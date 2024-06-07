from typing import Dict, Union
from math import sqrt, log

import torch
from openfold.utils.rigid_utils import Rigid, Rotation, invert_quat, quat_multiply, quat_multiply_by_vec

from .tools.quat import get_angle


class EGNNLayer(torch.nn.Module):
    """
    This layer does not take graph edge as input.
    It simply assumes every peptide residue is connected to every other peptide residue and every pocket residue.
    Thus it's an all-connected graph.
    """

    def __init__(self, node_input_size: int, edge_input_size: int, node_output_size: int, message_size: int):
        """
        Args:
            node_input_size: expected dimension for node features
            edge_input_size: expected dimension for edge features
            node_output_size: output dimension for updated node features
            message_size: dimension for internal message format
        """

        super().__init__()

        # dimension for transitional state
        transition_size = 64

        # updates node features
        self.feature_mlp = torch.nn.Sequential(
            torch.nn.Linear((node_input_size + message_size), transition_size),
            torch.nn.ReLU(),
            torch.nn.Linear(transition_size, node_output_size),
        )

        # computes a message from two nodes and a connecting edge
        self.message_mlp = torch.nn.Sequential(
            torch.nn.Linear(2 * node_input_size + edge_input_size + 9, transition_size),
            torch.nn.ReLU(),
            torch.nn.Linear(transition_size, message_size),
        )

        # computes a translation update from a message
        self.translation_mlp = torch.nn.Sequential(
            torch.nn.Linear(message_size, transition_size),
            torch.nn.ReLU(),
            torch.nn.Linear(transition_size, 3),
        )

        # computes a quaternion update from a message
        self.quat_mlp = torch.nn.Sequential(
            torch.nn.Linear(message_size, transition_size),
            torch.nn.ReLU(),
            torch.nn.Linear(transition_size, 4),
        )

        # computes a residue's torsion sin,cos angles from a message
        self.torsion_mlp = torch.nn.Sequential(
            torch.nn.Linear(message_size, transition_size),
            torch.nn.ReLU(),
            torch.nn.Linear(transition_size, 7 * 2),
        )

    def forward(
        self,
        frames: Rigid,
        torsions: torch.Tensor,
        h: torch.Tensor,
        e: torch.Tensor,
        node_mask: torch.Tensor,
        pocket_h: torch.Tensor,
        pocket_e: torch.Tensor,
        pocket_frames: Rigid,
        pocket_mask: torch.Tensor,
    ) -> Rigid:
        """
        Args:
            frames:     [*, N] (rotation + translation)
            torsions:   [*, N, 7, 2]
            h:          [*, N, H] (node features)
            e:          [*, N, N, E] (edge features)
            node_mask:  [*, N]
            pocket_h:    [*, P, H] (node features)
            pocket_e:    [*, N, P, E] (node features)
            pocket_frames:      [*, P] (rotation + translation)
            pocket_mask:        [*, P]
        Returns:
            updated frames:        [*, N]
            updated torsions:      [*, N, 7, 2]
            updated node features: [*, N, O]
        """

        N = h.shape[-2]  # max number of peptide nodes
        H = h.shape[-1]  # dimension of node features
        P = pocket_h.shape[-2]  # max number of pocket nodes

        # build a mask for the messages between the nodes.

        # [*, N, N]
        peptide_message_mask = torch.logical_and(node_mask.unsqueeze(-2), node_mask.unsqueeze(-1))
        peptide_message_mask = torch.logical_and(peptide_message_mask, torch.logical_not(torch.eye(node_mask.shape[-1], device=node_mask.device)[None, ...]))

        # [*, N, P]
        pocket_message_mask = torch.logical_and(node_mask.unsqueeze(-1), pocket_mask.unsqueeze(-2))

        # [*, N, N+P]
        message_mask = torch.cat((peptide_message_mask, pocket_message_mask), dim=-1)

        # [*]
        n_peptide_nodes = node_mask.sum(dim=-1)  # actual number of peptide nodes
        n_pocket_nodes = pocket_mask.sum(dim=-1)  # actual number of pocket nodes
        n_neighbours = n_peptide_nodes + n_pocket_nodes - 1  # actual number of neighbours that each peptide node has

        # [*, N, 3]
        x = frames.get_trans()

        # [*, P, 3]
        pocket_x = pocket_frames.get_trans()

        # [*, N, 4]
        q = frames.get_rots().get_quats()

        # [*, P, 4]
        pocket_q = pocket_frames.get_rots().get_quats()

        # [*, N, N+P] : transforms each peptide node to neighbour-node-local space
        global_to_local = Rigid.from_tensor_7(
            torch.cat(
                (
                    frames.invert().to_tensor_7()[..., None, :, :].expand(list(frames.shape) + [frames.shape[-1], 7]),
                    pocket_frames.invert().to_tensor_7()[..., None, :, :].expand(list(frames.shape) + [pocket_frames.shape[-1], 7]),
                ), dim=-2
            )
        )

        # representations of how distant two nodes' x and q are:
        # [*, N, N]
        peptide_d2 = torch.square(x[..., :, None, :] - x[..., None, :, :]).sum(dim=-1)
        peptide_qdot = torch.abs((q[..., :, None, :] * q[..., None, :, :]).sum(dim=-1))

        # [*, N, P]
        pocket_d2 = torch.square(x[..., :, None, :] - pocket_x[..., None, :, :]).sum(dim=-1)
        pocket_qdot = torch.abs((q[..., :, None, :] * pocket_q[..., None, :, :]).sum(dim=-1))

        # [*, N, N+P]
        d2 = torch.cat((peptide_d2, pocket_d2), dim=-1)
        qdot = torch.cat((peptide_qdot, pocket_qdot), dim=-1)

        # neighbour-node-local representations of rotation and translation:
        # [*, N, N+P, 3]
        local_x = global_to_local.apply(x[..., :, None, :])

        # [*, N, N+P, 4]
        local_q = quat_multiply(global_to_local.get_rots().get_quats(), q[..., :, None, :])

        # features of neighbouring nodes i and j
        # [*, N, N+P, H]
        hi = h.unsqueeze(-2).expand(-1, -1, N + P, -1)
        hj = torch.cat(
            (h.unsqueeze(-3).expand(-1, N, -1, -1), pocket_h.unsqueeze(-3).expand(-1, N, -1, -1)),
            dim=-2
        )

        # edge feature representation
        # [*, N, N+P, E]
        e = torch.cat((e, pocket_e), dim=-2)

        # gen message
        # [*, N, N+P, M]
        m = self.message_mlp(torch.cat((hi, hj, e, local_x, local_q, d2[..., None], qdot[..., None]), dim=-1)) * message_mask[..., None]

        # gen output feature
        # [*, N, O]
        o = self.feature_mlp(torch.cat((h, m.sum(dim=-2)), dim=-1))

        # gen torsion updates
        # [*, N, 8, 2]
        upd_torsions = torch.nn.functional.normalize(
            self.torsion_mlp(m.sum(dim=-2)).reshape(torsions.shape),
            dim=-1,
        )

        # gen local translation update
        # [*, N, N+P, 3]
        dx = self.translation_mlp(m) * message_mask[..., None]

        # gen local rotation update, identity where masked
        # [*, N, 4]
        dq = self.quat_mlp(m.sum(dim=-2))
        dq = torch.where(
            node_mask.unsqueeze(-1).expand(dq.shape),
            dq,
            torch.tensor([[[1.0, 0.0, 0.0, 0.0]]], device=dq.device).expand(dq.shape),
        )
        dq = torch.nn.functional.normalize(dq, dim=-1)

        # global rotation update, identity where masked
        # [*, N, 4]
        upd_q = quat_multiply(q, dq)
        upd_q = torch.where(
            node_mask.unsqueeze(-1).expand(upd_q.shape),
            upd_q,
            torch.tensor([[[1.0, 0.0, 0.0, 0.0]]], device=upd_q.device).expand(q.shape)
        )

        # transform local translation updates to global space
        # [*, N, N+P]
        rot_local_to_global = Rotation(
            quats=torch.cat(
                (
                    q[..., None, :, :].expand(list(q.shape[:-1]) + [N, 4]),
                    pocket_q[..., None, :, :].expand(list(q.shape[:-1]) + [P, 4])
                ),
                dim=-2
            ),
            normalize_quats=True
        )

        # [*, N, 3]
        upd_x = x + rot_local_to_global.apply(dx).sum(dim=-2) / n_neighbours[:, None, None]

        # output updated frames, torsions and node features
        return Rigid(Rotation(quats=upd_q, normalize_quats=True), upd_x), upd_torsions, o


class Model(torch.nn.Module):
    def __init__(self, max_len: int, node_input_size: int, T: int):
        """
        Args:
            max_len: max expected input length of peptide sequence (needed for encoding)
            node_input_size: expected dimension for node features
            T: expected max number of time steps
        """

        super().__init__()

        self.max_len = max_len

        relposenc_depth = max_len * 2 - 1
        self.relposenc_depth = relposenc_depth

        # [N]
        relposenc_range = torch.arange(max_len)

        # [N, N]
        relative_positions = (max_len - 1) + (relposenc_range[:, None] - relposenc_range[None, :])

        # [N, N, depth]
        self.relative_position_encodings = torch.nn.functional.one_hot(relative_positions, num_classes=relposenc_depth)

        # node features, H: 22 + time variable
        H = node_input_size + 1

        # edge features, E: one-hot relative position encoding for peptide, zero for pocket
        E = relposenc_depth

        I = 64
        M = 32

        self.gnn1 = EGNNLayer(H, E, I, M)
        self.gnn2 = EGNNLayer(I, E, 1, M)

        self.act = torch.nn.ReLU()

        self.T = T

    def forward(
        self,
        batch: Dict[str, Union[torch.Tensor, Rigid]],
        t: int,
    ) -> Rigid:

        # pointers to input data
        noised_frames = batch['frames']
        noised_torsions = batch["torsions"]
        node_features = batch['features']
        node_mask = batch['mask']
        pocket_frames = batch['pocket_frames']
        pocket_mask = batch['pocket_mask']
        pocket_features = batch['pocket_features']
        batch_size = node_mask.shape[0]

        # node feature, containing the time
        ft = torch.tensor([[[t / self.T]]]).expand(batch_size, node_features.shape[-2], 1)

        # input all coords + features to GNN
        h = torch.cat((node_features, ft), dim=-1)
        e = self.relative_position_encodings.clone().unsqueeze(0).expand(batch_size, -1, -1, -1)

        # do not include time with pocket nodes
        pocket_h = torch.cat((pocket_features, torch.zeros(list(pocket_mask.shape) + [1], device=pocket_features.device)), dim=-1)

        # leave pocket edge features blank
        pocket_e = torch.zeros(batch_size, e.shape[-3], pocket_mask.shape[-1], e.shape[-1], device=e.device)

        # gnn update layer 1
        frames, torsions, i = self.gnn1(noised_frames, noised_torsions, h, e, node_mask, pocket_h, pocket_e, pocket_frames, pocket_mask)

        # activation function on updated node features
        i = self.act(i)

        # expand pocket node features to the same dimensionality as the updated peptide features
        # pad with zeros
        pocket_i = torch.zeros(list(pocket_h.shape[:-1]) + [i.shape[-1]], device=i.device)
        pocket_i[..., :pocket_h.shape[-1]] = pocket_h

        # gnn update layer 2
        frames, torsions, o = self.gnn2(frames, torsions, i, e, node_mask, pocket_i, pocket_e, pocket_frames, pocket_mask)

        # output updated data as noise prediction
        return {
            "frames": frames,
            "torsions": torsions,
        }
