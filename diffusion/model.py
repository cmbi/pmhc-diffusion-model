from typing import Dict, Union
from math import sqrt, log

import torch
from openfold.utils.rigid_utils import Rigid, Rotation, invert_quat, quat_multiply, quat_multiply_by_vec

from .tools.angle import multiply_sin_cos


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

        torsions_flat_size = 7 * 2

        # dimension for transitional state
        transition_size = 256

        # updates node features from a message
        self.feature_mlp = torch.nn.Sequential(
            torch.nn.Linear((node_input_size + message_size), transition_size),
            torch.nn.ReLU(),
            torch.nn.Linear(transition_size, node_output_size),
        )

        # computes a message from two nodes and a connecting edge
        # from: node features, edge features, frames, distances, torsions
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

        # computes a residue's torsion sin & cos angles from a message
        self.torsion_mlp = torch.nn.Sequential(
            torch.nn.Linear((message_size + torsions_flat_size), transition_size),
            torch.nn.ReLU(),
            torch.nn.Linear(transition_size, torsions_flat_size),
        )

    def forward(
        self,
        peptide_node_frames: Rigid,
        peptide_node_torsions: torch.Tensor,
        peptide_node_features: torch.Tensor,
        peptide_edge_features: torch.Tensor,
        peptide_node_mask: torch.Tensor,
        pocket_node_features: torch.Tensor,
        pocket_node_frames: Rigid,
        pocket_node_mask: torch.Tensor,
    ) -> Rigid:
        """
        Args:
            peptide_node_frames:     [*, N] (rotation + translation)
            peptide_node_torsions:   [*, N, 7, 2]
            peptide_node_features:   [*, N, H] (node features)
            peptide_edge_features:   [*, N, N, E] (edge features)
            peptide_node_mask:       [*, N]
            pocket_node_features:    [*, P, H] (node features)
            pocket_node_frames:      [*, P] (rotation + translation)
            pocket_node_mask:        [*, P]
        Returns:
            updated frames:        [*, N]
            updated torsions:      [*, N, 7, 2]
            updated node features: [*, N, O]
        """

        # build a mask for the messages between the nodes.

        # [*, N, N]
        peptide_message_mask = torch.logical_and(peptide_node_mask.unsqueeze(-2), peptide_node_mask.unsqueeze(-1))
        peptide_message_mask = torch.logical_and(peptide_message_mask, torch.logical_not(torch.eye(peptide_node_mask.shape[-1], device=peptide_node_mask.device)[None, ...]))

        # [*, N, P]
        pocket_message_mask = torch.logical_and(peptide_node_mask.unsqueeze(-1), pocket_node_mask.unsqueeze(-2))

        # [*, N, N+P]
        message_mask = torch.cat((peptide_message_mask, pocket_message_mask), dim=-1)

        # [*, N, N+P, M]
        message = self._compute_message(
            peptide_node_features,
            peptide_edge_features,
            peptide_node_mask,
            peptide_node_frames,
            pocket_node_features,
            pocket_node_mask,
            pocket_node_frames,
        ) * message_mask[..., None]

        # gen output feature
        # [*, N, O]
        o = self.feature_mlp(torch.cat((peptide_node_features, message.sum(dim=-2)), dim=-1))

        N = peptide_node_frames.shape[-1]
        P = pocket_node_frames.shape[-1]

        # [*, N, N+P, 4]
        neighbour_quats = torch.cat(
            (
                peptide_node_frames.get_rots().get_quats()[..., None, :, :].expand(list(peptide_node_frames.shape) + [N, 4]),
                pocket_node_frames.get_rots().get_quats()[..., None, :, :].expand(list(peptide_node_frames.shape) + [P, 4]),
            ),
            dim=-2,
        )

        # [*, N, 3]
        upd_x = self._translation_update(peptide_node_frames.get_trans(), message, message_mask, neighbour_quats)

        # [*, N, 4]
        upd_q = self._rotation_update(peptide_node_frames.get_rots().get_quats(), message, message_mask)

        # [*, N, 7, 2]
        upd_torsions = self._torsion_update(peptide_node_torsions, message, message_mask)

        # Output updated frames, torsions and node features.
        # We must normalize the quats here, for the next layer.
        return Rigid(Rotation(quats=upd_q, normalize_quats=True), upd_x), upd_torsions, o

    def _compute_message(
        self,
        peptide_node_features: torch.Tensor,
        peptide_edge_features: torch.Tensor,
        peptide_node_mask: torch.Tensor,
        peptide_node_frames: Rigid,
        pocket_node_features: torch.Tensor,
        pocket_node_mask: torch.Tensor,
        pocket_node_frames: Rigid,
    ) -> torch.Tensor:

        N = peptide_node_features.shape[-2]
        P = pocket_node_features.shape[-2]
        E = peptide_edge_features.shape[-1]

        # features of neighbouring nodes i and j
        # [*, N, N+P, H]
        h_i = peptide_node_features[..., None, :].expand(-1, -1, N + P, -1)
        h_j = torch.cat(
            (
                peptide_node_features[..., None, :, :].expand(-1, N, -1, -1),
                pocket_node_features[..., None, :, :].expand(-1, N, -1, -1),
            ),
            dim=-2,
        )

        # zero edge features from pocket to peptide
        # [*, N, P, E]
        pocket_edge_features = torch.zeros(list(peptide_edge_features.shape[:-2]) + [P, E], device=peptide_edge_features.device)

        # [*, N, N+P, E]
        e = torch.cat(
            (
                peptide_edge_features,
                pocket_edge_features,
            ),
            dim=-2,
        )

        # transforms each peptide node to neighbour-node-local space
        # [*, N, N+P]
        global_to_local = Rigid.from_tensor_7(
            torch.cat(
                (
                    peptide_node_frames.invert().to_tensor_7()[..., None, :, :].expand(list(peptide_node_frames.shape[:-1]) + [N, N, 7]),
                    pocket_node_frames.invert().to_tensor_7()[..., None, :, :].expand(list(pocket_node_frames.shape[:-1]) + [N, P, 7]),
                ),
                dim=-2,
            )
        )

        # [*, N, 3]
        peptide_x = peptide_node_frames.get_trans()

        # [*, N, 4]
        peptide_q = peptide_node_frames.get_rots().get_quats()

        # [*, N, 3]
        pocket_x = pocket_node_frames.get_trans()

        # [*, N, 4]
        pocket_q = pocket_node_frames.get_rots().get_quats()

        # neighbour-node-local representations of rotation and translation:
        # [*, N, N+P, 3]
        local_x = global_to_local.apply(peptide_x[..., :, None, :])

        # [*, N, N+P, 4]
        local_q = quat_multiply(global_to_local.get_rots().get_quats(), peptide_q[..., :, None, :])

        # representations of how distant two nodes' x and q are:
        # [*, N, N]
        peptide_d2 = torch.square(peptide_x[..., :, None, :] - peptide_x[..., None, :, :]).sum(dim=-1)
        peptide_qdot = torch.abs((peptide_q[..., :, None, :] * peptide_q[..., None, :, :]).sum(dim=-1))

        # [*, N, P]
        pocket_d2 = torch.square(peptide_x[..., :, None, :] - pocket_x[..., None, :, :]).sum(dim=-1)
        pocket_qdot = torch.abs((peptide_q[..., :, None, :] * pocket_q[..., None, :, :]).sum(dim=-1))

        # [*, N, N+P]
        d2 = torch.cat((peptide_d2, pocket_d2), dim=-1)
        qdot = torch.cat((peptide_qdot, pocket_qdot), dim=-1)

        # gen message
        # [*, N, N+P, M]
        message = self.message_mlp(torch.cat((h_i, h_j, e, local_x, local_q, d2[..., None], qdot[..., None]), dim=-1))

        return message

    @staticmethod
    def _get_message_weight(message_mask: torch.Tensor) -> torch.Tensor:

        # [*, N]
        n_message = message_mask.sum(dim=-1)
        weight = torch.where(
            n_message > 0,
            1.0 / n_message,
            0.0,
        )

        return weight

    def _torsion_update(
        self,
        torsions: torch.Tensor,
        message: torch.Tensor,
        message_mask: torch.Tensor,
    ) -> torch.Tensor:

        # [*, N]
        c = self._get_message_weight(message_mask)

        # torsions representation
        # [*, N, 7 * 2]
        flat_torsions = torsions.reshape(list(torsions.shape[:-2]) + [torsions.shape[-2] * torsions.shape[-1]])

        # [*, N, 7, 2]
        delta = self.torsion_mlp(torch.cat((message.sum(dim=-2) * c[..., None], flat_torsions), dim=-1)).reshape(torsions.shape)
        delta = torch.nn.functional.normalize(delta, dim=-1)

        # [*, N, 7, 2]
        torsions = torch.where(
            (message_mask.sum(dim=-1) > 0)[..., None, None].expand(torsions.shape),
            multiply_sin_cos(torsions, delta),
            torsions,
        )

        return torsions

    def _rotation_update(
        self,
        quats: torch.Tensor,
        message: torch.Tensor,
        message_mask: torch.Tensor,
    ) -> torch.Tensor:

        # [*, N]
        c = self._get_message_weight(message_mask)

        # gen local rotation update, identity where masked
        # [*, N, 4]
        delta = self.quat_mlp(message.sum(dim=-2) * c[..., None])
        delta = torch.nn.functional.normalize(delta, dim=-1)

        # global rotation update
        # [*, N, 4]
        quats = torch.where(
            (message_mask.sum(dim=-1) > 0)[..., None].expand(delta.shape),
            quat_multiply(quats, delta),
            quats,
        )

        return quats

    def _translation_update(
        self,
        x: torch.Tensor,
        message: torch.Tensor,
        message_mask: torch.Tensor,
        neighbour_quats: torch.Tensor
    ) -> torch.Tensor:

        # [*, N]
        c = self._get_message_weight(message_mask)

        # gen local translation update
        # [*, N, N+P, 3]
        delta = self.translation_mlp(message) * message_mask[..., None]

        # transform local translation updates to global space
        # [*, N, N+P]
        rot_local_to_global = Rotation(quats=neighbour_quats, normalize_quats=False)

        # [*, N, 3]
        x = x + rot_local_to_global.apply(delta).sum(dim=-2) * c[..., None]

        return x


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
        M = 128

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

        # gnn update layer 1
        frames, torsions, i = self.gnn1(noised_frames, noised_torsions, h, e, node_mask, pocket_h, pocket_frames, pocket_mask)

        # activation function on updated node features
        i = self.act(i)

        # expand pocket node features to the same dimensionality as the updated peptide features
        # pad with zeros
        pocket_i = torch.zeros(list(pocket_h.shape[:-1]) + [i.shape[-1]], device=i.device)
        pocket_i[..., :pocket_h.shape[-1]] = pocket_h

        # gnn update layer 2
        frames, torsions, o = self.gnn2(frames, torsions, i, e, node_mask, pocket_i, pocket_frames, pocket_mask)

        # output updated data as noise prediction
        return {
            "frames": frames,
            "torsions": torsions,
        }
