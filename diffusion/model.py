from typing import Dict, Union
from math import sqrt, log, pi

import torch
from openfold.utils.rigid_utils import Rigid, Rotation, invert_quat, quat_multiply, quat_multiply_by_vec

from .tools.angle import inverse_sin_cos, multiply_sin_cos, shoemake_quat


epsilon = 1e-9
infinity = 1e9


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

        n_torsions = 7
        torsions_flat_size = n_torsions * 2

        # dimension for transitional state
        transition_size = 64

        # updates node features from a message
        self.feature_mlp = torch.nn.Sequential(
            torch.nn.Linear((node_input_size + message_size), transition_size),
            torch.nn.ReLU(),
            torch.nn.Linear(transition_size, node_output_size),
        )

        # computes a message from two nodes and a connecting edge
        # from: node features, edge features, frames, distances, torsions
        self.message_mlp = torch.nn.Sequential(
            torch.nn.Linear(2 * node_input_size + edge_input_size, transition_size),
            torch.nn.ReLU(),
            torch.nn.Linear(transition_size, message_size),
        )

        # attention weighs the neighbours
        self.attention_mlp = torch.nn.Sequential(
            torch.nn.Linear(message_size + 2, transition_size),
            torch.nn.ReLU(),
            torch.nn.Linear(transition_size, 1),
            torch.nn.Flatten(-2, -1),
        )

        # computes a translation update from a message
        self.translation_mlp = torch.nn.Sequential(
            torch.nn.Linear(message_size, transition_size),
            torch.nn.ReLU(),
            torch.nn.Linear(transition_size, 1),
        )

        # computes a quaternion update from a message
        self.rotation_mlp = torch.nn.Sequential(
            torch.nn.Linear(message_size + 4, transition_size),
            torch.nn.ReLU(),
            torch.nn.Linear(transition_size, 4),
            torch.nn.Sigmoid(),
        )

        # computes a residue's torsion sin & cos angles from a message
        self.torsion_mlp = torch.nn.Sequential(
            torch.nn.Linear((message_size + torsions_flat_size), transition_size),
            torch.nn.ReLU(),
            torch.nn.Linear(transition_size, n_torsions),
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

        neighbour_frames_shape = list(message_mask.shape) + [7]

        # [*, N, N+P]
        neighbour_frames = Rigid.from_tensor_7(
            torch.cat(
                (
                    peptide_node_frames.to_tensor_7()[..., None, :, :],
                    pocket_node_frames.to_tensor_7()[..., None, :, :],
                ),
                dim=-2,
            ).expand(neighbour_frames_shape)
        )

        # [*, N, N+P, M]
        message = self._compute_message(
            peptide_node_features,
            peptide_edge_features,
            peptide_node_mask,
            peptide_node_frames,
            pocket_node_features,
            pocket_node_mask,
            neighbour_frames,
        )

        # [*, N, N+P]
        neighbour_weights = self._compute_attention(message, message_mask, peptide_node_frames, neighbour_frames)

        # gen output feature
        # [*, N, O]
        o = self.feature_mlp(torch.cat((peptide_node_features, message.sum(dim=-2)), dim=-1))

        N = peptide_node_frames.shape[-1]
        P = pocket_node_frames.shape[-1]

        # [*, N, 4]
        upd_q = self._rotation_update(peptide_node_frames.get_rots().get_quats(), message, message_mask, neighbour_frames, neighbour_weights)

        # [*, N, 7, 2]
        upd_torsions = self._torsion_update(peptide_node_torsions, message, message_mask, neighbour_weights)

        # [*, N]
        peptide_upd_frames = Rigid(Rotation(quats=upd_q, normalize_quats=False), peptide_node_frames.get_trans())

        # [*, N, N+P]
        neighbour_frames = Rigid.from_tensor_7(
            torch.cat(
                (
                    peptide_upd_frames.to_tensor_7()[..., None, :, :],
                    pocket_node_frames.to_tensor_7()[..., None, :, :],
                ),
                dim=-2,
            ).expand(neighbour_frames_shape)
        )

        # [*, N, 3]
        upd_x = self._translation_update(peptide_node_frames.get_trans(), message, message_mask, neighbour_frames, neighbour_weights)

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
        neighbour_frames: Rigid,
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

        # gen message
        # [*, N, N+P, M]
        message = self.message_mlp(torch.cat((h_i, h_j, e), dim=-1))

        return message

    def _compute_attention(
        self,
        message: torch.Tensor,
        message_mask: torch.Tensor,
        node_frames: Rigid,
        neighbour_frames: Rigid,
    ) -> torch.Tensor:

        # representations of how distant two nodes' x and q are:
        # [*, N, N+P]
        d2 = torch.square(node_frames.get_trans()[..., :, None, :] - neighbour_frames.get_trans()).sum(dim=-1)
        qdot2 = torch.square((node_frames.get_rots().get_quats()[..., :, None, :] * neighbour_frames.get_rots().get_quats()).sum(dim=-1))

        # [*, N, N+P] (adds on to 1.0 in dimension -1, 0.0 where masked)
        neighbour_weights = self.attention_mlp(torch.cat((message, -d2[..., None], qdot2[..., None]), dim=-1))
        neighbour_weights = torch.nn.functional.softmax(neighbour_weights - torch.logical_not(message_mask) * infinity, dim=-1)

        return neighbour_weights

    def _torsion_update(
        self,
        torsions: torch.Tensor,
        message: torch.Tensor,
        message_mask: torch.Tensor,
        neighbour_weights: torch.Tensor,
    ) -> torch.Tensor:

        # torsions representation
        # [*, N, 7 * 2]
        flat_torsions = torsions.reshape(list(torsions.shape[:-2]) + [torsions.shape[-2] * torsions.shape[-1]])

        # [*, N, N+P, 7]
        m_delta_a = self.torsion_mlp(torch.cat((message, flat_torsions[..., None, :].expand(list(message.shape[:-1]) + [flat_torsions.shape[-1]])), dim=-1))

        # [*, N, 7]
        delta_a = (m_delta_a * neighbour_weights[..., None]).sum(dim=-2)

        # [*, N, 7, 2]
        delta_t = torch.cat((torch.sin(delta_a)[..., None], torch.cos(delta_a)[..., None]), dim=-1)

        # [*, N, 7, 2]
        torsions = multiply_sin_cos(delta_t, torsions)
        return torsions

    def _rotation_update(
        self,
        quats: torch.Tensor,
        message: torch.Tensor,
        message_mask: torch.Tensor,
        neighbour_frames: Rigid,
        neighbour_weights: torch.Tensor,
    ) -> torch.Tensor:

        # [*, N, N+P, 4]
        neighbour_quats = neighbour_frames.get_rots().get_quats()
        inverse_neighbour_quats = invert_quat(neighbour_quats)

        # convert node quats to neighbour-local space
        # [*, N, N+P, 4]
        local_quats = quat_multiply(inverse_neighbour_quats, quat_multiply(quats[..., :, None, :], neighbour_quats))

        # learn update rotation in neighbour-local space
        # [*, N, N+P, 4]
        local_delta_quats = self.rotation_mlp(torch.cat((message, local_quats), dim=-1))
        torch.nn.functional.normalize(local_delta_quats, dim=-1)

        # convert predicted quats to global space and take the mean (approximization)
        # [*, N, N+P, 4]
        m_global_delta_quats = quat_multiply(neighbour_quats, quat_multiply(local_delta_quats, inverse_neighbour_quats))

        # [*, N, 4]
        # must convert masked quats to identity before normalizing
        global_delta_quats = (m_global_delta_quats * neighbour_weights[..., None]).sum(dim=-2)
        global_delta_quats = torch.where(
            (message_mask.sum(dim=-1) > 0)[..., None].expand(global_delta_quats.shape),
            global_delta_quats,
            torch.tensor([[[1.0, 0.0, 0.0, 0.0]]], device=global_delta_quats.device),
        )
        global_delta_quats = torch.nn.functional.normalize(global_delta_quats, dim=-1)

        # global rotation update
        # [*, N, 4]
        quats = quat_multiply(global_delta_quats, quats)

        return quats

    def _translation_update(
        self,
        x: torch.Tensor,
        message: torch.Tensor,
        message_mask: torch.Tensor,
        neighbour_frames: Rigid,
        neighbour_weights: torch.Tensor,
    ) -> torch.Tensor:

        # gen local translation update
        # [*, N, N+P, 1]
        m = self.translation_mlp(message)

        # [*, N, N+P, 3]
        r = x[..., :, None, :] - neighbour_frames.get_trans()

        # [*, N, 3]
        x = x + (m * r * neighbour_weights[..., None]).sum(dim=-2)

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
        M = 64

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
