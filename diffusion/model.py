from typing import Dict, Union
from math import sqrt, log

import torch

from openfold.utils.rigid_utils import Rigid, Rotation, invert_quat, quat_multiply, quat_multiply_by_vec


class EGNNLayer(torch.nn.Module):
    def __init__(self, node_input_size: int, edge_input_size: int, node_output_size: int, message_size: int):
        super().__init__()

        self.feature_mlp = torch.nn.Linear((node_input_size + message_size), node_output_size)
        self.message_mlp = torch.nn.Linear(2 * node_input_size + edge_input_size + 9, message_size)
        self.frame_mlp = torch.nn.Linear(message_size, 6)

    def forward(
        self,
        frames: Rigid,
        h: torch.Tensor,
        e: torch.Tensor,
        node_mask: torch.Tensor,
    ) -> Rigid:
        """
        Args:
            frames:     [*, N]
            h:          [*, N, H] (node features)
            e:          [*, N, N, E] (edge features)
            node_mask:  [*, N]
        Returns:
            updated frames:        [*, N]
            updated node features: [*, N, O]
        """

        N = h.shape[-2]
        H = h.shape[-1]

        # [*, N, N]
        message_mask = torch.logical_and(node_mask.unsqueeze(-2), node_mask.unsqueeze(-1))
        message_mask = torch.logical_and(message_mask, torch.logical_not(torch.eye(node_mask.shape[-1], device=node_mask.device)[None, ...]))

        # [*]
        n_nodes = node_mask.sum(dim=-1)

        # [*, N, 3]
        x = frames.get_trans()

        # [*, N, 4]
        q = frames.get_rots().get_quats()

        conv_shape = list(frames.shape) + [frames.shape[-1], 7]

        # [*, N, N]
        local_to_global = Rigid.from_tensor_7(frames.to_tensor_7()[..., None, :, :].expand(conv_shape))
        global_to_local = Rigid.from_tensor_7(frames.invert().to_tensor_7()[..., None, :, :].expand(conv_shape))

        # [*, N, N, 1]
        d2 = torch.square(x[:, :, None, :] - x[:, None, :, :]).sum(dim=-1)
        dot = (q[..., :, None, :] * q[..., None, :, :]).sum(dim=-1).abs()

        # [*, N, N, 3]
        local_x = global_to_local.apply(x[..., :, None, :])

        # [*, N, N, 4]
        local_q = quat_multiply(
            quat_multiply(global_to_local.get_rots().get_quats(), q[..., :, None, :]),
            local_to_global.get_rots().get_quats()
        )

        # [*, N, N, H]
        hi = h.unsqueeze(-2).expand(-1, -1, N, -1)
        hj = h.unsqueeze(-3).expand(-1, N, -1, -1)

        # [*, N, N, M]
        m = self.message_mlp(torch.cat((hi, hj, e, local_x, local_q, d2[..., None], dot[..., None]), dim=-1)) * message_mask[..., None]

        # [*, N, O]
        o = self.feature_mlp(torch.cat((h, m.sum(dim=-2)), dim=-1))

        # [*, N, N, 7]
        delta = self.frame_mlp(m) * message_mask[..., None]

        # [*, N, 3]
        upd_x = (local_to_global.apply(local_x + delta[..., 3:])).sum(dim=-2) / (n_nodes[:, None, None] - 1)

        # [*, N, 4]
        local_upd_q = torch.nn.functional.normalize((local_q + quat_multiply_by_vec(local_q, delta[..., :3])).sum(dim=-2), dim=-1)
        upd_q = quat_multiply(quat_multiply(q, local_upd_q), invert_quat(q))

        return Rigid(Rotation(quats=upd_q), upd_x), o


class Model(torch.nn.Module):
    def __init__(self, max_len: int, sequence_onehot_depth: int, T: int):
        super().__init__()

        self.max_len = max_len

        relposenc_depth = max_len * 2 - 1

        # [N]
        relposenc_range = torch.arange(max_len)

        # [N, N]
        relative_positions = (max_len - 1) + (relposenc_range[:, None] - relposenc_range[None, :])

        # [N, N, depth]
        self.relative_position_encodings = torch.nn.functional.one_hot(relative_positions, num_classes=relposenc_depth)

        # node features: 22 + frame(=7) + time variable
        H = sequence_onehot_depth + 7 + 1

        # edge features: position encoding + bonded
        E = relposenc_depth

        I = 64
        M = 32

        self.gnn1 = EGNNLayer(H, E, I, M)
        self.gnn2 = EGNNLayer(I, E, I, M)
        self.gnn3 = EGNNLayer(I, E, I, M)
        self.gnn4 = EGNNLayer(I, E, 1, M)

        self.act = torch.nn.ReLU()

        self.T = T

    def forward(
        self,
        batch: Dict[str, Union[torch.Tensor, Rigid]],
        t: int,
    ) -> Rigid:

        noised_frames = batch['frames']
        node_features = batch['features']
        node_mask = batch['mask']

        batch_size = node_mask.shape[0]

        ft = torch.tensor([[[t / self.T]]]).expand(batch_size, self.max_len, 1)

        # input all coords + features to GNN
        h = torch.cat((noised_frames.to_tensor_7(), node_features, ft), dim=-1)
        e = self.relative_position_encodings.clone().unsqueeze(0).expand(batch_size, -1, -1, -1)

        frames, i = self.gnn1(noised_frames, h, e, node_mask)
        i = self.act(i)
        frames, i = self.gnn2(frames, i, e, node_mask)
        i = self.act(i)
        frames, i = self.gnn3(frames, i, e, node_mask)
        i = self.act(i)
        frames, o = self.gnn4(frames, i, e, node_mask)

        return frames
