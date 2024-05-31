from typing import Dict, Union
from math import sqrt, log

import torch

from openfold.utils.rigid_utils import Rigid, Rotation, invert_quat, quat_multiply_by_vec


class GNNLayer(torch.nn.Module):
    def __init__(self, node_input_size: int, edge_input_size: int, node_output_size: int, message_size: int):
        super().__init__()

        self.feature_mlp = torch.nn.Sequential(
            torch.nn.Linear((node_input_size + message_size), node_output_size),
        )
        self.message_mlp = torch.nn.Sequential(
            torch.nn.Linear(2 * node_input_size + edge_input_size, message_size),
        )

    def forward(
        self,
        h: torch.Tensor,
        e: torch.Tensor,
        node_mask: torch.Tensor,
    ) -> Rigid:
        """
        Args:
            h: [*, N, H] (node features)
            e: [*, N, N, E] (edge features)
            node_mask: [*, N]
        Returns:
            updated node features: [*, N, O]
        """

        N = h.shape[-2]
        H = h.shape[-1]

        # [*, N, N]
        mask2 = torch.logical_and(node_mask.unsqueeze(-2), node_mask.unsqueeze(-1))

        # [*, N, N, H]
        hi = h.unsqueeze(-2).expand(-1, -1, N, -1)
        hj = h.unsqueeze(-3).expand(-1, N, -1, -1)

        # [*, N, N, M]
        m = self.message_mlp(torch.cat((hi, hj, e), dim=-1)) * mask2.unsqueeze(-1)

        # [*, N, O]
        o = self.feature_mlp(torch.cat((h, m.sum(dim=-2)), dim=-1))

        return o


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
        M = 16
        J = 128

        self.gnn1 = GNNLayer(H, E, I, M)
        self.gnn2 = GNNLayer(I, E, I, M)
        self.gnn3 = GNNLayer(I, E, I, M)
        self.gnn4 = GNNLayer(I, E, I, M)
        self.act = torch.nn.ReLU()

        self.trans = torch.nn.Sequential(
            torch.nn.Linear(I, J),
            torch.nn.ReLU(),
            torch.nn.Linear(J, 7),
        )

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

        i = self.gnn1(h, e, node_mask)
        i = self.act(i)
        i = self.gnn2(i, e, node_mask)
        i = self.act(i)
        i = self.gnn3(i, e, node_mask)
        i = self.act(i)
        i = self.gnn4(i, e, node_mask)
        i = self.act(i)

        o = self.trans(i)

        noise_frames = Rigid.from_tensor_7(o, normalize_quats=True)

        return noise_frames
