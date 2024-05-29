from typing import Dict, Union
from math import sqrt, log

import torch

from openfold.utils.rigid_utils import Rigid, Rotation, invert_quat, quat_multiply_by_vec


class PositionalEncoding(torch.nn.Module):

    def __init__(self, d_model: int, dropout: float = 0.1, max_len: int = 5000):
        super().__init__()
        self.dropout = torch.nn.Dropout(p=dropout)

        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-log(10000.0) / d_model))
        pe = torch.zeros(1, max_len, d_model)
        pe[0, :, 0::2] = torch.sin(position * div_term)
        pe[0, :, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Arguments:
            x: Tensor, shape ``[*, N, d]``
        """
        x = x + self.pe[:, :x.shape[-2], :]
        return self.dropout(x)


class GNNLayer(torch.nn.Module):
    def __init__(self, M: int, H: int, O: int):
        super().__init__()

        self.feature_mlp = torch.nn.Sequential(
            torch.nn.Linear((M + H), O),
        )
        self.message_mlp = torch.nn.Sequential(
            torch.nn.Linear(2 * H, M),
        )

    def forward(
        self,
        h: torch.Tensor,
        node_mask: torch.Tensor,
        edge_mask: torch.Tensor
    ) -> Rigid:
        """
        Args:
            h: [*, N, H] (other features)
            node_mask: [*, N]
            edge_mask: [*, N, N]
        Returns:
            updated h: [*, N, O]
        """

        N = h.shape[-2]
        H = h.shape[-1]

        # [*, N, N]
        mask2 = torch.logical_and(node_mask.unsqueeze(-2), node_mask.unsqueeze(-1))
        mask2 = torch.logical_and(mask2, edge_mask)

        # [*, N, N, H]
        hi = h.unsqueeze(-2).expand(-1, -1, N, -1)
        hj = h.unsqueeze(-3).expand(-1, N, -1, -1)

        # [*, N, N, M]
        m = self.message_mlp(torch.cat((hi, hj), dim=-1)) * mask2.unsqueeze(-1)

        # [*, N, O]
        o = self.feature_mlp(torch.cat((h, m.sum(dim=-2)), dim=-1))

        return o


class Model(torch.nn.Module):
    def __init__(self, M: int, H: int, T: int):
        super().__init__()

        self._posenc_size = 32 - H
        self.posenc = PositionalEncoding(self._posenc_size)

        I = 64

        # 32 + frame(=7) + time variable = 40
        self.gnn1 = GNNLayer(M, 40, I)
        self.act = torch.nn.ReLU()
        self.gnn2 = GNNLayer(M, I, 7)  # output 7 for frames

        self.T = T

    def forward(
        self,
        batch: Dict[str, Union[torch.Tensor, Rigid]],
        t: int,
    ) -> Rigid:

        noised_frames = batch['frames']
        node_features = batch['features']
        node_mask = batch['mask']

        # connect residues {1 & 2, 2 & 3, 3 & 4, ...}
        edge_mask = (
            torch.abs(
                torch.arange(node_mask.shape[-1]).unsqueeze(0) -
                torch.arange(node_mask.shape[-1]).unsqueeze(-1)
            ) == 1
        ).unsqueeze(0).expand(node_mask.shape[0], node_mask.shape[-1], node_mask.shape[-1])

        ft = torch.tensor([[[t / self.T]]]).expand(list(node_features.shape[:-1]) + [1])

        # make residue position features
        p = self.posenc(torch.zeros(list(node_features.shape[:-1]) + [self._posenc_size]))

        # input all coords + features to GNN
        h = torch.cat((noised_frames.to_tensor_7(), node_features, p, ft), dim=-1)

        i = self.gnn1(h, node_mask, edge_mask)
        i = self.act(i)
        o = self.gnn2(i, node_mask, edge_mask)

        #o = self.mlp(h)

        noise_frames = Rigid.from_tensor_7(o, normalize_quats=True)

        return noise_frames
