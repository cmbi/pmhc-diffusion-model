#!/usr/bin/env python

import os
import random
import sys
import logging
from math import sqrt, log
from typing import Dict, Union

from diffusion.data import MhcpDataset

import torch
from torch.utils.data import DataLoader
from torch.nn.functional import one_hot

from openfold.utils.rigid_utils import Rigid, Rotation, invert_quat, quat_multiply_by_vec

from diffusion.tools.pdb import save
from diffusion.optimizer import DiffusionModelOptimizer


_log = logging.getLogger(__name__)


def square(x: float) -> float:
    return x * x


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


class EGNNLayer(torch.nn.Module):
    def __init__(self, M: int, H: int, O: int):
        super().__init__()

        I = 64

        self.feature_mlp = torch.nn.Sequential(
            torch.nn.Linear((M + H), I),
            torch.nn.ReLU(),
            torch.nn.Linear(I, I),
            torch.nn.ReLU(),
            torch.nn.Linear(I, O),
        )
        self.position_mlp = torch.nn.Sequential(
            torch.nn.Linear(M, I),
            torch.nn.ReLU(),
            torch.nn.Linear(I, I),
            torch.nn.ReLU(),
            torch.nn.Linear(I, 3),
        )
        self.message_mlp = torch.nn.Sequential(
            torch.nn.Linear(2 * H + 1, I),
            torch.nn.ReLU(),
            torch.nn.Linear(I, I),
            torch.nn.ReLU(),
            torch.nn.Linear(I, M),
        )

    def forward(self, frames: torch.Tensor, h: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: [*, N, 3] (positions)
            h: [*, N, H] (other features)
            mask: [*, N]

        Returns:
            updated x: [*, N, 3]
            updated h: [*, N, H]
        """

        N = frames.shape[-2]
        H = h.shape[-1]

        # [*, N, N]
        mask2 = torch.logical_and(mask.unsqueeze(-2), mask.unsqueeze(-1))
        mask2 = torch.logical_and(mask2, torch.logical_not(torch.eye(N, N, dtype=torch.bool).unsqueeze(0)))

        # [N]
        positions = torch.arange(N)
        # [1, N, N, 1]
        neighbours = (torch.abs(positions.unsqueeze(-2) - positions.unsqueeze(-1)) == 1).unsqueeze(0).unsqueeze(-1)

        # [*, N, N, 3]
        r = x.unsqueeze(-3) - x.unsqueeze(-2)

        # [*, N, N]
        d2 = (r ** 2).sum(-1)

        # [*, N, N, H]
        hi = h.unsqueeze(-3).expand(-1, N, -1, -1)
        hj = h.unsqueeze(-2).expand(-1, -1, N, -1)

        # [*, N, N, M]
        m = self.message_mlp(torch.cat((hi, hj, d2.unsqueeze(-1)), dim=-1)) * mask2.unsqueeze(-1) * neighbours

        # [*, N, N, 3]
        mx = self.position_mlp(m) * r

        x = x + mx.sum(dim=-2) / mask2.float().sum(dim=-1).unsqueeze(-1)

        h = self.feature_mlp(torch.cat((h, m.sum(dim=-2)), dim=-1))

        return x, h


class Model(torch.nn.Module):
    def __init__(self, M: int, H: int, T: int):
        super().__init__()

        self.posenc = PositionalEncoding(32 - H)

        I = 64

        self.frames_mlp = torch.nn.Sequential(
            torch.nn.Linear(40, I),
            torch.nn.ReLU(),
            torch.nn.Linear(I, I),
            torch.nn.ReLU(),
            torch.nn.Linear(I, 7),
        )

        self.T = T

    def forward(
        self,
        batch: Dict[str, Union[torch.Tensor, Rigid]],
        t: int,
    ) -> Rigid:

        noised_frames = batch['frames']

        noised_positions = noised_frames.get_trans()

        noised_quats = noised_frames.get_rots().get_quats()

        h = batch['features']
        mask = batch['mask']

        ft = torch.tensor([[[t / self.T]]]).expand(list(h.shape[:-1]) + [1])

        # make residue position features
        p = self.posenc(torch.zeros(list(h.shape[:-1]) + [32 - h.shape[-1]]))

        # input all frames + features to MLP
        upd = self.frames_mlp(torch.cat((noised_positions, noised_quats, h, p, ft), dim=-1))

        # predicted noise frames
        upd_positions = upd[..., 4:]
        upd_quats = upd[..., :4]

        return Rigid(Rotation(quats=upd_quats, normalize_quats=True), upd_positions)


if __name__ == "__main__":

    hdf5_path = sys.argv[1]

    model_path = "model.pth"

    logging.basicConfig(stream=sys.stdout, level=logging.DEBUG)

    device = torch.device("cpu")

    # init
    torch.autograd.detect_anomaly(check_nan=True)

    dataset = MhcpDataset(hdf5_path, device)
    data_loader = DataLoader(dataset, batch_size=64, shuffle=True)

    _log.debug(f"initializing model")
    T = 1000
    model = Model(16, 22, T).to(device=device)
    if os.path.isfile("model.pth"):
        model.load_state_dict(torch.load("model.pth", map_location=device))

    _log.debug(f"initializing diffusion model optimizer")
    dm = DiffusionModelOptimizer(T, model)

    # train
    nepoch = 100
    for epoch_index in range(nepoch):
        _log.debug(f"starting epoch {epoch_index}")

        for batch in data_loader:
            dm.optimize(batch, beta_max=0.8)

        torch.save(model.state_dict(), model_path)

    # sample
    model.load_state_dict(torch.load(model_path, map_location=device))

    s = next(iter(data_loader))

    batch = {
        "frames": torch.randn(1, 9, 7) * 10.0,
        "mask": torch.ones(1, 9, dtype=torch.bool),
        "features": s["features"][:1],
    }

    true_frames = Rigid.from_tensor_7(s["frames"])
    input_frames = Rigid.from_tensor_7(batch["frames"])

    save(true_frames[0], "dm-true.pdb")
    save(input_frames[0], "dm-input.pdb")

    with torch.no_grad():
        pred_frames = dm.sample(batch, true_frames)

    save(pred_frames[0], "dm-output.pdb")
