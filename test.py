#!/usr/bin/env python

import os
import random
import sys
import logging
from math import sqrt, log
from typing import Dict

from diffusion.data import MhcpDataset

import torch
from torch.utils.data import DataLoader
from torch.nn.functional import one_hot

from Bio.PDB.Structure import Structure
from Bio.PDB.Model import Model as PDBModel
from Bio.PDB.Chain import Chain
from Bio.PDB.Residue import Residue
from Bio.PDB.Atom import Atom
from Bio.PDB.PDBIO import PDBIO

from diffusion.optimizer import DiffusionModelOptimizer



_log = logging.getLogger(__name__)


def save(x: torch.Tensor, path: str):

    structure = Structure("")
    model = PDBModel(0)
    structure.add(model)
    chain = Chain('A')
    model.add(chain)
    for i, p in enumerate(x):
        res = Residue(("A", i + 1, " "), "ALA", "A")
        chain.add(res)

        atom = Atom("CA", p, 0.0, 1.0, ' ', " CA ", "C")
        res.add(atom)

    io = PDBIO()
    io.set_structure(structure)
    io.save(path)


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
    def __init__(self, M: int, H: int):
        super().__init__()

        self.feature_mlp = torch.nn.Linear((M + H), H)
        self.position_mlp = torch.nn.Linear(M, 3)
        self.message_mlp = torch.nn.Sequential(
            torch.nn.Linear(2 * H + 1, M),
            torch.nn.ReLU(),
        )

    def forward(self, x: torch.Tensor, h: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: [*, N, 3] (positions)
            h: [*, N, H] (other features)
            mask: [*, N]

        Returns:
            updated x: [*, N, 3]
            updated h: [*, N, H]
        """

        N = x.shape[-2]
        H = h.shape[-1]

        # [*, N, N]
        mask2 = torch.logical_and(mask.unsqueeze(-2), mask.unsqueeze(-1))
        diagonal_index = torch.arange(N).unsqueeze(-1).expand(N, 2)
        mask2[:, diagonal_index] = False

        # [*, N, N, 3]
        r = (x.unsqueeze(-3) - x.unsqueeze(-2)) * mask2.unsqueeze(-1)

        # [*, N, N]
        d2 = (r ** 2).sum(-1)

        # [*, N, N, H]
        hi = h.unsqueeze(-3).expand(-1, N, N, H)
        hj = h.unsqueeze(-2).expand(-1, N, N, H)

        # [*, N, N, M]
        m = self.message_mlp(torch.cat((hi, hj, d2.unsqueeze(-1)), dim=-1)) * mask2.unsqueeze(-1)

        # [*, N, N, 3]
        mx = self.position_mlp(m) * r

        x = x + mx.sum(dim=-2) / (N - 1)

        h = self.feature_mlp(torch.cat((h, m.sum(dim=-2)), dim=-1))

        return x, h


class Model(torch.nn.Module):
    def __init__(self, M: int, H: int):
        super().__init__()

        self.posenc = PositionalEncoding(H)

        self.egnn1 = EGNNLayer(M, H)
        self.egnn2 = EGNNLayer(M, H)
        self.act = torch.nn.ReLU()

    def forward(
        self,
        batch: Dict[str, torch.Tensor],
        t: int,
    ) -> torch.Tensor:

        x = batch['positions']
        h = batch['features']
        mask = batch['mask']

        h = self.posenc(h)

        x, h = self.egnn1(x, h, mask)
        h = self.act(h)
        x, h = self.egnn2(x, h, mask)

        return x


if __name__ == "__main__":

    hdf5_path = sys.argv[1]

    logging.basicConfig(filename="diffusion.log", filemode='a', level=logging.DEBUG)

    device = torch.device("cpu")

    data_loader = DataLoader(MhcpDataset(hdf5_path, device), batch_size=64, shuffle=True)

    _log.debug(f"initializing model")
    T = 100
    model = Model(16, 22).to(device=device)
    if os.path.isfile("model.pth"):
        model.load_state_dict(torch.load("model.pth", map_location=device))

    _log.debug(f"initializing diffusion model optimizer")
    dm = DiffusionModelOptimizer(T, model)

    nepoch = 100
    for epoch_index in range(nepoch):
        _log.debug(f"starting epoch {epoch_index}")

        for batch in data_loader:

            dm.optimize(batch)

        torch.save(model.state_dict(), "model.pth")

    model.load_state_dict(torch.load("model.pth", map_location=device))

    x = next(iter(data_loader))["positions"]

    batch = {
        "positions": x.std() * torch.randn(1, 9, 3),
        "mask": torch.ones(1, 9, dtype=torch.bool),
        "features": torch.nn.functional.one_hot(torch.randint(0, 21, (9,)), 22).unsqueeze(0),
    }

    x = dm.sample(batch)
    save(x[0], "dm.pdb")

