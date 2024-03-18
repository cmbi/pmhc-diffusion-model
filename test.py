#!/usr/bin/env python

import torch
import random
import sys

from Bio.PDB.Atom import Atom
from Bio.PDB.Residue import Residue
from Bio.PDB.Chain import Chain
from Bio.PDB.Model import Model
from Bio.PDB.Structure import Structure

from torch.utils.data import DataLoader

from diffusionmodel.optimizer import DiffusionModelOptimizer
from diffusionmodel.data import MhcpDataset

from openfold.np.residue_constants import restype_name_to_atom14_names, restype_1to3, restypes


class Model(torch.nn.Module):
    def __init__(self, T: int):
        super(Model, self).__init__()

        trans = 32
        n = 9 * 14
        dim = 3

        self.mlps = torch.nn.ModuleList([
            torch.nn.Sequential(
                torch.nn.Linear(n * dim, trans),
                torch.nn.ReLU(),
                torch.nn.Linear(trans, n * dim),
            )
            for t in range(T)]
        )

    def forward(self, z: torch.Tensor, t: int) -> torch.Tensor:

        shape = z.shape
        z = z.reshape(shape[0], -1)

        e = self.mlps[t](z)

        return e.reshape(shape)


if __name__ == "__main__":

    dim = 3
    n = 10
    b = 8
    trans = 32
    T = 10

    model = Model(T)

    dm = DiffusionModelOptimizer(T, model)

    m = torch.randn(n, dim)

    dataset = MhcpDataset(sys.argv[1])
    data_loader = DataLoader(dataset, shuffle=True)

    nepoch = 100

    for _ in range(nepoch):
        for batch in data_loader:

            x = batch["peptide_atom14_gt_positions"]

            dm.optimize(x)

    torch.save(model.state_dict(), "model.pth")

    aatype = torch.randint(0, 20, (1, 9))

    z0 = dm.sample(aatype)

    structure = Structure("sample-" + "".join([restypes[i] for i in aatype[0]]))

    model = Model('1')
    structure.add(model)

    chain_id = "P"
    chain = Chain(chain_id)
    model.add(chain)

    for aa_index in aatype:
        aa_letter = restypes[aa_index]
        aa_name = restype_1to3[aa_letter]

        residue = Residue(aa_index, aa_name, chain_id)
        chain.add(residue)

        for atom_index, atom_name in enumerate(restype_name_to_atom14_names[aa_name]):
            if len(atom_name) > 0:
                atom = Atom(name, z0[0, aa_index, atom_index, :], element=atom_name[0])
                residue.add(atom)
