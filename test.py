#!/usr/bin/env python

import random
import sys
import logging

from Bio.PDB.Atom import Atom
from Bio.PDB.Residue import Residue
from Bio.PDB.Chain import Chain
from Bio.PDB.Model import Model as StructureModel
from Bio.PDB.Structure import Structure

import torch
from torch.utils.data import DataLoader
from torch.nn.functional import one_hot

from diffusionmodel.optimizer import DiffusionModelOptimizer
from diffusionmodel.data import MhcpDataset

from openfold.np.residue_constants import restype_name_to_atom14_names, restype_1to3, restypes


_log = logging.getLogger(__name__)


class Model(torch.nn.Module):
    def __init__(self, T: int):
        super(Model, self).__init__()

        trans = 32

        self.mlp = torch.nn.Sequential(
            torch.nn.Linear(14 * 3 + 20, trans),
            torch.nn.ReLU(),
            torch.nn.Linear(trans, 14 * 3),
        )

    def forward(
        self, 
        z: torch.Tensor,
        aatype: torch.Tensor
    ) -> torch.Tensor:

        shape = z.shape
        z = z.reshape(shape[0], -1)

        e = self.mlp(torch.cat((z, one_hot(aatype, 20)), dim=-1))

        return e.reshape(shape)


if __name__ == "__main__":

    device = torch.device("cuda")

    logging.basicConfig(filename="diffusion.log", filemode='a', level=logging.DEBUG)

    dim = 3
    n = 10
    b = 8
    trans = 32
    T = 10

    model = Model(T).to(device=device)

    dm = DiffusionModelOptimizer(T, model)

    m = torch.randn(n, dim)

    dataset = MhcpDataset(sys.argv[1], device=device)
    data_loader = DataLoader(dataset, batch_size=8, shuffle=True)

    nepoch = 10

    for epoch_index in range(nepoch):
        _log.debug(f"starting epoch {epoch_index}")

        for batch in data_loader:

            x = batch["peptide_atom14_gt_positions"]
            aatype = batch["peptide_aatype"]

            dm.optimize(x, aatype)

        torch.save(model.state_dict(), "model.pth")

    aatype = torch.randint(0, 20, (1, 9))

    z0 = dm.sample(aatype)

    structure = Structure("sample-" + "".join([restypes[i] for i in aatype[0]]))

    structure_model = StructureModel('1')
    structure.add(model)

    chain_id = "P"
    chain = Chain(chain_id)
    structure_model.add(chain)

    for aa_index in aatype:
        aa_letter = restypes[aa_index]
        aa_name = restype_1to3[aa_letter]

        residue = Residue(aa_index, aa_name, chain_id)
        chain.add(residue)

        for atom_index, atom_name in enumerate(restype_name_to_atom14_names[aa_name]):
            if len(atom_name) > 0:
                atom = Atom(name, z0[0, aa_index, atom_index, :], element=atom_name[0])
                residue.add(atom)
