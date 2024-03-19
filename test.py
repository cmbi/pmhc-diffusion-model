#!/usr/bin/env python

import os
import random
import sys
import logging

from Bio.PDB.PDBIO import PDBIO
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

        self.T = T

        trans = 32

        self.mlp = torch.nn.Sequential(
            torch.nn.Linear(9 * (14 * 3 + 20) + T, trans),
            torch.nn.ReLU(),
            torch.nn.Linear(trans, 9 * 14 * 3),
        )

    def forward(
        self, 
        z: torch.Tensor,
        aatype: torch.Tensor,
        t: int,
    ) -> torch.Tensor:

        shape = z.shape
        z = z.reshape(shape[0], shape[1] * 14 * 3)

        t_onehot = torch.nn.functional.one_hot(torch.tensor(t, device=z.device), self.T).unsqueeze(0).expand(shape[0], self.T)

        aatype_onehot = one_hot(aatype.long(), 20).reshape(shape[0], 20 * 9)

        e = self.mlp(torch.cat((
            z,
            aatype_onehot,
            t_onehot,
        ), dim=-1))

        return e.reshape(shape)


def write_structure(x: torch.Tensor, aatype: torch.Tensor, path: str):

    structure = Structure("".join([restypes[i] for i in aatype]))

    structure_model = StructureModel('1')
    structure.add(structure_model)

    chain_id = "P"
    chain = Chain(chain_id)
    structure_model.add(chain)

    n = 0
    for residue_index, aa_index in enumerate(aatype):
        aa_letter = restypes[aa_index]
        aa_name = restype_1to3[aa_letter]

        residue = Residue((' ', residue_index, ' '), aa_name, chain_id)
        chain.add(residue)

        for atom_index, atom_name in enumerate(restype_name_to_atom14_names[aa_name]):
            if len(atom_name) > 0:
                n += 1
                atom = Atom(
                    atom_name, x[residue_index, atom_index, :],
                    bfactor=0.0,
                    occupancy=1.0,
                    altloc=' ',
                    fullname=f" {atom_name} ",
                    element=atom_name[0],
                    serial_number=n,
                )
                residue.add(atom)

    io = PDBIO()
    io.set_structure(structure)
    io.save(path)


if __name__ == "__main__":

    logging.basicConfig(filename="diffusion.log", filemode='a', level=logging.DEBUG)

    _log.debug(f"initializing model")
    T = 10
    device = torch.device("cuda")
    model = Model(T).to(device=device)
    if os.path.isfile("model.pth"):
        model.load_state_dict(torch.load("model.pth", map_location=device))

    _log.debug(f"initializing diffusion model optimizer")
    dm = DiffusionModelOptimizer(T, model)

    _log.debug(f"initializing dataset")
    dataset = MhcpDataset(sys.argv[1], device=device)
    data_size = len(dataset)
    data_loader = DataLoader(dataset, batch_size=64, shuffle=True)

    nepoch = 1000
    for epoch_index in range(nepoch):
        _log.debug(f"starting epoch {epoch_index}")

        n_passed = 0
        for batch in data_loader:

            x = batch["peptide_atom14_gt_positions"]
            aatype = batch["peptide_aatype"]

            dm.optimize(x, aatype)

            n_passed += batch["peptide_aatype"].shape[0]
            _log.debug(f"{n_passed}/{data_size} passed ({round(100.0 * n_passed / data_size, 1)} %)")

        torch.save(model.state_dict(), "model.pth")

    model.load_state_dict(torch.load("model.pth"))
    aatype = torch.randint(0, 20, (1, 9), device=device)
    sample_name = "sample-" + "".join([restypes[i] for i in aatype[0]])
    sample_path = sample_name + ".pdb"

    _log.debug(f"sampling")
    z0 = dm.sample(aatype)

    _log.debug(f"writing")
    write_structure(z0[0], aatype[0], sample_path)


