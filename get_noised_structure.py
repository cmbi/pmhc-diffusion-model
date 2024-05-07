#!/usr/bin/env python

from argparse import ArgumentParser
from math import sqrt

import torch

from diffusion.data import MhcpDataset

from Bio.PDB.Structure import Structure
from Bio.PDB.Model import Model as PDBModel
from Bio.PDB.Chain import Chain
from Bio.PDB.Residue import Residue
from Bio.PDB.Atom import Atom
from Bio.PDB.PDBIO import PDBIO


arg_parser = ArgumentParser()
arg_parser.add_argument("hdf5_path")
arg_parser.add_argument("beta", type=float)
arg_parser.add_argument("output_path")


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


if __name__ == "__main__":

    args = arg_parser.parse_args()

    device = torch.device("cpu")

    dataset = MhcpDataset(args.hdf5_path, device)

    batch = dataset[0]
    x = batch["positions"]

    alpha = 1.0 - args.beta
    sigma = sqrt(1.0 - alpha * alpha)

    eps = torch.randn(x.shape)

    z = alpha * x + sigma * eps

    save(z, args.output_path)
