#!/usr/bin/env python

from argparse import ArgumentParser
from math import sqrt

import torch

from openfold.utils.rigid_utils import Rigid

from diffusion.data import MhcpDataset
from diffusion.tools.pdb import save
from diffusion.optimizer import DiffusionModelOptimizer


arg_parser = ArgumentParser()
arg_parser.add_argument("hdf5_path")
arg_parser.add_argument("beta", type=float)
arg_parser.add_argument("output_path")


if __name__ == "__main__":

    args = arg_parser.parse_args()

    device = torch.device("cpu")

    dataset = MhcpDataset(args.hdf5_path, device)

    batch = dataset[0]
    frames = batch["frames"]

    alpha = 1.0 - args.beta
    sigma = sqrt(1.0 - alpha * alpha)

    noise = torch.randn(frames.shape)

    frames = DiffusionModelOptimizer.interpolate(Rigid.from_tensor_7(frames), Rigid.from_tensor_7(noise), alpha, sigma)

    save(frames, args.output_path)
