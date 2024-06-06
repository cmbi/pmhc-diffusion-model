#!/usr/bin/env python

import os
import sys
from argparse import ArgumentParser
import logging

import torch
from openfold.utils.rigid_utils import Rigid

from diffusion.optimizer import DiffusionModelOptimizer
from diffusion.model import Model
from diffusion.data import MhcpDataset
from diffusion.tools.pdb import save


_log = logging.getLogger(__name__)

arg_parser = ArgumentParser()
arg_parser.add_argument("model")
arg_parser.add_argument("test_hdf5")
arg_parser.add_argument("id")
arg_parser.add_argument("--debug", "-d", action="store_const", const=True, default=False)

if __name__ == "__main__":

    args = arg_parser.parse_args()

    log_level = logging.INFO
    if args.debug:
        log_level = logging.DEBUG

    logging.basicConfig(stream=sys.stdout, level=log_level)

    device = torch.device("cpu")
    if torch.cuda.is_available():
        device = torch.device("cuda")

    # init
    _log.debug(f"initializing model")
    T = 1000
    model = Model(16, 22, T).to(device=device)
    if os.path.isfile(args.model):
        model.load_state_dict(torch.load(args.model, map_location=device))

    _log.debug(f"initializing diffusion model optimizer")
    dm = DiffusionModelOptimizer(T, model)

    # sample
    model.load_state_dict(torch.load(args.model, map_location=device))

    test_dataset = MhcpDataset(args.test_hdf5, device)
    test_entry = test_dataset.get_entry(args.id)

    # for sampling, make a batch of size 1
    batch = {k: test_entry[k].unsqueeze(0) for k in test_entry}

    true_frames = Rigid.from_tensor_7(batch["frames"])
    input_frames = dm.gen_noise(true_frames.shape, device=device)

    batch["frames"] = input_frames

    save(true_frames[0], test_entry["mask"], test_entry["pocket_aatype"], test_entry["pocket_atom14_positions"], test_entry["pocket_atom14_exists"], f"{args.id}-true.pdb")
    save(input_frames[0], test_entry["mask"], test_entry["pocket_aatype"], test_entry["pocket_atom14_positions"], test_entry["pocket_atom14_exists"], f"{args.id}-input.pdb")

    with torch.no_grad():
        pred_frames = dm.sample(batch)

    save(pred_frames[0], test_entry["mask"], test_entry["pocket_aatype"], test_entry["pocket_atom14_positions"], test_entry["pocket_atom14_exists"], f"{args.id}-output.pdb")
