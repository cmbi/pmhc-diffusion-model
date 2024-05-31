#!/usr/bin/env python

import os
import sys
from argparse import ArgumentParser
import logging

import torch
from openfold.utils.rigid_utils import Rigid
from openfold.np.residue_constants import restypes

from diffusion.optimizer import DiffusionModelOptimizer
from diffusion.model import Model
from diffusion.data import MhcpDataset
from diffusion.tools.pdb import save


_log = logging.getLogger(__name__)

arg_parser = ArgumentParser()
arg_parser.add_argument("model")
arg_parser.add_argument("sequence")

if __name__ == "__main__":

    args = arg_parser.parse_args()
    max_len = 16
    if len(args.sequence) > max_len:
        raise RuntimeError(f"input sequence is too long, max {max_len} amino acids")

    logging.basicConfig(stream=sys.stdout, level=logging.DEBUG)

    device = torch.device("cpu")
    if torch.cuda.is_available():
        device = torch.device("cuda")

    # init
    _log.debug(f"initializing model")
    T = 1000
    H = 22
    model = Model(max_len, H, T).to(device=device)
    if os.path.isfile(args.model):
        model.load_state_dict(torch.load(args.model, map_location=device))

    _log.debug(f"initializing diffusion model optimizer")
    dm = DiffusionModelOptimizer(T, model)

    model.load_state_dict(torch.load(args.model, map_location=device))

    # build input data
    mask = torch.zeros(max_len, device=device, dtype=torch.bool)
    mask[:len(args.sequence)] = True

    one_hot_sequence = torch.nn.functional.one_hot(torch.tensor([restypes.index(args.sequence[i])
                                                                 for i in range(len(args.sequence))],
                                                                device=device),
                                                   num_classes=H)
    features = torch.zeros(max_len, H, device=device)
    features[mask] = one_hot_sequence.float()

    input_frames = dm.gen_noise([max_len], device=device)

    batch = {
        "frames": input_frames.unsqueeze(0),
        "mask": mask.unsqueeze(0),
        "features": features.unsqueeze(0),
    }

    save(input_frames, mask, f"{args.sequence}-input.pdb")

    with torch.no_grad():
        pred_frames = dm.sample(batch)

    save(pred_frames[0], mask, f"{args.sequence}-output.pdb")
