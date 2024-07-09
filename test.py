#!/usr/bin/env python

import os
import sys
from argparse import ArgumentParser
import logging

import torch
from torch.utils.data import DataLoader
from openfold.utils.rigid_utils import Rigid

from diffusion.optimizer import DiffusionModelOptimizer
from diffusion.model import Model
from diffusion.data import MhcpDataset
from diffusion.tools.pdb import save


_log = logging.getLogger(__name__)

arg_parser = ArgumentParser()
arg_parser.add_argument("model", help="model parameters file")
arg_parser.add_argument("test_hdf5", help="test data")
arg_parser.add_argument("--debug", "-d", action="store_const", const=True, default=False, help="run in debug mode")
arg_parser.add_argument("-T", type=int, default=1000, help="number of noise steps")
arg_parser.add_argument("--batch-size", "-b", type=int, help="data batch size", default=64)
arg_parser.add_argument("--num-workers", "-w", type=int, help="number of batch loading workers", default=4)

if __name__ == "__main__":

    args = arg_parser.parse_args()

    # init logger
    log_level = logging.INFO
    if args.debug:
        log_level = logging.DEBUG

    logging.basicConfig(stream=sys.stdout, level=log_level)

    # select device
    device = torch.device("cpu")
    if torch.cuda.is_available():
        device = torch.device("cuda")

    # init model & optimizer
    _log.debug(f"initializing model")
    model = Model(16, 22, args.T).to(device=device)
    if os.path.isfile(args.model):
        model.load_state_dict(torch.load(args.model, map_location=device))

    _log.debug(f"initializing diffusion model optimizer")
    dm = DiffusionModelOptimizer(args.T, model, 0.0)

    # load model state from input file
    model.load_state_dict(torch.load(args.model, map_location=device))

    # open dataset
    test_dataset = MhcpDataset(args.test_hdf5, device)

    # get output directory
    output_path = os.path.splitext(args.test_hdf5)[0] + "-sampled"
    if not os.path.isdir(output_path):
        os.mkdir(output_path)

    with torch.no_grad():
        for true_batch in DataLoader(test_dataset, batch_size=args.batch_size, num_workers=args.num_workers):

            # get entry names
            names = list(true_batch['name'][0])

            # noisify
            noise = dm.gen_noise(true_batch["frames"].shape[:-1], device=device)
            input_batch = {k: true_batch[k] for k in true_batch}
            input_batch["frames"] = noise["frames"].to_tensor_7()
            input_batch["torsions"] = noise["torsions"]

            # denoisify
            pred_batch = dm.sample(input_batch)

            # add all protein residues
            pred_batch.update(test_dataset.get_protein_positions(names))

            # save denoisified data
            for i, name in enumerate(names):
                save(pred_batch, i, f"{output_path}/{name}.pdb")
