#!/usr/bin/env python

import os
import random
import sys
import logging

from argparse import ArgumentParser


import torch
from torch.utils.data import DataLoader
from torch.nn.functional import one_hot

from diffusion.optimizer import DiffusionModelOptimizer
from diffusion.model import Model
from diffusion.data import MhcpDataset


_log = logging.getLogger(__name__)


arg_parser = ArgumentParser()
arg_parser.add_argument("train_hdf5")
arg_parser.add_argument("epoch_count", type=int)
arg_parser.add_argument("output_model")


if __name__ == "__main__":

    args = arg_parser.parse_args()

    logging.basicConfig(stream=sys.stdout, level=logging.DEBUG)

    device = torch.device("cpu")
    if torch.cuda.is_available():
        device = torch.device("cuda")

    # init
    torch.autograd.detect_anomaly(check_nan=True)

    _log.debug(f"initializing model")
    T = 1000
    model = Model(16, 22, T).to(device=device)
    if os.path.isfile(args.output_model):
        model.load_state_dict(torch.load(args.output_model, map_location=device))

    _log.debug(f"initializing diffusion model optimizer")
    dm = DiffusionModelOptimizer(T, model)

    train_dataset = MhcpDataset(args.train_hdf5, device)
    train_data_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)

    # train
    for epoch_index in range(args.epoch_count):
        _log.debug(f"starting epoch {epoch_index}")

        for batch in train_data_loader:
            dm.optimize(batch)

        torch.save(model.state_dict(), args.output_model)
