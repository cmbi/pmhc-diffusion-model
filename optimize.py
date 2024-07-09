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
from diffusion.tools.metrics import MetricsRecord


_log = logging.getLogger(__name__)


arg_parser = ArgumentParser()
arg_parser.add_argument("train_hdf5", help="train data")
arg_parser.add_argument("epoch_count", type=int, help="number of epochs over the data")
arg_parser.add_argument("output_model", help="output model parameters file")
arg_parser.add_argument("--debug", "-d", action="store_const", const=True, default=False, help="run in debug mode")
arg_parser.add_argument("-T", type=int, help="number of noise steps", default=1000)
arg_parser.add_argument("--batch-size", "-b", type=int, help="data batch size", default=64)
arg_parser.add_argument("--num-workers", "-w", type=int, help="number of batch loading workers", default=4)
arg_parser.add_argument("--lr", type=float, help="learning rate", default=0.001)


if __name__ == "__main__":

    args = arg_parser.parse_args()

    # init logger
    log_level = logging.INFO
    if args.debug:
        torch.autograd.detect_anomaly(check_nan=True)
        log_level = logging.DEBUG

    logging.basicConfig(stream=sys.stdout, level=log_level)

    # select device
    device = torch.device("cpu")
    if torch.cuda.is_available():
        device = torch.device("cuda")

    # init model & optimizer
    _log.debug(f"initializing model")
    model = Model(16, 22, args.T).to(device=device)
    if os.path.isfile(args.output_model):
        model.load_state_dict(torch.load(args.output_model, map_location=device), strict=True)

    _log.debug(f"initializing diffusion model optimizer")
    dm = DiffusionModelOptimizer(args.T, model, args.lr)

    # load dataset
    train_dataset = MhcpDataset(args.train_hdf5, device)
    train_data_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers)

    # train
    metrics_path = args.output_model.replace('.pth', '.csv')
    for epoch_index in range(args.epoch_count):
        _log.debug(f"starting epoch {epoch_index}")

        metrics = MetricsRecord()

        for i, batch in enumerate(train_data_loader):
            dm.optimize(batch, metrics)

            if i > 0 and i % 100 == 0:
                torch.save(model.state_dict(), args.output_model)
                _log.debug(f"saved {args.output_model}")

        torch.save(model.state_dict(), args.output_model)
        _log.debug(f"saved {args.output_model}")

        metrics.save(f"{metrics_path}", epoch_index)
