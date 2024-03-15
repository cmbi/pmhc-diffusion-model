from typing import Dict
from math import log

import numpy
import h5py

import torch
from torch.utils.data import Dataset


class MhcpDataset(Dataset):

    keys = ["atom14_gt_exists", "atom14_gt_positions"]
    treshold = 1.0 - log(500) / log(500000)

    def __init__(self, hdf5_path: str):
        self.hdf5_path = hdf5_path
        with h5py.File(self.hdf5_path, 'r') as f5:
            self.entry_names = list(f5.keys())

    def __getitem__(self, index: int) -> Dict[str, torch.Tensor]:
        return self._get_entry(self.hdf5_path, self.entry_names[index])

    def __len__(self) -> int:
        return len(self.entry_names)

    @staticmethod
    def _get_entry(hdf5_path: str, entry_name: str) -> Dict[str, torch.Tensor]:

        data = {}
        with h5py.File(hdf5_path, 'r') as f5:
            entry = f5[entry_name]

            peptide = entry["peptide"]
            protein = entry["protein"]

            for key in MhcpDataset.keys:
                data[f"peptide_{key}"] = torch.tensor(peptide[key][()])

            for key in MhcpDataset.keys:
                data[f"protein_{key}"] = torch.tensor(protein[key][()])

        return data
