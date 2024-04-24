from typing import Dict
from math import log
from typing import Optional

import numpy
import h5py

import torch
from torch.utils.data import Dataset


class MhcpDataset(Dataset):

    protein_size = 200
    peptide_size = 9

    keys = [
        ("aatype", torch.long),
        ("atom14_gt_exists", torch.float),
        ("atom14_gt_positions", torch.float)
    ]
    treshold = 1.0 - log(500) / log(500000)

    def __init__(self, hdf5_path: str, device: Optional[torch.device] = None):

        if device is not None:
            self.device = device
        else:
            self.device = torch.device("cpu")

        self.hdf5_path = hdf5_path
        with h5py.File(self.hdf5_path, 'r') as f5:
            self.entry_names = list(f5.keys())

    def __getitem__(self, index: int) -> Dict[str, torch.Tensor]:
        return self._get_entry(self.hdf5_path, self.entry_names[index], self.device)

    def __len__(self) -> int:
        return len(self.entry_names)

    @staticmethod
    def _get_entry(hdf5_path: str, entry_name: str, device: torch.device) -> Dict[str, torch.Tensor]:

        data = {}
        with h5py.File(hdf5_path, 'r') as f5:
            entry = f5[entry_name]

            peptide = entry["peptide"]

            x = peptide['atom14_gt_positions'][..., 1, :]
            mask = peptide['atom14_gt_exists'][..., 1]
            h = peptide['sequence_onehot']

            data['positions'] = torch.tensor(x, device=device)
            data['mask'] = torch.tensor(mask, device=device)
            data['features'] = torch.tensor(h, device=device)

        return data
