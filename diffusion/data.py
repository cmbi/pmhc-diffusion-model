from typing import Dict
from math import log
from typing import Optional

import numpy
import h5py

import torch
from torch.utils.data import Dataset

from openfold.utils.rigid_utils import Rigid


class MhcpDataset(Dataset):

    protein_size = 200
    peptide_size = 9

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

            # backbone rotation(quaternion) + c-alpha xyz
            frames = peptide['backbone_rigid_tensor'][:]

            # backbone reswise mask
            mask = peptide['backbone_rigid_mask'][:]

            # one-hot encoded amino acid sequence
            h = peptide['sequence_onehot'][:]

            data['mask'] = torch.tensor(mask, device=device)
            data['frames'] = Rigid.from_tensor_4x4(torch.tensor(frames, device=device)).to_tensor_7()
            data['features'] = torch.tensor(h, device=device)

        return data
