from typing import Dict
from math import log
from typing import Optional

import numpy
import h5py

import torch
from torch.utils.data import Dataset

from openfold.utils.rigid_utils import Rigid


class MhcpDataset(Dataset):

    peptide_maxlen = 16

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

    def _get_entry(self, hdf5_path: str, entry_name: str, device: torch.device) -> Dict[str, torch.Tensor]:

        data = {}
        with h5py.File(hdf5_path, 'r') as f5:
            entry = f5[entry_name]

            if "peptide" not in entry:
                raise ValueError(f"no peptide in {entry_name}")

            peptide = entry["peptide"]

            # backbone rotation(quaternion) + c-alpha xyz
            frames_data = peptide['backbone_rigid_tensor'][:]

            peptide_len = frames_data.shape[0]

            frames = torch.zeros([MhcpDataset.peptide_maxlen, 4, 4], device=self.device)
            frames[:peptide_len, :, :] = torch.tensor(frames_data, device=self.device)

            # backbone reswise mask
            mask = torch.zeros(MhcpDataset.peptide_maxlen, device=self.device, dtype=torch.bool)
            mask[:peptide_len] = True

            # one-hot encoded amino acid sequence
            onehot = torch.zeros([MhcpDataset.peptide_maxlen, 22], device=self.device)
            onehot[:peptide_len, :] = torch.tensor(peptide['sequence_onehot'][:], device=self.device)

            # output dict
            data['mask'] = mask
            data['frames'] = Rigid.from_tensor_4x4(frames).to_tensor_7()  # convert to tensor, for collation
            data['features'] = onehot

        return data
