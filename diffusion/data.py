from typing import List, Dict, Optional, Union
from math import log

import numpy
import h5py

import torch
from torch.utils.data import Dataset

from openfold.utils.rigid_utils import Rigid


class MhcpDataset(Dataset):

    peptide_maxlen = 16
    pocket_maxlen = 80

    def __init__(self, hdf5_path: str, device: Optional[torch.device] = None):

        if device is not None:
            self.device = device
        else:
            self.device = torch.device("cpu")

        self.hdf5_path = hdf5_path
        with h5py.File(self.hdf5_path, 'r') as f5:
            self.entry_names = list(f5.keys())

    def __getitem__(self, index: int) -> Dict[str, torch.Tensor]:
        return self.get_entry(self.entry_names[index])

    def __len__(self) -> int:
        return len(self.entry_names)

    def get_entry(self, entry_name: str) -> Dict[str, Union[List[str], torch.Tensor]]:

        data = {}
        with h5py.File(self.hdf5_path, 'r') as f5:
            entry = f5[entry_name]

            if "peptide" not in entry:
                raise ValueError(f"no peptide in {entry_name}")

            peptide = entry["peptide"]
            mhc = entry['protein']

            # backbone rotation(quaternion) + xyz
            frames_data = peptide['backbone_rigid_tensor'][:]
            mhc_frames_data = mhc['backbone_rigid_tensor'][:]
            mhc_atoms_data = mhc['atom14_gt_positions'][:]
            mhc_atoms_exist = mhc['atom14_gt_exists'][:]
            mhc_aatype = mhc['aatype'][:]
            mhc_pocket_mask = mhc['cross_residues_mask'][:]
            pocket_n_res = mhc_pocket_mask.sum().item()

            # fetch masked pocket
            pocket_frames_data = torch.zeros(MhcpDataset.pocket_maxlen, 4, 4, device=self.device)
            pocket_frames_data[:pocket_n_res] = torch.tensor(mhc_frames_data[mhc_pocket_mask], device=self.device)
            pocket_atoms_xyz = torch.zeros(MhcpDataset.pocket_maxlen, 14, 3, device=self.device)
            pocket_atoms_xyz[:pocket_n_res] = torch.tensor(mhc_atoms_data[mhc_pocket_mask], device=self.device)
            pocket_atoms_exist = torch.zeros(MhcpDataset.pocket_maxlen, 14, device=self.device, dtype=torch.bool)
            pocket_atoms_exist[:pocket_n_res] = torch.tensor(mhc_atoms_exist[mhc_pocket_mask], device=self.device)
            pocket_aatype = torch.zeros(MhcpDataset.pocket_maxlen, device=self.device, dtype=torch.long)
            pocket_aatype[:pocket_n_res] = torch.tensor(mhc_aatype[mhc_pocket_mask], device=self.device)
            pocket_mask = torch.zeros(MhcpDataset.pocket_maxlen, device=self.device, dtype=torch.bool)
            pocket_mask[:pocket_n_res] = True

            peptide_len = frames_data.shape[0]

            # masking frames with identity frames
            frames = torch.eye(4, 4, device=self.device).unsqueeze(-3).expand(MhcpDataset.peptide_maxlen, 4, 4).clone()
            frames[:peptide_len, :, :] = torch.tensor(frames_data, device=self.device)

            pocket_frames = torch.eye(4, 4, device=self.device).unsqueeze(-3).expand(MhcpDataset.pocket_maxlen, 4, 4).clone()
            pocket_frames[:pocket_n_res, :, :] = pocket_frames_data[:pocket_n_res]

            # backbone reswise mask
            mask = torch.zeros(MhcpDataset.peptide_maxlen, device=self.device, dtype=torch.bool)
            mask[:peptide_len] = True

            # aatype index
            aatype = torch.zeros(MhcpDataset.peptide_maxlen, device=self.device, dtype=torch.long)
            aatype[:peptide_len] = torch.tensor(peptide['aatype'][:], device=self.device, dtype=torch.long)

            # one-hot encoded amino acid sequence
            onehot = torch.zeros([MhcpDataset.peptide_maxlen, 22], device=self.device)
            onehot[:peptide_len, :] = torch.tensor(peptide['sequence_onehot'][:], device=self.device)

            pocket_onehot = torch.zeros(MhcpDataset.pocket_maxlen, 22, device=self.device)
            pocket_onehot[:pocket_n_res] = torch.tensor(mhc['sequence_onehot'][:], device=self.device)[mhc_pocket_mask]

            # torsion angles: pre-omega, phi, psi, chi-1, chi-2, chi-3, chi-4
            torsions = torch.empty(MhcpDataset.peptide_maxlen, 7, 2, device=self.device)
            torsions[:peptide_len] = torch.tensor(peptide['torsion_angles_sin_cos'][:], device=self.device)
            torsions_mask = torch.zeros(MhcpDataset.peptide_maxlen, 7, device=self.device, dtype=torch.bool)
            torsions_mask[:peptide_len] = torch.tensor(peptide['torsion_angles_mask'][:], device=self.device)
            # The frames determine the backbone structure,
            # so disable backbone torsions, except for the C-terminus.
            torsions_mask[:, :3] = False
            torsions_mask[peptide_len - 1, 2] = True
            # set identity, where masked
            torsions[torch.logical_not(torsions_mask)] = torch.tensor([0.0, 1.0], device=self.device)

            # output dict
            data['name'] = [entry_name]
            data['mask'] = mask
            data['frames'] = Rigid.from_tensor_4x4(frames).to_tensor_7()  # convert to tensor, for collation
            data['features'] = onehot
            data['aatype'] = aatype
            data['torsions'] = torsions
            data['torsions_mask'] = torsions_mask
            data['pocket_aatype'] = pocket_aatype
            data['pocket_features'] = pocket_onehot
            data['pocket_mask'] = pocket_mask
            data['pocket_frames'] = Rigid.from_tensor_4x4(pocket_frames).to_tensor_7()  # convert to tensor, for collation
            data['pocket_atom14_positions'] = pocket_atoms_xyz
            data['pocket_atom14_exists'] = pocket_atoms_exist

        return data

    def get_protein_positions(self, entry_name: str) -> Dict[str, numpy.ndarray]:

        data = {}
        with h5py.File(self.hdf5_path, 'r') as f5:
            entry = f5[entry_name]

            mhc = entry['protein']

            data["protein_aatype"] = torch.tensor(mhc['aatype'][:], device=self.device)
            data["protein_atom14_positions"] = torch.tensor(mhc['atom14_gt_positions'][:], device=self.device)
            data["protein_atom14_exists"] = torch.tensor(mhc['atom14_gt_exists'][:], device=self.device)

        return data
