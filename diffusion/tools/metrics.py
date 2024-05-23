
import torch
from openfold.utils.rigid_utils import Rigid


def get_rmsd(pred_frames: Rigid, true_frames: Rigid) -> torch.Tensor:
    return torch.sqrt(((true_frames.get_trans() - pred_frames.get_trans()) ** 2).sum(dim=(-2, -1)) / pred_frames.shape[-1])
