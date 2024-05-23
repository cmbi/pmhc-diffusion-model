import random
from typing import Tuple, Dict, Optional, Union
from math import sqrt, exp, pi
import logging

import torch
from openfold.utils.rigid_utils import Rigid, Rotation, invert_quat, quat_multiply

from .tools.metrics import get_rmsd


_log  = logging.getLogger(__name__)


def square(x: float) -> float:
    return x * x



class DiffusionModelOptimizer:

    def __init__(self, noise_step_count: int, model: torch.nn.Module):

        self.noise_step_count = noise_step_count
        self.model = model
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=0.001)

    @staticmethod
    def get_loss(frames_true: Rigid, frames_pred: Rigid, mask: torch.Tensor) -> torch.Tensor:

        # position vectors
        positions_loss = torch.square(frames_true.get_trans() - frames_pred.get_trans()).sum(dim=(-2, -1)) / mask.sum(dim=-1)

        # normalize rotation quaternions
        rotations_true = frames_true.get_rots().get_quats()
        rotations_pred = frames_pred.get_rots().get_quats()

        rotations_true_norm = torch.sqrt(torch.square(rotations_true).sum(dim=-1))
        rotations_pred_norm = torch.sqrt(torch.square(rotations_pred).sum(dim=-1))

        rotations_true = rotations_true / rotations_true_norm.unsqueeze(-1)
        rotations_pred = rotations_pred / rotations_pred_norm.unsqueeze(-1)

        dots = (rotations_pred * rotations_true).sum(dim=-1)
        angles = torch.acos(dots)

        rotations_loss = torch.square(angles).sum(dim=-1) / mask.sum(dim=-1)

        return positions_loss + rotations_loss

    @staticmethod
    def combine(signal: Rigid, noise: Rigid, alpha: float, sigma: float) -> Rigid:

        # position vectors
        positions = signal.get_trans() * alpha + sigma * noise.get_trans()

        # rotation quaternions
        rotations_signal = signal.get_rots().get_quats()
        rotations_noise = noise.get_rots().get_quats()

        rotations_signal_norm = torch.sqrt(torch.square(rotations_signal).sum(dim=-1))
        rotations_noise_norm = torch.sqrt(torch.square(rotations_noise).sum(dim=-1))

        rotations_signal = rotations_signal / rotations_signal_norm.unsqueeze(dim=-1)
        rotations_noise = rotations_noise / rotations_noise_norm.unsqueeze(dim=-1)

        # slerp
        dots = (rotations_signal * rotations_noise).sum(dim=-1)
        dots = torch.where(dots < 0.0, -dots, dots)
        angles = torch.acos(dots)
        rotations = rotations_signal * (torch.sin(alpha * angles) / torch.sin(angles)).unsqueeze(-1) + \
                    rotations_noise * (torch.sin(sigma * angles) / torch.sin(angles)).unsqueeze(-1)

        return Rigid(Rotation(quats=rotations, normalize_quats=True), positions)

    def optimize(self, batch: Dict[str, torch.Tensor], beta_max: Optional[float] = 1.0):

        frames = batch["frames"]

        t = random.randint(0, self.noise_step_count - 1)

        self.optimizer.zero_grad()

        epsilon = Rigid.from_tensor_7(torch.randn(frames.shape, device=frames.device) * 10.0)

        beta_min = 0.0001
        beta_delta = beta_max - beta_min
        beta = beta_min + beta_delta * t / self.noise_step_count
        alpha = 1.0 - beta
        sigma = sqrt(1.0 - square(alpha))

        zt = self.combine(Rigid.from_tensor_7(frames), epsilon, alpha, sigma)

        batch = {
            "frames": zt,
            "mask": batch["mask"],
            "features": batch["features"],
        }

        loss = self.get_loss(epsilon, self.model(batch, t), batch["mask"]).mean()
        if loss.isnan().any():
            raise RuntimeError("NaN loss")

        _log.debug(f"beta is {beta:.3f}, loss is {loss:.3f}")

        loss.backward()

        self.optimizer.step()

    def sample(self, batch: Dict[str, Union[torch.Tensor, Rigid]], true_x: Rigid) -> Rigid:

        data_shape = batch["frames"].shape

        zt = Rigid.from_tensor_7(batch["frames"])

        t = self.noise_step_count
        while t > 0:

            s = t - 1

            epsilon = Rigid.from_tensor_7(torch.randn(data_shape, device=zt.device))

            beta_t = 0.0001 + 0.9 * t / self.noise_step_count
            alpha_t = 1.0 - beta_t
            sigma_t = sqrt(1.0 - square(alpha_t))

            if sigma_t == 0.0:
                raise RuntimeError("zero sigma")

            beta_s = 0.0001 + 0.9 * s / self.noise_step_count
            alpha_s = 1.0 - beta_s
            sigma_s = sqrt(1.0 - square(alpha_s))

            if alpha_s == 0.0:
                raise RuntimeError("zero alpha")

            alpha_ts = alpha_t / alpha_s
            sqr_sigma_ts = square(sigma_t) - square(sigma_s) * alpha_ts

            sigma_ts = sqrt(sqr_sigma_ts)
            sigma_t2s = sigma_ts * sigma_s / sigma_t

            batch = {
                "frames": zt,
                "mask": batch["mask"],
                "features": batch["features"],
            }

            zs = self.combine(
                self.combine(zt, self.model(batch, t), (1.0 / alpha_ts), -sqr_sigma_ts / (alpha_ts * sigma_t)),
                epsilon,
                1.0,
                sigma_t2s
            )

            rmsd = get_rmsd(zt, true_x).mean().item()
            _log.debug(f"at t={t}, rmsd={rmsd}")

            zt = zs
            t = s

        x = zt
        return x

