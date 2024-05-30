import random
from typing import Tuple, Dict, Optional, Union, List
from math import sqrt, exp, pi
import logging

import torch
from openfold.utils.rigid_utils import Rigid, Rotation, invert_quat, quat_multiply, rot_to_quat
from openfold.utils.loss import compute_fape

from .tools.frame import get_rmsd
from .tools.pdb import save


_log  = logging.getLogger(__name__)


def square(x: float) -> float:
    return x * x


def partial_rot(rot: Rotation, amount: float) -> Rotation:

    q = rot.get_quats()
    a = torch.acos(q[..., :1])
    x = torch.nn.functional.normalize(q[..., 1:], dim=-1)

    return Rotation(quats=torch.cat((torch.cos(a * amount), torch.sin(a * amount) * x), dim=-1), normalize_quats=True)


def quat_multiply(q1: torch.Tensor, q2: torch.Tensor) -> torch.Tensor:

    w = (q1[..., 0] * q2[..., 0] - q1[..., 1] * q2[..., 1] - q1[..., 2] * q2[..., 2] - q1[..., 3] * q2[..., 3]).unsqueeze(-1)
    x = (q1[..., 2] * q2[..., 3] - q1[..., 3] * q2[..., 2] + q1[..., 1] * q2[..., 0] + q1[..., 0] * q2[..., 1]).unsqueeze(-1)
    y = (q1[..., 3] * q2[..., 0] - q1[..., 0] * q2[..., 3] + q1[..., 2] * q2[..., 0] + q1[..., 0] * q2[..., 2]).unsqueeze(-1)
    z = (q1[..., 1] * q2[..., 2] - q1[..., 2] * q2[..., 1] + q1[..., 3] * q2[..., 0] + q1[..., 0] * q2[..., 3]).unsqueeze(-1)

    qx = torch.cat((w, x, y, z), dim=-1)

    return qx


def conjugate(q: torch.Tensor) -> torch.Tensor:

    return torch.cat((q[..., :1], -q[..., 1:]), dim=-1)


class DiffusionModelOptimizer:

    def __init__(self, noise_step_count: int, model: torch.nn.Module):

        self.noise_step_count = noise_step_count
        self.model = model
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=0.001)

        self.beta_min = 0.02
        self.beta_max = 0.8

    @staticmethod
    def get_loss(frames_true: Rigid, frames_pred: Rigid, mask: torch.Tensor) -> torch.Tensor:

        # position vectors
        positions_loss = torch.square(frames_true.get_trans() - frames_pred.get_trans()).sum(dim=(-2, -1)) / mask.sum(dim=-1)

        # normalize rotation quaternions
        rotations_true = torch.nn.functional.normalize(frames_true.get_rots().get_quats(), dim=-1)
        rotations_pred = torch.nn.functional.normalize(frames_pred.get_rots().get_quats(), dim=-1)

        # when pred == true, q*q is a unit quaternion
        qq = quat_multiply(conjugate(rotations_pred), rotations_true)
        unit = torch.tensor([1.0, 0.0, 0.0, 0.0], device=qq.device)

        rotations_loss = ((qq - unit) ** 2).sum(dim=(-2, -1)) / mask.sum(dim=-1)

        return positions_loss + rotations_loss

    def get_beta_alpha_sigma(self, noise_step: int) -> Tuple[float, float, float]:

        beta = self.beta_min + (self.beta_max - self.beta_min) * (float(noise_step) / self.noise_step_count)

        alpha = 1.0 - beta

        sigma = sqrt(1.0 - alpha * alpha)

        return (beta, alpha, sigma)

    @staticmethod
    def gen_noise(shape: Union[List[int], Tuple[int]], device: torch.device) -> Rigid:

        # position: [..., 3]
        p = torch.randn(list(shape) + [3], device=device)

        # rotation axis: [..., 3]
        u = torch.randn(list(shape) + [3], device=device)
        u = torch.nn.functional.normalize(u, dim=-1)

        # rotation angle/2 sin, cos: [..., 2]
        a = torch.randn(list(shape) + [2], device=device)
        a = torch.nn.functional.normalize(a, dim=-1)

        # unit quaternion [..., 4]
        q = torch.cat((a[..., 1:], u * a[..., :1]), dim=-1)

        return Rigid(Rotation(quats=q), p)

    def add_noise(self, signal: Rigid, noise: Rigid, t: int) -> Rigid:

        beta, alpha, sigma = self.get_beta_alpha_sigma(t)

        signal_pos = signal.get_trans()
        signal_rot = signal.get_rots()

        noise_pos = noise.get_trans()
        noise_rot = noise.get_rots()

        # noise positions
        pos = signal_pos * alpha + noise_pos * sigma

        # noise rotations
        rot = partial_rot(noise_rot, beta).compose_r(signal_rot)

        return Rigid(rot, pos)

    def remove_noise(
        self,
        noised_signal: Rigid,
        predicted_noise: Rigid,
        t: int, s: int
    ) -> Rigid:

        beta_t, alpha_t, sigma_t = self.get_beta_alpha_sigma(t)
        beta_s, alpha_s, sigma_s = self.get_beta_alpha_sigma(s)

        random_noise = DiffusionModelOptimizer.gen_noise(noised_signal.shape, noised_signal.device)

        alpha_ts = alpha_t / alpha_s
        sqr_sigma_ts = square(sigma_t) - square(sigma_s) * alpha_ts

        sigma_ts = sqrt(sqr_sigma_ts)
        sigma_t2s = sigma_ts * sigma_s / sigma_t

        # denoisify position
        noised_signal_pos = noised_signal.get_trans()
        predicted_noise_pos = predicted_noise.get_trans()
        random_noise_pos = random_noise.get_trans()

        denoised_pos = noised_signal_pos / alpha_ts - \
                       (predicted_noise_pos * sqr_sigma_ts) / (alpha_ts * sigma_t) + \
                       sigma_t2s * random_noise_pos

        # denoisify rotation
        noised_signal_rot = noised_signal.get_rots()
        predicted_noise_rot = predicted_noise.get_rots()
        random_noise_rot = random_noise.get_rots()

        denoised_rot = partial_rot(random_noise_rot, beta_s).compose_r(
            partial_rot(predicted_noise_rot, beta_t).invert().compose_r(noised_signal_rot)
        )

        return Rigid(denoised_rot, denoised_pos)

    def optimize(self, batch: Dict[str, Union[Rigid, torch.Tensor]]):

        frames = Rigid.from_tensor_7(batch["frames"])
        frame_dimensions = list(range(len(frames.shape)))

        t = random.randint(0, self.noise_step_count - 1)

        self.optimizer.zero_grad()

        epsilon = self.gen_noise(frames.shape, frames.device)

        zt = self.add_noise(frames, epsilon, t)

        batch = {
            "frames": zt,
            "mask": batch["mask"],
            "features": batch["features"],
        }

        pred_epsilon = self.model(batch, t)

        loss = self.get_loss(epsilon, pred_epsilon, batch["mask"]).mean()
        if loss.isnan().any():
            raise RuntimeError("NaN loss")

        _log.debug(f"loss is {loss:.3f}")

        loss.backward()

        self.optimizer.step()

    def sample(self, batch: Dict[str, Union[torch.Tensor, Rigid]]) -> Rigid:

        beta_delta = self.beta_max - self.beta_min

        zt = batch["frames"]

        t = self.noise_step_count
        while t > 0:

            s = t - 1

            batch = {
                "frames": zt,
                "mask": batch["mask"],
                "features": batch["features"],
            }

            zs = self.remove_noise(
                zt, self.model(batch, t),
                t, s,
            )

            zt = zs
            t = s

        x = zt
        return x

