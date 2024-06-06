import random
from typing import Tuple, Dict, Optional, Union, List
from math import sqrt, exp, pi
import logging

import torch
from openfold.utils.rigid_utils import Rigid, Rotation, invert_quat, quat_multiply, rot_to_quat
from openfold.utils.loss import compute_fape
from openfold.np.residue_constants import rigid_group_atom_positions

from .tools.frame import get_rmsd
from .tools.pdb import save
from .tools.quat import get_angle


_log  = logging.getLogger(__name__)


def get_cn_bond_lengths(frames: Rigid) -> torch.Tensor:

    n_positions = frames.apply(torch.tensor(rigid_group_atom_positions["ALA"][0][2], device=frames.device))
    c_positions = frames.apply(torch.tensor(rigid_group_atom_positions["ALA"][2][2], device=frames.device))

    return torch.sqrt(torch.square(c_positions[..., :-1, :] - n_positions[..., 1:, :]).sum(dim=-1))


def square(x: float) -> float:
    return x * x


def partial_rot(rot: Rotation, amount: float) -> Rotation:

    q = torch.nn.functional.normalize(rot.get_quats(), dim=-1)
    a2 = torch.acos(torch.clamp(q[..., :1], -1.0, 1.0))
    a = torch.where(a2 * 2 > pi, a2 * 2 - 2 * pi, a2 * 2)
    x = torch.nn.functional.normalize(q[..., 1:], dim=-1)

    return Rotation(quats=torch.cat((torch.cos(a / 2 * amount), torch.sin(a / 2 * amount) * x), dim=-1), normalize_quats=False)


class DiffusionModelOptimizer:

    def __init__(self, noise_step_count: int, model: torch.nn.Module):

        self.noise_step_count = noise_step_count
        self.model = model
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=0.001)

        self.beta_min = 0.0001
        self.beta_max = 0.8

    @staticmethod
    def get_loss(noise_true: Rigid, noise_pred: Rigid, mask: torch.Tensor) -> torch.Tensor:

        # position square deviation
        positions_loss = (torch.square(noise_true.get_trans() - noise_pred.get_trans()).sum(dim=-1) * mask).sum(dim=-1) / mask.sum(dim=-1)

        # rotation angle deviation
        angle = get_angle(noise_true.get_rots().get_quats(), noise_pred.get_rots().get_quats())

        rotations_loss = (angle * mask).sum(dim=-1) / mask.sum(dim=-1)

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

