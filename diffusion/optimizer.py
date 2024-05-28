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



class DiffusionModelOptimizer:

    def __init__(self, noise_step_count: int, model: torch.nn.Module):

        self.noise_step_count = noise_step_count
        self.model = model
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=0.001)

        self.beta_min = 0.0001
        self.beta_max = 0.8

    @staticmethod
    def get_loss(frames_true: Rigid, frames_pred: Rigid, mask: torch.Tensor) -> torch.Tensor:

        #fape = compute_fape(
        #    frames_pred, frames_true, mask,
        #    frames_pred.get_trans(), frames_true.get_trans(), mask,
        #    1.0,
        #)

        #return fape

        # position vectors
        positions_loss = torch.square(frames_true.get_trans() - frames_pred.get_trans()).sum(dim=(-2, -1)) / mask.sum(dim=-1)

        # normalize rotation quaternions
        rotations_true = torch.nn.functional.normalize(frames_true.get_rots().get_quats(), dim=-1)
        rotations_pred = torch.nn.functional.normalize(frames_pred.get_rots().get_quats(), dim=-1)

        dots = (rotations_pred * rotations_true).sum(dim=-1)
        angles = torch.acos(dots)

        rotations_loss = torch.square(angles).sum(dim=-1) / mask.sum(dim=-1)

        return positions_loss + rotations_loss

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

    @staticmethod
    def add_noise(signal: Rigid, noise: Rigid, alpha: float, sigma: float) -> Rigid:

        signal_pos = signal.get_trans()
        signal_quat = signal.get_rots().get_quats()

        noise_pos = noise.get_trans()
        noise_quat = noise.get_rots().get_quats()

        # mean between positions
        pos = signal_pos * alpha + noise_pos * sigma

        # slerp on rotations
        ll = alpha * alpha + sigma * sigma
        s_signal = alpha * alpha / ll
        s_noise = sigma * sigma / ll

        dot = (signal_quat * noise_quat).sum(dim=-1)
        dot = torch.where(dot < 0.0, -dot, dot)
        angle = torch.acos(dot)
        quat = signal_quat * (torch.sin(s_signal * angle) / torch.sin(angle)).unsqueeze(-1) + \
               noise_quat * (torch.sin(s_noise * angle) / torch.sin(angle)).unsqueeze(-1)

        return Rigid(Rotation(quats=quat, normalize_quats=True), pos)

    @staticmethod
    def remove_noise(
        noised_signal: Rigid,
        predicted_noise: Rigid,
        alpha_t: float,
        sigma_t: float,
        alpha_s: float,
        sigma_s: float,
    ) -> Rigid:

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
        noised_signal_quat = torch.nn.functional.normalize(noised_signal.get_rots().get_quats(), dim=-1)
        predicted_noise_quat = torch.nn.functional.normalize(predicted_noise.get_rots().get_quats(), dim=-1)
        random_noise_quat = torch.nn.functional.normalize(random_noise.get_rots().get_quats(), dim=-1)

        noised_signal_matrix = noised_signal_quat[..., None] * noised_signal_quat[..., None, :]
        predicted_noise_matrix = predicted_noise_quat[..., None] * predicted_noise_quat[..., None, :]
        random_noise_matrix = random_noise_quat[..., None] * random_noise_quat[..., None, :]

        denoised_matrix = noised_signal_matrix / alpha_ts - \
                          (predicted_noise_matrix * sqr_sigma_ts) / (alpha_ts * sigma_t) + \
                          sigma_t2s * random_noise_matrix

        _, eigen_vectors = torch.linalg.eig(denoised_matrix)

        return Rigid(Rotation(quats=eigen_vectors[..., 0].real, normalize_quats=True), denoised_pos)
        #return Rigid(Rotation.identity(noised_signal.shape), denoised_pos)

    def optimize(self, batch: Dict[str, Union[Rigid, torch.Tensor]]):

        frames = Rigid.from_tensor_7(batch["frames"])
        frame_dimensions = list(range(len(frames.shape)))

        t = random.randint(0, self.noise_step_count - 1)

        self.optimizer.zero_grad()

        epsilon = self.gen_noise(frames.shape, frames.device)

        beta_delta = self.beta_max - self.beta_min
        beta = self.beta_min + beta_delta * t / self.noise_step_count
        alpha = 1.0 - beta
        sigma = sqrt(1.0 - square(alpha))

        zt = self.add_noise(frames, epsilon, alpha, sigma)

        batch = {
            "frames": zt,
            "mask": batch["mask"],
            "features": batch["features"],
        }

        pred_epsilon = self.model(batch, t)

        loss = self.get_loss(epsilon, pred_epsilon, batch["mask"]).mean()
        if loss.isnan().any():
            raise RuntimeError("NaN loss")

        _log.debug(f"beta is {beta:.3f}, loss is {loss:.3f}")

        loss.backward()

        self.optimizer.step()

    def sample(self, batch: Dict[str, Union[torch.Tensor, Rigid]]) -> Rigid:

        beta_delta = self.beta_max - self.beta_min

        zt = batch["frames"]

        t = self.noise_step_count
        while t > 0:

            s = t - 1

            beta_t = self.beta_min + beta_delta * t / self.noise_step_count
            alpha_t = 1.0 - beta_t
            sigma_t = sqrt(1.0 - square(alpha_t))

            if sigma_t == 0.0:
                raise RuntimeError("zero sigma")

            beta_s = self.beta_min + beta_delta * s / self.noise_step_count
            alpha_s = 1.0 - beta_s
            sigma_s = sqrt(1.0 - square(alpha_s))

            if alpha_s == 0.0:
                raise RuntimeError("zero alpha")

            batch = {
                "frames": zt,
                "mask": batch["mask"],
                "features": batch["features"],
            }

            zs = self.remove_noise(
                zt, self.model(batch, t),
                alpha_t, sigma_t,
                alpha_s, sigma_s,
            )

            if t % 100 == 0:
                save(zs[0], f"dm-output-{t}.pdb")

            zt = zs
            t = s

        x = zt
        return x

