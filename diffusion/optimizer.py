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
from .tools.angle import random_quat, random_sin_cos, partial_rot, partial_sin_cos, inverse_sin_cos, multiply_sin_cos
from .tools.metrics import MetricsRecord


_log  = logging.getLogger(__name__)


def linear_schedule(t: int, T: int, beta_min: float, beta_max: float) -> float:
    return beta_min + (beta_max - beta_min) * (float(t) / T)

def pow_schedule(t: int, T: int, beta_min: float, beta_max: float, p: int) -> float:
    tf = float(t) / T
    return beta_min + (beta_max - beta_min) * tf ** p

class DiffusionModelOptimizer:

    def __init__(self, noise_step_count: int, model: torch.nn.Module, lr: float):

        self.noise_step_count = noise_step_count
        self.model = model
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=lr)

        self.beta_min = 0.0
        self.beta_max = 0.8

    @staticmethod
    def get_loss(
        noise_true: Dict[str, Union[Rigid, torch.Tensor]],
        noise_pred: Dict[str, Union[Rigid, torch.Tensor]],
        residues_mask: torch.Tensor,
        torsions_mask: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:

        noise_frames_true = noise_true['frames']
        noise_frames_pred = noise_pred['frames']

        noise_torsions_true = noise_true['torsions']
        noise_torsions_pred = noise_pred['torsions']

        # position square deviation
        positions_loss = (torch.square(noise_frames_true.get_trans() - noise_frames_pred.get_trans()).sum(dim=-1) * residues_mask).sum(dim=-1) / residues_mask.sum(dim=-1)
        rmsd = torch.sqrt(positions_loss)

        # rotation angle deviation, the square quaternion dot product
        # represents the deviation
        quats_true = torch.nn.functional.normalize(noise_frames_true.get_rots().get_quats(), dim=-1)
        quats_pred = torch.nn.functional.normalize(noise_frames_pred.get_rots().get_quats(), dim=-1)
        quats_dots = (quats_true * quats_pred).sum(dim=-1)
        quats_deviation = 1.0 - quats_dots  # range 0.0 to 2.0
        rotations_loss = (quats_deviation * residues_mask).sum(dim=-1) / residues_mask.sum(dim=-1)

        # torsion angle deviation (sin, cos)
        noise_torsions_true = torch.nn.functional.normalize(noise_torsions_true, dim=-1)
        noise_torsions_pred = torch.nn.functional.normalize(noise_torsions_pred, dim=-1)
        torsion_dots = (noise_torsions_true * noise_torsions_pred).sum(dim=-1)
        torsion_deviation = 1.0 - torsion_dots  # range 0.0 to 2.0
        torsion_loss = (torsion_deviation * torsions_mask).sum(dim=(-2, -1)) / torsions_mask.sum(dim=(-2, -1))

        _log.debug(f"rotations loss mean is {rotations_loss.mean():.3f}, positions loss mean is {positions_loss.mean():.3f}, torsions loss mean is {torsion_loss.mean():.3f}")

        return {
            'total loss': 0.1 * positions_loss + rotations_loss + torsion_loss,
            'positions loss': positions_loss,
            'rotations loss': rotations_loss,
            'torsions loss': torsion_loss,
            'rmsd': rmsd,
        }

    def get_beta_alpha_sigma(self, noise_step: int) -> Tuple[float, float, float]:

        beta = linear_schedule(noise_step, self.noise_step_count, self.beta_min, self.beta_max)

        sigma = sqrt(beta)

        alpha = sqrt(1.0 - beta)

        _log.debug(f"at {noise_step}: beta={beta:.3f}")

        return (beta, alpha, sigma)

    @staticmethod
    def gen_noise(shape: Union[List[int], Tuple[int]], device: torch.device) -> Rigid:

        # position: [..., 3]
        p = torch.randn(list(shape) + [3], device=device) * 5.0

        # unit quaternion [..., 4]
        q = random_quat(shape, device=device)

        # sin, cos [..., 7, 2]
        torsions = random_sin_cos(list(shape) + [7], device)

        return {
            "frames": Rigid(Rotation(quats=q), p),
            "torsions": torsions,
        }

    def add_noise(self,
        signal: Dict[str, Union[Rigid, torch.Tensor]],
        noise: Dict[str, Union[Rigid, torch.Tensor]],
        t: int
    ) -> Dict[str, Union[Rigid, torch.Tensor]]:

        beta, alpha, sigma = self.get_beta_alpha_sigma(t)

        signal_pos = signal["frames"].get_trans()
        signal_rot = signal["frames"].get_rots()
        signal_torsion = signal["torsions"]

        noise_pos = noise["frames"].get_trans()
        noise_rot = noise["frames"].get_rots()
        noise_torsion = noise["torsions"]

        # noise_torsions
        torsion = multiply_sin_cos(partial_sin_cos(noise_torsion, beta), signal_torsion)

        # noise positions
        pos = signal_pos * alpha + noise_pos * sigma

        # noise rotations
        rot = partial_rot(noise_rot, beta).compose_r(signal_rot)

        result = {k: signal[k] for k in signal}
        result["frames"] = Rigid(rot, pos)
        result["torsions"] = torsion
        return result

    def remove_noise(
        self,
        noised_signal: Dict[str, Union[Rigid, torch.Tensor]],
        predicted_noise: Dict[str, Union[Rigid, torch.Tensor]],
        t: int,
        s: int,
    ) -> Dict[str, Union[Rigid, torch.Tensor]]:

        beta_t, alpha_t, sigma_t = self.get_beta_alpha_sigma(t)
        beta_s, alpha_s, sigma_s = self.get_beta_alpha_sigma(s)

        random_noise = DiffusionModelOptimizer.gen_noise(noised_signal["frames"].shape, noised_signal["frames"].device)

        alpha_ts = alpha_t / alpha_s
        sqr_sigma_ts = sigma_t ** 2 - sigma_s ** 2 * alpha_ts

        sigma_ts = sqrt(sqr_sigma_ts)
        sigma_t2s = sigma_ts * sigma_s / sigma_t

        # denoisify position by KL
        noised_signal_pos = noised_signal["frames"].get_trans()
        predicted_noise_pos = predicted_noise["frames"].get_trans()
        random_noise_pos = random_noise["frames"].get_trans()

        denoised_pos = noised_signal_pos / alpha_ts - \
                       (predicted_noise_pos * sqr_sigma_ts) / (alpha_ts * sigma_t) + \
                       sigma_t2s * random_noise_pos

        # denoisify rotation by inverting the rotation
        noised_signal_rot = noised_signal["frames"].get_rots()
        predicted_noise_rot = predicted_noise["frames"].get_rots()
        random_noise_rot = random_noise["frames"].get_rots()

        denoised_rot = partial_rot(random_noise_rot, beta_s).compose_r(
            partial_rot(predicted_noise_rot, beta_t).invert().compose_r(noised_signal_rot)
        )

        # denoisify torsion by inverting the rotation
        noised_signal_torsion = noised_signal["torsions"]
        predicted_noise_torsion = predicted_noise["torsions"]
        random_noise_torsion = random_noise["torsions"]

        denoised_torsion = multiply_sin_cos(
            partial_sin_cos(random_noise_torsion, beta_s),
            multiply_sin_cos(
                inverse_sin_cos(partial_sin_cos(predicted_noise_torsion, beta_t)),
                noised_signal_torsion,
            ),
        )

        result = {k: noised_signal[k] for k in noised_signal}
        result["frames"] = Rigid(denoised_rot, denoised_pos)
        result["torsions"] = denoised_torsion
        return result

    def optimize(self, batch: Dict[str, Union[Rigid, torch.Tensor]], metrics: Optional[MetricsRecord] = None):

        t = random.randint(0, self.noise_step_count - 1)

        self.optimizer.zero_grad()

        batch["frames"] = Rigid.from_tensor_7(batch["frames"])
        batch["pocket_frames"] = Rigid.from_tensor_7(batch["pocket_frames"])

        # noise
        epsilon = self.gen_noise(batch["frames"].shape, batch["frames"].device)

        # noised data
        zt = self.add_noise(batch, epsilon, t)

        # predict noise
        pred_epsilon = self.model(zt, t)

        # loss computation & backward propagation
        losses = self.get_loss(epsilon, pred_epsilon, batch["mask"], batch["torsions_mask"])

        total_loss = losses['total loss']
        if total_loss.isnan().any():
            raise RuntimeError("NaN loss")

        metrics.add_batch(losses)

        total_loss.mean().backward()

        self.optimizer.step()

    def sample(self, batch: Dict[str, Union[torch.Tensor, Rigid]]) -> Rigid:

        beta_delta = self.beta_max - self.beta_min

        # noised data to frames
        batch["pocket_frames"] = Rigid.from_tensor_7(batch["pocket_frames"])
        batch["frames"] = Rigid.from_tensor_7(batch["frames"])

        zt = batch

        t = self.noise_step_count
        while t > 0:

            _log.debug(f"sample t={t}")

            s = t - 1

            zs = self.remove_noise(
                zt, self.model(zt, t),
                t, s,
            )

            zt = zs
            t = s

        x = zt
        return x

