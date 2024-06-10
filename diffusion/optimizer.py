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



def square(x: float) -> float:
    return x * x


def random_sin_cos(shape: Union[List[int], Tuple[int]], device: torch.device) -> torch.Tensor:
    """
    Makes a random angle and outputs the sin,cos of that.
    """

    a = torch.rand(shape, device=device) * 2 * pi

    sin_cos = torch.cat((torch.sin(a).unsqueeze(-1), torch.cos(a).unsqueeze(-1)), dim=-1)

    return sin_cos


def random_quat(shape: Union[List[int], Tuple[int]], device: torch.device) -> torch.Tensor:
    """
    Makes a random axis with a random rotation angle.
    Output is a quaternion.
    """

    # spherical angles
    phi = torch.rand(shape, device=device) * 2 * pi
    theta = torch.rand(shape, device=device) * pi

    x = torch.cos(phi).unsqueeze(-1)
    y = torch.sin(phi).unsqueeze(-1)
    z = torch.cos(theta).unsqueeze(-1)
    xy = torch.cat((x, y), dim=-1)
    xyz = torch.cat((xy * torch.sin(theta).unsqueeze(-1), z), dim=-1)

    # quaternion angle
    a2 = torch.rand(shape, device=device) * pi
    w = torch.cos(a2).unsqueeze(-1)

    q = torch.cat((w, xyz * torch.sin(a2).unsqueeze(-1)), dim=-1)

    return q


def multiply_sin_cos(sin_cos1: torch.Tensor, sin_cos2: torch.Tensor) -> torch.Tensor:
    """
    Treats the inputs as complex numbers (sin=imaginary, cos=real) and takes the outer product.
    This means that in the output, the angles are added and the magnitudes are multiplied.
    The result is NOT normalized.
    """

    return torch.cat(
        (
            sin_cos1[..., 1:] * sin_cos2[..., 1:] - sin_cos1[..., :1] * sin_cos2[..., :1],
            sin_cos1[..., :1] * sin_cos2[..., 1:] + sin_cos1[..., 1:] * sin_cos2[..., :1],
        ),
        dim=-1
    )


def inverse_sin_cos(sin_cos: torch.Tensor) -> torch.Tensor:
    """
    Inverts the rotation angle and returns sin,cos
    """

    sin_cos = torch.nn.functional.normalize(sin_cos, dim=-1)
    a = torch.acos(torch.clamp(sin_cos[..., 1:], -1.0, 1.0))
    a = torch.where(sin_cos[..., :1] < 0.0, -a, a)

    return torch.cat((torch.sin(-a), torch.cos(-a)), dim=-1)


def partial_sin_cos(sin_cos: torch.Tensor, amount: float) -> torch.Tensor:
    """
    Multiplies the angle by the given amount.
    """

    sin_cos = torch.nn.functional.normalize(sin_cos, dim=-1)
    a = torch.acos(torch.clamp(sin_cos[..., 1:], -1.0, 1.0))
    a = torch.where(sin_cos[..., :1] < 0.0, -a, a)

    return torch.cat((torch.sin(a * amount), torch.cos(a * amount)), dim=-1)


def partial_rot(rot: Rotation, amount: float) -> Rotation:
    """
    Normalizes the axis and multiplies the angle by the given amount.
    """

    q = torch.nn.functional.normalize(rot.get_quats(), dim=-1)
    a2 = torch.acos(torch.clamp(q[..., :1], -1.0, 1.0))  # [0, pi]
    x = torch.nn.functional.normalize(q[..., 1:], dim=-1)

    return Rotation(quats=torch.cat((torch.cos(a2 * amount), torch.sin(a2 * amount) * x), dim=-1), normalize_quats=False)


class DiffusionModelOptimizer:

    def __init__(self, noise_step_count: int, model: torch.nn.Module):

        self.noise_step_count = noise_step_count
        self.model = model
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=0.001)

        self.beta_min = 0.0001
        self.beta_max = 0.8

    @staticmethod
    def get_loss(
        noise_true: Dict[str, Union[Rigid, torch.Tensor]],
        noise_pred: Dict[str, Union[Rigid, torch.Tensor]],
        residues_mask: torch.Tensor,
        torsions_mask: torch.Tensor,
    ) -> torch.Tensor:

        noise_frames_true = noise_true['frames']
        noise_frames_pred = noise_pred['frames']

        noise_torsions_true = noise_true['torsions']
        noise_torsions_pred = noise_pred['torsions']

        # position square deviation
        positions_loss = (torch.square(noise_frames_true.get_trans() - noise_frames_pred.get_trans()).sum(dim=-1) * residues_mask).sum(dim=-1) / residues_mask.sum(dim=-1)

        # rotation angle deviation
        angle = get_angle(noise_frames_true.get_rots().get_quats(), noise_frames_pred.get_rots().get_quats())

        rotations_loss = (angle * residues_mask).sum(dim=-1) / residues_mask.sum(dim=-1)

        # torsion angle deviation (sin, cos)
        torsion_dots = (torch.nn.functional.normalize(noise_torsions_true) * torch.nn.functional.normalize(noise_torsions_pred)).sum(dim=-1)
        torsion_angle_deviation = torch.acos(torch.clamp(torsion_dots, -1.0, 1.0))
        torsion_loss = (torsion_angle_deviation * torsions_mask).sum(dim=(-2, -1)) / torsions_mask.sum(dim=(-2, -1))

        _log.debug(f"rotations loss mean is {rotations_loss.mean():.3f}, positions loss mean is {positions_loss.mean():.3f}, torsions loss mean is {torsion_loss.mean():.3f}")

        return positions_loss + 10.0 * rotations_loss + torsion_loss

    def get_beta_alpha_sigma(self, noise_step: int) -> Tuple[float, float, float]:

        beta = self.beta_min + (self.beta_max - self.beta_min) * (float(noise_step) / self.noise_step_count)

        alpha = 1.0 - beta

        sigma = sqrt(1.0 - alpha * alpha)

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
        torsion = torch.nn.functional.normalize(
            multiply_sin_cos(partial_sin_cos(noise_torsion, beta), signal_torsion),
            dim=-1,
        )

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
        sqr_sigma_ts = square(sigma_t) - square(sigma_s) * alpha_ts

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

    def optimize(self, batch: Dict[str, Union[Rigid, torch.Tensor]]):

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
        loss = self.get_loss(epsilon, pred_epsilon, batch["mask"], batch["torsions_mask"]).mean()
        if loss.isnan().any():
            raise RuntimeError("NaN loss")

        loss.backward()

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

