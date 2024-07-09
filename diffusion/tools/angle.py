from typing import Union, List, Tuple
from math import pi

import torch

from openfold.utils.rigid_utils import Rotation


def get_quat_conjugate(q: torch.Tensor) -> torch.Tensor:
    """
    Gets conjugate quaternion
    """

    return torch.cat((q[..., :1], -q[..., 1:]), dim=-1)


def get_quat_angle(q1: torch.Tensor, q2: torch.Tensor) -> torch.Tensor:
    """
    Gets angle between two quaternions
    """

    q1 = torch.nn.functional.normalize(q1, dim=-1)
    q2 = torch.nn.functional.normalize(q2, dim=-1)

    dot = torch.clamp((q1 * q2).sum(dim=-1), -1.0, 1.0)

    # if dot is -1, the axis and angle are both flipped
    angle = torch.acos(torch.abs(dot))

    return angle


def random_sin_cos(shape: Union[List[int], Tuple[int]], device: torch.device) -> torch.Tensor:
    """
    Makes a random angle and outputs the sin,cos of that.
    Output is normalized.
    """

    a = torch.rand(shape, device=device) * 2 * pi

    return angle_to_sin_cos(a)


def angle_to_sin_cos(angle: torch.Tensor) -> torch.Tensor:
    """
    Returns a tensor with the angle's sin, cos values.
    """

    sin_cos = torch.cat(
        (
            torch.sin(angle).unsqueeze(-1),
            torch.cos(angle).unsqueeze(-1)
        ),
        dim=-1
    )

    return sin_cos

def random_quat(shape: Union[List[int], Tuple[int]], device: torch.device) -> torch.Tensor:
    """
    Makes a random axis with a random rotation angle.
    Output is a normalized quaternion.
    """

    x = torch.rand(list(shape) + [3], device=device)
    return shoemake_quat(x)


def shoemake_quat(x: torch.Tensor) -> torch.Tensor:
    """
    Converts shoemake coordinates into quaternions.
    This is a method to get uniform rotations.

    Args:
        x: [..., 3]
    Returns:
        quaternions: [..., 4]
    """

    x = x.clamp(0.0, 1.0)

    theta1 = (2 * pi * x[..., 1]).unsqueeze(-1)
    theta2 = (2 * pi * x[..., 2]).unsqueeze(-1)

    r1 = torch.sqrt(1.0 - x[..., 0]).unsqueeze(-1)
    r2 = torch.sqrt(x[..., 0]).unsqueeze(-1)

    q = torch.cat(
        (
            r2 * torch.cos(theta2),
            r1 * torch.sin(theta1),
            r1 * torch.cos(theta1),
            r2 * torch.sin(theta2),
        ),
        dim=-1
    )

    return q


def spherical_to_quat(
    axis_phi: torch.Tensor,
    axis_theta: torch.Tensor,
    alpha: torch.Tensor,
) -> torch.Tensor:
    """
    Turns an axis (spherical coordinates) and a rotation anglr
    into a normalized quaternion
    """

    x = torch.cos(axis_phi).unsqueeze(-1)
    y = torch.sin(axis_phi).unsqueeze(-1)
    z = torch.cos(axis_theta).unsqueeze(-1)
    xy = torch.cat((x, y), dim=-1)
    xyz = torch.cat((xy * torch.sin(axis_theta).unsqueeze(-1), z), dim=-1)

    a2 = alpha / 2
    w = torch.cos(a2).unsqueeze(-1)

    q = torch.cat((w, xyz * torch.sin(a2).unsqueeze(-1)), dim=-1)

    return q


def get_sin_cos_angle(sin_cos1: torch.Tensor, sin_cos2: torch.Tensor) -> torch.Tensor:
    """
    Gets the angle between two sin,cos vectors
    """

    sin_cos1 = torch.nn.functional.normalize(sin_cos1, dim=-1)
    sin_cos2 = torch.nn.functional.normalize(sin_cos2, dim=-1)

    dot = (sin_cos1 * sin_cos2).sum(dim=-1)
    a = torch.acos(dot.clamp(-1.0, 1.0))

    return a


def multiply_sin_cos(sin_cos1: torch.Tensor, sin_cos2: torch.Tensor) -> torch.Tensor:
    """
    Treats the inputs as complex numbers (sin=imaginary, cos=real) and takes the outer product.
    This means that in the output, the angles are added and the magnitudes are multiplied.
    The result is NOT normalized.
    """

    return torch.cat(
        (
            sin_cos1[..., :1] * sin_cos2[..., 1:] + sin_cos1[..., 1:] * sin_cos2[..., :1],
            sin_cos1[..., 1:] * sin_cos2[..., 1:] - sin_cos1[..., :1] * sin_cos2[..., :1],
        ),
        dim=-1
    )


def inverse_sin_cos(sin_cos: torch.Tensor) -> torch.Tensor:
    """
    Inverts the rotation angle and returns sin,cos
    """

    sqr_norm = (sin_cos ** 2).sum(dim=-1)

    return torch.cat((-sin_cos[..., :1], sin_cos[..., 1:]), dim=-1) / sqr_norm[..., None]


def partial_sin_cos(sin_cos: torch.Tensor, amount: float) -> torch.Tensor:
    """
    Multiplies the angle by the given amount.
    """

    sin_cos = torch.nn.functional.normalize(sin_cos, dim=-1)
    a = torch.acos(torch.clamp(sin_cos[..., 1:], -1.0, 1.0))  # [0, pi]
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
