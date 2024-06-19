from math import pi

import torch

from diffusion.tools.angle import angle_to_sin_cos, multiply_sin_cos


def test_sin_cos_multiplication():

    angles = torch.tensor(
        [pi, pi / 2, pi / 3, 0.0, -pi / 3, -pi / 2, -pi]
    )
    size = angles.shape[0]

    sum_of_angles = angles[:, None] + angles[None, :]

    sin_cos = angle_to_sin_cos(angles)

    multiplication = multiply_sin_cos(
        sin_cos[:, None, :].expand(-1, size, -1),
        sin_cos[None, :, :].expand(size, -1, -1),
    )

    sin_cos_of_sum = angle_to_sin_cos(sum_of_angles)
    assert torch.all(multiplication == sin_cos_of_sum), f"{multiplication} != {sum_of_angles}"
