from math import pi

import torch

from diffusion.tools.angle import angle_to_sin_cos, multiply_sin_cos, random_quat, inverse_sin_cos


epsilon = 1e-6


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
    assert torch.all(torch.abs(multiplication - sin_cos_of_sum) < epsilon), \
        f"{multiplication}\n!=\n{sin_cos_of_sum}"

    inverted = multiply_sin_cos(
        inverse_sin_cos(sin_cos),
        sin_cos,
    )

    # angle must be 0.0
    assert torch.all(inverted[..., :1] == 0.0), inverted[..., :1]
    assert torch.all(inverted[..., 1:] == 1.0), inverted[..., 1:]



def test_random_quat():

    q = random_quat((10, 10), torch.device("cpu"))

    l = (q ** 2).sum(dim=-1).sqrt()

    assert torch.all(torch.abs(l - 1.0) < epsilon)
