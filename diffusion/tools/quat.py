import torch

def get_conjugate(q: torch.Tensor) -> torch.Tensor:
    return torch.cat((q[..., :1], -q[..., 1:]), dim=-1)


def get_angle(q1: torch.Tensor, q2: torch.Tensor) -> torch.Tensor:
    q1 = torch.nn.functional.normalize(q1, dim=-1)
    q2 = torch.nn.functional.normalize(q2, dim=-1)

    dot = torch.clamp((q1 * q2).sum(dim=-1), -1.0, 1.0)

    # if dot is -1, the axis and angle are both flipped
    angle = torch.acos(torch.abs(dot))

    return angle
