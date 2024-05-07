import random
from typing import Tuple, Dict
from math import sqrt, exp
import logging

import torch


_log  = logging.getLogger(__name__)


def square(x: float) -> float:
    return x * x



class DiffusionModelOptimizer:

    def __init__(self, noise_step_count: int, model: torch.nn.Module):

        self.noise_step_count = noise_step_count
        self.model = model
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=0.001)

    def optimize(self, batch: Dict[str, torch.Tensor]):

        x = batch["positions"]

        t = random.randint(0, self.noise_step_count - 1)

        self.optimizer.zero_grad()

        epsilon = torch.randn(x.shape, device=x.device)

        beta = 0.0001 + 0.8 * t / self.noise_step_count
        alpha = 1.0 - beta
        sigma = sqrt(1.0 - square(alpha))

        zt = alpha * x + sigma * epsilon

        batch = {
            "positions": zt,
            "mask": batch["mask"],
            "features": batch["features"],
        }

        loss = (torch.square(epsilon - self.model(batch, t)).sum(dim=(-2, -1)) / x.shape[-2]).mean()

        if loss.isnan().any():
            raise RuntimeError("NaN loss")

        _log.debug(f"loss is {loss}")

        loss.backward()

        self.optimizer.step()

    def sample(self, batch: Dict[str, torch.Tensor]) -> torch.Tensor:

        zt = batch["positions"]

        t = self.noise_step_count
        while t > 0:

            s = t - 1

            epsilon = torch.randn(zt.shape, device=zt.device)

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

            _log.debug(f"at {t}: {1.0 / alpha_ts}, {sqr_sigma_ts / (alpha_ts * sigma_t)}, {sigma_t2s}")

            batch = {
                "positions": zt,
                "mask": batch["mask"],
                "features": batch["features"],
            }

            zs = (1.0 / alpha_ts) * zt - sqr_sigma_ts / (alpha_ts * sigma_t) * self.model(batch, t) + sigma_t2s * epsilon

            if zs.isnan().any():
                raise RuntimeError("NaN coords")

            zt = zs
            t = s

        x = zt
        return x

