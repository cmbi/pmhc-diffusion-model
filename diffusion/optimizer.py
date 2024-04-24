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

    def alpha_function(self, t: int) -> float:

        return exp(-2 * (t / self.noise_step_count))

    def optimize(self, batch: Dict[str, torch.Tensor]):

        x = batch["positions"]

        t = random.randint(0, self.noise_step_count - 1)

        self.optimizer.zero_grad()

        epsilon = torch.randn(x.shape, device=x.device)

        alpha = self.alpha_function(t)
        sigma = sqrt(1.0 - square(alpha))

        zt = alpha * x + sigma * epsilon

        batch = {
            "positions": zt,
            "mask": batch["mask"],
            "features": batch["features"],
        }

        loss = torch.square(epsilon - self.model(batch, t)).mean()

        _log.debug(f"loss is {loss}")

        loss.backward()

        self.optimizer.step()

    def sample(self, batch: Dict[str, torch.Tensor]) -> torch.Tensor:

        zt = batch["positions"]

        t = self.noise_step_count
        while t > 0:

            s = t - 1

            epsilon = torch.randn(zt.shape, device=zt.device)

            alpha_t = self.alpha_function(t)
            sigma_t = sqrt(1.0 - square(alpha_t))

            alpha_s = self.alpha_function(s)
            sigma_s = sqrt(1.0 - square(alpha_s))

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

            zt = zs
            t = s

        x = zt
        return x

