import torch
import random
import sys


from torch.utils.data import DataLoader

from diffusionmodel.optimizer import DiffusionModelOptimizer
from diffusionmodel.data import MhcpDataset


class Model(torch.nn.Module):
    def __init__(self, T: int):
        super(Model, self).__init__()

        trans = 32
        n = 9 * 14
        dim = 3

        self.mlps = torch.nn.ModuleList([
            torch.nn.Sequential(
                torch.nn.Linear(n * dim, trans),
                torch.nn.ReLU(),
                torch.nn.Linear(trans, n * dim),
            )
            for t in range(T)]
        )

    def forward(self, z: torch.Tensor, t: int) -> torch.Tensor:

        shape = z.shape
        z = z.reshape(shape[0], -1)

        e = self.mlps[t](z)

        return e.reshape(shape)


if __name__ == "__main__":

    dim = 3
    n = 10
    b = 8
    trans = 32
    T = 10

    dm = DiffusionModelOptimizer(T, Model(T))

    m = torch.randn(n, dim)

    dataset = MhcpDataset(sys.argv[1])
    data_loader = DataLoader(dataset, shuffle=True)

    nepoch = 100

    for _ in range(nepoch):
        for batch in data_loader:

            x = batch["peptide_atom14_gt_positions"]

            dm.optimize(x)

    z = dm.sample((1, n, dim))

    print(z)
