import os
from typing import Dict
import csv

import torch


class MetricsRecord:
    def __init__(self):
        self._sums = {}
        self._size = 0

    def add_batch(self, results: Dict[str, torch.Tensor]):

        batch_size = 0
        for key, data in results.items():
            self._sums[key] = self._sums.get(key, 0.0) + data.sum().item()
            batch_size = data.shape[0]

        self._size += batch_size

    def mean(self) -> Dict[str, float]:

        means = {key: sum_ / self._size for key, sum_ in self._sums.items()}
        return means

    def save(self, path: str, epoch_number: int):

        keys = list(self._sums.keys())

        add_header = not os.path.isfile(path)

        with open(path, 'at') as f:
            w = csv.writer(f, delimiter=',')

            if add_header:
                w.writerow(['epoch'] + keys)

            m = self.mean()
            w.writerow([epoch_number] + [round(m[key], 3) for key in keys])

