"""Explicit sample order for checkpoint-safe asynchronous policy loading."""

import torch
from torch.utils.data import Sampler


def legacy_first_epoch_order(rng_state, count):
    """Reproduce the original runner's monitor/train iterator seed draws."""
    generator = torch.Generator().set_state(rng_state)
    # Monitor iterator base seed, training iterator base seed, sampler seed.
    for _ in range(2):
        torch.empty((), dtype=torch.int64).random_(generator=generator)
    seed = int(torch.empty((), dtype=torch.int64).random_(generator=generator))
    return torch.randperm(count, generator=torch.Generator().manual_seed(seed))


class ResumableOrder(Sampler):
    def __init__(self, count):
        self.order = torch.arange(count)
        self.start = 0

    def __iter__(self):
        return iter(self.order[self.start :].tolist())

    def __len__(self):
        return len(self.order) - self.start
