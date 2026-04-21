import torch
from torch.utils.data import Sampler, BatchSampler
from typing import Optional
from abc import abstractmethod
from tqdm import tqdm


class BatchSamplerWithDataSkip(BatchSampler):
    def __init__(
            self,
            sampler: Sampler,
            batch_size: int,
            drop_last: bool = False,
            skip_steps: Optional[int] = None
    ):
        super().__init__(sampler, batch_size, drop_last)
        self.skip_steps = 0 if skip_steps is None else skip_steps

    @abstractmethod
    def _iter_step(self):
        return None

    def __iter__(self):
        iter_gen = self._iter_step()

        if iter_gen is None:
            iter_gen = super().__iter__()

        for i in tqdm(range(self.skip_steps)):
            res = next(iter_gen, None)
            if res is None:
                # New epoch
                iter_gen = self._iter_step()

                if iter_gen is None:
                    iter_gen = super().__iter__()

                next(iter_gen, None)

        self.skip_steps = 0

        yield from iter_gen

