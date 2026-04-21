from typing import Type
import torch
import re
from copy import deepcopy

from omegaconf import ListConfig
from torch.utils.data import RandomSampler, BatchSampler, SequentialSampler, DataLoader, Dataset
from collections import defaultdict
from functools import partial
from ..utils.dict import deep_update_dict

from .samplers import BatchSamplerWithDataSkip
from .data_processor import DataProcessor
from .data_provider import DataProvider
import torch.distributed as dist


class DataloaderSplits:
    """Build datasets and batch samplers for all splits from a DataProvider.

    Supports sample-based and token-based batching, length-based sampling,
    dataset subsetting (by file or by index range), and resumable data
    skipping for checkpoint recovery.

    Multiple dataloaders per split are supported: if the DataProvider contains
    keys like ``"validation_copy"`` and ``"validation_reverse"``, they are
    grouped under the ``"validation"`` split and returned as a list from
    :meth:`get_dataloader`.
    """
    VALIDATION_SYN = {'validation', 'val', 'valid', 'dev'}
    DEFAULT_BATCH_CONFIG = {
        'train': {
            'type': 'samples',
            'size': 32,
            'sampler': 'random',
            'shuffle': True
        },
        'validation': {
            'type': 'samples',
            'size': 32,
            'sampler': 'sequential',
            'shuffle': False
        },
        'test': {
            'type': 'samples',
            'size': 32,
            'sampler': 'sequential',
            'shuffle': False
        }
    }

    def __init__(
            self,
            data_provider: DataProvider,
            data_processor: Type[DataProcessor],
            data_processor_kwargs: dict,
            sampler_type: str = 'random',
            sampler_kwargs: dict = None,
            batch_config: dict = None,
            subset_config: dict = None,
            data_skip_steps: int = None
    ):
        batch_config = dict() if batch_config is None else self._fix_split_name(dict(batch_config))

        batch_config = deep_update_dict(deepcopy(self.DEFAULT_BATCH_CONFIG), batch_config)
        sampler_kwargs = dict() if sampler_kwargs is None else sampler_kwargs

        splits = defaultdict(list)
        batch_samplers = defaultdict(list)

        for split, data in data_provider.items():
            # Store original split name
            orig_split = split
            # Strip suffix from split name. Split name can have form validation_suffix
            split = re.sub(r"_[\d\w]+", "", split)

            batch_type = batch_config[split]['type']
            batch_size = batch_config[split]['size']

            data_processor_kwargs.update({"data": data, "split": split})
            ds = data_processor(**data_processor_kwargs)

            if len(ds) == 0:
                raise ValueError(f"Dataset is empty `split={orig_split}`")

            if ds.streaming and subset_config is not None:
                raise RuntimeError("`subset_config` does not work in streaming mode. Use settings for DataProvider (e.g. take, skip)")

            if subset_config is not None and orig_split in subset_config:
                if isinstance(subset_config[orig_split], str):
                    # Limit dataset by ids specified in a file
                    file_path = subset_config[orig_split]
                    ds = ds.subset_from_file(file_path)
                elif isinstance(subset_config[orig_split], list) or isinstance(subset_config[orig_split], tuple) \
                        or isinstance(subset_config[orig_split], ListConfig):
                    # Limit dataset by range
                    bl, br = list(subset_config[orig_split])
                    sample_ids = list(range(bl, br))
                    ds = torch.utils.data.Subset(ds, sample_ids)
                else:
                    raise ValueError(f"Wrong type in subset_config for split `{orig_split}`: {subset_config[orig_split]}")

            splits[split] += [ds]

            if split == 'train' and data_skip_steps is not None:
                _data_skip_steps = data_skip_steps
            else:
                _data_skip_steps = 0

            is_random = batch_config[split]['shuffle']
            if is_random:
                ds_sampler = self.get_sampler(ds, "random")
            else:
                ds_sampler = self.get_sampler(ds, "sequential")

            if _data_skip_steps != 0:
                batch_sampler = BatchSamplerWithDataSkip(ds_sampler, batch_size, False, skip_steps=_data_skip_steps)
            else:
                batch_sampler = BatchSampler(ds_sampler, batch_size, False)

            batch_samplers[split] += [batch_sampler]

        self.batch_config = batch_config
        self.datasets = splits
        self.batch_samplers = batch_samplers

    def _fix_split_name(self, config):
        for split, data in config.items():
            if split in self.VALIDATION_SYN:
                del config[split]
                config["validation"] = data
                break
        return config

    def get_dataset(self, split):
        """Return the list of datasets for *split*."""
        return self.datasets[split]

    def get_dataloader(self, split, collator, **kwargs):
        """Build DataLoader(s) for *split*. Returns a single DataLoader or a list."""
        datasets = self.datasets[split]
        batch_samplers = self.batch_samplers[split]

        if 'collator' in self.batch_config[split]:
            if not isinstance(collator, partial):
                raise TypeError(f"collator in split `{split}` is not callable. Make sure collator is of functools.partial")
            collator = collator(**self.batch_config[split]['collator'])


        dataloaders = [DataLoader(
                ds,
                batch_sampler=batch_sampler,
                collate_fn=collator,
                **kwargs
            ) for ds, batch_sampler in zip(datasets, batch_samplers)]

        if len(dataloaders) == 1:
            dataloaders = dataloaders[0]

        return dataloaders

    def get_sampler(self, ds, sampler_name, extra_kwargs=None):
        if dist.is_available() and dist.is_initialized():
            num_replicas = dist.get_world_size()
            rank = dist.get_rank()
        else:
            num_replicas = 1
            rank = 0

        extra_kwargs = dict() if extra_kwargs is None else dict(extra_kwargs)
        shuffle = False
        cls = torch.utils.data.distributed.DistributedSampler

        if sampler_name == "random":
            shuffle = True
        elif sampler_name == "sequential":
            pass
        else:
            raise ValueError(f"Unknown sampler `{sampler_name}`")

        sampler = cls(ds, num_replicas=num_replicas, rank=rank, shuffle=shuffle, **extra_kwargs)
        return sampler