from typing import Any, Dict, Optional, Tuple, Type

from lightning import LightningDataModule
from torch.utils.data import DataLoader

from .data_processor import DataProcessor
from .data_provider import DataProvider
from .data_splits import DataloaderSplits


class DataModule(LightningDataModule):
    """LightningDataModule that wires together data provider, processor, and samplers.

    Builds train/val/test DataLoaders from a :class:`DataProvider` (raw data
    source) and a :class:`DataProcessor` (tokenization, formatting). Supports
    token-level batching, length-based sampling, dataset subsetting, and
    resumable data skipping.
    """
    split_alias = {"validation": "val", "test": "test", "train": "train"}

    def __init__(
            self,
            data_provider: DataProvider,
            data_processor: Type[DataProcessor],
            data_processor_kwargs: dict = None,
            sampler_type: str = 'random',
            sampler_kwargs: dict = None,
            batch_config: dict = None,
            subset_config: dict = None,
            collator: Any = None,
            data_loader_kwargs: dict = None,
            data_skip_steps: int = None
    ):
        super().__init__()
        self.data_provider = data_provider
        self.data_processor = data_processor
        self.data_processor_kwargs = data_processor_kwargs
        self.sampler_type = sampler_type
        self.sampler_kwargs = sampler_kwargs
        self.batch_config = batch_config
        self.subset_config = subset_config
        self.collator = collator
        self.data_loader_kwargs = data_loader_kwargs if data_loader_kwargs is not None else {}
        self.data_skip_steps = data_skip_steps
        self.eval_split_name = 'test'

        if data_processor_kwargs and 'tokenizer' in data_processor_kwargs:
            self.tokenizer = data_processor_kwargs['tokenizer']


    def setup(self, stage):
        self.data_splits = DataloaderSplits(
            data_provider=self.data_provider,
            data_processor=self.data_processor,
            data_processor_kwargs=self.data_processor_kwargs,
            sampler_type=self.sampler_type,
            sampler_kwargs=self.sampler_kwargs,
            batch_config=self.batch_config,
            subset_config=self.subset_config,
            data_skip_steps=self.data_skip_steps
        )

    def set_eval_split(self, name):
        self.eval_split_name = name

    @property
    def num_classes(self) -> int:
        """Return the vocabulary size."""
        return self.hparams.vocab_size

    def prepare_data(self) -> None:
        """No-op. Data is loaded lazily in :meth:`setup`."""
        pass

    def get_dataloader(self, split_name):
        return self.data_splits.get_dataloader(split_name, collator=self.collator, **self.data_loader_kwargs)

    def train_dataloader(self) -> DataLoader[Any]:
        """Return the training DataLoader."""
        return self.get_dataloader('train')

    def val_dataloader(self) -> DataLoader[Any]:
        """Return the validation DataLoader (may be a list if multiple splits)."""
        return self.get_dataloader('validation')

    def test_dataloader(self) -> DataLoader[Any]:
        """Return the test DataLoader for the current :attr:`eval_split_name`."""
        return self.get_dataloader(self.eval_split_name)



