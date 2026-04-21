import sys
import torch
import numpy as np
from torch.utils.data import Dataset


class DataProcessor(Dataset):
    """Base dataset class with helpers for subsetting, masking, and text formatting."""

    def __len__(self):
        if hasattr(self.data, "streaming") and self.data.streaming:
            return sys.maxsize // 4 # Must be less than sys.maxsize to exlude overflow
        else:
            return len(self.data)

    @property
    def data(self):
        return self._data

    @data.setter
    def data(self, data):
        self._data = data

    @staticmethod
    def load_selected_ids(file_path, file_type):
        """Load sample indices from a file.

        :param file_path: Path to the file containing sample indices.
        :param file_type: ``"lines"`` for a text file with one index per line,
            ``"torch"`` for a torch-saved tensor.
        :return: List or tensor of integer indices.
        """
        if file_type == 'lines':
            with open(file_path, "r") as f:
                selected_ids = [int(line.strip()) for line in f.readlines()]
        elif file_type == 'torch':
            selected_ids = torch.load(file_path)
        else:
            raise ValueError(f"File type `{file_type}` is not defined")

        return selected_ids

    def subset_from_file(self, file_path, file_type='torch'):
        """Return a :class:`Subset` of this dataset using indices loaded from *file_path*."""
        selected_ids = self.load_selected_ids(file_path, file_type).tolist()
        return torch.utils.data.Subset(self, selected_ids)

    def random_subset(self, num_samples):
        """Return a :class:`Subset` of *num_samples* randomly chosen indices."""
        ids = np.random.randint(0, len(self), size=num_samples)
        return torch.utils.data.Subset(self, ids)

    @staticmethod
    def mask_tokens(token_ids: list[int], mask_prob: float = 0.0, mask_token_id: int = -1, vary_prob: bool = True):
        """
        Data augmentation with random token masking
        :return: list[int]
        """
        if mask_prob > 0:
            if vary_prob:
                mask_prob = np.random.randint(0, int(mask_prob * 100)) / 100

            ids = np.array(token_ids)
            masked = np.random.binomial(n=1, p=mask_prob, size=len(ids)) == 1
            # Dropping tokens
            ids[masked] = mask_token_id
            return ids.tolist()
        else:
            return token_ids

    @staticmethod
    def process_format(pattern: str, text: str, special_tokens: dict):
        """Format *pattern* by substituting ``{text}`` and special-token placeholders."""
        format_kwargs = dict(special_tokens)
        format_kwargs.update({'text': text})
        return pattern.format(**format_kwargs)


class DataProcessorWithTokenizer(DataProcessor):
    """DataProcessor that stores a tokenizer and its special-token map."""

    def __init__(self, tokenizer):
        self.streaming = False
        self.tokenizer = tokenizer
        self.special_tokens = dict(tokenizer.special_tokens_map)
        if 'additional_special_tokens' in self.special_tokens:
            del self.special_tokens['additional_special_tokens']


class SourceTargetProcessor(DataProcessorWithTokenizer):
    """Dataset that tokenizes source-target pairs into causal-LM training samples.

    Each ``__getitem__`` returns a dict with concatenated ``input_ids``
    (source + target), shifted ``targets``, and a ``target_mask`` indicating
    which positions contribute to the loss. When ``only_source=True``, the
    target is empty and the processor acts as a plain text dataset.

    :param data: Indexable data object (e.g. :class:`ZipLines`).
    :param split: Split name (``"train"``, ``"validation"``, ``"test"``).
    :param tokenizer: Tokenizer with ``encode``/``decode`` methods.
    :param tokenizer_kwargs: Optional per-field tokenizer kwargs. If a flat
        dict is given, it is duplicated to both ``"source"`` and ``"target"``.
        Pass ``{"source": {...}, "target": {...}}`` for asymmetric settings.
    :param only_source: If True, produce source-only samples (no target).
    :param src_pattern: Format string for source text (e.g. ``"{text}"``).
    :param trg_pattern: Format string for target text.
    :param src_name: Key for the source field in data items.
    :param trg_name: Key for the target field in data items.
    :param loss_type: ``"target"`` masks source positions from the loss;
        ``"all"`` includes both source and target in the loss.
    :param mask_prob: Token masking probability for data augmentation (train only).
    :param mask_type: Which tokens to mask: ``"source"``, ``"target"``, or ``"all"``.
    :param double_source: If True, duplicate source tokens (src + src + target).
    :param keep_columns: Extra columns from the data item to pass through.
    """

    def __init__(self,
                 data: object,
                 split: str,
                 tokenizer: object,
                 tokenizer_kwargs: dict = None,
                 only_source: bool = False,
                 src_pattern: str = '{text}',
                 trg_pattern: str = '{text}',
                 src_name: str = 'src',
                 trg_name: str = 'trg',
                 loss_type: str = 'target',
                 mask_prob: float = 0.0,
                 mask_type: str = 'source',
                 double_source: bool = False,
                 keep_columns: list = None
        ):
        if mask_type not in {'source', 'target', 'all', None}:
            raise ValueError(f"mask_type `{mask_type}` not supported")
        if only_source and src_name is None:
            raise ValueError("Source name is not specified when only_source=True")

        super().__init__(tokenizer)
        self.split = split
        self.data = data
        self.only_source = only_source
        self.source_format = src_pattern
        self.target_format = trg_pattern
        self.src_name = src_name
        self.trg_name = trg_name
        self.loss_type = loss_type if not only_source else 'source'
        self.mask_prob = mask_prob if split == 'train' else 0
        self.mask_type = mask_type if mask_prob > 0 else None
        self.double_source = double_source
        self.tokenizer_kwargs = dict() if tokenizer_kwargs is None else tokenizer_kwargs
        self.keep_columns = keep_columns

        if not(len(self.tokenizer_kwargs.keys()) == 2 and 'source' in self.tokenizer_kwargs):
            self.tokenizer_kwargs['source'] = self.tokenizer_kwargs
            self.tokenizer_kwargs['target'] = self.tokenizer_kwargs

    def __getitem__(self, idx, context_prefix=None):
        context_prefix = [] if context_prefix is None else context_prefix
        item = self.data[idx]

        # Source
        source_str = item[self.src_name] if self.src_name is not None else item
        source = self.process_format(self.source_format, source_str, self.special_tokens)
        source_ids = self.tokenizer.encode(source, **self.tokenizer_kwargs['source'])

        # Masking
        if self.mask_type in {'source', 'all'}:
            source_ids = self.mask_tokens(source_ids, self.mask_prob, self.tokenizer.mask_token_id)

        # Double source
        if self.double_source:
            source_ids = source_ids + source_ids

        source_ids = context_prefix + source_ids

        if not self.only_source:
            target_str = item[self.trg_name]
            target = self.process_format(self.target_format, target_str, self.special_tokens)

            target_ids = self.tokenizer.encode(target, **self.tokenizer_kwargs['target'])
            # Masking
            if self.mask_type in {'target', 'all'}:
                target_ids = self.mask_tokens(target_ids, self.mask_prob, self.tokenizer.mask_token_id)
        else:
            target_ids = []
            target_str = ''

        input_ids = source_ids + target_ids
        targets = input_ids[1:]
        input_ids = input_ids[:-1]
        target_mask = [0 if self.loss_type == 'target' else 1] * (len(source_ids) - 1) + [1] * (len(target_ids))

        sample = {
            'input_ids': input_ids,
            'source_ids': source_ids,
            'targets': targets,
            'target_mask': target_mask,
            'target_ids': target_ids,
            'target_str': target_str,
            'source_str': source_str
        }

        if self.keep_columns is not None:
            for col in self.keep_columns:
                if col not in item:
                    raise KeyError(f"Column `{col}` is not in the dataset")
                else:
                    sample[col] = item[col]

        return sample


class TokenClassificationProcessor(DataProcessorWithTokenizer):
    """Dataset for token-level classification tasks (e.g. NER, POS tagging).

    Tokenizes input and target fields, builds a per-token ``target_mask``, and
    optionally packs multiple examples to a fixed length (``pack_to_len``).

    :param data: Indexable data object.
    :param split: Split name.
    :param tokenizer: Tokenizer with ``encode`` method.
    :param tokenizer_kwargs: Optional kwargs passed to ``tokenizer.encode``.
    :param input_pattern: Format string for input text.
    :param target_pattern: Format string for target mask layout. Positions
        marked ``{no_loss}`` produce mask=0; the ``{text}`` slot produces mask=1.
    :param input_field: Key for the input field in data items.
    :param target_field: Key for the target field in data items.
    :param pack_to_len: If set, pack consecutive examples until this length.
    """

    def __init__(self,
                 data: object,
                 split: str,
                 tokenizer: object,
                 tokenizer_kwargs: dict = None,
                 input_pattern: str = '{bos_token}{text}{eos_token}',
                 target_pattern: str = '{no_loss}{text}{no_loss}',
                 input_field: str = 'src',
                 target_field: str = 'trg',
                 pack_to_len: int = None
                 ):
        super().__init__(tokenizer)
        self.split = split
        self.data = data
        self.input_pattern = input_pattern
        self.target_pattern = target_pattern
        self.input_field = input_field
        self.target_field = target_field
        self.tokenizer_kwargs = dict() if tokenizer_kwargs is None else tokenizer_kwargs
        self.pack_to_len = pack_to_len

        if self.data.streaming:
            self.iter_data = iter(self.data)
            self.streaming = True

    def __getitem__(self, idx):
        input_ids, targets, target_mask = [], [], []

        while True:
            item = self.get_next_item(idx)

            # Source
            inputs = item[self.input_field] if self.input_field is not None else item
            inputs = self.process_format(self.input_pattern, inputs, self.special_tokens)
            input_ids += self.tokenizer.encode(inputs, **self.tokenizer_kwargs)

            if self.target_field is not None:
                _targets = item[self.target_field]
            else:
                _targets = input_ids

            _target_mask = self.process_format(self.target_pattern, " 1 ", {'no_loss': '0'})
            _target_mask = list(map(int, _target_mask.split()))
            z_index = _target_mask.index(1)
            left_size = len(_target_mask[:z_index])
            right_size = len(_target_mask[z_index + 1:])

            target_mask += [0] * left_size + [1] * len(_targets) + [0] * right_size
            targets += [-1] * left_size + _targets + [-1] * right_size

            if self.pack_to_len is None:
                break
            elif len(input_ids) >= self.pack_to_len:
                input_ids = input_ids[:self.pack_to_len]
                targets = targets[:self.pack_to_len]
                target_mask = target_mask[:self.pack_to_len]
                break

        return {
            'input_ids': input_ids,
            'targets': targets,
            'target_mask': target_mask
        }

    def get_next_item(self, idx):
        if self.streaming:
            item = next(self.iter_data, None)
            if item is None:
                self.iter_data = iter(self.data)
                item = next(self.iter_data)
        else:
            item = self.data[idx]

        return item

