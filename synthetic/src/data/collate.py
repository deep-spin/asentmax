import torch
import math


class CollateObject:
    """Base collator with helpers for extracting, padding, and batching fields."""

    def __init__(self, tokenizer=None):
        self.tokenizer = tokenizer

    def extract_var(self, data, var_name, tensorize=True, is_reversed=False):
        """Extract *var_name* from each sample.

        :param data: List of sample dicts.
        :param var_name: Key to extract from each sample.
        :param tensorize: If True, wrap each value in a 1-D tensor.
        :param is_reversed: If True, reverse the sequence (used for left padding).
        :return: List of tensors (or raw values if ``tensorize=False``).
        """
        out = [d[var_name] if not is_reversed else list(reversed(d[var_name])) for d in data]
        if tensorize:
            out = [torch.tensor(d) for d in out]
        return out

    def pad(self, seq_list, pad_id):
        """Right-pad a list of 1-D tensors to equal length."""
        return torch.nn.utils.rnn.pad_sequence(seq_list, True, pad_id)

    def extract_and_pad(self, data, var_name, pad_id, right_pad=True):
        """Extract *var_name* from all samples and pad into a single tensor.

        :param right_pad: If True, pad on the right; if False, pad on the left.
        """
        if right_pad:
            return self.pad(self.extract_var(data, var_name), pad_id)
        else:
            res = self.pad(self.extract_var(data, var_name, is_reversed=True), pad_id)
            return res.flip([1])

    def concat_var(self, data, var_name):
        """Concatenate *var_name* tensors from all samples along dim 0."""
        return torch.concat(self.extract_var(data, var_name), dim=0)

    def pad_with_size(self, t, size, pad_id, right_pad=True):
        """Append (or prepend) *size* padding columns to tensor *t*."""
        B = t.shape[0]
        pads = torch.full((B, size), pad_id)
        to_cat = (t, pads) if right_pad else (pads, t)
        return torch.cat(to_cat, dim=1)


class CollateByChunk(CollateObject):
    """Collator that pads sequences so their length is divisible by *chunk_size*.

    If *chunk_size* is None, behaves as a standard pad-to-longest collator.

    :param pad_id: Token ID used for padding.
    :param chunk_size: Target chunk alignment. Sequences are padded to the
        nearest multiple of this value.
    """

    def __init__(self, pad_id, chunk_size=None):
        self.pad_id = pad_id
        self.chunk_size = chunk_size

    def __call__(self, data):
        input_ids = self.extract_and_pad(data, 'input_ids', self.pad_id)
        if self.chunk_size is not None:
            B, L = input_ids.shape
            num_to_add = math.ceil(L / self.chunk_size) * self.chunk_size - L
            extra_pads = torch.zeros(B, num_to_add, dtype=int) + self.pad_id
            input_ids = torch.cat((input_ids, extra_pads), dim=-1)

        return {
            'input_ids': input_ids,
            'attention_mask': (input_ids != self.pad_id).int(),
            'targets': torch.stack(self.extract_var(data, 'target'))
        }


class CollateSourceTarget(CollateObject):
    """Collator for source-target sequence tasks (e.g. copy, reverse, sort).

    Produces a batch dict with ``input_ids``, ``targets``, ``input_mask``,
    ``target_mask``, ``source_ids``, ``source_mask``, and string fields.

    :param tokenizer: Tokenizer (must have ``pad_token_id``).
    :param right_pad: If True, pad on the right; if False, pad on the left.
    :param extra_paddings: Fixed number of extra padding tokens to append.
    :param pad_to_len: If set, pad all sequences to exactly this length
        (overrides *extra_paddings*).
    """

    def __init__(self, tokenizer, right_pad: bool = False, extra_paddings: int = 0, pad_to_len: int = None):
        self.tokenizer = tokenizer
        self.right_pad = right_pad
        self.extra_paddings = extra_paddings
        self.pad_to_len = pad_to_len

    def __call__(self, data):
        input_ids = self.extract_and_pad(data, 'input_ids', self.tokenizer.pad_token_id, right_pad=self.right_pad)
        target_mask = self.extract_and_pad(data, 'target_mask', 0, right_pad=self.right_pad)
        targets = self.extract_and_pad(data, 'targets', -1, right_pad=self.right_pad)

        if self.pad_to_len is not None:
            cur_len = input_ids.size(-1)
            extra_paddings = self.pad_to_len - cur_len
        else:
            extra_paddings = self.extra_paddings

        if extra_paddings > 0:
            input_ids = self.pad_with_size(input_ids, extra_paddings, self.tokenizer.pad_token_id, right_pad=self.right_pad)
            target_mask = self.pad_with_size(target_mask, extra_paddings, 0, right_pad=self.right_pad)
            targets = self.pad_with_size(targets, extra_paddings, -1, right_pad=self.right_pad)

        input_mask = input_ids != self.tokenizer.pad_token_id

        target_str = self.extract_var(data, 'target_str', tensorize=False)
        source_str = self.extract_var(data, 'source_str', tensorize=False)
        source_ids = self.extract_and_pad(data, 'source_ids', self.tokenizer.pad_token_id, right_pad=self.right_pad)
        source_mask = source_ids != self.tokenizer.pad_token_id

        return {
            'input_ids': input_ids,
            'targets': targets,
            'input_mask': input_mask,
            'target_mask': target_mask.bool(),
            'target_str': target_str,
            'source_str': source_str,
            'source_ids': source_ids,
            'source_mask': source_mask,
            'batch_size': input_ids.shape[0]
        }


class CollateTokenClassification(CollateObject):
    """Collator for token-classification tasks (e.g. NER, POS tagging).

    Produces a batch dict with ``input_ids``, ``targets``, ``input_mask``,
    and ``target_mask``.

    :param tokenizer: Tokenizer (must have ``pad_token_id``).
    :param right_pad: If True, pad on the right; if False, pad on the left.
    """

    def __init__(self, tokenizer, right_pad=False, extra_paddings=0):
        self.tokenizer = tokenizer
        self.right_pad = right_pad

    def __call__(self, data):
        input_ids = self.extract_and_pad(data, 'input_ids', self.tokenizer.pad_token_id, right_pad=self.right_pad)
        target_mask = self.extract_and_pad(data, 'target_mask', 0, right_pad=self.right_pad)
        targets = self.extract_and_pad(data, 'targets', -1, right_pad=self.right_pad)
        input_mask = input_ids != self.tokenizer.pad_token_id

        return {
            'input_ids': input_ids,
            'targets': targets,
            'input_mask': input_mask,
            'target_mask': target_mask.bool(),
            'batch_size': input_ids.shape[0]
        }