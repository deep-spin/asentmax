class SimpleTokenizer:
    """HuggingFace-compatible tokenizer for space-separated integer sequences.

    Tokens are plain integers (e.g. ``"3 14 7"`` → ``[3, 14, 7]``). Five
    special tokens (pad, bos, eos, mask, sep) are appended above
    ``max_vocab_id``, so ``vocab_size = max_vocab_id + 6``.

    :param max_vocab_id: Highest data token ID. Special tokens are assigned
        IDs starting from ``max_vocab_id + 1``. Required.
    :param max_len: Optional maximum sequence length; longer inputs are
        truncated during encoding.
    """

    def __init__(self, *args, **kwargs):
        self.max_vocab_id = kwargs.get("max_vocab_id", None)
        self.max_len = kwargs.get("max_len", None)
        assert self.max_vocab_id is not None

        self.special_tokens_map = dict()

        for token in ["pad_token", "bos_token", "eos_token", "mask_token", "sep_token"]:
            self._add_special_token(token)

        self.vocab_size = self.max_vocab_id + 1

    def _add_special_token(self, token_name):
        """Register a special token with the next available ID."""
        self.max_vocab_id += 1
        setattr(self, f"{token_name}_id", self.max_vocab_id)
        setattr(self, f"{token_name}", f"{self.max_vocab_id}")
        self.special_tokens_map[token_name] = self.max_vocab_id

    def __call__(self, s, max_length=None, **kwargs):
        """Encode a space-separated integer string into a list of ints."""
        tokens = [int(x) for x in s.split()]
        if self.max_len is not None:
            tokens = tokens[:self.max_len]

        return tokens

    def encode(self, *args, **kwargs):
        """Alias for ``__call__``."""
        return self.__call__(*args, **kwargs)

    def decode(self, token_ids, **kwargs):
        """Decode token IDs back into a space-separated string."""
        if not isinstance(token_ids, list):
            token_ids = token_ids.cpu().tolist()
        return ' '.join([str(x) for x in token_ids])

    def batch_decode(self, decode_ids, *args, **kwargs):
        """Decode a batch of token ID tensors into a list of strings."""
        return [self.decode(decode_ids[i]) for i in range(decode_ids.shape[0])]

    def __len__(self):
        """Return the total vocabulary size (data tokens + special tokens)."""
        return self.vocab_size