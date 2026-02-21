from abc import ABC, abstractmethod
import argparse
from pathlib import Path
from tqdm import tqdm
import numpy as np
import random


class SourceTargetGenerator(ABC):
    """
    Abstract base class for generating source-target sequence pairs.

    Subclasses implement ``sample_func`` to define how individual
    (source, target) pairs are created.  The ``generate`` method
    writes ``num_samples`` pairs to parallel ``.src`` / ``.trg``
    text files.

    Attributes:
        config: Namespace holding all generation hyper-parameters.
    """

    def __init__(self, config):
        """
        Initialise the generator and validate the supplied configuration.

        Args:
            config: a config with at least ``seq_len`` and
                    ``vary_len`` attributes.
        """
        self.validate_config(config)
        self.config = config

    @abstractmethod
    def sample_func(self) -> tuple[list[int], list[int]]:
        """
        Produce a single (source, target) pair using ``self.config``.

        Must be implemented by every concrete subclass.

        Returns:
            tuple[list[int], list[int]]: source and target token lists.
        """
        pass

    def validate_config(self, config):
        """
        Base validation shared by all tasks.
        """
        if config.vary_len >= config.seq_len:
            raise ValueError(
                f"vary_len ({config.vary_len}) must be less than seq_len {config.seq_len})"
            )
        if config.vocab_size < 2:
            raise ValueError(
                f"vocab_size ({config.vocab_size}) must be at least 2."
            )

    def pre_generate(self):
        """
        Optional hook called once before the sample loop in ``generate``.

        Subclasses may override this to build vocabularies or other
        shared state.
        """
        pass

    def generate(self, filename: Path, num_samples: int):
        """
        Write ``num_samples`` (source, target) pairs to disk.

        Two files are created: ``<filename>.src`` and ``<filename>.trg``,
        each containing one space-separated token sequence per line.

        Args:
            filename: pathlib.Path (without extension) used as the base
                      name for the output files.
            num_samples: Number of sample pairs to generate.
        """
        src_file = filename.with_suffix('.src')
        trg_file = filename.with_suffix('.trg')

        self.pre_generate()

        src_file = open(src_file, 'w',  encoding='utf-8')
        trg_file = open(trg_file, 'w',  encoding='utf-8')

        for _ in tqdm(range(num_samples)):
            source, target = self.sample_func()

            src_file.write(f"{SourceTargetGenerator.sample_to_str(source)}\n")
            trg_file.write(f"{SourceTargetGenerator.sample_to_str(target)}\n")

        src_file.close()
        trg_file.close()

    def sample_length(self) -> int:
        """
        Sample a random sequence length within the configured range.

        The length is drawn uniformly from
        ``[seq_len - vary_len, seq_len + vary_len]``.

        Returns:
            int: Sampled sequence length.
        """
        return np.random.randint(
            self.config.seq_len - self.config.vary_len,
            self.config.seq_len + self.config.vary_len + 1,
            1
        ).item()

    
    @staticmethod
    def sample_to_str(sample: list[int]) -> str:
        """
        Convert a list of token ids to a space-separated string.

        Args:
            sample: List of integer token ids.

        Returns:
            str: Space-delimited string representation.
        """
        return " ".join(map(str, sample))


class MultiQueryMultiTokenRecall(SourceTargetGenerator):
    """
    Multi-query multi-token associative recall task.

    A sequence of key-value pairs is generated.  The source contains
    the key-value pairs followed by a set of query keys; the target
    contains the corresponding values.

    Vocabulary size must be <= ``abc_size ** k/v_len`` to guarantee
    unique keys and values.

    Attributes:
        k_vocab (np.ndarray): Generated key vocabulary matrix.
        v_vocab (np.ndarray): Generated value vocabulary matrix.
    """

    def validate_config(self, config):
        """
        Validate configuration specific to the MQMTAR task.

        Checks that:
        - ``k_len`` and ``v_len`` are positive.
        - ``num_kv`` proportion is in (0, 1].
        - ``vocab_size`` does not exceed the number of unique tokens
          that ``abc_size`` and key/value lengths can produce.
        - The sequence is long enough to hold at least ``num_q`` kv pairs.
        - ``seq_len // num_kv > 1`` (room for kv pairs and spacing).
        - ``num_q <= num_kv`` for any possible sampled num_kv.

        Args:
            config: argparse.Namespace to validate.

        Raises:
            ValueError: If any constraint is violated.
        """
        super().validate_config(config)
        if config.k_len <= 0 or config.v_len <= 0:
            raise ValueError("k_len and v_len must be positive integers.")
        if not (0 < config.num_kv <= 1):
            raise ValueError(
                f"num_kv ({config.num_kv}) must be in the range (0, 1]."
            )
        max_possible_vocab = config.abc_size ** max(config.k_len, config.v_len)
        if config.vocab_size > max_possible_vocab:
            raise ValueError(
                f"vocab_size ({config.vocab_size}) exceeds the maximum number of "
                f"unique words that abc_size={config.abc_size} and "
                f"k/v_len={config.k_len}/{config.v_len} can produce ({max_possible_vocab})."
            )
        # Constraints previously asserted inside gen_kv_pairs
        min_seq_len = config.seq_len - config.vary_len
        kv_unit = config.k_len + config.v_len + 2  # +2 for sep and bound
        min_num_kv = int(min_seq_len * config.num_kv // kv_unit)
        min_num_kv = min(min_num_kv, config.vocab_size)
        if min_seq_len // max(min_num_kv, 1) < 2:
            raise ValueError(
                f"Minimum sequence length ({min_seq_len}) is too short relative to "
                f"num_kv proportion ({config.num_kv}). seq_len // num_kv must be > 1."
            )
        if config.num_q > min_num_kv:
            raise ValueError(
                f"num_q ({config.num_q}) exceeds the minimum number of kv pairs "
                f"that can fit in the shortest possible sequence ({min_num_kv})."
            )

    def pre_generate(self):
        """
        Build unique key and value vocabularies.

        Generates all possible key and value token combinations from the
        alphabet, then randomly pairs ``vocab_size`` unique keys with
        ``vocab_size`` unique values to form the kv-pair vocabulary.
        """
        ids = np.arange(self.config.start_ids_from,
                        self.config.abc_size + self.config.start_ids_from)

        all_keys = np.array(
            np.meshgrid(*[ids] * self.config.k_len)
        ).T.reshape(-1, self.config.k_len)

        all_vals = np.array(
            np.meshgrid(*[ids] * self.config.v_len)
        ).T.reshape(-1, self.config.v_len)

        k_idx = np.random.choice(len(all_keys), self.config.vocab_size, replace=False)
        v_idx = np.random.choice(len(all_vals), self.config.vocab_size, replace=False)

        self.k_vocab = all_keys[k_idx]
        self.v_vocab = all_vals[v_idx]

    def sample_func(self):
        """
        Generate one MQMTAR (source, target) pair.

        The source is a sequence of interleaved key-value pairs
        (separated by special tokens) followed by query keys.
        The target contains the values that correspond to the queries.

        Returns:
            tuple[list[int], list[int]]: (source, target) token lists.
        """
        config = self.config
        seq_len = self.sample_length()

        num_kv = int(seq_len * config.num_kv // (config.k_len + config.v_len + 2))  # +2 for sep and bound
        num_kv = num_kv if num_kv <= self.k_vocab.shape[0] else self.k_vocab.shape[0]

        k, v, src = self.gen_kv_pairs(seq_len, num_kv)

        q_ids = np.random.randint(0, num_kv, config.num_q)
        q_sep = np.full((config.num_q, 1), config.q_sep_token)
        q = np.concatenate((q_sep, k[q_ids]), axis=1).flatten()

        src = np.concatenate((src, q))
        trg = np.concatenate((q_sep, v[q_ids]), axis=1).flatten()

        return src.tolist(), trg.tolist()

    def gen_kv_pairs(self, seq_len, num_kv):
        """
        Construct a noisy key-value pair sequence.

        Keys and values are drawn from the pre-generated vocabularies,
        concatenated with separator tokens, and embedded within a
        background of space tokens.

        Args:
            seq_len: Total desired length of the source prefix
                     (before queries are appended).
            num_kv: Number of key-value pairs to include.

        Returns:
            tuple[np.ndarray, np.ndarray, np.ndarray]:
                keys, values, and the assembled source array.
        """
        del_token_id = -100
        rep_token_id = -200

        k_idx = np.random.choice(self.k_vocab.shape[0], num_kv, replace=False)
        v_idx = np.random.choice(self.v_vocab.shape[0], num_kv, replace=False)
        k = self.k_vocab[k_idx]
        v = self.v_vocab[v_idx]

        sep = np.full((num_kv, 1), self.config.kv_sep_token)
        # To be replaced with space token
        bound = np.full((num_kv, 1), rep_token_id)

        kv = np.concatenate((k, sep, v, bound), axis=1)

        zeros_len = (seq_len // num_kv + 1) * 2
        sample = np.full((num_kv, zeros_len), self.config.space_token)

        ins_from = zeros_len // 2 - self.config.kv_len // 2
        ins_to = ins_from + self.config.kv_len + 1  # +1 for bound token
        sample[:, ins_from:ins_to] = kv

        zeros_pos = np.concatenate((np.arange(ins_from), np.arange(ins_to, zeros_len))).reshape(1, -1).repeat(num_kv, 0)
        rng = np.random.default_rng()
        slice_size = zeros_len - seq_len // num_kv
        y_ids = rng.permuted(zeros_pos, axis=1)[:, :slice_size].flatten()
        x_ids = np.arange(num_kv).repeat(slice_size)
        sample[x_ids, y_ids] = del_token_id
        sample = sample[sample != del_token_id]

        # Replace with space token
        sample[sample == rep_token_id] = self.config.space_token

        return k, v, sample


class Sort(SourceTargetGenerator):
    """
    Sorting task.

    The source is a random integer sequence and the target is the
    same sequence sorted in ascending order.
    """

    def sample_func(self):
        """
        Generate one sorting (source, target) pair.
        """
        seq_len = self.sample_length()
        src = np.random.randint(0, self.config.vocab_size, seq_len)
        trg = np.sort(src)

        return src.tolist(), trg.tolist()


class Copy(SourceTargetGenerator):
    """
    Identity copying task.

    The target is an exact copy of the source sequence.
    """

    def sample_func(self):
        """
        Generate one copying (source, target) pair.
        """
        seq_len = self.sample_length()
        src = np.random.randint(0, self.config.vocab_size, seq_len)
        src = src.tolist()

        return src, src


class Reverse(SourceTargetGenerator):
    """
    Sequence reversal task.

    The target is the source sequence in reverse order.
    """

    def sample_func(self):
        """
        Generate one reverse (source, target) pair.
        """
        seq_len = self.sample_length()
        src = np.random.randint(0, self.config.vocab_size, seq_len)
        trg = np.flip(src)

        return src.tolist(), trg.tolist()


class Parity(SourceTargetGenerator):
    """
    Parity check task.

    The source is a binary sequence and the target is a single token
    indicating whether the number of 1s is even (1) or odd (0).
    """

    def sample_func(self):
        """
        Generate one parity (source, target) pair.

        Returns:
            tuple[list[int], list[int]]: (binary sequence, [parity bit]).
        """
        seq_len = self.sample_length()
        src = np.random.randint(0, 2, seq_len)
        trg = [int(not src.sum() % 2)]

        return src.tolist(), trg


class LocalCount(SourceTargetGenerator):
    """
    A task to check the model’s ability to count repetitions of local tokens.

    The source is a sequence of repeated numbers (each number repeated
    a random number of times), and the target is the 0-based repetition
    index (the number of repeated tokens so far) for each token position.
    """

    def validate_config(self, config):
        """
        Validate configuration for the local counting task.

        Checks that:
        - ``max_reps`` is at least 2.
        """
        super().validate_config(config)
        if config.max_reps < 2:
            raise ValueError(
                f"max_reps ({config.max_reps}) must be at least 2."
            )

    def sample_func(self):
        """
        Generate (source, target) pair.

        Returns:
            tuple[list[int], list[int]]: (numbers, 0-based counts).
        """
        max_reps = self.config.max_reps
        seq_len = self.sample_length()
        num_blocks = int(seq_len // ((max_reps + 2) / 2))

        counts = np.tile(np.arange(1, max_reps + 1), (num_blocks, 1))
        numbers = self.non_repeated_sequence(self.config.vocab_size, num_blocks).reshape(-1, 1)
        numbers = np.tile(numbers, (1, max_reps))

        mask_block = np.random.randint(2, max_reps + 1, size=(num_blocks, 1))
        mask = counts <= mask_block
        counts = counts * mask
        numbers = numbers * mask

        counts = counts[counts != 0] - 1
        numbers = numbers[numbers != 0] - 1

        return numbers.tolist(), counts.tolist()

    def non_repeated_sequence(self, vocab_size, seq_size):
        """
        Generate a sequence of integers with no two consecutive equal values.

        Args:
            vocab_size: Upper bound (exclusive) for token ids (sampled
                        from ``[1, vocab_size]``).
            seq_size: Desired length of the output sequence.

        Returns:
            np.ndarray: 1-D array of length ``seq_size`` with no
                        adjacent duplicates.
        """
        numbers = np.random.randint(1, vocab_size + 1, seq_size)
        if seq_size % 2 != 0:
            numbers = np.concatenate((numbers, np.array([-1])))

        while True:
            rep_mask = numbers[:-1] == numbers[1:]
            num_reps = rep_mask.sum()
            if num_reps == 0:
                break
            rep_indices = np.where(rep_mask)[0]
            numbers[rep_indices] = np.random.randint(1, vocab_size + 1, num_reps)

        if seq_size % 2 != 0:
            numbers = numbers[:-1]

        return numbers


class NBack(SourceTargetGenerator):
    """
    N-back memory task.

    The target at each position is the source token that appeared
    ``num_steps`` positions earlier, or 0 for the first ``num_steps``
    positions.
    """

    def validate_config(self, config):
        """
        Validate configuration for the n-back task.

        Checks that:
        - ``num_steps`` is at least 1.
        - ``num_steps`` < minimum possible sequence length.
        """
        super().validate_config(config)
        if config.num_steps < 1:
            raise ValueError(
                f"num_steps ({config.num_steps}) must be at least 1."
            )
        min_len = config.seq_len - config.vary_len
        if config.num_steps >= min_len:
            raise ValueError(
                f"num_steps ({config.num_steps}) must be less than the minimum "
                f"sequence length ({min_len})."
            )

    def sample_func(self):
        """
        Generate one n-back (source, target) pair.

        Returns:
            tuple[list[int], list[int]]: (source, shifted-source) token lists.
        """
        seq_len = self.sample_length()
        num_steps = self.config.num_steps

        source = np.concatenate((
            np.array([0]), 
            np.random.randint(1, self.config.vocab_size, size=seq_len)
        ))
        target = np.roll(source, num_steps)
        target[:num_steps] = 0
        
        return source.tolist(), target.tolist()


class FlipFlop(SourceTargetGenerator):
    """
    Flip-flop latent-state recall task.

    A sequence of (command, bit) pairs is generated.  Commands are
    either *write* (token 2) or *ignore* (token 3), with a final
    *read* command (token 4).  The target is the bit value of the
    last write operation.
    """

    def validate_config(self, config):
        """
        Validate configuration for the flip-flop task.

        Checks that:
        - ``write_prob`` is in (0, 1).
        - The minimum sequence length is at least 4 (need room for
          at least one write, one ignore, and the read command).
        """
        super().validate_config(config)
        if not (0 < config.write_prob < 1):
            raise ValueError(
                f"write_prob ({config.write_prob}) must be in the open interval (0, 1)."
            )
        min_len = config.seq_len - config.vary_len
        if min_len < 4:
            raise ValueError(
                f"Minimum sequence length ({min_len}) must be at least 4 for flip-flop."
            )

    def sample_func(self):
        """
        Generate one flip-flop (source, target) pair.

        The source is a flattened sequence of (command, bit) pairs
        (minus the last bit).  The target is the bit associated with
        the last write command.

        Returns:
            tuple[list[int], list[int]]: (flattened commands+bits, [recalled bit]).
        """
        config = self.config
        seq_len = self.sample_length() // 2

        bits = np.random.randint(0, 2, size=(seq_len,))
        cmd = np.random.choice(np.array([2, 3]), p=[config.write_prob, 1 - config.write_prob], size=(seq_len - 2,))
        cmd = np.concatenate((np.array([2]), cmd, np.array([4])))

        last_write_pos = np.argwhere(cmd == 2)[-1].item()

        inp = np.stack((cmd, bits)).transpose()
        target = [inp[last_write_pos][1].item()]
        source = inp.flatten()[:-1].tolist()

        return source, target



def generate_splits(config, task_func):
    """
    Generate train / validation / test splits for a given task.

    If MDPS (Multiple Datasets Per Split) arguments are present, each
    evaluation split is generated once per parameter combination.

    Args:
        config: argparse.Namespace with output directory, split sizes,
                and task-specific hyper-parameters.
        task_func: A ``SourceTargetGenerator`` subclass to instantiate.
    """
    split_info = mdps_info(config)
    if not len(split_info):
        split_info = [('train', config.train_size, {}), ('validation', config.dev_size, {}), ('test', config.test_size, {})]

    for name, num_samples, params in split_info:
        for key, value in params.items():
            setattr(config, key, value)

        task = task_func(config)
        task.generate(config.out_dir / name, num_samples=num_samples)


def mdps_info(args):
    """
    Parse Multiple Datasets Per Split (MDPS) arguments.

    MDPS allows generating multiple evaluation datasets with different
    hyper-parameter values (e.g. different sequence lengths) in a single
    run.  Arguments prefixed with ``mdps_`` define per-split parameter
    sweeps.

    Args:
        args: argparse.Namespace containing ``mdps_*`` attributes.

    Returns:
        list[tuple[str, int, dict]]: List of (filename, num_samples,
        param_overrides) triples.  Empty if no MDPS arguments are found.
    """
    param_lists = {}  # param_name -> list of typed values
    cur_info = dict()

    for key, value in args._get_kwargs():
        if 'mdps_' in key and key not in {'mdps_file_format', 'mdps_key_for_file'}:
            param_name = key[5:]
            cur_info[param_name] = getattr(args, param_name)
            arg_type = type(cur_info[param_name])
            param_lists[param_name] = list(map(arg_type, value.split()))

    if not param_lists:
        return []

    # Validate all param lists have the same length
    lengths = [len(v) for v in param_lists.values()]
    if len(set(lengths)) != 1:
        raise ValueError(
            f"All mdps_ parameters must have the same number of values, "
            f"got lengths: {dict(zip(param_lists.keys(), lengths))}"
        )

    num_combos = lengths[0]
    param_names = list(param_lists.keys())

    # Build a list of dicts, one per parameter combination index
    mdps_args = [
        {name: param_lists[name][i] for name in param_names}
        for i in range(num_combos)
    ]

    file_name_key = args.mdps_key_for_file
    info = [
        (
            args.mdps_file_format.format(split=split, index=i, val=params[file_name_key]),
            size,
            params
        )
        for split, size in [('validation', args.dev_size), ('test', args.test_size)]
            for i, params in enumerate(mdps_args)
    ]

    info += [('train', args.train_size, cur_info)]

    return info


def generate_task(args):
    """
    Top-level dispatcher: set up output directory and special tokens,
    then delegate to ``generate_splits`` with the correct task class.

    Args:
        args: argparse.Namespace with ``task_type`` and all relevant
              hyper-parameters.

    Raises:
        NotImplementedError: If ``task_type`` is unrecognised.
    """
    out_dir = Path(args.out_dir)
    out_dir.mkdir(exist_ok=True)
    args.out_dir = out_dir

    if args.task_type == "mqmtar":
        args.space_token = 0
        args.kv_sep_token = 1
        args.q_sep_token = 2
        args.start_ids_from = 3
        args.kv_len = args.k_len + args.v_len + 1

        generate_splits(args, MultiQueryMultiTokenRecall)
    elif args.task_type == "sort":
        generate_splits(args, Sort)
    elif args.task_type == "count":
        generate_splits(args, LocalCount)
    elif args.task_type == "nback":
        generate_splits(args, NBack)
    elif args.task_type == "reverse":
        generate_splits(args, Reverse)
    elif args.task_type == "copy":
        generate_splits(args, Copy)
    elif args.task_type == "parity":
        generate_splits(args, Parity)
    elif args.task_type == "flip-flop":
        generate_splits(args, FlipFlop)
    else:
        raise NotImplementedError


if __name__ == "__main__":
    root_path = Path(__file__).parent.resolve()

    ap = argparse.ArgumentParser(
        description="Generate synthetic source-target sequence datasets for "
                    "various algorithmic tasks (sorting, copying, reversal, "
                    "parity, associative recall, counting, n-back, flip-flop)."
    )
    ap.add_argument(
        "--seed", type=int, default=42,
        help="Random seed for reproducibility."
    )
    ap.add_argument(
        "--out_dir",
        type=str,
        default="",
        help="Path to the output directory where .src/.trg files will be written.",
    )
    ap.add_argument(
        "--train_size", type=int, default=10_000_000,
        help="Number of training samples to generate."
    )
    ap.add_argument(
        "--dev_size", type=int, default=1000,
        help="Number of validation (dev) samples to generate."
    )
    ap.add_argument(
        "--test_size", type=int, default=1000,
        help="Number of test samples to generate."
    )

    # Multiple Datasets Per Split (MDPS).
    ap.add_argument(
        "--mdps_seq_len", type=str, default="64 128 256 512 1024 2048 4096",
        help="Space-separated sequence lengths for MDPS evaluation splits."
    )
    ap.add_argument(
        "--mdps_vary_len", type=str, default="0 0 0 0 0 0 0",
        help="Space-separated vary_len values corresponding to each MDPS seq_len."
    )
    ap.add_argument(
        "--mdps_key_for_file", type=str, default="seq_len",
        help="MDPS parameter name used in output file naming."
    )
    ap.add_argument(
        "--mdps_file_format", type=str, default="{split}_{index}_{val}",
        help="Format string for MDPS output filenames. Available placeholders: "
             "{split}, {index}, {val}."
    )

    # Tasks' parameters
    ap.add_argument(
        "--abc_size", type=int, default=256,
        help="[MQMTAR] Alphabet size — number of distinct token ids available for "
             "vocabulary generation."
    )
    ap.add_argument(
        "--vocab_size", type=int, default=32,
        help="Number of unique words in the vocabulary. "
             "For the MQMTAR task this is the number of key-value pairs; "
             "for sorting/copying/reverse tasks it is the token range."
    )
    ap.add_argument(
        "--k_len", type=int, default=2,
        help="[MQMTAR] Length of each key token sequence."
    )
    ap.add_argument(
        "--v_len", type=int, default=2,
        help="[MQMTAR] Length of each value token sequence."
    )
    ap.add_argument(
        "--num_kv", type=float, default=0.8,
        help="[MQMTAR] Proportion of the sequence length devoted to key-value "
             "pairs, in (0, 1]."
    )
    ap.add_argument(
        "--num_q", type=int, default=4,
        help="[MQMTAR] Number of query keys appended to the source; must be "
             "<= effective num_kv."
    )
    ap.add_argument(
        "--seq_len", type=int, default=48,
        help="Base sequence length before vary_len randomisation."
    )
    ap.add_argument(
        "--vary_len", type=int, default=16,
        help="Half-width of the uniform length variation around seq_len. "
             "Actual length is sampled from [seq_len - vary_len, seq_len + vary_len]. "
             "Must be strictly less than seq_len."
    )
    ap.add_argument(
        "--max_reps", type=int, default=12,
        help="[LocalCount] Maximum number of repetitions per block."
    )
    ap.add_argument(
        "--num_steps", type=int, default=1,
        help="[NBack] Number of positions to look back."
    )
    ap.add_argument(
        "--write_prob", type=float, default=0.1,
        help="[FlipFlop] Probability of a write command at each position, "
             "in (0, 1)."
    )

    ap.add_argument(
        "--task_type",
        choices=["mqmtar", "sort", "count", "reverse", "nback", "copy", "parity", "flip-flop"],
        default="sort",
        help="Type of algorithmic task to generate data for. "
             "Choices: mqmtar (multi-query multi-token associative recall), "
             "sort, count (local counting), reverse, nback, copy, parity, flip-flop."
    )

    args = ap.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)

    generate_task(args)