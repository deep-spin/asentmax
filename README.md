# asentmax

Code for [Long-Context Generalization with Sparse Attention](https://arxiv.org/abs/2506.16640).

For the attention module, see [attention.py](https://github.com/deep-spin/asentmax/blob/main/attention.py#L452), which can be integrated into torchtitan.

## Requirements

Install [AdaSplash](https://github.com/deep-spin/adasplash) for the entmax attention kernel.

WIP

## Data Generation

The data generation script `scripts/generate_data.py` creates two files per split, e.g., `train.src` and `train.trg`.
The script generates evaluation data for each sequence length.
The `--mdps_seq_len` argument specifies all sequence lengths required for generation.
You can also specify any other argument that you want to modify for each sequence length.
For example, to maintain the same length per sample, one needs to set `--mdps_vary_len` to all zeros; otherwise, the script will use the `--vary_len` value that is used for the training split. 
By default, `--mdps_seq_len` is set to `"64 128 256 512 1024 2048 4096"`, while `--mdps_vary_len` is set to `"0 0 0 0 0 0 0"`.


### MQMTAR
```
python3 scripts/generate_data.py \
    --task_type mqmtar \
    --out_dir <path> \
    --train_size 50000000 \
    --vocab_size 10000 \
    --seq_len 48 \
    --vary_len 16 \
    --mdps_seq_len "64 128 256 512 1024 2048 4096 8192 16384 32768 65536" \
    --mdps_vary_len "0 0 0 0 0 0 0 0 0 0 0" \
    --abc_size 256 \
    --k_len 2 \
    --v_len 2 \
    --num_q 4 \
    --num_kv 0.8
```

### Sort
```
python3 scripts/generate_data.py \
    --task_type sort \
    --out_dir <path> \
    --train_size 40000000 \
    --seq_len 48 \
    --vary_len 16 \
    --vocab_size 32
```

### Copy
```
python3 scripts/generate_data.py \
    --task_type copy \
    --out_dir <path> \
    --train_size 20000000 \
    --seq_len 48 \
    --vary_len 16 \
    --vocab_size 32
```
### Reverse
```
python3 scripts/generate_data.py \
    --task_type reverse \
    --out_dir <path> \
    --train_size 30000000 \
    --seq_len 48 \
    --vary_len 16 \
    --vocab_size 32
```

### LocalCount
```
python3 scripts/generate_data.py \
    --task_type count \
    --out_dir <path> \
    --train_size 10000000 \
    --seq_len 96 \
    --vary_len 32 \
    --vocab_size 16 \
    --max_reps 48
```

### 2Back
```
python3 scripts/generate_data.py \
    --task_type nback \
    --out_dir <path> \
    --train_size 10000000 \
    --vocab_size 16 \
    --seq_len 48 \
    --vary_len 16 \
    --num_steps 2
```

### Flip-Flop
```
python3 scripts/generate_data.py \
    --task_type flip-flop \
    --out_dir <path> \
    --train_size 10000000 \
    --write_prob 0.8 \
    --seq_len 48 \
    --vary_len 16
```


