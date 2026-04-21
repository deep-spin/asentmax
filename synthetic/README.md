
# ASEntmax

Adaptive Sparse Entmax attention for Gemma-2 models with learnable sparsity and length-based attention scaling.

Built on [Lightning-Hydra template](https://github.com/ashleve/lightning-hydra-template/).

## Installation

```bash
pip install -r requirements.txt
```

### Environment variables

Create a `.env` file in the root directory:

```
WANDB_API_KEY=<your key>
PROJECT_ROOT=<absolute path to this repo>
```

## Project Structure

The project uses [Hydra](https://hydra.cc/) for configuration management.
Configs are composed hierarchically: base configs in `configs/` are overridden
by experiment-specific configs in `configs/experiment/`.

```
configs/
  model/sparse_gemma.yaml   # Base model config (architecture, optimizer, scheduler)
  experiment/entmax/         # Per-task experiment configs (copy, mqmtar, reverse, sort, flip-flop)
  data/                      # Dataset configs
  trainer/                   # PyTorch Lightning Trainer settings
  callbacks/                 # Checkpoint, early stopping, progress bar
  logger/                    # Weights & Biases logger
  paths/                     # Input/output directory paths
```

### Key model parameters (`configs/model/sparse_gemma.yaml`)

| Parameter | Description |
|-----------|-------------|
| `net.entmax_alpha` | Sparsity parameter: `1.0` = softmax, `1.5` = entmax 1.5 |
| `net.attn_scale_type` | Length-based scaling: `null`, `"nakanishi"`, `"adapt-softplus-tanh"`, or `"learn"` |
| `net.attn_scale_proj_bias` | Enable bias in the scaling projection layers (for `adapt-softplus-tanh`) |
| `net.apply_rotary` | Use RoPE positional encoding |
| `net.apply_nape` | Use NAPE (NoPE + ALiBi hybrid) positional encoding |
| `net.attn_implementation` | `"eager"` for entmax/adasplash, `"flash_attention_2"` for softmax |
| `net.use_fast_attn` | Use Triton-based adasplash kernel for sparse attention |
| `net.hidden_size` | Hidden dimension (overridden per experiment) |
| `net.num_attention_heads` | Number of attention heads (overridden per experiment) |
| `net.num_hidden_layers` | Number of transformer layers (overridden per experiment) |

### Experiment configs (`configs/experiment/entmax/`)

Each experiment config (`copy.yaml`, `mqmtar.yaml`, `reverse.yaml`, `sort.yaml`,
`flip-flop.yaml`) overrides the base model config with task-specific settings:
architecture size, learning rate schedule, generation parameters, and
validation monitoring.

Parameters from both the base model config and the experiment config can be
overridden from the command line using Hydra syntax. Use `++` to add new keys
or override existing ones.

## Training

### General command

```bash
python3 train.py \
    'experiment=entmax/<task>' \
    "++logger.wandb.name=\"<run name>\"" \
    data.data_provider.path=<path to data folder> \
    model.optimizer.lr=<learning rate> \
    seed=<seed>
```

Where `<task>` is one of: `copy`, `mqmtar`, `reverse`, `sort`, `flip-flop`.

### Positional encoding

RoPE:
```
model.net.apply_rotary=True \
model.net.apply_nape=False
```

NAPE:
```
model.net.apply_rotary=False \
model.net.apply_nape=True
```

### Attention variants

ASEntmax (sparse attention + adaptive scaling):
```
model.net.entmax_alpha=1.5 \
++model.net.attn_scale_type="adapt-softplus-tanh" \
++model.net.attn_scale_proj_bias=True
```

SSMax (softmax + Nakanishi scaling):
```
model.net.entmax_alpha=1.0 \
++model.net.attn_implementation=flash_attention_2 \
++model.net.attn_scale_type="nakanishi"
```

Softmax (baseline, no scaling):
```
model.net.entmax_alpha=1.0 \
++model.net.attn_implementation=flash_attention_2 \
++model.net.attn_scale_type=null
```

Entmax (sparse attention, no scaling):
```
model.net.entmax_alpha=1.5 \
++model.net.attn_scale_type=null
```

## Reproducing ASEntmax results

> Results may differ slightly due to minor config differences.

### Multi-Query Multi-Token Associative Recall (MQMTAR)

```bash
python3 train.py \
    'experiment=entmax/mqmtar' \
    "++logger.wandb.name=\"ASEntmax NAPE\"" \
    seed=1 \
    model.optimizer.lr=2e-4 \
    model.net.entmax_alpha=1.5 \
    ++model.net.apply_rotary=False \
    ++model.net.apply_nape=True \
    ++model.net.attn_scale_type="adapt-softplus-tanh" \
    ++model.net.attn_scale_proj_bias=True
```

### Reverse

```bash
python3 train.py \
    'experiment=entmax/reverse' \
    "++logger.wandb.name=\"ASEntmax NAPE\"" \
    seed=4 \
    model.optimizer.lr=4e-4 \
    model.net.entmax_alpha=1.5 \
    ++model.net.apply_rotary=False \
    ++model.net.apply_nape=True \
    ++model.net.attn_scale_type="adapt-softplus-tanh" \
    ++model.net.attn_scale_proj_bias=True
```

### Sort

```bash
python3 train.py \
    'experiment=entmax/sort' \
    "++logger.wandb.name=\"ASEntmax NAPE\"" \
    seed=4 \
    model.optimizer.lr=4e-4 \
    model.net.entmax_alpha=1.5 \
    ++model.net.apply_rotary=False \
    ++model.net.apply_nape=True \
    ++model.net.attn_scale_type="adapt-softplus-tanh" \
    ++model.net.attn_scale_proj_bias=True
```

### Copy

```bash
python3 train.py \
    'experiment=entmax/copy' \
    "++logger.wandb.name=\"ASEntmax NAPE\"" \
    data.data_provider.path=<path to data folder> \
    model.optimizer.lr=1e-3 \
    seed=1 \
    model.net.entmax_alpha=1.5 \
    ++model.net.apply_rotary=False \
    ++model.net.apply_nape=True \
    ++model.net.attn_scale_type="adapt-softplus-tanh" \
    ++model.net.attn_scale_proj_bias=True
```
