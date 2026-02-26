# AutoMAX

### Install
```
conda create -n AutoMAX python=3.10
conda activate AutoMAX
conda install gxx_linux-64 gcc_linux-64 swig
pip install -r requirements.txt
```


### AutoTune AUROC on PneumoniaMNIST

```
python -m src.ui.auto_trainer --config_file ./recipes/automax/config_auc_PneumoniaMNIST.yaml
```

### AutoTune AUROC on RIP-Dataset

```
python -m src.ui.auto_transformers_trainer --config_file ./recipes/automax/config_auc_RIP.yaml
```

### Run single experiment (without autotune)

```
python -m src.ui.run_trainer --config_file ./recipes/libauc_trainer/config_auc.yaml
```


# AutoMAX Configuration

The AutoMAX configuration file is a YAML document passed to `run.py` via `--config_file`. It is divided into five top-level sections: `dataset`, `model`, `metrics`, `training`, and `automax`. Each section is described in full below.

---

## Quick Start

```bash
python -m src.ui.auto_trainer --config_file config.yaml
```

Any `training` field can be overridden at the command line:

```bash
python -m src.ui.auto_trainer --config_file config.yaml --epochs 50 --seed 0 --output_path ./runs
```

---

## Top-Level Sections

| Section | Required | Purpose |
|---|---|---|
| `dataset` | Yes | Dataset name, splits, and loading kwargs |
| `model` | Yes | Architecture and checkpoint settings |
| `metrics` | No | Evaluation metrics (default: `[AUROC]`) |
| `training` | Yes | Optimizer, loss, schedule, and checkpointing |
| `automax` | Yes | Hyperparameter search configuration |

---

## `dataset`

Controls which dataset is loaded and how it is split.

```yaml
dataset:
  name: cifar10
  eval_splits: [val, test]
  kwargs:
    imratio: 0.1
```

| Field | Type | Required | Description |
|---|---|---|---|
| `name` | string | Yes | Dataset identifier passed to `load_dataset()` |
| `eval_splits` | list[string] | No | Splits to evaluate each epoch. Default: `[val]` |
| `kwargs` | mapping | No | Additional keyword arguments forwarded to `load_dataset()` |

**`kwargs` fields:**

| Field | Type | Description |
|---|---|---|
| `imratio` | float | Positive-class imbalance ratio (e.g. `0.1` = 10% positive samples) |

---

## `model`

Defines the model architecture and optional pretrained weights.

```yaml
model:
  name: resnet20
  pretrained: true
  pretrained_path: /path/to/checkpoint.pth
  num_classes: 1
  in_channels: 3
```

| Field | Type | Required | Description |
|---|---|---|---|
| `name` | string | Yes | Model architecture identifier passed to `build_model()` |
| `pretrained` | bool | No | Whether to load pretrained weights. Default: `false` |
| `pretrained_path` | string | No | Path to a `.pth` checkpoint file. Required if `pretrained: true` |
| `num_classes` | int | Yes | Number of output classes. Use `1` for binary classification |
| `in_channels` | int | No | Number of input channels. Default: `3` |

---

## `metrics`

A list of metric names to compute on every evaluation split after each epoch. The **first entry** is used as the optimization target by AutoMAX.

```yaml
metrics:
  - AUROC
```

| Value | Description |
|---|---|
| `AUROC` | Area Under the ROC Curve |
| `AUPRC` | Area Under the Precision-Recall Curve |
| `ACC` | Accuracy (threshold 0.5) |

If omitted, defaults to `[AUROC]`.

---

## `training`

Controls all aspects of the training loop, including the loss function, optimizer, schedule, and checkpointing. Fields in this section can be overridden from the command line (see [CLI Overrides](#cli-overrides)).

```yaml
training:
  project_name: libauc
  experiment_name: AUCMLoss_cifar10

  epochs: 100
  batch_size: 128
  eval_batch_size: 256
  sampling_rate: 0.2
  num_workers: 0
  SEED: 123

  loss: AUCMLoss
  optimizer: PESG

  decay_epochs: [0.5, 0.75]

  output_path: ./output
  resume_from_checkpoint: false
  save_checkpoint_every: 10
```

### Experiment Tracking

| Field | Type | Required | Default | Description |
|---|---|---|---|---|
| `project_name` | string | No | `libauc` | Top-level project name for run tracking |
| `experiment_name` | string | **Yes** | — | Unique name for this run |

### Core Training Parameters

| Field | Type | Required | Default | Description |
|---|---|---|---|---|
| `epochs` | int | No | `50` | Total number of training epochs |
| `batch_size` | int | No | `128` | Training mini-batch size |
| `eval_batch_size` | int | No | `128` | Batch size used during evaluation |
| `sampling_rate` | float | No | `0.5` | Positive-sample ratio for `DualSampler` |
| `num_workers` | int | No | `2` | DataLoader worker processes. Use `0` on Windows |
| `SEED` | int | No | `42` | Global random seed for reproducibility |

### Loss and Optimizer

| Field | Type | Required | Description |
|---|---|---|---|
| `loss` | string | Yes | Loss function name. See [Supported Loss / Optimizer Pairs](#supported-loss--optimizer-pairs) |
| `optimizer` | string | Yes | Optimizer name. Must be compatible with the chosen loss |
| `loss_kwargs` | mapping | No | Override individual loss hyperparameters (merged over defaults) |
| `optimizer_kwargs` | mapping | No | Override individual optimizer hyperparameters (merged over defaults) |

`loss_kwargs` and `optimizer_kwargs` accept the same hyperparameter definition format as the search spaces — either a plain scalar (constant) or a dict with `val`, `default`, and `log` keys. Values provided here are merged on top of the defaults defined in the corresponding space class, so only the fields you want to change need to be specified.

```yaml
# Example: fix learning rate and override the margin search space
training:
  optimizer_kwargs:
    lr: 0.05                     # scalar → treated as a constant
  loss_kwargs:
    margin:
      val: [0.8, 1.0]            # narrow the categorical choices
      default: 1.0
```

### Learning Rate Schedule

| Field | Type | Required | Default | Description |
|---|---|---|---|---|
| `decay_epochs` | list[float] | No | `[]` | Epoch fractions at which the learning rate is decayed (e.g. `[0.5, 0.75]` decays at 50% and 75% of training) |

### Checkpointing

| Field | Type | Required | Default | Description |
|---|---|---|---|---|
| `output_path` | string | No | `./output` | Directory where checkpoints and results are written |
| `resume_from_checkpoint` | bool | No | `true` | If `true`, resume training from the latest checkpoint in `output_path` |
| `save_checkpoint_every` | int | No | `5` | Save a `.pt` checkpoint file every N epochs |

---

## `automax`

Configures the AutoMAX hyperparameter search.

```yaml
automax:
  deterministic: true
  n_trials: 5
  SEED: 42
  name: n5_AUCMLoss_cifar10
  output_directory: ./automax_output
  overwrite: true
```

| Field | Type | Required | Default | Description |
|---|---|---|---|---|
| `name` | string | Yes | — | Unique identifier for this search run |
| `n_trials` | int | No | `5` | Number of hyperparameter configurations to evaluate |
| `SEED` | int | No | `42` | Random seed for the hyperparameter sampler |
| `deterministic` | bool | No | `true` | If `true`, enforce fully deterministic training across trials |
| `output_directory` | string | No | `./automax_output` | Directory where search results and trial logs are saved |
| `overwrite` | bool | No | `true` | Overwrite an existing search run with the same `name` |

> **Note:** The field is `output_directory` in code (`AutoMAXConfigration`), but the example YAML uses `output_path`. Use `output_directory` to ensure the value is correctly read.

---

## Supported Loss / Optimizer Pairs

Each loss function has a canonical optimizer and a predefined hyperparameter search space. Mismatched pairs will raise an error.

| `loss` | `optimizer` | Key Hyperparameters |
|---|---|---|
| `AUCMLoss` | `PESG` | `lr`, `epoch_decay`, `weight_decay`, `momentum`, `margin` |
| `CompositionalAUCLoss` | `PDSCA` | `lr`, `epoch_decay`, `weight_decay`, `margin`, `k` |
| `APLoss` | `SOAP` | `lr`, `epoch_decay`, `momentum`, `weight_decay`, `gamma`, `margin` |
| `pAUC_CVaR_Loss` | `SOPA` | `lr`, `epoch_decay`, `weight_decay`, `margin`, `beta`, `eta` |
| `pAUC_DRO_Loss` | `SOPAs` | `lr`, `epoch_decay`, `momentum`, `weight_decay`, `gamma`, `margin`, `Lambda` |
| `tpAUC_KL_Loss` | `SOTAs` | `lr`, `epoch_decay`, `momentum`, `weight_decay`, `tau`, `gammas`, `margin`, `Lambda` |
| `NDCGLoss` | `SONG` | `lr`, `epoch_decay`, `momentum`, `weight_decay`, `gamma0`, `gamma1`, `eta0`, `margin`, `sigmoid_alpha` |
| `CrossEntropyLoss` | `SGD` | `lr`, `epoch_decay`, `momentum`, `weight_decay` |
| `CrossEntropyLoss` | `Adam` | `lr`, `epoch_decay`, `weight_decay` |

---

## CLI Overrides

The following `training` fields can be overridden directly on the command line. CLI values always take precedence over the config file.

| CLI Flag | Config Field | Type |
|---|---|---|
| `--epochs` | `training.epochs` | int |
| `--batch_size` | `training.batch_size` | int |
| `--eval_batch_size` | `training.eval_batch_size` | int |
| `--sampling_rate` | `training.sampling_rate` | float |
| `--num_workers` | `training.num_workers` | int |
| `--output_path` | `training.output_path` | string |
| `--seed` | `training.SEED` | int |
| `--resume` / `--no-resume` | `training.resume_from_checkpoint` | bool |
| `--save_checkpoint_every` | `training.save_checkpoint_every` | int |

---

## Hyperparameter Definition Format

When specifying `optimizer_kwargs` or `loss_kwargs`, each value can be written in one of two ways.

**Scalar constant** — the value is fixed and not searched:
```yaml
lr: 0.01
```

**Search space dict** — the value is sampled by AutoMAX:
```yaml
lr:
  val: [0.0001, 0.1]   # tuple → uniform range; list → categorical choices; scalar → constant
  default: 0.001        # starting/default value
  log: true             # sample on log scale (valid for numeric ranges only)
```

| Sub-field | Type | Description |
|---|---|---|
| `val` | scalar / tuple / list | Scalar: constant. Tuple `(low, high)`: uniform range. List: categorical choices |
| `default` | scalar | Default value. Falls back to the lower bound for ranges, first element for lists |
| `log` | bool | Sample the range on a log scale. Ignored for categorical values |