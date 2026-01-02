# AutoMAX CLI Usage Guide

## Quick Start

The AutoMAX CLI can be accessed through the main entry point:

```bash
python main.py --interface cli [OPTIONS]
```

Or directly:

```bash
python -m src.ui.cli [OPTIONS]
```

## Basic Usage

### Minimal Example

```bash
python main.py --interface cli
```

This runs with default settings:
- Model: ResNet18
- Dataset: CIFAR10
- Target metric: AUROC
- Optimizer: PESG
- Loss: AUCMLoss
- Trials: 5
- Epochs: 50

### Common Examples

**Optimize AUROC on CIFAR10 with ResNet20:**
```bash
python main.py --interface cli --model resnet20 --dataset CIFAR10 --target AUROC --n_trials 10
```

**Optimize AUPRC with custom epochs:**
```bash
python main.py --interface cli --target AUPRC --epochs 100 --batch_size 64
```

**Use pretrained model:**
```bash
python main.py --interface cli --pretrained --model_path ./pretrain/model.pth
```

**Optimize OPAUC (Partial AUC):**
```bash
python main.py --interface cli --target OPAUC --max_fpr 0.3
```

**Optimize TPAUC (Two-way Partial AUC):**
```bash
python main.py --interface cli --target TPAUC --max_fpr 0.3 --min_tpr 0.7
```

## Command-Line Arguments

### Model & Dataset Options

| Argument | Choices | Default | Description |
|----------|---------|---------|-------------|
| `--model` | resnet18, resnet20, resnet32 | resnet18 | Model architecture |
| `--dataset` | CIFAR10, CIFAR100 | CIFAR10 | Dataset to use |
| `--pretrained` | - | False | Use pretrained model |
| `--model_path` | - | - | Path to pretrained model weights |

### Target Metric Options

| Argument | Choices | Default | Description |
|----------|---------|---------|-------------|
| `--target` | AUROC, AUPRC, OPAUC, TPAUC, Rank | AUROC | Target metric to optimize |
| `--max_fpr` | float | 0.3 | Maximum FPR for OPAUC/TPAUC |
| `--min_tpr` | float | 0.7 | Minimum TPR for TPAUC |

### Training Options

| Argument | Type | Default | Description |
|----------|------|---------|-------------|
| `--seed` | int | 123 | Random seed |
| `--batch_size` | int | 128 | Training batch size |
| `--eval_batch_size` | int | 128 | Evaluation batch size |
| `--sampling_rate` | float | 0.2 | Sampling rate |
| `--epochs` | int | 50 | Number of training epochs |
| `--decay_epochs` | str (JSON) | "[0.5, 0.75]" | Learning rate decay epochs |
| `--num_workers` | int | 0 | Number of data loader workers |
| `--output_path` | str | ./output | Output directory |

### AutoMAX Configuration

| Argument | Type | Default | Description |
|----------|------|---------|-------------|
| `--n_trials` | int | 5 | Number of optimization trials |
| `--n_configs` | int | 1 | Number of initial configurations |

### Optimizer & Loss Options

| Argument | Choices | Default | Description |
|----------|---------|---------|-------------|
| `--optimizer` | PESG, PDSCA, SOAP, SOPA, SOPAs, SOTAs, SONG, SGD, Adam | PESG | Optimizer to use |
| `--loss` | AUCMLoss, CompositionalAUCLoss, APLoss, pAUC_CVaR_Loss, pAUC_DRO_Loss, tpAUC_KL_Loss, NDCGLoss, CrossEntropyLoss | AUCMLoss | Loss function to use |

## Advanced Examples

**Full configuration example:**
```bash
python main.py --interface cli \
    --model resnet32 \
    --dataset CIFAR100 \
    --target AUROC \
    --optimizer SOAP \
    --loss AUCMLoss \
    --batch_size 64 \
    --epochs 100 \
    --n_trials 20 \
    --seed 42 \
    --output_path ./results \
    --decay_epochs "[0.3, 0.6, 0.9]"
```

**Custom learning rate schedule:**
```bash
python main.py --interface cli \
    --epochs 200 \
    --decay_epochs "[0.25, 0.5, 0.75]"
```

## Getting Help

To see all available options:

```bash
python main.py --interface cli --help
```

Or:

```bash
python -m src.ui.cli --help
```

## Output

Results are saved to the directory specified by `--output_path` (default: `./output`). The optimization process will display progress information in the console.

