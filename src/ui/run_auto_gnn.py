"""
Entry point for AutoMAX hyperparameter search with GNNTrainer.

Usage:
    python run_auto_gnn.py --config_file config.yaml
"""

import argparse
import logging
import sys
from functools import partial

import numpy as np
import torch
import yaml
from libauc.trainer import TrainingArguments, CLICallback

from ..config.args import (
    AutoMAXConfigration,
    parse_defaultconfig,
    parse_hyperparameters_from_dict,
    autopartial,
)
from ..core.auto_auc import AutoMAX
from ..data.datasets import load_dataset
from .helpers import build_metric
from libauc.trainer import GNNTrainer   # adjust import path as needed

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s - %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Argument parsing
# ---------------------------------------------------------------------------

def parse_args():
    parser = argparse.ArgumentParser(
        description="AutoMAX hyperparameter search for GNN training"
    )

    parser.add_argument("--config_file", type=str, required=True,
                        help="Path to a YAML configuration file.")

    parser.add_argument("--epochs",                type=int,   default=None)
    parser.add_argument("--batch_size",            type=int,   default=None)
    parser.add_argument("--eval_batch_size",       type=int,   default=None)
    parser.add_argument("--sampling_rate",         type=float, default=None)
    parser.add_argument("--num_workers",           type=int,   default=None)
    parser.add_argument("--output_path",           type=str,   default=None)
    parser.add_argument("--seed",                  type=int,   default=None)
    parser.add_argument(
        "--resume_from_checkpoint",
        action=argparse.BooleanOptionalAction,
        default=None,
    )
    parser.add_argument("--save_checkpoint_every", type=int,   default=None)

    return parser.parse_args()


# ---------------------------------------------------------------------------
# Config helpers
# ---------------------------------------------------------------------------

def load_config(config_path: str) -> dict:
    with open(config_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def apply_cli_overrides(cfg: dict, args) -> dict:
    overrides = {
        "epochs":                  args.epochs,
        "batch_size":              args.batch_size,
        "eval_batch_size":         args.eval_batch_size,
        "sampling_rate":           args.sampling_rate,
        "num_workers":             args.num_workers,
        "output_path":             args.output_path,
        "SEED":                    args.seed,
        "resume_from_checkpoint":  args.resume_from_checkpoint,
        "save_checkpoint_every":   args.save_checkpoint_every,
    }
    for key, value in overrides.items():
        if value is not None:
            logger.info(f"CLI override: training.{key} = {value}")
            cfg["training"][key] = value
    return cfg


# ---------------------------------------------------------------------------
# Reproducibility
# ---------------------------------------------------------------------------

def set_seed(seed: int):
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    logger.info(f"Global seed set to {seed}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    args = parse_args()

    logger.info(f"Loading config from: {args.config_file}")
    cfg = load_config(args.config_file)
    cfg = apply_cli_overrides(cfg, args)

    training_cfg  = cfg["training"]
    automax_cfg   = cfg["automax"]
    dataset_cfg   = cfg["dataset"]
    model_cfg     = cfg["model"]
    metric_names  = cfg.get("metrics", ["AUROC"])
    metric_kwargs = cfg.get("metric_kwargs", [])

    # GNN-specific optional keys
    decay_epochs = training_cfg.get("decay_epochs", [])
    decay_factor = training_cfg.get("decay_factor", 10.0)

    set_seed(training_cfg.get("SEED", 42))

    # ── Datasets ────────────────────────────────────────────────────────────
    dataset_name   = dataset_cfg["name"]
    dataset_kwargs = dataset_cfg.get("kwargs", {})
    eval_splits    = dataset_cfg.get("eval_splits", ["val"])

    logger.info(f"Loading train and {eval_splits} splits of dataset: {dataset_name}")
    train_dataset, eval_datasets = load_dataset(
        dataset_name, splits=eval_splits, **dataset_kwargs
    )

    labels = np.array(train_dataset.targets)
    if labels.ndim == 1:
        num_tasks = len(np.unique(labels))
    else:
        num_tasks = labels.shape[-1]
    logger.info(f"Number of tasks: {num_tasks}")

    multilabel = num_tasks >= 2

    # ── Metric ──────────────────────────────────────────────────────────────
    metric_fn = build_metric(metric_names, metric_kwargs)

    # ── Loss / optimizer kwargs with AutoMAX search-space defaults ──────────
    optimizer_kwargs = training_cfg.get("optimizer_kwargs", {})
    default_optimizer_config = parse_defaultconfig(
        training_cfg["optimizer"], multilabel, optimizer_kwargs
    )["optimizer"]
    optimizer_kwargs = {**default_optimizer_config["space"], **optimizer_kwargs}

    loss_kwargs = training_cfg.get("loss_kwargs", {})
    if multilabel:
        loss_kwargs["num_labels"] = num_tasks
    default_loss_config = parse_defaultconfig(
        training_cfg["loss"], multilabel, loss_kwargs
    )["loss"]
    loss_kwargs = {**default_loss_config["space"], **loss_kwargs}

    # ── autopartial TrainingArguments (carries search-space distributions) ──
    train_args = autopartial(
        partial(
            TrainingArguments,
            optimizer       = default_optimizer_config["type"],
            loss            = default_loss_config["type"],
            SEED            = training_cfg.get("SEED", 42),
            batch_size      = training_cfg.get("batch_size", 128),
            eval_batch_size = training_cfg.get("eval_batch_size", 128),
            sampling_rate   = training_cfg.get("sampling_rate", 0.5),
            epochs          = training_cfg.get("epochs", 50),
            decay_epochs    = decay_epochs,
            num_workers     = training_cfg.get("num_workers", 2),
            output_path     = training_cfg.get("output_path", "./output"),
            resume_from_checkpoint  = training_cfg.get("resume_from_checkpoint", True),
            save_checkpoint_every   = training_cfg.get("save_checkpoint_every", 5),
            project_name    = training_cfg.get("project_name", "libauc"),
            experiment_name = training_cfg["experiment_name"],
            verbose         = training_cfg.get("verbose", 1),
        ),
        optimizer_kwargs = autopartial(
            dict, **parse_hyperparameters_from_dict(optimizer_kwargs)
        ),
        loss_kwargs = autopartial(
            dict, **parse_hyperparameters_from_dict(loss_kwargs)
        ),
    )

    # ── AutoMAX configuration ────────────────────────────────────────────────
    automax_args = AutoMAXConfigration(
        deterministic    = automax_cfg.get("deterministic", True),
        n_trials         = automax_cfg.get("n_trials", 5),
        n_configs        = automax_cfg.get("n_configs", 1),
        SEED             = automax_cfg.get("SEED", 42),
        name             = automax_cfg.get("name"),
        output_directory = automax_cfg.get("output_directory", "./automax_output"),
        overwrite        = automax_cfg.get("overwrite", True),
    )

    # ── autopartial GNNTrainer ───────────────────────────────────────────────
    logger.info("Initialising GNNTrainer (autopartial)...")
    trainer = autopartial(
        partial(
            GNNTrainer,
            model_cfg     = model_cfg,
            train_dataset = train_dataset,
            eval_dataset  = eval_datasets if eval_datasets else None,
            metric        = metric_fn,
            callbacks     = [CLICallback()],
            decay_epochs  = decay_epochs,
            decay_factor  = decay_factor,
        ),
        train_args = train_args,
    )

    print("\n[Hyperparameter Space]")
    for key, value in trainer.cs.items():
        print(f"{value}")

    tuner = AutoMAX(trainer, automax_args, target=metric_names[0])
    tuner.optimize()


if __name__ == "__main__":
    main()