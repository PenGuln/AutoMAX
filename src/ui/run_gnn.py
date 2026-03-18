"""
Entry point for GNN training with GNNTrainer.

Usage:
    python run_gnn.py --config_file config.yaml
"""

import argparse
import logging
import sys
import numpy as np
import torch
import yaml
from libauc.trainer import TrainingArguments, CLICallback

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
    parser = argparse.ArgumentParser(description="GNN training script")

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
    """Merge CLI overrides into the loaded config dict (mutates in-place)."""
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
    dataset_cfg   = cfg["dataset"]
    model_cfg     = cfg["model"]
    metric_names  = cfg.get("metrics", ["AUROC"])
    metric_kwargs = cfg.get("metric_kwargs", [])

    # GNN-specific optional keys (with safe defaults)
    decay_epochs  = training_cfg.get("decay_epochs", [])
    decay_factor  = training_cfg.get("decay_factor", 10.0)

    set_seed(training_cfg.get("SEED", 42))

    # ── Datasets ────────────────────────────────────────────────────────────
    dataset_name   = dataset_cfg["name"]
    dataset_kwargs = dataset_cfg.get("kwargs", {})
    eval_splits    = dataset_cfg.get("eval_splits", ["val"])

    logger.info(f"Loading train and {eval_splits} splits of dataset: {dataset_name}")
    train_dataset, eval_datasets = load_dataset(
        dataset_name, splits=eval_splits, **dataset_kwargs
    )

    # ── Metric ──────────────────────────────────────────────────────────────
    metric_fn = build_metric(metric_names, metric_kwargs)

    # ── Loss / optimizer kwargs ─────────────────────────────────────────────
    optimizer_kwargs = training_cfg.get("optimizer_kwargs", {})
    loss_kwargs      = training_cfg.get("loss_kwargs", {})

    # ── TrainingArguments ───────────────────────────────────────────────────
    train_args = TrainingArguments(
        optimizer        = training_cfg["optimizer"],
        optimizer_kwargs = optimizer_kwargs,
        loss             = training_cfg["loss"],
        loss_kwargs      = loss_kwargs,
        SEED             = training_cfg.get("SEED", 42),
        batch_size       = training_cfg.get("batch_size", 128),
        eval_batch_size  = training_cfg.get("eval_batch_size", 128),
        sampling_rate    = training_cfg.get("sampling_rate", 0.5),
        epochs           = training_cfg.get("epochs", 50),
        decay_epochs     = decay_epochs,
        num_workers      = training_cfg.get("num_workers", 2),
        output_path      = training_cfg.get("output_path", "./output"),
        num_tasks        = 1,
        resume_from_checkpoint  = training_cfg.get("resume_from_checkpoint", True),
        save_checkpoint_every   = training_cfg.get("save_checkpoint_every", 5),
        project_name     = training_cfg.get("project_name", "libauc"),
        experiment_name  = training_cfg["experiment_name"],
        verbose          = training_cfg.get("verbose", 1),
    )

    # ── GNNTrainer ──────────────────────────────────────────────────────────
    logger.info("Initialising GNNTrainer...")
    trainer = GNNTrainer(
        train_args   = train_args,
        model_cfg    = model_cfg,
        train_dataset = train_dataset,
        eval_dataset  = eval_datasets if eval_datasets else None,
        metric        = metric_fn,
        callbacks     = [CLICallback()],
        decay_epochs  = decay_epochs,
        decay_factor  = decay_factor,
    )

    logger.info("Starting GNN training...")
    train_log = trainer.train()
    logger.info("Training complete.")


if __name__ == "__main__":
    main()