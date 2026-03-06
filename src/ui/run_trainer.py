"""
Entry point for AutoMAX training.

Usage:
    python run.py --config_file config.yaml
"""

import argparse
import logging
import sys
import numpy as np
import torch
import yaml
from libauc.trainer import TrainingArguments, Trainer, CLICallback

from ..data.datasets import load_dataset
from .helpers import build_metric

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
    parser = argparse.ArgumentParser(description="AutoMAX training script")

    parser.add_argument(
        "--config_file",
        type=str,
        required=True,
        help="Path to a YAML configuration file.",
    )

    # Each flag below mirrors a TrainingArguments field and, when supplied
    # on the CLI, overrides the value from the config file.
    parser.add_argument("--epochs",                type=int,   default=None)
    parser.add_argument("--batch_size",            type=int,   default=None)
    parser.add_argument("--eval_batch_size",       type=int,   default=None)
    parser.add_argument("--sampling_rate",         type=float, default=None)
    parser.add_argument("--num_workers",           type=int,   default=None)
    parser.add_argument("--output_path",           type=str,   default=None)
    parser.add_argument("--seed",                  type=int,   default=None)
    parser.add_argument(
        "--resume_from_checkpoint",
        action=argparse.BooleanOptionalAction,   # --resume / --no-resume
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

    # 1. Load config and apply CLI overrides
    logger.info(f"Loading config from: {args.config_file}")
    cfg = load_config(args.config_file)
    cfg = apply_cli_overrides(cfg, args)

    training_cfg  = cfg["training"]
    dataset_cfg   = cfg["dataset"]
    model_cfg     = cfg["model"]
    metric_names  = cfg.get("metrics", ["AUROC"])
    metric_kwargs = cfg.get("metric_kwargs", [])

    # 2. Reproducibility
    set_seed(training_cfg.get("SEED", 42))

    # 3. Load datasets
    dataset_name   = dataset_cfg["name"]
    dataset_kwargs = dataset_cfg.get("kwargs", {})

    eval_splits   = dataset_cfg.get("eval_splits", ["val"])
    logger.info(f"Loading train and {eval_splits} split of dataset: {dataset_name}")
    train_dataset, eval_datasets = load_dataset(dataset_name, splits=eval_splits, **dataset_kwargs)

    labels = np.array(train_dataset.targets)
    if len(labels.shape) == 1:
        num_tasks = len(np.unique(labels)) 
    else: 
        num_tasks = labels.shape[-1]
    
    logger.info(f"Number of tasks: {num_tasks}")

    if num_tasks >= 2:
        multilable = True
    else:
        multilable = False

    # 5. Build metric function
    metric_fn = build_metric(metric_names, metric_kwargs)

    optimizer_kwargs=training_cfg.get("optimizer_kwargs", {})
    loss_kwargs=training_cfg.get("loss_kwargs", {})
    if multilable:
        loss_kwargs["num_labels"] = num_tasks

    # 6. Construct TrainingArguments
    #    trainset/evalsets store human-readable identifiers; the actual
    #    Dataset objects are passed separately to Trainer below.
    train_args = TrainingArguments(
        optimizer=training_cfg["optimizer"],
        optimizer_kwargs=optimizer_kwargs,
        loss=training_cfg["loss"],
        loss_kwargs=loss_kwargs,
        SEED=training_cfg.get("SEED", 42),
        batch_size=training_cfg.get("batch_size", 128),
        eval_batch_size=training_cfg.get("eval_batch_size", 128),
        sampling_rate=training_cfg.get("sampling_rate", 0.5),
        epochs=training_cfg.get("epochs", 50),
        decay_epochs=training_cfg.get("decay_epochs", []),
        num_workers=training_cfg.get("num_workers", 2),
        output_path=training_cfg.get("output_path", "./output"),
        resume_from_checkpoint=training_cfg.get("resume_from_checkpoint", True),
        save_checkpoint_every=training_cfg.get("save_checkpoint_every", 5),
        project_name=training_cfg.get("project_name", "libauc"),
        experiment_name=training_cfg["experiment_name"]
    )

    # 8. Initialise and run Trainer
    logger.info("Initialising Trainer...")
    trainer = Trainer(
        train_args=train_args,
        model_cfg=model_cfg,
        train_dataset=train_dataset,
        eval_dataset=eval_datasets if eval_datasets else None,
        metric=metric_fn,
        callbacks = [CLICallback()]
    )

    logger.info("Starting training...")
    train_log = trainer.train()
    logger.info("Training complete.")

if __name__ == "__main__":
    main()