"""
Command-line interface for AutoAUC.
"""

import argparse
import json
import logging
from pathlib import Path
from ..data.loaders import get_dataset_full
from ..config.args import _SETTINGS, _OPTIMIZERS, _LOSSES, autopartial, parse_hyperparameters_from_dict
from ..core.callbacks import CLICallback
from ..utils.helpers import create_auto_auc_components

logger = logging.getLogger(__name__)


def load_config_from_file_or_string(config_input):
    """
    Load configuration from a JSON file path or JSON string.
    
    Args:
        config_input: Path to JSON file or JSON string
        
    Returns:
        Parsed configuration dictionary
    """
    if config_input is None:
        return None
    
    # Try to load as file first
    config_path = Path(config_input)
    if config_path.exists() and config_path.is_file():
        with open(config_path, 'r') as f:
            return json.load(f)
    
    # Try to parse as JSON string
    try:
        return json.loads(config_input)
    except json.JSONDecodeError:
        raise ValueError(f"Invalid JSON format: {config_input}")


def merge_config(default_config, override_config):
    """
    Deep merge override_config into default_config.
    
    Args:
        default_config: Default configuration dictionary
        override_config: Configuration dictionary to override with
        
    Returns:
        Merged configuration dictionary
    """
    if override_config is None:
        return default_config
    
    merged = default_config.copy()
    
    # Merge optimizer config if present
    if "optimizer" in override_config:
        if "optimizer" not in merged:
            merged["optimizer"] = {}
        merged["optimizer"].update(override_config["optimizer"])
        # Deep merge space if present
        if "space" in override_config["optimizer"]:
            if "space" not in merged["optimizer"]:
                merged["optimizer"]["space"] = {}
            merged["optimizer"]["space"].update(override_config["optimizer"]["space"])
    
    # Merge loss config if present
    if "loss" in override_config:
        if "loss" not in merged:
            merged["loss"] = {}
        merged["loss"].update(override_config["loss"])
        # Deep merge space if present
        if "space" in override_config["loss"]:
            if "space" not in merged["loss"]:
                merged["loss"]["space"] = {}
            merged["loss"]["space"].update(override_config["loss"]["space"])
    
    return merged


def parse_args(args=None):
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='AutoAUC CLI')
    
    # Model and dataset arguments
    parser.add_argument('--model', type=str, choices=['resnet18', 'resnet20', 'resnet32'], 
                       default='resnet18', help='Model architecture')
    parser.add_argument('--dataset', type=str, choices=['CIFAR10', 'CIFAR100'], 
                       default='CIFAR10', help='Dataset to use')
    parser.add_argument('--pretrained', action='store_true', help='Use pretrained model')
    parser.add_argument('--model_path', type=str, help='Path to pretrained model weights')
    
    # Target metric arguments
    parser.add_argument('--target', type=str, choices=list(_SETTINGS.keys()), 
                       default='AUROC', help='Target metric to optimize')
    parser.add_argument('--max_fpr', type=float, default=0.3, help='Maximum FPR for OPAUC/TPAUC')
    parser.add_argument('--min_tpr', type=float, default=0.7, help='Minimum TPR for TPAUC')
    
    # Training arguments
    parser.add_argument('--seed', type=int, default=123, help='Random seed')
    parser.add_argument('--batch_size', type=int, default=128, help='Training batch size')
    parser.add_argument('--eval_batch_size', type=int, default=128, help='Evaluation batch size')
    parser.add_argument('--sampling_rate', type=float, default=0.2, help='Sampling rate')
    parser.add_argument('--epochs', type=int, default=50, help='Number of epochs')
    parser.add_argument('--decay_epochs', type=str, default='[0.5, 0.75]', 
                       help='Learning rate decay epochs as JSON list')
    parser.add_argument('--num_workers', type=int, default=0, help='Number of data loader workers')
    parser.add_argument('--output_path', type=str, default='./output', help='Output directory')
    
    # AutoAUC configuration
    parser.add_argument('--n_trials', type=int, default=5, help='Number of trials')
    parser.add_argument('--n_configs', type=int, default=1, help='Number of initial configurations')
    
    # Optimizer and loss function arguments
    parser.add_argument('--optimizer', type=str, choices=list(_OPTIMIZERS.keys()), 
                       default='PESG', help='Optimizer to use')
    parser.add_argument('--loss', type=str, choices=list(_LOSSES.keys()), 
                       default='AUCMLoss', help='Loss function to use')
    
    # Configuration override arguments
    parser.add_argument('--optimizer_config', type=str, default=None,
                       help='Path to JSON file or JSON string for optimizer config override')
    parser.add_argument('--loss_config', type=str, default=None,
                       help='Path to JSON file or JSON string for loss config override')
    
    return parser.parse_args(args)


def main(args=None, optimizer_config=None, loss_config=None):
    """
    Main CLI entry point.
    
    Args:
        args: Command-line arguments (parsed if None)
        optimizer_config: Optional optimizer config dict to override defaults
        loss_config: Optional loss config dict to override defaults
    """
    parsed_args = parse_args(args)
    
    # Setup logging
    logging.basicConfig(level=logging.INFO)
    
    # Get datasets
    train_dataset, eval_dataset = get_dataset_full(parsed_args.dataset)
    
    # Parse decay epochs
    decay_epochs = json.loads(parsed_args.decay_epochs)
    
    # Use the same approach as GUI - get default config and modify
    from ..config.args import parse_defaultconfig
    
    # Get default configuration for the selected optimizer and loss
    default_optimizer_config = parse_defaultconfig(parsed_args.optimizer)
    default_loss_config = parse_defaultconfig(parsed_args.loss)
    
    # Load override configs from command-line if provided
    cli_optimizer_config = None
    cli_loss_config = None
    
    if parsed_args.optimizer_config:
        cli_optimizer_config = load_config_from_file_or_string(parsed_args.optimizer_config)
    
    if parsed_args.loss_config:
        cli_loss_config = load_config_from_file_or_string(parsed_args.loss_config)
    
    # Merge configs: function parameters override CLI args, which override defaults
    if optimizer_config is None:
        optimizer_config = default_optimizer_config
        if cli_optimizer_config:
            optimizer_config = merge_config(default_optimizer_config, cli_optimizer_config)
    else:
        # If function parameter provided, merge with default
        optimizer_config = merge_config(default_optimizer_config, optimizer_config)
    
    if loss_config is None:
        loss_config = default_loss_config
        if cli_loss_config:
            loss_config = merge_config(default_loss_config, cli_loss_config)
    else:
        # If function parameter provided, merge with default
        loss_config = merge_config(default_loss_config, loss_config)
    
    # Prepare training arguments (same structure as GUI)
    train_args = {
        "optimizer": optimizer_config["optimizer"]["type"],
        "loss": loss_config["loss"]["type"],
        "optimizer_kwargs": autopartial(dict, **parse_hyperparameters_from_dict(optimizer_config["optimizer"]["space"])),
        "loss_kwargs": autopartial(dict, **parse_hyperparameters_from_dict(loss_config["loss"]["space"])),
        "SEED": parsed_args.seed,
        "batch_size": parsed_args.batch_size,
        "eval_batch_size": parsed_args.eval_batch_size,
        "sampling_rate": parsed_args.sampling_rate,
        "epochs": parsed_args.epochs,
        "decay_epochs": decay_epochs,
        "num_workers": parsed_args.num_workers,
        "output_path": parsed_args.output_path,
        "target": parsed_args.target,
    }
    
    # Add target-specific parameters
    if parsed_args.target in ['OPAUC', 'TPAUC']:
        train_args["max_fpr"] = parsed_args.max_fpr
    if parsed_args.target == 'TPAUC':
        train_args["min_tpr"] = parsed_args.min_tpr
    
    # Create components
    _, _, _, _, tuner = create_auto_auc_components(
        train_args=train_args,
        n_trials=parsed_args.n_trials,
        n_configs=parsed_args.n_configs,
        model_name=parsed_args.model,
        pretrained=parsed_args.pretrained,
        model_path=parsed_args.model_path,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        callback_class=CLICallback
    )
    
    # Start optimization
    print("Starting AutoAUC optimization...")
    tuner.optimize()
    print("Optimization completed!")


if __name__ == "__main__":
    main()
