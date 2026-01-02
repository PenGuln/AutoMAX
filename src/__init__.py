"""
AutoAUC - Automated AUC Optimization Framework

A comprehensive framework for automated hyperparameter optimization
focused on AUC-based metrics for machine learning models.
"""

__version__ = "1.0.0"
__author__ = "AutoAUC Team"

# Core imports
from .core.trainer import Trainer
from .core.auto_auc import AutoAUC
from .core.callbacks import TrainerCallback, CallbackHandler, TrainerState

# Model imports
from .models.factory import create_model

# Data imports
from .data.datasets import ImageDataset
from .data.loaders import get_dataset, get_dataset_full

# Config imports
from .config.args import TrainingArguments, AutoAUCConfigration

# UI imports
from .ui.gui import AutoAUCApp

# Interface imports
from .ui.cli import main as cli_main
from .ui.gui_main import main as gui_main

__all__ = [
    # Core
    "Trainer",
    "AutoAUC", 
    "TrainerCallback",
    "CallbackHandler", 
    "TrainerState",
    
    # Models
    "create_model",
    
    # Data
    "ImageDataset",
    "get_dataset",
    "get_dataset_full",
    
    # Config
    "TrainingArguments",
    "AutoAUCConfigration",
    
    # UI
    "AutoAUCApp",
    
    # Interfaces
    "cli_main",
    "gui_main",
]
