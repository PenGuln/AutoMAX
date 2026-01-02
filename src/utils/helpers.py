"""
Helper functions for AutoAUC.
"""

import torch
import logging
from functools import partial

from ..config.args import AutoAUCConfigration
from ..core.trainer import Trainer
from ..core.auto_auc import AutoAUC
from ..models.factory import create_model
from libauc.metrics import auc_roc_score, auc_prc_score, pauc_roc_score

logger = logging.getLogger(__name__)


def create_auto_auc_components(train_args, n_trials, n_configs, model_name=None, 
                              pretrained=False, model_path=None, train_dataset=None, 
                              eval_dataset=None, callback_class=None, app=None, session_id=None):
    """
    Helper function to create AutoAUC components.
    
    Args:
        train_args (dict): Training arguments
        n_trials (int): Number of trials
        n_configs (int): Number of initial configurations
        model_name (str, optional): Model name if creating new model
        pretrained (bool, optional): Whether to use pretrained model
        model_path (str, optional): Path to pretrained model weights
        train_dataset (Dataset, optional): Training dataset
        eval_dataset (Dataset, optional): Evaluation dataset
        callback_class (class, optional): Callback class to use (GuiCallback or CLICallback)
        app (object, optional): App instance for GuiCallback
    
    Returns:
        tuple: (metric, auto_config, model, trainer, tuner)
    """
    
    # Create metric function based on target
    if train_args["target"] == "AUROC":
        metric = lambda test_true, test_pred: {
            "AUROC": auc_roc_score(test_true, test_pred)
        }
    elif train_args["target"] == "AUPRC":
        metric = lambda test_true, test_pred: {
            "AUPRC": auc_prc_score(test_true, test_pred)
        }
    elif train_args["target"] == 'OPAUC':
        metric = lambda test_true, test_pred: {
            "OPAUC": auc_roc_score(test_true, test_pred, max_fpr=float(train_args.get('max_fpr', 0.3)))
        }
    elif train_args["target"] == 'TPAUC':
        metric = lambda test_true, test_pred: {
            "TPAUC": pauc_roc_score(test_true, test_pred, 
                                   max_fpr=float(train_args.get('max_fpr', 0.3)), 
                                   min_tpr=float(train_args.get('min_tpr', 0.7)))
        }
    else:
        raise ValueError(f"Unsupported target metric: {train_args['target']}")
    
    # Create AutoAUC configuration
    auto_config = AutoAUCConfigration(deterministic=True, n_trials=n_trials, n_configs=n_configs)
    
    # Initialize model
    if model_name:
        model = create_model(model_name, pretrained, model_path)
    else:
        model = None
    
    # Print training arguments
    print("\n" + "="*50)
    print("TRAINING CONFIGURATION".center(50))
    print("="*50)
    
    print("\n[Optimizer Configuration]")
    print(f"Type: {train_args['optimizer']}")
    print("Parameters:")
    for param, value in train_args['optimizer_kwargs'].cs.items():
        print(f"  - {value}")
    
    print("\n[Loss Function Configuration]")
    print(f"Type: {train_args['loss']}")
    print("Parameters:")
    for param, value in train_args['loss_kwargs'].cs.items():
        print(f"  - {value}")
    
    print("\n[Training Parameters]")
    for key, value in train_args.items():
        if key not in ['optimizer', 'optimizer_kwargs', 'loss', 'loss_kwargs']:
            print(f"  - {key}: {value}")

    print("\n[AutoAUC Configuration]")
    for key, value in auto_config.__dict__.items():
        print(f"  - {key}: {value}")
    
    print("\n" + "="*50 + "\n")

    # Create trainer
    if train_dataset and eval_dataset:
        # Special handling for different callback types
        if callback_class is not None:
            if callback_class.__name__ == 'GuiCallback' and app is not None:
                callbacks = [callback_class(app, auto_config)]
            elif callback_class.__name__ == 'SessionCallback' and app is not None and session_id is not None:
                # For SessionCallback, app is the session_manager and session_id is provided
                callbacks = [callback_class(app, session_id, auto_config)]
            else:
                callbacks = [callback_class(auto_config)]
        else:
            callbacks = None
            
        trainer = partial(Trainer,
            model=model,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            metric=metric,
            callbacks=callbacks
        )
    else:
        trainer = None
    
    # Create tuner
    if trainer and model:
        tuner = AutoAUC(trainer, auto_config, train_args, model, target='test_' + train_args["target"])
    else:
        tuner = None
    
    return metric, auto_config, model, trainer, tuner
