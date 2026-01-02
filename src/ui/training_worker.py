#!/usr/bin/env python3
"""
Independent training worker that can run even when GUI is closed.
This script runs the actual training in a separate process.
"""

import os
import sys
import json
import logging
import argparse
from datetime import datetime
from pathlib import Path

# Add the project root to the path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from src.data.loaders import get_dataset_full
from src.utils.helpers import create_auto_auc_components
from src.config.args import parse_hyperparameters_from_dict, autopartial
import wandb

logger = logging.getLogger(__name__)

def convert_typed_spaces(space_dict):
    """Convert spaces with type indicators back to original format."""
    converted = {}
    for key, value in space_dict.items():
        if isinstance(value, dict) and 'type' in value and 'val' in value:
            # New format with type indicators
            param = {
                'default': value.get('default'),
                'log': value.get('log', False)
            }
            
            if value['type'] == 'interval':
                # Convert back to tuple for continuous ranges
                param['val'] = tuple(value['val'])
            elif value['type'] == 'categorical':
                # Keep as list for categorical
                param['val'] = value['val']
            else:
                # Single value or other types
                param['val'] = value['val']
            
            converted[key] = param
        else:
            # Old format without type indicators - keep as is
            converted[key] = value
    return converted

def run_training(config_file, session_id):
    """Run training with the given configuration."""
    try:
        # Load configuration
        with open(config_file, 'r') as f:
            config = json.load(f)
        
        logger.info(f"Starting training for session {session_id}")
        logger.info(f"Configuration: {config}")
        
        # Initialize wandb
        wandb.init(
            project="auto-auc",
            config=config,
            name=f"session_{session_id}"
        )
        
        # Initialize session manager
        from src.ui.session_manager import get_session_manager
        session_manager = get_session_manager()
        
        # Load datasets
        train_dataset, eval_dataset = get_dataset_full(config['dataset'])
        logger.info("Datasets loaded successfully")
        
        # Prepare training arguments
        # Convert spaces with type indicators back to original format
        optimizer_space = convert_typed_spaces(config.get('optimizer_space', {}))
        loss_space = convert_typed_spaces(config.get('loss_space', {}))
        
        args = {
            "optimizer": config['optimizer'],
            "loss": config['loss'],
            "optimizer_kwargs": autopartial(dict, **parse_hyperparameters_from_dict(optimizer_space)),
            "loss_kwargs": autopartial(dict, **parse_hyperparameters_from_dict(loss_space)),
            "SEED": config.get('seed', 42),
            "batch_size": config['batch_size'],
            "eval_batch_size": config.get('eval_batch_size', config['batch_size']),
            "sampling_rate": config.get('sampling_rate', 1.0),
            "epochs": config['epochs'],
            "decay_epochs": config.get('decay_epochs', []),
            "num_workers": config.get('num_workers', 4),
            "output_path": config['output_path'],
            "target": config['target']
        }
        
        # Add target-specific parameters
        if config['target'] == 'OPAUC' or config['target'] == 'TPAUC':
            args["max_fpr"] = config.get('max_fpr', 0.1)
        if config['target'] == 'TPAUC':
            args["min_tpr"] = config.get('min_tpr', 0.8)
        
        logger.info("Starting auto-auc optimization...")
        
        # Create and run the tuner with SessionCallback
        from src.core.callbacks import SessionCallback
        
        _, _, _, _, tuner = create_auto_auc_components(
            train_args=args,
            n_trials=config.get('n_trials', 10),
            n_configs=config.get('n_configs', 5),
            model_name=config['model'],
            pretrained=config.get('pretrained', False),
            model_path=config.get('model_path'),
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            callback_class=SessionCallback,
            app=session_manager,
            session_id=session_id
        )
        
        # Run optimization
        tuner.optimize()
        
        logger.info("Training completed successfully")
        
        # Update session status
        update_session_status(session_id, 'completed')
        
    except Exception as e:
        logger.error(f"Training failed: {str(e)}")
        update_session_status(session_id, 'failed', str(e))
        raise
    finally:
        wandb.finish()

def update_session_status(session_id, status, error_message=None):
    """Update session status in the session manager."""
    try:
        from src.ui.session_manager import get_session_manager
        
        session_manager = get_session_manager()
        
        if status == 'completed':
            session_manager.complete_session(session_id)
            session_manager.write_session_log(session_id, "Training completed successfully", "SUCCESS")
        elif status == 'failed':
            session_manager.complete_session(session_id)
            session_manager.write_session_log(session_id, f"Training failed: {error_message}", "ERROR")
        
        logger.info(f"Updated session {session_id} status to {status}")
        
    except Exception as e:
        logger.error(f"Failed to update session status: {str(e)}")

def main():
    parser = argparse.ArgumentParser(description='Training Worker')
    parser.add_argument('--config', required=True, help='Path to configuration file')
    parser.add_argument('--session-id', required=True, help='Session ID')
    
    args = parser.parse_args()
    
    # Log the process start
    logger.info(f"Training worker started for session {args.session_id}")
    logger.info(f"PID: {os.getpid()}")
    
    # Update session with PID
    try:
        from src.ui.session_manager import get_session_manager
        session_manager = get_session_manager()
        session_manager.update_session(args.session_id, pid=os.getpid())
        session_manager.write_session_log(args.session_id, f"Training process started (PID: {os.getpid()})", "INFO")
    except Exception as e:
        logger.error(f"Failed to update session with PID: {str(e)}")
    
    # Run training
    run_training(args.config, args.session_id)

if __name__ == "__main__":
    main()
