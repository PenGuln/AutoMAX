import json
from typing import Any

import logging
logger = logging.getLogger(__name__)

# Optimizer mappings
_OPTIMIZERS = {
    "PESG": ("libauc.optimizers", "PESG"),
    "PDSCA": ("libauc.optimizers", "PDSCA"),
    "SOAP": ("libauc.optimizers", "SOAP"),
    "SOPA": ("libauc.optimizers", "SOPA"),
    "SOPAs": ("libauc.optimizers", "SOPAs"),
    "SOTAs": ("libauc.optimizers", "SOTAs"),
    "SONG": ("libauc.optimizers", "SONG"),
    "SGD": ("libauc.optimizers", "SGD"),
    "Adam": ("libauc.optimizers", "Adam")
}

# Loss function mappings
_LOSSES = {
    "AUCMLoss": ("libauc.losses", "AUCMLoss"),
    "CompositionalAUCLoss": ("libauc.losses", "CompositionalAUCLoss"),
    "APLoss": ("libauc.losses", "APLoss"),
    "pAUC_CVaR_Loss": ("libauc.losses", "pAUC_CVaR_Loss"),
    "pAUC_DRO_Loss": ("libauc.losses", "pAUC_DRO_Loss"),
    "tpAUC_KL_Loss": ("libauc.losses", "tpAUC_KL_Loss"),
    "NDCGLoss": ("libauc.losses", "NDCGLoss"),
    "CrossEntropyLoss": ("libauc.losses", "CrossEntropyLoss")
}

class TrainingArguments:
    """Training configuration arguments."""
    
    def __init__(self, **kwargs):
        # Training args
        self.optimizer = kwargs.pop("optimizer")
        self.optimizer_kwargs = kwargs.pop("optimizer_kwargs")
        self.loss = kwargs.pop("loss")
        self.loss_kwargs = kwargs.pop("loss_kwargs")
        self.SEED = kwargs.pop("SEED")
        self.batch_size = kwargs.pop("batch_size")
        self.eval_batch_size = kwargs.pop("eval_batch_size")
        self.sampling_rate = kwargs.pop("sampling_rate")
        self.epochs = kwargs.pop("epochs")
        self.decay_epochs = kwargs.pop("decay_epochs")
        for i in range(len(self.decay_epochs)):
            if isinstance(self.decay_epochs[i], float):
                self.decay_epochs[i] = int(self.decay_epochs[i] * self.epochs)
        self.num_workers = kwargs.pop("num_workers")
        self.output_path = kwargs.pop("output_path")
        
        # Checkpoint parameters
        self.resume_from_checkpoint = kwargs.pop("resume_from_checkpoint")
        self.save_checkpoint_every = kwargs.pop("save_checkpoint_every")

        # wandb config
        self.project_name = kwargs.pop("project_name")
        self.experiment_name = kwargs.pop("experiment_name")