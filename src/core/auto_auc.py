"""
AutoAUC optimization framework.
"""

from typing import Dict, Any, Optional
from smac import HyperparameterOptimizationFacade, Scenario
from ConfigSpace import ConfigurationSpace

from ..config.args import AutoAUCConfigration, TrainingArguments


class AutoAUC:
    """
    Automated AUC optimization using SMAC3.
    
    This class handles hyperparameter optimization for AUC-based metrics
    using the SMAC3 framework.
    """
    
    def __init__(self, trainer, config: AutoAUCConfigration, data: dict, model, target: Optional[str] = None):
        """
        Initialize AutoAUC optimizer.
        
        Args:
            trainer: Trainer function/class for training models
            config: AutoAUC configuration
            data: Training data configuration
            model: Model to optimize
            target: Target metric to optimize (optional)
        """
        # Setup configuration space
        cs = ConfigurationSpace(seed=0)
        cs.add_configuration_space(prefix="", configuration_space=data["optimizer_kwargs"].cs, delimiter="")
        cs.add_configuration_space(prefix="", configuration_space=data["loss_kwargs"].cs, delimiter="")

        self.data = data  
        self.configspace = cs        
        self.trainer = trainer
        self.target = target
        
        # Setup SMAC scenario
        scenario = Scenario(
            self.configspace,
            n_trials=config.n_trials,
            deterministic=config.deterministic,
            seed=config.seed
        )
        
        # Store initial model state for resetting between trials
        self.init_model_dict = model.state_dict()
        self.model = model

        # Setup initial design
        initial_design = HyperparameterOptimizationFacade.get_initial_design(scenario, n_configs=config.n_configs)
        
        # Initialize SMAC
        self.smac = HyperparameterOptimizationFacade(
            scenario,
            self.train,
            initial_design=initial_design,
            overwrite=True,
        )

    def train(self, space, seed: int = 0) -> float:
        """
        Train a model with given hyperparameters.
        
        Args:
            space: Hyperparameter configuration
            seed: Random seed for reproducibility
            
        Returns:
            Negative score (for minimization)
        """
        print(f"Training with configuration: {space}")
        
        # Reset model to initial state
        self.model.load_state_dict(self.init_model_dict)
        
        # Update configuration with new hyperparameters
        d = {
            "optimizer_kwargs": self.data["optimizer_kwargs"](space=space),
            "loss_kwargs": self.data["loss_kwargs"](space=space)
        }
        newd = {**self.data, **d}
        args = TrainingArguments(**newd)
        
        # Train model
        trainer = self.trainer(train_args=args)
        train_log = trainer.train()
        
        # Validate target metric exists
        if self.target and self.target not in train_log[0].keys():
            raise ValueError(f"Target {self.target} is not in train_log")

        # Calculate score
        if self.target is None:
            # Use negative loss for minimization
            score = -min([item['loss'] for item in train_log])
        else:
            # Use target metric (maximize)
            score = max([item[self.target] for item in train_log])

        print(f"Score: {score}")
        return -score  # Return negative for minimization

    def optimize(self):
        """
        Run the optimization process.
        
        Returns:
            Best configuration found
        """
        incumbent = self.smac.optimize()
        print(f"Best configuration found: {incumbent}")
        return incumbent
