"""
AutoMAX optimization framework.
"""
import os
from typing import Dict, Any, Optional
from smac import HyperparameterOptimizationFacade, Scenario
from ConfigSpace import ConfigurationSpace
from ..config.args import AutoMAXConfigration
import shutil
from smac.runhistory import TrialInfo, TrialValue


class AutoMAX:
    """
    Automated AUC optimization using SMAC3.
    
    This class handles hyperparameter optimization for AUC-based metrics
    using the SMAC3 framework.
    """
    
    def __init__(self, trainer, config: AutoMAXConfigration, target: Optional[str]):
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
        self.configspace = trainer.cs        
        self.trainer = trainer
        self.target = target

        # Setup SMAC scenario
        scenario = Scenario(
            self.configspace,
            name=config.name,
            output_directory=config.output_directory,
            n_trials=config.n_trials,
            deterministic=config.deterministic,
            seed=config.SEED
        )
        
        # Initialize SMAC
        self.smac = HyperparameterOptimizationFacade(
            scenario,
            self.train,
            overwrite=False,
        )
        # print(self.smac.runhistory.finished)

        self.best_score = float('-inf')

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
        trainer = self.trainer(space=space)
        train_log = trainer.train()
        print(train_log)
        score = max([v['metrics'][-1][self.target] for v in train_log])
        print(f"Score: {score}")

        if score > self.best_score:
            self.best_score = score
            best_dir = os.path.join(trainer.args.output_path, trainer.args.experiment_name + '_best')
            if os.path.exists(best_dir):
                shutil.rmtree(best_dir)
            os.rename(os.path.join(trainer.args.output_path, trainer.args.experiment_name), best_dir)

        return -score  # Return negative for minimization

    def optimize(self):
        """
        Run the optimization process.
        
        Returns:
            Best configuration found
        """

        for trial_info in self.smac.runhistory.get_running_trials():
            # Re-run the trial
            cost = self.train(trial_info.config, seed=trial_info.seed)
            trial_value = TrialValue(cost=cost, time=0.0)
            self.smac.tell(trial_info, trial_value)
            
        incumbent = self.smac.optimize()
        print(f"Best configuration found: {incumbent}")
        return incumbent
