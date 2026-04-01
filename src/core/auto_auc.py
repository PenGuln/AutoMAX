"""
AutoMAX optimization framework.
"""
import os
import sys
import logging
from typing import Dict, Any, Optional
from smac import HyperparameterOptimizationFacade, Scenario
from smac.initial_design import DefaultInitialDesign
from ..config.args import AutoMAXConfigration
import shutil
from smac.runhistory import TrialInfo, TrialValue
import pickle
from libauc.trainer import Trainer

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s - %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger(__name__)

class AutoMAX:
    """
    Automated AUC optimization using SMAC3.
    
    This class handles hyperparameter optimization for AUC-based metrics
    using the SMAC3 framework.
    """
    
    def __init__(self, trainer, config: AutoMAXConfigration, target: str):
        """
        Initialize AutoAUC optimizer.
        
        Args:
            trainer: Trainer function/class for training models
            config: AutoAUC configuration
            data: Training data configuration
            model: Model to optimize
            target: Target metric to optimize
        """
        # Setup configuration space
        self.configspace = trainer.cs        
        self.trainer = trainer
        self.target = target
        self.config = config
        self.log = []

        # Setup SMAC scenario
        scenario = Scenario(
            self.configspace,
            name=config.name,
            output_directory=config.output_directory,
            n_trials=config.n_trials,
            deterministic=config.deterministic,
            seed=config.SEED
        )

        initial_design = DefaultInitialDesign(scenario)
        
        # Initialize SMAC
        self.smac = HyperparameterOptimizationFacade(
            scenario,
            self.train,
            initial_design=initial_design,
            overwrite=config.overwrite,
        )

        self.finished = self.smac.runhistory.finished

        self.best_score = float('-inf')
        if not config.overwrite and self.finished > 0:
            filepath = os.path.join(self.config.output_directory, self.config.name, "state.pkl")
            if os.path.exists(filepath):
                with open(filepath, "rb") as f:
                    self.log = pickle.load(f)
                    
        assert len(self.log) == self.finished

    def train(self, space, seed: int = 0) -> float:
        """
        Train a model with given hyperparameters.
        
        Args:
            space: Hyperparameter configuration
            seed: Random seed for reproducibility
            
        Returns:
            Negative score (for minimization)
        """
        
        logger.info(f"Starting Trail {self.finished + 1} with configuration: {space}.")
        trainer: Trainer = self.trainer(space=space)
        trainer.train()

        trail_log = trainer.state.train_summary
        trail_log["space"] = space

        score = trail_log['val']
        
        if score > self.best_score:
            logger.info(f"Found new best configuration! Updating the best checkpoint.")
            self.best_score = score
            best_dir = os.path.join(trainer.args.output_path, trainer.args.experiment_name + '_best')
            if os.path.exists(best_dir):
                shutil.rmtree(best_dir)
            os.rename(os.path.join(trainer.args.output_path, trainer.args.experiment_name), best_dir)
        
        self.finished += 1
        self.log.append(trail_log)
        with open(os.path.join(self.config.output_directory, self.config.name, "state.pkl"), "wb") as f:
            pickle.dump(self.log, f)
        
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
        logger.info(f"Best configuration found: {incumbent}")

        # Print trial history as a table
        has_test = any("test" in entry for entry in self.log)
        header = f"{'Trial':<8} {'Val ' + self.target:<20}" + (f" {'Test ' + self.target:<20}" if has_test else "")
        separator = "-" * len(header)
        print("\nTrial History:")
        print(separator)
        print(header)
        print(separator)
        for i, entry in enumerate(self.log, 1):
            val = f"{entry.get('val', float('nan')):.6f}"
            row = f"{i:<8} {val:<20}"
            if has_test:
                test = f"{entry.get('test', float('nan')):.6f}"
                row += f" {test:<20}"
            print(row)
        print(separator)

        return incumbent