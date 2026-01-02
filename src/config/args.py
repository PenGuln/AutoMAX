"""
Configuration and argument parsing for AutoAUC.
"""

import json
from typing import Any
from ConfigSpace import ConfigurationSpace
from ConfigSpace.hyperparameters import (
    CategoricalHyperparameter,
    Constant,
    Hyperparameter,
    UniformFloatHyperparameter,
    UniformIntegerHyperparameter,
)

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

# Settings for different target metrics
_SETTINGS = {
    "AUROC": ["AUCMLoss", "CompositionalAUCLoss", "SGD", "Adam"],
    "AUPRC": ["APLoss", "SGD", "Adam"],
    "OPAUC": ["pAUC_CVaR_Loss", "pAUC_DRO_Loss", "SGD", "Adam"],
    "TPAUC": ["tpAUC_KL_Loss", "SGD", "Adam"],
    "Rank": ["NDCGLoss"]
}


def parse_hyperparameters_from_dict(items: dict[str, Any]):
    """
    Parse hyperparameters from dictionary format.
    
    Args:
        items: Dictionary of hyperparameter definitions
        
    Returns:
        Dictionary of parsed hyperparameters
    """
    ret = {}
    for name, hp in items.items():
        # Additional support for AutoAUC
        if isinstance(hp, dict):
            val = hp.get("val")
            if val is None:
                raise ValueError("Can't find the key 'val' in {hp}")
            
            logarithmic = hp.get("log", False)
            if isinstance(val, set) and logarithmic:
                logger.warn("Both categorical and logarithmic are set to True, ignore the logarithmic.")
                    
            if isinstance(val, (int, str, float)):
                ret[name] = Constant(name, val)
            elif isinstance(val, tuple):
                if len(val) != 2: 
                    raise ValueError(f"'{name}' must be (lower, upper) bound, got {val}")
                default = hp.get("default", val[0])
                if isinstance(val[0], float) or isinstance(val[1], float):
                    ret[name] = UniformFloatHyperparameter(
                        name=name,
                        lower=val[0],
                        upper=val[1],
                        default_value=default,
                        q=None,
                        log=logarithmic,
                        meta=None,
                    )
                else:
                    ret[name] = UniformIntegerHyperparameter(
                        name=name,
                        lower=val[0],
                        upper=val[1],
                        q=None,
                        log=logarithmic,
                        default_value=default,
                        meta=None,
                    )
            elif isinstance(val, list):
                if len(val) == 0:
                    raise ValueError(f"Can't have empty list for categorical {name}")
                default = hp.get("default", val[0])
                ret[name] = CategoricalHyperparameter(
                        name=name,
                        choices=val,
                        default_value=default,
                        weights=None,
                        meta=None,
                    )
            else:
                raise ValueError(f"Unknown value '{val}' for '{name}'")
            
        # Anything that is a Hyperparameter already is good
        elif isinstance(hp, Hyperparameter):
            ret[name] = hp

        # Tuples are bounds, check if float or int
        elif isinstance(hp, tuple):
            if len(hp) != 2:
                raise ValueError(f"'{name}' must be (lower, upper) bound, got {hp}")

            lower, upper = hp
            if isinstance(lower, float):
                ret[name] = UniformFloatHyperparameter(name, lower, upper)
            else:
                ret[name] = UniformIntegerHyperparameter(name, lower, upper)

        # Lists are categoricals
        elif isinstance(hp, list):
            if len(hp) == 0:
                raise ValueError(f"Can't have empty list for categorical {name}")

            ret[name] = CategoricalHyperparameter(name, hp)

        # If it's an allowed type, it's a constant
        elif isinstance(hp, (int, str, float)):
            ret[name] = Constant(name, hp)
        else:
            raise ValueError(f"Unknown value '{hp}' for '{name}'")
    return ret


def parse_defaultconfig(type_name):
    """
    Parse default configuration for a given type.
    
    Args:
        type_name: Type of configuration to parse
        
    Returns:
        Dictionary with optimizer and loss configurations
    """
    if type_name in ['AUCMLoss', 'PESG']:
        from .spaces import AUCMLossSpace as Sp
    elif type_name in ['CompositionalAUCLoss', 'PDSCA']:
        from .spaces import CompositionalAUCLossSpace as Sp
    elif type_name in ['APLoss', 'SOAP']:
        from .spaces import APLossSpace as Sp
    elif type_name in ['pAUC_CVaR_Loss', 'SOPA']:
        from .spaces import pAUC_CVaR_LossSpace as Sp
    elif type_name in ['pAUC_DRO_Loss', 'SOPAs']:
        from .spaces import pAUC_DRO_LossSpace as Sp
    elif type_name in ['tpAUC_KL_Loss', 'SOTAs']:
        from .spaces import tpAUC_KL_LossSpace as Sp
    elif type_name in ['NDCGLoss', 'SONG']:
        from .spaces import NDCGLossSpace as Sp
    elif type_name in ['SGD']:
        from .spaces import CrossEntropyLossSpace1 as Sp
    elif type_name in ['Adam']:
        from .spaces import CrossEntropyLossSpace2 as Sp
    elif type_name in ['CrossEntropyLoss']:
        from .spaces import CrossEntropyLossSpace2 as Sp
    else:
        raise ValueError(f"unsupported loss {type_name}")
    
    return {"optimizer": Sp.optimizer, "loss": Sp.loss}


class autopartial:
    """
    Partial function with hyperparameter support for AutoAUC.
    """
    __slots__ = "autofunc", "args", "kwds", "cs", "__dict__", "__weakref__"

    def __new__(cls, autofunc, /, *args, **kwds):
        if not callable(autofunc):
            raise TypeError("the first argument must be callable")
        
        cs = ConfigurationSpace(seed=0)
        
        for i, arg in enumerate(args):
            if isinstance(arg, Hyperparameter):
                cs.add_hyperparameter(arg)
        
        for k, v in kwds.items():
            if isinstance(v, Hyperparameter):
                cs.add_hyperparameter(v)

        if hasattr(autofunc, "autofunc"):
            args = autofunc.args + args
            kwds = {**autofunc.kwds, **kwds}
            cs.add_configuration_space(prefix="", configuration_space=autofunc.cs, delimiter="")
            autofunc = autofunc.autofunc
        
        self = super(autopartial, cls).__new__(cls)

        self.autofunc = autofunc
        self.args = args
        self.kwds = kwds
        self.cs = cs
        return self

    def __call__(self, /, *args, **kwds):
        args = self.args + args
        kwds = {**self.kwds, **kwds}
        
        space = kwds.pop("space", None)

        # replace the Hyperparameter slot with true argument
        if space:
            for i, arg in enumerate(args):
                if isinstance(arg, Hyperparameter):
                    args[i] = space[arg.name]
            for k, v in kwds.items():
                if isinstance(v, Hyperparameter):
                    kwds[k] = space[v.name]
        else:
            raise ValueError("Calling Autopartial must have 'space' keyword")

        return self.autofunc(*args, **kwds)


class AutoAUCConfigration:
    """Configuration for AutoAUC optimization."""
    
    def __init__(self, **kwargs):
        self.deterministic = kwargs.pop("deterministic")
        self.n_trials = kwargs.pop("n_trials")
        self.n_configs = kwargs.pop("n_configs")
        self.seed = kwargs.pop("seed", 0)

    @classmethod
    def from_json(cls, json_file: str, **kwargs):
        """Load configuration from JSON file."""
        config_dict = cls.dict_from_json_file(json_file)
        config = cls(**config_dict)
        return config
    
    @classmethod
    def dict_from_json_file(cls, json_file: str):
        """Load dictionary from JSON file."""
        with open(json_file, "r", encoding="utf-8") as reader:
            text = reader.read()
        return json.loads(text)


class TrainingArguments:
    """Training configuration arguments."""
    
    def __init__(self, **kwargs):
        self.optimizer = kwargs.pop("optimizer")
        self.optimizer_kwargs = kwargs.pop("optimizer_kwargs")
        self.loss = kwargs.pop("loss")
        self.loss_kwargs = kwargs.pop("loss_kwargs")
        self.SEED = kwargs.pop("SEED", 42)
        self.batch_size = kwargs.pop("batch_size", 128)
        self.eval_batch_size = kwargs.pop("eval_batch_size", 128)
        self.sampling_rate = kwargs.pop("sampling_rate", 0.5)
        self.epochs = kwargs.pop("epochs", 50)
        self.decay_epochs = kwargs.pop("decay_epochs")
        for i in range(len(self.decay_epochs)):
            if isinstance(self.decay_epochs[i], float):
                self.decay_epochs[i] = int(self.decay_epochs[i] * self.epochs)
        self.num_workers = kwargs.pop("num_workers", 2)
        self.output_path = kwargs.pop("output_path", "./output")
        
        # Checkpoint parameters
        self.resume_from_checkpoint = kwargs.pop("resume_from_checkpoint", True)
        self.save_checkpoint_every = kwargs.pop("save_checkpoint_every", 5)  # Save every N epochs