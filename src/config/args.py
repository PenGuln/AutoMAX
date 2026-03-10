"""
Configuration and argument parsing for AutoAUC.
"""

from audioop import mul
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
                logger.warning("Both categorical and logarithmic are set to True, ignore the logarithmic.")
                    
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
                        log=logarithmic,
                        meta=None,
                    )
                else:
                    ret[name] = UniformIntegerHyperparameter(
                        name=name,
                        lower=val[0],
                        upper=val[1],
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


def parse_defaultconfig(type_name, multilable = False, kwargs = {}):
    """
    Parse default configuration for a given type.
    
    Args:
        type_name: Type of configuration to parse
        
    Returns:
        Dictionary with optimizer and loss configurations
    """
    if type_name in ['AUCMLoss', 'PESG']:
        if multilable:
            from .spaces import MultiLabelAUCMLossSpace as Sp
        else:
            from .spaces import AUCMLossSpace as Sp
    elif type_name in ['CompositionalAUCLoss', 'PDSCA']:
            from .spaces import CompositionalAUCLossSpace as Sp
    elif type_name in ['APLoss', 'SOAP']:
        if multilable:
            from .spaces import mAPLossSpace as Sp
        else:
            from .spaces import APLossSpace as Sp
    elif type_name in ['pAUC_CVaR_Loss', 'SOPA'] or (type_name == 'pAUCLoss' and kwargs.get("mode", None) == 'SOPA'):
        if multilable:
            from .spaces import MultiLabelpAUC_CVaR_LossSpace as Sp
        else:
            from .spaces import pAUC_CVaR_LossSpace as Sp
    elif type_name in ['pAUC_DRO_Loss', 'SOPAs'] or (type_name == 'pAUCLoss' and kwargs.get("mode", None) == '1w'):
        if multilable:
            from .spaces import MultiLabelpAUC_DRO_LossSpace as Sp
        else:
            from .spaces import pAUC_DRO_LossSpace as Sp
    elif type_name in ['tpAUC_KL_Loss', 'SOTAs'] or (type_name == 'pAUCLoss' and kwargs.get("mode", None) == '2w'):
        if multilable:
            from .spaces import MultiLabeltpAUC_KL_LossSpace as Sp
        else:
            from .spaces import tpAUC_KL_LossSpace as Sp
    elif type_name in ['tpAUC_CVaR_loss', 'STACO']:
        from .spaces import tpAUC_CVaR_lossSpace as Sp
    elif type_name in ['NDCGLoss', 'SONG']:
        from .spaces import NDCGLossSpace as Sp
    elif type_name in ['CrossEntropyLoss', 'SGD']:
        from .spaces import SGDSpace as Sp
    elif type_name in ['Adam']:
        from .spaces import AdamSpace as Sp
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
            elif hasattr(arg, "autofunc"):
                cs.add_configuration_space(prefix=f"{i}", configuration_space=arg.cs, delimiter=":")
            else:
                raise ValueError("Autopartial must have Hyperparameter or autopartial arguments.")
        
        for k, v in kwds.items():
            if isinstance(v, Hyperparameter):
                cs.add_hyperparameter(v)
            elif hasattr(v, "autofunc"):
                cs.add_configuration_space(prefix=f"{k}", configuration_space=v.cs, delimiter=":")
            else:
                raise ValueError("Autopartial must have Hyperparameter or autopartial keywords.")

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

    def __call__(self, space, prefix = ""):
        new_args = []
        new_kwds = {}
        for i, arg in enumerate(self.args):
            if isinstance(arg, Hyperparameter):
                new_args.append(space[prefix + arg.name])
            else:
                new_args.append(self.args[i](space, prefix + f"{i}" + ":"))
        for k, v in self.kwds.items():
            if isinstance(v, Hyperparameter):
                new_kwds[k] = space[prefix + v.name]
            else:
                new_kwds[k] = self.kwds[k](space, prefix + f"{k}" + ":")
        return self.autofunc(*new_args, **new_kwds)


class AutoMAXConfigration:
    """Configuration for AutoAUC optimization."""
    
    def __init__(self, **kwargs):
        self.deterministic = kwargs.pop("deterministic")
        self.n_trials = kwargs.pop("n_trials")
        self.n_configs = kwargs.pop("n_configs")
        self.SEED = kwargs.pop("SEED", 0)
        self.name = kwargs.pop("name")
        self.output_directory = kwargs.pop("output_directory", "./automax_output")
        self.overwrite = kwargs.pop("overwrite")