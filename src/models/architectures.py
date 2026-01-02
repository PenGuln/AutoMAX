"""
Model architecture utilities for AutoAUC.
"""

import importlib
from ..config.args import _OPTIMIZERS, _LOSSES


def get_optimizer(name):
    """
    Get an optimizer class by name.
    
    Args:
        name: Name of the optimizer
        
    Returns:
        Optimizer class
    """
    opt_name, opt_cls_name = _OPTIMIZERS[name]
    opt = importlib.import_module(opt_name)
    opt_cls = getattr(opt, opt_cls_name, None)
    return opt_cls


def get_loss(name):
    """
    Get a loss function class by name.
    
    Args:
        name: Name of the loss function
        
    Returns:
        Loss function class
    """
    loss_name, loss_cls_name = _LOSSES[name]
    loss = importlib.import_module(loss_name)
    loss_cls = getattr(loss, loss_cls_name, None)
    return loss_cls


def get_class(mod_name, cls_name):
    """
    Get a class from a module.
    
    Args:
        mod_name: Name of the module
        cls_name: Name of the class
        
    Returns:
        Class object
    """
    mod = importlib.import_module(mod_name)
    cls = getattr(mod, cls_name, None)
    return cls
