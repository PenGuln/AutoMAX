"""
Model factory functions for AutoAUC.
"""

import torch
import logging

logger = logging.getLogger(__name__)


def create_model(name, pretrained=False, model_path=None):
    """
    Create a model by name.
    
    Args:
        name: Name of the model architecture
        pretrained: Whether to use pretrained weights
        model_path: Path to pretrained model weights
        
    Returns:
        Model instance
    """
    model = get_model(name)
    
    if pretrained and model_path:
        state_dict = torch.load(model_path)
        filtered = {k: v for k, v in state_dict.items() if 'linear' not in k and 'fc' not in k}
        msg = model.load_state_dict(filtered, False)
        logger.info(msg)
        
        # Reset classifier layers
        if "linear" in state_dict:
            model.linear.reset_parameters()
        if "fc" in state_dict:
            model.fc.reset_parameters()
    
    return model


def get_model(name):
    """
    Get a model architecture by name.
    
    Args:
        name: Name of the model architecture
        
    Returns:
        Model instance
    """
    if name == "resnet20":
        from libauc.models import resnet20
        model = resnet20(pretrained=False, last_activation=None, num_classes=1)
        return model
    elif name == "resnet18":
        from libauc.models import resnet18
        model = resnet18(pretrained=False, last_activation=None, num_classes=1)
        return model
    else:
        raise ValueError(f"Unsupported model: {name}")
