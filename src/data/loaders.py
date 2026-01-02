"""
Data loading utilities for AutoAUC.
"""

import importlib
from pathlib import Path
from libauc.utils import ImbalancedDataGenerator
from .datasets import ImageDataset, UniformImageDataset


def get_dataset(name):
    """
    Get a dataset by name.
    
    Args:
        name: Name of the dataset (e.g., 'CIFAR10')
        
    Returns:
        Tuple of (train_dataset, valid_dataset)
    """
    if name == "CIFAR10":
        from libauc.datasets import CIFAR10
        mod = importlib.import_module("libauc.datasets")
        cls = getattr(mod, "CIFAR10", None)
        trainset = cls(root='./data', train=True)
        validset = cls(root='./data', train=False)
        return trainset, validset
    else:
        raise ValueError(f"Unsupported dataset: {name}")


def get_dataset_full(name, train_imratio=0.1, valid_imratio=0.5):
    """
    Get a full dataset with imbalanced data generation.
    
    Args:
        name: Name of the dataset (e.g., 'CIFAR10')
        train_imratio: Imbalance ratio for training data (default: 0.1)
        valid_imratio: Imbalance ratio for validation data (default: 0.5)
        
    Returns:
        Tuple of (train_dataset, eval_datasets_dict)
    """
    if name == "CIFAR10":
        from libauc.datasets import CIFAR10
        
        # Load raw data
        train_data, train_targets = CIFAR10(root='./data', train=True).as_array()
        test_data, test_targets = CIFAR10(root='./data', train=False).as_array()

        # Generate imbalanced data
        generator = ImbalancedDataGenerator(verbose=True, random_seed=0)
        (train_images, train_labels) = generator.transform(
            train_data, train_targets, imratio=train_imratio
        )
        (test_images, test_labels) = generator.transform(
            test_data, test_targets, imratio=valid_imratio
        )

        # Create datasets
        trainSet = ImageDataset(train_images, train_labels)
        trainSet_eval = ImageDataset(train_images, train_labels, mode='test')
        testSet = ImageDataset(test_images, test_labels, mode='test')

        return trainSet, {"train": trainSet_eval, "test": testSet}
    else:
        raise ValueError(f"Unsupported dataset: {name}")


def get_uniform_dataset(annotation_file, root_dir=None, image_size=32, crop_size=30, mode='train'):
    """
    Load a uniform dataset from JSON annotation file.
    
    Args:
        annotation_file: Path to the JSON annotation file
        root_dir: Root directory for resolving relative image paths (optional)
        image_size: Target image size (default: 32)
        crop_size: Crop size for training (default: 30)
        mode: Dataset mode - 'train' or 'test' (default: 'train')
        
    Returns:
        UniformImageDataset instance
    """
    return UniformImageDataset(
        annotation_file=annotation_file,
        image_size=image_size,
        crop_size=crop_size,
        mode=mode
    )


def get_uniform_dataset_full(annotation_file_train, annotation_file_test=None, image_size=32, crop_size=30):
    """
    Get uniform datasets for both training and evaluation.
    
    Args:
        annotation_file_train: Path to training set annotation file
        annotation_file_test: Path to test set annotation file (optional)
        root_dir: Root directory for resolving relative image paths (optional)
        image_size: Target image size (default: 32)
        crop_size: Crop size for training (default: 30)
        
    Returns:
        Tuple of (train_dataset, eval_datasets_dict)
        If annotation_file_test is None, eval_datasets_dict only contains 'train'
    """
    train_dataset = UniformImageDataset(
        annotation_file=annotation_file_train,
        image_size=image_size,
        crop_size=crop_size,
        mode='train'
    )
    
    train_dataset_eval = UniformImageDataset(
        annotation_file=annotation_file_train,
        image_size=image_size,
        crop_size=crop_size,
        mode='test'
    )
    
    eval_datasets = {"train": train_dataset_eval}
    
    if annotation_file_test:
        test_dataset = UniformImageDataset(
            annotation_file=annotation_file_test,
            image_size=image_size,
            crop_size=crop_size,
            mode='test'
        )
        eval_datasets["test"] = test_dataset
    
    return train_dataset, eval_datasets


# uniform dataset format
# more datasets (medical, molecule, etc.)
# more models
# AUC visualization
# multi-label AUC visualization 
# SSH compatibility
# libauc load state dict
# libauc tensor initialization (device)