"""
Dataset classes for AutoAUC.
"""

import json
import os
import numpy as np
from PIL import Image
import torchvision.transforms as transforms
from torch.utils.data import Dataset
from pathlib import Path


class ImageDataset(Dataset):
    """
    Custom dataset class for image data with train/test transforms.
    
    Args:
        images: Array of images
        targets: Array of target labels
        image_size: Target image size (default: 32)
        crop_size: Crop size for training (default: 30)
        mode: Dataset mode - 'train' or 'test' (default: 'train')
    """
    
    def __init__(self, images, targets, image_size=32, crop_size=30, mode='train'):
        self.images = images.astype(np.uint8)
        self.targets = targets
        self.mode = mode
        
        # Training transforms with data augmentation
        self.transform_train = transforms.Compose([
            transforms.ToTensor(),
            transforms.RandomCrop((crop_size, crop_size), padding=None),
            transforms.RandomHorizontalFlip(),
            transforms.Resize((image_size, image_size), antialias=True),
        ])
        
        # Test transforms without augmentation
        self.transform_test = transforms.Compose([
            transforms.ToTensor(),
            transforms.Resize((image_size, image_size), antialias=True),
        ])
    
    def __len__(self):
        """Return the number of samples in the dataset."""
        return len(self.images)

    def __getitem__(self, idx):
        """
        Get a sample from the dataset.
        
        Args:
            idx: Index of the sample
            
        Returns:
            Tuple of (image, target, index)
        """
        image = self.images[idx]
        target = self.targets[idx]
        image = Image.fromarray(image.astype('uint8'))
        
        if self.mode == 'train':
            image = self.transform_train(image)
        else:
            image = self.transform_test(image)
            
        return image, target, idx


class UniformImageDataset(Dataset):
    """
    Uniform dataset class that loads images from disk based on JSON annotations.
    
    The dataset structure should be:
    - A JSON annotation file containing a list of dictionaries, each with:
      - "image_path": relative or absolute path to the image file
      - "label": integer label for the image
    - Images stored at the paths specified in the JSON file
    
    Args:
        annotation_file: Path to the JSON annotation file
        root_dir: Root directory for resolving relative image paths (optional)
        image_size: Target image size (default: 32)
        crop_size: Crop size for training (default: 30)
        mode: Dataset mode - 'train' or 'test' (default: 'train')
    """
    
    def __init__(self, annotation_file, image_size=32, crop_size=30, mode='train'):
        self.annotation_file = annotation_file
        self.root_dir = Path(annotation_file).parent
        self.mode = mode
        
        # Load annotations
        with open(annotation_file, 'r') as f:
            self.annotations = json.load(f)
        
        # Validate annotations
        if not isinstance(self.annotations, list):
            raise ValueError("Annotation file must contain a JSON array")
        
        for i, ann in enumerate(self.annotations):
            if 'image_path' not in ann or 'label' not in ann:
                raise ValueError(f"Annotation at index {i} missing 'image_path' or 'label'")
        
        # Training transforms with data augmentation
        self.transform_train = transforms.Compose([
            transforms.ToTensor(),
            transforms.RandomCrop((crop_size, crop_size), padding=None),
            transforms.RandomHorizontalFlip(),
            transforms.Resize((image_size, image_size), antialias=True),
        ])
        
        # Test transforms without augmentation
        self.transform_test = transforms.Compose([
            transforms.ToTensor(),
            transforms.Resize((image_size, image_size), antialias=True),
        ])
    
    def __len__(self):
        """Return the number of samples in the dataset."""
        return len(self.annotations)
    
    def __getitem__(self, idx):
        """
        Get a sample from the dataset.
        
        Args:
            idx: Index of the sample
            
        Returns:
            Tuple of (image, target, index)
        """
        ann = self.annotations[idx]
        image_path = ann['image_path']
        target = int(ann['label'])
        
        # Resolve image path
        if self.root_dir:
            # If root_dir is provided, resolve relative paths
            if os.path.isabs(image_path):
                full_path = Path(image_path)
            else:
                full_path = self.root_dir / image_path
        else:
            # Use image_path as-is (can be absolute or relative to current working directory)
            full_path = Path(image_path)
        
        # Load image
        if not full_path.exists():
            raise FileNotFoundError(f"Image not found: {full_path}")
        
        try:
            image = Image.open(full_path).convert('RGB')
        except Exception as e:
            raise RuntimeError(f"Error loading image {full_path}: {e}")
        
        # Apply transforms
        if self.mode == 'train':
            image = self.transform_train(image)
        else:
            image = self.transform_test(image)
        
        return image, target, idx
