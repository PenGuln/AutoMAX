from typing import List
from torch.utils.data import Dataset
import torchvision.transforms as transforms
import numpy as np
from PIL import Image
from libauc.utils import ImbalancedDataGenerator

# ---------------------------------------------------------------------------
# Dataset loading
# ---------------------------------------------------------------------------
class IndexedDataset(Dataset):
    def __init__(self, dataset):
        self.dataset = dataset
        self.targets = self._load_targets()
    
    def _load_targets(self):
        targets = [self.dataset[i][1] for i in range(len(self.dataset))]
        return np.array(targets)

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        image, target = self.dataset[idx]
        return image, target, idx

class ImageDataset(Dataset):
    def __init__(self, images, targets, image_size=32, crop_size=30, mode='train'):
        self.images = images.astype(np.uint8)
        self.targets = targets
        self.mode = mode
        self.transform_train = transforms.Compose([
                                transforms.ToTensor(),
                                transforms.RandomCrop((crop_size, crop_size), padding=None),
                                transforms.RandomHorizontalFlip(),
                                transforms.Resize((image_size, image_size), antialias=True),
                                ])
        self.transform_test = transforms.Compose([
                                transforms.ToTensor(),
                                transforms.Resize((image_size, image_size), antialias=True),
                                ])
    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        image = self.images[idx]
        target = self.targets[idx]
        image = Image.fromarray(image.astype('uint8'))
        if self.mode == 'train':
            image = self.transform_train(image)
        else:
            image = self.transform_test(image)
        return image, target, idx

def load_dataset(name: str, splits: List[str], **kwargs) -> Dataset:
    """
    Load a dataset by name and split.

    Args:
        name:     Dataset identifier (e.g. "catvsdog", "chexpert").
        splits:    Evaluation splits.
        **kwargs: Extra dataset-specific keyword arguments from the config.

    Returns:
        A torch.utils.data.Dataset whose __getitem__ yields
        (data, label, index) tuples, as expected by the Trainer.

    TODO: Implement each dataset branch below.
    """
    name = name.lower()

    if name == "catvsdog":
        raise NotImplementedError(f"Dataset '{name}' is not yet implemented.")

    elif name == "chexpert":
        raise NotImplementedError(f"Dataset '{name}' is not yet implemented.")

    elif name == "cifar10":
        from libauc.datasets import CIFAR10
        # load data as numpy arrays
        train_data, train_targets = CIFAR10(root='./data', train=True).as_array()
        test_data, test_targets  = CIFAR10(root='./data', train=False).as_array()

        imratio = kwargs.get("imratio", 0.1)
        # generate imbalanced data
        generator = ImbalancedDataGenerator(verbose=True, random_seed=2023)
        (train_images, train_labels) = generator.transform(train_data, train_targets, imratio=imratio)
        (test_images, test_labels) = generator.transform(test_data, test_targets, imratio=0.5)

        train_dataset = ImageDataset(train_images, train_labels)
        eval_datasets = []
        for split in splits:
            if split == 'val':
                eval_datasets.append(ImageDataset(train_images, train_labels, mode='test'))
            elif split == 'test':
                eval_datasets.append(ImageDataset(test_images, test_labels, mode='test'))
            else:
                raise NotImplementedError(f"Split '{split}' is not yet implemented for dataset '{name}'.")
        return train_dataset, eval_datasets
    elif name == "pneumoniamnist":
        from medmnist import PneumoniaMNIST
        train_transform = transforms.Compose([
            transforms.RandomHorizontalFlip(),
            transforms.RandomRotation(10),
            transforms.ToTensor(),
            transforms.Normalize(mean=[.5], std=[.5])
        ])

        # No augmentation for val/test
        test_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[.5], std=[.5])
        ])
        train_dataset = IndexedDataset(PneumoniaMNIST(split='train', transform=train_transform, download=True))
        eval_datasets = []
        for split in splits:
            if split == 'val':
                eval_datasets.append(IndexedDataset(PneumoniaMNIST(split='val',   transform=test_transform,  download=True)))
            elif split == 'test':
                eval_datasets.append(IndexedDataset(PneumoniaMNIST(split='test',  transform=test_transform,  download=True)))
            else:
                raise NotImplementedError(f"Split '{split}' is not yet implemented for dataset '{name}'.")
        return train_dataset, eval_datasets
    else:
        raise ValueError(
            f"Unknown dataset: '{name}'. "
            "Please add a branch for it inside load_dataset()."
        )