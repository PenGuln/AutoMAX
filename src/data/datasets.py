from typing import List
from torch.utils.data import Dataset
import torchvision.transforms as transforms
import numpy as np
from PIL import Image
from libauc.utils import ImbalancedDataGenerator
import pandas as pd
from ogb.graphproppred import PygGraphPropPredDataset
import torch

# ---------------------------------------------------------------------------
# Dataset loading
# ---------------------------------------------------------------------------
class IndexedDataset(Dataset):
    def __init__(self, dataset, class_id = None):
        self.dataset = dataset
        self.targets = self._load_targets()
        if len(self.targets.shape) == 2 and class_id is not None:
            self.targets = self.targets[:, class_id : class_id + 1]
    
    def _load_targets(self):
        targets = [self.dataset[i][1] for i in range(len(self.dataset))]
        return np.array(targets).astype(np.float32)

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        image, _ = self.dataset[idx]
        target = self.targets[idx]
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

class GraphDataset(PygGraphPropPredDataset):
   def __getitem__(self, idx):
      if isinstance(idx, (int,np.int64)):
            item = self.get(self.indices()[idx])
            item.idx = torch.LongTensor([idx])
            return item
      else:
            return self.index_select(idx)

class TextDataset(Dataset):
    def __init__(self, dataframe, text_col, label_col):
        self.len = len(dataframe)
        self.data = dataframe
        self.text_col = text_col
        self.targets = self.data[label_col].to_numpy().astype(np.float32)
        self.texts = self.data[text_col]

    def __getitem__(self, index):
        text_inputs = self.texts[index]
        targets = self.targets[index]
        return text_inputs, targets, index    
    

    def __len__(self):
        return self.len

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
    root_path = kwargs.get("root_path", "./data")

    if name == "catvsdog":
        raise NotImplementedError(f"Dataset '{name}' is not yet implemented.")

    elif name == "chexpert":
        from libauc.datasets import CheXpert
        root = os.path.join(root_path, "CheXpert-v1.0-small")
        train_dataset = IndexedDataset(CheXpert(csv_path=root+'train.csv', image_root_path=root, use_upsampling=False, use_frontal=True, image_size=224, mode='train', class_index=-1, verbose=False))
        eval_datasets = []
        for split in splits:
            if split == 'val':
                eval_datasets.append(IndexedDataset(CheXpert(csv_path=root+'valid.csv',  image_root_path=root, use_upsampling=False, use_frontal=True, image_size=224, mode='valid', class_index=-1, verbose=False)))
            else:
                raise NotImplementedError(f"Split '{split}' is not yet implemented for dataset '{name}'.")
        return train_dataset, eval_datasets
    elif name == "cifar10":
        from libauc.datasets import CIFAR10
        # load data as numpy arrays
        train_data, train_targets = CIFAR10(root=root_path, train=True).as_array()
        test_data, test_targets  = CIFAR10(root=root_path, train=False).as_array()

        imratio = kwargs.get("imratio", 0.1)
        # generate imbalanced data
        generator = ImbalancedDataGenerator(verbose=True, random_seed=0)
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
        test_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[.5], std=[.5])
        ])
        train_dataset = IndexedDataset(PneumoniaMNIST(split='train', transform=train_transform, download=True, root=root_path))
        eval_datasets = []
        for split in splits:
            if split == 'val':
                eval_datasets.append(IndexedDataset(PneumoniaMNIST(split='val',   transform=test_transform,  download=True, root=root_path)))
            elif split == 'test':
                eval_datasets.append(IndexedDataset(PneumoniaMNIST(split='test',  transform=test_transform,  download=True, root=root_path)))
            else:
                raise NotImplementedError(f"Split '{split}' is not yet implemented for dataset '{name}'.")
        return train_dataset, eval_datasets
    elif name == "breastmnist":
        from medmnist import BreastMNIST
        train_transform = transforms.Compose([
            transforms.ToTensor(),
        ])
        test_transform = transforms.Compose([
            transforms.ToTensor(),
        ])
        train_dataset = IndexedDataset(BreastMNIST(split='train', transform=train_transform, download=True, root=root_path))
        eval_datasets = []
        for split in splits:
            if split == 'val':
                eval_datasets.append(IndexedDataset(BreastMNIST(split='val', transform=test_transform, download=True, root=root_path)))
            elif split == 'test':
                eval_datasets.append(IndexedDataset(BreastMNIST(split='test', transform=test_transform, download=True, root=root_path)))
            else:
                raise NotImplementedError(f"Split '{split}' is not yet implemented for dataset '{name}'.")
        return train_dataset, eval_datasets
    elif name == "chestmnist":
        from medmnist import ChestMNIST
        train_transform = transforms.Compose([
            transforms.ToTensor(),
        ])
        test_transform = transforms.Compose([
            transforms.ToTensor(),
        ])
        task = kwargs.get("task", None)
        train_dataset = IndexedDataset(ChestMNIST(split='train', transform=train_transform, download=True, root=root_path), task)
        eval_datasets = []
        for split in splits:
            if split == 'val':
                eval_datasets.append(IndexedDataset(ChestMNIST(split='val', transform=test_transform, download=True, root=root_path), task))
            elif split == 'test':
                eval_datasets.append(IndexedDataset(ChestMNIST(split='test', transform=test_transform, download=True, root=root_path), task))
            else:
                raise NotImplementedError(f"Split '{split}' is not yet implemented for dataset '{name}'.")
        return train_dataset, eval_datasets
    
    elif name == "ogbg-molpcba":
        import os
        dataset = GraphDataset(name = 'ogbg-molpcba', root = root_path)
        labels = pd.read_csv(os.path.join(root_path, 'ogbg_molpcba/raw', 'graph-label.csv.gz'), compression='gzip', header = None).values

        #### get the official train_val_test split
        split_idx = dataset.get_idx_split()
        #### get training lable for task_0
        train_labels = labels[split_idx["train"]][:,0]

        #### remove samples which have 'nan' as their labels
        not_nan = ~np.isnan(train_labels)
        train_labels = train_labels[not_nan]
        train_dataset = dataset[split_idx["train"]][not_nan]
        eval_datasets = []
        for split in splits:
            if split == 'val':
                eval_datasets.append(dataset[split_idx["test"]][~np.isnan(labels[split_idx["test"]][:,0] )])
            else:
                raise NotImplementedError(f"Split '{split}' is not yet implemented for dataset '{name}'.")
        return train_dataset, eval_datasets

    elif name == "rip":
        train_df = pd.read_parquet("hf://datasets/ShantanuT01/RIP-Dataset/train.parquet")
        test_df = pd.read_parquet("hf://datasets/ShantanuT01/RIP-Dataset/test.parquet")
        train_dataset = TextDataset(train_df, "text", "target")
        eval_datasets = []
        for split in splits:
            if split == 'test':
                eval_datasets.append(TextDataset(test_df,"text", "target"))
            else:
                raise NotImplementedError(f"Split '{split}' is not yet implemented for dataset '{name}'.")
        
        return train_dataset, eval_datasets
    
    elif name == "raid":
        train_df = pd.read_parquet(os.path.join(root_path, "raid-training-stratified.parquet"))
        test_df = pd.read_parquet(os.path.join(root_path, "raid-training-stratified.parquet"))
        train_dataset = TextDataset(train_df, "generation", "target")
        eval_datasets = []
        for split in splits:
            if split == 'test':
                eval_datasets.append(TextDataset(test_df,"generation", "target"))
            else:
                raise NotImplementedError(f"Split '{split}' is not yet implemented for dataset '{name}'.")
        
        return train_dataset, eval_datasets

    else:
        raise ValueError(
            f"Unknown dataset: '{name}'. "
            "Please add a branch for it inside load_dataset()."
        )