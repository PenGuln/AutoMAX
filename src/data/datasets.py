from typing import List
from torch.utils.data import Dataset
import torchvision.transforms as transforms
import numpy as np
from PIL import Image
from libauc.utils import ImbalancedDataGenerator
import pandas as pd
from ogb.graphproppred import PygGraphPropPredDataset
import torch
import os

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
        # Unpack task_id if TriSampler is being used
        task_id = None
        if isinstance(idx, (tuple, list)):
            idx, task_id = idx
            
        image, _ = self.dataset[idx]
        target = self.targets[idx]
        return image, target, (idx, task_id) if task_id is not None else idx

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

class ChemicalDataset(Dataset):
    def __init__(self, dataset, class_id):
        self.targets = []
        # dataset.indices() gives the actual indices into the full dataset
        indices = dataset.indices()
        assert(len(dataset.data.y.shape) == 2)
        y = dataset.data.y[indices, class_id]
        not_nan = ~np.isnan(y.numpy())  # shape matches len(dataset)
        self.targets = y[not_nan].float()
        self.dataset = dataset[not_nan]
        try:
            tmp=np.array(self.targets)
            pos = len(tmp[tmp==1])
            print('positive: ' + str(pos))
            print('positive rate: '+ str(float(pos)/len(tmp)))
        except:
            print('positive rate error ')

    def __len__(self):
        return len(self.targets)

    def __getitem__(self, idx):
        return self.dataset[idx], self.targets[idx], int(idx)


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

class MedicalImageCSVDataset(Dataset):
    """
    General-purpose CSV-backed medical image dataset.

    Expects a CSV with at least an image path column and a binary label column.
    Image paths in the CSV may be relative (resolved against ``image_root``) or
    absolute.

    Args:
        csv_path:   Path to the metadata CSV.
        image_root: Directory that image paths are resolved against when they
                    are not absolute.  Ignored for absolute paths.
        image_col:  Column name containing the image filename / path.
        label_col:  Column name containing the binary label (0 / 1).
        transform:  torchvision transform applied to each PIL image.
    """

    def __init__(
        self,
        csv_path: str,
        image_root: str,
        image_col: str,
        label_col: str,
        transform,
    ):
        df = pd.read_csv(csv_path)
        # Drop rows with missing labels
        df = df.dropna(subset=[label_col]).reset_index(drop=True)

        self.image_root = image_root
        self.image_col = image_col
        self.transform = transform
        self.targets = df[label_col].to_numpy().astype(np.float32)
        self.image_paths = df[image_col].tolist()

        pos = int((self.targets == 1).sum())
        total = len(self.targets)
        print(f"[MedicalImageCSVDataset] positive: {pos} | rate: {pos/total:.4f}")

    def __len__(self):
        return len(self.targets)

    def __getitem__(self, idx):
        rel = self.image_paths[idx]
        path = rel if os.path.isabs(rel) else os.path.join(self.image_root, rel)
        image = Image.open(path).convert("RGB")
        if self.transform is not None:
            image = self.transform(image)
        return image, self.targets[idx], idx


# ---------------------------------------------------------------------------
# Shared transform factories
# ---------------------------------------------------------------------------
def _medical_train_transform(image_size: int = 224) -> transforms.Compose:
    return transforms.Compose([
        transforms.Resize((image_size, image_size), antialias=True),
        transforms.RandomHorizontalFlip(),
        transforms.RandomVerticalFlip(),
        transforms.RandomRotation(10),
        transforms.ColorJitter(brightness=0.2, contrast=0.2),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225]),
    ])


def _medical_test_transform(image_size: int = 224) -> transforms.Compose:
    return transforms.Compose([
        transforms.Resize((image_size, image_size), antialias=True),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225]),
    ])


# ---------------------------------------------------------------------------
# OGB graph dataset helpers (shared by ogbg-molhiv and existing datasets)
# ---------------------------------------------------------------------------
def _safe_import_pyg_globals():
    """Register PyG safe globals for torch.serialization when available."""
    import torch.serialization
    try:
        from torch_geometric.data.data import DataEdgeAttr, DataTensorAttr
        from torch_geometric.data.storage import GlobalStorage
        torch.serialization.add_safe_globals(
            [DataEdgeAttr, DataTensorAttr, GlobalStorage]
        )
    except ImportError:
        pass

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

    elif name == "ogbg-moltox21":
        _safe_import_pyg_globals()
        dataset = GraphDataset(name='ogbg-moltox21', root=root_path)
        split_idx = dataset.get_idx_split()
        train_dataset = ChemicalDataset(dataset[split_idx["train"]], class_id=0)

        eval_datasets = []
        for split in splits:
            if split == 'val':
                eval_datasets.append(ChemicalDataset(dataset[split_idx["valid"]], class_id=0))
            elif split == 'test':
                eval_datasets.append(ChemicalDataset(dataset[split_idx["test"]], class_id=0))
            else:
                raise NotImplementedError(f"Split '{split}' is not yet implemented for dataset '{name}'.")
        return train_dataset, eval_datasets

    elif name == "ogbg-molmuv":
        _safe_import_pyg_globals()
        dataset = GraphDataset(name='ogbg-molmuv', root=root_path)
        split_idx = dataset.get_idx_split()
        train_dataset = ChemicalDataset(dataset[split_idx["train"]], class_id=1)

        eval_datasets = []
        for split in splits:
            if split == 'val':
                eval_datasets.append(ChemicalDataset(dataset[split_idx["valid"]], class_id=1))
            elif split == 'test':
                eval_datasets.append(ChemicalDataset(dataset[split_idx["test"]], class_id=1))
            else:
                raise NotImplementedError(f"Split '{split}' is not yet implemented for dataset '{name}'.")
        return train_dataset, eval_datasets

    elif name == "ogbg-molpcba":
        _safe_import_pyg_globals()
        dataset = GraphDataset(name='ogbg-molpcba', root=root_path)
        split_idx = dataset.get_idx_split()
        train_dataset = ChemicalDataset(dataset[split_idx["train"]], class_id = 0)

        eval_datasets = []
        for split in splits:
            if split == 'val':
                eval_datasets.append(ChemicalDataset(dataset[split_idx["valid"]], class_id = 0))
            elif split == 'test':
                eval_datasets.append(ChemicalDataset(dataset[split_idx["test"]], class_id = 0))
            else:
                raise NotImplementedError(f"Split '{split}' is not yet implemented for dataset '{name}'.")
        return train_dataset, eval_datasets

    # -----------------------------------------------------------------------
    # OGB-HIV  (ogbg-molhiv)
    # Binary: active (1) vs inactive (0) against HIV replication.
    # Single-task dataset — class_id is always 0.
    # Scaffold split provided by OGB.
    # -----------------------------------------------------------------------
    elif name == "ogbg-molhiv":
        _safe_import_pyg_globals()
        dataset = GraphDataset(name='ogbg-molhiv', root=root_path)
        split_idx = dataset.get_idx_split()
        train_dataset = ChemicalDataset(dataset[split_idx["train"]], class_id=0)

        eval_datasets = []
        for split in splits:
            if split == 'val':
                eval_datasets.append(ChemicalDataset(dataset[split_idx["valid"]], class_id=0))
            elif split == 'test':
                eval_datasets.append(ChemicalDataset(dataset[split_idx["test"]], class_id=0))
            else:
                raise NotImplementedError(f"Split '{split}' is not yet implemented for dataset '{name}'.")
        return train_dataset, eval_datasets

    elif name == "melanoma":
        from libauc.datasets import Melanoma
        root = os.path.join(root_path, "melanoma")
        train_dataset = Melanoma(root=root, is_test=False, test_size=0.2)
        eval_datasets = []
        for split in splits:
            if split == 'val':
                eval_datasets.append(Melanoma(root=root, is_test=False, test_size=0.2))
            elif split == 'test':
                eval_datasets.append(Melanoma(root=root, is_test=True, test_size=0.2))
            else:
                raise NotImplementedError(f"Split '{split}' is not yet implemented for dataset '{name}'.")
        return train_dataset, eval_datasets
    
    # -----------------------------------------------------------------------
    # DDSM+  (CBIS-DDSM — Curated Breast Imaging Subset of DDSM)
    # Binary mammogram classification: malignant (1) vs benign/normal (0).
    # Expected directory layout:
    #   <root_path>/ddsm/
    #       train.csv          — columns: image_path, pathology  (or custom via kwargs)
    #       test.csv
    #       <image directories referenced by image_path column>
    # CBIS-DDSM CSVs use "pathology" as label: MALIGNANT=1, BENIGN/BENIGN_WITHOUT_CALLBACK=0.
    # A "pathology_binary" column is created automatically if not present.
    # kwargs:
    #   image_size  (int,  default 224)
    #   image_col   (str,  default "image_path")
    #   label_col   (str,  default "pathology_binary")
    #   malignant_value (str, default "MALIGNANT") — raw label treated as positive
    # -----------------------------------------------------------------------
    elif name == "ddsm" or name == "ddsm+":
        image_size       = kwargs.get("image_size",       224)
        image_col        = kwargs.get("image_col",        "image_path")
        label_col        = kwargs.get("label_col",        "pathology_binary")
        malignant_value  = kwargs.get("malignant_value",  "MALIGNANT")
        data_root        = os.path.join(root_path, "ddsm")

        def _load_ddsm_csv(csv_path: str) -> pd.DataFrame:
            df = pd.read_csv(csv_path)
            # Auto-binarise "pathology" column if the target label column is absent
            if label_col not in df.columns and "pathology" in df.columns:
                df[label_col] = (df["pathology"].str.upper() == malignant_value.upper()).astype(np.float32)
            return df

        train_df = _load_ddsm_csv(os.path.join(data_root, "train.csv"))
        train_dataset = MedicalImageCSVDataset(
            csv_path   = os.path.join(data_root, "train.csv"),
            image_root = data_root,
            image_col  = image_col,
            label_col  = label_col,
            transform  = _medical_train_transform(image_size),
        )
        # Patch: re-apply binarisation to the already-loaded dataframe
        train_df = _load_ddsm_csv(os.path.join(data_root, "train.csv"))
        train_dataset.targets    = train_df[label_col].to_numpy().astype(np.float32)
        train_dataset.image_paths = train_df[image_col].tolist()

        eval_datasets = []
        for split in splits:
            if split in ('val', 'test'):
                csv_name = "test.csv" if split == 'test' else "val.csv"
                ev_df = _load_ddsm_csv(os.path.join(data_root, csv_name))
                ev_ds = MedicalImageCSVDataset(
                    csv_path   = os.path.join(data_root, csv_name),
                    image_root = data_root,
                    image_col  = image_col,
                    label_col  = label_col,
                    transform  = _medical_test_transform(image_size),
                )
                ev_ds.targets     = ev_df[label_col].to_numpy().astype(np.float32)
                ev_ds.image_paths = ev_df[image_col].tolist()
                eval_datasets.append(ev_ds)
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