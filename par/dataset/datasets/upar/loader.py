"""
PyTorch Dataset Loader for UPAR_UNIFIED dataset in D:\\AI DATASET.
"""
import os
import pickle
import torch
import numpy as np
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms as T


def get_upar_transforms(height: int = 256, width: int = 128):
    """
    Standard data transforms for Pedestrian Attribute Recognition (PAR).
    """
    normalize = T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

    train_transform = T.Compose([
        T.Resize((height, width)),
        T.Pad(10),
        T.RandomCrop((height, width)),
        T.RandomHorizontalFlip(),
        T.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1),
        T.ToTensor(),
        normalize,
        T.RandomErasing(p=0.2, scale=(0.02, 0.2))
    ])

    val_transform = T.Compose([
        T.Resize((height, width)),
        T.ToTensor(),
        normalize
    ])

    return train_transform, val_transform


class UPARDataset(Dataset):
    """
    PyTorch Dataset class for UPAR_UNIFIED dataset in D:\\AI DATASET.
    Loads unified annotations and outputs [3, H, W] images and [40] targets.
    """
    def __init__(self, root_dir: str = r"D:\AI DATASET\UPAR_UNIFIED", split: str = 'train', transform=None, sample_ratio: float = 1.0):
        self.root_dir = root_dir
        self.split = split
        self.transform = transform
        
        if split in ['train', 'val', 'test']:
            self.pkl_path = os.path.join(root_dir, "annotations", f"{split}.pkl")
        else:
            self.pkl_path = os.path.join(root_dir, "annotations", "unified_annotations.pkl")
            
        if not os.path.exists(self.pkl_path):
            raise FileNotFoundError(f"Annotation file not found: {self.pkl_path}")
            
        with open(self.pkl_path, "rb") as f:
            data = pickle.load(f)
            
        self.image_names = np.array(data["image_name"])
        self.labels = np.array(data["label"])  # shape (N, 40)
        self.dataset_ids = np.array(data["dataset_ids"])
        self.dataset_names = np.array(data["dataset_names"])
        self.partition = np.array(data["partition"])
        self.attr_names = list(data["attr_name"])
        self.num_attributes = len(self.attr_names)
        
        if sample_ratio < 1.0 and len(self.image_names) > 0:
            num_samples = max(10, int(len(self.image_names) * sample_ratio))
            self.image_names = self.image_names[:num_samples]
            self.labels = self.labels[:num_samples]
            self.dataset_ids = self.dataset_ids[:num_samples]
            self.dataset_names = self.dataset_names[:num_samples]
            self.partition = self.partition[:num_samples]

        self.datasets_disk_root = r"D:\AI DATASET\3 Datasets"
        self.base_root = r"D:\AI DATASET"

    def __len__(self):
        return len(self.image_names)

    def _resolve_image_path(self, rel_path: str):
        p1 = os.path.join(self.datasets_disk_root, rel_path)
        if os.path.exists(p1):
            return p1
            
        p2 = os.path.join(self.datasets_disk_root, rel_path.replace("PA100k", "PA-100K"))
        if os.path.exists(p2):
            return p2
            
        p3 = os.path.join(self.base_root, rel_path)
        if os.path.exists(p3):
            return p3

        return None

    def __getitem__(self, index: int):
        rel_path = self.image_names[index]
        label = self.labels[index]
        dataset_id = self.dataset_ids[index]
        
        img_path = self._resolve_image_path(rel_path)
        
        if img_path is not None:
            try:
                img = Image.open(img_path).convert('RGB')
            except Exception:
                img = Image.new('RGB', (128, 256), color=(128, 128, 128))
        else:
            img = Image.new('RGB', (128, 256), color=(128, 128, 128))
            
        if self.transform is not None:
            img = self.transform(img)

        label_tensor = torch.tensor(label, dtype=torch.float32)
        return img, label_tensor, dataset_id, rel_path
