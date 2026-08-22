import sys
import os
import torch
from torch.utils.data import DataLoader

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from datasets.upar.loader import UPARDataset, get_upar_transforms

def main():
    print("=" * 60)
    print("TEST: TESTING UPAR DATASET LOADER (D:\\AI DATASET)")
    print("=" * 60)
    
    train_transform, val_transform = get_upar_transforms(height=256, width=128)
    
    dataset = UPARDataset(split='all', transform=val_transform)
    ds_len = len(dataset)
    print(f"Dataset length: {ds_len}")
    assert ds_len == 145656, f"Expected dataset length 145656, got {ds_len}"
    
    img, label, ds_id, img_name = dataset[0]
    print(f"Sample image name: {img_name}")
    print(f"Sample dataset ID: {ds_id}")
    print(f"Single image shape: {img.shape}")
    print(f"Single label shape: {label.shape}")
    print(f"Single label dtype: {label.dtype}")
    print(f"Unique label values in sample: {torch.unique(label).tolist()}")
    
    assert img.shape == torch.Size([3, 256, 128]), f"Expected image shape [3, 256, 128], got {img.shape}"
    assert label.shape == torch.Size([40]), f"Expected label shape [40], got {label.shape}"
    
    loader = DataLoader(dataset, batch_size=16, shuffle=True, num_workers=0)
    batch_imgs, batch_labels, batch_ids, batch_names = next(iter(loader))
    
    print(f"Batch images shape: {batch_imgs.shape}")
    print(f"Batch labels shape: {batch_labels.shape}")
    print(f"Unique values in batch labels: {torch.unique(batch_labels).tolist()}")
    
    assert batch_imgs.shape == torch.Size([16, 3, 256, 128]), f"Expected batch imgs shape [16, 3, 256, 128], got {batch_imgs.shape}"
    assert batch_labels.shape == torch.Size([16, 40]), f"Expected batch labels shape [16, 40], got {batch_labels.shape}"
    
    print("\n" + "=" * 60)
    print("TEST DATASET RESULT: PASS")
    print("=" * 60)

if __name__ == "__main__":
    main()
