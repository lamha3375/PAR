import sys
import os
import torch
from torch.utils.data import DataLoader

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from datasets.upar.loader import UPARDataset, get_upar_transforms
from models.hydraplus.par_model import UnifiedPARModel

def main():
    print("=" * 60)
    print("TEST: TESTING MODEL FORWARD PASS (D:\\AI DATASET)")
    print("=" * 60)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    train_transform, _ = get_upar_transforms(height=256, width=128)
    dataset = UPARDataset(split='train', transform=train_transform)
    loader = DataLoader(dataset, batch_size=8, shuffle=True)
    
    images, labels, _, _ = next(iter(loader))
    images = images.to(device)
    labels = labels.to(device)
    
    print(f"Input images shape: {images.shape}")
    print(f"Input labels shape: {labels.shape}")
    
    model = UnifiedPARModel(num_attributes=40, backbone_name='resnet50', pretrained=True).to(device)
    model.eval()
    
    with torch.no_grad():
        outputs = model(images)
        
    print(f"Model outputs shape: {outputs.shape}")
    assert outputs.shape == torch.Size([8, 40]), f"Expected outputs shape [8, 40], got {outputs.shape}"
    
    print("\n" + "=" * 60)
    print("TEST MODEL RESULT: PASS")
    print("=" * 60)

if __name__ == "__main__":
    main()
