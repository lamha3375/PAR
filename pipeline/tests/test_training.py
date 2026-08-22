import sys
import os
import torch
from torch.utils.data import DataLoader

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from datasets.upar.loader import UPARDataset, get_upar_transforms
from models.hydraplus.par_model import UnifiedPARModel
from training.loss import MaskedBCEWithLogitsLoss

def main():
    print("=" * 60)
    print("TEST: TESTING FORWARD + LOSS + BACKWARD (D:\\AI DATASET)")
    print("=" * 60)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    train_transform, _ = get_upar_transforms(height=256, width=128)
    dataset = UPARDataset(split='train', transform=train_transform)
    loader = DataLoader(dataset, batch_size=8, shuffle=True)
    
    images, labels, _, _ = next(iter(loader))
    images = images.to(device)
    labels = labels.to(device)
    
    # 1. Forward Pass
    model = UnifiedPARModel(num_attributes=40, backbone_name='resnet50', pretrained=True).to(device)
    model.train()
    
    outputs = model(images)
    print(f"Forward Pass: PASS (outputs shape: {outputs.shape})")
    assert outputs.shape == torch.Size([8, 40]), "Outputs shape mismatch!"
    
    # 2. Loss Computation
    criterion = MaskedBCEWithLogitsLoss().to(device)
    loss = criterion(outputs, labels)
    print(f"Loss Pass: PASS (loss value: {loss.item():.4f})")
    assert not torch.isnan(loss) and not torch.isinf(loss), "Loss is NaN or Inf!"
    
    # 3. Backward Pass
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
    optimizer.zero_grad()
    loss.backward()
    
    grad_count = 0
    for param in model.parameters():
        if param.grad is not None:
            grad_count += 1
    print(f"Backward Pass: PASS ({grad_count} parameters received gradients)")
    assert grad_count > 0, "No gradients computed!"
    
    optimizer.step()
    
    print("\n" + "=" * 60)
    print("TEST TRAINING RESULT: ALL 3 PASS (Forward: PASS, Loss: PASS, Backward: PASS)")
    print("=" * 60)

if __name__ == "__main__":
    main()
