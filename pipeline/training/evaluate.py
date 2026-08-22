"""
Evaluation Metrics for Pedestrian Attribute Recognition (PAR) in D:\\AI DATASET.
Calculates Mean Accuracy (mA), Precision, Recall, F1-score per attribute and overall.
Ignores Unknown labels (target == 2.0).
"""
import torch
import numpy as np

UPAR_ATTRIBUTES = [
    "Age-Young", "Age-Adult", "Age-Old", "Gender-Female",
    "Hair-Length-Short", "Hair-Length-Long", "Hair-Length-Bald",
    "UpperBody-Length-Short", "UpperBody-Color-Black", "UpperBody-Color-Blue",
    "UpperBody-Color-Brown", "UpperBody-Color-Green", "UpperBody-Color-Grey",
    "UpperBody-Color-Orange", "UpperBody-Color-Pink", "UpperBody-Color-Purple",
    "UpperBody-Color-Red", "UpperBody-Color-White", "UpperBody-Color-Yellow",
    "UpperBody-Color-Other", "LowerBody-Length-Short", "LowerBody-Color-Black",
    "LowerBody-Color-Blue", "LowerBody-Color-Brown", "LowerBody-Color-Green",
    "LowerBody-Color-Grey", "LowerBody-Color-Orange", "LowerBody-Color-Pink",
    "LowerBody-Color-Purple", "LowerBody-Color-Red", "LowerBody-Color-White",
    "LowerBody-Color-Yellow", "LowerBody-Color-Other", "LowerBody-Type-Trousers&Shorts",
    "LowerBody-Type-Skirt&Dress", "Accessory-Backpack", "Accessory-Bag",
    "Accessory-Glasses-Normal", "Accessory-Glasses-Sun", "Accessory-Hat"
]

def compute_par_metrics(probs: np.ndarray, targets: np.ndarray, thresholds: dict = None) -> dict:
    N, num_attrs = probs.shape
    thresh_vec = np.full(num_attrs, 0.5)

    preds = (probs >= thresh_vec).astype(int)

    acc_list = []
    precision_list = []
    recall_list = []
    f1_list = []

    per_attribute_metrics = {}

    for i in range(num_attrs):
        attr_name = UPAR_ATTRIBUTES[i] if i < len(UPAR_ATTRIBUTES) else f"attr_{i}"
        
        valid_mask = (targets[:, i] != 2)
        valid_preds = preds[:, i][valid_mask]
        valid_targets = targets[:, i][valid_mask]

        pos_gt = np.sum(valid_targets == 1)
        neg_gt = np.sum(valid_targets == 0)

        tp = np.sum((valid_preds == 1) & (valid_targets == 1))
        tn = np.sum((valid_preds == 0) & (valid_targets == 0))
        fp = np.sum((valid_preds == 1) & (valid_targets == 0))
        fn = np.sum((valid_preds == 0) & (valid_targets == 1))

        pos_acc = tp / (pos_gt + 1e-5) if pos_gt > 0 else 1.0
        neg_acc = tn / (neg_gt + 1e-5) if neg_gt > 0 else 1.0
        attr_ma = 0.5 * (pos_acc + neg_acc)

        precision = tp / (tp + fp + 1e-5)
        recall = tp / (tp + fn + 1e-5)
        f1 = 2 * precision * recall / (precision + recall + 1e-5)

        acc_list.append(attr_ma)
        precision_list.append(precision)
        recall_list.append(recall)
        f1_list.append(f1)

        per_attribute_metrics[attr_name] = {
            "mA": float(attr_ma),
            "precision": float(precision),
            "recall": float(recall),
            "f1": float(f1),
            "threshold": float(thresh_vec[i])
        }

    overall_mA = float(np.mean(acc_list))
    overall_precision = float(np.mean(precision_list))
    overall_recall = float(np.mean(recall_list))
    overall_f1 = float(np.mean(f1_list))

    return {
        "mA": overall_mA,
        "precision": overall_precision,
        "recall": overall_recall,
        "f1": overall_f1,
        "per_attribute": per_attribute_metrics
    }


def evaluate_model(model: torch.nn.Module, dataloader: torch.utils.data.DataLoader, device: torch.device, thresholds: dict = None, criterion = None) -> dict:
    model.eval()
    all_probs = []
    all_targets = []
    val_loss = 0.0
    val_acc = 0.0
    total_batches = len(dataloader)

    with torch.no_grad():
        for imgs, labels, _, _ in dataloader:
            imgs = imgs.to(device)
            labels_device = labels.to(device)
            logits = model(imgs)
            probs = torch.sigmoid(logits)
            
            if criterion is not None:
                loss = criterion(logits, labels_device)
                val_loss += loss.item()
                
            preds = (probs > 0.5).float()
            valid_mask = (labels_device != 2.0)
            batch_acc = (preds[valid_mask] == labels_device[valid_mask]).float().mean().item() if valid_mask.sum() > 0 else 1.0
            val_acc += batch_acc
            
            all_probs.append(probs.cpu().numpy())
            all_targets.append(labels.numpy())

    all_probs = np.vstack(all_probs)
    all_targets = np.vstack(all_targets)

    metrics = compute_par_metrics(all_probs, all_targets, thresholds=thresholds)
    metrics["loss"] = val_loss / total_batches if total_batches > 0 else 0.0
    metrics["accuracy"] = val_acc / total_batches if total_batches > 0 else 0.0
    return metrics
