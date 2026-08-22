"""
Loss Functions for Multi-Label Pedestrian Attribute Recognition (PAR) in D:\\AI DATASET.
Supports Masked BCE Loss for handling Unknown labels (2.0).
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class MaskedBCEWithLogitsLoss(nn.Module):
    """
    Masked Binary Cross Entropy with Logits Loss.
    Masks out unknown labels (where target == 2.0) so they contribute 0 to loss and gradient.
    Supports optional positive class weighting (pos_weight) for label imbalance.
    """
    def __init__(self, pos_weights: torch.Tensor = None):
        super(MaskedBCEWithLogitsLoss, self).__init__()
        if pos_weights is not None:
            self.pos_weight = pos_weights.float()
        else:
            self.pos_weight = None

    def compute_pos_weights(self, labels: np.ndarray, eps: float = 1e-5) -> torch.Tensor:
        pos_counts = np.sum(labels == 1, axis=0)
        neg_counts = np.sum(labels == 0, axis=0)
        weights = np.clip(neg_counts / (pos_counts + eps), 0.1, 10.0)
        pos_weight_tensor = torch.tensor(weights, dtype=torch.float32)
        self.pos_weight = pos_weight_tensor
        return pos_weight_tensor

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        mask = (targets != 2.0).float()
        targets_clamped = torch.clamp(targets, min=0.0, max=1.0)
        
        pos_weight = self.pos_weight.to(logits.device) if self.pos_weight is not None else None
        
        loss_element = F.binary_cross_entropy_with_logits(
            logits,
            targets_clamped,
            pos_weight=pos_weight,
            reduction='none'
        )
        
        masked_loss = loss_element * mask
        total_valid = mask.sum()
        if total_valid > 0:
            return masked_loss.sum() / total_valid
        else:
            return masked_loss.sum() * 0.0


class WeightedBCEWithLogitsLoss(MaskedBCEWithLogitsLoss):
    pass
