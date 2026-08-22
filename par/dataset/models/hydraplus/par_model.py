"""
Unified PAR Multi-Label Classification Neural Network Model in D:\\AI DATASET.
Includes Spatial Region Attention for localization of Head vs Body attributes.
Uses LayerNorm(512) in projection head for complete batch-size independence.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from models.hydraplus.backbone import build_backbone


class SpatialAttention(nn.Module):
    def __init__(self, in_channels: int):
        super(SpatialAttention, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, in_channels // 4, kernel_size=1),
            nn.BatchNorm2d(in_channels // 4),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels // 4, 1, kernel_size=3, padding=1),
            nn.Sigmoid()
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        attn_map = self.conv(x)  # (B, 1, H, W)
        return x * attn_map + x   # Residual attention feature map


class UnifiedPARModel(nn.Module):
    def __init__(self, num_attributes: int = 40, backbone_name: str = 'resnet50', pretrained: bool = True, dropout: float = 0.4):
        super(UnifiedPARModel, self).__init__()
        self.num_attributes = num_attributes
        self.backbone_name = backbone_name
        
        self.backbone, self.feature_dim = build_backbone(name=backbone_name, pretrained=pretrained)
        self.attention = SpatialAttention(self.feature_dim)
        self.pooling = nn.AdaptiveAvgPool2d((1, 1))
        
        self.head = nn.Sequential(
            nn.Linear(self.feature_dim, 512),
            nn.LayerNorm(512),
            nn.ReLU(inplace=True),
            nn.Dropout(p=dropout),
            nn.Linear(512, num_attributes)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        features = self.backbone(x)
        attn_features = self.attention(features)
        pooled = self.pooling(attn_features)
        flattened = torch.flatten(pooled, 1)
        logits = self.head(flattened)
        return logits
