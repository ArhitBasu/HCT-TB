import torch
import torch.nn as nn

from models.backbones.mobilenetv2 import MultiScaleMobileNetV2
from models.backbones.swin_tiny import SwinTinyBackbone
from models.fusion.feature_pyramid import FeaturePyramidNetwork
from models.fusion.token_projection import MultiScaleTokenProjection
from models.attention.cross_attention import HierarchicalCrossAttention
from models.fusion.transformer_fusion import TransformerFusion
from models.classifier.classification_head import ClassificationHead

class ROIGuidedHybridNetwork(nn.Module):
    """
    1. Purpose: Master module assembling the V3 frozen architecture.
    2. Mathematical Intuition: Defines the forward graph connecting CNN -> FPN -> Tokens -> Cross Attention -> Fusion -> Classifier.
    3. Input Tensor Shape: (B, 3, 224, 224). Output Tensor Shape: (B, num_classes).
    4. IEEE Baseline Improvement: Fully modular, allowing config-driven ablation of individual pathways (e.g., bypassing Swin).
    5. Benefit for TB Detection: Creates the ultimate end-to-end multi-scale TB anomaly detector.
    """
    def __init__(self, config: dict):
        super(ROIGuidedHybridNetwork, self).__init__()
        
        pretrained = config.get('pretrained', True)
        num_classes = config.get('classes', 3)
        dropout = config.get('dropout', 0.1)
        
        # 1. Backbones
        self.cnn_backbone = MultiScaleMobileNetV2(pretrained=pretrained)
        self.swin_backbone = SwinTinyBackbone(pretrained=pretrained)
        embed_dim = self.swin_backbone.embed_dim # 768
        cnn_channels = [24, 32, 96, 320]
        
        # 2. FPN
        self.fpn = FeaturePyramidNetwork(cnn_channels, embed_dim)
        
        # 3. Token Projection
        self.token_projection = MultiScaleTokenProjection(embed_dim, num_scales=4, dropout=dropout)
        
        # 4. Cross Attention
        self.cross_attn = HierarchicalCrossAttention(
            embed_dim=embed_dim, 
            num_heads=config.get('attention_heads', 8), 
            num_stages=4, 
            dropout=dropout
        )
        
        # 5. Fusion
        self.fusion = TransformerFusion(
            embed_dim=embed_dim, 
            num_heads=config.get('attention_heads', 8), 
            encoder_depth=1, 
            dropout=dropout
        )
        
        # 6. Classifier
        self.classifier = ClassificationHead(embed_dim=embed_dim, num_classes=num_classes, dropout=dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Backbones
        cnn_features = self.cnn_backbone(x)
        swin_tokens = self.swin_backbone(x)
        
        # FPN & Projection
        fpn_features = self.fpn(cnn_features)
        cnn_tokens_list = self.token_projection(fpn_features)
        
        # Attention & Fusion
        attended_tokens = self.cross_attn(cnn_tokens_list, swin_tokens)
        fused_vector = self.fusion(attended_tokens, swin_tokens)
        
        # Classification
        logits = self.classifier(fused_vector)
        
        return logits
