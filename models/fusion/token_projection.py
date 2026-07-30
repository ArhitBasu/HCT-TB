import torch
import torch.nn as nn

class TokenProjection(nn.Module):
    """
    1. Purpose: Projects FPN-enhanced CNN spatial maps into sequences of Transformer tokens.
    2. Mathematical Intuition: Applies BN, GELU, and Dropout before flattening spatial dimensions to (H*W), mapping (B, C, H, W) -> (B, H*W, D).
    3. Input Tensor Shape: (B, D, H, W). Output Tensor Shape: (B, H*W, D).
    4. IEEE Baseline Improvement: Introduces Batch Normalization and Dropout into the projection pathway, stabilizing training gradients.
    5. Benefit for TB Detection: Regularizes local feature tokens, preventing the network from overfitting to specific patients' rib structures.
    """
    def __init__(self, embed_dim: int, dropout: float = 0.1):
        super(TokenProjection, self).__init__()
        
        self.projection = nn.Sequential(
            nn.Conv2d(embed_dim, embed_dim, kernel_size=1),
            nn.BatchNorm2d(embed_dim),
            nn.GELU(),
            nn.Dropout(dropout)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # (B, D, H, W) -> (B, D, H, W)
        x = self.projection(x)
        
        # Flatten spatial dimensions: (B, D, H*W)
        B, D, H, W = x.shape
        x = x.view(B, D, H * W)
        
        # Transpose to token sequence: (B, H*W, D)
        x = x.transpose(1, 2)
        
        return x

class MultiScaleTokenProjection(nn.Module):
    def __init__(self, embed_dim: int, num_scales: int = 4, dropout: float = 0.1):
        super(MultiScaleTokenProjection, self).__init__()
        
        self.projections = nn.ModuleList([
            TokenProjection(embed_dim, dropout) for _ in range(num_scales)
        ])

    def forward(self, fpn_features: list[torch.Tensor]) -> list[torch.Tensor]:
        assert len(fpn_features) == len(self.projections)
        
        tokens_list = []
        for feature, proj in zip(fpn_features, self.projections):
            tokens = proj(feature)
            tokens_list.append(tokens)
            
        return tokens_list
