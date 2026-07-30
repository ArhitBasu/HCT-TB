import torch
import torch.nn as nn

class ClassificationHead(nn.Module):
    """
    1. Purpose: Maps the fused Transformer representation into final classification logits/probabilities.
    2. Mathematical Intuition: Applies double dropout with a non-linear GELU bottleneck to strongly regularize the feature space before projecting to the probability simplex.
    3. Input Tensor Shape: (B, D). Output Tensor Shape: (B, num_classes).
    4. IEEE Baseline Improvement: Replaces a naked linear layer with a deep, regularized MLP, combating the severe overfitting seen in baseline models.
    5. Benefit for TB Detection: TBX11K has class imbalances and limited samples. Heavy dropout forces the network to distribute decision logic across multiple features rather than relying on a single memorized node.
    """
    def __init__(self, embed_dim: int, num_classes: int = 3, dropout: float = 0.3):
        super(ClassificationHead, self).__init__()
        
        hidden_dim = embed_dim // 2
        
        # We don't need GAP here since TransformerFusion already outputs a 1D CLS token (B, D)
        # If input was spatial, we'd add nn.AdaptiveAvgPool2d(1)
        
        self.mlp = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(embed_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, num_classes)
            # Softmax is typically handled by the CrossEntropyLoss criterion directly,
            # but if strictly required as output, it can be added during inference.
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.mlp(x)
