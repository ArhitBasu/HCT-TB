import torch
import torch.nn as nn
import torch.nn.functional as F

class FocalLoss(nn.Module):
    """
    1. Purpose: Computes Focal Loss to handle severe class imbalances in medical datasets.
    2. Mathematical Intuition: FL(p_t) = -alpha_t * (1 - p_t)^gamma * log(p_t). Reduces the relative loss for well-classified examples, putting more focus on hard, misclassified examples.
    3. Input Tensor Shape: logits (B, C), targets (B). Output Tensor Shape: Scalar (1).
    4. IEEE Baseline Improvement: CrossEntropy strictly heavily biases models toward the majority 'Healthy' class. Focal loss dynamically scales gradients based on prediction confidence.
    5. Benefit for TB Detection: In TBX11K, 'Tuberculosis' is a minority class. This forces the model to dedicate capacity to finding subtle TB lesions rather than over-optimizing 'Healthy' scans.
    """
    def __init__(self, alpha: list = None, gamma: float = 2.0, reduction: str = 'mean'):
        super(FocalLoss, self).__init__()
        self.gamma = gamma
        self.reduction = reduction
        # Alpha should be a list of weights for each class
        if alpha is not None:
            self.alpha = torch.tensor(alpha, dtype=torch.float32)
        else:
            self.alpha = None

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        ce_loss = F.cross_entropy(logits, targets, reduction='none')
        pt = torch.exp(-ce_loss)
        
        focal_loss = (1 - pt) ** self.gamma * ce_loss
        
        if self.alpha is not None:
            if self.alpha.device != logits.device:
                self.alpha = self.alpha.to(logits.device)
            alpha_t = self.alpha[targets]
            focal_loss = alpha_t * focal_loss
            
        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        else:
            return focal_loss
