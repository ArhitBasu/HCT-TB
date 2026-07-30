import torch
import torch.nn as nn
import torch.nn.functional as F

class LabelSmoothingLoss(nn.Module):
    """
    1. Purpose: Regularization technique that prevents the model from predicting classes with absolute certainty.
    2. Mathematical Intuition: y_k = (1 - alpha) * delta_{k,y} + alpha / K. Distributes a small fraction of probability mass to all non-target classes.
    3. Input Tensor Shape: logits (B, C), targets (B). Output Tensor Shape: Scalar (1).
    4. IEEE Baseline Improvement: Deep networks easily memorize small datasets and become over-confident, degrading generalization.
    5. Benefit for TB Detection: Radiological labels have inherent inter-reader variability. A 'Healthy' scan might have minor abnormalities missed by the annotator. Smoothing prevents the network from treating labels as absolute truth.
    """
    def __init__(self, num_classes: int, smoothing: float = 0.1, reduction: str = 'mean'):
        super(LabelSmoothingLoss, self).__init__()
        self.num_classes = num_classes
        self.smoothing = smoothing
        self.reduction = reduction

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        # Note: PyTorch's native F.cross_entropy now supports label_smoothing directly.
        # This wrapper provides a modular interface for the experiment configuration system.
        return F.cross_entropy(
            logits, 
            targets, 
            label_smoothing=self.smoothing, 
            reduction=self.reduction
        )
