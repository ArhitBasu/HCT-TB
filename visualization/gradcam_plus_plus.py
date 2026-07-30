import torch
import torch.nn.functional as F
import cv2
import numpy as np

class GradCAMPlusPlus:
    """
    1. Purpose: Generates high-fidelity Grad-CAM++ heatmaps to localize TB anomalies.
    2. Mathematical Intuition: Uses higher-order derivatives to weight the activations, effectively highlighting multiple occurrences of an object (e.g., multiple miliary nodules) better than standard Grad-CAM.
    3. Input Tensor Shape: (B, C, H, W). Output Tensor Shape: (H, W).
    4. IEEE Baseline Improvement: Grad-CAM++ resolves the 'blob' issue of standard CAM where scattered lesions merge into one useless heatmap.
    5. Benefit for TB Detection: TB often manifests as bilateral or multi-focal lesions. Grad-CAM++ captures these disparate features simultaneously.
    """
    def __init__(self, model, target_layer):
        self.model = model
        self.target_layer = target_layer
        self.gradients = None
        self.activations = None
        
        target_layer.register_forward_hook(self.save_activation)
        target_layer.register_full_backward_hook(self.save_gradient)

    def save_activation(self, module, input, output):
        self.activations = output.detach()

    def save_gradient(self, module, grad_input, grad_output):
        self.gradients = grad_output[0].detach()

    def __call__(self, x: torch.Tensor, class_idx: int = None):
        logits = self.model(x)
        if class_idx is None:
            class_idx = logits.argmax(dim=1).item()
            
        self.model.zero_grad()
        target = logits[0, class_idx]
        target.backward(retain_graph=True)
        
        # Grad-CAM++ weights formulation
        b, k, u, v = self.gradients.size()
        
        alpha_num = self.gradients.pow(2)
        alpha_denom = 2.0 * alpha_num + self.activations.mul(self.gradients.pow(3)).view(b, k, u*v).sum(-1, keepdim=True).view(b, k, 1, 1)
        # Avoid division by zero
        alpha_denom = torch.where(alpha_denom != 0.0, alpha_denom, torch.ones_like(alpha_denom))
        
        alphas = alpha_num / alpha_denom
        weights = (alphas * F.relu(self.gradients)).view(b, k, u*v).sum(-1)
        weights = weights.view(b, k, 1, 1)
        
        cam = torch.sum(weights * self.activations, dim=1).squeeze()
        cam = F.relu(cam)
        cam -= torch.min(cam)
        cam /= torch.max(cam)
        
        return cam.cpu().numpy()
