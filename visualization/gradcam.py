import torch
import torch.nn.functional as F
import cv2
import numpy as np
import matplotlib.pyplot as plt

class GradCAM:
    """
    Implements Gradient-weighted Class Activation Mapping (Grad-CAM)
    for the CNN branch of the hybrid network to provide visual 
    explainability for the medical predictions.
    """
    def __init__(self, model, target_layer):
        self.model = model
        self.target_layer = target_layer
        self.gradients = None
        self.activations = None
        
        # Register hooks
        target_layer.register_forward_hook(self.save_activation)
        target_layer.register_full_backward_hook(self.save_gradient)

    def save_activation(self, module, input, output):
        self.activations = output.detach()

    def save_gradient(self, module, grad_input, grad_output):
        # grad_output is a tuple
        self.gradients = grad_output[0].detach()

    def __call__(self, x: torch.Tensor, class_idx: int):
        # Forward pass
        logits = self.model(x)
        
        if class_idx is None:
            class_idx = logits.argmax(dim=1).item()
            
        self.model.zero_grad()
        # Backward pass for the target class
        target = logits[0, class_idx]
        target.backward(retain_graph=True)
        
        # Compute weights (Global Average Pooling of the gradients)
        weights = torch.mean(self.gradients, dim=[2, 3], keepdim=True)
        
        # Multiply activations by weights
        cam = torch.sum(weights * self.activations, dim=1).squeeze()
        
        # Apply ReLU to only keep features that have a positive influence
        cam = F.relu(cam)
        
        # Normalize between 0 and 1
        cam -= torch.min(cam)
        cam /= torch.max(cam)
        
        return cam.cpu().numpy()

def overlay_cam_on_image(img_tensor, cam_mask, alpha=0.5, colormap=cv2.COLORMAP_JET):
    """
    Overlays the CAM heatmap onto the original image tensor.
    """
    # Convert tensor image to numpy array (H, W, C)
    img_np = img_tensor.squeeze().permute(1, 2, 0).cpu().numpy()
    
    # Resize cam mask to match image size
    cam_resized = cv2.resize(cam_mask, (img_np.shape[1], img_np.shape[0]))
    
    # Convert mask to heatmap
    heatmap = cv2.applyColorMap(np.uint8(255 * cam_resized), colormap)
    heatmap = np.float32(heatmap) / 255.0
    
    # Combine original image and heatmap
    cam_img = heatmap * alpha + img_np * (1 - alpha)
    cam_img = cam_img / np.max(cam_img)
    
    return np.uint8(255 * cam_img)
