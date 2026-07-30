import torch
import torch.nn as nn
import timm

class SwinTinyBackbone(nn.Module):
    """
    Swin Transformer Backbone (Swin-Tiny) for Tuberculosis Detection.
    
    Replaces the baseline DeiT model. Swin Transformer utilizes hierarchical 
    feature maps and shifted window attention, which is computationally 
    efficient and highly effective for medical image analysis.
    
    This module extracts the raw patch tokens from the final stage of Swin 
    without applying pooling, preserving spatial and global context for 
    the downstream Cross-Attention module.
    """
    def __init__(self, pretrained: bool = True):
        super(SwinTinyBackbone, self).__init__()
        
        # Load Swin-Tiny pretrained on ImageNet
        self.swin = timm.create_model(
            'swin_tiny_patch4_window7_224',
            pretrained=pretrained,
            num_classes=0,       # Remove the classification head
            global_pool=''       # Disable global pooling to get raw tokens
        )
        
        self.embed_dim = self.swin.num_features # Typically 768 for Swin-Tiny

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass for Swin-Tiny Backbone.
        
        Args:
            x (torch.Tensor): Input tensor of shape (B, 3, 224, 224)
            
        Returns:
            torch.Tensor: Swin tokens of shape (B, N, D)
                          where N is the sequence length (e.g., 49) 
                          and D is the embedding dimension (e.g., 768).
        """
        # Forward through the transformer stages
        # Expected output shape: (B, 7, 7, 768)
        features = self.swin.forward_features(x)
        
        # Flatten the spatial dimensions to create a sequence of tokens
        # (B, 7, 7, 768) -> (B, 49, 768)
        B, H, W, C = features.shape
        tokens = features.view(B, H * W, C)
        
        return tokens

if __name__ == "__main__":
    # Test the tensor shapes
    model = SwinTinyBackbone(pretrained=False)
    dummy_input = torch.randn(2, 3, 224, 224)
    swin_tokens = model(dummy_input)
    
    print("Swin-Tiny Backbone Feature Shape:")
    print(f"Output shape: {swin_tokens.shape}") # Should be (2, 49, 768)
