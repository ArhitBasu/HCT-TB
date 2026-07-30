import torch
import torch.nn as nn
from torchvision.models import mobilenet_v2, MobileNet_V2_Weights

class MultiScaleMobileNetV2(nn.Module):
    """
    Multi-Scale MobileNetV2 Backbone for Tuberculosis Detection.
    
    Extracts intermediate feature maps at 4 different spatial scales 
    (Stage 1, Stage 2, Stage 3, Stage 4) while bypassing the final 
    Global Average Pooling layer. This preserves the local spatial 
    context necessary for detecting both tiny micro-nodules and large 
    lung cavities.
    """
    def __init__(self, pretrained: bool = True):
        super(MultiScaleMobileNetV2, self).__init__()
        
        # Load the base MobileNetV2 with or without ImageNet weights
        weights = MobileNet_V2_Weights.DEFAULT if pretrained else None
        base_model = mobilenet_v2(weights=weights)
        features = base_model.features
        
        # We split the features into 4 distinct stages based on the
        # inverted residual block structure of MobileNetV2.
        
        # Stage 1: Initial Conv + early Bottlenecks
        # Output spatial dims (assuming 224x224 input): 56x56
        # Output channels: 24
        self.stage1 = features[:4]
        
        # Stage 2: Middle Bottlenecks
        # Output spatial dims: 28x28
        # Output channels: 32
        self.stage2 = features[4:7]
        
        # Stage 3: Deep Bottlenecks
        # Output spatial dims: 14x14
        # Output channels: 96
        self.stage3 = features[7:14]
        
        # Stage 4: Final Bottlenecks & 1x1 Conv
        # Output spatial dims: 7x7
        # Output channels: 1280
        # (We use up to the last feature layer before the classifier)
        self.stage4 = features[14:]
        
    def forward(self, x: torch.Tensor) -> list[torch.Tensor]:
        """
        Forward pass for Multi-Scale MobileNetV2.
        
        Args:
            x (torch.Tensor): Input tensor of shape (B, 3, H, W)
            
        Returns:
            list[torch.Tensor]: A list of 4 feature maps extracted from 
                                different stages of the network.
        """
        # (B, 3, 224, 224) -> (B, 24, 56, 56)
        f1 = self.stage1(x)
        
        # (B, 24, 56, 56) -> (B, 32, 28, 28)
        f2 = self.stage2(f1)
        
        # (B, 32, 28, 28) -> (B, 96, 14, 14)
        f3 = self.stage3(f2)
        
        # (B, 96, 14, 14) -> (B, 1280, 7, 7)
        f4 = self.stage4(f3)
        
        # Return all stages preserving spatial information 
        # No Global Average Pooling is applied here!
        return [f1, f2, f3, f4]

if __name__ == "__main__":
    # Test the tensor shapes
    model = MultiScaleMobileNetV2(pretrained=False)
    dummy_input = torch.randn(2, 3, 224, 224)
    features = model(dummy_input)
    
    print("Multi-Scale MobileNetV2 Feature Shapes:")
    for i, f in enumerate(features):
        print(f"Stage {i+1}: {f.shape}")
