import torch
import torch.nn as nn
import torch.nn.functional as F

class FeaturePyramidNetwork(nn.Module):
    """
    1. Purpose: Enhances multi-scale MobileNetV2 features using a top-down lateral architecture (FPN).
    2. Mathematical Intuition: P_i = Conv1x1(C_i) + Upsample(P_{i+1}). Merges high-level semantic context with low-level spatial detail.
    3. Input Tensor Shape: List of 4 tensors [(B, 24, 56, 56), (B, 32, 28, 28), (B, 96, 14, 14), (B, 320, 7, 7)]
       Output Tensor Shape: List of 4 tensors all with channel dimension `embed_dim`.
    4. IEEE Baseline Improvement: Baseline relies on a single final CNN feature map. FPN explicitly mines multi-scale anomalies.
    5. Benefit for TB Detection: Small miliary TB nodules need high spatial resolution (Stage 1), but deep semantic context (Stage 4) to differentiate from noise. FPN fuses both.
    """
    def __init__(self, in_channels_list: list[int], embed_dim: int):
        super(FeaturePyramidNetwork, self).__init__()
        
        # Lateral 1x1 convolutions to align channel dimensions to embed_dim
        self.lateral_convs = nn.ModuleList([
            nn.Conv2d(in_channels, embed_dim, kernel_size=1) 
            for in_channels in in_channels_list
        ])
        
        # Smoothing 3x3 convolutions to reduce aliasing from upsampling
        self.fpn_convs = nn.ModuleList([
            nn.Conv2d(embed_dim, embed_dim, kernel_size=3, padding=1)
            for _ in range(len(in_channels_list))
        ])

    def forward(self, features: list[torch.Tensor]) -> list[torch.Tensor]:
        # 1. Apply lateral 1x1 convs
        laterals = [lat_conv(f) for lat_conv, f in zip(self.lateral_convs, features)]
        
        # 2. Top-down pathway
        # Start from the deepest layer (Stage 4) and go backwards
        for i in range(len(laterals) - 1, 0, -1):
            # Upsample the deeper layer to match the spatial dimensions of the shallower layer
            target_shape = laterals[i-1].shape[2:]
            top_down = F.interpolate(laterals[i], size=target_shape, mode='nearest')
            # Add to the shallower lateral feature
            laterals[i-1] = laterals[i-1] + top_down
            
        # 3. Apply smoothing 3x3 convs
        fpn_outs = [fpn_conv(lat) for fpn_conv, lat in zip(self.fpn_convs, laterals)]
        
        return fpn_outs
