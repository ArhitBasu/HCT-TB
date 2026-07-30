import torch
import torch.nn as nn
from models.attention.cross_attention import CrossAttentionBlock

class HierarchicalCrossAttention(nn.Module):
    """
    Hierarchical Multi-Scale Multi-Token Cross-Attention Module.
    
    This encapsulates 4 independent Cross-Attention blocks. 
    Each block receives CNN tokens from a specific scale (Stage 1 to 4) 
    as Queries, and uses the identical Swin Transformer tokens as Keys/Values.
    
    This ensures that CNN features at EVERY scale interact with the 
    global anatomical context, perfectly fulfilling the user's primary novelty.
    """
    def __init__(self, embed_dim: int, num_heads: int, num_stages: int = 4):
        super(HierarchicalCrossAttention, self).__init__()
        
        self.num_stages = num_stages
        
        # Instantiate 4 independent Cross-Attention blocks
        self.blocks = nn.ModuleList([
            CrossAttentionBlock(embed_dim=embed_dim, num_heads=num_heads)
            for _ in range(num_stages)
        ])

    def forward(self, cnn_tokens_list: list[torch.Tensor], swin_tokens: torch.Tensor) -> list[torch.Tensor]:
        """
        Forward pass.
        
        Args:
            cnn_tokens_list (list[torch.Tensor]): 4 tensors of shape (B, N_i, D)
            swin_tokens (torch.Tensor): 1 tensor of shape (B, N_swin, D)
            
        Returns:
            list[torch.Tensor]: 4 tensors of shape (B, N_i, D) (Cross-attended)
        """
        assert len(cnn_tokens_list) == self.num_stages
        
        attended_tokens = []
        for i, block in enumerate(self.blocks):
            # cnn_tokens is the query, swin_tokens acts as Key and Value
            out = block(cnn_tokens_list[i], swin_tokens)
            attended_tokens.append(out)
            
        return attended_tokens

if __name__ == "__main__":
    # Test tensor shapes
    B, D = 2, 768
    num_heads = 8
    
    # Mock CNN tokens of varying sequence lengths (3136, 784, 196, 49)
    cnn_tokens_list = [
        torch.randn(B, 3136, D),
        torch.randn(B, 784, D),
        torch.randn(B, 196, D),
        torch.randn(B, 49, D)
    ]
    
    # Mock Swin Tokens (49 tokens from a 7x7 grid)
    swin_tokens = torch.randn(B, 49, D)
    
    model = HierarchicalCrossAttention(embed_dim=D, num_heads=num_heads)
    
    out_list = model(cnn_tokens_list, swin_tokens)
    
    print("Hierarchical Cross-Attention Output Shapes:")
    for i, out in enumerate(out_list):
        print(f"Stage {i+1} Attended Tokens: {out.shape}")
