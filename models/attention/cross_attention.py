import torch
import torch.nn as nn
import torch.nn.functional as F

class CrossAttentionBlock(nn.Module):
    """
    1. Purpose: Full Transformer-style Hierarchical Cross-Attention Block.
    2. Mathematical Intuition: X = X + MHA(LN(X), LN(Y), LN(Y)), Out = X + FFN(LN(X)).
    3. Input Tensor Shape: Q (CNN) -> (B, N_q, D), KV (Swin) -> (B, N_kv, D). Output: (B, N_q, D).
    4. IEEE Baseline Improvement: Implements a rigorous Pre-LN Transformer decoder block architecture rather than naked attention.
    5. Benefit for TB Detection: The dual residual pathways prevent the vanishing gradient problem, allowing the model to learn deep contextual links between local TB lesions and global lung structure.
    """
    def __init__(self, embed_dim: int, num_heads: int, mlp_ratio: float = 4.0, dropout: float = 0.1):
        super(CrossAttentionBlock, self).__init__()
        
        self.norm1_q = nn.LayerNorm(embed_dim)
        self.norm1_kv = nn.LayerNorm(embed_dim)
        
        self.cross_attn = nn.MultiheadAttention(
            embed_dim=embed_dim, 
            num_heads=num_heads, 
            dropout=dropout,
            batch_first=True
        )
        
        self.norm2 = nn.LayerNorm(embed_dim)
        
        # Feed Forward Network (MLP)
        hidden_dim = int(embed_dim * mlp_ratio)
        self.mlp = nn.Sequential(
            nn.Linear(embed_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, embed_dim),
            nn.Dropout(dropout)
        )

    def forward(self, x_q: torch.Tensor, x_kv: torch.Tensor) -> torch.Tensor:
        # Pre-Norm Formulation
        q = self.norm1_q(x_q)
        kv = self.norm1_kv(x_kv)
        
        # Cross Attention + First Residual
        attn_out, _ = self.cross_attn(query=q, key=kv, value=kv)
        x = x_q + attn_out
        
        # FFN + Second Residual
        out = x + self.mlp(self.norm2(x))
        
        return out

class HierarchicalCrossAttention(nn.Module):
    def __init__(self, embed_dim: int, num_heads: int, num_stages: int = 4, dropout: float = 0.1):
        super(HierarchicalCrossAttention, self).__init__()
        self.num_stages = num_stages
        self.blocks = nn.ModuleList([
            CrossAttentionBlock(embed_dim=embed_dim, num_heads=num_heads, dropout=dropout)
            for _ in range(num_stages)
        ])

    def forward(self, cnn_tokens_list: list[torch.Tensor], swin_tokens: torch.Tensor) -> list[torch.Tensor]:
        attended_tokens = []
        for i, block in enumerate(self.blocks):
            out = block(cnn_tokens_list[i], swin_tokens)
            attended_tokens.append(out)
        return attended_tokens
