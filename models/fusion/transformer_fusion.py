import torch
import torch.nn as nn

class TransformerFusion(nn.Module):
    """
    1. Purpose: Fuses multi-scale cross-attended tokens using a true Transformer Encoder.
    2. Mathematical Intuition: Multi-Head Self-Attention over pooled scale-specific tokens establishes inter-scale dependencies.
    3. Input Tensor Shape: List of 4 tensors (B, N_i, D) + 1 tensor (B, N_s, D). Output: (B, D).
    4. IEEE Baseline Improvement: Replaces naive 1D concatenation with dynamic self-attention over scale embeddings.
    5. Benefit for TB Detection: Autonomously learns whether the critical TB features in this specific patient reside in the micro-scale (Stage 1) or macro-scale (Stage 4) structures.
    """
    def __init__(self, embed_dim: int, num_heads: int = 8, encoder_depth: int = 1, dropout: float = 0.1):
        super(TransformerFusion, self).__init__()
        
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim,
            nhead=num_heads,
            dim_feedforward=embed_dim * 4,
            dropout=dropout,
            activation="gelu",
            batch_first=True
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=encoder_depth)
        
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.norm = nn.LayerNorm(embed_dim)

    def forward(self, attended_cnn_tokens_list: list[torch.Tensor], swin_tokens: torch.Tensor) -> torch.Tensor:
        B = swin_tokens.shape[0]
        
        # Mean pool each sequence down to 1 token per scale
        pooled_cnn = [tokens.mean(dim=1, keepdim=True) for tokens in attended_cnn_tokens_list]
        pooled_swin = swin_tokens.mean(dim=1, keepdim=True)
        
        # Expand CLS token
        cls_tokens = self.cls_token.expand(B, -1, -1)
        
        # Sequence: [CLS, Swin, Stage1, Stage2, Stage3, Stage4] -> length 6
        fusion_sequence = torch.cat([cls_tokens, pooled_swin] + pooled_cnn, dim=1) 
        
        # Self-Attention across scales
        encoded_sequence = self.transformer_encoder(fusion_sequence)
        
        # Extract fused CLS token
        cls_out = self.norm(encoded_sequence[:, 0, :])
        
        return cls_out
