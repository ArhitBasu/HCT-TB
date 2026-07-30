import torch
import torch.nn as nn

class FeatureFusion(nn.Module):
    """
    Final Feature Fusion Module.
    
    Takes the 4 Cross-Attended CNN Token sequences (Stage 1 to 4) and the 
    original Swin Tokens, condenses them into representative semantic vectors, 
    and passes them through a final Transformer Encoder.
    
    This fulfills the requirement of deep fusion, ensuring that the model 
    holistically understands the relationship between the 4 scales of 
    anomalies before making a final classification.
    """
    def __init__(self, embed_dim: int, num_heads: int = 8, encoder_depth: int = 1):
        super(FeatureFusion, self).__init__()
        
        # We use a standard Transformer Encoder for the final fusion
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim,
            nhead=num_heads,
            dim_feedforward=embed_dim * 4,
            dropout=0.1,
            activation="gelu",
            batch_first=True
        )
        
        self.transformer_encoder = nn.TransformerEncoder(
            encoder_layer,
            num_layers=encoder_depth
        )
        
        # We append a learnable CLS token to aggregate the final representation
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        
        self.norm = nn.LayerNorm(embed_dim)

    def forward(self, attended_cnn_tokens_list: list[torch.Tensor], swin_tokens: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.
        
        Args:
            attended_cnn_tokens_list (list[torch.Tensor]): 4 tensors of (B, N_i, D)
            swin_tokens (torch.Tensor): (B, N_swin, D)
            
        Returns:
            torch.Tensor: A single unified feature vector of shape (B, D)
        """
        B = swin_tokens.shape[0]
        
        # 1. Global Average Pool each sequence to extract its core semantic essence
        # This reduces (B, 3136, D) -> (B, 1, D) for each scale
        pooled_cnn = []
        for tokens in attended_cnn_tokens_list:
            pooled_cnn.append(tokens.mean(dim=1, keepdim=True))
            
        pooled_swin = swin_tokens.mean(dim=1, keepdim=True)
        
        # 2. Expand CLS token to match batch size
        cls_tokens = self.cls_token.expand(B, -1, -1)
        
        # 3. Concatenate all pooled vectors into a sequence of length 6
        # Sequence: [CLS, Swin_Pool, CNN_Stage1_Pool, CNN_Stage2_Pool, CNN_Stage3_Pool, CNN_Stage4_Pool]
        fusion_sequence = torch.cat([cls_tokens, pooled_swin] + pooled_cnn, dim=1) # (B, 6, D)
        
        # 4. Pass through Transformer Encoder
        encoded_sequence = self.transformer_encoder(fusion_sequence)
        
        # 5. Extract the CLS token which now contains the fused global representation
        cls_out = encoded_sequence[:, 0, :] # (B, D)
        
        # 6. Final LayerNorm
        cls_out = self.norm(cls_out)
        
        return cls_out

if __name__ == "__main__":
    B, D = 2, 768
    model = FeatureFusion(embed_dim=D)
    
    # Mock inputs
    attended_cnn = [
        torch.randn(B, 3136, D),
        torch.randn(B, 784, D),
        torch.randn(B, 196, D),
        torch.randn(B, 49, D)
    ]
    swin_tokens = torch.randn(B, 49, D)
    
    fused_rep = model(attended_cnn, swin_tokens)
    print(f"Fused Representation Shape: {fused_rep.shape}") # Should be (2, 768)
