import torch
import torch.nn as nn

class ModelProfiler:
    """
    1. Purpose: Computes parameter counts, estimated model size, and trainable parameters.
    2. Mathematical Intuition: Sum(w_i) for all w in Theta. Size = Params * 4 bytes (Float32).
    3. Input Tensor Shape: nn.Module. Output Tensor Shape: Dict of metrics.
    4. IEEE Baseline Improvement: Transparently reports computational overhead.
    5. Benefit for TB Detection: Proves that the hierarchical cross-attention does not make the model too heavy for deployment in resource-constrained clinics.
    """
    
    @staticmethod
    def profile_model(model: nn.Module) -> dict:
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        
        # Estimate size in Megabytes (Assuming Float32 = 4 bytes)
        model_size_mb = (total_params * 4) / (1024 ** 2)
        
        return {
            "Total Parameters": total_params,
            "Trainable Parameters": trainable_params,
            "Model Size (MB)": round(model_size_mb, 2)
        }
