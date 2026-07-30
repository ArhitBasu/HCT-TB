import torch
from models.main_architecture import ROIGuidedHybridNetwork

def test_full_forward_pass():
    """
    1. Purpose: Verifies the integrity of the frozen V3 architecture and tensor routing.
    2. Mathematical Intuition: Validates that the FPN interpolation and Transformer projection dimensions algebraically align.
    3. Input Tensor Shape: (2, 3, 224, 224). Output Tensor Shape: (2, 3).
    4. IEEE Baseline Improvement: Test-Driven Development (TDD) approach ensures research code is free of silent tensor dimension mismatches.
    5. Benefit for TB Detection: Ensures the computational graph is intact before beginning expensive, long-running medical image training.
    """
    print("Initializing V3 Research Framework Architecture...")
    
    config = {
        'pretrained': False, # Set false for offline testing
        'classes': 3,
        'attention_heads': 4,
        'dropout': 0.1
    }
    
    try:
        model = ROIGuidedHybridNetwork(config)
        model.eval()
        
        # Create a dummy batch of 2 RGB images (224x224)
        dummy_input = torch.randn(2, 3, 224, 224)
        
        print("Executing forward pass...")
        with torch.no_grad():
            outputs = model(dummy_input)
            
        print(f"Forward pass successful! Output shape: {outputs.shape}")
        
        assert outputs.shape == (2, 3), f"Expected (2, 3), got {outputs.shape}"
        print("Tensor routing verification passed.")
        
    except Exception as e:
        print(f"Architecture Test Failed! Error: {str(e)}")
        raise e

if __name__ == "__main__":
    test_full_forward_pass()
