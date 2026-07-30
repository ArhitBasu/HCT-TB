import torch
import time
from models.main_architecture import ROIGuidedHybridNetwork

def test_architecture():
    print("Initializing ROI-Guided Hierarchical Multi-Scale Cross-Attention Network...")
    
    # Initialize the model without pretrained weights to save time for testing
    # In practice, you will set pretrained=True
    model = ROIGuidedHybridNetwork(pretrained=False, num_classes=3)
    model.eval()
    
    print("Model initialized successfully!")
    
    # Calculate parameter count
    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total Trainable Parameters: {total_params:,}")
    
    # Simulate a batch of 2 ROI-segmented Lung X-Rays
    batch_size = 2
    dummy_input = torch.randn(batch_size, 3, 224, 224)
    print(f"\nPassing dummy tensor of shape {dummy_input.shape} through the network...")
    
    # Time the forward pass
    start_time = time.time()
    with torch.no_grad():
        logits = model(dummy_input)
    end_time = time.time()
    
    print(f"\nForward Pass Successful!")
    print(f"Output Logits Shape: {logits.shape} (Expected: {batch_size}, 3)")
    print(f"Inference Time for batch of {batch_size}: {(end_time - start_time) * 1000:.2f} ms")

if __name__ == "__main__":
    test_architecture()
