import torch
import numpy as np
import random
import os

def seed_everything(seed: int = 42):
    """
    1. Purpose: Enforces deterministic behavior across all pseudo-random number generators (PRNG).
    2. Mathematical Intuition: Initializes the seed state S_0 such that the sequence of generated numbers S_t is identical across runs.
    3. Input Tensor Shape: N/A (Integer Seed). Output Tensor Shape: N/A.
    4. IEEE Baseline Improvement: Solves the reproducibility crisis commonly found in baseline papers.
    5. Benefit for TB Detection: Medical imaging research mandates strict determinism to ensure that reported accuracy improvements are architectural and not due to lucky weight initialization.
    """
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    
    # Ensures deterministic convolution algorithms
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    
    print(f"Global seed set to {seed} for reproducibility.")
