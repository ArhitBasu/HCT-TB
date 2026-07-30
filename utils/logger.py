import logging
import sys
import os

def get_logger(name: str, log_dir: str, experiment_name: str) -> logging.Logger:
    """
    1. Purpose: Provides dual-channel logging (Console + File) for experiment tracking.
    2. Mathematical Intuition: N/A
    3. Input Tensor Shape: N/A. Output Tensor Shape: N/A.
    4. IEEE Baseline Improvement: Replaces ephemeral 'print' statements with persistent, parseable training histories.
    5. Benefit for TB Detection: Crucial for long-running medical dataset trainings, allowing researchers to track overnight epoch losses safely.
    """
    logger = logging.Logger(name)
    logger.setLevel(logging.INFO)
    
    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    
    # Console Handler
    ch = logging.StreamHandler(sys.stdout)
    ch.setFormatter(formatter)
    logger.addHandler(ch)
    
    # File Handler
    if log_dir and experiment_name:
        os.makedirs(log_dir, exist_ok=True)
        log_file = os.path.join(log_dir, f"{experiment_name}.log")
        fh = logging.FileHandler(log_file)
        fh.setFormatter(formatter)
        logger.addHandler(fh)
        
    return logger
