import argparse
import os
import datetime
import shutil

from utils.config import ConfigParser
from utils.seed import seed_everything
from utils.logger import get_logger
from models.main_architecture import ROIGuidedHybridNetwork
from training.trainer import ModelTrainer
from losses.focal_loss import FocalLoss
from losses.label_smoothing import LabelSmoothingLoss
import torch.nn as nn

def main():
    """
    1. Purpose: Entry point for the framework. Parses YAML, initializes the registry, and launches training.
    2. Mathematical Intuition: N/A
    3. Input Tensor Shape: N/A. Output Tensor Shape: N/A.
    4. IEEE Baseline Improvement: Creates a fully reproducible experiment registry instead of ad-hoc Jupyter notebooks.
    5. Benefit for TB Detection: Ensures that the results submitted to the journal (IEEE Access / CBM) are perfectly reproducible simply by sharing the generated configuration file.
    """
    parser = argparse.ArgumentParser(description="TB Detection Framework V3")
    parser.add_argument('--config', type=str, required=True, help="Path to YAML config file")
    args = parser.parse_args()
    
    # 1. Load Configuration
    config = ConfigParser.load_config(args.config)
    
    # 2. Setup Experiment Registry
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M")
    exp_name = config.get('experiment_name', 'baseline')
    experiment_dir = os.path.join('experiments', f"{timestamp}_{exp_name}")
    os.makedirs(experiment_dir, exist_ok=True)
    
    # 3. Save a copy of the config inside the registry for reproducibility
    ConfigParser.save_config(config, os.path.join(experiment_dir, 'config.yaml'))
    
    # 4. Initialize Logger
    logger = get_logger("TB_Framework", os.path.join(experiment_dir, 'logs'), exp_name)
    logger.info(f"Initialized Experiment: {experiment_dir}")
    
    # 5. Seed Everything
    seed_everything(config.get('seed', 42))
    
    # 6. Initialize Architecture
    logger.info("Initializing Frozen Architecture...")
    model = ROIGuidedHybridNetwork(config)
    
    # 7. Loss Selection
    loss_type = config.get('loss_function', 'cross_entropy')
    if loss_type == 'focal':
        criterion = FocalLoss(gamma=2.0)
    elif loss_type == 'label_smoothing':
        criterion = LabelSmoothingLoss(num_classes=config.get('classes', 3), smoothing=0.1)
    else:
        criterion = nn.CrossEntropyLoss()
        
    logger.info(f"Selected Loss Function: {loss_type}")
    
    # 8. DataLoaders (Mocked for pipeline architecture)
    # train_loader, val_loader = get_dataloaders(config)
    train_loader = [] 
    val_loader = []
    
    # 9. Initialize Trainer
    # trainer = ModelTrainer(model, train_loader, val_loader, criterion, config, experiment_dir, logger)
    # trainer.fit()
    
    logger.info("Framework successfully booted! Waiting for dataloaders to commence training.")

if __name__ == "__main__":
    main()
