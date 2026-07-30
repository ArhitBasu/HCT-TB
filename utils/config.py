import yaml
import os

class ConfigParser:
    """
    1. Purpose: Parses YAML configuration files to manage experiment hyperparameters dynamically.
    2. Mathematical Intuition: Represents a mapping function f: YAML -> Dict for immutable configuration state.
    3. Input Tensor Shape: N/A (String Path). Output Tensor Shape: N/A (Python Dictionary).
    4. IEEE Baseline Improvement: Replaces hardcoded training scripts with a scalable experiment tracking registry.
    5. Benefit for TB Detection: Enables rapid configuration sweeps (e.g., trying focal loss vs CE) to find the optimal setup for identifying TB lesions.
    """
    
    @staticmethod
    def load_config(config_path: str) -> dict:
        """Loads a YAML configuration file into a dictionary."""
        if not os.path.exists(config_path):
            raise FileNotFoundError(f"Configuration file not found: {config_path}")
            
        with open(config_path, 'r') as file:
            config = yaml.safe_load(file)
            
        return config
        
    @staticmethod
    def save_config(config: dict, save_path: str):
        """Saves a configuration dictionary to a YAML file for reproducibility."""
        with open(save_path, 'w') as file:
            yaml.dump(config, file, default_flow_style=False)
