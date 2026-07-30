import os
import json
import yaml
from datetime import datetime

class ExperimentReporter:
    """
    1. Purpose: Automates the generation of a publication-ready Markdown report containing all metrics, curves, and configurations.
    2. Mathematical Intuition: N/A.
    3. Input Tensor Shape: N/A. Output Tensor Shape: Generates experiment_report.md.
    4. IEEE Baseline Improvement: Eliminates the possibility of manual transcription errors when copying results from notebooks to the final dissertation manuscript.
    5. Benefit for TB Detection: Ensures clinical transparency by perfectly logging every hyperparameter alongside the resulting ROC/PR curves.
    """
    @staticmethod
    def generate_report(experiment_dir: str):
        report_path = os.path.join(experiment_dir, 'experiment_report.md')
        
        # Load configs and metrics
        config_path = os.path.join(experiment_dir, 'config.yaml')
        metrics_path = os.path.join(experiment_dir, 'results', 'metrics.json')
        
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
            
        with open(metrics_path, 'r') as f:
            metrics = json.load(f)
            
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M")
        
        md_content = f"# Experiment Report\n"
        md_content += f"**Generated:** {timestamp}\n"
        md_content += f"**Experiment Name:** {config.get('experiment_name', 'Unnamed')}\n\n"
        
        md_content += "## 1. Hyperparameters\n```yaml\n"
        md_content += yaml.dump(config, default_flow_style=False)
        md_content += "```\n\n"
        
        md_content += "## 2. Evaluation Metrics\n"
        md_content += "| Metric | Score |\n|---|---|\n"
        for k, v in metrics.items():
            md_content += f"| {k} | {v:.4f} |\n"
            
        md_content += "\n## 3. Visual Results\n"
        md_content += "### Confusion Matrix\n"
        md_content += f"![Confusion Matrix](results/confusion_matrix.png)\n\n"
        md_content += "### ROC Curve\n"
        md_content += f"![ROC Curve](results/roc_curve.png)\n\n"
        md_content += "### PR Curve\n"
        md_content += f"![PR Curve](results/precision_recall_curve.png)\n\n"
        
        md_content += "## 4. Conclusion\n"
        md_content += "_Add manual qualitative analysis of the Explainability heatmaps here._\n"
        
        with open(report_path, 'w') as f:
            f.write(md_content)
            
        print(f"Experiment report generated successfully at: {report_path}")
