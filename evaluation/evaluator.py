import os
import json
import csv
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import roc_curve, auc, precision_recall_curve

from evaluation.metrics import MedicalMetrics

class Evaluator:
    """
    1. Purpose: Autonomous Result Generation engine.
    2. Mathematical Intuition: Translates raw probability distributions into clinical performance curves (ROC, PR).
    3. Input Tensor Shape: Labels and Probabilities. Output Tensor Shape: Saved .png and .csv artifacts.
    4. IEEE Baseline Improvement: Eliminates manual plotting. Automatically generates publication-ready artifacts inside the experiment registry.
    5. Benefit for TB Detection: Reviewers require these exact curves to validate the clinical utility of the model.
    """
    def __init__(self, results_dir: str, class_names: list = ['Healthy', 'Non-TB', 'TB']):
        self.results_dir = results_dir
        self.class_names = class_names
        self.num_classes = len(class_names)
        
    def evaluate(self, labels, preds, probs):
        metrics, cm = MedicalMetrics.compute_all(labels, preds, probs, self.num_classes)
        
        # Save Metrics
        self._save_metrics(metrics)
        
        # Generate Plots
        self._plot_confusion_matrix(cm)
        self._plot_roc_curve(labels, probs)
        self._plot_pr_curve(labels, probs)
        
        return metrics

    def _save_metrics(self, metrics):
        # JSON
        with open(os.path.join(self.results_dir, 'metrics.json'), 'w') as f:
            json.dump(metrics, f, indent=4)
        
        # CSV
        with open(os.path.join(self.results_dir, 'metrics.csv'), 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['Metric', 'Value'])
            for k, v in metrics.items():
                writer.writerow([k, f"{v:.4f}"])

    def _plot_confusion_matrix(self, cm):
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                    xticklabels=self.class_names, yticklabels=self.class_names)
        plt.ylabel('Actual Label')
        plt.xlabel('Predicted Label')
        plt.title('Confusion Matrix')
        plt.tight_layout()
        plt.savefig(os.path.join(self.results_dir, 'confusion_matrix.png'), dpi=300)
        plt.close()

    def _plot_roc_curve(self, labels, probs):
        plt.figure(figsize=(8, 6))
        labels_one_hot = np.eye(self.num_classes)[labels]
        
        for i in range(self.num_classes):
            fpr, tpr, _ = roc_curve(labels_one_hot[:, i], probs[:, i])
            roc_auc = auc(fpr, tpr)
            plt.plot(fpr, tpr, lw=2, label=f'{self.class_names[i]} (area = {roc_auc:.2f})')
            
        plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('Receiver Operating Characteristic (ROC)')
        plt.legend(loc="lower right")
        plt.savefig(os.path.join(self.results_dir, 'roc_curve.png'), dpi=300)
        plt.close()

    def _plot_pr_curve(self, labels, probs):
        plt.figure(figsize=(8, 6))
        labels_one_hot = np.eye(self.num_classes)[labels]
        
        for i in range(self.num_classes):
            precision, recall, _ = precision_recall_curve(labels_one_hot[:, i], probs[:, i])
            pr_auc = auc(recall, precision)
            plt.plot(recall, precision, lw=2, label=f'{self.class_names[i]} (area = {pr_auc:.2f})')
            
        plt.xlabel('Recall')
        plt.ylabel('Precision')
        plt.title('Precision-Recall Curve')
        plt.legend(loc="lower left")
        plt.savefig(os.path.join(self.results_dir, 'precision_recall_curve.png'), dpi=300)
        plt.close()
