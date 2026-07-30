import numpy as np
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score, 
    roc_auc_score, confusion_matrix, cohen_kappa_score, 
    matthews_corrcoef, balanced_accuracy_score, average_precision_score
)

class MedicalMetrics:
    """
    1. Purpose: Computes an exhaustive suite of clinical metrics.
    2. Mathematical Intuition: Computes MCC, Kappa, PR-AUC, and Specificity which are heavily penalized by false positives/negatives in imbalanced distributions.
    3. Input Tensor Shape: 1D numpy arrays for labels and predictions. Output Tensor Shape: Dictionary.
    4. IEEE Baseline Improvement: Standard baselines use Accuracy. Medical imaging requires MCC and PR-AUC.
    5. Benefit for TB Detection: Reliably identifies if the model is just guessing 'Healthy' to achieve 90% accuracy on imbalanced data.
    """
    @staticmethod
    def compute_all(labels, preds, probs, num_classes=3):
        metrics = {}
        metrics['Accuracy'] = accuracy_score(labels, preds)
        metrics['Balanced_Accuracy'] = balanced_accuracy_score(labels, preds)
        metrics['Precision_Macro'] = precision_score(labels, preds, average='macro', zero_division=0)
        metrics['Recall_Macro'] = recall_score(labels, preds, average='macro', zero_division=0)
        metrics['F1_Macro'] = f1_score(labels, preds, average='macro', zero_division=0)
        metrics['Cohen_Kappa'] = cohen_kappa_score(labels, preds)
        metrics['MCC'] = matthews_corrcoef(labels, preds)
        
        # Specificity
        cm = confusion_matrix(labels, preds, labels=list(range(num_classes)))
        specificities = []
        for i in range(num_classes):
            tn = np.sum(np.delete(np.delete(cm, i, axis=0), i, axis=1))
            fp = np.sum(np.delete(cm[:, i], i))
            spec = tn / (tn + fp) if (tn + fp) > 0 else 0
            specificities.append(spec)
        metrics['Specificity_Macro'] = np.mean(specificities)
        
        # ROC-AUC & PR-AUC
        try:
            metrics['ROC_AUC_Macro'] = roc_auc_score(labels, probs, multi_class='ovr', average='macro')
            
            # For PR-AUC, we need to one-hot encode labels
            labels_one_hot = np.eye(num_classes)[labels]
            metrics['PR_AUC_Macro'] = average_precision_score(labels_one_hot, probs, average='macro')
        except ValueError:
            metrics['ROC_AUC_Macro'] = -1.0
            metrics['PR_AUC_Macro'] = -1.0
            
        return metrics, cm
