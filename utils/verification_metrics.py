import numpy as np
import torch
from sklearn.metrics import roc_curve, auc, precision_recall_curve
from scipy.optimize import brentq
from scipy.interpolate import interp1d
import matplotlib.pyplot as plt


class VerificationMetrics:
    """
    Verification metrics for twin face verification task.
    
    Computes EER, AUC, ROC, TAR/FAR, and verification accuracy
    focusing on positive pairs (same person) and hard-negative pairs (twin pairs).
    """
    
    def __init__(self):
        self.reset()
    
    def reset(self):
        """Reset all metrics."""
        self.scores = []
        self.labels = []
        self.person_ids = []
        self.twin_ids = []
    
    def update(self, scores, labels, person_ids=None, twin_ids=None):
        """
        Update metrics with new batch of predictions.
        
        Args:
            scores: Similarity scores [batch_size]
            labels: Binary labels (1 for same person, 0 for twin) [batch_size]
            person_ids: Person IDs for analysis [batch_size]
            twin_ids: Twin IDs for analysis [batch_size]
        """
        if isinstance(scores, torch.Tensor):
            scores = scores.detach().cpu().numpy()
        if isinstance(labels, torch.Tensor):
            labels = labels.detach().cpu().numpy()
        
        self.scores.extend(scores)
        self.labels.extend(labels)
        
        if person_ids is not None:
            self.person_ids.extend(person_ids)
        if twin_ids is not None:
            self.twin_ids.extend(twin_ids)
    
    def compute_eer(self):
        """
        Compute Equal Error Rate (EER).
        
        Returns:
            eer: Equal Error Rate
            threshold: Threshold at EER
        """
        if len(self.scores) == 0:
            return 0.0, 0.0
        
        scores = np.array(self.scores)
        labels = np.array(self.labels)
        
        # Compute ROC curve
        fpr, tpr, thresholds = roc_curve(labels, scores)
        
        # Find EER (where FPR = 1 - TPR)
        fnr = 1 - tpr
        eer_threshold = thresholds[np.nanargmin(np.absolute(fnr - fpr))]
        eer = fpr[np.nanargmin(np.absolute(fnr - fpr))]
        
        return eer, eer_threshold
    
    def compute_auc(self):
        """
        Compute Area Under ROC Curve (AUC).
        
        Returns:
            auc_score: AUC value
        """
        if len(self.scores) == 0:
            return 0.0
        
        scores = np.array(self.scores)
        labels = np.array(self.labels)
        
        # Compute AUC
        auc_score = auc(roc_curve(labels, scores)[0], roc_curve(labels, scores)[1])
        
        return auc_score
    
    def compute_tar_far(self, threshold=None):
        """
        Compute True Accept Rate (TAR) and False Accept Rate (FAR) at given threshold.
        
        Args:
            threshold: Decision threshold (if None, use EER threshold)
            
        Returns:
            tar: True Accept Rate
            far: False Accept Rate
            threshold: Used threshold
        """
        if len(self.scores) == 0:
            return 0.0, 0.0, 0.0
        
        scores = np.array(self.scores)
        labels = np.array(self.labels)
        
        if threshold is None:
            _, threshold = self.compute_eer()
        
        # Compute TAR and FAR
        predictions = (scores >= threshold).astype(int)
        
        # True Accept Rate (TPR)
        tar = np.sum((predictions == 1) & (labels == 1)) / np.sum(labels == 1)
        
        # False Accept Rate (FPR)
        far = np.sum((predictions == 1) & (labels == 0)) / np.sum(labels == 0)
        
        return tar, far, threshold
    
    def compute_verification_accuracy(self, threshold=None):
        """
        Compute verification accuracy at given threshold.
        
        Args:
            threshold: Decision threshold (if None, use EER threshold)
            
        Returns:
            accuracy: Verification accuracy
        """
        if len(self.scores) == 0:
            return 0.0
        
        scores = np.array(self.scores)
        labels = np.array(self.labels)
        
        if threshold is None:
            _, threshold = self.compute_eer()
        
        predictions = (scores >= threshold).astype(int)
        accuracy = np.mean(predictions == labels)
        
        return accuracy
    
    def compute_precision_recall(self):
        """
        Compute precision and recall.
        
        Returns:
            precision: Precision value
            recall: Recall value
        """
        if len(self.scores) == 0:
            return 0.0, 0.0
        
        scores = np.array(self.scores)
        labels = np.array(self.labels)
        
        precision, recall, _ = precision_recall_curve(labels, scores)
        
        # Return precision and recall at EER threshold
        _, threshold = self.compute_eer()
        predictions = (scores >= threshold).astype(int)
        
        tp = np.sum((predictions == 1) & (labels == 1))
        fp = np.sum((predictions == 1) & (labels == 0))
        fn = np.sum((predictions == 0) & (labels == 1))
        
        precision_val = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall_val = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        
        return precision_val, recall_val
    
    def get_all_metrics(self):
        """
        Compute all verification metrics.
        
        Returns:
            metrics: Dictionary containing all metrics
        """
        if len(self.scores) == 0:
            return {
                'eer': 0.0,
                'auc': 0.0,
                'tar': 0.0,
                'far': 0.0,
                'accuracy': 0.0,
                'precision': 0.0,
                'recall': 0.0,
                'threshold': 0.0
            }
        
        eer, threshold = self.compute_eer()
        auc_score = self.compute_auc()
        tar, far, _ = self.compute_tar_far(threshold)
        accuracy = self.compute_verification_accuracy(threshold)
        precision, recall = self.compute_precision_recall()
        
        metrics = {
            'eer': eer,
            'auc': auc_score,
            'tar': tar,
            'far': far,
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'threshold': threshold,
            'num_samples': len(self.scores),
            'num_positive': np.sum(self.labels),
            'num_negative': len(self.labels) - np.sum(self.labels)
        }
        
        return metrics
    
    def plot_roc_curve(self, save_path=None):
        """
        Plot ROC curve.
        
        Args:
            save_path: Path to save the plot
        """
        if len(self.scores) == 0:
            return
        
        scores = np.array(self.scores)
        labels = np.array(self.labels)
        
        fpr, tpr, thresholds = roc_curve(labels, scores)
        auc_score = auc(fpr, tpr)
        
        plt.figure(figsize=(8, 6))
        plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (AUC = {auc_score:.3f})')
        plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('Receiver Operating Characteristic (ROC) Curve')
        plt.legend(loc="lower right")
        plt.grid(True)
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        else:
            plt.show()
        
        plt.close()
    
    def plot_precision_recall_curve(self, save_path=None):
        """
        Plot Precision-Recall curve.
        
        Args:
            save_path: Path to save the plot
        """
        if len(self.scores) == 0:
            return
        
        scores = np.array(self.scores)
        labels = np.array(self.labels)
        
        precision, recall, _ = precision_recall_curve(labels, scores)
        
        plt.figure(figsize=(8, 6))
        plt.plot(recall, precision, color='blue', lw=2)
        plt.xlabel('Recall')
        plt.ylabel('Precision')
        plt.title('Precision-Recall Curve')
        plt.grid(True)
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        else:
            plt.show()
        
        plt.close()


def compute_verification_metrics_batch(scores, labels, person_ids=None, twin_ids=None):
    """
    Compute verification metrics for a single batch.
    
    Args:
        scores: Similarity scores [batch_size]
        labels: Binary labels [batch_size]
        person_ids: Person IDs [batch_size]
        twin_ids: Twin IDs [batch_size]
        
    Returns:
        metrics: Dictionary with batch metrics
    """
    metrics_calculator = VerificationMetrics()
    metrics_calculator.update(scores, labels, person_ids, twin_ids)
    return metrics_calculator.get_all_metrics() 