import torch
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import roc_auc_score, accuracy_score, precision_recall_curve, average_precision_score
from sklearn.metrics import roc_curve, confusion_matrix, classification_report
from typing import Dict, List, Tuple, Optional, Union, Any
import seaborn as sns
from pathlib import Path
import json


def convert_numpy_types(obj):
    """Convert NumPy types to native Python types for JSON serialization"""
    if isinstance(obj, np.integer):
        return int(obj)
    elif isinstance(obj, np.floating):
        return float(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, dict):
        return {key: convert_numpy_types(value) for key, value in obj.items()}
    elif isinstance(obj, list):
        return [convert_numpy_types(item) for item in obj]
    else:
        return obj


class TwinVerificationMetrics:
    """
    Metrics for twin verification evaluation
    """
    
    def __init__(self, thresholds: Optional[List[float]] = None):
        """
        Args:
            thresholds: List of thresholds to evaluate
        """
        if thresholds is None:
            thresholds = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
        self.thresholds = thresholds
        
    def compute_metrics(self, 
                       similarity_scores: Union[torch.Tensor, np.ndarray],
                       labels: Union[torch.Tensor, np.ndarray],
                       threshold: Optional[float] = None) -> Dict[str, float]:
        """
        Compute verification metrics
        
        Args:
            similarity_scores: Similarity scores between pairs
            labels: Binary labels (1 for same person, 0 for different)
            threshold: Decision threshold (if None, find optimal)
            
        Returns:
            Dictionary of metrics
        """
        # Convert to numpy if needed
        if isinstance(similarity_scores, torch.Tensor):
            similarity_scores = similarity_scores.cpu().numpy()
        if isinstance(labels, torch.Tensor):
            labels = labels.cpu().numpy()
            
        # Find optimal threshold if not provided
        if threshold is None:
            threshold = self.find_optimal_threshold(similarity_scores, labels)
            
        # Make predictions
        predictions = (similarity_scores > threshold).astype(int)
        
        # Compute metrics
        metrics = {}
        
        # Basic metrics
        metrics['accuracy'] = accuracy_score(labels, predictions)
        metrics['auc'] = roc_auc_score(labels, similarity_scores)
        metrics['average_precision'] = average_precision_score(labels, similarity_scores)
        metrics['threshold'] = threshold
        
        # Confusion matrix metrics
        tn, fp, fn, tp = confusion_matrix(labels, predictions).ravel()
        metrics['true_negatives'] = tn
        metrics['false_positives'] = fp
        metrics['false_negatives'] = fn
        metrics['true_positives'] = tp
        
        # Derived metrics
        metrics['precision'] = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        metrics['recall'] = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        metrics['f1_score'] = 2 * (metrics['precision'] * metrics['recall']) / (metrics['precision'] + metrics['recall']) if (metrics['precision'] + metrics['recall']) > 0 else 0.0
        
        # Verification-specific metrics
        metrics['far'] = fp / (fp + tn) if (fp + tn) > 0 else 0.0  # False Accept Rate
        metrics['frr'] = fn / (fn + tp) if (fn + tp) > 0 else 0.0  # False Reject Rate
        metrics['tar'] = tp / (tp + fn) if (tp + fn) > 0 else 0.0  # True Accept Rate
        metrics['trr'] = tn / (tn + fp) if (tn + fp) > 0 else 0.0  # True Reject Rate
        
        # Equal Error Rate (EER)
        metrics['eer'] = self.compute_eer(similarity_scores, labels)
        
        return metrics
        
    def find_optimal_threshold(self, 
                              similarity_scores: np.ndarray,
                              labels: np.ndarray,
                              metric: str = 'f1_score') -> float:
        """
        Find optimal threshold based on specified metric
        
        Args:
            similarity_scores: Similarity scores
            labels: Binary labels
            metric: Metric to optimize ('f1_score', 'accuracy', 'eer')
            
        Returns:
            Optimal threshold
        """
        best_threshold = 0.5
        best_score = 0.0
        
        for threshold in self.thresholds:
            predictions = (similarity_scores > threshold).astype(int)
            
            if metric == 'f1_score':
                score = self._compute_f1_score(labels, predictions)
            elif metric == 'accuracy':
                score = accuracy_score(labels, predictions)
            elif metric == 'eer':
                score = -self.compute_eer(similarity_scores, labels)  # Negative because we want to minimize EER
            else:
                raise ValueError(f"Unknown metric: {metric}")
                
            if score > best_score:
                best_score = score
                best_threshold = threshold
                
        return best_threshold
        
    def compute_eer(self, similarity_scores: np.ndarray, labels: np.ndarray) -> float:
        """
        Compute Equal Error Rate (EER)
        
        Args:
            similarity_scores: Similarity scores
            labels: Binary labels
            
        Returns:
            Equal Error Rate
        """
        fpr, tpr, thresholds = roc_curve(labels, similarity_scores)
        fnr = 1 - tpr
        
        # Find threshold where FPR = FNR
        eer_threshold_idx = np.argmin(np.abs(fpr - fnr))
        eer = (fpr[eer_threshold_idx] + fnr[eer_threshold_idx]) / 2
        
        return eer
        
    def _compute_f1_score(self, labels: np.ndarray, predictions: np.ndarray) -> float:
        """Compute F1 score"""
        tn, fp, fn, tp = confusion_matrix(labels, predictions).ravel()
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0
        return f1
        
    def compute_tar_at_far(self, 
                          similarity_scores: np.ndarray,
                          labels: np.ndarray,
                          target_far: float = 0.01) -> float:
        """
        Compute True Accept Rate at specific False Accept Rate
        
        Args:
            similarity_scores: Similarity scores
            labels: Binary labels
            target_far: Target False Accept Rate
            
        Returns:
            True Accept Rate at target FAR
        """
        fpr, tpr, thresholds = roc_curve(labels, similarity_scores)
        
        # Find threshold closest to target FAR
        far_idx = np.argmin(np.abs(fpr - target_far))
        tar_at_far = tpr[far_idx]
        
        return tar_at_far


class TwinVerificationEvaluator:
    """
    Comprehensive evaluator for twin verification
    """
    
    def __init__(self, save_dir: str = "./evaluation_results"):
        """
        Args:
            save_dir: Directory to save evaluation results
        """
        self.save_dir = Path(save_dir)
        self.save_dir.mkdir(exist_ok=True)
        self.metrics_calculator = TwinVerificationMetrics()
        
    def evaluate_model(self,
                      model,
                      test_loader,
                      device: str = 'cuda',
                      save_results: bool = True) -> Dict[str, Any]:
        """
        Evaluate model on test set
        
        Args:
            model: Twin verification model
            test_loader: Test data loader
            device: Device to use
            save_results: Whether to save results
            
        Returns:
            Dictionary of evaluation results
        """
        model.eval()
        all_similarities = []
        all_labels = []
        all_predictions = []
        
        with torch.no_grad():
            for img1, img2, labels in test_loader:
                img1 = img1.to(device)
                img2 = img2.to(device)
                labels = labels.to(device)
                
                # Forward pass
                embeddings1, embeddings2, similarity_scores = model(img1, img2, training=False)
                
                # Store results
                all_similarities.extend(similarity_scores.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())
                
        # Convert to numpy arrays
        all_similarities = np.array(all_similarities)
        all_labels = np.array(all_labels)
        
        # Compute metrics
        metrics = self.metrics_calculator.compute_metrics(all_similarities, all_labels)
        
        # Generate plots
        plots = self.generate_evaluation_plots(all_similarities, all_labels)
        
        # Prepare results
        results = {
            'metrics': metrics,
            'similarity_scores': all_similarities,
            'labels': all_labels,
            'plots': plots
        }
        
        # Save results
        if save_results:
            self.save_evaluation_results(results)
            
        return results
        
    def generate_evaluation_plots(self,
                                 similarity_scores: np.ndarray,
                                 labels: np.ndarray) -> Dict[str, plt.Figure]:
        """
        Generate evaluation plots
        
        Args:
            similarity_scores: Similarity scores
            labels: Binary labels
            
        Returns:
            Dictionary of matplotlib figures
        """
        plots = {}
        
        # ROC Curve
        fig_roc, ax_roc = plt.subplots(figsize=(8, 6))
        fpr, tpr, thresholds = roc_curve(labels, similarity_scores)
        auc = roc_auc_score(labels, similarity_scores)
        
        ax_roc.plot(fpr, tpr, label=f'ROC Curve (AUC = {auc:.3f})')
        ax_roc.plot([0, 1], [0, 1], 'k--', label='Random')
        ax_roc.set_xlabel('False Positive Rate')
        ax_roc.set_ylabel('True Positive Rate')
        ax_roc.set_title('ROC Curve')
        ax_roc.legend()
        ax_roc.grid(True)
        plots['roc_curve'] = fig_roc
        
        # Precision-Recall Curve
        fig_pr, ax_pr = plt.subplots(figsize=(8, 6))
        precision, recall, _ = precision_recall_curve(labels, similarity_scores)
        ap = average_precision_score(labels, similarity_scores)
        
        ax_pr.plot(recall, precision, label=f'PR Curve (AP = {ap:.3f})')
        ax_pr.set_xlabel('Recall')
        ax_pr.set_ylabel('Precision')
        ax_pr.set_title('Precision-Recall Curve')
        ax_pr.legend()
        ax_pr.grid(True)
        plots['pr_curve'] = fig_pr
        
        # Similarity Score Distribution
        fig_dist, ax_dist = plt.subplots(figsize=(10, 6))
        
        same_person_scores = similarity_scores[labels == 1]
        different_person_scores = similarity_scores[labels == 0]
        
        ax_dist.hist(same_person_scores, bins=50, alpha=0.7, label='Same Person', density=True)
        ax_dist.hist(different_person_scores, bins=50, alpha=0.7, label='Different Person', density=True)
        ax_dist.set_xlabel('Similarity Score')
        ax_dist.set_ylabel('Density')
        ax_dist.set_title('Similarity Score Distribution')
        ax_dist.legend()
        ax_dist.grid(True)
        plots['score_distribution'] = fig_dist
        
        # Threshold Analysis
        fig_thresh, ax_thresh = plt.subplots(figsize=(10, 6))
        
        thresholds = np.linspace(0, 1, 100)
        accuracies = []
        f1_scores = []
        
        for threshold in thresholds:
            predictions = (similarity_scores > threshold).astype(int)
            accuracies.append(accuracy_score(labels, predictions))
            f1_scores.append(self.metrics_calculator._compute_f1_score(labels, predictions))
            
        ax_thresh.plot(thresholds, accuracies, label='Accuracy')
        ax_thresh.plot(thresholds, f1_scores, label='F1 Score')
        ax_thresh.set_xlabel('Threshold')
        ax_thresh.set_ylabel('Score')
        ax_thresh.set_title('Performance vs Threshold')
        ax_thresh.legend()
        ax_thresh.grid(True)
        plots['threshold_analysis'] = fig_thresh
        
        return plots
        
    def save_evaluation_results(self, results: Dict[str, Any]):
        """Save evaluation results"""
        # Save metrics
        metrics_file = self.save_dir / "metrics.json"
        with open(metrics_file, 'w') as f:
            # Convert NumPy types to native Python types for JSON serialization
            metrics_converted = convert_numpy_types(results['metrics'])
            json.dump(metrics_converted, f, indent=2)
            
        # Save plots
        for plot_name, fig in results['plots'].items():
            plot_file = self.save_dir / f"{plot_name}.png"
            fig.savefig(plot_file, dpi=300, bbox_inches='tight')
            plt.close(fig)
            
        # Save similarity scores and labels
        scores_file = self.save_dir / "similarity_scores.npz"
        np.savez(scores_file, 
                similarity_scores=results['similarity_scores'],
                labels=results['labels'])
                
        print(f"Evaluation results saved to {self.save_dir}")
        
    def evaluate_hard_pairs(self,
                           model,
                           hard_pairs_loader,
                           device: str = 'cuda') -> Dict[str, Any]:
        """
        Evaluate model specifically on hard pairs (twins and same-person)
        
        Args:
            model: Twin verification model
            hard_pairs_loader: Data loader with hard pairs only
            device: Device to use
            
        Returns:
            Dictionary of hard pairs evaluation results
        """
        print("Evaluating on hard pairs (twins and same-person pairs)...")
        
        results = self.evaluate_model(model, hard_pairs_loader, device, save_results=False)
        
        # Add hard pairs specific analysis
        similarity_scores = results['similarity_scores']
        labels = results['labels']
        
        # Analyze twin pairs vs same-person pairs
        # This assumes the data loader provides information about pair types
        # In practice, you'd need to modify the data loader to provide this information
        
        hard_pairs_metrics = {
            'overall_metrics': results['metrics'],
            'hard_pairs_count': len(similarity_scores),
            'same_person_pairs': np.sum(labels == 1),
            'twin_pairs': np.sum(labels == 0),  # Assuming twins are labeled as 0
            'average_similarity_same_person': np.mean(similarity_scores[labels == 1]),
            'average_similarity_twins': np.mean(similarity_scores[labels == 0]),
            'std_similarity_same_person': np.std(similarity_scores[labels == 1]),
            'std_similarity_twins': np.std(similarity_scores[labels == 0])
        }
        
        # Save hard pairs results
        hard_pairs_file = self.save_dir / "hard_pairs_metrics.json"
        with open(hard_pairs_file, 'w') as f:
            # Convert NumPy types to native Python types for JSON serialization
            hard_pairs_converted = convert_numpy_types(hard_pairs_metrics)
            json.dump(hard_pairs_converted, f, indent=2)
            
        print("Hard pairs evaluation completed!")
        return hard_pairs_metrics


def evaluate_twin_model(model,
                       test_loader,
                       device: str = 'cuda',
                       save_dir: str = "./evaluation_results",
                       evaluate_hard_pairs: bool = True,
                       hard_pairs_loader: Optional[object] = None) -> Dict[str, Any]:
    """
    Convenience function to evaluate twin verification model
    
    Args:
        model: Twin verification model
        test_loader: Test data loader
        device: Device to use
        save_dir: Directory to save results
        evaluate_hard_pairs: Whether to evaluate on hard pairs
        hard_pairs_loader: Data loader for hard pairs
        
    Returns:
        Dictionary of evaluation results
    """
    evaluator = TwinVerificationEvaluator(save_dir)
    
    # Evaluate on full test set
    results = evaluator.evaluate_model(model, test_loader, device)
    
    # Evaluate on hard pairs if requested
    if evaluate_hard_pairs and hard_pairs_loader is not None:
        hard_pairs_results = evaluator.evaluate_hard_pairs(model, hard_pairs_loader, device)
        results['hard_pairs'] = hard_pairs_results
        
    return results


def plot_attention_maps(model,
                       image1: torch.Tensor,
                       image2: torch.Tensor,
                       save_path: Optional[str] = None):
    """
    Plot attention maps for twin verification
    
    Args:
        model: Twin verification model
        image1: First image
        image2: Second image
        save_path: Path to save attention map
    """
    model.eval()
    
    with torch.no_grad():
        # Get attention rollout
        # This requires the model to have attention rollout functionality
        if hasattr(model, 'get_attention_maps'):
            attention_maps1 = model.get_attention_maps(image1.unsqueeze(0))
            attention_maps2 = model.get_attention_maps(image2.unsqueeze(0))
            
            # Plot attention maps
            fig, axes = plt.subplots(2, 2, figsize=(12, 8))
            
            # Image 1 attention
            axes[0, 0].imshow(attention_maps1[0].cpu().numpy(), cmap='hot')
            axes[0, 0].set_title('Image 1 Attention Map')
            axes[0, 0].axis('off')
            
            # Image 2 attention
            axes[0, 1].imshow(attention_maps2[0].cpu().numpy(), cmap='hot')
            axes[0, 1].set_title('Image 2 Attention Map')
            axes[0, 1].axis('off')
            
            # Original images
            axes[1, 0].imshow(image1.permute(1, 2, 0).cpu().numpy())
            axes[1, 0].set_title('Image 1')
            axes[1, 0].axis('off')
            
            axes[1, 1].imshow(image2.permute(1, 2, 0).cpu().numpy())
            axes[1, 1].set_title('Image 2')
            axes[1, 1].axis('off')
            
            plt.tight_layout()
            
            if save_path:
                plt.savefig(save_path, dpi=300, bbox_inches='tight')
            else:
                plt.show()
        else:
            print("Model does not have attention map functionality") 