import torch
import os
import torch.nn as nn
from timm.scheduler.cosine_lr import CosineLRScheduler
from collections import OrderedDict
from engine.optimizer import make_optimizer
from engine.triplet_loss import TripletLoss
from utils.misc import AverageMeter, EarlyStop
from utils.setup_logging import get_logger
from utils.verification_metrics import VerificationMetrics
from timm.utils import update_summary
import numpy as np
import json

logger = get_logger("Prompt_CAM")
torch.backends.cudnn.benchmark = False


class TwinFaceTrainer():
    """
    Trainer for twin face verification using triplet loss.
    
    Features:
    - Triplet loss training with hard negative mining
    - Verification metrics evaluation (EER, AUC, TAR/FAR)
    - Attention map saving for visualization
    """

    def __init__(self, model, tune_parameters, params):
        self.params = params
        self.model = model
        self.device = params.device
        
        # Triplet loss
        self.triplet_criterion = TripletLoss(
            margin=getattr(params, 'triplet_margin', 0.3),
            distance_metric=getattr(params, 'distance_metric', 'cosine')
        )
        
        # Verification metrics
        self.verification_metrics = VerificationMetrics()
        
        if not hasattr(params, 'test_data') or params.test_data is None:
            # Setup optimizer and scheduler
            logger.info("\tSetting up the optimizer...")
            self.optimizer = make_optimizer(tune_parameters, params)
            self.scheduler = CosineLRScheduler(
                self.optimizer, 
                t_initial=params.epoch,
                warmup_t=params.warmup_epoch, 
                lr_min=params.lr_min,
                warmup_lr_init=params.warmup_lr_init
            )
            self.total_epoch = self.params.epoch
            if self.params.early_patience > 0:
                self.early_stop_check = EarlyStop(self.params.early_patience)

    def forward_one_batch_triplet(self, batch_data, is_train):
        """
        Forward pass for triplet training.
        
        Args:
            batch_data: Dictionary with 'anchor', 'positive', 'negative' images
            is_train: Whether in training mode
            
        Returns:
            loss: Triplet loss
            outputs: Triplet outputs
            stats: Training statistics
        """
        # Move data to device
        anchor = batch_data['anchor'].to(self.device, non_blocking=True)
        positive = batch_data['positive'].to(self.device, non_blocking=True)
        negative = batch_data['negative'].to(self.device, non_blocking=True)
        
        # Forward pass
        with torch.set_grad_enabled(is_train):
            triplet_output = self.model.forward_triplet(anchor, positive, negative)
            
            # Extract embeddings
            anchor_emb = triplet_output['anchor_features']
            positive_emb = triplet_output['positive_features']
            negative_emb = triplet_output['negative_features']
            
            # Compute triplet loss
            loss, stats = self.triplet_criterion(anchor_emb, positive_emb, negative_emb)
            
            if loss == float('inf') or torch.isnan(loss).any():
                logger.info("Encountered infinite/nan loss, skip gradient updating for this batch!")
                return -1, triplet_output, (-1, -1, -1)
        
        # Backward pass for training
        if is_train:
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
        
        return loss, triplet_output, stats

    def train_one_epoch(self, epoch, loader):
        """Train one epoch using triplet loss."""
        loss_m = AverageMeter()
        triplet_acc_m = AverageMeter()
        pos_dist_m = AverageMeter()
        neg_dist_m = AverageMeter()
        
        lr = self.scheduler._get_lr(epoch)
        logger.info(
            "Training {} / {} epoch, with learning rate {}".format(
                epoch + 1, self.total_epoch, lr
            )
        )
        
        # Enable training mode
        self.model.train()
        
        num_updates = epoch * len(loader)
        for idx, batch_data in enumerate(loader):
            train_loss, _, stats = self.forward_one_batch_triplet(batch_data, True)
            
            if not isinstance(train_loss, int):
                loss_m.update(train_loss.item(), batch_data['anchor'].shape[0])
                triplet_acc_m.update(stats['accuracy'], batch_data['anchor'].shape[0])
                pos_dist_m.update(stats['pos_dist'], batch_data['anchor'].shape[0])
                neg_dist_m.update(stats['neg_dist'], batch_data['anchor'].shape[0])
            
            del train_loss, stats, batch_data
            num_updates += 1
            self.scheduler.step_update(num_updates=num_updates, metric=loss_m.avg)
        
        logger.info(
            "Epoch {} / {}: ".format(epoch + 1, self.total_epoch)
            + "average train loss: {:.4f}, ".format(loss_m.avg)
            + "average triplet accuracy: {:.4f}, ".format(triplet_acc_m.avg)
            + "average pos dist: {:.4f}, ".format(pos_dist_m.avg)
            + "average neg dist: {:.4f}".format(neg_dist_m.avg)
        )
        
        return OrderedDict([
            ('loss', round(loss_m.avg, 4)), 
            ('triplet_acc', round(triplet_acc_m.avg, 4)),
            ('pos_dist', round(pos_dist_m.avg, 4)),
            ('neg_dist', round(neg_dist_m.avg, 4))
        ])

    def eval_verification(self, loader, prefix):
        """
        Evaluate verification performance using verification pairs.
        
        Args:
            loader: DataLoader with verification pairs
            prefix: Evaluation prefix ('val' or 'test')
            
        Returns:
            metrics: Verification metrics
        """
        self.verification_metrics.reset()
        
        # Enable eval mode
        self.model.eval()
        
        with torch.no_grad():
            for batch_idx, batch_data in enumerate(loader):
                # Extract image pairs and labels
                img1 = batch_data['img1'].to(self.device, non_blocking=True)
                img2 = batch_data['img2'].to(self.device, non_blocking=True)
                labels = batch_data['label'].to(self.device, non_blocking=True)
                
                # Compute verification scores
                scores = self.model.compute_verification_score(img1, img2)
                
                # Update metrics
                self.verification_metrics.update(
                    scores, labels,
                    person_ids=batch_data.get('person_id', None),
                    twin_ids=batch_data.get('twin_id', None)
                )
        
        # Compute final metrics
        metrics = self.verification_metrics.get_all_metrics()
        
        logger.info(
            f"Verification ({prefix}): "
            + "EER: {:.4f}, ".format(metrics['eer'])
            + "AUC: {:.4f}, ".format(metrics['auc'])
            + "TAR: {:.4f}, ".format(metrics['tar'])
            + "FAR: {:.4f}, ".format(metrics['far'])
            + "Accuracy: {:.4f}".format(metrics['accuracy'])
        )
        
        return OrderedDict([
            ('eer', round(metrics['eer'], 4)),
            ('auc', round(metrics['auc'], 4)),
            ('tar', round(metrics['tar'], 4)),
            ('far', round(metrics['far'], 4)),
            ('accuracy', round(metrics['accuracy'], 4)),
            ('threshold', round(metrics['threshold'], 4))
        ])

    def train_classifier(self, train_loader, val_loader, test_loader):
        """
        Train the twin face verification model.
        """
        best_eer = float('inf')
        best_metrics = None
        
        for epoch in range(self.total_epoch):
            train_metrics = self.train_one_epoch(epoch, train_loader)
            
            # Evaluate at specified frequency
            if (epoch % self.params.eval_freq == 0) or epoch == self.total_epoch - 1:
                if test_loader is not None:
                    eval_metrics = self.eval_verification(test_loader, "test")
                elif val_loader is not None:
                    eval_metrics = self.eval_verification(val_loader, "val")
                else:
                    raise Exception('Both val and test loaders are missing.')
                
                # Save best model based on EER
                if eval_metrics['eer'] < best_eer:
                    best_eer = eval_metrics['eer']
                    best_metrics = eval_metrics
                    if self.params.store_ckp:
                        torch.save({
                            'model_state_dict': self.model.state_dict(),
                            'epoch': epoch,
                            'best_eer': best_eer
                        }, os.path.join(self.params.output_dir, 'model.pt'))
                
                # Early stopping
                if self.params.early_patience > 0:
                    stop, save_model = self.early_stop_check.early_stop(eval_metrics)
                    if stop:
                        logger.info(f"Early stopping at epoch {epoch}")
                        return train_metrics, best_metrics, eval_metrics
                
                # Log to summary
                if self.params.debug:
                    update_summary(
                        epoch, train_metrics, eval_metrics, 
                        os.path.join(self.params.output_dir, 'summary.csv'),
                        write_header=epoch == 0
                    )
            
            self.scheduler.step(epoch)
        
        # Save final model if not already saved
        if self.params.store_ckp and not os.path.isfile(os.path.join(self.params.output_dir, 'model.pt')):
            torch.save({
                'model_state_dict': self.model.state_dict(),
                'epoch': self.total_epoch - 1,
                'best_eer': best_eer
            }, os.path.join(self.params.output_dir, 'model.pt'))
        
        return train_metrics, best_metrics, eval_metrics

    def load_weight(self):
        """Load trained model weights."""
        checkpoint = torch.load(os.path.join(self.params.output_dir, 'model.pt'))
        self.model.load_state_dict(checkpoint['model_state_dict'])
        logger.info(f"Loaded model from epoch {checkpoint.get('epoch', 'unknown')}")

    @torch.no_grad()
    def collect_verification_scores(self, loader):
        """
        Collect verification scores for analysis.
        
        Returns:
            scores: All verification scores
            labels: All labels
            person_ids: Person IDs
            twin_ids: Twin IDs
        """
        self.model.eval()
        all_scores = []
        all_labels = []
        all_person_ids = []
        all_twin_ids = []
        
        with torch.no_grad():
            for batch_idx, batch_data in enumerate(loader):
                img1 = batch_data['img1'].to(self.device, non_blocking=True)
                img2 = batch_data['img2'].to(self.device, non_blocking=True)
                labels = batch_data['label']
                
                scores = self.model.compute_verification_score(img1, img2)
                
                all_scores.append(scores.cpu().detach().numpy())
                all_labels.append(labels.numpy())
                all_person_ids.extend(batch_data.get('person_id', []))
                all_twin_ids.extend(batch_data.get('twin_id', []))
        
        return (
            np.concatenate(all_scores, axis=0),
            np.concatenate(all_labels, axis=0),
            all_person_ids,
            all_twin_ids
        )

    def save_attention_maps(self, loader, save_dir, num_samples=10):
        """
        Save attention maps for visualization.
        
        Args:
            loader: DataLoader with images
            save_dir: Directory to save attention maps
            num_samples: Number of samples to save
        """
        os.makedirs(save_dir, exist_ok=True)
        
        self.model.eval()
        sample_count = 0
        
        with torch.no_grad():
            for batch_idx, batch_data in enumerate(loader):
                if sample_count >= num_samples:
                    break
                
                # Get images from batch
                if 'anchor' in batch_data:
                    images = batch_data['anchor']
                elif 'img1' in batch_data:
                    images = batch_data['img1']
                else:
                    continue
                
                images = images.to(self.device, non_blocking=True)
                
                # Extract attention maps for both classes
                for i in range(min(images.shape[0], num_samples - sample_count)):
                    img = images[i:i+1]
                    
                    # Get attention maps for same person (class 0) and twin (class 1)
                    same_attention = self.model.get_attention_maps(img, target_class=0)
                    twin_attention = self.model.get_attention_maps(img, target_class=1)
                    
                    if same_attention is not None and twin_attention is not None:
                        # Save attention maps
                        attention_data = {
                            'same_person_attention': same_attention.cpu().numpy(),
                            'twin_attention': twin_attention.cpu().numpy(),
                            'image_index': sample_count + i
                        }
                        
                        torch.save(
                            attention_data,
                            os.path.join(save_dir, f'attention_maps_{sample_count + i}.pt')
                        )
                
                sample_count += images.shape[0]
        
        logger.info(f"Saved attention maps for {sample_count} samples to {save_dir}") 