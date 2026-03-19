"""
Validation mixin for Trainer.

Provides validation state management and helper utilities.
The main test_epoch logic remains in the trainer for now due to its
complexity and deep integration with model and data loaders.
"""
from typing import Dict, Optional, List, Any
from collections import defaultdict
import time


class ValidationMixin:
    """
    Mixin that provides validation state management and utilities.
    
    Assumes the base class has:
    - self.config: Training configuration dict
    - self.logger: Logger instance
    - self.model: The PyTorch model
    """
    
    def init_validation(self) -> None:
        """Initialize validation state. Call from __init__."""
        # Primary validation metric tracking
        self.metric_scoring = self.config.get('metric_scoring', 'auc')
        self.best_val_metric = 0.0
        self.best_val_epoch = 0
        
        # Validation loaders (set by prepare_data or similar)
        self.val_in_dist_loader = None
        self.val_holdout_loader = None
        self.ood_loader = None
        
        # Validation timing
        self._val_start_time = None
        self._val_videos_processed = 0
        
        self.logger.info(f"✅ Validation initialized: scoring metric = {self.metric_scoring}")
    
    def start_validation_timer(self) -> None:
        """Start timing a validation run."""
        self._val_start_time = time.time()
        self._val_videos_processed = 0
    
    def get_validation_duration(self) -> float:
        """Get elapsed time since validation started."""
        if self._val_start_time is None:
            return 0.0
        return time.time() - self._val_start_time
    
    def update_validation_progress(self, videos_processed: int) -> None:
        """Update the count of videos processed."""
        self._val_videos_processed = videos_processed
    
    def get_validation_progress(self) -> int:
        """Get current validation progress."""
        return self._val_videos_processed
    
    def should_run_validation(self, epoch: int, iteration: int, step_cnt: int) -> bool:
        """
        Determine if validation should run at current training state.
        
        Args:
            epoch: Current epoch
            iteration: Current iteration within epoch
            step_cnt: Global step count
            
        Returns:
            True if validation should run
        """
        # Only run on primary rank
        if self.config.get('local_rank', 0) != 0:
            return False
        
        # Check if we're at a validation interval
        val_interval = self.config.get('val_interval', 1)
        if val_interval <= 0:
            return False
        
        # Epoch-based validation
        if (epoch + 1) % val_interval == 0:
            # Could add iteration-based validation here
            return True
        
        return False
    
    def is_primary_validation_improved(self, current_metric: float) -> bool:
        """
        Check if the current metric represents an improvement.
        
        Args:
            current_metric: The metric value to check
            
        Returns:
            True if this is an improvement over the best
        """
        min_delta = getattr(self, 'early_stopping_min_delta', 0.0)
        return current_metric > self.best_val_metric + min_delta
    
    def update_best_metric(self, metric_value: float, epoch: int) -> None:
        """
        Update the best validation metric.
        
        Args:
            metric_value: New best metric value
            epoch: Epoch when best metric was achieved
        """
        self.best_val_metric = metric_value
        self.best_val_epoch = epoch
        self.logger.info(
            f"🚀 New best {self.metric_scoring}: {metric_value:.4f} at epoch {epoch}"
        )
    
    def get_validation_summary(self) -> Dict[str, Any]:
        """Get a summary of validation state."""
        return {
            'best_metric': self.best_val_metric,
            'best_epoch': self.best_val_epoch,
            'metric_type': self.metric_scoring,
            'has_in_dist_loader': self.val_in_dist_loader is not None,
            'has_holdout_loader': self.val_holdout_loader is not None,
            'has_ood_loader': self.ood_loader is not None,
        }
    
    def create_validation_log_dict(
        self,
        log_prefix: str,
        epoch: int,
        step_cnt: int
    ) -> Dict[str, Any]:
        """
        Create a base log dictionary for validation metrics.
        
        Args:
            log_prefix: Prefix for metric names
            epoch: Current epoch
            step_cnt: Current step count
            
        Returns:
            Dictionary with base validation metrics
        """
        return {
            f"{log_prefix}/epoch": epoch + 1,
            "train/step": step_cnt,
            'val_primary/best_metric': self.best_val_metric,
            'val_primary/best_epoch': self.best_val_epoch,
        }
    
    def collect_method_predictions(
        self,
        method_preds: Dict[str, List[float]],
        method_labels: Dict[str, List[int]]
    ) -> Dict[str, Dict[str, Any]]:
        """
        Calculate per-method accuracy metrics.
        
        Args:
            method_preds: Predictions per method
            method_labels: Labels per method
            
        Returns:
            Dictionary mapping method names to their metrics
        """
        import numpy as np
        
        metrics = {}
        real_sources = self.config.get('dataset_methods', {}).get('use_real_sources', [])
        
        for method in sorted(method_preds.keys()):
            preds = np.array(method_preds[method])
            labels = np.array(method_labels[method])
            
            if len(labels) == 0:
                continue
            
            # Calculate threshold-based accuracy
            predictions_binary = (preds >= 0.5).astype(int)
            correct = np.sum(predictions_binary == labels)
            accuracy = correct / len(labels)
            
            metrics[method] = {
                'acc': accuracy,
                'n_samples': len(labels),
                'n_correct': int(correct),
                'is_real_source': method in real_sources
            }
        
        return metrics
    
    def calculate_macro_accuracy(
        self,
        method_metrics: Dict[str, Dict[str, Any]]
    ) -> Optional[float]:
        """
        Calculate macro accuracy across all methods.
        
        Args:
            method_metrics: Per-method metrics from collect_method_predictions
            
        Returns:
            Macro accuracy or None if no metrics available
        """
        import numpy as np
        
        accuracies = [
            m['acc'] for m in method_metrics.values()
            if m.get('acc') is not None
        ]
        
        if not accuracies:
            return None
        
        return float(np.mean(accuracies))
