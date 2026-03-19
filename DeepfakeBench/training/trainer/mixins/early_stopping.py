"""
Early stopping mixin for Trainer.

Provides patience-based early stopping functionality to prevent overfitting
and save compute resources.
"""
from typing import Optional


class EarlyStoppingMixin:
    """
    Mixin that provides early stopping functionality for the Trainer.
    
    Assumes the base class has:
    - self.config: Training configuration dict
    - self.logger: Logger instance
    """
    
    def init_early_stopping(self) -> None:
        """Initialize early stopping state. Call from __init__."""
        self.early_stopping_config = self.config.get('early_stopping', {})
        self.early_stopping_enabled = self.early_stopping_config.get('enabled', False)
        
        self.early_stopping_patience = self.early_stopping_config.get('patience', 3)
        self.early_stopping_min_delta = self.early_stopping_config.get('min_delta', 0.0001)
        self.early_stopping_mode = self.early_stopping_config.get('mode', 'max')
        
        self.epochs_without_improvement = 0
        self.early_stop_triggered = False
        self._early_stopping_best_metric: Optional[float] = None
        
        if self.early_stopping_enabled:
            self.logger.info(
                f"✅ Early stopping enabled: patience={self.early_stopping_patience}, "
                f"min_delta={self.early_stopping_min_delta}, mode={self.early_stopping_mode}"
            )
        else:
            self.logger.info(
                "Early stopping is disabled. Training will run for all configured epochs."
            )
    
    def check_early_stopping(self, current_metric: float) -> bool:
        """
        Check if early stopping criterion is met.
        
        Args:
            current_metric: The current validation metric value
            
        Returns:
            True if training should stop, False otherwise
        """
        if not self.early_stopping_enabled:
            return False
        
        # Initialize best metric on first call
        if self._early_stopping_best_metric is None:
            self._early_stopping_best_metric = current_metric
            return False
        
        # Check for improvement based on mode
        if self.early_stopping_mode == 'max':
            improved = current_metric > (self._early_stopping_best_metric + self.early_stopping_min_delta)
        else:  # mode == 'min'
            improved = current_metric < (self._early_stopping_best_metric - self.early_stopping_min_delta)
        
        if improved:
            self._early_stopping_best_metric = current_metric
            self.epochs_without_improvement = 0
            self.logger.info(
                f"📈 Metric improved to {current_metric:.4f}. "
                f"Resetting early stopping counter."
            )
        else:
            self.epochs_without_improvement += 1
            self.logger.info(
                f"📉 No improvement for {self.epochs_without_improvement}/{self.early_stopping_patience} epochs. "
                f"Best: {self._early_stopping_best_metric:.4f}, Current: {current_metric:.4f}"
            )
        
        if self.epochs_without_improvement >= self.early_stopping_patience:
            self.early_stop_triggered = True
            self.logger.info(
                f"🛑 Early stopping triggered after {self.early_stopping_patience} epochs "
                f"without improvement. Best metric: {self._early_stopping_best_metric:.4f}"
            )
            return True
        
        return False
    
    def reset_early_stopping(self) -> None:
        """Reset early stopping state (useful for curriculum learning)."""
        self.epochs_without_improvement = 0
        self.early_stop_triggered = False
        self._early_stopping_best_metric = None
        self.logger.info("🔄 Early stopping state reset.")
