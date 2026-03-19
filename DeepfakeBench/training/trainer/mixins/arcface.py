"""
ArcFace mixin for Trainer.

Provides ArcFace head parameter annealing functionality for curriculum learning.
"""
from typing import Optional


class ArcFaceMixin:
    """
    Mixin that provides ArcFace parameter annealing for the Trainer.
    
    ArcFace uses a scale parameter 's' that can be annealed during training
    to gradually increase the difficulty of the classification task.
    
    Assumes the base class has:
    - self.config: Training configuration dict
    - self.model: The PyTorch model
    - self.logger: Logger instance
    """
    
    def init_arcface(self) -> None:
        """Initialize ArcFace state. Call from __init__ if using ArcFace."""
        self.use_arcface_head = self.config.get('use_arcface_head', False)
        self.train_arcface = self.config.get('train_arcface', True)
        
        if not self.use_arcface_head:
            return
        
        # Annealing parameters
        self.arcface_s_start = self.config.get('s_start', self.config.get('arcface_s', 30.0))
        self.arcface_s_end = self.config.get('s_end', self.config.get('arcface_s', 30.0))
        self.arcface_anneal_steps = self.config.get('anneal_steps', 0)
        
        self.logger.info(
            f"✅ ArcFace initialized: s_start={self.arcface_s_start}, "
            f"s_end={self.arcface_s_end}, anneal_steps={self.arcface_anneal_steps}"
        )
        
        if self.arcface_anneal_steps > 0:
            self.logger.info(
                f"   Annealing will run for {self.arcface_anneal_steps} steps "
                f"from s={self.arcface_s_start} to s={self.arcface_s_end}"
            )
    
    def update_arcface_s(self, step_cnt: int) -> Optional[float]:
        """
        Anneals the 's' parameter of the ArcFace head if configured.
        
        Args:
            step_cnt: Current training step
            
        Returns:
            The new 's' value, or None if not using ArcFace
        """
        if not self.use_arcface_head or not self.train_arcface:
            return None
        
        if self.arcface_anneal_steps <= 0:
            return None
        
        # Get model instance (handle DDP)
        model_instance = self.model.module if self.config.get('ddp') else self.model
        
        if not hasattr(model_instance, 'head') or not hasattr(model_instance.head, 's'):
            return None
        
        # Linear annealing from s_start to s_end
        progress = min(1.0, step_cnt / self.arcface_anneal_steps)
        new_s = self.arcface_s_start + progress * (self.arcface_s_end - self.arcface_s_start)
        
        # Update the parameter
        model_instance.head.s.data.fill_(new_s)
        
        return new_s
    
    def get_current_arcface_s(self) -> Optional[float]:
        """Get the current ArcFace 's' parameter value."""
        if not self.use_arcface_head:
            return None
        
        model_instance = self.model.module if self.config.get('ddp') else self.model
        
        if hasattr(model_instance, 'head') and hasattr(model_instance.head, 's'):
            return float(model_instance.head.s)
        
        return None
    
    def log_arcface_state(self, step_cnt: int) -> None:
        """Log current ArcFace state for monitoring."""
        current_s = self.get_current_arcface_s()
        if current_s is not None:
            self.logger.debug(f"Step {step_cnt}: ArcFace s = {current_s:.4f}")
