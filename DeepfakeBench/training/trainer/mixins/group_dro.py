"""
Group-DRO (Distributionally Robust Optimization) mixin for Trainer.

Implements Group-DRO loss that upweights groups (methods) with higher loss,
encouraging the model to perform well across all generation methods.
"""
from typing import Any, Dict, Optional

import torch


class GroupDROMixin:
    """
    Mixin that provides Group-DRO functionality for the Trainer.
    
    Group-DRO upweights samples from groups (methods) that have higher loss,
    which helps the model generalize across different deepfake generation methods.
    
    Assumes the base class has:
    - self.config: Training configuration dict
    - self.model: The PyTorch model
    - self.logger: Logger instance
    """
    
    def init_group_dro(self) -> None:
        """Initialize Group-DRO state. Call from __init__ if use_group_dro is True."""
        self.logger.info("Initializing Group-DRO loss strategy.")
        
        dro_params = self.config.get('group_dro_params', {})
        self.group_dro_beta = dro_params.get('beta', 3.0)
        self.group_dro_clip_min = dro_params.get('clip_min', 1.0)
        self.group_dro_clip_max = dro_params.get('clip_max', 4.0)
        self.ema_alpha = dro_params.get('ema_alpha', 0.1)

        # PREREQUISITE: Config must contain the method_mapping
        method_mapping = self.config.get('data_params', {}).get('method_mapping')
        if not method_mapping:
            raise ValueError("Group-DRO requires 'data_params.method_mapping' in the config.")
        
        self.num_methods = len(method_mapping)
        self.method_mapping = method_mapping
        
        # Initialize EMA of group losses
        device = next(self.model.parameters()).device
        self.group_losses_ema = torch.zeros(self.num_methods, device=device)
        
        self.logger.info(
            f"Group-DRO initialized with {self.num_methods} methods, "
            f"beta={self.group_dro_beta}, clip=[{self.group_dro_clip_min}, {self.group_dro_clip_max}]"
        )

    def calculate_group_dro_loss(
        self,
        data_dict: Dict[str, Any],
        per_sample_loss: torch.Tensor
    ) -> Dict[str, Any]:
        """
        Computes the Group-DRO loss for a given batch.
        
        Args:
            data_dict: Data dictionary containing 'method_id' tensor
            per_sample_loss: Per-sample loss tensor (before reduction)
            
        Returns:
            Dictionary with 'overall' (weighted loss) and 'group_weights' tensors
        """
        # The dataloader must provide the 'method_id' tensor
        method_ids = data_dict['method_id']
        device = per_sample_loss.device

        # Handle video-style batches where loss is per-frame but method_id is per-video.
        # The model reshapes [B, T, C, H, W] → [B*T, C, H, W] for per-frame predictions,
        # so per_sample_loss has B*T elements while method_ids has only B elements.
        # We expand method_ids to match by repeating each method_id T times.
        if per_sample_loss.shape[0] > method_ids.shape[0]:
            B = method_ids.shape[0]
            T = per_sample_loss.shape[0] // B
            method_ids = method_ids.repeat_interleave(T)

        # Step 1: Compute per-group average loss for the current batch
        batch_group_losses = torch.zeros(self.num_methods, device=device)
        batch_group_counts = torch.zeros(self.num_methods, device=device)

        # Efficient scatter_add operation for calculating group means
        batch_group_counts.index_add_(0, method_ids, torch.ones_like(per_sample_loss))
        batch_group_losses.index_add_(0, method_ids, per_sample_loss.detach())

        valid_groups_mask = batch_group_counts > 0
        batch_group_losses[valid_groups_mask] /= batch_group_counts[valid_groups_mask]

        # Step 2: Update the EMA of group losses
        current_ema = self.group_losses_ema[valid_groups_mask]
        current_batch_loss = batch_group_losses[valid_groups_mask]
        updated_ema = (1 - self.ema_alpha) * current_ema + self.ema_alpha * current_batch_loss
        self.group_losses_ema[valid_groups_mask] = updated_ema

        # Step 3: Compute group weights based on the EMA
        # Use only the EMA of groups present in the current batch
        valid_ema_losses = self.group_losses_ema[self.group_losses_ema > 0]
        avg_ema_loss = valid_ema_losses.mean() if len(valid_ema_losses) > 0 else 0

        relative_losses = self.group_losses_ema - avg_ema_loss

        # Exponential weighting based on relative loss
        weights = torch.exp(self.group_dro_beta * relative_losses)
        weights = weights * (self.num_methods / torch.sum(weights))  # Normalize
        clipped_weights = torch.clip(weights, self.group_dro_clip_min, self.group_dro_clip_max)

        # Step 4: Calculate the final weighted loss for the batch
        sample_weights = clipped_weights[method_ids].detach()
        weighted_loss = (per_sample_loss * sample_weights).mean()

        return {
            'overall': weighted_loss,
            'group_weights': clipped_weights,
            'group_losses_ema': self.group_losses_ema.clone(),
        }
    
    def get_group_dro_stats(self) -> Dict[str, Any]:
        """Get current Group-DRO statistics for logging."""
        if not hasattr(self, 'group_losses_ema'):
            return {}
        
        # Get method names from mapping (reverse lookup)
        reverse_mapping = {v: k for k, v in self.method_mapping.items()}
        
        stats = {}
        for idx in range(self.num_methods):
            method_name = reverse_mapping.get(idx, f"method_{idx}")
            stats[f"group_dro/ema_loss/{method_name}"] = self.group_losses_ema[idx].item()
        
        return stats
