"""
Setup utilities for training initialization.

This module provides functions for:
- Random seed initialization
- Optimizer selection and configuration
- Learning rate scheduler selection
- Metric selection

Usage:
    from utils.setup import init_seed, choose_optimizer, choose_scheduler, choose_metric
    
    init_seed(config)
    optimizer = choose_optimizer(model, config)
    scheduler = choose_scheduler(config, optimizer)
    metric = choose_metric(config)
"""

import random
import torch
import torch.optim as optim
from transformers import get_cosine_schedule_with_warmup


def init_seed(config: dict) -> None:
    """
    Initialize random seeds for reproducibility.
    
    Sets seeds for Python's random module, PyTorch CPU, and CUDA.
    If no manual seed is provided in config, generates a random one.
    
    Args:
        config: Configuration dictionary with keys:
            - 'manualSeed': Optional seed value (int or None)
            - 'cuda': Whether CUDA is being used (bool)
    """
    if config['manualSeed'] is None:
        config['manualSeed'] = random.randint(1, 10000)
    random.seed(config['manualSeed'])
    if config['cuda']:
        torch.manual_seed(config['manualSeed'])
        torch.cuda.manual_seed_all(config['manualSeed'])


def choose_optimizer(model: torch.nn.Module, config: dict) -> optim.Optimizer:
    """
    Create an optimizer based on configuration.

    Currently supports:
        - 'adam': Adam optimizer with configurable lr, eps, weight_decay

    IMPORTANT (Jan 11, 2026): SVD residual parameters (U_residual, S_residual, V_residual)
    are placed in a separate param group with weight_decay=0.0 to prevent interference
    with orthogonality and keepsv constraints. Weight decay on these params creates a
    constant tug-of-war with the regularization losses, causing them to grow unbounded.

    Optional capability (Apr 25, 2026): a per-group LR multiplier for the unfrozen
    backbone parameters (SVD residuals + visual.proj + visual.ln_post when those flags
    are enabled). Set `optimizer.adam.backbone_lr_mult` in yaml to apply
    `lr * backbone_lr_mult` to those params while keeping the head at the base lr.
    Default is 1.0, which preserves the previous single-LR behavior bit-identically.

    Optional capability (May 12, 2026): a `lora` param group with weight_decay=0.0
    and its own LR multiplier `optimizer.adam.lora_lr_mult`. Parameters whose names
    contain `lora_A` or `lora_B` (i.e., the LoRA A/B matrices introduced by
    `detectors/lora_adapter.py`) are routed here. Standard LoRA practice uses zero
    weight decay on these params. When LoRA is disabled (no matching names), this
    group is simply empty and behavior is bit-identical to the pre-LoRA path.

    Args:
        model: PyTorch model whose parameters will be optimized
        config: Configuration dictionary with structure:
            optimizer:
              type: "adam"
              adam:
                lr: 0.0002
                eps: 1e-8
                weight_decay: 0.0005
                backbone_lr_mult: 1.0   # optional, default 1.0

    Returns:
        Configured optimizer instance

    Raises:
        NotImplementedError: If optimizer type is not supported
    """
    opt_name = config['optimizer']['type']
    if opt_name == 'adam':
        adam_cfg = config['optimizer'][opt_name]
        base_lr = adam_cfg['lr']
        weight_decay = adam_cfg['weight_decay']
        backbone_lr_mult = adam_cfg.get('backbone_lr_mult', 1.0)
        lora_lr_mult = adam_cfg.get('lora_lr_mult', 1.0)
        backbone_lr = base_lr * backbone_lr_mult
        lora_lr = base_lr * lora_lr_mult

        # Four categories of trainable params:
        #   1. SVD residual params (no weight decay; tracked separately since
        #      Jan 11 2026 to prevent orthogonality/keepsv interference).
        #   2. Native CLIP backbone params that may be unfrozen via the
        #      backbone.unfreeze_final_proj / unfreeze_final_ln flags.
        #   3. LoRA A/B matrices (no weight decay; standard LoRA practice;
        #      added May 12 2026 with the layers-10-11 LoRA packet).
        #   4. Everything else (head, ArcFace, etc.).
        # Categories (1) and (2) share `backbone_lr`; (3) uses `lora_lr`;
        # (4) uses `base_lr`. When all multipliers are 1.0 and no LoRA is
        # installed, behavior is bit-identical to the prior 3-group split.
        svd_param_names = ('U_residual', 'S_residual', 'V_residual')
        backbone_native_names = ('visual.proj', 'visual.ln_post')
        lora_param_names = ('lora_A', 'lora_B')

        svd_params = []
        backbone_native_params = []
        lora_params = []
        other_params = []

        for name, param in model.named_parameters():
            if not param.requires_grad:
                continue
            if any(svd_name in name for svd_name in svd_param_names):
                svd_params.append(param)
            elif any(bn_name in name for bn_name in backbone_native_names):
                backbone_native_params.append(param)
            elif any(lora_name in name for lora_name in lora_param_names):
                lora_params.append(param)
            else:
                other_params.append(param)

        svd_param_count = sum(p.numel() for p in svd_params)
        backbone_native_param_count = sum(p.numel() for p in backbone_native_params)
        lora_param_count = sum(p.numel() for p in lora_params)
        other_param_count = sum(p.numel() for p in other_params)

        param_groups = []
        if svd_params:
            param_groups.append({
                'params': svd_params,
                'weight_decay': 0.0,  # CRITICAL: No weight decay for SVD residual params
                'lr': backbone_lr,
                'name': 'svd_residual',
            })
        if backbone_native_params:
            param_groups.append({
                'params': backbone_native_params,
                'weight_decay': weight_decay,
                'lr': backbone_lr,
                'name': 'backbone_native',
            })
        if lora_params:
            param_groups.append({
                'params': lora_params,
                'weight_decay': 0.0,  # standard LoRA practice
                'lr': lora_lr,
                'name': 'lora',
            })
        if other_params:
            param_groups.append({
                'params': other_params,
                'weight_decay': weight_decay,
                'lr': base_lr,
                'name': 'other',
            })

        print(
            f"INFO: Optimizer param groups (backbone_lr_mult={backbone_lr_mult}, "
            f"lora_lr_mult={lora_lr_mult}):"
        )
        print(f"  - SVD residual params: {svd_param_count:,} (lr={backbone_lr:g}, weight_decay=0.0)")
        print(f"  - Backbone native params: {backbone_native_param_count:,} (lr={backbone_lr:g}, weight_decay={weight_decay})")
        print(f"  - LoRA params: {lora_param_count:,} (lr={lora_lr:g}, weight_decay=0.0)")
        print(f"  - Other params: {other_param_count:,} (lr={base_lr:g}, weight_decay={weight_decay})")

        optimizer = optim.Adam(
            param_groups,
            lr=base_lr,
            eps=adam_cfg['eps'],
        )
        return optimizer
    else:
        raise NotImplementedError(f'Optimizer {opt_name} is not implemented')


def choose_scheduler(config: dict, optimizer: optim.Optimizer):
    """
    Create a learning rate scheduler based on configuration.
    
    Supports:
        - None/'none'/'null': No scheduler
        - 'cosine': CosineAnnealingLR (epoch-based)
        - 'cosine_with_warmup': Cosine schedule with linear warmup (step-based)
    
    Args:
        config: Configuration dictionary with keys:
            - 'lr_scheduler': Scheduler type string
            - 'nEpochs': Number of epochs (for 'cosine')
            - 'total_training_steps': Total steps (for 'cosine_with_warmup')
            - 'lr_scheduler_warmup_steps': Warmup steps (for 'cosine_with_warmup')
            - 'optimizer.adam.lr': Base learning rate
        optimizer: The optimizer to schedule
    
    Returns:
        Scheduler instance or None
    
    Raises:
        ValueError: If 'cosine_with_warmup' is selected without 'total_training_steps'
        NotImplementedError: If scheduler type is not supported
    """
    scheduler_type = config.get('lr_scheduler')
    if scheduler_type is None or scheduler_type.lower() == 'none' or scheduler_type.lower() == 'null':
        return None

    if scheduler_type == 'cosine':
        # Epoch-based cosine annealing (backward compatibility)
        return optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=config['nEpochs'], eta_min=config['optimizer']['adam']['lr'] / 100
        )
    elif scheduler_type == 'cosine_with_warmup':
        if not config.get('total_training_steps'):
            raise ValueError("'total_training_steps' must be configured for the 'cosine_with_warmup' scheduler.")

        warmup_steps = config.get('lr_scheduler_warmup_steps', 0)
        total_steps = config['total_training_steps']

        print(
            f"INFO: Using cosine_with_warmup scheduler with {warmup_steps} warmup steps and {total_steps} total steps.")

        return get_cosine_schedule_with_warmup(
            optimizer,
            num_warmup_steps=warmup_steps,
            num_training_steps=total_steps
        )

    raise NotImplementedError(f"Scheduler '{scheduler_type}' is not implemented")


def choose_metric(config: dict) -> str:
    """
    Validate and return the metric scoring method.
    
    Args:
        config: Configuration dictionary with 'metric_scoring' key
    
    Returns:
        The validated metric name string
    
    Raises:
        NotImplementedError: If metric is not one of: 'eer', 'auc', 'acc', 'ap'
    """
    metric_scoring = config['metric_scoring']
    if metric_scoring not in ['eer', 'auc', 'acc', 'ap']:
        raise NotImplementedError(f'metric {metric_scoring} is not implemented')
    return metric_scoring
