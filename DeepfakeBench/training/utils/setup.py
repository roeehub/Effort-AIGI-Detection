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
    
    Args:
        model: PyTorch model whose parameters will be optimized
        config: Configuration dictionary with structure:
            optimizer:
              type: "adam"
              adam:
                lr: 0.0002
                eps: 1e-8
                weight_decay: 0.0005
    
    Returns:
        Configured optimizer instance
    
    Raises:
        NotImplementedError: If optimizer type is not supported
    """
    opt_name = config['optimizer']['type']
    if opt_name == 'adam':
        # Separate SVD residual parameters from other parameters
        # SVD params should NOT have weight decay (interferes with orthogonality constraints)
        svd_param_names = ['U_residual', 'S_residual', 'V_residual']
        
        svd_params = []
        other_params = []
        
        for name, param in model.named_parameters():
            if not param.requires_grad:
                continue
            if any(svd_name in name for svd_name in svd_param_names):
                svd_params.append(param)
            else:
                other_params.append(param)
        
        # Log the parameter group split
        svd_param_count = sum(p.numel() for p in svd_params)
        other_param_count = sum(p.numel() for p in other_params)
        
        weight_decay = config['optimizer'][opt_name]['weight_decay']
        
        # Create param groups: SVD params get weight_decay=0, others get normal weight_decay
        param_groups = []
        if svd_params:
            param_groups.append({
                'params': svd_params,
                'weight_decay': 0.0,  # CRITICAL: No weight decay for SVD residual params
                'name': 'svd_residual'
            })
        if other_params:
            param_groups.append({
                'params': other_params,
                'weight_decay': weight_decay,
                'name': 'other'
            })
        
        print(f"INFO: Optimizer param groups:")
        print(f"  - SVD residual params: {svd_param_count:,} (weight_decay=0.0)")
        print(f"  - Other params: {other_param_count:,} (weight_decay={weight_decay})")
        
        optimizer = optim.Adam(
            param_groups,
            lr=config['optimizer'][opt_name]['lr'],
            eps=config['optimizer'][opt_name]['eps'],
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
