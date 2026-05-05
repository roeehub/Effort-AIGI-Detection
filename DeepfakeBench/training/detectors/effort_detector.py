import os
import math
import logging
import numpy as np  # noqa
from sklearn import metrics  # noqa

import torch  # noqa
import torch.nn as nn  # noqa
import torch.nn.functional as F  # noqa
import torch.optim as optim  # noqa
from torch.nn import DataParallel  # noqa

from metrics.base_metrics_class import calculate_metrics_for_train  # noqa

from detectors import DETECTOR  # noqa
from networks import BACKBONE  # noqa
from loss import LOSSFUNC  # noqa

import loralib as lora  # noqa
from transformers import AutoConfig, AutoProcessor, CLIPModel, CLIPVisionModel, ViTModel, ViTConfig  # noqa

logger = logging.getLogger(__name__)


class ArcMarginProduct(nn.Module):
    """
    Implementation of ArcFace head for binary classification.
    This module replaces the final nn.Linear layer.
    """

    def __init__(self, in_features, out_features, s=30.0, m=0.35):
        super(ArcMarginProduct, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.m = m
        self.weight = nn.Parameter(torch.FloatTensor(out_features, in_features))
        nn.init.xavier_uniform_(self.weight)

        # This tells PyTorch to include 's' in the state_dict
        self.register_buffer('s', torch.tensor(s))

    def forward(self, features, label=None, return_raw_logits=False):

        # 1. Normalize and compute cosine similarity (the expensive part, done only once)
        cosine = F.linear(F.normalize(features), F.normalize(self.weight))

        # The raw, unpenalized logits are simply the scaled cosine similarity.
        # These are what we need for metrics.
        raw_logits = self.s * cosine

        # 2. Handle inference case (no label) or if only raw logits are needed.
        # If no label is provided, we can't apply a margin, so we must return raw logits.
        if label is None:
            return raw_logits

        # 3. Apply margin logic for the training path
        theta = torch.acos(torch.clamp(cosine, -1.0 + 1e-7, 1.0 - 1e-7))
        one_hot = torch.zeros(cosine.size(), device=features.device)
        one_hot.scatter_(1, label.view(-1, 1).long(), 1)
        M_theta = torch.where(one_hot.bool(), theta + self.m, theta)
        marginal_target_logit = torch.cos(M_theta)

        # The final, penalized logits for the loss function.
        final_logits = self.s * marginal_target_logit

        # 4. Return what the caller asked for
        if return_raw_logits:
            # Return both the final (for loss) and raw (for metrics) logits
            return final_logits, raw_logits
        else:
            # Default behavior: just return the final logits
            return final_logits

    def __repr__(self):
        return self.__class__.__name__ + '(' \
            + 'in_features=' + str(self.in_features) \
            + ', out_features=' + str(self.out_features) \
            + ', s=' + str(self.s) \
            + ', m=' + str(self.m) + ')'


class FocalLoss(nn.Module):
    """
    Focal Loss, as described in https://arxiv.org/abs/1708.02002.
    It is used to address the issue of class imbalance and difficulty imbalance.
    """

    def __init__(self, gamma=2.0, alpha=None, reduction='mean'):
        super(FocalLoss, self).__init__()
        self.gamma = gamma
        self.alpha = alpha
        self.reduction = reduction

    # ++ MODIFIED SIGNATURE: Added optional 'reduction' argument ++
    def forward(self, inputs, targets, reduction=None):
        """
        Args:
            inputs: model predictions (logits) of shape [N, C]
            targets: ground truth labels of shape [N]
            reduction (str, optional): Overrides the default reduction method.
                                       Can be 'mean', 'sum', or 'none'.
        """
        # Determine which reduction to use: the one passed in, or the default
        reduction = reduction if reduction is not None else self.reduction

        # Calculate cross-entropy loss without reduction (this part is correct)
        ce_loss = F.cross_entropy(inputs, targets, reduction='none')

        # Get the probability of the correct class
        pt = torch.exp(-ce_loss)

        # Calculate the focal loss term
        focal_loss = (1 - pt) ** self.gamma * ce_loss

        # Apply alpha weighting for class imbalance
        if self.alpha is not None:
            alpha_t = torch.where(targets == 1, self.alpha, 1 - self.alpha)
            focal_loss = alpha_t * focal_loss

        # Apply the determined reduction
        if reduction == 'mean':
            return focal_loss.mean()
        elif reduction == 'sum':
            return focal_loss.sum()
        else:  # This handles 'none'
            return focal_loss


class CrossEntropyLossWithReduction(nn.Module):
    """
    Standard CrossEntropyLoss that allows the 'reduction' parameter
    to be passed during the forward call. This is necessary for training
    strategies like Group-DRO that require per-sample losses.

    Supports ``label_smoothing`` (PyTorch ≥ 1.10) for calibration.
    """

    def __init__(self, reduction='mean', label_smoothing: float = 0.0):
        super(CrossEntropyLossWithReduction, self).__init__()
        self.default_reduction = reduction
        self.label_smoothing = label_smoothing

    def forward(self, inputs, targets, reduction=None):
        """
        Args:
            inputs: model predictions (logits) of shape [N, C]
            targets: ground truth labels of shape [N]
            reduction (str, optional): Overrides the default reduction method.
                                       Can be 'mean', 'sum', or 'none'.
        """
        reduction_to_use = reduction if reduction is not None else self.default_reduction
        return F.cross_entropy(
            inputs, targets,
            reduction=reduction_to_use,
            label_smoothing=self.label_smoothing,
        )


class OpenCLIPVisionModelWrapper(nn.Module):
    """
    Wrapper to make OpenCLIP visual encoder output compatible with HuggingFace format.
    
    HuggingFace CLIPVisionModel returns: {'pooler_output': tensor, 'last_hidden_state': tensor, ...}
    OpenCLIP visual encoder returns: tensor (pooled output)
    
    This wrapper standardizes the output format.
    """
    
    def __init__(self, openclip_visual, intermediate_layer=None):
        super().__init__()
        self.visual = openclip_visual
        # NEW (P17): if intermediate_layer is set, hook that resblock and
        # capture its [CLS] token as the model's "pooled" output. The full
        # forward still runs but the head reads the captured features.
        self.intermediate_layer = intermediate_layer
        self._captured = None
        if intermediate_layer is not None:
            blocks = openclip_visual.transformer.resblocks
            assert 0 <= intermediate_layer < len(blocks), \
                f"intermediate_layer={intermediate_layer} out of range [0, {len(blocks)})"
            blocks[intermediate_layer].register_forward_hook(self._capture_hook)
            logger.info(f"OpenCLIPVisionModelWrapper: intermediate-layer readout at block {intermediate_layer}")

    def _capture_hook(self, module, inputs, output):
        # OpenCLIP resblock output shape: (seq, batch, dim) by default.
        # The [CLS] token is index 0 in the seq dimension.
        if output.dim() == 3:
            if output.shape[0] >= output.shape[1]:
                cls = output[0]      # seq-first: take token 0
            else:
                cls = output[:, 0]   # batch-first
        elif output.dim() == 2:
            cls = output
        else:
            raise RuntimeError(f"unexpected resblock output shape: {output.shape}")
        self._captured = cls

    def forward(self, pixel_values, **kwargs):
        """
        Forward pass that returns HuggingFace-compatible output dict.

        Args:
            pixel_values: Input images tensor [B, C, H, W]

        Returns:
            dict with 'pooler_output' key containing the pooled features
        """
        # OpenCLIP visual encoder returns pooled output directly
        # The visual encoder's forward method signature varies by version
        # Most OpenCLIP models: visual(x) returns the pooled CLS token
        pooled_output = self.visual(pixel_values)

        # NEW (P17): if intermediate-layer hook is active, return that as the
        # primary "pooler_output" — head reads from intermediate features.
        if self.intermediate_layer is not None and self._captured is not None:
            return {
                'pooler_output': self._captured.clone(),
                'final_pooler_output': pooled_output,  # kept for diagnostics
            }
        return {
            'pooler_output': pooled_output,
        }
    
    def named_modules(self, *args, **kwargs):
        """Delegate named_modules to the wrapped visual encoder."""
        return self.visual.named_modules(*args, **kwargs)
    
    def named_parameters(self, *args, **kwargs):
        """Delegate named_parameters to the wrapped visual encoder."""
        return self.visual.named_parameters(*args, **kwargs)
    
    def modules(self, *args, **kwargs):
        """Delegate modules to the wrapped visual encoder."""
        return self.visual.modules(*args, **kwargs)


# =============================================================================
# Gradient Reversal Layer & Quality-Domain Adversarial Head
# =============================================================================

class GradientReversalFunction(torch.autograd.Function):
    """Reverses gradient by factor lambda during backprop (DANN, Ganin 2015)."""

    @staticmethod
    def forward(ctx, x, lambda_val):
        ctx.lambda_val = lambda_val
        return x.clone()

    @staticmethod
    def backward(ctx, grad_output):
        return -ctx.lambda_val * grad_output, None


class GradientReversalLayer(nn.Module):
    def __init__(self, lambda_val: float = 1.0):
        super().__init__()
        self.lambda_val = lambda_val

    def set_lambda(self, val: float):
        self.lambda_val = val

    def forward(self, x):
        return GradientReversalFunction.apply(x, self.lambda_val)


class QualityDomainHead(nn.Module):
    """
    Small MLP that predicts quality domain (or method-domain) from backbone
    features. Attached via a gradient reversal layer so the backbone learns to
    REMOVE that information from its representations.

    Number of output domains is controlled by the ``num_domains`` constructor
    arg (set from the yaml ``quality_domain_count`` field). The labels passed
    in must lie in [0, num_domains).

    Two domain conventions are supported in the codebase:

    LEGACY 4-class (matches `DOMAIN_MAP` below; matches
    ``data.sources.combined_paired.QUALITY_DOMAIN_MAP``):
        0 = clean_academic  (DF40 reals — soft, smooth, low-noise)
        1 = webcam_codec    (VCD reals, external webcam — sharp, noisy, codec artifacts)
        2 = studio_capture  (DeepLive/VisoMaster reals — studio lighting, variable quality)
        3 = social_media    (YouTube reals — heavier compression, variable resolution)

    NEW 12-class (Phase 3 method-conditional GRL; canonical map at
    ``data.sources.method_domain_map.METHOD_DOMAIN_NAMES``):
        0 = df40, 1 = deeplive_basic, 2 = deeplive_enhanced (Phase 1A axis),
        3 = deeplive_teams, 4 = visomaster_inswapper, 5 = visomaster_ghost,
        6 = visomaster_other, 7 = visomaster_enhanced (reserved),
        8 = visomaster_teams_recap (reserved),
        9 = proper_visomaster_clean (reserved),
        10 = external_vcd_real, 11 = realpool_real.

    The 4-class DOMAIN_MAP attribute below is kept for backwards-compat
    consistency with ``QUALITY_DOMAIN_MAP`` in combined_paired.py (legacy
    callers + the test_domain_map_consistency check). New code should use
    ``data.sources.method_domain_map.lookup_method_domain_with_label`` and pass
    ``quality_domain_count: 12`` in the yaml.
    """

    DOMAIN_MAP = {
        "df40": 0,
        "external": 1,
        "deeplive": 2,
        "visomaster": 2,
        "deeplive_teams": 1,
        "youtube": 3,
    }

    def __init__(self, in_features: int, num_domains: int = 4, hidden_dim: int = 128):
        super().__init__()
        self.grl = GradientReversalLayer()
        self.classifier = nn.Sequential(
            nn.Linear(in_features, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(hidden_dim, num_domains),
        )

    def set_lambda(self, val: float):
        self.grl.set_lambda(val)

    def forward(self, features):
        reversed_features = self.grl(features)
        return self.classifier(reversed_features)


@DETECTOR.register_module(module_name='effort')
class EffortDetector(nn.Module):
    def __init__(self, config=None):
        super(EffortDetector, self).__init__()
        self.config = config
        
        # Lambda regularization with optional annealing
        # lambda_reg controls orthogonal constraint strength (Effort method)
        self.lambda_reg = config.get('lambda_reg', 1.0)  # Default to 1.0 if not in config
        
        # Lambda annealing parameters (anneal from start to end over N steps)
        self.lambda_reg_start = config.get('lambda_reg_start', self.lambda_reg)
        self.lambda_reg_end = config.get('lambda_reg_end', self.lambda_reg)
        self.lambda_reg_anneal_steps = config.get('lambda_reg_anneal_steps', 0)
        
        # If annealing is configured, start at lambda_reg_start
        if self.lambda_reg_anneal_steps > 0:
            self.lambda_reg = self.lambda_reg_start
            logger.info(f"Lambda annealing ENABLED: {self.lambda_reg_start} → {self.lambda_reg_end} over {self.lambda_reg_anneal_steps} steps")
        
        self.rank = config.get('rank', 1023)
        
        # Get backbone path - support both old and new config structures
        self.clip_backbone_path = self._resolve_backbone_path(config)
        
        # Get hidden size from backbone config or use default
        self.hidden_size = self._get_hidden_size(config)
        
        self.backbone = self.build_backbone(config)  # Initialize Backbone model

        # Controlled initialization of the head based on config
        self.use_arcface_head = config.get('use_arcface_head', False)
        
        # Feature normalization before head (helps stabilize ArcFace)
        self.normalize_features_before_head = config.get('normalize_features_before_head', False)
        if self.normalize_features_before_head:
            logger.info("Feature normalization ENABLED before head (L2 norm to unit length)")
        
        if self.use_arcface_head:
            s = config.get('arcface_s', 30.0)
            m = config.get('arcface_m', 0.35)

            # Annealing parameters
            self.s_start = config.get('s_start', s)
            self.s_end = config.get('s_end', s)
            self.anneal_steps = config.get('anneal_steps', 0)

            initial_s = self.s_start if self.anneal_steps > 0 else s

            logger.info(f"Using ArcFace head with s={initial_s} (initial) and m={m}")
            if self.anneal_steps > 0:
                logger.info(f"Annealing s from {self.s_start} to {self.s_end} over {self.anneal_steps} steps.")

            self.head = ArcMarginProduct(in_features=self.hidden_size, out_features=2, s=initial_s, m=m)
            logger.info(f"[INIT] ArcFace head created with in_features={self.hidden_size}")
        else:
            logger.info("Using standard Linear head")
            self.head = nn.Linear(self.hidden_size, 2)

        # Embedding-space mixup (R10+): interpolate features before the head
        # to smooth the decision boundary.  Gated by mixup_alpha (default 0 = off).
        self.mixup_alpha = config.get('mixup_alpha', 0.0)
        if self.mixup_alpha > 0:
            logger.info(f"Embedding-space mixup ENABLED: alpha={self.mixup_alpha}")

        # Feature-norm regularization (packet-6B 2026-04-23): pulls real and fake
        # feature-norm distributions toward each other so ‖feat‖ stops being a
        # usable cue. Targets the diagnosed shortcut where low-norm enhancement-
        # style real frames get classified as fake. Default 0 = off.
        self.feat_norm_reg_lambda = config.get('feat_norm_reg_lambda', 0.0)
        if self.feat_norm_reg_lambda > 0:
            logger.info(f"Feature-norm regularization ENABLED: lambda={self.feat_norm_reg_lambda}")

        # Controlled initialization of the loss function
        # If ArcFace is used, we MUST use CrossEntropyLoss, not FocalLoss.
        # The margin 'm' in ArcFace serves a similar purpose to Focal Loss's gamma.
        self.label_smoothing = config.get('label_smoothing', 0.0)
        if self.use_arcface_head:
            logger.info("ArcFace head is active. Switching to standard CrossEntropyLoss.")
            self.loss_func = CrossEntropyLossWithReduction(label_smoothing=self.label_smoothing)
        else:
            # Setup loss function for non-ArcFace case (may use Focal Loss)
            self._setup_loss_function(config, logger)

        # Quality domain adversarial head (optional — gradient reversal for
        # quality-invariant representations, per REAL_ROBUSTNESS_PLAN Part C)
        self.use_quality_head = config.get('use_quality_domain_head', False)
        self.quality_domain_loss_weight = config.get('quality_domain_loss_weight', 0.1)
        self.quality_domain_require_labels = config.get('quality_domain_require_labels', True)
        self._quality_head_warning_emitted = False
        if self.use_quality_head:
            num_domains = config.get('quality_domain_count', 4)
            self.quality_head = QualityDomainHead(
                in_features=self.hidden_size,
                num_domains=num_domains,
                hidden_dim=config.get('quality_head_hidden_dim', 128),
            )
            logger.info(
                f"Quality domain adversarial head ENABLED: "
                f"{num_domains} domains, hidden_dim={config.get('quality_head_hidden_dim', 128)}, "
                f"loss_weight={self.quality_domain_loss_weight}, "
                f"require_labels={self.quality_domain_require_labels}"
            )

        # Initialize tracking variables and log parameter analysis
        self._setup_tracking_vars()
    
    def update_lambda_reg(self, step_cnt: int) -> float:
        """
        Updates lambda_reg based on annealing schedule if configured.
        
        Args:
            step_cnt: Current training step
            
        Returns:
            Current lambda_reg value
        """
        if self.lambda_reg_anneal_steps <= 0:
            return self.lambda_reg
        
        if step_cnt <= self.lambda_reg_anneal_steps:
            # Linear annealing from start to end
            progress = step_cnt / self.lambda_reg_anneal_steps
            self.lambda_reg = self.lambda_reg_start + progress * (self.lambda_reg_end - self.lambda_reg_start)
        else:
            # Fix at end value after annealing completes
            self.lambda_reg = self.lambda_reg_end
        
        return self.lambda_reg

    def _setup_loss_function(self, config: dict, logger) -> None:
        """
        Sets up the loss function based on configuration.

        This is called from __init__ if not using ArcFace head.
        """
        self.use_focal_loss = config.get('use_focal_loss', False)
        if self.use_focal_loss:
            gamma = config.get('focal_loss_gamma', 2.0)
            alpha = config.get('focal_loss_alpha', None)
            logger.info(f"Using Focal Loss with gamma={gamma} and alpha={alpha}")
            self.loss_func = FocalLoss(gamma=gamma, alpha=alpha)
        else:
            logger.info("Using standard CrossEntropyLoss")
            self.loss_func = CrossEntropyLossWithReduction(label_smoothing=self.label_smoothing)

        # Optional: shortcut-avoidance correlation penalty (added 2026-05-05)
        from loss.correlation_penalty import build_correlation_penalty_from_config
        self.corr_penalty = build_correlation_penalty_from_config(config)
        if self.corr_penalty is not None:
            logger.info(
                f"Correlation penalty ENABLED: lambda={self.corr_penalty.lambda_}, "
                f"axes={self.corr_penalty.axes}"
            )
        else:
            logger.info("Correlation penalty disabled (no `correlation_penalty.enabled: true` in config)")

    def _setup_tracking_vars(self) -> None:
        """Initialize tracking variables for metrics."""
        self.prob, self.label = [], []
        self.correct, self.total = 0, 0
        
        # Flag to enable first-batch dimension verification 
        self._first_forward_done = False
        
        # === DIAGNOSTIC: Log trainable parameter breakdown ===
        total_params = 0
        trainable_params = 0
        trainable_by_component = {}
        trainable_param_names = []
        for name, param in self.named_parameters():
            total_params += param.numel()
            if param.requires_grad:
                trainable_params += param.numel()
                trainable_param_names.append((name, param.shape, param.numel()))
                # Extract component name (e.g., 'backbone', 'head')
                component = name.split('.')[0] if '.' in name else name
                trainable_by_component[component] = trainable_by_component.get(component, 0) + param.numel()
        
        logger.info(f"="*60)
        logger.info(f"🔬 PARAMETER ANALYSIS")
        logger.info(f"   Total parameters: {total_params:,}")
        logger.info(f"   Trainable parameters: {trainable_params:,}")
        logger.info(f"   Frozen parameters: {total_params - trainable_params:,}")
        logger.info(f"   Trainable ratio: {100*trainable_params/total_params:.2f}%")
        logger.info(f"   Trainable by component:")
        for comp, count in sorted(trainable_by_component.items(), key=lambda x: -x[1]):
            logger.info(f"      {comp}: {count:,}")
        
        # Log first 30 trainable parameter names to help debug
        logger.info(f"   First {min(30, len(trainable_param_names))} trainable parameters:")
        for name, shape, numel in trainable_param_names[:30]:
            logger.info(f"      {name}: {list(shape)} ({numel:,})")
        if len(trainable_param_names) > 30:
            logger.info(f"      ... and {len(trainable_param_names) - 30} more")
        logger.info(f"="*60)
    
    def _resolve_backbone_path(self, config) -> str:
        """
        Resolve the backbone path from config.
        
        Supports:
        1. Direct backbone_path in config (HuggingFace model ID like 'openai/clip-vit-base-patch16')
        2. New backbone config: config['backbone']['local_path_override'] or resolved from registry
        3. Legacy config: config['gcs_assets']['clip_backbone']['local_path']
        
        Returns:
            str: Path to the backbone model (can be HuggingFace ID or local path)
        """
        # Check for direct backbone_path (HuggingFace model ID)
        if config.get('backbone_path'):
            path = config['backbone_path']
            logger.info(f"Using backbone_path from config: {path}")
            return path
        
        # Try new backbone config first
        backbone_config = config.get('backbone', {})
        if backbone_config:
            # Check for explicit override
            local_override = backbone_config.get('local_path_override')
            if local_override:
                logger.info(f"Using backbone from explicit override: {local_override}")
                return local_override
            
            # Check for huggingface_id in backbone config
            hf_id = backbone_config.get('huggingface_id')
            if hf_id:
                logger.info(f"Using HuggingFace model ID: {hf_id}")
                return hf_id
        
        # Fall back to gcs_assets path (either resolved from registry or legacy config)
        gcs_assets = config.get('gcs_assets', {})
        clip_backbone = gcs_assets.get('clip_backbone', {})
        local_path = clip_backbone.get('local_path')
        
        if local_path:
            logger.info(f"Using backbone from gcs_assets: {local_path}")
            return local_path
        
        # Final fallback to default
        default_path = "./weights/models--openai--clip-vit-large-patch14/"
        logger.warning(f"No backbone path found in config. Using default: {default_path}")
        return default_path
    
    def _get_hidden_size(self, config) -> int:
        """
        Get the hidden size for the backbone model.
        
        Supports:
        1. Explicit config['backbone']['hidden_size']
        2. Inferred from variant in backbone_registry
        3. Default based on known variants
        
        Returns:
            int: Hidden size for the head layer
        """
        # Known hidden sizes by variant
        VARIANT_HIDDEN_SIZES = {
            'ViT-B-16': 768,
            'ViT-B-32': 768,  # Fixed: was incorrectly 512, but vision hidden_size is 768
            'ViT-L-14': 1024,
            'ViT-L-14-336': 1024,
            'ViT-H-14': 1280,
            'ViT-G-14': 1664,
        }
        
        # Try explicit backbone config
        backbone_config = config.get('backbone', {})
        if backbone_config:
            # Check for explicit hidden_size
            if 'hidden_size' in backbone_config:
                return backbone_config['hidden_size']
            
            # Try to infer from variant
            variant = backbone_config.get('variant', 'ViT-L-14')
            if variant in VARIANT_HIDDEN_SIZES:
                return VARIANT_HIDDEN_SIZES[variant]
            
            # Check backbone registry
            source = backbone_config.get('source', 'openai')
            registry = config.get('backbone_registry', {})
            source_entry = registry.get(source, {})
            variant_entry = source_entry.get(variant, {})
            if 'hidden_size' in variant_entry:
                return variant_entry['hidden_size']
        
        # Default to ViT-L-14 hidden size
        return 1024

    def build_backbone(self, config):
        """
        Build the vision backbone.
        
        Supports:
        1. HuggingFace transformers CLIP models (source: openai)
        2. OpenCLIP models (source: laion/openclip)
        
        The backbone path is set by _resolve_backbone_path() in __init__.
        """
        # Log backbone configuration
        backbone_config = config.get('backbone', {})
        source = backbone_config.get('source', 'openai') if backbone_config else 'openai'
        
        if backbone_config:
            logger.info(f"--- Backbone Configuration ---")
            logger.info(f"   Type: {backbone_config.get('type', 'clip')}")
            logger.info(f"   Variant: {backbone_config.get('variant', 'ViT-L-14')}")
            logger.info(f"   Source: {source}")
            logger.info(f"   Resolution: {backbone_config.get('resolution', 224)}")
            logger.info(f"   Hidden Size: {self.hidden_size}")
        
        # Route to appropriate loader based on source
        if source in ('laion', 'openclip'):
            return self._build_openclip_backbone(config, backbone_config)
        else:
            return self._build_huggingface_backbone(config)
    
    def _build_huggingface_backbone(self, config):
        """Build backbone using HuggingFace transformers (OpenAI CLIP models)."""
        logger.info(f"Loading HuggingFace CLIP backbone from: {self.clip_backbone_path}")
        logger.info(f"Current working directory: {os.getcwd()}")
        
        # Check if it's a HuggingFace model ID (contains '/') or a local path
        is_hf_model_id = '/' in self.clip_backbone_path and not os.path.exists(self.clip_backbone_path)
        backbone_config = config.get('backbone', {}) if config else {}
        forced_type = backbone_config.get('type')
        use_clip_vision = False

        if forced_type in ('clip_vision', 'clip_vision_model'):
            use_clip_vision = True
            logger.info(f"Backbone type forced to vision-only: {forced_type}")
        elif not is_hf_model_id:
            try:
                local_cfg = AutoConfig.from_pretrained(
                    self.clip_backbone_path,
                    local_files_only=True
                )
                if getattr(local_cfg, 'model_type', None) == 'clip_vision_model':
                    use_clip_vision = True
                    logger.info("Detected CLIPVisionModel config in local backbone path.")
            except Exception as e:
                logger.warning(f"Could not inspect local backbone config: {e}")
        
        if use_clip_vision:
            if is_hf_model_id:
                logger.info(f"Downloading CLIPVisionModel from HuggingFace Hub: {self.clip_backbone_path}")
                vision_model = CLIPVisionModel.from_pretrained(
                    self.clip_backbone_path,
                    local_files_only=False
                )
            else:
                logger.info(f"Loading CLIPVisionModel from local path: {self.clip_backbone_path}")
                vision_model = CLIPVisionModel.from_pretrained(
                    self.clip_backbone_path,
                    local_files_only=True
                )

            logger.info(f"Applying SVD with rank={self.rank}")
            vision_model = apply_svd_residual_to_self_attn(vision_model, r=self.rank)
            log_svd_layer_dimensions(vision_model, configured_rank=self.rank)
            return vision_model

        if is_hf_model_id:
            logger.info(f"Downloading model from HuggingFace Hub: {self.clip_backbone_path}")
            clip_model = CLIPModel.from_pretrained(
                self.clip_backbone_path,
                local_files_only=False
            )
        else:
            logger.info(f"Loading model from local path: {self.clip_backbone_path}")
            clip_model = CLIPModel.from_pretrained(
                self.clip_backbone_path,
                local_files_only=True
            )

        # Apply SVD to self_attn layers
        logger.info(f"Applying SVD with rank={self.rank}")
        clip_model.vision_model = apply_svd_residual_to_self_attn(clip_model.vision_model, r=self.rank)
        
        # === DIAGNOSTIC: Verify SVD layer dimensions (Jan 11, 2026) ===
        log_svd_layer_dimensions(clip_model.vision_model, configured_rank=self.rank)

        return clip_model.vision_model
    
    def _build_openclip_backbone(self, config, backbone_config):
        """
        Build backbone using OpenCLIP library (LAION models).
        
        OpenCLIP models have different module structure than HuggingFace:
        - HuggingFace: model.vision_model.encoder.layers[i].self_attn
        - OpenCLIP: model.visual.transformer.resblocks[i].attn
        
        We wrap the OpenCLIP visual encoder to return HuggingFace-compatible output.
        """
        try:
            import open_clip
        except ImportError:
            raise ImportError(
                "OpenCLIP library required for LAION backbones. "
                "Install with: pip install open_clip_torch"
            )
        
        # Get OpenCLIP-specific config
        # Support both dedicated keys (openclip_model/openclip_pretrained) and
        # checkpoint-embedded keys (model_name/pretrained).
        openclip_model = (backbone_config.get('openclip_model')
                          or backbone_config.get('model_name', 'ViT-B-16'))
        openclip_pretrained = (backbone_config.get('openclip_pretrained')
                               or backbone_config.get('pretrained', 'datacomp_xl_s13b_b90k'))
        
        logger.info(f"Loading OpenCLIP backbone: {openclip_model} (pretrained: {openclip_pretrained})")
        
        # Check if we have a local cache path
        local_path = backbone_config.get('local_path_override') or self.clip_backbone_path
        
        # Try to load from local cache first, otherwise download
        try:
            if local_path and os.path.exists(local_path):
                logger.info(f"Loading from local cache: {local_path}")
                
                # OpenCLIP expects a direct path to the weight file, not a directory
                # Check if local_path is a directory and find the weight file
                weight_file = local_path
                if os.path.isdir(local_path):
                    # Look for common OpenCLIP weight file names
                    possible_names = [
                        'open_clip_pytorch_model.bin',
                        'pytorch_model.bin',
                        'model.pt',
                        'open_clip_model.pt',
                    ]
                    for name in possible_names:
                        candidate = os.path.join(local_path, name)
                        if os.path.exists(candidate):
                            weight_file = candidate
                            logger.info(f"Found weight file: {weight_file}")
                            break
                    else:
                        # No known weight file found, list directory contents for debugging
                        contents = os.listdir(local_path)
                        logger.warning(f"No known weight file in {local_path}. Contents: {contents}")
                        # Try to find any .bin or .pt file
                        for f in contents:
                            if f.endswith('.bin') or f.endswith('.pt'):
                                weight_file = os.path.join(local_path, f)
                                logger.info(f"Using weight file: {weight_file}")
                                break
                
                model, _, _ = open_clip.create_model_and_transforms(
                    openclip_model, 
                    pretrained=weight_file
                )
            else:
                logger.info(f"Downloading OpenCLIP model...")
                model, _, _ = open_clip.create_model_and_transforms(
                    openclip_model, 
                    pretrained=openclip_pretrained
                )
        except Exception as e:
            logger.error(f"Failed to load OpenCLIP model: {e}")
            raise
        
        # Get the visual encoder
        visual_encoder = model.visual
        
        # Apply SVD to attention layers
        # OpenCLIP uses 'attn' instead of 'self_attn'
        # Check if we should apply SVD to in_proj_weight (q, k, v) in addition to out_proj
        apply_to_in_proj = backbone_config.get('apply_svd_to_in_proj', True)  # Default True for full coverage
        svd_blocks = backbone_config.get('svd_blocks', None)  # Optional: only apply to specific blocks
        apply_svd_to_mlp = backbone_config.get('apply_svd_to_mlp', False)  # NEW: apply SVD to MLP layers
        logger.info(f"Applying SVD with rank={self.rank} (OpenCLIP structure)")
        logger.info(f"   apply_svd_to_in_proj={apply_to_in_proj} (covers q, k, v projections)")
        logger.info(f"   apply_svd_to_mlp={apply_svd_to_mlp} (covers MLP fc layers)")
        if svd_blocks is not None:
            logger.info(f"   svd_blocks={svd_blocks} (selective block coverage)")
        visual_encoder = apply_svd_residual_to_openclip_attn(
            visual_encoder, r=self.rank, apply_to_in_proj=apply_to_in_proj,
            block_indices=svd_blocks, apply_to_mlp=apply_svd_to_mlp
        )
        
        # === NEW: Optionally unfreeze final projection and LayerNorm ===
        unfreeze_final_proj = backbone_config.get('unfreeze_final_proj', False)
        unfreeze_final_ln = backbone_config.get('unfreeze_final_ln', False)
        
        if unfreeze_final_proj:
            # OpenCLIP visual.proj is the final 768->output_dim projection
            if hasattr(visual_encoder, 'proj') and visual_encoder.proj is not None:
                visual_encoder.proj.requires_grad = True
                logger.info(f"✓ Unfroze visual.proj: {visual_encoder.proj.shape}")
            else:
                logger.warning("unfreeze_final_proj=True but visual.proj not found")
        
        if unfreeze_final_ln:
            # OpenCLIP visual.ln_post is the final LayerNorm
            if hasattr(visual_encoder, 'ln_post'):
                for param in visual_encoder.ln_post.parameters():
                    param.requires_grad = True
                logger.info(f"✓ Unfroze visual.ln_post")
            else:
                logger.warning("unfreeze_final_ln=True but visual.ln_post not found")

        # P17 frozen intermediate-layer readout: SVD residual components are
        # trainable by default after SVD wrapping. Freeze them explicitly when
        # the experiment is intended to train only the new head.
        if backbone_config.get('freeze_svd_residuals', False):
            frozen_params = 0
            frozen_tensors = 0
            for name, param in visual_encoder.named_parameters():
                if any(svd_name in name for svd_name in ('U_residual', 'S_residual', 'V_residual')):
                    param.requires_grad = False
                    frozen_params += param.numel()
                    frozen_tensors += 1
            logger.info(
                "Frozen SVD residual backbone params: "
                f"{frozen_params:,} parameters across {frozen_tensors} tensors"
            )
        
        # === DIAGNOSTIC: Verify SVD layer dimensions (Jan 11, 2026) ===
        # This helps catch mismatched rank configs (e.g., rank=511 for 768-dim layers)
        log_svd_layer_dimensions(visual_encoder, configured_rank=self.rank)
        
        # Wrap to make output compatible with HuggingFace format
        # NEW (P17): pass intermediate_layer through if set in yaml
        intermediate_layer = backbone_config.get('intermediate_layer', None)
        wrapped_encoder = OpenCLIPVisionModelWrapper(visual_encoder, intermediate_layer=intermediate_layer)

        return wrapped_encoder

    def features(self, data_dict: dict) -> torch.tensor:
        feat = self.backbone(data_dict['image'])['pooler_output']
        return feat

    def classifier(self, features: torch.tensor) -> torch.tensor:
        return self.head(features)

    def compute_weight_loss(self):
        weight_sum_dict = {}
        num_weight_dict = {}
        for name, module in self.backbone.named_modules():
            if isinstance(module, SVDResidualLinear):
                weight_curr = module.compute_current_weight()
                if str(weight_curr.size()) not in weight_sum_dict.keys():
                    weight_sum_dict[str(weight_curr.size())] = weight_curr
                    num_weight_dict[str(weight_curr.size())] = 1
                else:
                    weight_sum_dict[str(weight_curr.size())] += weight_curr
                    num_weight_dict[str(weight_curr.size())] += 1

        loss2 = 0.0
        for k in weight_sum_dict.keys():
            _, S_sum, _ = torch.linalg.svd(weight_sum_dict[k], full_matrices=False)
            loss2 += -torch.mean(S_sum)
        loss2 /= len(weight_sum_dict.keys())
        return loss2

    def get_losses(self, data_dict: dict, pred_dict: dict, reduction: str = 'mean') -> dict:
        """
        Calculates losses. Supports both mean reduction (default) and per-sample
        reduction for advanced training strategies like Group-DRO.

        Args:
            data_dict (dict): Dictionary containing ground truth labels.
            pred_dict (dict): Dictionary containing model predictions.
            reduction (str): The reduction to apply to the loss.
                             'mean': returns a single scalar loss.
                             'none': returns a loss for each sample in the batch.

        Returns:
            dict: A dictionary of losses. The 'overall' key will contain a
                  scalar or a tensor depending on the reduction method.
                  
        Loss Breakdown (for diagnostic logging):
            - cls_loss: Raw classification loss (CrossEntropy/Focal) BEFORE regularization
            - reg_loss: Total regularization term (lambda_reg * avg(orthogonal + keepsv))
            - orthogonal_loss: Sum of orthogonal losses across all SVDResidualLinear layers
            - keepsv_loss: Sum of keepsv losses across all SVDResidualLinear layers
            - overall = cls_loss + reg_loss
        """
        label = data_dict['label']
        pred = pred_dict['cls']

        # Handle reshaped video batches by repeating labels for each frame
        if pred.shape[0] > label.shape[0]:
            B = label.shape[0]
            T = pred.shape[0] // B
            label = label.repeat_interleave(T)

        # --- Calculate Regularization Terms with DETAILED BREAKDOWN ---
        # Initialize all regularization tracking tensors on the correct device
        device = pred.device
        reg_term = torch.tensor(0.0, device=device)
        orthogonal_loss_total = torch.tensor(0.0, device=device)
        keepsv_loss_total = torch.tensor(0.0, device=device)
        num_reg = 0
        
        if self.training:
            lambda_reg = self.lambda_reg
            
            for module in self.backbone.modules():
                # Handle standard SVDResidualLinear (out_proj and HuggingFace q, k, v)
                if isinstance(module, SVDResidualLinear):
                    # Get individual loss components
                    orth_loss = module.compute_orthogonal_loss()
                    keep_loss = module.compute_keepsv_loss()
                    
                    # Handle case where loss might be a float 0.0 instead of tensor
                    # (happens when S_residual is None)
                    if isinstance(orth_loss, (int, float)):
                        orth_loss = torch.tensor(float(orth_loss), device=device)
                    if isinstance(keep_loss, (int, float)):
                        keep_loss = torch.tensor(float(keep_loss), device=device)
                    
                    orthogonal_loss_total = orthogonal_loss_total + orth_loss
                    keepsv_loss_total = keepsv_loss_total + keep_loss
                    num_reg += 1
                
                # Handle SVDInProjLinear (OpenCLIP fused q, k, v projections)
                if isinstance(module, SVDInProjLinear):
                    orth_loss = module.compute_orthogonal_loss()
                    keep_loss = module.compute_keepsv_loss()
                    
                    if isinstance(orth_loss, (int, float)):
                        orth_loss = torch.tensor(float(orth_loss), device=device)
                    if isinstance(keep_loss, (int, float)):
                        keep_loss = torch.tensor(float(keep_loss), device=device)
                    
                    orthogonal_loss_total = orthogonal_loss_total + orth_loss
                    keepsv_loss_total = keepsv_loss_total + keep_loss
                    # Count as 3 layers since it contains q, k, v
                    num_reg += 3
            
            if num_reg > 0:
                # reg_term = lambda_reg * (sum of all losses) / num_layers
                raw_reg_sum = orthogonal_loss_total + keepsv_loss_total
                reg_term = lambda_reg * raw_reg_sum / num_reg

        # --- Quality domain adversarial loss (gradient reversal) ---
        quality_loss_raw = torch.tensor(0.0, device=device)
        quality_loss = torch.tensor(0.0, device=device)
        quality_domain_has_logits = torch.tensor(0.0, device=device)
        quality_domain_has_labels = torch.tensor(0.0, device=device)
        quality_domain_unique_count = torch.tensor(0.0, device=device)
        if self.use_quality_head and self.training:
            has_quality_logits = 'quality_domain_logits' in pred_dict
            quality_domain_has_logits = torch.tensor(
                1.0 if has_quality_logits else 0.0, device=device
            )
            domain_labels = data_dict.get('quality_domain')

            if domain_labels is not None and not isinstance(domain_labels, torch.Tensor):
                try:
                    domain_labels = torch.as_tensor(domain_labels, dtype=torch.long)
                except Exception as exc:
                    msg = (
                        "Quality head enabled but `quality_domain` labels could not be "
                        f"converted to a tensor: {exc}"
                    )
                    if self.quality_domain_require_labels:
                        raise RuntimeError(msg)
                    if not self._quality_head_warning_emitted:
                        logger.warning(msg)
                        self._quality_head_warning_emitted = True
                    domain_labels = None

            if not has_quality_logits:
                msg = (
                    "Quality head enabled but `quality_domain_logits` is missing from pred_dict. "
                    "Ensure model.forward adds quality-domain logits during training."
                )
                if self.quality_domain_require_labels:
                    raise RuntimeError(msg)
                if not self._quality_head_warning_emitted:
                    logger.warning(msg)
                    self._quality_head_warning_emitted = True
            elif domain_labels is None:
                msg = (
                    "Quality head enabled but `quality_domain` is missing from data_dict. "
                    "Ensure the dataloader/collate function provides quality-domain labels."
                )
                if self.quality_domain_require_labels:
                    raise RuntimeError(msg)
                if not self._quality_head_warning_emitted:
                    logger.warning(msg)
                    self._quality_head_warning_emitted = True
            elif domain_labels.numel() == 0:
                msg = (
                    "Quality head enabled but `quality_domain` tensor is empty. "
                    "Cannot compute quality-domain adversarial loss on an empty batch."
                )
                if self.quality_domain_require_labels:
                    raise RuntimeError(msg)
                if not self._quality_head_warning_emitted:
                    logger.warning(msg)
                    self._quality_head_warning_emitted = True
            else:
                quality_domain_has_labels = torch.tensor(1.0, device=device)
                domain_labels = domain_labels.to(device)
                quality_domain_unique_count = torch.tensor(
                    float(domain_labels.unique().numel()), device=device
                )
                # Expand for video batches
                if pred_dict['quality_domain_logits'].shape[0] > domain_labels.shape[0]:
                    B_q = domain_labels.shape[0]
                    T_q = pred_dict['quality_domain_logits'].shape[0] // B_q
                    domain_labels = domain_labels.repeat_interleave(T_q)
                quality_loss_raw = F.cross_entropy(
                    pred_dict['quality_domain_logits'],
                    domain_labels.long(),
                )
                quality_loss = quality_loss_raw * self.quality_domain_loss_weight

        # --- Feature-norm regularization (aux loss) ---
        # Penalize divergence between real-side and fake-side feature norms so the
        # model can't use ‖feat‖ as a shortcut cue. Diagnosis (2026-04-23 dor): the
        # current model places enhancement-style reals (low ‖feat‖) inside the fake
        # cluster purely on norm; this loss pulls the two distributions together.
        feat_norm_loss = torch.tensor(0.0, device=device)
        if self.feat_norm_reg_lambda > 0 and self.training:
            feat = pred_dict.get('feat', None)
            if feat is not None and feat.dim() == 2 and feat.shape[0] == label.shape[0]:
                feat_norms = feat.norm(dim=-1)
                mask_real_fn = label == 0
                mask_fake_fn = label == 1
                if mask_real_fn.sum() > 0 and mask_fake_fn.sum() > 0:
                    real_norms = feat_norms[mask_real_fn]
                    fake_norms = feat_norms[mask_fake_fn]
                    feat_norm_loss = (
                        (real_norms.mean() - fake_norms.mean()).pow(2)
                        + 0.1 * (real_norms.var() + fake_norms.var())
                    )

        # --- Main Loss Calculation based on reduction type ---
        # Check for embedding-space mixup metadata
        _has_mixup = '_mixup_lam' in pred_dict

        if reduction == 'mean':
            # --- DEFAULT BEHAVIOR: Return a single scalar loss ---
            if _has_mixup:
                # Mixup: blend CE losses for the two label sets
                lam = pred_dict['_mixup_lam']
                idx = pred_dict['_mixup_index']
                label_b = label[idx]
                cls_loss = lam * self.loss_func(pred, label) + (1.0 - lam) * self.loss_func(pred, label_b)
            else:
                cls_loss = self.loss_func(pred, label)  # Classification loss ONLY

            # Combined loss = classification + regularization + quality reversal + feat-norm reg
            overall_loss = cls_loss + reg_term if self.training else cls_loss
            if self.training:
                overall_loss = overall_loss + quality_loss
                if self.feat_norm_reg_lambda > 0:
                    overall_loss = overall_loss + self.feat_norm_reg_lambda * feat_norm_loss

            # Correlation-penalty regularizer (added 2026-05-05) — penalize batch
            # Pearson(score, axis) for each configured nuisance axis.
            corr_penalty_loss = torch.tensor(0.0, device=device)
            corr_per_axis_r: dict = {}
            if self.training and self.corr_penalty is not None:
                from loss.correlation_penalty import compute_pixel_axes
                # P(fake) — softmax over class dim, take fake column
                score = pred.softmax(dim=-1)[:, 1]
                axis_values = compute_pixel_axes(data_dict['image'])
                corr_penalty_loss, corr_per_axis_r = self.corr_penalty(score, axis_values)
                overall_loss = overall_loss + corr_penalty_loss

            # For logging, calculate separate real/fake losses
            mask_real = label == 0
            mask_fake = label == 1
            loss_real = self.loss_func(pred[mask_real], label[mask_real]) if mask_real.sum() > 0 else torch.tensor(0.0, device=device)
            loss_fake = self.loss_func(pred[mask_fake], label[mask_fake]) if mask_fake.sum() > 0 else torch.tensor(0.0, device=device)

            loss_dict = {
                'overall': overall_loss,
                'real_loss': loss_real,
                'fake_loss': loss_fake,
                # === DIAGNOSTIC LOSS COMPONENTS ===
                'cls_loss': cls_loss.detach(),  # Classification loss before regularization
                'reg_loss': reg_term.detach() if isinstance(reg_term, torch.Tensor) else torch.tensor(reg_term, device=device),
                'orthogonal_loss': orthogonal_loss_total.detach() if isinstance(orthogonal_loss_total, torch.Tensor) else torch.tensor(orthogonal_loss_total, device=device),
                'keepsv_loss': keepsv_loss_total.detach() if isinstance(keepsv_loss_total, torch.Tensor) else torch.tensor(keepsv_loss_total, device=device),
                'num_svd_layers': torch.tensor(float(num_reg), device=device),  # For sanity checking
                # === KEY DIAGNOSTIC RATIO: If this grows >> 1.0, regularization is dominating ===
                'reg_cls_ratio': (reg_term / (cls_loss + 1e-8)).detach() if self.training else torch.tensor(0.0, device=device),
                'quality_domain_loss_raw': quality_loss_raw.detach(),
                'quality_domain_loss': quality_loss.detach(),
                'quality_domain_has_logits': quality_domain_has_logits.detach(),
                'quality_domain_has_labels': quality_domain_has_labels.detach(),
                'quality_domain_unique_count': quality_domain_unique_count.detach(),
                'feat_norm_loss': feat_norm_loss.detach(),
                'corr_penalty_loss': corr_penalty_loss.detach(),
                **{f'corr_r_{ax}': r.detach() for ax, r in corr_per_axis_r.items()},
            }

        elif reduction == 'none':
            # --- NEW BEHAVIOR: Return per-sample losses for Group-DRO ---
            if _has_mixup:
                lam = pred_dict['_mixup_lam']
                idx = pred_dict['_mixup_index']
                label_b = label[idx]
                per_sample_cls_loss = (
                    lam * self.loss_func(pred, label, reduction='none')
                    + (1.0 - lam) * self.loss_func(pred, label_b, reduction='none')
                )
            else:
                per_sample_cls_loss = self.loss_func(pred, label, reduction='none')

            # Add scalar regularization term + quality loss + feat-norm reg (PyTorch broadcasts)
            per_sample_loss = per_sample_cls_loss + reg_term if self.training else per_sample_cls_loss
            if self.training:
                per_sample_loss = per_sample_loss + quality_loss
                if self.feat_norm_reg_lambda > 0:
                    per_sample_loss = per_sample_loss + self.feat_norm_reg_lambda * feat_norm_loss

            # Correlation-penalty regularizer (scalar, broadcasts across per-sample losses)
            corr_penalty_loss = torch.tensor(0.0, device=device)
            corr_per_axis_r: dict = {}
            if self.training and self.corr_penalty is not None:
                from loss.correlation_penalty import compute_pixel_axes
                score = pred.softmax(dim=-1)[:, 1]
                axis_values = compute_pixel_axes(data_dict['image'])
                corr_penalty_loss, corr_per_axis_r = self.corr_penalty(score, axis_values)
                per_sample_loss = per_sample_loss + corr_penalty_loss

            # For logging, calculate the mean of the per-sample losses for each class
            mask_real = label == 0
            mask_fake = label == 1
            loss_real = per_sample_loss[mask_real].mean() if mask_real.sum() > 0 else torch.tensor(0.0, device=device)
            loss_fake = per_sample_loss[mask_fake].mean() if mask_fake.sum() > 0 else torch.tensor(0.0, device=device)

            loss_dict = {
                'overall': per_sample_loss,  # This is a TENSOR
                'real_loss': loss_real.detach(),  # Scalar for logging
                'fake_loss': loss_fake.detach(),  # Scalar for logging
                # === DIAGNOSTIC LOSS COMPONENTS (scalar summaries) ===
                'cls_loss': per_sample_cls_loss.mean().detach(),  # Mean classification loss
                'reg_loss': reg_term.detach() if isinstance(reg_term, torch.Tensor) else torch.tensor(reg_term, device=device),
                'orthogonal_loss': orthogonal_loss_total.detach() if isinstance(orthogonal_loss_total, torch.Tensor) else torch.tensor(orthogonal_loss_total, device=device),
                'keepsv_loss': keepsv_loss_total.detach() if isinstance(keepsv_loss_total, torch.Tensor) else torch.tensor(keepsv_loss_total, device=device),
                'num_svd_layers': torch.tensor(float(num_reg), device=device),
                # === KEY DIAGNOSTIC RATIO ===
                'reg_cls_ratio': (reg_term / (per_sample_cls_loss.mean() + 1e-8)).detach() if self.training else torch.tensor(0.0, device=device),
                'quality_domain_loss_raw': quality_loss_raw.detach(),
                'quality_domain_loss': quality_loss.detach(),
                'quality_domain_has_logits': quality_domain_has_logits.detach(),
                'quality_domain_has_labels': quality_domain_has_labels.detach(),
                'quality_domain_unique_count': quality_domain_unique_count.detach(),
                'feat_norm_loss': feat_norm_loss.detach(),
                'corr_penalty_loss': corr_penalty_loss.detach(),
                **{f'corr_r_{ax}': r.detach() for ax, r in corr_per_axis_r.items()},
            }
        else:
            raise ValueError(f"Unsupported reduction type: '{reduction}'. Must be 'mean' or 'none'.")

        return loss_dict

    def get_train_metrics(self, data_dict: dict, pred_dict: dict) -> dict:
        label = data_dict['label']
        # --- Use 'raw_logits' for metrics ---
        pred = pred_dict['raw_logits']

        # If predictions are per-frame (B*T) and labels are per-video (B),
        # we must expand the labels to match the predictions for per-frame metric calculation.
        # This aligns with how the loss is calculated in get_losses().
        if pred.shape[0] > label.shape[0]:
            B = label.shape[0]
            # Calculate T (number of frames) from the discrepancy
            T = pred.shape[0] // B
            # Repeat each label T times to match the reshaped predictions
            label = label.repeat_interleave(T)

        # compute metrics for batch data
        # Now, `label` and `pred` will have compatible shapes
        auc, eer, acc, ap = calculate_metrics_for_train(label.detach(), pred.detach())
        
        # --- Diagnostic logging for debugging 50% accuracy issue ---
        # Log every 100 batches to avoid spam
        if not hasattr(self, '_train_metrics_counter'):
            self._train_metrics_counter = 0
        self._train_metrics_counter += 1
        
        if self._train_metrics_counter % 100 == 1:
            with torch.no_grad():
                # Check logit statistics
                logit_diff = pred[:, 1] - pred[:, 0]  # fake_logit - real_logit
                pred_class = torch.argmax(pred, dim=1)
                real_mask = (label == 0)
                fake_mask = (label == 1)
                
                logger.info(f"[DIAG batch #{self._train_metrics_counter}] Label distribution: real={real_mask.sum().item()}, fake={fake_mask.sum().item()}")
                logger.info(f"[DIAG batch #{self._train_metrics_counter}] Logit diff stats (fake-real): mean={logit_diff.mean().item():.4f}, std={logit_diff.std().item():.4f}")
                logger.info(f"[DIAG batch #{self._train_metrics_counter}] Predictions: pred_fake={pred_class.sum().item()}, pred_real={(pred_class==0).sum().item()}")
                
                if real_mask.any():
                    real_pred_correct = ((pred_class == 0) & real_mask).sum().item()
                    logger.info(f"[DIAG batch #{self._train_metrics_counter}] Real frames: {real_pred_correct}/{real_mask.sum().item()} predicted correctly")
                
                if fake_mask.any():
                    fake_pred_correct = ((pred_class == 1) & fake_mask).sum().item()
                    logger.info(f"[DIAG batch #{self._train_metrics_counter}] Fake frames: {fake_pred_correct}/{fake_mask.sum().item()} predicted correctly")
                
                # Feature statistics
                feat = pred_dict.get('feat')
                if feat is not None:
                    logger.info(f"[DIAG batch #{self._train_metrics_counter}] Feature shape: {feat.shape}, norm mean: {feat.norm(dim=1).mean().item():.4f}")
        
        metric_batch_dict = {'acc': acc, 'auc': auc, 'eer': eer, 'ap': ap}
        return metric_batch_dict

    def forward(self, data_dict: dict, inference=False) -> dict:
        image = data_dict['image']
        label = data_dict.get('label', None)

        # 1. Handle video tensor shape
        # Input for video is [B, T, C, H, W], e.g., [1, 32, 3, 224, 224]
        is_video = image.dim() == 5
        if is_video:
            B, T, C, H, W = image.shape
            # Reshape to a single batch of frames: [B * T, C, H, W], e.g., [32, 3, 224, 224]
            image = image.view(B * T, C, H, W)
            # --- Ensure labels are also expanded if present ---
            if label is not None:
                label = label.repeat_interleave(T)

        # 2. Extract features directly from the backbone
        # This is the core fix: Pass the reshaped 'image' tensor directly (new)
        # to the backbone, avoiding the 'features' helper method and the
        # intermediate dictionary. This ensures the entire batch is processed.
        features = self.backbone(image)['pooler_output']
        
        # First forward dimension verification - log only once
        if not self._first_forward_done:
            self._first_forward_done = True
            logger.info(f"[FIRST FORWARD] Input image shape: {data_dict['image'].shape}")
            logger.info(f"[FIRST FORWARD] Reshaped to: {image.shape}")
            logger.info(f"[FIRST FORWARD] Backbone output (features) shape: {features.shape}")
            logger.info(f"[FIRST FORWARD] Expected hidden_size: {self.hidden_size}")
            if features.shape[-1] != self.hidden_size:
                logger.error(f"[DIMENSION MISMATCH] Backbone output dim {features.shape[-1]} != expected hidden_size {self.hidden_size}")
            if label is not None:
                logger.info(f"[FIRST FORWARD] Label shape after expand: {label.shape}")
                unique_labels = label.unique().tolist()
                logger.info(f"[FIRST FORWARD] Unique labels in batch: {unique_labels}")

        # Optional: Normalize features before head (helps stabilize ArcFace training)
        if self.normalize_features_before_head:
            features = F.normalize(features, p=2, dim=1)
            if not self._first_forward_done:
                logger.info(f"[FIRST FORWARD] Features normalized to unit length")

        # --- Embedding-space mixup (R10+) ---
        # Interpolate backbone features before the classification head.
        # When active, we skip the ArcFace angular margin (use raw cosine logits)
        # because the margin assumes one-hot labels.  The mixed CE loss in
        # get_losses() handles the soft label combination instead.
        _mixup_active = (
            self.mixup_alpha > 0
            and self.training
            and not inference
            and label is not None
        )
        if _mixup_active:
            lam = float(np.random.beta(self.mixup_alpha, self.mixup_alpha))
            shuffle_idx = torch.randperm(features.size(0), device=features.device)
            features = lam * features + (1.0 - lam) * features[shuffle_idx]

        # 3. Pass features through the appropriate head
        if self.use_arcface_head:
            if inference:
                # During inference, we only need the raw logits.
                # `label` is passed as None to the head.
                raw_logits = self.head(features, label=None)
                pred_for_loss = raw_logits
            elif _mixup_active:
                # Mixup: skip angular margin — use raw cosine logits for both
                # loss and metrics.  The mixed CE is computed in get_losses().
                raw_logits = self.head(features, label=None)
                pred_for_loss = raw_logits
            else:  # Normal training (no mixup)
                # Make ONE call to get both penalized and raw logits efficiently.
                pred_for_loss, raw_logits = self.head(features, label=label, return_raw_logits=True)
        else:
            # Standard linear head logic
            # The original self.classifier(features) is just self.head(features)
            raw_logits = self.head(features)
            pred_for_loss = raw_logits

        # 4. Calculate probabilities from raw (unpenalized) logits
        prob = torch.softmax(raw_logits, dim=1)[:, 1]

        pred_dict = {
            'cls': pred_for_loss,
            'prob': prob,
            'feat': features,
            'raw_logits': raw_logits
        }

        # Store mixup metadata so get_losses() can compute the mixed CE
        if _mixup_active:
            pred_dict['_mixup_lam'] = lam
            pred_dict['_mixup_index'] = shuffle_idx

        # 5. Quality domain prediction (gradient reversal head)
        if self.use_quality_head and not inference:
            pred_dict['quality_domain_logits'] = self.quality_head(features)

        return pred_dict


# Custom module to represent the residual using SVD components
class SVDResidualLinear(nn.Module):
    def __init__(self, in_features, out_features, r, bias=True, init_weight=None):
        super(SVDResidualLinear, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.r = r  # Number of top singular values to exclude

        # Original weights (fixed)
        self.weight_main = nn.Parameter(torch.Tensor(out_features, in_features), requires_grad=False)
        if init_weight is not None:
            self.weight_main.data.copy_(init_weight)
        else:
            nn.init.kaiming_uniform_(self.weight_main, a=math.sqrt(5))

        # Bias
        if bias:
            self.bias = nn.Parameter(torch.Tensor(out_features))
            nn.init.zeros_(self.bias)
        else:
            self.register_parameter('bias', None)

    def compute_current_weight(self):
        if self.S_residual is not None:
            return self.weight_main + self.U_residual @ torch.diag(self.S_residual) @ self.V_residual
        else:
            return self.weight_main

    @property
    def weight(self):
        """
        Property to return the effective weight (main + residual).
        
        This is required for compatibility with nn.MultiheadAttention, which
        directly accesses `out_proj.weight` in its forward() method via
        F.multi_head_attention_forward. Without this property, replacing
        out_proj with SVDResidualLinear would break MHA.
        """
        return self.compute_current_weight()

    def forward(self, x):
        if hasattr(self, 'U_residual') and hasattr(self, 'V_residual') and self.S_residual is not None:
            # Reconstruct the residual weight
            residual_weight = self.U_residual @ torch.diag(self.S_residual) @ self.V_residual
            # Total weight is the fixed main weight plus the residual
            weight = self.weight_main + residual_weight
        else:
            # If residual components are not set, use only the main weight
            weight = self.weight_main

        return F.linear(x, weight, self.bias)

    def compute_orthogonal_loss(self):
        if self.S_residual is not None:
            # According to the properties of orthogonal matrices: A^TA = I
            UUT = torch.cat((self.U_r, self.U_residual), dim=1) @ torch.cat((self.U_r, self.U_residual), dim=1).t()
            VVT = torch.cat((self.V_r, self.V_residual), dim=0) @ torch.cat((self.V_r, self.V_residual), dim=0).t()
            # print(self.U_r.size(), self.U_residual.size())  # torch.Size([1024, 1023]) torch.Size([1024, 1])
            # print(self.V_r.size(), self.V_residual.size())  # torch.Size([1023, 1024]) torch.Size([1, 1024])
            # UUT = self.U_residual @ self.U_residual.t()
            # VVT = self.V_residual @ self.V_residual.t()

            # Construct an identity matrix
            UUT_identity = torch.eye(UUT.size(0), device=UUT.device)
            VVT_identity = torch.eye(VVT.size(0), device=VVT.device)

            # Using frobenius norm to compute loss
            loss = 0.5 * torch.norm(UUT - UUT_identity, p='fro') + 0.5 * torch.norm(VVT - VVT_identity, p='fro')
        else:
            loss = 0.0

        return loss

    def compute_keepsv_loss(self):
        if (self.S_residual is not None) and (self.weight_original_fnorm is not None):
            # Total current weight is the fixed main weight plus the residual
            weight_current = self.weight_main + self.U_residual @ torch.diag(self.S_residual) @ self.V_residual
            # Frobenius norm of current weight
            weight_current_fnorm = torch.norm(weight_current, p='fro')

            loss = torch.abs(weight_current_fnorm ** 2 - self.weight_original_fnorm ** 2)
            # loss = torch.abs(weight_current_fnorm ** 2 + 0.01 * self.weight_main_fnorm ** 2 - 1.01 * self.weight_original_fnorm ** 2)
        else:
            loss = 0.0

        return loss

    def compute_fn_loss(self):
        if (self.S_residual is not None):
            weight_current = self.weight_main + self.U_residual @ torch.diag(self.S_residual) @ self.V_residual
            weight_current_fnorm = torch.norm(weight_current, p='fro')

            loss = weight_current_fnorm ** 2
        else:
            loss = 0.0

        return loss


def log_svd_layer_dimensions(model, configured_rank: int):
    """
    DIAGNOSTIC (Jan 11, 2026): Log the dimensions of all SVD layers.
    
    This function helps verify that the rank configuration is correct for the actual
    layer dimensions. The Effort paper uses k=1 trainable singular direction as default.
    
    If attention layers are 768x768 (ViT-B-16) but rank=511, you get k=768-511=257
    trainable directions instead of k=1, which causes regularization loss to dominate.
    
    CORRECT CONFIG:
    - For 768x768 attention layers: rank=767 → k=1
    - For 1024x1024 attention layers: rank=1023 → k=1
    - For 512x512 layers: rank=511 → k=1
    
    Args:
        model: The model containing SVDResidualLinear and SVDInProjLinear layers
        configured_rank: The rank value from config
    """
    logger.info("=" * 70)
    logger.info("🔬 SVD LAYER DIMENSION ANALYSIS (CRITICAL FOR EFFORT METHOD)")
    logger.info("=" * 70)
    logger.info(f"   Configured rank: {configured_rank}")
    logger.info(f"   Effort paper default: k=1 trainable singular direction")
    logger.info("-" * 70)
    
    svd_layers = []
    svd_in_proj_layers = []
    total_k = 0
    
    for name, module in model.named_modules():
        # Handle SVDResidualLinear (out_proj and HuggingFace q, k, v)
        if isinstance(module, SVDResidualLinear):
            if hasattr(module, 'U_residual') and module.U_residual is not None:
                U_shape = tuple(module.U_residual.shape)
                S_shape = tuple(module.S_residual.shape)
                V_shape = tuple(module.V_residual.shape)
                k = S_shape[0]  # Number of trainable singular directions
                d = U_shape[0]  # Full dimension (out_features)
                
                svd_layers.append({
                    'name': name,
                    'd': d,
                    'k': k,
                    'U_shape': U_shape,
                    'V_shape': V_shape,
                    'params': module.U_residual.numel() + module.S_residual.numel() + module.V_residual.numel()
                })
                total_k += k
        
        # Handle SVDInProjLinear (OpenCLIP fused q, k, v)
        if isinstance(module, SVDInProjLinear):
            for proj_name, proj in [('q', module.svd_q), ('k', module.svd_k), ('v', module.svd_v)]:
                if hasattr(proj, 'U_residual') and proj.U_residual is not None:
                    U_shape = tuple(proj.U_residual.shape)
                    S_shape = tuple(proj.S_residual.shape)
                    V_shape = tuple(proj.V_residual.shape)
                    k = S_shape[0]
                    d = U_shape[0]
                    
                    svd_in_proj_layers.append({
                        'name': f"{name}.svd_{proj_name}",
                        'd': d,
                        'k': k,
                        'U_shape': U_shape,
                        'V_shape': V_shape,
                        'params': proj.U_residual.numel() + proj.S_residual.numel() + proj.V_residual.numel()
                    })
                    total_k += k
    
    # Combine all layers for reporting
    all_svd_layers = svd_layers + svd_in_proj_layers
    
    if not all_svd_layers:
        logger.warning("   ⚠️ NO SVD layers found with residual parameters!")
        logger.info("=" * 70)
        return
    
    # Log summary
    first_layer = all_svd_layers[0]
    logger.info(f"   Found {len(svd_layers)} SVDResidualLinear layers (out_proj)")
    logger.info(f"   Found {len(svd_in_proj_layers)} SVDInProjLinear sublayers (q, k, v)")
    logger.info(f"   Total SVD layers: {len(all_svd_layers)}")
    logger.info(f"   Layer dimension (d): {first_layer['d']}")
    logger.info(f"   Trainable singular directions per layer (k): {first_layer['k']}")
    logger.info(f"   Total trainable directions across all layers: {total_k}")
    
    # Check if k matches Effort paper recommendation
    if first_layer['k'] == 1:
        logger.info(f"   ✅ CORRECT: k=1 matches Effort paper default")
    else:
        logger.warning(f"   ⚠️ WARNING: k={first_layer['k']} != 1 (Effort paper default)")
        logger.warning(f"   ⚠️ This means you're training {first_layer['k']}× more directions than intended!")
        correct_rank = first_layer['d'] - 1
        logger.warning(f"   ⚠️ For k=1 with {first_layer['d']}-dim layers, set rank={correct_rank}")
    
    # Log first few layers as examples
    logger.info("-" * 70)
    logger.info("   First 3 SVD layers (examples):")
    for layer_info in all_svd_layers[:3]:
        logger.info(f"      {layer_info['name']}")
        logger.info(f"         U_residual: {layer_info['U_shape']}, V_residual: {layer_info['V_shape']}")
        logger.info(f"         d={layer_info['d']}, k={layer_info['k']}, params={layer_info['params']:,}")
    
    # Total params in SVD layers
    total_svd_params = sum(l['params'] for l in all_svd_layers)
    logger.info("-" * 70)
    logger.info(f"   Total trainable params in SVD layers: {total_svd_params:,}")
    expected_params_k1 = len(all_svd_layers) * (first_layer['d'] + 1 + first_layer['d'])  # U[d,1] + S[1] + V[1,d]
    logger.info(f"   Expected if k=1: ~{expected_params_k1:,}")
    if total_svd_params > expected_params_k1 * 2:
        logger.warning(f"   ⚠️ SVD params are {total_svd_params / expected_params_k1:.1f}× higher than k=1 baseline!")
    logger.info("=" * 70)


# Function to replace nn.Linear modules within self_attn modules with SVDResidualLinear
def apply_svd_residual_to_self_attn(model, r):
    for name, module in model.named_children():
        if 'self_attn' in name:
            # Replace nn.Linear layers in this module
            for sub_name, sub_module in module.named_modules():
                if isinstance(sub_module, nn.Linear):
                    # Get parent module within self_attn
                    parent_module = module
                    sub_module_names = sub_name.split('.')
                    for module_name in sub_module_names[:-1]:
                        parent_module = getattr(parent_module, module_name)
                    # Replace the nn.Linear layer with SVDResidualLinear
                    setattr(parent_module, sub_module_names[-1], replace_with_svd_residual(sub_module, r))
        else:
            # Recursively apply to child modules
            apply_svd_residual_to_self_attn(module, r)
    # After replacing, set requires_grad for residual components
    for param_name, param in model.named_parameters():
        if any(x in param_name for x in ['S_residual', 'U_residual', 'V_residual']):
            param.requires_grad = True
        else:
            param.requires_grad = False
    return model


# Function to apply SVD to OpenCLIP models
# OpenCLIP uses different module naming: 'attn' instead of 'self_attn'
# Structure: visual.transformer.resblocks[i].attn (MultiheadAttention)


class SVDMultiheadAttentionWrapper(nn.Module):
    """
    Wrapper for nn.MultiheadAttention that applies SVD decomposition to the output projection.
    
    This wrapper is necessary because nn.MultiheadAttention uses F.multi_head_attention_forward
    which directly accesses out_proj.weight, making it incompatible with SVDResidualLinear.
    
    Instead, we keep the original MHA and apply SVD as a post-processing step on the output.
    """
    
    def __init__(self, mha_module, rank):
        super().__init__()
        self.mha = mha_module
        self.embed_dim = mha_module.embed_dim
        self.num_heads = mha_module.num_heads
        
        # Store original out_proj for reference
        out_features = mha_module.out_proj.out_features
        in_features = mha_module.out_proj.in_features
        
        # Create SVD residual layer for the output projection
        # We'll apply this AFTER the MHA forward pass
        self.svd_out_proj = SVDResidualLinear(
            in_features, out_features, rank, 
            bias=False,  # MHA already has bias in out_proj
            init_weight=mha_module.out_proj.weight.data.clone()
        )
        
        # Perform SVD on the original out_proj weight
        U, S, Vh = torch.linalg.svd(mha_module.out_proj.weight.data, full_matrices=False)
        r = min(rank, len(S))
        
        # Keep top r singular components (main weight)
        U_r = U[:, :r]
        S_r = S[:r]
        Vh_r = Vh[:r, :]
        
        # Reconstruct the main weight (fixed)
        weight_main = U_r @ torch.diag(S_r) @ Vh_r
        
        # Store in SVD module
        self.svd_out_proj.weight_main = nn.Parameter(weight_main, requires_grad=False)
        self.svd_out_proj.weight_original_fnorm = torch.norm(mha_module.out_proj.weight.data, p='fro')
        self.svd_out_proj.weight_main_fnorm = torch.norm(weight_main, p='fro')
        
        # Set up residual components
        self.svd_out_proj.U_r = nn.Parameter(U_r, requires_grad=False)
        self.svd_out_proj.V_r = nn.Parameter(Vh_r, requires_grad=False)
        
        # Initialize residual singular values
        remaining_S = S[r:] if len(S) > r else torch.zeros(1)
        self.svd_out_proj.S_residual = nn.Parameter(remaining_S[:1] if len(remaining_S) > 0 else torch.zeros(1), requires_grad=True)
        
        # Initialize residual U and V
        U_remaining = U[:, r:] if U.shape[1] > r else torch.zeros(out_features, 1)
        V_remaining = Vh[r:, :] if Vh.shape[0] > r else torch.zeros(1, in_features)
        self.svd_out_proj.U_residual = nn.Parameter(U_remaining[:, :1] if U_remaining.shape[1] > 0 else torch.randn(out_features, 1) * 0.01, requires_grad=True)
        self.svd_out_proj.V_residual = nn.Parameter(V_remaining[:1, :] if V_remaining.shape[0] > 0 else torch.randn(1, in_features) * 0.01, requires_grad=True)
        
        # Store original bias
        self.out_proj_bias = mha_module.out_proj.bias
        
        # Zero out the original MHA's out_proj weight (we'll use our SVD version)
        # But keep the bias
        mha_module.out_proj.weight.requires_grad = False
        nn.init.zeros_(mha_module.out_proj.weight)
        
    def forward(self, query, key=None, value=None, key_padding_mask=None, 
                need_weights=False, attn_mask=None, average_attn_weights=True,
                is_causal=False):
        """Forward pass with SVD applied to output projection."""
        
        # Handle the case where key and value are not provided
        if key is None:
            key = query
        if value is None:
            value = query
            
        # Run original MHA (but with zeroed out_proj weight, so output is just the bias)
        # Actually, we need to be smarter - let's compute attention output WITHOUT out_proj
        # and then apply our SVD out_proj
        
        # The MHA forward will apply its internal computations
        # Since we zeroed the out_proj weight, the output will just have the bias
        attn_output, attn_weights = self.mha(
            query, key, value,
            key_padding_mask=key_padding_mask,
            need_weights=need_weights,
            attn_mask=attn_mask,
            average_attn_weights=average_attn_weights,
            is_causal=is_causal
        )
        
        # The attn_output currently has shape [L, N, E] or [N, L, E] depending on batch_first
        # We need to apply our SVD projection
        # But wait - the MHA already applied the zeroed out_proj, so we need to "undo" and redo
        
        # Actually, let's use a different approach: just apply our SVD weight directly
        # attn_output = attn_output - self.out_proj_bias  # Remove bias
        
        # Compute the SVD-based output
        svd_weight = self.svd_out_proj.weight_main + \
                     self.svd_out_proj.U_residual @ torch.diag(self.svd_out_proj.S_residual) @ self.svd_out_proj.V_residual
        
        # Apply the SVD weight (attn_output is [L, N, E] for non-batch_first)
        # We need to apply linear transformation on the last dimension
        attn_output = F.linear(attn_output, svd_weight, self.out_proj_bias)
        
        if need_weights:
            return attn_output, attn_weights
        return attn_output, None


class SVDInProjLinear(nn.Module):
    """
    SVD decomposition for fused in_proj_weight in nn.MultiheadAttention.
    
    nn.MultiheadAttention stores q, k, v projections in a single fused weight:
    in_proj_weight has shape [3*embed_dim, embed_dim] where:
      - rows 0:embed_dim are Q projection
      - rows embed_dim:2*embed_dim are K projection  
      - rows 2*embed_dim:3*embed_dim are V projection
    
    This module applies SVD residual learning to each projection independently,
    providing 3x more trainable directions compared to out_proj only.
    
    The module exposes a `weight` property that reconstructs the fused weight
    from the three SVD components, maintaining compatibility with
    F.multi_head_attention_forward.
    """
    
    def __init__(self, in_proj_weight, in_proj_bias, embed_dim, r):
        super().__init__()
        self.embed_dim = embed_dim
        self.r = r
        
        # Split the fused weight into q, k, v
        W_q = in_proj_weight[:embed_dim, :].clone()       # [embed_dim, embed_dim]
        W_k = in_proj_weight[embed_dim:2*embed_dim, :].clone()  # [embed_dim, embed_dim]
        W_v = in_proj_weight[2*embed_dim:, :].clone()     # [embed_dim, embed_dim]
        
        # Create SVD components for each projection
        self.svd_q = self._create_svd_components(W_q, r, "q_proj")
        self.svd_k = self._create_svd_components(W_k, r, "k_proj")
        self.svd_v = self._create_svd_components(W_v, r, "v_proj")
        
        # Handle bias (also fused)
        if in_proj_bias is not None:
            self.bias = nn.Parameter(in_proj_bias.clone())
        else:
            self.register_parameter('bias', None)
    
    def _create_svd_components(self, weight, r, name):
        """Create SVD decomposition components for a single projection weight."""
        components = nn.Module()
        components.weight_original_fnorm = torch.norm(weight, p='fro')
        
        # Perform SVD
        U, S, Vh = torch.linalg.svd(weight, full_matrices=False)
        r = min(r, len(S))
        
        # Top r singular components (fixed main weight)
        U_r = U[:, :r]
        S_r = S[:r]
        Vh_r = Vh[:r, :]
        weight_main = U_r @ torch.diag(S_r) @ Vh_r
        
        components.weight_main = nn.Parameter(weight_main, requires_grad=False)
        components.weight_main_fnorm = torch.norm(weight_main, p='fro')
        components.U_r = nn.Parameter(U_r.clone(), requires_grad=False)
        components.V_r = nn.Parameter(Vh_r.clone(), requires_grad=False)
        
        # Residual components (trainable)
        U_residual = U[:, r:]
        S_residual = S[r:]
        Vh_residual = Vh[r:, :]
        
        if len(S_residual) > 0:
            components.S_residual = nn.Parameter(S_residual.clone(), requires_grad=True)
            components.U_residual = nn.Parameter(U_residual.clone(), requires_grad=True)
            components.V_residual = nn.Parameter(Vh_residual.clone(), requires_grad=True)
        else:
            components.S_residual = nn.Parameter(torch.zeros(1), requires_grad=True)
            components.U_residual = nn.Parameter(torch.randn(weight.shape[0], 1) * 0.01, requires_grad=True)
            components.V_residual = nn.Parameter(torch.randn(1, weight.shape[1]) * 0.01, requires_grad=True)
        
        return components
    
    def _compute_weight(self, svd_components):
        """Compute effective weight from SVD components."""
        if svd_components.S_residual is not None:
            residual = svd_components.U_residual @ torch.diag(svd_components.S_residual) @ svd_components.V_residual
            return svd_components.weight_main + residual
        return svd_components.weight_main
    
    @property
    def weight(self):
        """
        Reconstruct the fused in_proj_weight from individual SVD components.
        
        This property is accessed by F.multi_head_attention_forward.
        Returns shape [3*embed_dim, embed_dim].
        """
        W_q = self._compute_weight(self.svd_q)
        W_k = self._compute_weight(self.svd_k)
        W_v = self._compute_weight(self.svd_v)
        return torch.cat([W_q, W_k, W_v], dim=0)
    
    def compute_orthogonal_loss(self):
        """Compute orthogonality loss for all three projections."""
        loss = 0.0
        for svd_comp in [self.svd_q, self.svd_k, self.svd_v]:
            if svd_comp.S_residual is not None:
                UUT = torch.cat((svd_comp.U_r, svd_comp.U_residual), dim=1) @ \
                      torch.cat((svd_comp.U_r, svd_comp.U_residual), dim=1).t()
                VVT = torch.cat((svd_comp.V_r, svd_comp.V_residual), dim=0) @ \
                      torch.cat((svd_comp.V_r, svd_comp.V_residual), dim=0).t()
                UUT_identity = torch.eye(UUT.size(0), device=UUT.device)
                VVT_identity = torch.eye(VVT.size(0), device=VVT.device)
                loss += 0.5 * torch.norm(UUT - UUT_identity, p='fro') + \
                        0.5 * torch.norm(VVT - VVT_identity, p='fro')
        return loss / 3.0  # Average over q, k, v
    
    def compute_keepsv_loss(self):
        """Compute singular value preservation loss for all three projections."""
        loss = 0.0
        for svd_comp in [self.svd_q, self.svd_k, self.svd_v]:
            if svd_comp.S_residual is not None and svd_comp.weight_original_fnorm is not None:
                weight_current = self._compute_weight(svd_comp)
                weight_current_fnorm = torch.norm(weight_current, p='fro')
                loss += torch.abs(weight_current_fnorm ** 2 - svd_comp.weight_original_fnorm ** 2)
        return loss / 3.0


def _install_svd_in_proj_routing(mha_module):
    """
    Patch an nn.MultiheadAttention instance so its forward routes the in_proj
    weight through the autograd-tracked ``self._svd_in_proj.weight`` property,
    instead of through the leaf ``self.in_proj_weight`` parameter.

    -------------------------------------------------------------------------
    BUG CONTEXT (silent zero-gradient on in_proj SVD residuals; fixed 2026-04-26)
    -------------------------------------------------------------------------
    Prior implementation registered a ``forward_pre_hook`` that did

        module.in_proj_weight.data.copy_(module._svd_in_proj.weight)

    before each MHA forward. ``in_proj_weight`` is a leaf ``nn.Parameter`` with
    ``requires_grad=False`` (intentionally frozen by design). ``.data.copy_``
    writes values in place WITHOUT participating in autograd: the weight tensor
    that ``F.multi_head_attention_forward`` actually reads is never connected to
    the SVD residual parameters in the autograd graph.

    Consequence: ``_svd_in_proj.svd_{q,k,v}.{U_residual, V_residual,
    S_residual}`` received gradient ONLY from ``compute_orthogonal_loss`` /
    ``compute_keepsv_loss`` (when ``lambda_reg > 0``), never from the
    classification loss. They drifted toward orthogonality but learned nothing
    about the task. Adam silently skipped them whenever grad was None.

    Every R12 / RLP / P-* run with ``apply_svd_to_in_proj=True`` was affected.
    P8A's anchor-pool improvement therefore came from the MLP-SVD path +
    unfrozen ``visual.proj`` + unfrozen ``ln_post`` only — the in_proj-SVD
    capacity it was supposed to test was a no-op.

    -------------------------------------------------------------------------
    THE FIX
    -------------------------------------------------------------------------
    Override the MHA instance's forward to call
    ``F.multi_head_attention_forward`` with ``self._svd_in_proj.weight``
    (an autograd-tracked ``@property`` that recomputes the fused weight from
    the residual parameters on each access). The frozen leaf
    ``self.in_proj_weight`` becomes unused; we keep it on the module purely
    for checkpoint backward-compatibility.

    The MLP-SVD path (``SVDResidualLinear``) and the out_proj-SVD path
    (``SVDResidualLinear`` replacing ``out_proj``) are unaffected — they
    already route through autograd-tracked tensors via their ``forward`` and
    ``weight`` property respectively.
    """
    import types
    import torch.nn.functional as F  # noqa: F401  (F is imported at module top; alias here for clarity)

    def forward(self, query, key=None, value=None,
                key_padding_mask=None, need_weights=True, attn_mask=None,
                average_attn_weights=True, is_causal=False):
        # Mirrors the slow path of nn.MultiheadAttention.forward, with one
        # change: the in_proj weight comes from the autograd-tracked SVD
        # property, NOT from the frozen leaf parameter.
        if key is None:
            key = query
        if value is None:
            value = query

        is_batched = query.dim() == 3
        if self.batch_first and is_batched:
            query, key, value = (x.transpose(1, 0) for x in (query, key, value))

        in_proj_weight = self._svd_in_proj.weight   # autograd-tracked property
        in_proj_bias = self.in_proj_bias            # frozen leaf, kept as-is

        # out_proj is itself an SVDResidualLinear when out_proj-SVD is enabled;
        # its .weight is also an autograd-tracked property.
        out_proj_weight = self.out_proj.weight
        out_proj_bias = self.out_proj.bias

        attn_output, attn_output_weights = F.multi_head_attention_forward(
            query, key, value,
            self.embed_dim, self.num_heads,
            in_proj_weight, in_proj_bias,
            self.bias_k, self.bias_v, self.add_zero_attn,
            self.dropout, out_proj_weight, out_proj_bias,
            training=self.training,
            key_padding_mask=key_padding_mask,
            need_weights=need_weights,
            attn_mask=attn_mask,
            use_separate_proj_weight=False,
            average_attn_weights=average_attn_weights,
            is_causal=is_causal,
        )

        if self.batch_first and is_batched:
            attn_output = attn_output.transpose(1, 0)

        return attn_output, attn_output_weights

    mha_module.forward = types.MethodType(forward, mha_module)
    # Mark the patch so callers (e.g. checkpoint loaders, audits) can detect it.
    mha_module._svd_in_proj_routing_active = True


def apply_svd_residual_to_openclip_attn(model, r, apply_to_in_proj=True, block_indices=None, _current_block_idx=None, apply_to_mlp=False):
    """
    Apply SVD decomposition to OpenCLIP vision transformer attention layers.
    
    OpenCLIP structure differs from HuggingFace:
    - HuggingFace: encoder.layers[i].self_attn.{q_proj, k_proj, v_proj, out_proj}
    - OpenCLIP: transformer.resblocks[i].attn (nn.MultiheadAttention)
    
    With apply_to_in_proj=True (NEW default), we apply SVD to both:
    1. out_proj: Output projection [embed_dim, embed_dim]
    2. in_proj_weight: Fused Q, K, V projections [3*embed_dim, embed_dim]
    
    This provides 4x more trainable parameters (matching HuggingFace coverage).
    
    Args:
        model: The OpenCLIP model
        r: Rank for SVD decomposition
        apply_to_in_proj: If True, also apply SVD to in_proj_weight (q, k, v).
                         Default True for full coverage matching HuggingFace.
        block_indices: Optional list of block indices (0-indexed) to apply SVD to.
                      If None, apply to all blocks. E.g., [9, 10, 11] for last 3 blocks.
        _current_block_idx: Internal tracker for block index during recursion.
        apply_to_mlp: If True, also apply SVD to MLP fc layers (c_fc and c_proj).
                     Default False. NEW: enables adapting MLP pathway.
    """
    for name, module in model.named_children():
        # Track block index when entering resblocks
        if name == 'resblocks':
            # This is the Sequential container of blocks
            for block_idx, (block_name, block_module) in enumerate(module.named_children()):
                apply_svd_residual_to_openclip_attn(
                    block_module, r, apply_to_in_proj=apply_to_in_proj,
                    block_indices=block_indices, _current_block_idx=block_idx,
                    apply_to_mlp=apply_to_mlp
                )
            continue
        
        # === NEW: Apply SVD to MLP layers (c_fc and c_proj) ===
        if apply_to_mlp and name == 'mlp':
            # Check if we should skip this block
            if block_indices is not None and _current_block_idx is not None:
                if _current_block_idx not in block_indices:
                    logger.info(f"Skipping SVD for MLP in block {_current_block_idx} (not in block_indices)")
                    for param in module.parameters():
                        param.requires_grad = False
                    continue
                else:
                    logger.info(f"Applying SVD to MLP in block {_current_block_idx}")
            
            # OpenCLIP MLP structure: mlp.c_fc (768->3072) and mlp.c_proj (3072->768)
            for mlp_name, mlp_module in module.named_children():
                if isinstance(mlp_module, nn.Linear) and mlp_name in ('c_fc', 'c_proj'):
                    logger.info(f"Applying SVD residual to MLP.{mlp_name}: {mlp_module.weight.shape}")
                    new_linear = replace_with_svd_residual(mlp_module, r)
                    setattr(module, mlp_name, new_linear)
            continue
        
        if isinstance(module, nn.MultiheadAttention):
            # Check if we should skip this block
            if block_indices is not None and _current_block_idx is not None:
                if _current_block_idx not in block_indices:
                    logger.info(f"Skipping SVD for block {_current_block_idx} (not in block_indices={block_indices})")
                    # Freeze this attention layer's parameters
                    for param in module.parameters():
                        param.requires_grad = False
                    continue
                else:
                    logger.info(f"Applying SVD to block {_current_block_idx} (in block_indices={block_indices})")
            try:
                # 1. Apply SVD to out_proj (existing behavior)
                logger.info(f"Applying SVD residual to MultiheadAttention.out_proj for module: {name}")
                new_out = replace_with_svd_residual(module.out_proj, r)
                if hasattr(module.out_proj, 'bias') and module.out_proj.bias is not None:
                    if hasattr(new_out, 'bias') and new_out.bias is not None:
                        new_out.bias.data.copy_(module.out_proj.bias.data)
                module.out_proj = new_out
                
                # 2. Apply SVD to in_proj_weight (q, k, v projections)
                if apply_to_in_proj and module.in_proj_weight is not None:
                    logger.info(f"Applying SVD residual to MultiheadAttention.in_proj (q, k, v) for module: {name}")
                    embed_dim = module.embed_dim

                    # Create SVDInProjLinear which handles the fused q, k, v
                    svd_in_proj = SVDInProjLinear(
                        module.in_proj_weight.data,
                        module.in_proj_bias.data if module.in_proj_bias is not None else None,
                        embed_dim,
                        r
                    )

                    module._svd_in_proj = svd_in_proj

                    # Route the MHA forward through the autograd-tracked SVD weight.
                    # See _install_svd_in_proj_routing for the bug context (2026-04-26 fix).
                    _install_svd_in_proj_routing(module)

                    # Freeze the original (now unused) in_proj_weight / in_proj_bias.
                    # The forward override never reads these; they remain on the module
                    # only for checkpoint compatibility.
                    module.in_proj_weight.requires_grad = False
                    if module.in_proj_bias is not None:
                        module.in_proj_bias.requires_grad = False
                    # Also freeze the SVDInProjLinear's bias copy: the original code
                    # intent was to freeze the bias, but SVDInProjLinear creates it
                    # with requires_grad=True by default. Match the freeze intent.
                    if svd_in_proj.bias is not None:
                        svd_in_proj.bias.requires_grad = False
                    
            except Exception as e:
                logger.exception(f"Failed to replace projections for module {name}: {e}")
        else:
            # Recursively apply to child modules
            apply_svd_residual_to_openclip_attn(
                module, r, apply_to_in_proj=apply_to_in_proj,
                block_indices=block_indices, _current_block_idx=_current_block_idx,
                apply_to_mlp=apply_to_mlp
            )
    
    # After replacing, set requires_grad for residual components. Keep only
    # residual parameters trainable and freeze other backbone parameters by
    # default (this mirrors the behavior used elsewhere in the codebase).
    for param_name, param in model.named_parameters():
        if any(x in param_name for x in ['S_residual', 'U_residual', 'V_residual']):
            param.requires_grad = True
        else:
            # Do not override explicit requires_grad set by user if already True
            # but default to False to avoid accidental full fine-tuning.
            try:
                # Only set False if param is currently requires_grad True and
                # the name doesn't belong to a head/residual
                param.requires_grad = False
            except Exception:
                pass
    
    return model


# Function to replace a module with SVDResidualLinear
def replace_with_svd_residual(module, r):
    if isinstance(module, nn.Linear):
        in_features = module.in_features
        out_features = module.out_features
        bias = module.bias is not None

        # Create SVDResidualLinear module
        new_module = SVDResidualLinear(in_features, out_features, r, bias=bias, init_weight=module.weight.data.clone())

        if bias and module.bias is not None:
            new_module.bias.data.copy_(module.bias.data)

        new_module.weight_original_fnorm = torch.norm(module.weight.data, p='fro')

        # Perform SVD on the original weight
        U, S, Vh = torch.linalg.svd(module.weight.data, full_matrices=False)

        # Determine r based on the rank of the weight matrix
        r = min(r, len(S))  # Ensure r does not exceed the number of singular values

        # Keep top r singular components (main weight)
        U_r = U[:, :r]  # Shape: (out_features, r)
        S_r = S[:r]  # Shape: (r,)
        Vh_r = Vh[:r, :]  # Shape: (r, in_features)

        # Reconstruct the main weight (fixed)
        weight_main = U_r @ torch.diag(S_r) @ Vh_r

        # Calculate the frobenius norm of main weight
        new_module.weight_main_fnorm = torch.norm(weight_main.data, p='fro')

        # Set the main weight
        new_module.weight_main.data.copy_(weight_main)

        # Residual components (trainable)
        U_residual = U[:, r:]  # Shape: (out_features, n - r)
        S_residual = S[r:]  # Shape: (n - r,)
        Vh_residual = Vh[r:, :]  # Shape: (n - r, in_features)

        if len(S_residual) > 0:
            new_module.S_residual = nn.Parameter(S_residual.clone())
            new_module.U_residual = nn.Parameter(U_residual.clone())
            new_module.V_residual = nn.Parameter(Vh_residual.clone())

            new_module.S_r = nn.Parameter(S_r.clone(), requires_grad=False)
            new_module.U_r = nn.Parameter(U_r.clone(), requires_grad=False)
            new_module.V_r = nn.Parameter(Vh_r.clone(), requires_grad=False)
        else:
            new_module.S_residual = None
            new_module.U_residual = None
            new_module.V_residual = None

            new_module.S_r = None
            new_module.U_r = None
            new_module.V_r = None

        return new_module
    else:
        return module
