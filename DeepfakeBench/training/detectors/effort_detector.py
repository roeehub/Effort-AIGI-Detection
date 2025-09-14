import os
import math
import datetime
import logging
import numpy as np  # noqa
from sklearn import metrics  # noqa
from typing import Union
from collections import defaultdict

import torch  # noqa
import torch.nn as nn  # noqa
import torch.nn.functional as F  # noqa
import torch.optim as optim  # noqa
from torch.nn import DataParallel  # noqa

from metrics.base_metrics_class import calculate_metrics_for_train  # noqa

from .base_detector import AbstractDetector
from detectors import DETECTOR  # noqa
from networks import BACKBONE  # noqa
from loss import LOSSFUNC  # noqa

import loralib as lora  # noqa
from transformers import AutoProcessor, CLIPModel, ViTModel, ViTConfig  # noqa

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
        self.s = s
        self.m = m
        # The weight parameter is equivalent to the class centers in the feature space
        self.weight = nn.Parameter(torch.FloatTensor(out_features, in_features))
        nn.init.xavier_uniform_(self.weight)

    def forward(self, features, label=None):
        # 1. Normalize feature vectors and weight vectors
        cosine = F.linear(F.normalize(features), F.normalize(self.weight))

        # If no label is provided (i.e., during inference), we can't add a margin.
        # We return the scaled cosine similarity, which is a valid logit for ranking.
        if label is None:
            return cosine * self.s

        # 2. Calculate the angle (theta) from the cosine similarity
        # Add a small epsilon for numerical stability
        theta = torch.acos(torch.clamp(cosine, -1.0 + 1e-7, 1.0 - 1e-7))

        # 3. Add the angular margin 'm' to the angle of the correct class
        # Create a one-hot vector from the labels
        one_hot = torch.zeros(cosine.size(), device=features.device)
        one_hot.scatter_(1, label.view(-1, 1).long(), 1)

        # Add margin 'm' where the one_hot vector is 1
        # M_theta = theta + m * one_hot
        M_theta = torch.where(one_hot.bool(), theta + self.m, theta)

        # 4. Convert the new angle back to a cosine value
        marginal_target_logit = torch.cos(M_theta)

        # 5. Scale the logits
        # This sharpens the distribution and helps convergence
        logits = self.s * marginal_target_logit

        return logits

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
    """

    def __init__(self, reduction='mean'):
        super(CrossEntropyLossWithReduction, self).__init__()
        self.default_reduction = reduction

    def forward(self, inputs, targets, reduction=None):
        """
        Args:
            inputs: model predictions (logits) of shape [N, C]
            targets: ground truth labels of shape [N]
            reduction (str, optional): Overrides the default reduction method.
                                       Can be 'mean', 'sum', or 'none'.
        """
        reduction_to_use = reduction if reduction is not None else self.default_reduction
        return F.cross_entropy(inputs, targets, reduction=reduction_to_use)


@DETECTOR.register_module(module_name='effort')
class EffortDetector(nn.Module):
    def __init__(self, config=None):
        super(EffortDetector, self).__init__()
        self.config = config
        self.lambda_reg = config.get('lambda_reg', 1.0)  # Default to 1.0 if not in config
        self.rank = config.get('rank', 1023)
        self.clip_backbone_path = config['gcs_assets']['clip_backbone']['local_path']
        self.backbone = self.build_backbone(config)  # Initialize Backbone model

        # Controlled initialization of the head based on config
        self.use_arcface_head = config.get('use_arcface_head', False)
        if self.use_arcface_head:
            s = config.get('arcface_s', 30.0)
            m = config.get('arcface_m', 0.35)
            logger.info(f"Using ArcFace head with s={s} and m={m}")
            self.head = ArcMarginProduct(in_features=1024, out_features=2, s=s, m=m)
        else:
            logger.info("Using standard Linear head")
            self.head = nn.Linear(1024, 2)

        # Controlled initialization of the loss function
        # If ArcFace is used, we MUST use CrossEntropyLoss, not FocalLoss.
        # The margin 'm' in ArcFace serves a similar purpose to Focal Loss's gamma.
        if self.use_arcface_head:
            logger.info("ArcFace head is active. Switching to standard CrossEntropyLoss.")
            self.loss_func = CrossEntropyLossWithReduction()
        else:
            self.use_focal_loss = config.get('use_focal_loss', False)
            if self.use_focal_loss:
                gamma = config.get('focal_loss_gamma', 2.0)
                alpha = config.get('focal_loss_alpha', None)
                logger.info(f"Using Focal Loss with gamma={gamma} and alpha={alpha}")
                self.loss_func = FocalLoss(gamma=gamma, alpha=alpha)
            else:
                logger.info("Using standard CrossEntropyLoss")
                self.loss_func = CrossEntropyLossWithReduction()

        self.prob, self.label = [], []
        self.correct, self.total = 0, 0

    def build_backbone(self, config):
        # ⚠⚠⚠ Download CLIP model using the below link
        # https://drive.google.com/drive/folders/1fm3Jd8lFMiSP1qgdmsxfqlJZGpr_bXsx?usp=drive_link

        # mean: [0.48145466, 0.4578275, 0.40821073]
        # std: [0.26862954, 0.26130258, 0.27577711]

        # ViT-L/14 224*224
        print("The CLIP is in: ", os.getcwd())
        clip_model = CLIPModel.from_pretrained(
            self.clip_backbone_path,  # Path to the downloaded CLIP model
            local_files_only=True)  # the path of this folder in your disk (download from the above link)
        # clip_model = CLIPModel.from_pretrained("openai/clip-vit-large-patch14", cache_dir="./clip-vit-large-patch14")

        # Apply SVD to self_attn layers only
        # ViT-L/14 224*224: 1024-1
        clip_model.vision_model = apply_svd_residual_to_self_attn(clip_model.vision_model, r=self.rank)

        # for name, param in clip_model.vision_model.named_parameters():
        #    print('{}: {}'.format(name, param.requires_grad))
        # num_param = sum(p.numel() for p in clip_model.vision_model.parameters() if p.requires_grad)
        # num_total_param = sum(p.numel() for p in clip_model.vision_model.parameters())
        # print('Number of total parameters: {}, tunable parameters: {}'.format(num_total_param, num_param))

        return clip_model.vision_model

    def features(self, data_dict: dict) -> torch.tensor:
        feat = self.backbone(data_dict['image'])['pooler_output']
        return feat

    def classifier(self, features: torch.tensor) -> torch.tensor:
        return self.head(features)

    # def get_losses(self, data_dict: dict, pred_dict: dict) -> dict:
    #    label = data_dict['label']
    #    pred = pred_dict['cls']
    #    loss = self.loss_func(pred, label)
    #
    #    if self.training:
    #        # Regularization term
    #        lambda_reg = 1.0
    #        reg_term = 0.0
    #        num_reg = 0
    #        for module in self.backbone.modules():
    #            if isinstance(module, SVDResidualLinear):
    #                reg_term += module.compute_orthogonal_loss()
    #                reg_term += module.compute_keepsv_loss()
    #                num_reg += 1
    #
    #        loss += lambda_reg * reg_term / num_reg
    #
    #    loss_dict = {'overall': loss}
    #    return loss_dict

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

    # def get_losses(self, data_dict: dict, pred_dict: dict) -> dict:
    #     label = data_dict['label']
    #     pred = pred_dict['cls']
    #
    #     # Check if the batch was originally 5D by comparing label and pred batch sizes
    #     # If len(pred) > len(label), it means we reshaped a video batch
    #     if pred.shape[0] > label.shape[0]:
    #         B = label.shape[0]
    #         # Calculate T (number of frames) from the discrepancy
    #         T = pred.shape[0] // B
    #         # Repeat each label T times to match the reshaped predictions
    #         label = label.repeat_interleave(T)
    #
    #     # The rest of the loss calculation logic remains the same
    #     # It now works correctly for both 4D and 5D original inputs
    #
    #     # Compute overall loss using all samples
    #     loss = self.loss_func(pred, label)
    #
    #     # Create masks for real and fake classes
    #     mask_real = label == 0
    #     mask_fake = label == 1
    #
    #     # Compute loss for real class
    #     if mask_real.sum() > 0:
    #         loss_real = self.loss_func(pred[mask_real], label[mask_real])
    #     else:
    #         loss_real = torch.tensor(0.0, device=pred.device)
    #
    #     # Compute loss for fake class
    #     if mask_fake.sum() > 0:
    #         loss_fake = self.loss_func(pred[mask_fake], label[mask_fake])
    #     else:
    #         loss_fake = torch.tensor(0.0, device=pred.device)
    #
    #     loss_dict = {
    #         'overall': loss,
    #         'real_loss': loss_real,
    #         'fake_loss': loss_fake,
    #     }
    #     return loss_dict

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
        """
        label = data_dict['label']
        pred = pred_dict['cls']

        # Handle reshaped video batches by repeating labels for each frame
        if pred.shape[0] > label.shape[0]:
            B = label.shape[0]
            T = pred.shape[0] // B
            label = label.repeat_interleave(T)

        # --- Calculate Regularization Term (as a scalar) ---
        reg_term = torch.tensor(0.0, device=pred.device)
        if self.training:
            lambda_reg = self.lambda_reg
            num_reg = 0
            current_reg_term = 0.0
            for module in self.backbone.modules():
                if isinstance(module, SVDResidualLinear):
                    current_reg_term += module.compute_orthogonal_loss()
                    current_reg_term += module.compute_keepsv_loss()
                    num_reg += 1
            if num_reg > 0:
                reg_term = lambda_reg * current_reg_term / num_reg

        # --- Main Loss Calculation based on reduction type ---
        if reduction == 'mean':
            # --- DEFAULT BEHAVIOR: Return a single scalar loss ---
            loss = self.loss_func(pred, label)  # Assumes self.loss_func defaults to 'mean'

            # Add scalar regularization term
            if self.training:
                loss += reg_term

            # For logging, calculate separate real/fake losses
            mask_real = label == 0
            mask_fake = label == 1
            loss_real = self.loss_func(pred[mask_real], label[mask_real]) if mask_real.sum() > 0 else torch.tensor(0.0,
                                                                                                                   device=pred.device)
            loss_fake = self.loss_func(pred[mask_fake], label[mask_fake]) if mask_fake.sum() > 0 else torch.tensor(0.0,
                                                                                                                   device=pred.device)

            loss_dict = {
                'overall': loss,
                'real_loss': loss_real,
                'fake_loss': loss_fake,
            }

        elif reduction == 'none':
            # --- NEW BEHAVIOR: Return per-sample losses for Group-DRO ---
            # CRITICAL: Assumes self.loss_func (e.g., nn.CrossEntropyLoss, FocalLoss)
            # supports the `reduction` argument. This is standard in PyTorch.
            per_sample_loss = self.loss_func(pred, label, reduction='none')

            # Add scalar regularization term (PyTorch broadcasts this correctly)
            if self.training:
                per_sample_loss += reg_term

            # For logging, calculate the mean of the per-sample losses for each class
            mask_real = label == 0
            mask_fake = label == 1
            loss_real = per_sample_loss[mask_real].mean() if mask_real.sum() > 0 else torch.tensor(0.0,
                                                                                                   device=pred.device)
            loss_fake = per_sample_loss[mask_fake].mean() if mask_fake.sum() > 0 else torch.tensor(0.0,
                                                                                                   device=pred.device)

            loss_dict = {
                'overall': per_sample_loss,  # This is a TENSOR
                'real_loss': loss_real.detach(),  # Scalar for logging
                'fake_loss': loss_fake.detach(),  # Scalar for logging
            }
        else:
            raise ValueError(f"Unsupported reduction type: '{reduction}'. Must be 'mean' or 'none'.")

        return loss_dict

    def get_train_metrics(self, data_dict: dict, pred_dict: dict) -> dict:
        label = data_dict['label']
        pred = pred_dict['cls']

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
        metric_batch_dict = {'acc': acc, 'auc': auc, 'eer': eer, 'ap': ap}
        return metric_batch_dict

    def forward(self, data_dict: dict, inference=False) -> dict:
        image = data_dict['image']
        label = data_dict.get('label', None)  # Get label, if available

        # Check if the input is a 5D tensor (batch of videos)
        if image.dim() == 5:
            # data_dict['image'] has shape [B, T, C, H, W]
            B, T, C, H, W = image.shape
            # Reshape to treat every frame as a separate sample: [B * T, C, H, W]
            image = image.view(B * T, C, H, W)
            # --- Ensure labels are also expanded if present ---
            if label is not None:
                label = label.repeat_interleave(T)

        # Create a temporary dict for the backbone
        temp_data_dict = {'image': image}

        # Now the backbone receives a 4D tensor as expected
        features = self.features(temp_data_dict)

        # The logic for calling the head must now be inside the forward pass,
        # because the ArcFace head requires the label during training.
        if self.use_arcface_head:
            # Pass the label to the head during training.
            # During inference, label might be None, and the head is designed to handle this.
            pred = self.head(features, label)
        else:
            # The standard linear head does not need the label.
            pred = self.classifier(features)

        prob = torch.softmax(pred, dim=1)[:, 1]
        # build the prediction dict for each output
        pred_dict = {'cls': pred, 'prob': prob, 'feat': features}

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
