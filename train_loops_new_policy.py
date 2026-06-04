from utils import cal_param_size, cal_multi_adds, AverageMeter, adjust_lr, DistillKL, correct_num
from metrics import dice_score, iou_score, get_evaluation_metrics
import random
import time
import math
import torch
import torch.nn as nn
import torch.optim as optim
import torch.backends.cudnn as cudnn
import torch.nn.functional as F

import os
import shutil
import argparse
import numpy as np
from distiller_zoo import FeatureKLLoss, FeatureMSELoss, FeatureKLLoss3D, FeatureMSELoss3D
from meta_teacher_optimizer import get_meta_teacher_optimizer
import torch.nn.functional as F
from tqdm import tqdm


def compute_tp_fp_fn(predictions, targets, axes=None):
    """
    Compute True Positives, False Positives, False Negatives for segmentation.
    This follows nnUNet's implementation.
    
    Args:
        predictions: Binary predictions [B, C, ...] or [B, ...] (already thresholded)
        targets: Ground truth [B, C, ...] or [B, ...]
        axes: Axes to sum over (default: all except channel dim)
    
    Returns:
        tp, fp, fn: Arrays of shape [C] with per-channel/region counts
    """
    if axes is None:
        # Sum over all dims except the channel dimension (dim 1)
        axes = tuple(range(predictions.ndim))
        axes = tuple([i for i in axes if i != 1])
    
    # Ensure both are boolean or float
    predictions = predictions.float()
    targets = targets.float()
    
    tp = ((predictions == 1) & (targets == 1)).sum(dim=axes).cpu().numpy()
    fp = ((predictions == 1) & (targets == 0)).sum(dim=axes).cpu().numpy()
    fn = ((predictions == 0) & (targets == 1)).sum(dim=axes).cpu().numpy()
    
    return tp, fp, fn


def compute_dice_per_sample(predictions, targets, channels_to_include=None, keep_batch_dim=True):
    """
    Compute Dice score per sample for segmentation tasks.
    This uses the same formula as test(): Dice = 2*TP / (2*TP + FP + FN).
    
    Args:
        predictions: Binary predictions [B, C, D, H, W] (already thresholded to 0/1)
        targets: Ground truth [B, C, D, H, W] (0/1)
        channels_to_include: List of channel indices to include (e.g., [1, 2] for PCa/csPCa).
                            If None, includes all channels.
        keep_batch_dim: If True, returns [B] with per-sample Dice averaged across channels.
                        If False, returns [B, C] with per-sample, per-channel Dice.
    
    Returns:
        dice_scores: Tensor of shape [B] or [B, C] depending on keep_batch_dim
    """
    predictions = predictions.float()
    targets = targets.float()
    
    # If channels specified, select them
    if channels_to_include is not None:
        predictions = predictions[:, channels_to_include]  # [B, len(channels), D, H, W]
        targets = targets[:, channels_to_include]
    
    # Compute per-sample, per-channel Dice
    # Dice = 2*intersection / (pred_sum + target_sum) = 2*TP / (2*TP + FP + FN)
    batch_size, num_channels = predictions.shape[:2]
    dice_per_sample_per_channel = []
    
    for c in range(num_channels):
        pred_c = predictions[:, c]  # [B, D, H, W]
        target_c = targets[:, c]  # [B, D, H, W]
        
        # Compute intersection and union per sample
        intersection = (pred_c * target_c).sum(dim=(1, 2, 3))  # [B]
        pred_sum = pred_c.sum(dim=(1, 2, 3))  # [B]
        target_sum = target_c.sum(dim=(1, 2, 3))  # [B]
        
        # Dice = 2*TP / (2*TP + FP + FN) = 2*intersection / (pred_sum + target_sum)
        dice_c = (2.0 * intersection + 1e-7) / (pred_sum + target_sum + 1e-7)  # [B]
        dice_per_sample_per_channel.append(dice_c)
    
    # Stack to [B, C]
    dice_scores = torch.stack(dice_per_sample_per_channel, dim=1)  # [B, C]
    
    if keep_batch_dim:
        # Average across channels: [B]
        return dice_scores.mean(dim=1)
    else:
        # Keep per-channel: [B, C]
        return dice_scores


# =====================================================================
# Cancer-correctness masking for feature distillation
# =====================================================================

def compute_cancer_correctness_mask(teacher_logits, targets, teacher_name, dilation_radius=3):
    """
    Compute a binary spatial mask indicating where a teacher is correct about cancer.
    
    Mask = 1 (distill) for:
      - True Negatives: teacher correctly says "no cancer" (background/prostate voxels)
      - True Positives: teacher correctly says "cancer" + dilated neighborhood
    Mask = 0 (skip) for:
      - False Positives: teacher wrongly says "cancer"
      - False Negatives: teacher misses cancer (outside TP dilation zone)
    
    Args:
        teacher_logits: [B, C_teacher, D, H, W] raw logits from one teacher
        targets: [B, C_target, D, H, W] region-based GT (C=2 or C=3)
                 C=2: [prostate, cancer]
                 C=3: [prostate, PCa, csPCa]
        teacher_name: str, one of 'nnunet', 'provicnet', 'prostatlasdiff'
        dilation_radius: int, radius of 3D dilation around TP voxels (default 3)
    
    Returns:
        mask: [B, 1, D, H, W] float tensor, 1=distill, 0=skip
              Broadcastable over channel dim for feature masking
    """
    with torch.no_grad():
        B = teacher_logits.shape[0]
        target_channels = targets.shape[1]
        
        # --- Step 1: Get binary cancer GT [B, D, H, W] ---
        if target_channels == 2:
            gt_cancer = targets[:, 1].float()  # [B, D, H, W]
        else:
            # C=3: cancer = PCa (ch1) | csPCa (ch2)
            gt_cancer = torch.clamp(targets[:, 1] + targets[:, 2], 0, 1).float()
        
        # --- Step 2: Get binary cancer prediction per teacher ---
        name = teacher_name.lower()
        if 'prostatlasdiff' in name:
            # ProstAtlasDiff: logits [B, 3, D, H, W] — 3 modality copies of 1 cancer channel
            # Use channel 1 (same as prostatlasdiff_get_pred_dice)
            probs = torch.sigmoid(teacher_logits)
            # Normalize + scale + clip (same pipeline as prostatlasdiff_get_pred_dice)
            max_val = probs.max()
            if max_val > 0:
                probs = probs / max_val
            probs = torch.clamp(probs / 0.2, 0, 1)
            pred_cancer = (probs[:, 1] > 0.1).float()  # [B, D, H, W]
        elif 'provicnet' in name:
            # ProViCNet: logits [B, 4, D, H, W] — softmax classes
            # Cancer = ch2 (ciPCa) or ch3 (csPCa)
            probs = F.softmax(teacher_logits, dim=1)
            cancer_prob = probs[:, 2] + probs[:, 3]  # [B, D, H, W]
            pred_cancer = (cancer_prob > 0.5).float()
        else:
            # nnUNet: sigmoid, region-based
            # 3-class logits [B, 3, D, H, W]: Cancer = ch1 (PCa) or ch2 (csPCa)
            # Binary  logits [B, 2, D, H, W]: Cancer = ch1 (combined by dataloader)
            probs = torch.sigmoid(teacher_logits)
            if probs.shape[1] == 2:
                pred_cancer = (probs[:, 1] > 0.5).float()
            else:
                pred_cancer = ((probs[:, 1] > 0.5) | (probs[:, 2] > 0.5)).float()
        
        # --- Step 3: Compute TP, FP, FN per voxel ---
        tp = pred_cancer * gt_cancer           # both say cancer
        fp = pred_cancer * (1 - gt_cancer)     # teacher says cancer, GT says no
        fn = (1 - pred_cancer) * gt_cancer     # teacher misses cancer
        # tn = (1 - pred_cancer) * (1 - gt_cancer)  # implied: mask=1
        
        # --- Step 4: Dilate TP in 3D ---
        if dilation_radius > 0 and tp.sum() > 0:
            kernel_size = 2 * dilation_radius + 1  # radius=3 → kernel=7
            # Use max_pool3d for dilation (equivalent to binary dilation)
            # tp: [B, D, H, W] → [B, 1, D, H, W] for pooling
            tp_dilated = F.max_pool3d(
                tp.unsqueeze(1),
                kernel_size=kernel_size,
                stride=1,
                padding=dilation_radius
            ).squeeze(1)  # [B, D, H, W]
        else:
            tp_dilated = tp
        
        # --- Step 5: Build mask ---
        # Mask = 1 everywhere EXCEPT: FP voxels and FN voxels not covered by TP dilation
        # Equivalently: mask = 1 - FP - (FN that's outside TP dilation)
        # Or more clearly: mask = TN + dilated_TP region
        # mask = 1 where: (no FP) AND (not FN outside dilation)
        
        # Start with all 1s, then zero out FP and un-rescued FN
        mask = torch.ones_like(gt_cancer)
        mask[fp > 0] = 0              # zero out false positives
        fn_outside_dilation = fn * (1 - tp_dilated)  # FN voxels not rescued by TP dilation
        mask[fn_outside_dilation > 0] = 0
        
        # [B, D, H, W] → [B, 1, D, H, W] for broadcasting over channels
        return mask.unsqueeze(1)


def compute_teacher_disagreement(teacher_logits, targets=None, is_region_based=True, 
                                 channels_to_include=None, method='variance'):
    """
    Compute per-sample teacher disagreement as a weighting factor for rewards.
    
    High disagreement → more learning opportunity (agent should explore)
    Low disagreement → teachers agree (simple averaging suffices)
    
    Args:
        teacher_logits: list of teacher logit tensors, each [B, C, D, H, W]
        targets: ground truth [B, C, D, H, W] (optional, not used for variance method)
        is_region_based: bool, whether using region-based training (sigmoid) vs class-based (softmax)
        channels_to_include: list of channels to compute disagreement on (e.g., [1, 2] for PCa/csPCa)
        method: str, 'variance' (default) or 'pairwise_dice'
            - 'variance': std of teacher probabilities (fast, works well)
            - 'pairwise_dice': avg pairwise Dice disagreement (more expensive but precise)
    
    Returns:
        disagreement_weight: tensor [B], higher values = more disagreement
                            Normalized to [0.5, 1.5] range to avoid extreme scaling
    """
    num_teachers = len(teacher_logits)
    if num_teachers < 2:
        # No disagreement with single teacher
        batch_size = teacher_logits[0].size(0)
        return torch.ones(batch_size, device=teacher_logits[0].device)
    
    if method == 'variance':
        # Method 1: Variance of teacher probabilities (FAST)
        # Higher variance = more disagreement
        
        # Convert logits to probabilities
        if is_region_based:
            teacher_probs = [torch.sigmoid(t) for t in teacher_logits]
        else:
            teacher_probs = [F.softmax(t, dim=1) for t in teacher_logits]
        
        # Stack: [num_teachers, B, C, D, H, W]
        probs_stack = torch.stack(teacher_probs, dim=0)
        
        # Select channels if specified
        if channels_to_include is not None:
            probs_stack = probs_stack[:, :, channels_to_include]
        
        # Compute variance across teachers (dim=0), then average over spatial dims and channels
        # Variance: [B, C, D, H, W]
        variance = probs_stack.var(dim=0)
        
        # Average over all dimensions except batch: [B, C, D, H, W] -> [B]
        disagreement = variance.view(variance.size(0), -1).mean(dim=1)
        
    elif method == 'pairwise_dice':
        # Method 2: Pairwise Dice disagreement (MORE EXPENSIVE but more interpretable)
        # Compute Dice between all teacher pairs, then average
        
        # Convert to predictions
        if is_region_based:
            teacher_preds = [(torch.sigmoid(t) > 0.5).float() for t in teacher_logits]
        else:
            teacher_preds = [torch.argmax(t, dim=1, keepdim=True) for t in teacher_logits]
            # Convert to one-hot for Dice computation
            num_classes = teacher_logits[0].size(1)
            teacher_preds = [
                F.one_hot(pred.squeeze(1).long(), num_classes).permute(0, 3, 1, 2, 3).float()
                for pred in teacher_preds
            ]
        
        # Compute pairwise Dice disagreement
        pairwise_disagreements = []
        for i in range(num_teachers):
            for j in range(i + 1, num_teachers):
                # Dice between teacher i and j
                dice_ij = compute_dice_per_sample(
                    teacher_preds[i], 
                    teacher_preds[j],  # Use teacher j as "target"
                    channels_to_include=channels_to_include,
                    keep_batch_dim=True
                )  # [B]
                # Disagreement = 1 - Dice (higher when teachers differ)
                pairwise_disagreements.append(1.0 - dice_ij)
        
        # Average disagreement across all pairs: [B]
        disagreement = torch.stack(pairwise_disagreements).mean(dim=0)
    
    else:
        raise ValueError(f"Unknown disagreement method: {method}")
    
    # Normalize disagreement to [0.5, 1.5] range to avoid extreme reward scaling
    # This ensures disagreement modulates reward by ±50% rather than completely dominating it
    disagreement_min = disagreement.min()
    disagreement_max = disagreement.max()
    
    if disagreement_max - disagreement_min < 1e-8:
        # No variation in disagreement across batch
        disagreement_weight = torch.ones_like(disagreement)
    else:
        # Normalize to [0, 1] then scale to [0.5, 1.5]
        disagreement_normalized = (disagreement - disagreement_min) / (disagreement_max - disagreement_min + 1e-8)
        disagreement_weight = 0.5 + disagreement_normalized  # [0.5, 1.5]
    
    return disagreement_weight


def compute_deep_supervision_loss(logits, targets, criterion_ce, use_deep_supervision=False):
    """
    Compute deep supervision loss with nnUNet-style weighting.
    
    Args:
        logits: Either a single tensor [B, C, D, H, W] or list of tensors at different scales
        targets: Ground truth labels [B, D, H, W]
        criterion_ce: Base classification criterion
        use_deep_supervision: Whether to apply deep supervision (only if logits is a list)
    
    Returns:
        loss_cls: Weighted classification loss
        logits_main: Main output (highest resolution) for KD and feature losses
    """
    # Check if deep supervision is enabled AND logits is a list
    if use_deep_supervision and isinstance(logits, (list, tuple)):
        # Deep supervision: compute weighted loss across scales
        num_outputs = len(logits)
        
        # nnUNet weighting scheme: exponentially decreasing, last weight = 0, then normalize
        weights = torch.tensor([0.5 ** i for i in range(num_outputs)], device=logits[0].device)
        weights[-1] = 0.0  # Don't use coarsest output
        weights = weights / weights.sum()  # Normalize
        
        # Compute loss at each scale
        total_loss = 0.0
        for i, output in enumerate(logits):
            if weights[i] == 0.0:
                continue
            
            # Downsample targets to match output resolution
            # Check if targets are region-based [B, C, D, H, W] or class-based [B, D, H, W]
            is_region_based = (targets.dim() == 5)
            
            if is_region_based:
                # Region-based targets: [B, C, D, H, W], compare with output spatial dims
                if output.shape[2:] != targets.shape[2:]:
                    # Resize keeping channel dimension
                    target_resized = F.interpolate(
                        targets,
                        size=output.shape[2:],
                        mode='nearest'
                    )  # Keep as float for BCE
                else:
                    target_resized = targets
            else:
                # Class-based targets: [B, D, H, W]
                if output.shape[2:] != targets.shape[1:]:
                    target_resized = F.interpolate(
                        targets.float().unsqueeze(1),
                        size=output.shape[2:],
                        mode='nearest'
                    ).squeeze(1).long()
                else:
                    target_resized = targets
            
            # Compute loss at this scale
            loss_at_scale = criterion_ce(output, target_resized)
            total_loss += weights[i] * loss_at_scale
        
        # Print logits statistics after sigmoid (only for main output, only once per run)
        if not hasattr(compute_deep_supervision_loss, '_printed_logits_stats'):
            logits_main = logits[0]  # Highest resolution output
            probs_main = torch.sigmoid(logits_main)
            print(f"[Logits Stats] Main output after sigmoid - Overall: mean={probs_main.mean().item():.4f}, std={probs_main.std().item():.4f}", flush=True)
            # Print per-channel statistics
            for c in range(probs_main.shape[1]):
                channel_data = probs_main[:, c]
                print(f"  Channel {c}: mean={channel_data.mean().item():.4f}, std={channel_data.std().item():.4f}, "
                      f"min={channel_data.min().item():.4f}, max={channel_data.max().item():.4f}", flush=True)
            compute_deep_supervision_loss._printed_logits_stats = True
        
        # Return weighted loss and main output (first in list, highest resolution)
        return total_loss, logits[0]
    else:
        # No deep supervision OR logits is not a list: return loss and logits as-is
        # Handle case where model returns list but deep supervision is disabled
        if isinstance(logits, (list, tuple)):
            # Use only the main output (first in list)
            return criterion_ce(logits[0], targets), logits[0]
        else:
            return criterion_ce(logits, targets), logits


class LossBalancer:
    """
    EMA-based adaptive loss normalization for multi-task learning.
    
    Tracks running means of each loss component using exponential moving average,
    then normalizes each loss by its historical magnitude to ensure balanced
    gradient contributions across all objectives.
    
    This prevents loss domination where one component (e.g., feature loss) 
    overwhelms others (e.g., classification loss) due to magnitude differences.
    
    Args:
        momentum: float, EMA momentum (default: 0.9, ~10-batch memory)
        eps: float, epsilon for numerical stability
        warmup_steps: int, number of steps before normalization starts (default: 10)
    
    Example:
        balancer = LossBalancer(momentum=0.9)
        
        for batch in dataloader:
            loss_cls, loss_kd, loss_feat = compute_losses(...)
            
            # Normalize losses
            loss_cls_norm, loss_kd_norm, loss_feat_norm = balancer.normalize_losses(
                loss_cls, loss_kd, loss_feat
            )
            
            # Combine with user weights
            loss = w_cls * loss_cls_norm + w_kd * loss_kd_norm + w_feat * loss_feat_norm
    """
    def __init__(self, momentum=0.9, eps=1e-8, warmup_steps=10):
        self.momentum = momentum
        self.eps = eps
        self.warmup_steps = warmup_steps
        
        # Running means (initialized to 1.0)
        self.running_mean_cls = 1.0
        self.running_mean_kd = 1.0
        self.running_mean_feat = 1.0
        
        # Step counter for warmup
        self.step_count = 0
    
    def update_running_means(self, loss_cls, loss_kd, loss_feat):
        """
        Update running means with current batch losses.
        
        Args:
            loss_cls: tensor or float, current classification loss
            loss_kd: tensor or float, current KD loss
            loss_feat: tensor or float, current feature loss
        """
        # Convert tensors to scalars
        if torch.is_tensor(loss_cls):
            loss_cls = loss_cls.item()
        if torch.is_tensor(loss_kd):
            loss_kd = loss_kd.item()
        if torch.is_tensor(loss_feat):
            loss_feat = loss_feat.item()
        
        # EMA update
        self.running_mean_cls = (
            self.momentum * self.running_mean_cls + 
            (1 - self.momentum) * loss_cls
        )
        self.running_mean_kd = (
            self.momentum * self.running_mean_kd + 
            (1 - self.momentum) * loss_kd
        )
        self.running_mean_feat = (
            self.momentum * self.running_mean_feat + 
            (1 - self.momentum) * loss_feat
        )
        
        self.step_count += 1
    
    def normalize_losses(self, loss_cls, loss_kd, loss_feat):
        """
        Normalize losses by their running means.
        
        During warmup period, just update running means without normalization.
        After warmup, divide each loss by its running mean.
        
        Args:
            loss_cls: tensor, classification loss
            loss_kd: tensor, knowledge distillation loss
            loss_feat: tensor, feature matching loss
        
        Returns:
            loss_cls_norm: tensor, normalized classification loss
            loss_kd_norm: tensor, normalized KD loss
            loss_feat_norm: tensor, normalized feature loss
        """
        # During warmup (before updating), return unnormalized losses
        if self.step_count < self.warmup_steps:
            # Update running means
            self.update_running_means(loss_cls, loss_kd, loss_feat)
            return loss_cls, loss_kd, loss_feat
        
        # After warmup: update running means
        self.update_running_means(loss_cls, loss_kd, loss_feat)
        
        # Then normalize by running means
        loss_cls_norm = loss_cls / (self.running_mean_cls + self.eps)
        loss_kd_norm = loss_kd / (self.running_mean_kd + self.eps)
        loss_feat_norm = loss_feat / (self.running_mean_feat + self.eps)
        
        return loss_cls_norm, loss_kd_norm, loss_feat_norm
    
    def get_stats(self):
        """Return current running means for logging."""
        return {
            'running_mean_cls': self.running_mean_cls,
            'running_mean_kd': self.running_mean_kd,
            'running_mean_feat': self.running_mean_feat,
            'step_count': self.step_count
        }
    
    def __repr__(self):
        return (f"LossBalancer(momentum={self.momentum}, "
                f"running_means=[cls={self.running_mean_cls:.3f}, "
                f"kd={self.running_mean_kd:.3f}, "
                f"feat={self.running_mean_feat:.3f}], "
                f"steps={self.step_count})")


class EpisodeBuffer:
    """
    Memory-efficient buffer for episode-based RL rewards.
    Stores states and actions to recompute log probs with gradients during training.
    Can optionally store logits actions and/or neg2 feature actions in addition to neg1 feature actions.
    """
    def __init__(self, store_logits_actions=False, store_neg2_actions=False):
        self.batch_rewards = []     # List of scalars (per-step blended rewards: dice + loss)
        self.states = []            # List of agent states (on CPU)
        self.feature_actions = []   # List of neg1 action tensors (on CPU), can be None entries
        self.store_logits_actions = store_logits_actions
        self.store_neg2_actions = store_neg2_actions
        if store_logits_actions:
            self.logits_actions = []  # List of logits action tensors (on CPU)
        if store_neg2_actions:
            self.neg2_feature_actions = []  # List of neg2 action tensors (on CPU)
        
    def add(self, batch_reward, agent_state, feature_action, neg2_feature_action=None, logits_action=None):
        """
        Add one batch experience to buffer.
        
        Args:
            batch_reward: scalar, blended per-step reward (dice-based + loss-based)
            agent_state: tuple of tensors representing state (will be moved to CPU)
                        Format: (neg1_features, neg2_features, teacher_logits, teacher_scalars)
                        neg1_features and neg2_features can be None if not enabled
            feature_action: tensor [B, num_teachers] for neg1, or None if neg1 not enabled
            neg2_feature_action: tensor [B, num_teachers] for neg2, or None if neg2 not enabled
            logits_action: optional tensor [B, num_teachers], action weights for logits (will be moved to CPU)
        """
        self.batch_rewards.append(batch_reward)
        # Store state on CPU to save GPU memory
        def to_cpu(x):
            if x is None:
                return None
            elif isinstance(x, (int, float)):
                return x  # Scalars (e.g. site_idx) don't need device transfer
            elif isinstance(x, list):
                return [t.detach().cpu() for t in x]
            else:
                return x.detach().cpu()
        state_cpu = tuple([to_cpu(item) for item in agent_state])
        self.states.append(state_cpu)
        # Store neg1 feature action (can be None)
        self.feature_actions.append(feature_action.detach().cpu() if feature_action is not None else None)
        # Store neg2 feature action if enabled
        if self.store_neg2_actions:
            self.neg2_feature_actions.append(neg2_feature_action.detach().cpu() if neg2_feature_action is not None else None)
        if self.store_logits_actions and logits_action is not None:
            self.logits_actions.append(logits_action.detach().cpu())
    
    def compute_returns(self, terminal_reward, gamma=0.95):
        """
        Compute discounted returns with terminal validation reward.
        
        G_t = r_t + gamma*r_{t+1} + ... + gamma^(T-t)*terminal_reward
        
        Args:
            terminal_reward: scalar, validation performance (e.g., Dice score)
            gamma: float, discount factor
            
        Returns:
            returns: list of scalars, one per batch
        """
        T = len(self.batch_rewards)
        returns = []
        
        # Backward pass to compute returns
        G = terminal_reward
        for t in reversed(range(T)):
            # Per-step reward already has correct sign (higher is better)
            G = self.batch_rewards[t] + gamma * G
            returns.insert(0, G)
        
        return returns
    
    def clear(self):
        """Clear buffer after agent update with explicit memory cleanup."""
        # Explicitly delete tensor references to help garbage collector
        for state in self.states:
            if state is not None:
                for item in state:
                    if item is not None:
                        if isinstance(item, list):
                            for t in item:
                                del t
                        else:
                            del item
        
        self.batch_rewards.clear()
        self.states.clear()
        self.feature_actions.clear()
        if self.store_logits_actions:
            self.logits_actions.clear()
        if self.store_neg2_actions:
            self.neg2_feature_actions.clear()
    
    def __len__(self):
        return len(self.batch_rewards)


def get_feature_kd_func(feat_kd_type, kd_T, use_3d=False, normalize=True, robust=False):
    """
    Get appropriate feature KD loss function based on dimensionality
    
    Args:
        feat_kd_type: str, 'mse' or 'kl'
        kd_T: float, temperature for KL divergence
        use_3d: bool, whether to use 3D-compatible versions
        normalize: bool, whether to normalize features before MSE (recommended: True)
        robust: bool, whether to use robust (Smooth-L1) loss instead of MSE
        
    Returns:
        feat_kd_func: The appropriate loss function
        
    Note:
        Setting normalize=True fixes the batch 230/400 loss spikes caused by
        feature magnitude outliers (60+ vs 20). Highly recommended for PIMED.
    """
    if feat_kd_type == 'mse':
        # Use normalized/robust MSE to handle feature outliers
        if normalize or robust:
            from distiller_zoo import get_feature_mse_loss
            return get_feature_mse_loss(use_3d=use_3d, normalize=normalize, robust=robust)
        else:
            # Original MSE (backward compatible, but prone to spikes)
            return FeatureMSELoss3D() if use_3d else FeatureMSELoss()
    elif feat_kd_type == 'kl':
        return FeatureKLLoss3D(kd_T) if use_3d else FeatureKLLoss(kd_T)
    else:
        raise ValueError(f"Unknown feature KD type: {feat_kd_type}")


def get_actions(agent_pred):
    batch_size = agent_pred.size(0)
    teacher_num = agent_pred.size(1)
    index = torch.from_numpy(np.random.randint(0, teacher_num, batch_size).astype(np.int64)).cuda(args.gpu)
    random_select =  F.one_hot(index, num_classes=teacher_num).float().cuda(args.gpu)
    actions = torch.where(agent_pred>=0.5, torch.ones_like(agent_pred), torch.zeros_like(agent_pred)).cuda(args.gpu)
    is_random = (actions.sum(1) ==  0)[:, None].float().cuda(args.gpu)
    actions = actions + random_select * is_random
    return actions


def train_agent(args, epoch, agent_state, agent_rewards, feature_agent_actions, agent, agent_optimizer):
    """
    Train agent with new policy format.
    State format: (neg1_features, neg2_features, teacher_logits, teacher_scalars)
    neg1_features and neg2_features can be None if not enabled.
    Note: Only trains on feature actions (no logit distillation anymore).
    """
    # Unwrap DDP if necessary to avoid DDP reduction errors
    if hasattr(agent, 'module'):
        agent_module = agent.module
    else:
        agent_module = agent
    
    agent_module.train()
    agent_loss = AverageMeter('agent_loss', ':.4e')

    for state, rewards, actions in zip(agent_state, agent_rewards, feature_agent_actions):
        # Move state from CPU back to CUDA
        # New policy state format: (neg1_features, neg2_features, teacher_logits, teacher_scalars)
        def to_cuda(x):
            if x is None:
                return None
            elif isinstance(x, list):
                return [t.cuda(args.gpu) for t in x]
            else:
                return x.cuda(args.gpu)
        state_cuda = tuple([to_cuda(item) for item in state])
        
        # Move rewards and actions back to CUDA
        rewards = rewards.cuda(args.gpu)
        actions = actions.cuda(args.gpu)
        
        # Check for NaN/Inf in input state before forward pass
        neg1_features, neg2_features, teacher_logits, teacher_scalars = state_cuda
        if neg1_features is not None and any(torch.isnan(t).any() or torch.isinf(t).any() for t in neg1_features):
            print(f"WARNING: NaN/Inf in neg1_features at epoch {epoch}, skipping this agent update", flush=True)
            continue
        if neg2_features is not None and any(torch.isnan(t).any() or torch.isinf(t).any() for t in neg2_features):
            print(f"WARNING: NaN/Inf in neg2_features at epoch {epoch}, skipping this agent update", flush=True)
            continue
        if any(torch.isnan(t).any() or torch.isinf(t).any() for t in teacher_logits):
            print(f"WARNING: NaN/Inf in teacher_logits at epoch {epoch}, skipping this agent update", flush=True)
            continue
        if any(torch.isnan(t).any() or torch.isinf(t).any() for t in teacher_scalars):
            print(f"WARNING: NaN/Inf in teacher_scalars at epoch {epoch}, skipping this agent update", flush=True)
            continue
        if torch.isnan(rewards).any() or torch.isinf(rewards).any():
            print(f"WARNING: NaN/Inf in rewards at epoch {epoch}, skipping this agent update", flush=True)
            continue
        
        # Use unwrapped agent_module to avoid DDP issues
        agent_pred = agent_module(state_cuda)
        
        # Check for invalid values in agent output
        if torch.isnan(agent_pred).any():
            print(f"WARNING: NaN detected in agent_pred at epoch {epoch}, skipping this agent update", flush=True)
            continue  # Skip this update instead of returning
        
        if torch.isinf(agent_pred).any():
            print(f"WARNING: Inf detected in agent_pred at epoch {epoch}, skipping this agent update", flush=True)
            continue  # Skip this update instead of returning
        
        agent_optimizer.zero_grad() 
        
        # Clamp values to [0, 1] range to prevent assertion errors
        agent_pred_clamped = torch.clamp(agent_pred, 0, 1)
        
        action_label = torch.ones_like(agent_pred_clamped).detach()
        # Agent outputs probabilities (from softmax), so use binary_cross_entropy (not _with_logits)
        loss = F.binary_cross_entropy(agent_pred_clamped, action_label, weight=rewards.unsqueeze(-1))
        
        # Check if loss is NaN before backward
        if torch.isnan(loss) or torch.isinf(loss):
            print(f"WARNING: NaN/Inf in agent loss at epoch {epoch}, skipping this agent update", flush=True)
            continue
        
        loss.backward()
        
        # Gradient clipping to prevent exploding gradients
        torch.nn.utils.clip_grad_norm_(agent.parameters(), max_norm=1.0)
        
        agent_optimizer.step()

        agent_loss.update(loss.item(), actions.size(0))

    if args.rank == 0:
        args.logger.info('Epoch:{}, agent Loss:{:.6f}'.format(epoch, agent_loss.avg))


def train_agent_episode(args, epoch, episode_buffer, val_metric, agent, agent_optimizer):
    """
    Memory-efficient agent update using episode returns with validation feedback.
    
    Args:
        args: training arguments
        epoch: current epoch
        episode_buffer: EpisodeBuffer with stored experiences
        val_metric: float, validation performance (e.g., Dice score) as terminal reward
        agent: policy network (may be DDP wrapped)
        agent_optimizer: optimizer for agent
    """
    if len(episode_buffer) == 0:
        return
    
    # Unwrap DDP if necessary to avoid DDP reduction errors
    # DDP expects all forward passes to contribute to loss, but agent training
    # is separate from main model training
    if hasattr(agent, 'module'):
        agent_module = agent.module
    else:
        agent_module = agent
    
    agent_module.train()
    agent_loss = AverageMeter('agent_loss_episode', ':.4e')
    
    # Compute returns with validation feedback
    # Scale validation metric to reasonable magnitude (0.7 Dice → 70)
    terminal_reward = val_metric * 100.0
    returns = episode_buffer.compute_returns(terminal_reward, gamma=args.reward_gamma)
    
    # Normalize returns to reduce variance (optional but recommended)
    returns_tensor = torch.tensor(returns, dtype=torch.float32)
    returns_mean = returns_tensor.mean()
    returns_std = returns_tensor.std() + 1e-8
    returns_normalized = (returns_tensor - returns_mean) / returns_std
    
    # Update agent using policy gradient
    # Recompute log probabilities with gradients from stored states and actions
    batch_size = 4  # Process 4 experiences at a time to limit memory
    num_batches = len(episode_buffer)
    has_logits_actions = episode_buffer.store_logits_actions
    has_neg2_actions = episode_buffer.store_neg2_actions
    
    for start_idx in range(0, num_batches, batch_size):
        end_idx = min(start_idx + batch_size, num_batches)
        
        # Collect batch data
        log_probs_list = []  # Combined log probs from all enabled action types
        log_probs_logits_list = [] if has_logits_actions else None
        returns_batch = []
        
        for i in range(start_idx, end_idx):
            # Move state to GPU with support for None elements
            # State format: (neg1_features, neg2_features, teacher_logits, teacher_scalars)
            def to_cuda(x):
                if x is None:
                    return None
                elif isinstance(x, list):
                    return [t.cuda(args.gpu) for t in x]
                else:
                    return x.cuda(args.gpu)
            state_gpu = tuple([to_cuda(item) for item in episode_buffer.states[i]])
            
            # Recompute action probabilities with gradient tracking
            # Use unwrapped agent_module to avoid DDP issues
            agent_output = agent_module(state_gpu)
            
            # Handle conditional output based on agent configuration
            # Agent can return: single tensor, or tuple of (neg1, neg2), (neg1, logits), (neg2, logits), etc.
            if isinstance(agent_output, tuple):
                # Multiple outputs - need to figure out what they are based on buffer flags
                output_list = list(agent_output)
            else:
                output_list = [agent_output]
            
            # Determine which outputs correspond to which action types
            # Order is: neg1 (if enabled), neg2 (if enabled), logits (if enabled)
            output_idx = 0
            log_prob_total = None
            batch_size_sample = None
            
            # Get neg1 feature actions if present
            neg1_action = episode_buffer.feature_actions[i]
            if neg1_action is not None:
                neg1_action = neg1_action.cuda(args.gpu)
                neg1_probs = output_list[output_idx]
                output_idx += 1
                log_prob_neg1 = torch.log(neg1_probs + 1e-8).sum(dim=1)  # [B]
                log_prob_total = log_prob_neg1
                batch_size_sample = log_prob_neg1.size(0)
            
            # Get neg2 feature actions if present
            if has_neg2_actions:
                neg2_action = episode_buffer.neg2_feature_actions[i]
                if neg2_action is not None:
                    neg2_action = neg2_action.cuda(args.gpu)
                    neg2_probs = output_list[output_idx]
                    output_idx += 1
                    log_prob_neg2 = torch.log(neg2_probs + 1e-8).sum(dim=1)  # [B]
                    if log_prob_total is None:
                        log_prob_total = log_prob_neg2
                        batch_size_sample = log_prob_neg2.size(0)
                    else:
                        log_prob_total = log_prob_total + log_prob_neg2
            
            # Handle logits actions if present
            if has_logits_actions and output_idx < len(output_list):
                logits_probs = output_list[output_idx]
                logits_action = episode_buffer.logits_actions[i].cuda(args.gpu)
                log_prob_logits = torch.log(logits_probs + 1e-8).sum(dim=1)  # [B]
                log_probs_logits_list.append(log_prob_logits)
            
            if log_prob_total is not None:
                log_probs_list.append(log_prob_total)
            
            # Broadcast the same return to all samples in this batch
            if batch_size_sample is not None:
                for _ in range(batch_size_sample):
                    returns_batch.append(returns_normalized[i].item())
        
        if not log_probs_list:
            continue
        
        # Concatenate batch
        log_probs = torch.cat(log_probs_list, dim=0)
        returns_gpu = torch.tensor(returns_batch, dtype=torch.float32).cuda(args.gpu)
        
        # Policy gradient loss: -E[G_t * log π(a_t|s_t)]
        # Negative because we want to maximize expected return
        loss = -(log_probs * returns_gpu).mean()
        
        # Add logits loss if present
        if has_logits_actions and log_probs_logits_list:
            log_probs_logits = torch.cat(log_probs_logits_list, dim=0)
            loss_logits = -(log_probs_logits * returns_gpu).mean()
            loss = loss + loss_logits
        
        # Check for NaN
        if torch.isnan(loss) or torch.isinf(loss):
            print(f"WARNING: NaN/Inf in episode agent loss at epoch {epoch}, skipping this batch", flush=True)
            continue
        
        agent_optimizer.zero_grad()
        loss.backward()
        
        # Gradient clipping
        torch.nn.utils.clip_grad_norm_(agent.parameters(), max_norm=1.0)
        
        agent_optimizer.step()
        
        agent_loss.update(loss.item(), end_idx - start_idx)
    
    if args.rank == 0:
        args.logger.info('Epoch:{}, agent Episode Loss:{:.6f}, Terminal Reward (val_dice*100):{:.2f}, Avg Return:{:.2f}'.format(
            epoch, agent_loss.avg, terminal_reward, returns_mean.item()))


def get_agent_state(trans_student_features, teacher_features, student_logits, teacher_logits, teacher_embeddings, targets, criterion_div):
    """
    Compute agent state from student and teacher features.
    
    NEW FORMAT: Returns spatial features + scalars separately to allow policy network
    to use convolutional encoding before pooling. Supports region-based training.
    
    Args:
        trans_student_features: list of transformed student features (layer -2), each [B, C, D, H, W] for 3D
        teacher_features: list of teacher features (layer -2), each [B, C, D, H, W] for 3D (may be None if not loaded)
        student_logits: student logits [B, num_cls, D, H, W]
        teacher_logits: list of teacher logits, each [B, num_cls, D, H, W]
        teacher_embeddings: list of teacher embeddings (layer -1), each [B, C, D, H, W] for 3D
        targets: ground truth labels [B, D, H, W] or [B, C, D, H, W] for region-based
        criterion_div: KL divergence criterion
    
    Returns:
        teacher_embeddings: list of teacher embedding tensors [B, C, D, H, W] (spatial)
        teacher_logits: list of teacher logit tensors [B, num_cls, D, H, W] (spatial)
        teacher_scalars: list of [B, 3] tensors containing [cos_sim, kl_div, ce_loss] per teacher
    """
    # Handle deep supervision: logits might be a list
    if isinstance(student_logits, (list, tuple)):
        # Deep supervision: use the final output (highest resolution)
        student_logits_for_agent = student_logits[0]
    else:
        student_logits_for_agent = student_logits
    
    # Check if using region-based training (multi-hot targets)
    is_region_based = targets.dim() == 5  # [B, C, D, H, W] for regions vs [B, D, H, W] for classes
    
    teacher_scalars_list = []
    
    for idx in range(len(teacher_embeddings)):  # Use teacher_embeddings length since teacher_features may be None
        # Compute scalar metrics for this teacher
        
        # 1. Feature cosine similarity (pooled student vs pooled teacher features)
        # Use teacher_features if available, otherwise use teacher_embeddings
        teacher_feat_for_cos = teacher_features[idx] if teacher_features[idx] is not None else teacher_embeddings[idx]
        
        if trans_student_features[idx].dim() == 5:
            # 3D: pool to [B, C]
            student_emb = F.adaptive_avg_pool3d(trans_student_features[idx], (1, 1, 1))
            student_emb = student_emb.view(student_emb.size(0), -1)
            teacher_feat_emb = F.adaptive_avg_pool3d(teacher_feat_for_cos, (1, 1, 1))
            teacher_feat_emb = teacher_feat_emb.view(teacher_feat_emb.size(0), -1)
        else:
            # 2D: pool to [B, C]
            student_emb = F.adaptive_avg_pool2d(trans_student_features[idx], (1, 1))
            student_emb = student_emb.view(student_emb.size(0), -1)
            teacher_feat_emb = F.adaptive_avg_pool2d(teacher_feat_for_cos, (1, 1))
            teacher_feat_emb = teacher_feat_emb.view(teacher_feat_emb.size(0), -1)
        
        feat_cos_sim = F.cosine_similarity(student_emb, teacher_feat_emb).unsqueeze(-1)  # [B, 1]
        
        # 2. Logit KL divergence (no pooling needed - criterion_div handles spatial dims)
        # For segmentation: [B, C, D, H, W] or [B, C, H, W] -> mean over spatial dims
        # For classification: [B, C] -> already scalar
        logit_kl_raw = criterion_div(student_logits_for_agent, teacher_logits[idx], unreduce=True)  # [B] or [B, D, H, W] or [B, H, W]
        if logit_kl_raw.dim() > 1:
            # Segmentation: average over spatial dimensions
            logit_kl = logit_kl_raw.view(logit_kl_raw.size(0), -1).mean(dim=1, keepdim=True)  # [B, 1]
        else:
            # Classification: already [B]
            logit_kl = logit_kl_raw.unsqueeze(-1)  # [B, 1]
        
        # 3. Cross-entropy loss (pooled) - supports both class-based and region-based
        teacher_logits_resized = teacher_logits[idx]
        
        if is_region_based:
            # Region-based: use BCE instead of CE
            # targets: [B, C, D, H, W], teacher_logits: [B, C, D, H, W]
            if teacher_logits[idx].dim() == 5:
                B, C, D_t, H_t, W_t = teacher_logits[idx].shape
                B_tgt, C_tgt, D_tgt, H_tgt, W_tgt = targets.shape
                if D_t != D_tgt or H_t != H_tgt or W_t != W_tgt:
                    teacher_logits_resized = F.interpolate(teacher_logits[idx], size=(D_tgt, H_tgt, W_tgt), 
                                                          mode='trilinear', align_corners=True)
            teachers_ce_raw = F.binary_cross_entropy_with_logits(teacher_logits_resized, targets.float(), reduction='none')
            # Average over channels and spatial dims: [B, C, D, H, W] -> [B]
            teachers_ce = teachers_ce_raw.mean(dim=[1, 2, 3, 4]).unsqueeze(-1)  # [B, 1]
        else:
            # Class-based: use CE
            if teacher_logits[idx].dim() == 5 and targets.dim() == 4:
                # Resize teacher logits to match target size if needed
                B, C, D_t, H_t, W_t = teacher_logits[idx].shape
                B_tgt, D_tgt, H_tgt, W_tgt = targets.shape
                if D_t != D_tgt or H_t != H_tgt or W_t != W_tgt:
                    teacher_logits_resized = F.interpolate(teacher_logits[idx], size=(D_tgt, H_tgt, W_tgt), 
                                                          mode='trilinear', align_corners=True)
            
            teachers_ce_raw = F.cross_entropy(teacher_logits_resized, targets, reduction='none')
            if teachers_ce_raw.dim() > 1:
                # Pool spatial dimensions: [B, D, H, W] -> [B] or [B, H, W] -> [B]
                teachers_ce = teachers_ce_raw.view(teachers_ce_raw.size(0), -1).mean(dim=1, keepdim=True)  # [B, 1]
            else:
                teachers_ce = teachers_ce_raw.unsqueeze(-1)  # [B, 1]
        
        # Concatenate scalars: [B, 3] = [cos_sim, kl_div, ce_loss]
        teacher_scalars = torch.cat([feat_cos_sim, logit_kl, teachers_ce], dim=1)  # [B, 3]
        teacher_scalars_list.append(teacher_scalars)
    
    # Return: spatial features (teacher_embeddings, teacher_logits) + scalars
    return teacher_embeddings, teacher_logits, teacher_scalars_list

def nnunet_get_pred_dice(logits, targets):
    """
    Compute per-sample cancer Dice score from nnUNet predictions.
    
    nnUNet uses sigmoid activation for region-based segmentation.
    In binary mode (C=2): channel 1 is cancer (already combined by dataloader).
    In 3-class mode (C=3): channels 1,2 are PCa,csPCa — combine into single cancer
    (same approach as provicnet combining ciPCA + csPCA).
    
    Args:
        logits: tensor [B, C, D, H, W], raw logits from nnUNet model
                C=2 for binary (prostate, cancer) or C=3 for 3-class (prostate, PCa, csPCa)
        targets: tensor [B, C, D, H, W], multi-hot ground truth labels
    
    Returns:
        dice_per_sample: tensor [B, C], Dice score for each sample and each region
        dice_mean_per_sample: tensor [B], combined cancer Dice per sample
    """
    predicted_probs = torch.sigmoid(logits)  # [B, C, D, H, W]
    target_channels = targets.shape[1]
    
    if target_channels == 2:
        # Binary: channel 1 is already combined cancer
        cancer_pred = (predicted_probs[:, 1] > 0.5).float()  # [B, D, H, W]
        cancer_target = targets[:, 1].float()  # [B, D, H, W]
    else:
        # 3-class: combine PCa (ch1) | csPCa (ch2) into single cancer
        cancer_pred = ((predicted_probs[:, 1] > 0.5) | (predicted_probs[:, 2] > 0.5)).float()
        cancer_target = torch.clamp(targets[:, 1] + targets[:, 2], 0, 1).float()
    
    eps = 1e-7
    intersection = (cancer_pred * cancer_target).sum(dim=(1, 2, 3))  # [B]
    pred_sum = cancer_pred.sum(dim=(1, 2, 3))  # [B]
    target_sum = cancer_target.sum(dim=(1, 2, 3))  # [B]
    dice_cancer = (2.0 * intersection + eps) / (pred_sum + target_sum + eps)  # [B]
    
    # Build compatible [B, C] return format
    dice_per_sample = torch.zeros(logits.shape[0], target_channels, device=logits.device)
    if target_channels == 2:
        dice_per_sample[:, 1] = dice_cancer
    else:
        dice_per_sample[:, 1] = dice_cancer  # combined cancer in both cancer channels
        dice_per_sample[:, 2] = dice_cancer
    
    return dice_per_sample, dice_cancer

def nnunet_get_pred_bce(logits, targets):
    """
    Compute per-sample cancer BCE from nnUNet predictions.
    
    In binary mode (C=2): channel 1 is cancer (already combined by dataloader).
    In 3-class mode (C=3): combine PCa (ch1) and csPCa (ch2) into single cancer
    probability using max (consistent with dataloader's binary conversion).
    
    Args:
        logits: tensor [B, C, D, H, W], raw logits from nnUNet model
                C=2 for binary or C=3 for 3-class
        targets: tensor [B, C, D, H, W], multi-hot ground truth labels
    
    Returns:
        bce_per_sample: tensor [B], mean cancer BCE per sample
    """
    probs = torch.sigmoid(logits)  # [B, C, D, H, W]
    target_channels = targets.shape[1]
    
    if target_channels == 2:
        # Binary: channel 1 is already combined cancer
        cancer_prob = probs[:, 1]  # [B, D, H, W]
        cancer_target = targets[:, 1].float()
    else:
        # 3-class: combine PCa and csPCa using max (same as dataloader)
        cancer_prob = torch.max(probs[:, 1], probs[:, 2])  # [B, D, H, W]
        cancer_target = torch.clamp(targets[:, 1] + targets[:, 2], 0, 1).float()
    
    # Manual BCE (since we combined probs, can't use BCEWithLogitsLoss)
    eps = 1e-7
    cancer_prob_clamped = torch.clamp(cancer_prob, eps, 1 - eps)
    bce = -(cancer_target * torch.log(cancer_prob_clamped) + 
            (1 - cancer_target) * torch.log(1 - cancer_prob_clamped))
    
    bce_per_sample = bce.mean(dim=(1, 2, 3))  # [B]
    
    return bce_per_sample

def provicnet_get_pred_bce(logits, targets):
    """
    Compute per-sample BCE score from ProViCNet predictions.
    
    ProViCNet uses softmax activation for class-based segmentation with 4 classes:
    [background, prostate, ciPCA, csPCA]. We need to convert this to region-based
    format and compute BCE against the cancer channel(s).
    
    Args:
        logits: tensor [B, 4, D, H, W], raw logits from ProViCNet model
                4 channels: [background, prostate, ciPCA, csPCA]
        targets: tensor [B, C, D, H, W], multi-hot ground truth labels
                For binary mode: [prostate, cancer]
                For 3-class mode: [prostate, PCa, csPCa]
    
    Returns:
        bce_per_sample: tensor [B], mean BCE for cancer channel(s)
    """
    # Apply softmax to get probabilities (provicnet is class-based)
    probs = F.softmax(logits, dim=1)  # [B, 4, D, H, W]
    
    # Compute cancer probability as sum of ciPCA (ch2) and csPCA (ch3)
    cancer_prob = probs[:, 2] + probs[:, 3]  # [B, D, H, W]
    cancer_prob = torch.clamp(cancer_prob, 0, 1)
    
    # Determine target format: binary (2 ch) or 3-class (3 ch)
    target_channels = targets.shape[1]
    
    if target_channels == 2:
        # Binary mode: targets[:, 1] is cancer
        target_cancer = targets[:, 1].float()  # [B, D, H, W]
    else:
        # 3-class mode: combine PCa and csPCa channels
        # targets[:, 1] is PCa, targets[:, 2] is csPCa
        target_cancer = torch.clamp(targets[:, 1] + targets[:, 2], 0, 1).float()  # [B, D, H, W]
    
    # Compute BCE between cancer_prob and target_cancer
    # Add small epsilon to avoid log(0)
    eps = 1e-7
    cancer_prob_clamped = torch.clamp(cancer_prob, eps, 1 - eps)
    bce = -(target_cancer * torch.log(cancer_prob_clamped) + 
            (1 - target_cancer) * torch.log(1 - cancer_prob_clamped))
    
    # Mean BCE across spatial dimensions: [B, D, H, W] -> [B]
    bce_per_sample = bce.mean(dim=(1, 2, 3))  # [B]
    
    return bce_per_sample

def provicnet_get_pred_dice(logits, targets):
    """
    Compute per-sample cancer Dice score from ProViCNet predictions.
    
    ProViCNet uses softmax activation with 4 classes:
    [background, prostate, ciPCA, csPCA]. We compute Dice on the combined
    cancer channel (ciPCA + csPCA) against the cancer ground truth.
    
    Args:
        logits: tensor [B, 4, D, H, W], raw logits from ProViCNet model
        targets: tensor [B, C, D, H, W], multi-hot ground truth labels
                For binary mode: [prostate, cancer]
                For 3-class mode: [prostate, PCa, csPCa]
    
    Returns:
        dice_mean_per_sample: tensor [B], cancer Dice score per sample
    """
    # Apply softmax to get probabilities (provicnet is class-based)
    probs = F.softmax(logits, dim=1)  # [B, 4, D, H, W]
    
    # Compute cancer probability as sum of ciPCA (ch2) and csPCA (ch3)
    cancer_prob = probs[:, 2] + probs[:, 3]  # [B, D, H, W]
    cancer_prob = torch.clamp(cancer_prob, 0, 1)
    cancer_pred = (cancer_prob > 0.5).float()  # [B, D, H, W]
    
    # Determine target format: binary (2 ch) or 3-class (3 ch)
    target_channels = targets.shape[1]
    
    if target_channels == 2:
        # Binary mode: targets[:, 1] is cancer
        target_cancer = targets[:, 1].float()  # [B, D, H, W]
    else:
        # 3-class mode: combine PCa and csPCa channels
        target_cancer = torch.clamp(targets[:, 1] + targets[:, 2], 0, 1).float()  # [B, D, H, W]
    
    # Compute Dice: 2*TP / (2*TP + FP + FN)
    eps = 1e-7
    intersection = (cancer_pred * target_cancer).sum(dim=(1, 2, 3))  # [B]
    pred_sum = cancer_pred.sum(dim=(1, 2, 3))  # [B]
    target_sum = target_cancer.sum(dim=(1, 2, 3))  # [B]
    
    dice_mean_per_sample = (2.0 * intersection + eps) / (pred_sum + target_sum + eps)  # [B]
    
    return dice_mean_per_sample


def prostatlasdiff_get_pred_dice(logits, targets):
    """
    Compute per-sample Dice score from ProstatLasDiff predictions.
    
    ProstatLasDiff may use different activation/prediction strategy than nnUNet.
    TODO: Fill in the correct computation based on ProstatLasDiff's prediction method.
    
    Args:
        logits: tensor [B, C, D, H, W], raw logits from ProstatLasDiff model
        targets: tensor [B, C, D, H, W], ground truth labels (region-based)
    
    Returns:
        dice_per_sample: tensor [B, C], Dice score for each sample and each region
                         For ProstatLasDiff, C=1 since it outputs single binary mask
        dice_mean_per_sample: tensor [B], mean Dice across regions for each sample
    """
    B, C, D, H, W = logits.shape
    
    # ProstatLasDiff prediction pipeline
    predicted_probs = torch.sigmoid(logits)
    
    # Normalize by max if non-zero
    max_val = predicted_probs.max()
    if max_val > 0:
        predicted_probs = predicted_probs / max_val
    
    # Scale and clip
    predicted_probs = predicted_probs / 0.2
    predicted_probs = torch.clamp(predicted_probs, 0, 1)
    
    # Get binary predictions from channel 1 (cancer channel): [B, D, H, W]
    predicted_labels = (predicted_probs[:, 1, ...] > 0.1).float()  # [B, D, H, W]
    
    # Determine target format: binary (2 ch) or 3-class (3 ch)
    target_channels = targets.shape[1]
    
    if target_channels == 2:
        # Binary mode: targets[:, 1] is cancer (PCa+csPCa combined)
        agg_gt_labels = targets[:, 1, ...].float()  # [B, D, H, W]
    else:
        # 3-class mode: combine PCa (ch1) + csPCa (ch2) into single cancer GT
        agg_gt_labels = torch.clamp(targets[:, 1] + targets[:, 2], 0, 1).float()  # [B, D, H, W]
    
    # Compute Dice for this binary prediction: [B, D, H, W] vs [B, D, H, W]
    # Sum over spatial dimensions (D, H, W), keep B
    intersection = (predicted_labels * agg_gt_labels).sum(dim=(1, 2, 3))  # [B]
    pred_sum = predicted_labels.sum(dim=(1, 2, 3))  # [B]
    target_sum = agg_gt_labels.sum(dim=(1, 2, 3))  # [B]
    
    # Dice = 2 * intersection / (pred_sum + target_sum + eps)
    eps = 1e-7
    dice_cancer = (2.0 * intersection + eps) / (pred_sum + target_sum + eps)  # [B]
    
    # For compatibility with nnunet_get_pred_dice return format:
    # Return [B, C] where C matches target_channels
    dice_per_sample = torch.zeros(B, target_channels, device=logits.device)
    if target_channels == 2:
        dice_per_sample[:, 1] = dice_cancer  # Binary: cancer channel
    else:
        dice_per_sample[:, 2] = dice_cancer  # 3-class: csPCa channel
    
    # Mean Dice - only cancer is valid for ProstatLasDiff
    dice_mean_per_sample = dice_cancer  # [B]
    
    return dice_per_sample, dice_mean_per_sample


def prostatlasdiff_get_pred_bce(logits, targets):
    """
    Compute per-sample BCE score from ProstatLasDiff predictions.
    
    ProstatLasDiff outputs cancer probability in channel 1.
    We only compute BCE for channel 1 (cancer channel) between teacher and target.
    
    Args:
        logits: tensor [B, C, D, H, W], raw logits from ProstatLasDiff model
                Channel 1 is the cancer/csPCa channel
        targets: tensor [B, C, D, H, W], ground truth labels (region-based)
                Channel 1 is the cancer channel (PCa+csPCa in binary mode, or PCa in 3-class)
    
    Returns:
        bce_per_sample: tensor [B], mean BCE for channel 1 only
    """
    # Only use channel 1 (cancer channel) for ProstatLasDiff
    # logits[:, 1] is the cancer logits, targets[:, 1] is cancer ground truth
    logits_ch1 = logits[:, 1:2]  # [B, 1, D, H, W] - keep dim for BCE
    targets_ch1 = targets[:, 1:2].float()  # [B, 1, D, H, W]
    
    # Use BCEWithLogitsLoss (applies sigmoid internally, don't apply sigmoid beforehand)
    bce_loss = nn.BCEWithLogitsLoss(reduction='none')
    bce_per_sample = bce_loss(logits_ch1, targets_ch1)  # [B, 1, D, H, W]
    
    # Mean over spatial dimensions: [B, 1, D, H, W] -> [B]
    bce_per_sample = bce_per_sample.mean(dim=(1, 2, 3, 4))  # [B]
    
    return bce_per_sample


def normalize_provicnet_logits(logits, target_channels=3):
    """
    Normalize provicnet 4-channel softmax logits to region-based format.
    
    provicnet has 4 channels: [background, prostate, ciPCA, csPCA] with softmax output.
    We convert to 3-channel region-based format: [prostate, PCa, csPCa] where:
    - prostate = 1 - background (prostate region probability)
    - PCa = prostate_ch (prostate channel, not useful for cancer detection)
    - cancer = ciPCA + csPCA (combined cancer probability)
    
    For binary mode (target_channels=2):
    - prostate = 1 - background
    - cancer = ciPCA + csPCA
    
    Args:
        logits: tensor [B, 4, D, H, W] raw logits from provicnet
        target_channels: int, 2 for binary mode, 3 for 3-class mode
    
    Returns:
        normalized_probs: tensor [B, target_channels, D, H, W] in region-based format
    """
    # Apply softmax to get probabilities (provicnet is class-based)
    probs = F.softmax(logits, dim=1)  # [B, 4, D, H, W]
    
    # probs channels: [background, prostate_gland, ciPCA, csPCA]
    # background = probs[:, 0]
    # prostate_gland = probs[:, 1]
    ciPCA = probs[:, 2]
    csPCA = probs[:, 3]
    
    # Compute cancer probability as sum of ciPCA and csPCA
    cancer_prob = ciPCA + csPCA  # [B, D, H, W]
    # Clip to [0, 1] since these are probabilities
    cancer_prob = torch.clamp(cancer_prob, 0, 1)
    
    # Compute prostate region probability (everything that's not background)
    prostate_prob = 1.0 - probs[:, 0]  # [B, D, H, W]
    prostate_prob = torch.clamp(prostate_prob, 0, 1)
    
    if target_channels == 2:
        # Binary mode: [prostate, cancer]
        return torch.stack([prostate_prob, cancer_prob], dim=1)  # [B, 2, D, H, W]
    else:
        # 3-class mode: [prostate, PCa (ciPCA), csPCa]
        # Note: PCa channel here uses ciPCA prob; for better consistency could use prostate_gland
        return torch.stack([prostate_prob, ciPCA, csPCA], dim=1)  # [B, 3, D, H, W]


def compute_per_teacher_disagreement(teacher_logits, is_region_based=True, channels_to_include=None,
                                      teacher_names=None):
    """
    Compute per-teacher disagreement with ensemble average.
    
    Each teacher gets a unique disagreement score showing how much it deviates from
    the ensemble consensus. This gives the agent information about which teachers
    have unique/different perspectives.
    
    Args:
        teacher_logits: list of teacher logit tensors, each [B, C, D, H, W]
        is_region_based: bool, whether using region-based (sigmoid) vs class-based (softmax)
        channels_to_include: list of channels to focus on (e.g., [1, 2] for PCa/csPCa)
        teacher_names: list of teacher name strings (e.g., ['nnunet', 'provicnet', 'prostatlasdiff'])
                      If None, all teachers are treated as region-based (sigmoid)
    
    Returns:
        disagreement_list: list of [B, 1] tensors, one per teacher
                          Higher value = more disagreement with ensemble
    """
    num_teachers = len(teacher_logits)
    if num_teachers < 2:
        # Single teacher: no disagreement
        batch_size = teacher_logits[0].size(0)
        return [torch.zeros(batch_size, 1, device=teacher_logits[0].device)]
    
    # Determine target number of channels for provicnet normalization
    # Use the first non-provicnet teacher's channel count, or default to 3
    target_channels = 3
    if teacher_names is not None:
        for idx, name in enumerate(teacher_names):
            if 'provicnet' not in name.lower() and teacher_logits[idx].shape[1] in [2, 3]:
                target_channels = teacher_logits[idx].shape[1]
                break
    
    # Convert each teacher's logits to probabilities
    # Handle provicnet specially (4-channel softmax -> region-based format)
    teacher_probs_list = []
    for idx, t in enumerate(teacher_logits):
        teacher_name = teacher_names[idx].lower() if teacher_names and idx < len(teacher_names) else ''
        
        if 'provicnet' in teacher_name.lower() and t.shape[1] == 4:
            # provicnet: 4-channel softmax -> convert to region-based format
            probs = normalize_provicnet_logits(t, target_channels=target_channels)
        elif is_region_based:
            # Region-based teachers (nnunet, etc.): use sigmoid
            probs = torch.sigmoid(t)
        else:
            # Class-based: use softmax
            probs = F.softmax(t, dim=1)
        
        teacher_probs_list.append(probs)
    
    # Stack and compute ensemble average (all should now have same format)
    probs_stack = torch.stack(teacher_probs_list, dim=0)  # [num_teachers, B, C, D, H, W]
    ensemble_probs = probs_stack.mean(dim=0)  # [B, C, D, H, W]
    
    # Select channels if specified
    if channels_to_include is not None:
        ensemble_probs = ensemble_probs[:, channels_to_include]
        teacher_probs_list = [t[:, channels_to_include] for t in teacher_probs_list]
    
    # Compute per-teacher disagreement with ensemble
    disagreement_list = []
    for teacher_probs in teacher_probs_list:
        # L1 distance between teacher and ensemble probabilities
        disagreement = torch.abs(teacher_probs - ensemble_probs)  # [B, C, D, H, W]
        
        # Average over all dims except batch: [B, C, D, H, W] -> [B, 1]
        teacher_disagreement = disagreement.view(disagreement.size(0), -1).mean(dim=1, keepdim=True)
        
        disagreement_list.append(teacher_disagreement)
    
    return disagreement_list


def get_agent_state_neg1(trans_student_features_neg1, teacher_features_neg1, teacher_logits, targets_region_based, teacher_names=None, use_disagreement=False):
    """
    Compute agent state from student and teacher features at layer -1 only.
    
    This version uses teacher BCE scores AND optionally per-teacher disagreement to inform the agent.
    Each teacher gets a unique disagreement score showing how much it deviates from
    the ensemble consensus (if use_disagreement=True).
    
    PolicyTrans will convolve teacher_features_neg1 and teacher_logits separately,
    so we return them along with the scalar metrics.
    
    Args:
        trans_student_features_neg1: list of transformed student features (layer -1), each [B, C, D, H, W] for 3D
        teacher_features_neg1: list of teacher features (layer -1), each [B, C, D, H, W] for 3D  
        teacher_logits: list of teacher logits, each [B, num_cls, D, H, W]
        targets_region_based: ground truth labels [B, C, D, H, W] for region-based training
        teacher_names: list of strings indicating teacher type ('nnunet' or 'prostatlasdiff')
                      If None, defaults to all 'nnunet'
        use_disagreement: bool, whether to include per-teacher disagreement in agent state
    
    Returns:
        teacher_features_neg1: list of teacher feature tensors [B, C, D, H, W] (for PolicyTrans convolution)
        None: placeholder for neg2 features (not used in neg1-only mode)
        teacher_logits: list of teacher logit tensors [B, num_cls, D, H, W] (for PolicyTrans convolution)
        teacher_scalars_list: list of [B, 2] or [B, 3] tensors containing:
            - If use_disagreement=False: [feat_cos_sim, dice_mean]
            - If use_disagreement=True: [feat_cos_sim, dice_mean, disagreement]
        teacher_dice_means: list of [B] tensors containing cancer Dice for each teacher (for logging)
    """
    num_teachers = len(teacher_features_neg1)
    
    # Default teacher names to nnunet if not specified
    if teacher_names is None:
        teacher_names = ['nnunet'] * num_teachers
    
    # Compute per-teacher disagreement with ensemble (focus on cancer channels) only if needed
    if use_disagreement:
        is_region_based = targets_region_based.dim() == 5
        # Derive num_channels from the GROUND TRUTH (targets), not teacher logits.
        # For prostatlasdiff, logits shape[1]==3 is 2.5D modality slices (NOT classes),
        # so inferring num_channels from teacher logits would give the wrong answer.
        # Targets are region-based: [B, C, D, H, W] where C=2 for binary, 3 for 3-class.
        if is_region_based:
            num_channels = targets_region_based.shape[1]
        else:
            num_channels = 3

        if num_channels == 2:
            channels_to_include = [1]  # Binary: cancer only (center slice for prostatlasdiff)
        else:
            channels_to_include = [1, 2]  # 3-class: PCa and csPCa
        disagreement_list = compute_per_teacher_disagreement(
            teacher_logits,
            is_region_based=is_region_based,
            channels_to_include=channels_to_include if is_region_based else None,
            teacher_names=teacher_names
        )
    
    teacher_scalars_list = []
    teacher_dice_means = []  # For logging teacher Dice scores
    
    for idx in range(num_teachers):
        # 1. Feature cosine similarity on flattened spatial features (no pooling)
        student_flat = trans_student_features_neg1[idx].view(trans_student_features_neg1[idx].size(0), -1)  # [B, C*D*H*W]
        teacher_flat = teacher_features_neg1[idx].view(teacher_features_neg1[idx].size(0), -1)  # [B, C*D*H*W]
        
        feat_cos_sim = F.cosine_similarity(student_flat, teacher_flat).unsqueeze(-1)  # [B, 1]
        
        # 2. Compute teacher cancer Dice score based on teacher type
        # Dice is sharper than BCE for quality differentiation:
        # a teacher with 5% FP gets Dice~0 on normal cases, while on cancer cases
        # nnUNet Dice=0.7 vs ProViCNet Dice=0.3 creates much bigger weight differences
        teacher_name = teacher_names[idx] if idx < len(teacher_names) else 'nnunet'
        
        if 'prostatlasdiff' in teacher_name.lower():
            # ProstatLasDiff: uses channel 1 for cancer
            _, dice_mean = prostatlasdiff_get_pred_dice(teacher_logits[idx], targets_region_based)
        elif 'provicnet' in teacher_name.lower():
            # ProViCNet: 4-channel softmax -> merge cancer channels -> Dice
            dice_mean = provicnet_get_pred_dice(teacher_logits[idx], targets_region_based)
        else:
            # Default to nnUNet: sigmoid + threshold at 0.5
            _, dice_mean = nnunet_get_pred_dice(teacher_logits[idx], targets_region_based)
        
        dice_mean_unsqueezed = dice_mean.unsqueeze(-1)  # [B, 1]
        
        # 3. Concatenate scalars based on use_disagreement flag
        if use_disagreement:
            # Get per-teacher disagreement (already computed above)
            teacher_disagreement = disagreement_list[idx]  # [B, 1]
            # Concatenate scalars: [B, 3] = [cos_sim, dice_mean, disagreement]
            teacher_scalars = torch.cat([feat_cos_sim, dice_mean_unsqueezed, teacher_disagreement], dim=1)
        else:
            # Concatenate scalars: [B, 2] = [cos_sim, dice_mean]
            teacher_scalars = torch.cat([feat_cos_sim, dice_mean_unsqueezed], dim=1)
        
        teacher_scalars_list.append(teacher_scalars)
        teacher_dice_means.append(dice_mean)  # Store for logging
    
    # Return: (neg1_features, neg2_features=None, logits, scalars, dice_means)
    return teacher_features_neg1, None, teacher_logits, teacher_scalars_list, teacher_dice_means


def _GET_AGENT_STATE_NEG2_MARKER_(trans_student_features_neg2, teacher_features_neg2, teacher_logits, targets_region_based, teacher_names=None, use_disagreement=False):
    """
    Compute agent state from student and teacher features at layer -2 only.
    
    Similar to get_agent_state_neg1 but uses layer -2 features instead.
    
    Args:
        trans_student_features_neg2: list of transformed student features (layer -2), each [B, C, D, H, W] for 3D
        teacher_features_neg2: list of teacher features (layer -2), each [B, C, D, H, W] for 3D  
        teacher_logits: list of teacher logits, each [B, num_cls, D, H, W]
        targets_region_based: ground truth labels [B, C, D, H, W] for region-based training
        teacher_names: list of strings indicating teacher type
        use_disagreement: bool, whether to include per-teacher disagreement in agent state
    
    Returns:
        None: placeholder for neg1 features (not used in neg2-only mode)
        teacher_features_neg2: list of teacher feature tensors [B, C, D, H, W] (for PolicyTrans convolution)
        teacher_logits: list of teacher logit tensors [B, num_cls, D, H, W]
        teacher_scalars_list: list of [B, 2] or [B, 3] tensors
        teacher_dice_means: list of [B] tensors
    """
    num_teachers = len(teacher_features_neg2)
    
    if teacher_names is None:
        teacher_names = ['nnunet'] * num_teachers
    
    # Compute per-teacher disagreement if needed
    if use_disagreement:
        is_region_based = targets_region_based.dim() == 5
        num_channels = 3
        for idx, t_logits in enumerate(teacher_logits):
            t_name = teacher_names[idx].lower() if teacher_names and idx < len(teacher_names) else ''
            if 'provicnet' not in t_name and t_logits.shape[1] in [2, 3]:
                num_channels = t_logits.shape[1]
                break
        
        channels_to_include = [1] if num_channels == 2 else [1, 2]
        disagreement_list = compute_per_teacher_disagreement(
            teacher_logits,
            is_region_based=is_region_based,
            channels_to_include=channels_to_include if is_region_based else None,
            teacher_names=teacher_names
        )
    
    teacher_scalars_list = []
    teacher_dice_means = []
    
    for idx in range(num_teachers):
        # Skip if teacher doesn't have neg2 features
        if teacher_features_neg2[idx] is None:
            # Create dummy scalars
            batch_size = teacher_logits[idx].size(0)
            device = teacher_logits[idx].device
            if use_disagreement:
                teacher_scalars = torch.zeros(batch_size, 3, device=device)
            else:
                teacher_scalars = torch.zeros(batch_size, 2, device=device)
            teacher_scalars_list.append(teacher_scalars)
            teacher_dice_means.append(torch.zeros(batch_size, device=device))
            continue
        
        # 1. Feature cosine similarity on flattened spatial features
        student_flat = trans_student_features_neg2[idx].view(trans_student_features_neg2[idx].size(0), -1)
        teacher_flat = teacher_features_neg2[idx].view(teacher_features_neg2[idx].size(0), -1)
        
        feat_cos_sim = F.cosine_similarity(student_flat, teacher_flat).unsqueeze(-1)
        
        # 2. Compute teacher cancer Dice score
        teacher_name = teacher_names[idx] if idx < len(teacher_names) else 'nnunet'
        
        if 'prostatlasdiff' in teacher_name.lower():
            _, dice_mean = prostatlasdiff_get_pred_dice(teacher_logits[idx], targets_region_based)
        elif 'provicnet' in teacher_name.lower():
            dice_mean = provicnet_get_pred_dice(teacher_logits[idx], targets_region_based)
        else:
            _, dice_mean = nnunet_get_pred_dice(teacher_logits[idx], targets_region_based)
        
        dice_mean_unsqueezed = dice_mean.unsqueeze(-1)
        
        # 3. Concatenate scalars
        if use_disagreement:
            teacher_disagreement = disagreement_list[idx]
            teacher_scalars = torch.cat([feat_cos_sim, dice_mean_unsqueezed, teacher_disagreement], dim=1)
        else:
            teacher_scalars = torch.cat([feat_cos_sim, dice_mean_unsqueezed], dim=1)
        
        teacher_scalars_list.append(teacher_scalars)
        teacher_dice_means.append(dice_mean)
    
    # Return: (neg1_features=None, neg2_features, logits, scalars, dice_means)
    return None, teacher_features_neg2, teacher_logits, teacher_scalars_list, teacher_dice_means


def get_agent_state_neg1_neg2(trans_student_features_neg1, trans_student_features_neg2, 
                               teacher_features_neg1, teacher_features_neg2, 
                               teacher_logits, targets_region_based, teacher_names=None, use_disagreement=False):
    """
    Compute agent state from student and teacher features at BOTH layer -1 and layer -2.
    
    This combines information from both layers for richer agent state representation.
    The scalars include feature cosine similarity from BOTH layers.
    
    Args:
        trans_student_features_neg1: list of transformed student features (layer -1)
        trans_student_features_neg2: list of transformed student features (layer -2)
        teacher_features_neg1: list of teacher features (layer -1)
        teacher_features_neg2: list of teacher features (layer -2)
        teacher_logits: list of teacher logits
        targets_region_based: ground truth labels
        teacher_names: list of teacher name strings
        use_disagreement: bool, whether to include disagreement scores
    
    Returns:
        teacher_features_neg1: list of teacher neg1 feature tensors
        teacher_features_neg2: list of teacher neg2 feature tensors
        teacher_logits: list of teacher logit tensors
        teacher_scalars_list: list of [B, 3] or [B, 4] tensors containing:
            - If use_disagreement=False: [neg1_cos_sim, neg2_cos_sim, dice_mean]
            - If use_disagreement=True: [neg1_cos_sim, neg2_cos_sim, dice_mean, disagreement]
        teacher_dice_means: list of [B] tensors
    """
    num_teachers = len(teacher_features_neg1)
    
    if teacher_names is None:
        teacher_names = ['nnunet'] * num_teachers
    
    # Compute per-teacher disagreement if needed
    if use_disagreement:
        is_region_based = targets_region_based.dim() == 5
        num_channels = 3
        for idx, t_logits in enumerate(teacher_logits):
            t_name = teacher_names[idx].lower() if teacher_names and idx < len(teacher_names) else ''
            if 'provicnet' not in t_name and t_logits.shape[1] in [2, 3]:
                num_channels = t_logits.shape[1]
                break
        
        channels_to_include = [1] if num_channels == 2 else [1, 2]
        disagreement_list = compute_per_teacher_disagreement(
            teacher_logits,
            is_region_based=is_region_based,
            channels_to_include=channels_to_include if is_region_based else None,
            teacher_names=teacher_names
        )
    
    teacher_scalars_list = []
    teacher_dice_means = []
    
    for idx in range(num_teachers):
        # 1. Feature cosine similarity for NEG1 layer
        student_flat_neg1 = trans_student_features_neg1[idx].view(trans_student_features_neg1[idx].size(0), -1)
        teacher_flat_neg1 = teacher_features_neg1[idx].view(teacher_features_neg1[idx].size(0), -1)
        feat_cos_sim_neg1 = F.cosine_similarity(student_flat_neg1, teacher_flat_neg1).unsqueeze(-1)
        
        # 2. Feature cosine similarity for NEG2 layer
        if teacher_features_neg2[idx] is not None:
            student_flat_neg2 = trans_student_features_neg2[idx].view(trans_student_features_neg2[idx].size(0), -1)
            teacher_flat_neg2 = teacher_features_neg2[idx].view(teacher_features_neg2[idx].size(0), -1)
            feat_cos_sim_neg2 = F.cosine_similarity(student_flat_neg2, teacher_flat_neg2).unsqueeze(-1)
        else:
            # If neg2 not available for this teacher, use neg1 similarity as proxy
            feat_cos_sim_neg2 = feat_cos_sim_neg1.clone()
        
        # 3. Compute teacher cancer Dice score
        teacher_name = teacher_names[idx] if idx < len(teacher_names) else 'nnunet'
        
        if 'prostatlasdiff' in teacher_name.lower():
            _, dice_mean = prostatlasdiff_get_pred_dice(teacher_logits[idx], targets_region_based)
        elif 'provicnet' in teacher_name.lower():
            dice_mean = provicnet_get_pred_dice(teacher_logits[idx], targets_region_based)
        else:
            _, dice_mean = nnunet_get_pred_dice(teacher_logits[idx], targets_region_based)
        
        dice_mean_unsqueezed = dice_mean.unsqueeze(-1)
        
        # 4. Concatenate scalars: [neg1_cos_sim, neg2_cos_sim, dice_mean, (optional) disagreement]
        if use_disagreement:
            teacher_disagreement = disagreement_list[idx]
            teacher_scalars = torch.cat([feat_cos_sim_neg1, feat_cos_sim_neg2, dice_mean_unsqueezed, teacher_disagreement], dim=1)
        else:
            teacher_scalars = torch.cat([feat_cos_sim_neg1, feat_cos_sim_neg2, dice_mean_unsqueezed], dim=1)
        
        teacher_scalars_list.append(teacher_scalars)
        teacher_dice_means.append(dice_mean)
    
    # Return: (neg1_features, neg2_features, logits, scalars, dice_means)
    return teacher_features_neg1, teacher_features_neg2, teacher_logits, teacher_scalars_list, teacher_dice_means


def train_avg(train_loader, model, criterion_list, optimizer, epoch, device, 
          args, feat_trans, teacher_models, teacher_names=None):
    
    train_loss = AverageMeter('train_loss', ':.4e')
    train_loss_cls = AverageMeter('train_loss_cls', ':.4e')
    train_loss_kd = AverageMeter('train_loss_kd', ':.4e')
    train_loss_feat = AverageMeter('train_loss_feat', ':.4e')

    # Initialize metrics based on dataset type
    if hasattr(args, 'dataset') and args.dataset == 'pimed':
        # Segmentation: only count total samples
        total = 0
        top1_num = 0  # Initialize to avoid errors, but won't be used
        top5_num = 0  # Initialize to avoid errors, but won't be used
    else:
        # Classification: track accuracy metrics
        top1_num = 0
        top5_num = 0
        total = 0

    lr = adjust_lr(optimizer, epoch, args)

    start_time = time.time()
    criterion_ce = criterion_list[0]
    criterion_div = criterion_list[1]

    model.train()
    
    # Detect if using pre-extracted features (PIMED) or online inference (CIFAR)
    use_preextracted_features = (teacher_models is None)
    
    for batch_idx, batch_data in enumerate(train_loader):
        batch_start_time = time.time()
        
        # Unpack batch based on dataset type
        if use_preextracted_features:
            # PIMED: (img, label, teacher_features, teacher_logits, case_ids, label_original, prostate_mask)
            if len(batch_data) == 7:
                inputs, targets, preextracted_teacher_features, preextracted_teacher_logits, case_ids, targets_original, prostate_mask = batch_data
            elif len(batch_data) == 6:
                inputs, targets, preextracted_teacher_features, preextracted_teacher_logits, case_ids, targets_original = batch_data
                prostate_mask = None
            else:
                # Backward compatibility if dataloader doesn't return original labels
                inputs, targets, preextracted_teacher_features, preextracted_teacher_logits, case_ids = batch_data
                targets_original = targets
                prostate_mask = None
        else:
            # CIFAR: (img, label)
            inputs, targets = batch_data
            targets_original = targets
            prostate_mask = None
        
        inputs = inputs.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)
        
        optimizer.zero_grad()
        
        features, logits = model(inputs, is_feat=True) 
        trans_student_features = feat_trans(features[-2])
        
        teacher_logits = []
        teacher_features = []
        teacher_embeddings = []
        
        if use_preextracted_features:
            # PIMED: Use pre-extracted features from dataloader
            with torch.no_grad():
                for i in range(len(preextracted_teacher_features)):
                    # preextracted_teacher_features[i] is a dict with 'layer_minus_1' (and optionally 'layer_minus_2')
                    t_feat_dict = preextracted_teacher_features[i]
                    t_logits = preextracted_teacher_logits[i].to(device, non_blocking=True)
                    
                    # Extract features and move to device
                    # layer_minus_2 is optional (may not be loaded to save memory)
                    t_feat_neg2 = t_feat_dict.get('layer_minus_2', None)
                    if t_feat_neg2 is not None:
                        t_feat_neg2 = t_feat_neg2.to(device, non_blocking=True)  # [B, C, D, H, W]
                    t_feat_neg1 = t_feat_dict['layer_minus_1'].to(device, non_blocking=True)  # [B, C, D, H, W]
                    
                    teacher_features.append(t_feat_neg2)  # May be None
                    teacher_logits.append(t_logits)
                    teacher_embeddings.append(t_feat_neg1)
        else:
            # CIFAR: Run teacher inference online
            with torch.no_grad():
                for t_model in teacher_models:
                    t_features, t_logits = t_model(inputs, is_feat=True)
                    t_feature = t_features[-1]
                    t_feature = t_feature.detach() 
                    t_logits = t_logits.detach() 
                    
                    teacher_features.append(t_features[-2])
                    teacher_logits.append(t_logits)
                    teacher_embeddings.append(t_features[-1])

        loss_cls = criterion_ce(logits, targets)
        
        # No logit distillation - set to 0
        loss_kd = torch.tensor(0.).cuda(args.gpu)
        
        loss_feat = torch.tensor(0.).cuda(args.gpu)
        teacher_num = len(teacher_embeddings)  # Use embeddings length since features may be None
        
        # Use MaskedFeatureMSELoss3D with cancer-correctness masking
        from distiller_zoo import MaskedFeatureMSELoss3D
        feat_kd_func = MaskedFeatureMSELoss3D(normalize=True)
        
        use_feature_masking = (getattr(args, 'use_feature_masking', False) 
                               and teacher_names is not None 
                               and use_preextracted_features
                               and targets.dim() == 5)

        for idx in range(teacher_num):
            # Use teacher_features if available, otherwise use teacher_embeddings
            teacher_feat = teacher_features[idx] if teacher_features[idx] is not None else teacher_embeddings[idx]
            
            # Compute cancer-correctness mask if enabled
            mask = None
            if use_feature_masking:
                t_name = teacher_names[idx] if idx < len(teacher_names) else 'nnunet'
                mask = compute_cancer_correctness_mask(
                    teacher_logits[idx], targets, t_name,
                    dilation_radius=getattr(args, 'mask_dilation_radius', 3)
                )
                # Handle spatial mismatch: ProViCNet features are 128x128, mask is 256x256
                feat_h, feat_w = teacher_feat.shape[-2], teacher_feat.shape[-1]
                mask_h, mask_w = mask.shape[-2], mask.shape[-1]
                if feat_h != mask_h or feat_w != mask_w:
                    mask = F.interpolate(mask.float(), size=(teacher_feat.shape[2], feat_h, feat_w),
                                        mode='nearest')
                # Debug: print mask statistics every 50 batches
                if batch_idx % 50 == 0:
                    mask_frac = mask.mean().item()
                    print(f'[MASK] train_avg batch {batch_idx} teacher {t_name}: '
                          f'mask_frac={mask_frac:.4f} ({mask_frac*100:.1f}% kept), '
                          f'mask_shape={list(mask.shape)}, feat_shape={list(teacher_feat.shape)}', flush=True)
            
            loss_feat = loss_feat + feat_kd_func(trans_student_features[idx], teacher_feat, mask=mask).mean()
        loss_feat = loss_feat / teacher_num
        loss_cls = args.ce_weight * loss_cls
        loss_feat = args.feat_weight * loss_feat
        
        loss = loss_cls + loss_feat
        loss.backward()
        optimizer.step()
        
        train_loss.update(loss.item(), inputs.size(0))
        train_loss_cls.update(loss_cls.item(), inputs.size(0))
        train_loss_kd.update(loss_kd.item(), inputs.size(0))
        train_loss_feat.update(loss_feat.item(), inputs.size(0))
        
        # Compute appropriate metrics based on dataset type
        if hasattr(args, 'dataset') and args.dataset == 'pimed':
            # Segmentation metrics: Dice and IoU (but not computed during training for speed)
            total += targets.size(0)
            metric_name = "Samples"
            metric_value = total
        else:
            # Classification metrics: Top-1 and Top-5 accuracy
            top1, top5 = correct_num(logits, targets, topk=(1, 5))
            top1_num += top1
            top5_num += top5
            total += targets.size(0)
            metric_name = "Top-1 Acc"
            metric_value = (top1_num/total*100.).item()

        if args.rank == 0 and batch_idx % 10 == 0:
            print('Epoch:{}, batch_idx:{}/{}, lr:{:.5f}, Duration:{:.2f}, CLS Loss:{:.2f},' 
                'KD Loss:{:.2f}, Feature Loss:{:.2f}, {}:{:.2f}'.format(
                epoch, batch_idx, len(train_loader), lr, time.time()-batch_start_time, 
                train_loss_cls.avg, train_loss_kd.avg, train_loss_feat.avg, 
                metric_name, metric_value))
    
    # Compute final metrics based on dataset type
    if hasattr(args, 'dataset') and args.dataset == 'pimed':
        # For segmentation, return total samples processed (no accuracy computed during training)
        final_metric = total
        metric_desc = "samples processed"
    else:
        # For classification, compute accuracies
        acc1 = top1_num / total
        acc5 = top5_num / total
        final_metric = acc1*100.
        metric_desc = "top-1 accuracy"

    if args.rank == 0:
        args.logger.info('Epoch:{}\t lr:{:.4f}\t Duration:{:.3f}'
                    '\n Train_loss:{:.5f}'
                    '\t Train_loss_cls:{:.5f}'
                    '\t Train_loss_kd:{:.5f}'
                    '\t Train_loss_feat:{:.5f}'
                    '\nTrain {}:{:.2f}'
                    .format(epoch, lr, time.time() - start_time,
                            train_loss.avg,
                            train_loss_cls.avg,
                            train_loss_kd.avg,
                            train_loss_feat.avg,
                            metric_desc, final_metric))


def train_meta(train_loader, model, criterion_list, optimizer, epoch, device, 
               args, feat_trans, teacher_models, meta_optimizer=None, scaler=None, gradient_accumulation_steps=1,
               visualizer=None, teacher_names=None, amp_dtype=torch.float16):
    """
    Training function using meta-learning for teacher weight optimization
    
    Args:
        train_loader: Data loader
        model: Student model
        criterion_list: [criterion_ce, criterion_div]
        optimizer: Model optimizer
        epoch: Current epoch
        device: Device to use
        args: Training arguments
        feat_trans: Feature transformation module
        teacher_models: List of teacher models (None for PIMED)
        meta_optimizer: Meta-learning optimizer for teacher weights
        scaler: GradScaler for mixed precision training (optional)
        gradient_accumulation_steps: Number of steps to accumulate gradients
        visualizer: ValidationVisualizer for saving predictions (optional)
        teacher_names: List of teacher names for visualization (optional)
        amp_dtype: Data type for automatic mixed precision (torch.float16 or torch.bfloat16)
    """
    
    train_loss = AverageMeter('train_loss', ':.4e')
    train_loss_cls = AverageMeter('train_loss_cls', ':.4e')
    train_loss_kd = AverageMeter('train_loss_kd', ':.4e')
    train_loss_feat = AverageMeter('train_loss_feat', ':.4e')

    # Initialize metrics based on dataset type
    if hasattr(args, 'dataset') and args.dataset == 'pimed':
        # Segmentation: only count total samples
        total = 0
        top1_num = 0  # Initialize to avoid errors, but won't be used
        top5_num = 0  # Initialize to avoid errors, but won't be used
    else:
        # Classification: track accuracy metrics
        top1_num = 0
        top5_num = 0
        total = 0

    lr = adjust_lr(optimizer, epoch, args)

    start_time = time.time()
    criterion_ce = criterion_list[0]
    criterion_div = criterion_list[1]

    model.train()
    
    # Track training cases for visualization
    train_vis_counter = 0
    max_train_vis = 10  # Visualize 10 training cases
    
    # Detect if using pre-extracted features (PIMED) or online inference (CIFAR)
    use_preextracted_features = (teacher_models is None)
    
    # Use automatic mixed precision if scaler is provided
    use_amp = (scaler is not None and scaler.is_enabled())
    
    # Initialize meta-optimizer if not provided
    if meta_optimizer is None:
        num_teachers = len(args.teacher_name_list) if use_preextracted_features else len(teacher_models)
        meta_optimizer = get_meta_teacher_optimizer(
            num_teachers=num_teachers,
            mode=args.meta_mode if hasattr(args, 'meta_mode') else 'adaptive',
            inner_steps=args.meta_inner_steps if hasattr(args, 'meta_inner_steps') else 3,
            lr=args.meta_lr if hasattr(args, 'meta_lr') else 0.05,
            temperature=args.meta_temperature if hasattr(args, 'meta_temperature') else 1.5,
            regularization=args.meta_regularization if hasattr(args, 'meta_regularization') else 0.01,
            momentum=args.meta_momentum if hasattr(args, 'meta_momentum') else 0.9
        )
    
    # Update meta-optimizer hyperparameters based on epoch
    meta_optimizer.update_hyperparams(epoch)
    
    # Initialize loss balancer if enabled
    loss_balancer = None
    if hasattr(args, 'use_loss_balancing') and args.use_loss_balancing:
        loss_balancer = LossBalancer(
            momentum=args.loss_balance_momentum if hasattr(args, 'loss_balance_momentum') else 0.9,
            eps=1e-8,
            warmup_steps=args.loss_balance_warmup if hasattr(args, 'loss_balance_warmup') else 10
        )
        if args.rank <= 0:
            print(f"===> Loss Balancing enabled: momentum={loss_balancer.momentum}, warmup={loss_balancer.warmup_steps}", flush=True)
    
    # Print gradient accumulation info
    if epoch == 0:
        effective_batch_size = args.batch_size * gradient_accumulation_steps
        print(f"===> Gradient Accumulation: accumulating {gradient_accumulation_steps} steps", flush=True)
        print(f"===> Batch size: {args.batch_size}, Effective batch size: {effective_batch_size}", flush=True)
    
    # Wrap dataloader with tqdm for progress bar
    import sys
    is_tty = sys.stdout.isatty()
    if is_tty:
        train_loader_iter = tqdm(train_loader, desc=f"Epoch {epoch}", disable=(args.rank != 0))
    else:
        train_loader_iter = train_loader
        print(f"===> Starting training epoch {epoch} with {len(train_loader)} batches...", flush=True)
    
    for batch_idx, batch_data in enumerate(train_loader_iter):
        batch_start_time = time.time()
        
        # Unpack batch based on dataset type
        if use_preextracted_features:
            # PIMED: (img, label, teacher_features, teacher_logits, case_ids, label_original, prostate_mask)
            if len(batch_data) == 7:
                inputs, targets, preextracted_teacher_features, preextracted_teacher_logits, case_ids, targets_original, prostate_mask = batch_data
            elif len(batch_data) == 6:
                inputs, targets, preextracted_teacher_features, preextracted_teacher_logits, case_ids, targets_original = batch_data
                prostate_mask = None
            else:
                # Backward compatibility
                inputs, targets, preextracted_teacher_features, preextracted_teacher_logits, case_ids = batch_data
                targets_original = targets
                prostate_mask = None
        else:
            # CIFAR: (img, label)
            inputs, targets = batch_data
            targets_original = targets
            prostate_mask = None
        
        inputs = inputs.to(device, non_blocking=True)
        # For region-based training, targets are already float multi-hot [B, C, D, H, W]
        # For class-based training, targets should be long [B, D, H, W]
        if targets.dim() == 5:  # [B, C, D, H, W] - region-based
            targets = targets.to(device, non_blocking=True).float()
        else:  # [B, D, H, W] - class-based
            targets = targets.to(device, non_blocking=True).long()
        
        # Zero gradients only at the start of accumulation cycle
        if batch_idx % gradient_accumulation_steps == 0:
            optimizer.zero_grad()
        
        teacher_logits = []
        teacher_features = []
        teacher_embeddings = []
        
        # Use autocast for model forward pass and loss computation
        with torch.cuda.amp.autocast(enabled=use_amp, dtype=amp_dtype):
            # For DataParallel, pass is_feat as a tuple in the second argument
            if isinstance(model, torch.nn.DataParallel):
                features, logits = model(inputs, (True,))
            else:
                features, logits = model(inputs, is_feat=True)
            trans_student_features = feat_trans(features[-2])
        
        if use_preextracted_features:
            # PIMED: Use pre-extracted features from dataloader
            with torch.no_grad():
                for i in range(len(preextracted_teacher_features)):
                    # preextracted_teacher_features[i] is a dict with 'layer_minus_1' (and optionally 'layer_minus_2')
                    t_feat_dict = preextracted_teacher_features[i]
                    t_logits = preextracted_teacher_logits[i].to(device, non_blocking=True)
                    
                    # Extract features and move to device
                    # layer_minus_2 is optional (may not be loaded to save memory)
                    t_feat_neg2 = t_feat_dict.get('layer_minus_2', None)
                    if t_feat_neg2 is not None:
                        t_feat_neg2 = t_feat_neg2.to(device, non_blocking=True)  # [B, C, D, H, W]
                    t_feat_neg1 = t_feat_dict['layer_minus_1'].to(device, non_blocking=True)  # [B, C, D, H, W]
                    
                    teacher_features.append(t_feat_neg2)  # May be None
                    teacher_logits.append(t_logits)
                    teacher_embeddings.append(t_feat_neg1)
        else:
            # CIFAR: Run teacher inference online
            with torch.no_grad():
                for t_model in teacher_models:
                    t_features, t_logits = t_model(inputs, is_feat=True)
                    t_feature = t_features[-1]
                    t_feature = t_feature.detach() 
                    t_logits = t_logits.detach() 
                    
                    teacher_features.append(t_features[-2])
                    teacher_logits.append(t_logits)
                    teacher_embeddings.append(t_features[-1])

        # Compute losses with autocast
        with torch.cuda.amp.autocast(enabled=use_amp, dtype=amp_dtype):
            # Compute base classification loss (with deep supervision support)
            use_deep_supervision = hasattr(args, 'use_deep_supervision') and args.use_deep_supervision
            loss_cls, logits_main = compute_deep_supervision_loss(logits, targets, criterion_ce, use_deep_supervision)
            
            # For KD and feature losses, use main output (highest resolution)
            # If deep supervision is enabled, logits_main is logits[0]
            # Otherwise, logits_main is just logits
            
            # Use RobustFeatureMSELoss3D: L2 normalize + Smooth L1 (VkD-style)
            from distiller_zoo import get_feature_mse_loss
            use_3d = (trans_student_features[0].dim() == 5) if trans_student_features else False
            feat_kd_func = get_feature_mse_loss(use_3d=use_3d, normalize=True, robust=True)
            
            # Use meta-learning to optimize teacher weights (only for features now)
            try:
                logit_weights, feature_weights = meta_optimizer.optimize_teacher_weights(
                    student_features=trans_student_features,
                    teacher_features=teacher_features,
                    student_logits=logits_main,  # Use main output for meta-optimization
                    teacher_logits=teacher_logits,
                    targets=targets,
                    criterion_div=criterion_div,
                    feat_kd_func=feat_kd_func
                )
            except Exception as e:
                # Fallback to uniform weights if optimization fails
                print(f"Meta-optimization failed: {e}, using uniform weights", flush=True)
                num_teachers = len(teacher_logits)
                logit_weights = torch.ones(num_teachers, device=device) / num_teachers
                feature_weights = torch.ones(num_teachers, device=device) / num_teachers
            
            # No logit distillation - set to 0
            loss_kd = torch.tensor(0.).to(device)
            teacher_num = len(teacher_embeddings)  # Use embeddings length since features may be None
            
            loss_feat = torch.tensor(0.).to(device)
            for idx in range(teacher_num):
                # Use teacher_features if available, otherwise use teacher_embeddings
                teacher_feat = teacher_features[idx] if teacher_features[idx] is not None else teacher_embeddings[idx]
                loss_feat = loss_feat + feature_weights[idx] * feat_kd_func(trans_student_features[idx], teacher_feat).mean()
            
            loss_cls = args.ce_weight * loss_cls
            loss_feat = args.feat_weight * loss_feat
            
            # Apply loss balancing if enabled (only cls and feat now)
            if loss_balancer is not None:
                loss_cls, loss_kd, loss_feat = loss_balancer.normalize_losses(
                    loss_cls, loss_kd, loss_feat
                )
            
            # Total loss (no logit KD)
            loss = loss_cls + loss_feat
        
        # Visualize training samples (only for PIMED dataset)
        # Note: Use targets_original for visualization (class labels, not region-based)
        if (use_preextracted_features and visualizer is not None and teacher_names is not None 
            and train_vis_counter < max_train_vis):
            try:
                teacher_logits_vis = preextracted_teacher_logits if isinstance(preextracted_teacher_logits, list) else [preextracted_teacher_logits]
                visualizer.visualize_batch(
                    inputs=inputs.detach(),
                    labels=targets_original.detach(),
                    student_logits=logits_main.detach(),  # Use main output for visualization
                    teacher_logits=[t.detach() for t in teacher_logits_vis],
                    case_ids=case_ids,
                    epoch=epoch,
                    teacher_names=teacher_names,
                    mode='train'
                )
                train_vis_counter += 1
                if args.rank <= 0 and train_vis_counter == max_train_vis:
                    print(f"  ✓ Visualized {max_train_vis} training cases for epoch {epoch}", flush=True)
            except Exception as e:
                pass  # Silently skip visualization errors
        
        # Scale loss by gradient accumulation steps
        loss = loss / gradient_accumulation_steps
        
        # Backward pass with optional mixed precision scaling
        if scaler is not None:
            scaler.scale(loss).backward()
            if (batch_idx + 1) % gradient_accumulation_steps == 0:
                # Unscale gradients before clipping
                scaler.unscale_(optimizer)
                # Clip gradients to prevent exploding gradients
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                scaler.step(optimizer)
                scaler.update()
        else:
            loss.backward()
            if (batch_idx + 1) % gradient_accumulation_steps == 0:
                # Clip gradients to prevent exploding gradients
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()
        
        # For logging, scale losses back to normal
        train_loss.update(loss.item() * gradient_accumulation_steps, inputs.size(0))
        train_loss_cls.update((loss_cls.item() / gradient_accumulation_steps), inputs.size(0))
        train_loss_kd.update((loss_kd.item() / gradient_accumulation_steps), inputs.size(0))
        train_loss_feat.update((loss_feat.item() / gradient_accumulation_steps), inputs.size(0))
        
        # Log per-batch losses to TensorBoard
        if args.rank <= 0 and hasattr(args, 'writer'):
            global_step = epoch * len(train_loader) + batch_idx
            args.writer.add_scalar('Batch/train_total_loss', loss.item() * gradient_accumulation_steps, global_step)
            args.writer.add_scalar('Batch/train_cls_loss', loss_cls.item() / gradient_accumulation_steps, global_step)
            args.writer.add_scalar('Batch/train_kd_loss', loss_kd.item() / gradient_accumulation_steps, global_step)
            args.writer.add_scalar('Batch/train_feat_loss', loss_feat.item() / gradient_accumulation_steps, global_step)
        
        # Compute appropriate metrics based on dataset type
        if hasattr(args, 'dataset') and args.dataset == 'pimed':
            # Segmentation metrics: just count samples
            total += targets.size(0)
            metric_name = "Samples"
            metric_value = total
        else:
            # Classification metrics: Top-1 and Top-5 accuracy (use main output)
            top1, top5 = correct_num(logits_main, targets, topk=(1, 5))
            top1_num += top1
            top5_num += top5
            total += targets.size(0)
            metric_name = "Top-1 Acc"
            metric_value = (top1_num/total*100.).item()

        # Update tqdm progress bar with current metrics
        if args.rank <= 0 and is_tty and hasattr(train_loader_iter, 'set_postfix'):
            logit_weights_str = ', '.join([f'{w:.2f}' for w in logit_weights])
            train_loader_iter.set_postfix({
                'CLS': f'{train_loss_cls.avg:.3f}',
                'KD': f'{train_loss_kd.avg:.3f}',
                'Feat': f'{train_loss_feat.avg:.3f}',
                'W': logit_weights_str
            })
        elif args.rank <= 0 and batch_idx % args.print_freq == 0:
            # Log teacher weights for monitoring
            logit_weights_str = ', '.join([f'{w:.3f}' for w in logit_weights])
            feature_weights_str = ', '.join([f'{w:.3f}' for w in feature_weights])
            
            # Add loss balancer stats if enabled
            balancer_info = ""
            if loss_balancer is not None:
                stats = loss_balancer.get_stats()
                balancer_info = f"\n  Running means: cls={stats['running_mean_cls']:.3f}, kd={stats['running_mean_kd']:.3f}, feat={stats['running_mean_feat']:.3f}"
            
            print(f'Epoch:{epoch}, batch_idx:{batch_idx}/{len(train_loader)}, lr:{lr:.5f}, Duration:{time.time()-batch_start_time:.2f}, '
                  f'CLS Loss:{train_loss_cls.avg:.3f}, KD Loss:{train_loss_kd.avg:.3f}, Feature Loss:{train_loss_feat.avg:.3f}, '
                  f'{metric_name}:{metric_value:.2f}\n  Logit W: [{logit_weights_str}], Feature W: [{feature_weights_str}]{balancer_info}', flush=True)
    
    # Compute final metrics based on dataset type
    if hasattr(args, 'dataset') and args.dataset == 'pimed':
        final_metric = total
        metric_desc = "samples processed"
    else:
        acc1 = top1_num / total
        acc5 = top5_num / total
        final_metric = acc1*100.
        metric_desc = "top-1 accuracy"

    if args.rank <= 0:
        log_msg = ('Epoch:{}\t lr:{:.4f}\t Duration:{:.3f}'
                    '\n Train_loss:{:.5f}'
                    '\t Train_loss_cls:{:.5f}'
                    '\t Train_loss_kd:{:.5f}'
                    '\t Train_loss_feat:{:.5f}'
                    '\nTrain {}:{:.2f}'
                    .format(epoch, lr, time.time() - start_time,
                            train_loss.avg,
                            train_loss_cls.avg,
                            train_loss_kd.avg,
                            train_loss_feat.avg,
                            metric_desc, final_metric))
        
        # Add loss balancer stats summary
        if loss_balancer is not None:
            stats = loss_balancer.get_stats()
            log_msg += (f'\nLoss Balancer: running_mean_cls={stats["running_mean_cls"]:.3f}, '
                       f'running_mean_kd={stats["running_mean_kd"]:.3f}, '
                       f'running_mean_feat={stats["running_mean_feat"]:.3f}, '
                       f'steps={stats["step_count"]}')
        
        args.logger.info(log_msg)
    
    # Return losses and metrics for logging
    if hasattr(args, 'dataset') and args.dataset == 'pimed':
        return {
            'train_loss': train_loss.avg,
            'train_loss_cls': train_loss_cls.avg,
            'train_loss_kd': train_loss_kd.avg,
            'train_loss_feat': train_loss_feat.avg,
            'train_acc': 0.0
        }
    else:
        return {
            'train_loss': train_loss.avg,
            'train_loss_cls': train_loss_cls.avg,
            'train_loss_kd': train_loss_kd.avg,
            'train_loss_feat': train_loss_feat.avg,
            'train_acc': final_metric
        }


def train(train_loader, model, criterion_list, optimizer, epoch, device, 
          args, agent, feat_trans, teacher_models, agent_optimizer, scaler=None, gradient_accumulation_steps=1,
          visualizer=None, teacher_names=None, amp_dtype=torch.float16, distill_neg1=True, distill_neg2=False):
    """
    Training function with configurable feature distillation layers.
    
    Args:
        distill_neg1: bool, whether to distill layer -1 features (default: True)
        distill_neg2: bool, whether to distill layer -2 features (default: False)
        
    When both are True, feat_trans is a tuple (feat_trans_neg2, feat_trans_neg1).
    When only one is True, feat_trans is a single TransFeat module.
    """
    
    train_loss = AverageMeter('train_loss', ':.4e')
    train_loss_cls = AverageMeter('train_loss_cls', ':.4e')
    train_loss_kd = AverageMeter('train_loss_kd', ':.4e')
    train_loss_feat = AverageMeter('train_loss_feat', ':.4e')

    top1_num = 0
    top5_num = 0
    total = 0
    
    # Determine distillation mode
    distill_both = distill_neg1 and distill_neg2

    lr = adjust_lr(optimizer, epoch, args)

    start_time = time.time()
    criterion_ce = criterion_list[0]
    criterion_div = criterion_list[1]

    model.train()
    agent.eval()
    
    # Initialize episode buffer for episode-based rewards, or lists for per-step rewards
    if hasattr(args, 'use_episode_reward') and args.use_episode_reward:
        store_logits_actions = hasattr(args, 'logits_actions') and args.logits_actions
        store_neg2_actions = distill_neg2  # Store neg2 actions if distilling neg2
        episode_buffer = EpisodeBuffer(store_logits_actions=store_logits_actions, store_neg2_actions=store_neg2_actions)
    else:
        agent_states = []
        feature_agent_actions = []
        agent_rewards = []
    
    # Track training cases for visualization
    train_vis_counter = 0
    max_train_vis = 10  # Visualize 10 training cases per epoch
    
    # Detect if using pre-extracted features (PIMED) or online inference (CIFAR)
    use_preextracted_features = (teacher_models is None)
    
    # Use automatic mixed precision if scaler is provided
    use_amp = (scaler is not None and scaler.is_enabled())
    
    # Initialize loss balancer if enabled
    loss_balancer = None
    if hasattr(args, 'use_loss_balancing') and args.use_loss_balancing:
        loss_balancer = LossBalancer(
            momentum=args.loss_balance_momentum if hasattr(args, 'loss_balance_momentum') else 0.9,
            eps=1e-8,
            warmup_steps=args.loss_balance_warmup if hasattr(args, 'loss_balance_warmup') else 10
        )
        if args.rank <= 0 and epoch == 0:
            print(f"===> Loss Balancing enabled: momentum={loss_balancer.momentum}, warmup={loss_balancer.warmup_steps}", flush=True)
    
    # Print gradient accumulation info
    if epoch == 0:
        effective_batch_size = args.batch_size * gradient_accumulation_steps
        print(f"===> Gradient Accumulation: accumulating {gradient_accumulation_steps} steps", flush=True)
        print(f"===> Batch size: {args.batch_size}, Effective batch size: {effective_batch_size}", flush=True)
    
    # Wrap dataloader with tqdm for progress bar
    import sys
    is_tty = sys.stdout.isatty()
    if is_tty:
        # Only use tqdm in interactive mode
        train_loader_iter = tqdm(train_loader, desc=f"Epoch {epoch}", disable=(args.rank != 0))
    else:
        # In non-interactive mode (logging to file), just use plain iterator
        train_loader_iter = train_loader
        batches_info = f"{args.batches_per_epoch}" if hasattr(args, 'batches_per_epoch') and args.batches_per_epoch else f"{len(train_loader)}"
        print(f"===> Starting training epoch {epoch} with {batches_info} batches...", flush=True)
    
    for batch_idx, batch_data in enumerate(train_loader_iter):
        # Check if we've reached the batch limit for this epoch
        if hasattr(args, 'batches_per_epoch') and args.batches_per_epoch is not None:
            if batch_idx >= args.batches_per_epoch:
                if args.rank == 0:
                    print(f"===> Reached batch limit ({args.batches_per_epoch}), ending epoch early", flush=True)
                break
        
        batch_start_time = time.time()
        
        # Unpack batch based on dataset type
        if use_preextracted_features:
            # PIMED: (img, label, teacher_features, teacher_logits, case_ids, label_original, prostate_mask)
            if len(batch_data) == 7:
                inputs, targets, preextracted_teacher_features, preextracted_teacher_logits, case_ids, targets_original, prostate_mask = batch_data
            elif len(batch_data) == 6:
                inputs, targets, preextracted_teacher_features, preextracted_teacher_logits, case_ids, targets_original = batch_data
                prostate_mask = None
            else:
                # Backward compatibility
                inputs, targets, preextracted_teacher_features, preextracted_teacher_logits, case_ids = batch_data
                targets_original = targets
                prostate_mask = None
        else:
            # CIFAR: (img, label)
            inputs, targets = batch_data
            targets_original = targets
            prostate_mask = None
        
        inputs = inputs.to(device, non_blocking=True)
        # For region-based training, targets are already float multi-hot [B, C, D, H, W]
        # For class-based training, targets should be long [B, D, H, W]
        if targets.dim() == 5:  # [B, C, D, H, W] - region-based
            targets = targets.to(device, non_blocking=True).float()
        else:  # [B, D, H, W] - class-based
            targets = targets.to(device, non_blocking=True).long()
        
        # Zero gradients only at the start of accumulation cycle
        if batch_idx % gradient_accumulation_steps == 0:
            optimizer.zero_grad()
        
        teacher_logits = []
        teacher_features = []
        teacher_embeddings = []
        
        # Use autocast for model forward pass and loss computation
        with torch.cuda.amp.autocast(enabled=use_amp, dtype=amp_dtype):
            # For DataParallel, pass is_feat as a tuple in the second argument
            if isinstance(model, torch.nn.DataParallel):
                features, logits = model(inputs, (True,))
            else:
                features, logits = model(inputs, is_feat=True)
            
            # Transform student features based on distillation mode
            if distill_both:
                # Both layers: feat_trans is a tuple: (feat_trans_neg2, feat_trans_neg1)
                feat_trans_neg2, feat_trans_neg1 = feat_trans
                trans_student_features_neg2 = feat_trans_neg2(features[-2])
                trans_student_features_neg1 = feat_trans_neg1(features[-1])
            elif distill_neg2:
                # Only neg2: single TransFeat for layer -2
                trans_student_features_neg2 = feat_trans(features[-2])
                trans_student_features_neg1 = None
            else:
                # Only neg1 (default): single TransFeat for layer -1
                trans_student_features_neg1 = feat_trans(features[-1])
                trans_student_features_neg2 = None
        
        if use_preextracted_features:
            # PIMED: Use pre-extracted features from dataloader
            with torch.no_grad():
                all_teacher_info = []  # Not used, but kept for consistency with CIFAR branch
                for i in range(len(preextracted_teacher_features)):
                    # preextracted_teacher_features[i] is a dict with 'layer_minus_1' (and optionally 'layer_minus_2')
                    t_feat_dict = preextracted_teacher_features[i]
                    t_logits = preextracted_teacher_logits[i].to(device, non_blocking=True)
                    
                    # Extract features and move to device
                    # layer_minus_2 is optional (may not be loaded to save memory)
                    t_feat_neg2 = t_feat_dict.get('layer_minus_2', None)
                    if t_feat_neg2 is not None:
                        t_feat_neg2 = t_feat_neg2.to(device, non_blocking=True)  # [B, C, D, H, W]
                    t_feat_neg1 = t_feat_dict['layer_minus_1'].to(device, non_blocking=True)  # [B, C, D, H, W]
                    
                    teacher_features.append(t_feat_neg2)  # May be None
                    teacher_logits.append(t_logits)
                    teacher_embeddings.append(t_feat_neg1)
                    
                    # Note: teacher_info is not used for PIMED (pre-extracted features)
                    # Skip building it to avoid dimension mismatch with 3D data
        else:
            # CIFAR: Run teacher inference online
            with torch.no_grad():
                all_teacher_info = []
                for t_model in teacher_models:
                    t_features, t_logits = t_model(inputs, is_feat=True)
                    t_feature = t_features[-1]
                    t_feature = t_feature.detach() 
                    t_logits = t_logits.detach() 
                    
                    teacher_features.append(t_features[-2])
                    teacher_logits.append(t_logits)
                    teacher_embeddings.append(t_features[-1])
                    teacher_info = []
                    teacher_info.append(t_feature)
                    teacher_info.append(t_logits)
                    teacher_info.append(F.cross_entropy(t_logits, targets, reduction='none').unsqueeze(-1)) # 128*1
                    teacher_info = torch.cat(teacher_info, dim=1) # teacher logtis , teacher feature , CE_loss , student_teacher_gap
                    all_teacher_info.append(teacher_info)

        # New policy: get_agent_state returns (neg1_features, neg2_features, teacher_logits, teacher_scalars, teacher_dice_means)
        # Call appropriate function based on distillation mode
        use_disagreement = hasattr(args, 'use_disagreement_reward') and args.use_disagreement_reward
        
        if distill_both:
            # Both layers: use get_agent_state_neg1_neg2
            agent_state_full = get_agent_state_neg1_neg2(
                trans_student_features_neg1, trans_student_features_neg2,
                teacher_embeddings, teacher_features,
                teacher_logits, targets, teacher_names, use_disagreement=use_disagreement
            )
        elif distill_neg2:
            # Only neg2: use get_agent_state_neg2
            agent_state_full = get_agent_state_neg2(
                trans_student_features_neg2, teacher_features,
                teacher_logits, targets, teacher_names, use_disagreement=use_disagreement
            )
        else:
            # Only neg1 (default): use get_agent_state_neg1
            agent_state_full = get_agent_state_neg1(
                trans_student_features_neg1, teacher_embeddings,
                teacher_logits, targets, teacher_names, use_disagreement=use_disagreement
            )
        
        # Unpack agent_state_full: (neg1_features, neg2_features, logits, scalars, dice_means)
        agent_neg1_features, agent_neg2_features, _, agent_scalars, teacher_dice_means = agent_state_full
        agent_state = (agent_neg1_features, agent_neg2_features, teacher_logits, agent_scalars)
        
        with torch.no_grad():
            agent_output = agent(agent_state)
            # Handle conditional output based on agent configuration
            # Output format depends on distill_neg1, distill_neg2, and enable_logits_actions
            # PolicyTrans returns results in order: [neg1 if enabled], [neg2 if enabled], [logits if enabled]
            enable_logits_actions = hasattr(args, 'logits_actions') and args.logits_actions
            
            if isinstance(agent_output, tuple):
                output_list = list(agent_output)
            else:
                output_list = [agent_output]
            
            # Parse output based on enabled actions
            idx = 0
            if distill_neg1:
                feature_actions = output_list[idx]
                idx += 1
            else:
                feature_actions = None
            
            if distill_neg2:
                neg2_feature_actions = output_list[idx]
                idx += 1
            else:
                neg2_feature_actions = None
            
            if enable_logits_actions and idx < len(output_list):
                logits_actions = output_list[idx]
            else:
                logits_actions = None
        
        # Check if agent produced NaN or Inf - if so, skip this entire batch
        # Check the primary action tensor (whichever is enabled)
        primary_actions = feature_actions if feature_actions is not None else neg2_feature_actions
        if primary_actions is not None and torch.isnan(primary_actions).any():
            print(f"WARNING: NaN detected in agent output at epoch {epoch}, batch {batch_idx}. Skipping this batch.", flush=True)
            continue  # Skip to next batch
        
        if primary_actions is not None and torch.isinf(primary_actions).any():
            print(f"WARNING: Inf detected in agent output at epoch {epoch}, batch {batch_idx}. Skipping this batch.", flush=True)
            continue  # Skip to next batch
        
        # For non-episode reward mode: save agent state for later training
        if not (hasattr(args, 'use_episode_reward') and args.use_episode_reward):
            # Move agent_state to CPU and detach to save GPU memory
            # agent_state is a tuple of (neg1_features, neg2_features, teacher_logits, teacher_scalars_list)
            # neg1_features and neg2_features can be None if not enabled
            def to_cpu(x):
                if x is None:
                    return None
                elif isinstance(x, list):
                    return [t.detach().cpu() for t in x]
                else:
                    return x.detach().cpu()
            agent_state_cpu = tuple([to_cpu(item) for item in agent_state])
            agent_states.append(agent_state_cpu)
        
        # Gradual warmup: blend uniform weights with agent weights over warmup_epochs
        # This ensures smooth transition and consistent loss scale
        warmup_epochs = getattr(args, 'agent_warmup_epochs', 5)  # Default 5 epochs warmup
        if epoch < warmup_epochs:
            # alpha goes from 0 to 1 over warmup period
            # At epoch 0: alpha=0 (100% uniform), at epoch warmup_epochs-1: alpha close to 1
            alpha = epoch / warmup_epochs
            if feature_actions is not None:
                uniform_weights = torch.ones_like(feature_actions)
                feature_actions = (1 - alpha) * uniform_weights + alpha * feature_actions
            if neg2_feature_actions is not None:
                uniform_neg2 = torch.ones_like(neg2_feature_actions)
                neg2_feature_actions = (1 - alpha) * uniform_neg2 + alpha * neg2_feature_actions
            if logits_actions is not None:
                uniform_logits = torch.ones_like(logits_actions)
                logits_actions = (1 - alpha) * uniform_logits + alpha * logits_actions
        if feature_actions is not None:
            feature_actions = feature_actions.detach()  # batch_size x teacher_number
        if neg2_feature_actions is not None:
            neg2_feature_actions = neg2_feature_actions.detach()
        if logits_actions is not None:
            logits_actions = logits_actions.detach()

        # For non-episode reward mode: save actions
        if not (hasattr(args, 'use_episode_reward') and args.use_episode_reward):
            # Move actions to CPU to save GPU memory
            # Use whichever actions are available
            if feature_actions is not None:
                feature_agent_actions.append(feature_actions.cpu())
            elif neg2_feature_actions is not None:
                feature_agent_actions.append(neg2_feature_actions.cpu())

        if args.rank <= 0 and batch_idx % 10 == 0:
            if feature_actions is not None:
                args.logger.info('feature_actions (neg1):{}'.format(str(feature_actions[0])))
            if neg2_feature_actions is not None:
                args.logger.info('feature_actions (neg2):{}'.format(str(neg2_feature_actions[0])))
            # Log teacher Dice scores (mean across batch for each teacher)
            teacher_dice_str = ', '.join([f'{d.mean().item():.4f}' for d in teacher_dice_means])
            args.logger.info('teacher_dices:[{}]'.format(teacher_dice_str))
        
        # Compute losses with autocast
        with torch.cuda.amp.autocast(enabled=use_amp, dtype=amp_dtype):
            # Compute base classification loss (with deep supervision support)
            use_deep_supervision = hasattr(args, 'use_deep_supervision') and args.use_deep_supervision
            loss_cls, logits_main = compute_deep_supervision_loss(logits, targets, criterion_ce, use_deep_supervision)
            
            # Print per-channel statistics after sigmoid (every 100 batches)
            if args.rank <= 0 and batch_idx % 50 == 0 and hasattr(args, 'use_region_based_training') and args.use_region_based_training:
                predicted_probs = torch.sigmoid(logits_main)  # [B, 3, D, H, W]
                for c in range(predicted_probs.size(1)):
                    # Compute mean/std across spatial dimensions (D, H, W) for each sample in batch
                    channel_probs = predicted_probs[:, c]  # [B, D, H, W]
                    target_channel = targets[:, c] if targets.dim() == 5 else None
                    # Mean across all voxels in the batch
                    global_mean = channel_probs.mean().item()
                    global_std = channel_probs.std().item()
                    target_sum = target_channel.sum().item() if target_channel is not None else 0
                    print(f"[TRAIN Batch {batch_idx}] Channel {c}: pred_mean={global_mean:.4f}, pred_std={global_std:.4f}, target_sum={target_sum:.0f}", flush=True)
            
            # No logit distillation - set to 0
            loss_kd = torch.tensor(0., device=logits_main.device, dtype=logits_main.dtype)
            # teacher_num = len(teacher_logits)
            teacher_num = len(teacher_features)
            if hasattr(args, 'logits_actions') and args.logits_actions and logits_actions is not None:
                # Check if region-based training (BCE) vs class-based (CE+KL)
                is_region_based = targets.dim() == 5  # [B, C, D, H, W]
                
                # Convert logits_actions to match the dtype of logits_main for mixed precision
                logits_actions_typed = logits_actions.to(dtype=logits_main.dtype)
                
                for idx in range(teacher_num):
                    teacher_n = teacher_names[idx]
                    if is_region_based:
                        # Region-based: use L1 loss on probabilities (after sigmoid) for logits distillation
                        # This is more stable than MSE on raw logits since probabilities are bounded [0,1]
                        student_probs = torch.sigmoid(logits_main)
                        
                        if 'provicnet' in teacher_n.lower() and teacher_logits[idx].shape[1] == 4:
                            # ProViCNet: 4-channel softmax -> convert to region-based format
                            # Channels: [background, prostate, ciPCA, csPCA]
                            teacher_probs = normalize_provicnet_logits(
                                teacher_logits[idx].detach(), 
                                target_channels=student_probs.shape[1]
                            )
                        else:
                            teacher_probs = torch.sigmoid(teacher_logits[idx].detach())
                        # Convert to matching dtype
                        teacher_probs = teacher_probs.to(dtype=student_probs.dtype)
                        kd_loss = F.l1_loss(student_probs, teacher_probs, reduction='none')
                        # Average over all dimensions except batch: [B, C, D, H, W] -> [B]
                        kd_loss = kd_loss.view(kd_loss.size(0), -1).mean(dim=1)
                        loss_kd = loss_kd + (logits_actions_typed[:, idx] * kd_loss).mean()
                    else:
                        # Class-based: use KL divergence
                        # For 3D segmentation, reshape [B, C, D, H, W] to [B*D*H*W, C] for KL divergence
                        # Use logits_main (not logits) which is the main output from deep supervision
                        if logits_main.dim() == 5:
                            B, C, D, H, W = logits_main.shape
                            logits_flat = logits_main.permute(0, 2, 3, 4, 1).reshape(-1, C)  # [B*D*H*W, C]
                            teacher_logits_flat = teacher_logits[idx].detach().permute(0, 2, 3, 4, 1).reshape(-1, C)  # [B*D*H*W, C]
                            kd_loss = criterion_div(logits_flat, teacher_logits_flat, unreduce=True)  # [B*D*H*W]
                            kd_loss = kd_loss.view(B, D, H, W).mean(dim=(1, 2, 3))  # [B]
                            loss_kd = loss_kd + (logits_actions_typed[:, idx] * kd_loss).mean()
                        else:
                            # For 2D or classification, compute KL normally
                            loss_kd = loss_kd + (logits_actions_typed[:, idx] * criterion_div(logits_main, teacher_logits[idx].detach(), unreduce=True)).mean()

            loss_feat = torch.tensor(0., device=logits_main.device, dtype=logits_main.dtype)
            loss_feat_neg2 = torch.tensor(0., device=logits_main.device, dtype=logits_main.dtype)
            
            # Use MaskedFeatureMSELoss3D with cancer-correctness masking
            from distiller_zoo import MaskedFeatureMSELoss3D
            feat_kd_func = MaskedFeatureMSELoss3D(normalize=True)
            
            use_feature_masking = (getattr(args, 'use_feature_masking', False)
                                   and teacher_names is not None
                                   and use_preextracted_features
                                   and targets.dim() == 5)

            for idx in range(teacher_num):
                # Compute cancer-correctness mask once per teacher (shared by NEG1 and NEG2)
                mask_neg1 = None
                mask_neg2 = None
                if use_feature_masking:
                    t_name = teacher_names[idx] if idx < len(teacher_names) else 'nnunet'
                    base_mask = compute_cancer_correctness_mask(
                        teacher_logits[idx], targets, t_name,
                        dilation_radius=getattr(args, 'mask_dilation_radius', 3)
                    )  # [B, 1, D, H, W] at logit resolution (256x256)
                
                # === NEG1 feature distillation (using teacher_embeddings / teacher features neg1) ===
                if distill_neg1 and trans_student_features_neg1 is not None:
                    teacher_feat_neg1 = teacher_embeddings[idx]
                    # Debug: print shapes and values
                    if batch_idx == 0:
                        print(f'[DEBUG] teacher {teacher_names[idx]} NEG1: trans_student shape={trans_student_features_neg1[idx].shape}, teacher shape={teacher_feat_neg1.shape}', flush=True)
                        print(f'[DEBUG] teacher {teacher_names[idx]} NEG1: trans_student mean={trans_student_features_neg1[idx].mean().item():.6f}, std={trans_student_features_neg1[idx].std().item():.6f}', flush=True)
                        print(f'[DEBUG] teacher {teacher_names[idx]} NEG1: teacher mean={teacher_feat_neg1.mean().item():.6f}, std={teacher_feat_neg1.std().item():.6f}', flush=True)
                    
                    # Adapt mask to feature spatial resolution if needed
                    if use_feature_masking:
                        feat_shape = teacher_feat_neg1.shape  # [B, C, D, H, W]
                        mask_neg1 = F.interpolate(base_mask.float(),
                                                  size=feat_shape[2:],  # (D, H, W)
                                                  mode='nearest') if base_mask.shape[2:] != feat_shape[2:] else base_mask
                        # Debug: print mask statistics every 50 batches
                        if batch_idx % 50 == 0:
                            mask_frac = mask_neg1.mean().item()
                            t_name = teacher_names[idx]
                            print(f'[MASK] train_meta batch {batch_idx} teacher {t_name} NEG1: '
                                  f'mask_frac={mask_frac:.4f} ({mask_frac*100:.1f}% kept), '
                                  f'base_mask_shape={list(base_mask.shape)}, feat_shape={list(feat_shape)}', flush=True)
                    
                    feat_kd_loss = feat_kd_func(trans_student_features_neg1[idx], teacher_feat_neg1, mask=mask_neg1)
                    loss_feat = loss_feat + (feature_actions[:, idx] * feat_kd_loss).mean()
                
                # === NEG2 feature distillation (using teacher_features / teacher features neg2) ===
                if distill_neg2 and trans_student_features_neg2 is not None and teacher_features[idx] is not None:
                    teacher_feat_neg2 = teacher_features[idx]
                    if batch_idx == 0:
                        print(f'[DEBUG] teacher {teacher_names[idx]} NEG2: trans_student shape={trans_student_features_neg2[idx].shape}, teacher shape={teacher_feat_neg2.shape}', flush=True)
                        print(f'[DEBUG] teacher {teacher_names[idx]} NEG2: trans_student mean={trans_student_features_neg2[idx].mean().item():.6f}, std={trans_student_features_neg2[idx].std().item():.6f}', flush=True)
                        print(f'[DEBUG] teacher {teacher_names[idx]} NEG2: teacher mean={teacher_feat_neg2.mean().item():.6f}, std={teacher_feat_neg2.std().item():.6f}', flush=True)
                    
                    # Adapt mask to neg2 feature spatial resolution if needed
                    if use_feature_masking:
                        feat_shape = teacher_feat_neg2.shape
                        mask_neg2 = F.interpolate(base_mask.float(),
                                                  size=feat_shape[2:],
                                                  mode='nearest') if base_mask.shape[2:] != feat_shape[2:] else base_mask
                        # Debug: print mask statistics every 50 batches
                        if batch_idx % 50 == 0:
                            t_name = teacher_names[idx]
                            mask_frac = mask_neg2.mean().item()
                            print(f'[MASK] train_meta batch {batch_idx} teacher {t_name} NEG2: '
                                  f'mask_frac={mask_frac:.4f} ({mask_frac*100:.1f}% kept), '
                                  f'feat_shape={list(feat_shape)}', flush=True)
                    
                    feat_kd_loss_neg2 = feat_kd_func(trans_student_features_neg2[idx], teacher_feat_neg2, mask=mask_neg2)
                    loss_feat_neg2 = loss_feat_neg2 + (neg2_feature_actions[:, idx] * feat_kd_loss_neg2).mean()

            loss_cls = args.ce_weight * loss_cls
            # Combine neg1 and neg2 feature losses with configurable neg2 weight
            neg2_weight = getattr(args, 'neg2_weight', 1.0)
            total_loss_feat = loss_feat + neg2_weight * loss_feat_neg2
            loss_feat = args.feat_weight * total_loss_feat
            loss_kd = args.kd_weight * loss_kd
            
            # Apply loss balancing if enabled
            if loss_balancer is not None:
                loss_cls, loss_kd, loss_feat = loss_balancer.normalize_losses(
                    loss_cls, loss_kd, loss_feat
                )
            
            # Total loss
            loss = loss_cls + loss_kd + loss_feat
        
        # Visualize training samples (only for PIMED dataset, use original labels)
        if (use_preextracted_features and visualizer is not None and teacher_names is not None 
            and train_vis_counter < max_train_vis):
            try:
                # Prepare teacher logits for visualization
                # preextracted_teacher_logits is a list of [B, C, D, H, W] tensors
                teacher_logits_vis = preextracted_teacher_logits if isinstance(preextracted_teacher_logits, list) else [preextracted_teacher_logits]
                
                # Call visualizer with mode='train' to distinguish from validation
                # IMPORTANT: Detach logits before visualization to avoid gradient issues
                # Use logits_main (main output) for visualization
                # Use targets_original for visualization (class labels, not region-based)
                visualizer.visualize_batch(
                    inputs=inputs.detach(),
                    labels=targets_original.detach(),
                    student_logits=logits_main.detach(),
                    teacher_logits=[t.detach() for t in teacher_logits_vis],
                    case_ids=case_ids,
                    epoch=epoch,
                    teacher_names=teacher_names,
                    mode='train'  # Add mode to distinguish train/val in saved filenames
                )
                train_vis_counter += 1
                if args.rank <= 0 and train_vis_counter == max_train_vis:
                    print(f"  ✓ Visualized {max_train_vis} training cases for epoch {epoch}", flush=True)
            except Exception as e:
                pass  # Silently skip visualization errors
        
        # Scale loss by gradient accumulation steps
        loss = loss / gradient_accumulation_steps
        
        # Backward pass with optional FP16 scaling
        if scaler is not None:
            scaler.scale(loss).backward()
            # Only update weights after accumulating gradients
            if (batch_idx + 1) % gradient_accumulation_steps == 0:
                scaler.step(optimizer)
                scaler.update()
        else:
            loss.backward()
            # Only update weights after accumulating gradients
            if (batch_idx + 1) % gradient_accumulation_steps == 0:
                optimizer.step()

        
        # Compute sample-wise losses for reward (need to reduce to [B] for agent)
        # Use logits_main (main output) for reward computation
        
        # Check if region-based training
        is_region_based = targets.dim() == 5  # [B, C, D, H, W]
        
        if is_region_based:
            # Region-based: use weighted BCE (same as training loss)
            # Detect number of channels to use appropriate weights
            num_channels = targets.size(1)  # C from [B, C, D, H, W]
            
            sample_ce_loss_raw = F.binary_cross_entropy_with_logits(logits_main, targets.float(), reduction='none')
            # [B, C, D, H, W]
            
            # Apply pos_weight: multiply foreground pixels by pos_w, background by 1
            if num_channels == 2:
                # Binary mode: [prostate, cancer]
                pos_w = torch.tensor([1.0, 10.0], dtype=sample_ce_loss_raw.dtype, device=sample_ce_loss_raw.device)
                channel_w = torch.tensor([1.0, 5.0], dtype=sample_ce_loss_raw.dtype, device=sample_ce_loss_raw.device)
            else:
                # 3-class mode: [prostate, PCa, csPCa]
                pos_w = torch.tensor([1.0, 5.0, 10.0], dtype=sample_ce_loss_raw.dtype, device=sample_ce_loss_raw.device)
                channel_w = torch.tensor([1.0, 3.0, 5.0], dtype=sample_ce_loss_raw.dtype, device=sample_ce_loss_raw.device)
            
            pos_w = pos_w.view(1, -1, 1, 1, 1)  # [1, C, 1, 1, 1] for broadcasting
            weight_mask = pos_w * targets.float() + (1 - targets.float())  # foreground gets pos_w, background gets 1
            sample_ce_loss_raw = sample_ce_loss_raw * weight_mask
            
            # Apply channel_weights
            channel_w = channel_w.view(1, -1, 1, 1, 1)  # [1, C, 1, 1, 1]
            sample_ce_loss_raw = sample_ce_loss_raw * channel_w
            
            # Average over channels and spatial dims: [B, C, D, H, W] -> [B]
            sample_ce_loss = sample_ce_loss_raw.view(sample_ce_loss_raw.size(0), -1).mean(dim=1)
        else:
            # Class-based: use CE with class weights
            # Apply class weights: [1.0, 10.0, 30.0] for [background, peripheral zone, transition zone]
            ce_class_weights = torch.tensor([1.0, 10.0, 30.0], dtype=torch.float32, device=targets.device)
            sample_ce_loss_raw = F.cross_entropy(logits_main, targets, weight=ce_class_weights, reduction='none')
            # For 3D segmentation: [B, D, H, W] -> [B], for 2D: [B, H, W] -> [B]
            if sample_ce_loss_raw.dim() > 1:
                sample_ce_loss = sample_ce_loss_raw.view(sample_ce_loss_raw.size(0), -1).mean(dim=1)
            else:
                sample_ce_loss = sample_ce_loss_raw
            
        sample_kd_loss = torch.zeros(inputs.size(0)).cuda(args.gpu)
        sample_feat_loss = torch.zeros(inputs.size(0)).cuda(args.gpu)
        
        for idx in range(teacher_num):
            # Logit KD for reward computation (if enabled)
            if hasattr(args, 'logits_actions') and args.logits_actions and logits_actions is not None:
                if is_region_based:
                    # Region-based: use L1 loss on probabilities
                    student_probs = torch.sigmoid(logits_main)
                    
                    # Handle provicnet's 4-channel softmax output
                    teacher_n = teacher_names[idx] if idx < len(teacher_names) else ''
                    if 'provicnet' in teacher_n.lower() and teacher_logits[idx].shape[1] == 4:
                        teacher_probs = normalize_provicnet_logits(
                            teacher_logits[idx].detach(),
                            target_channels=student_probs.shape[1]
                        )
                    else:
                        teacher_probs = torch.sigmoid(teacher_logits[idx].detach())
                    
                    kd_loss_raw = F.l1_loss(student_probs, teacher_probs, reduction='none')
                    # Average over all dimensions except batch: [B, C, D, H, W] -> [B]
                    kd_loss_per_sample = kd_loss_raw.view(kd_loss_raw.size(0), -1).mean(dim=1)
                    sample_kd_loss = sample_kd_loss + (logits_actions[:, idx] * kd_loss_per_sample)
                else:
                    # Class-based: use KL divergence
                    if logits_main.dim() == 5:
                        # 3D segmentation
                        B, C, D, H, W = logits_main.shape
                        logits_flat = logits_main.permute(0, 2, 3, 4, 1).reshape(-1, C)
                        teacher_logits_flat = teacher_logits[idx].detach().permute(0, 2, 3, 4, 1).reshape(-1, C)
                        kd_loss_raw = criterion_div(logits_flat, teacher_logits_flat, unreduce=True)
                        kd_loss_per_sample = kd_loss_raw.view(B, D, H, W).mean(dim=(1, 2, 3))  # [B]
                        sample_kd_loss = sample_kd_loss + (logits_actions[:, idx] * kd_loss_per_sample)
                    else:
                        # 2D or classification
                        kd_loss_raw = criterion_div(logits_main, teacher_logits[idx].detach(), unreduce=True)
                        sample_kd_loss = sample_kd_loss + (logits_actions[:, idx] * kd_loss_raw)
            
            # Feature loss for reward: reduce to [B]
            # Combine neg1 and neg2 feature losses based on what's enabled
            feat_loss_per_sample = torch.zeros(inputs.size(0)).cuda(args.gpu)
            
            if distill_neg1 and trans_student_features_neg1 is not None and feature_actions is not None:
                teacher_feat_neg1 = teacher_embeddings[idx]
                feat_loss_raw = feat_kd_func(trans_student_features_neg1[idx], teacher_feat_neg1)
                if feat_loss_raw.dim() > 1:
                    feat_loss_sample = feat_loss_raw.view(feat_loss_raw.size(0), -1).mean(dim=1)
                else:
                    feat_loss_sample = feat_loss_raw
                feat_loss_per_sample = feat_loss_per_sample + (feature_actions[:, idx] * feat_loss_sample)
            
            if distill_neg2 and trans_student_features_neg2 is not None and neg2_feature_actions is not None:
                if teacher_features[idx] is not None:
                    teacher_feat_neg2 = teacher_features[idx]
                    feat_loss_raw_neg2 = feat_kd_func(trans_student_features_neg2[idx], teacher_feat_neg2)
                    if feat_loss_raw_neg2.dim() > 1:
                        feat_loss_sample_neg2 = feat_loss_raw_neg2.view(feat_loss_raw_neg2.size(0), -1).mean(dim=1)
                    else:
                        feat_loss_sample_neg2 = feat_loss_raw_neg2
                    feat_loss_per_sample = feat_loss_per_sample + (neg2_feature_actions[:, idx] * feat_loss_sample_neg2)
            
            sample_feat_loss = sample_feat_loss + feat_loss_per_sample
        
        # Compute ensemble-based reward: student Dice - ensemble Dice
        # This directly optimizes for beating the ensemble average
        with torch.no_grad():
            # Compute ensemble prediction (average of teacher probabilities)
            # Need to handle provicnet's different format (4-ch softmax vs region-based sigmoid)
            target_channels = targets.shape[1]  # 2 for binary, 3 for 3-class
            
            normalized_teacher_probs = []
            for idx, t in enumerate(teacher_logits):
                teacher_n = teacher_names[idx].lower() if idx < len(teacher_names) else ''
                if 'provicnet' in teacher_n and t.shape[1] == 4:
                    # ProViCNet: normalize to region-based format
                    probs = normalize_provicnet_logits(t.detach(), target_channels=target_channels)
                else:
                    # Region-based teachers: apply sigmoid
                    probs = torch.sigmoid(t.detach())
                normalized_teacher_probs.append(probs)
            
            # Average teacher probabilities to get ensemble
            ensemble_probs = torch.mean(torch.stack(normalized_teacher_probs), dim=0)
            
            if is_region_based:
                # Region-based: compute per-sample Dice for each channel
                # Student predictions
                student_probs = torch.sigmoid(logits_main)
                student_pred = (student_probs > 0.5).float()
                # Ensemble predictions (already in probability space)
                ensemble_pred = (ensemble_probs > 0.5).float()
                
                # Compute Dice per sample, averaged across foreground channels
                # Determine channels based on number of channels:
                # Binary mode (2 channels): use [1] (cancer only)
                # 3-class mode (3 channels): use [1, 2] (PCa and csPCa)
                num_channels = targets.shape[1]
                channels_to_include = [1] if num_channels == 2 else [1, 2]
                
                student_dice = compute_dice_per_sample(
                    student_pred, targets, 
                    channels_to_include=channels_to_include,
                    keep_batch_dim=True
                )  # [B]
                
                ensemble_dice = compute_dice_per_sample(
                    ensemble_pred, targets,
                    channels_to_include=channels_to_include,
                    keep_batch_dim=True
                )  # [B]
            else:
                # Class-based: use accuracy or other metric
                # For simplicity, use negative CE loss as proxy for "goodness"
                # Note: ensemble_probs is already computed above (averaged teacher probs)
                student_pred = torch.argmax(logits_main, dim=1)
                ensemble_pred = torch.argmax(ensemble_probs, dim=1)
                student_dice = (student_pred == targets).float().view(targets.size(0), -1).mean(dim=1)
                ensemble_dice = (ensemble_pred == targets).float().view(targets.size(0), -1).mean(dim=1)
            
            # Reward: positive if student beats ensemble, negative otherwise
            # Scale by 100 to make reward magnitudes reasonable
            dice_based_reward = (student_dice - ensemble_dice) * 100.0
        
        # Combine dice-based reward with loss-based components
        # IMPORTANT: Use only CE loss for reward, NOT distillation losses (feat/kd).
        # Including feat_loss in reward would reward the agent for matching teachers better,
        # rather than for improving the student's actual segmentation quality.
        # This decouples the reward from the distillation objective so the agent can
        # discover which teacher actions actually help segmentation vs introduce noise.
        loss_based_reward = -sample_ce_loss
        
        # Blend rewards: prioritize dice improvement over loss minimization
        # Alpha controls the blend: 1.0 = pure dice-based, 0.0 = pure loss-based
        reward_alpha = getattr(args, 'reward_alpha', 0.7)  # Default: 70% dice-based, 30% loss-based
        reward = reward_alpha * dice_based_reward + (1 - reward_alpha) * loss_based_reward
        rewards_mean = reward.mean() 
        rewards_std = reward.std() 
        
        # Episode-based rewards: store state and actions for later policy gradient
        if hasattr(args, 'use_episode_reward') and args.use_episode_reward:
            # Add to episode buffer: store the blended dice+loss reward per step
            # reward = alpha * (student_dice - ensemble_dice) * 100 + (1-alpha) * (-CE_loss)
            # This gives the agent a per-step signal that reflects both segmentation
            # quality improvement over the ensemble AND classification loss reduction.
            step_reward = reward.mean().item()
            episode_buffer.add(step_reward, agent_state, feature_actions, neg2_feature_actions, logits_actions)
            
            # Periodic agent update to prevent memory explosion
            # Default: update every 1000 batches (configurable via args.agent_update_interval)
            agent_update_interval = getattr(args, 'agent_update_interval', 1000)
            if len(episode_buffer) >= agent_update_interval:
                # Use average blended reward as terminal reward for intermediate updates
                avg_reward = sum(episode_buffer.batch_rewards) / len(episode_buffer.batch_rewards)
                
                if args.rank <= 0:
                    print(f"[Agent Update] batch {batch_idx}: buffer size={len(episode_buffer)}, avg_reward={avg_reward:.4f}", flush=True)
                
                # Synchronize all ranks before agent training
                if torch.distributed.is_initialized():
                    torch.distributed.barrier()
                
                # Only rank 0 trains the agent
                if args.rank == 0:
                    train_agent_episode(args, epoch, episode_buffer, avg_reward, agent, agent_optimizer)
                
                # Synchronize after training and broadcast weights
                if torch.distributed.is_initialized():
                    torch.distributed.barrier()
                    for param in agent.parameters():
                        torch.distributed.broadcast(param.data, src=0)
                
                # Clear buffer after update
                episode_buffer.clear()
                
                # Force garbage collection to free CPU memory from episode buffer
                import gc
                gc.collect()
                torch.cuda.empty_cache()
        else:
            # Original per-step reward normalization
            # Handle case where batch_size=1 or std is too small (avoid division by zero)
            if rewards_std < 1e-8 or torch.isnan(rewards_std) or torch.isinf(rewards_std):
                # For single sample or constant rewards, just use the raw reward normalized to [0,1]
                normalized_reward = torch.sigmoid(reward)  # Maps to [0,1] range
            else:
                normalized_reward = (reward - rewards_mean) / rewards_std 
                normalized_reward = torch.clamp(normalized_reward, min=-3, max=3)  # Clip outliers before sigmoid
                normalized_reward = torch.sigmoid(normalized_reward)  # Map to [0,1] range
            
            normalized_reward = normalized_reward.detach()
            #print('normalized_reward', normalized_reward)
            # Move reward to CPU to save GPU memory
            agent_rewards.append(normalized_reward.cpu())
        
        # For logging, scale losses back to normal (multiply by accumulation steps)
        # since we divided by gradient_accumulation_steps earlier
        train_loss.update(loss.item() * gradient_accumulation_steps, inputs.size(0))
        train_loss_cls.update((loss_cls.item() / gradient_accumulation_steps), inputs.size(0))
        train_loss_kd.update((loss_kd.item() / gradient_accumulation_steps), inputs.size(0))
        train_loss_feat.update((loss_feat.item() / gradient_accumulation_steps), inputs.size(0))
        
        # Log per-batch losses to TensorBoard
        if args.rank <= 0 and hasattr(args, 'writer'):
            global_step = epoch * len(train_loader) + batch_idx
            args.writer.add_scalar('Batch/train_total_loss', loss.item() * gradient_accumulation_steps, global_step)
            args.writer.add_scalar('Batch/train_cls_loss', loss_cls.item() / gradient_accumulation_steps, global_step)
            args.writer.add_scalar('Batch/train_kd_loss', loss_kd.item() / gradient_accumulation_steps, global_step)
            args.writer.add_scalar('Batch/train_feat_loss', loss_feat.item() / gradient_accumulation_steps, global_step)
        
        # For 3D segmentation, skip top-k accuracy (use Dice/IoU metrics instead during validation)
        # Use logits_main (main output) for metrics
        if logits_main.dim() == 5:
            # 3D segmentation: just count samples
            total += targets.size(0)
            metric_str = f'Samples: {total}'
        else:
            # 2D classification: compute top-k accuracy
            top1, top5 = correct_num(logits_main, targets, topk=(1, 5))
            top1_num += top1
            top5_num += top5
            total += targets.size(0)
            metric_str = f'Top-1 Acc: {(top1_num/total*100.).item():.2f}'

        # Update tqdm progress bar with current metrics (only if using tqdm)
        if args.rank <= 0 and is_tty and hasattr(train_loader_iter, 'set_postfix'):
            train_loader_iter.set_postfix({
                'CLS': f'{train_loss_cls.avg:.3f}',
                'KD': f'{train_loss_kd.avg:.3f}',
                'Feat': f'{train_loss_feat.avg:.3f}',
                'Total': f'{train_loss.avg:.3f}'
            })
        elif args.rank <= 0 and batch_idx % 10 == 0:
            # Print progress every 10 batches when not using tqdm
            balancer_info = ""
            if loss_balancer is not None:
                stats = loss_balancer.get_stats()
                balancer_info = f" | RM: cls={stats['running_mean_cls']:.2f}, kd={stats['running_mean_kd']:.2f}, feat={stats['running_mean_feat']:.2f}"
            
            # Monitor GPU memory usage
            gpu_mem_allocated = torch.cuda.memory_allocated(args.gpu) / 1024**3  # GB
            gpu_mem_reserved = torch.cuda.memory_reserved(args.gpu) / 1024**3  # GB
            
            # Monitor agent state buffer size (if not using episode rewards)
            buffer_info = ""
            if not (hasattr(args, 'use_episode_reward') and args.use_episode_reward):
                buffer_info = f" | Buffer: {len(agent_states)} states"
            
            print(f"Batch {batch_idx}/{len(train_loader)} - CLS: {train_loss_cls.avg:.3f}, KD: {train_loss_kd.avg:.3f}, Feat: {train_loss_feat.avg:.3f}, Total: {train_loss.avg:.3f}{balancer_info}{buffer_info} | GPU: {gpu_mem_allocated:.1f}/{gpu_mem_reserved:.1f}GB", flush=True)
            
        if batch_idx % args.agent_step == 0 and batch_idx != 0:
            # Skip per-step agent update if using episode-based rewards
            if not (hasattr(args, 'use_episode_reward') and args.use_episode_reward):
                # Synchronize all ranks before agent training
                if torch.distributed.is_initialized():
                    torch.distributed.barrier()
                
                # Only rank 0 trains the agent
                if args.rank == 0:
                    # Clear GPU cache before agent training to prevent fragmentation OOM
                    torch.cuda.empty_cache()
                    train_agent(args, epoch, agent_states, agent_rewards, feature_agent_actions, agent, agent_optimizer)
                
                # Synchronize after training
                if torch.distributed.is_initialized():
                    torch.distributed.barrier()
                    # Broadcast updated agent weights to all ranks
                    for param in agent.parameters():
                        torch.distributed.broadcast(param.data, src=0)
                
                # Clear agent state lists and force garbage collection
                agent_states = []
                feature_agent_actions = []
                agent_rewards = []
                
                # Force Python garbage collection to free CPU memory
                import gc
                gc.collect()
        
    
    # Compute final metrics based on dataset type
    if hasattr(args, 'dataset') and args.dataset == 'pimed':
        # For 3D segmentation, report samples processed (actual metrics computed during validation)
        if args.rank <= 0:
            log_msg = ('Epoch:{}\t lr:{:.4f}\t Duration:{:.3f}'
                        '\n Train_loss:{:.5f}'
                        '\t Train_loss_cls:{:.5f}'
                        '\t Train_loss_kd:{:.5f}'
                        '\t Train_loss_feat:{:.5f}'
                        '\nTrain samples processed: {}'
                        .format(epoch, lr, time.time() - start_time,
                                train_loss.avg,
                                train_loss_cls.avg,
                                train_loss_kd.avg,
                                train_loss_feat.avg,
                                total))
            
            # Add loss balancer stats summary
            if loss_balancer is not None:
                stats = loss_balancer.get_stats()
                log_msg += (f'\nLoss Balancer: running_mean_cls={stats["running_mean_cls"]:.3f}, '
                           f'running_mean_kd={stats["running_mean_kd"]:.3f}, '
                           f'running_mean_feat={stats["running_mean_feat"]:.3f}, '
                           f'steps={stats["step_count"]}')
            
            args.logger.info(log_msg)
        if hasattr(args, 'use_episode_reward') and args.use_episode_reward:
            # Return remaining buffer to caller for update with validation Dice as terminal reward
            # This ensures the agent gets the true generalization signal (val Dice) instead of avg loss
            if args.rank <= 0 and len(episode_buffer) > 0:
                print(f"[Episode Buffer] End of epoch (PIMED): returning buffer with {len(episode_buffer)} experiences for val_dice update", flush=True)
        elif len(agent_states) != 0:
            # Synchronize all ranks before final agent training
            if torch.distributed.is_initialized():
                torch.distributed.barrier()
            
            # Only rank 0 trains the agent
            if args.rank == 0:
                train_agent(args, epoch, agent_states, agent_rewards, feature_agent_actions, agent, agent_optimizer)
            
            # Synchronize after training and broadcast weights
            if torch.distributed.is_initialized():
                torch.distributed.barrier()
                for param in agent.parameters():
                    torch.distributed.broadcast(param.data, src=0)
        
        # Return losses, metrics, and remaining episode buffer for val_dice update
        result = {
            'train_loss': train_loss.avg,
            'train_loss_cls': train_loss_cls.avg,
            'train_loss_kd': train_loss_kd.avg,
            'train_loss_feat': train_loss_feat.avg,
            'train_acc': 0.0  # Placeholder for segmentation
        }
        if hasattr(args, 'use_episode_reward') and args.use_episode_reward and len(episode_buffer) > 0:
            result['episode_buffer'] = episode_buffer
        return result
    else:
        # For classification, compute accuracies
        acc1 = top1_num / total
        acc5 = top5_num / total

        if args.rank <= 0:
            log_msg = ('Epoch:{}\t lr:{:.4f}\t Duration:{:.3f}'
                        '\n Train_loss:{:.5f}'
                        '\t Train_loss_cls:{:.5f}'
                        '\t Train_loss_kd:{:.5f}'
                        '\t Train_loss_feat:{:.5f}'
                        '\nTrain top-1 accuracy:{:.2f}'
                        .format(epoch, lr, time.time() - start_time,
                                train_loss.avg,
                                train_loss_cls.avg,
                                train_loss_kd.avg,
                                train_loss_feat.avg,
                                acc1*100.))
            
            # Add loss balancer stats summary
            if loss_balancer is not None:
                stats = loss_balancer.get_stats()
                log_msg += (f'\nLoss Balancer: running_mean_cls={stats["running_mean_cls"]:.3f}, '
                           f'running_mean_kd={stats["running_mean_kd"]:.3f}, '
                           f'running_mean_feat={stats["running_mean_feat"]:.3f}, '
                           f'steps={stats["step_count"]}')
            
            args.logger.info(log_msg)
        if hasattr(args, 'use_episode_reward') and args.use_episode_reward:
            # Return remaining buffer to caller for update with validation metric as terminal reward
            if args.rank <= 0 and len(episode_buffer) > 0:
                print(f"[Episode Buffer] End of epoch (Classification): returning buffer with {len(episode_buffer)} experiences for val update", flush=True)
        elif len(agent_states) != 0:
            # Synchronize all ranks before final agent training
            if torch.distributed.is_initialized():
                torch.distributed.barrier()
            
            # Only rank 0 trains the agent
            if args.rank == 0:
                train_agent(args, epoch, agent_states, agent_rewards, feature_agent_actions, agent, agent_optimizer)
            
            # Synchronize after training and broadcast weights
            if torch.distributed.is_initialized():
                torch.distributed.barrier()
                for param in agent.parameters():
                    torch.distributed.broadcast(param.data, src=0)
        
        # Return losses, metrics, and remaining episode buffer for val update
        result = {
            'train_loss': train_loss.avg,
            'train_loss_cls': train_loss_cls.avg,
            'train_loss_kd': train_loss_kd.avg,
            'train_loss_feat': train_loss_feat.avg,
            'train_acc': acc1.item() * 100
        }
        if hasattr(args, 'use_episode_reward') and args.use_episode_reward and len(episode_buffer) > 0:
            result['episode_buffer'] = episode_buffer
        return result


def test(epoch, net, device, val_loader, criterion_ce, args, verbose=True, visualizer=None, teacher_names=None, amp_dtype=torch.float16):
    test_loss_cls = AverageMeter('Loss', ':.4e')
    
    # Initialize metrics based on dataset type
    if args.dataset == 'pimed':
        # Segmentation metrics - accumulate TP, FP, FN like nnUNet
        if hasattr(args, 'use_region_based_training') and args.use_region_based_training:
            # Region-based: detect number of regions from n_cls (2 for binary, 3 for 3-class)
            num_regions = args.n_cls if hasattr(args, 'n_cls') else 3
            tp_accumulated = np.zeros(num_regions)
            fp_accumulated = np.zeros(num_regions)
            fn_accumulated = np.zeros(num_regions)
            # Define region names based on number of channels
            if num_regions == 2:
                # Binary mode: [prostate, cancer]
                region_names = ['prostate', 'cancer']
            else:
                # 3-class mode: [prostate, PCa, csPCa]
                region_names = ['prostate', 'PCa', 'csPCa']
        else:
            # Class-based: 3 foreground classes
            num_classes = args.n_cls
            tp_accumulated = np.zeros(num_classes - 1)  # Exclude background
            fp_accumulated = np.zeros(num_classes - 1)
            fn_accumulated = np.zeros(num_classes - 1)
    else:
        # Classification metrics
        top1_num = 0
        top5_num = 0
        total = 0
    
    # Reset visualizer for new epoch
    if visualizer is not None:
        visualizer.reset_for_new_epoch()
    
    net.eval()
    
    # Wrap validation loader with tqdm (only in interactive mode)
    import sys
    is_tty = sys.stdout.isatty()
    if is_tty and verbose:
        val_loader_iter = tqdm(val_loader, desc=f"Validation Epoch {epoch}", disable=(args.rank != 0))
    else:
        val_loader_iter = val_loader
        if args.rank <= 0 and verbose:
            print(f"===> Starting validation epoch {epoch} with {len(val_loader)} batches...", flush=True)
    
    with torch.no_grad():
        for batch_idx, batch_data in enumerate(val_loader_iter):
            batch_start_time = time.time()
            
            # Handle different batch formats
            if args.dataset == 'pimed':
                # PIMED with features: (img, label, teacher_features, teacher_logits, case_ids, label_original, prostate_mask)
                # PIMED without features: (img, label, case_ids, label_original, prostate_mask)
                if len(batch_data) == 7:
                    inputs, targets, teacher_features, teacher_logits, case_ids, targets_original, prostate_mask = batch_data
                elif len(batch_data) == 6:
                    inputs, targets, teacher_features, teacher_logits, case_ids, targets_original = batch_data
                    prostate_mask = None
                elif len(batch_data) == 5:
                    # No features loaded (validation mode)
                    inputs, targets, case_ids, targets_original, prostate_mask = batch_data
                    teacher_features = None
                    teacher_logits = None
                else:
                    # Backward compatibility
                    inputs, targets, teacher_features, teacher_logits, case_ids = batch_data
                    targets_original = targets
                    prostate_mask = None
            else:
                # CIFAR/ImageNet: (img, label)
                inputs, targets = batch_data
                teacher_logits = None
                case_ids = None
                targets_original = targets
                prostate_mask = None
                
            inputs = inputs.to(device, non_blocking=True)
            # For region-based training, targets are already float multi-hot [B, C, D, H, W]
            # For class-based training, targets should be long [B, D, H, W]
            if targets.dim() == 5:  # [B, C, D, H, W] - region-based
                targets = targets.to(device, non_blocking=True).float()
            else:  # [B, D, H, W] - class-based
                targets = targets.to(device, non_blocking=True).long()
            
            # For DataParallel, pass is_feat as a tuple in the second argument
            if isinstance(net, torch.nn.DataParallel):
                features, logits = net(inputs, (True,))
            else:
                features, logits = net(inputs, is_feat=True)
            
            # Handle deep supervision: if logits is a list, use the final output
            if isinstance(logits, (list, tuple)):
                logits_final = logits[0]  # First element is the final full-resolution output
            else:
                logits_final = logits
            
            # Debug: Print logits statistics for first batch
            if batch_idx == 0 and args.rank <= 0:
                nan_count = torch.isnan(logits_final).sum().item()
                total_elements = logits_final.numel()
                if nan_count > 0:
                    print(f"  [DEBUG] WARNING: {nan_count}/{total_elements} NaN values in validation logits!", flush=True)
                else:
                    print(f"  [DEBUG] Validation logits stats: min={logits_final.min().item():.4f}, max={logits_final.max().item():.4f}, mean={logits_final.mean().item():.4f}, std={logits_final.std().item():.4f}", flush=True)
                print(f"  [DEBUG] Logits shape: {logits_final.shape}, non-zero elements: {(logits_final != 0).sum().item()}/{logits_final.numel()}", flush=True)
            
            # Visualize validation samples (only for PIMED dataset, first batch only)
            # Use targets_original for visualization (class labels, not region-based)
            if (batch_idx == 0 and args.dataset == 'pimed' and visualizer is not None 
                and teacher_names is not None):
                try:
                    visualizer.visualize_batch(
                        inputs=inputs,
                        labels=targets_original,
                        student_logits=logits_final,
                        teacher_logits=teacher_logits,
                        case_ids=case_ids,
                        epoch=epoch,
                        teacher_names=teacher_names,
                        mode='val'  # Add mode to distinguish train/val in saved filenames
                    )
                except Exception as e:
                    pass  # Silently skip visualization errors
            
            # Compute loss (use original logits for deep supervision loss)
            loss_cls = criterion_ce(logits_final, targets)
            test_loss_cls.update(loss_cls.item(), inputs.size(0))
            
            # Compute appropriate metrics
            if args.dataset == 'pimed':
                if hasattr(args, 'use_region_based_training') and args.use_region_based_training:
                    # Region-based: compute TP, FP, FN per region
                    # Apply sigmoid and threshold at 0.5
                    predicted_probs = torch.sigmoid(logits_final)  # [B, 3, D, H, W]
                    
                    # Print per-channel statistics (every 25 batches)
                    if args.rank <= 0 and batch_idx % 25 == 0:
                        for c in range(predicted_probs.size(1)):
                            # Compute mean/std across spatial dimensions for this channel
                            channel_probs = predicted_probs[:, c]  # [B, D, H, W]
                            target_channel = targets[:, c] if targets.dim() == 5 else None
                            global_mean = channel_probs.mean().item()
                            global_std = channel_probs.std().item()
                            target_sum = target_channel.sum().item() if target_channel is not None else 0
                            print(f"[VAL Batch {batch_idx}] Channel {c}: pred_mean={global_mean:.4f}, pred_std={global_std:.4f}, target_sum={target_sum:.0f}", flush=True)
                    
                    predicted_regions = (predicted_probs > 0.5).long()  # [B, 3, D, H, W]
                    target_regions = targets.long()  # [B, 3, D, H, W]
                    
                    # Compute TP, FP, FN per region (sum over B, D, H, W, keep C dimension)
                    tp, fp, fn = compute_tp_fp_fn(predicted_regions, target_regions, axes=(0, 2, 3, 4))
                    
                    # Accumulate
                    tp_accumulated += tp
                    fp_accumulated += fp
                    fn_accumulated += fn
                    
                    # Compute running dice for display (not used for final metrics)
                    running_dice = np.array([2 * t / (2 * t + f_p + f_n) if (2 * t + f_p + f_n) > 0 else 0.0 
                                            for t, f_p, f_n in zip(tp_accumulated, fp_accumulated, fn_accumulated)])
                    
                    # Update progress (tqdm or print)
                    if args.rank <= 0 and verbose:
                        if is_tty and hasattr(val_loader_iter, 'set_postfix'):
                            # Dynamic postfix based on number of regions
                            postfix_dict = {name: f'{running_dice[i]:.4f}' for i, name in enumerate(region_names)}
                            postfix_dict['Loss'] = f'{test_loss_cls.avg:.4f}'
                            val_loader_iter.set_postfix(postfix_dict)
                        elif batch_idx % 10 == 0:
                            dice_str = '/'.join([f'{running_dice[i]:.4f}' for i in range(num_regions)])
                            region_str = '/'.join(region_names)
                            print(f"Val batch {batch_idx}/{len(val_loader)} - "
                                  f"Dice[{region_str}]: {dice_str}, "
                                  f"Loss: {test_loss_cls.avg:.4f}", flush=True)
                else:
                    # Standard per-class segmentation metrics - also use TP/FP/FN accumulation
                    # Get predictions
                    predicted_classes = torch.argmax(logits_final, dim=1, keepdim=True)  # [B, 1, D, H, W]
                    # Convert to one-hot [B, n_cls, D, H, W]
                    predicted_onehot = torch.zeros_like(logits_final)
                    predicted_onehot.scatter_(1, predicted_classes, 1)
                    
                    # Convert targets to one-hot if needed
                    if targets.dim() == 4:  # [B, D, H, W]
                        target_onehot = torch.zeros(targets.size(0), args.n_cls, *targets.shape[1:], device=targets.device)
                        target_onehot.scatter_(1, targets.unsqueeze(1).long(), 1)
                    else:
                        target_onehot = targets
                    
                    # Compute TP, FP, FN (exclude background - class 0)
                    tp, fp, fn = compute_tp_fp_fn(predicted_onehot[:, 1:], target_onehot[:, 1:], axes=(0, 2, 3, 4))
                    
                    # Accumulate
                    tp_accumulated += tp
                    fp_accumulated += fp
                    fn_accumulated += fn
                    
                    # Compute running dice for display
                    running_dice = np.mean([2 * t / (2 * t + f_p + f_n) if (2 * t + f_p + f_n) > 0 else 0.0 
                                           for t, f_p, f_n in zip(tp_accumulated, fp_accumulated, fn_accumulated)])
                    
                    # Update progress (tqdm or print)
                    if args.rank <= 0 and verbose:
                        if is_tty and hasattr(val_loader_iter, 'set_postfix'):
                            val_loader_iter.set_postfix({
                                'Dice': f'{running_dice:.4f}',
                                'Loss': f'{test_loss_cls.avg:.4f}'
                            })
                        elif batch_idx % 10 == 0:
                            print(f"Val batch {batch_idx}/{len(val_loader)} - Dice: {running_dice:.4f}, Loss: {test_loss_cls.avg:.4f}", flush=True)
            else:
                # Classification metrics: Top-1 and Top-5 accuracy
                top1, top5 = correct_num(logits_final, targets, topk=(1, 5))
                top1_num += top1
                top5_num += top5
                total += targets.size(0)
                
                # Update progress (tqdm or print)
                if args.rank <= 0 and verbose:
                    if is_tty and hasattr(val_loader_iter, 'set_postfix'):
                        val_loader_iter.set_postfix({
                            'Top1': f'{(top1_num/total*100.).item():.2f}%',
                            'Top5': f'{(top5_num/total*100.).item():.2f}%',
                            'Loss': f'{test_loss_cls.avg:.4f}'
                        })
                    elif batch_idx % 10 == 0:
                        print(f"Val batch {batch_idx}/{len(val_loader)} - Top1: {(top1_num/total*100.).item():.2f}%, Top5: {(top5_num/total*100.).item():.2f}%, Loss: {test_loss_cls.avg:.4f}", flush=True)
    
    # Compute final metrics and log
    if args.dataset == 'pimed':
        if hasattr(args, 'use_region_based_training') and args.use_region_based_training:
            # Region-based: compute final dice from accumulated TP, FP, FN (nnUNet style)
            # Dice = 2*TP / (2*TP + FP + FN)
            final_dice_per_region = [2 * tp / (2 * tp + fp + fn) if (2 * tp + fp + fn) > 0 else 0.0 
                                     for tp, fp, fn in zip(tp_accumulated, fp_accumulated, fn_accumulated)]
            
            # Map to region names (already defined at the beginning)
            final_dice = {name: dice for name, dice in zip(region_names, final_dice_per_region)}
            
            # Compute foreground mean (average over all regions, matching nnUNet)
            dice_foreground_mean = np.nanmean(final_dice_per_region)
            
            if args.rank <= 0 and verbose:
                # Build dynamic log message based on number of regions
                log_lines = [f'Test epoch:{epoch}\t Test_loss_cls:{test_loss_cls.avg:.5f}',
                            'Test Dice Scores (Region-based, nnUNet-compatible, computed from accumulated TP/FP/FN):']
                for i, name in enumerate(region_names):
                    log_lines.append(f'  - {name}: {final_dice_per_region[i]:.4f} (TP={int(tp_accumulated[i])}, FP={int(fp_accumulated[i])}, FN={int(fn_accumulated[i])})')
                log_lines.append(f'  - Foreground Mean: {dice_foreground_mean:.4f}')
                args.logger.info('\n'.join(log_lines))
            
            # Compute IoU from accumulated TP/FP/FN: IoU = TP / (TP + FP + FN)
            final_iou_per_region = [tp / (tp + fp + fn) if (tp + fp + fn) > 0 else 0.0
                                    for tp, fp, fn in zip(tp_accumulated, fp_accumulated, fn_accumulated)]
            iou_foreground_mean = np.nanmean(final_iou_per_region)
            final_iou = {name: iou for name, iou in zip(region_names, final_iou_per_region)}

            if args.rank <= 0 and verbose:
                for i, name in enumerate(region_names):
                    args.logger.info(f'  - {name} IoU: {final_iou_per_region[i]:.4f}')
                args.logger.info(f'  - Foreground Mean IoU: {iou_foreground_mean:.4f}')

            # Return as dict for easier unpacking in train_student_rl.py
            # Handle both binary and 3-class modes
            result = {
                'dice': dice_foreground_mean,  # Use foreground mean for checkpointing (matches nnUNet)
                'dice_prostate': final_dice.get('prostate', 0.0),
                'iou': iou_foreground_mean,
                'iou_prostate': final_iou.get('prostate', 0.0),
                'loss': test_loss_cls.avg
            }
            # Add cancer-related dice/iou based on mode
            if num_regions == 2:
                # Binary mode: single cancer channel
                result['dice_cancer'] = final_dice.get('cancer', 0.0)
                result['iou_cancer'] = final_iou.get('cancer', 0.0)
                # For compatibility with 3-class code, also set PCa and csPCa to cancer value
                result['dice_PCa'] = final_dice.get('cancer', 0.0)
                result['dice_csPCa'] = final_dice.get('cancer', 0.0)
            else:
                # 3-class mode
                result['dice_PCa'] = final_dice.get('PCa', 0.0)
                result['dice_csPCa'] = final_dice.get('csPCa', 0.0)
                result['iou_cancer'] = final_iou.get('csPCa', 0.0)
            return result
        else:
            # Standard per-class metrics - use accumulated TP/FP/FN
            final_dice_per_class = [2 * tp / (2 * tp + fp + fn) if (2 * tp + fp + fn) > 0 else 0.0 
                                   for tp, fp, fn in zip(tp_accumulated, fp_accumulated, fn_accumulated)]
            final_dice = np.mean(final_dice_per_class)
            
            # Compute IoU from accumulated TP/FP/FN: IoU = TP / (TP + FP + FN)
            final_iou_per_class = [tp / (tp + fp + fn) if (tp + fp + fn) > 0 else 0.0 
                                   for tp, fp, fn in zip(tp_accumulated, fp_accumulated, fn_accumulated)]
            final_iou = np.mean(final_iou_per_class)
            
            if args.rank <= 0 and verbose:
                args.logger.info('Test epoch:{}\t Test_loss_cls:{:.5f}\n'
                               'Test Dice Score (computed from accumulated TP/FP/FN): {:.4f}\n'
                               'Test IoU Score (computed from accumulated TP/FP/FN): {:.4f}\n'
                               'Per-class Dice: {}\n'
                               'Per-class IoU: {}'
                            .format(epoch, test_loss_cls.avg, final_dice, final_iou,
                                   [f'{d:.4f}' for d in final_dice_per_class],
                                   [f'{i:.4f}' for i in final_iou_per_class]))
            
            # Return as dict for easier unpacking in train_student_rl.py
            return {
                'dice': final_dice,
                'iou': final_iou,
                'loss': test_loss_cls.avg
            }
    else:
        class_acc1 = round((top1_num/total*100.).item(), 4)
        class_acc5 = round((top5_num/total*100.).item(), 4)
        
        if args.rank <= 0 and verbose:
            args.logger.info('Test epoch:{}\t Test_loss_cls:{:.5f}\nTest top-1 accuracy: {}\nTest top-5 accuracy: {}'
                        .format(epoch, test_loss_cls.avg, str(class_acc1), str(class_acc5)))
        
        return class_acc1, test_loss_cls.avg  # Return accuracy and loss for classification
    