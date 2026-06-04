"""
3D Segmentation Losses for Medical Image Segmentation
Designed for MTKD-RL framework with PIMED dataset

Handles:
- Input logits: [B, C, D, H, W] (e.g., [2, 2, 20, 256, 256])
- Target labels: [B, D, H, W] (e.g., [2, 20, 256, 256])
- Binary segmentation: C=2 (background vs prostate)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class DiceLoss(nn.Module):
    """
    Dice Loss for segmentation tasks (nnUNet-compatible)
    Better than CrossEntropyLoss for imbalanced segmentation (small prostate vs large background)
    
    Uses smooth=1.0 by default to match nnUNet's behavior
    """
    def __init__(self, smooth=1.0, ignore_index=None, class_weights=None, include_background=True):
        super(DiceLoss, self).__init__()
        self.smooth = smooth  # Default 1.0 (nnUNet standard)
        self.ignore_index = ignore_index
        self.class_weights = class_weights
        self.include_background = include_background  # Whether to include class 0 (background) in Dice calculation

    def forward(self, inputs, targets):
        """
        Args:
            inputs: [B, C, D, H, W] - predicted logits
            targets: [B, D, H, W] - ground truth labels (long tensor)
        Returns:
            dice_loss: scalar tensor
        """
        # Convert logits to probabilities
        inputs = F.softmax(inputs, dim=1)  # [B, C, D, H, W]
        
        # Convert targets to one-hot encoding
        num_classes = inputs.size(1)
        targets_one_hot = F.one_hot(targets.long(), num_classes=num_classes)  # [B, D, H, W, C]
        targets_one_hot = targets_one_hot.permute(0, 4, 1, 2, 3).float()  # [B, C, D, H, W]
        
        # Handle ignore_index
        if self.ignore_index is not None:
            mask = (targets != self.ignore_index).float()
            mask = mask.unsqueeze(1).expand_as(inputs)  # [B, C, D, H, W]
            inputs = inputs * mask
            targets_one_hot = targets_one_hot * mask
        
        # Flatten for calculation
        inputs_flat = inputs.view(inputs.size(0), inputs.size(1), -1)  # [B, C, D*H*W]
        targets_flat = targets_one_hot.view(targets_one_hot.size(0), targets_one_hot.size(1), -1)  # [B, C, D*H*W]
        
        # Calculate intersection and union
        intersection = (inputs_flat * targets_flat).sum(dim=2)  # [B, C]
        union = inputs_flat.sum(dim=2) + targets_flat.sum(dim=2)  # [B, C]
        
        # Dice coefficient per class
        dice = (2. * intersection + self.smooth) / (union + self.smooth)  # [B, C]
        
        # Exclude background if requested
        if not self.include_background:
            dice = dice[:, 1:]  # Remove class 0 (background), keep only [B, C-1]
        
        # Apply class weights if provided
        if self.class_weights is not None:
            class_weights = self.class_weights.to(dice.device)
            if not self.include_background:
                class_weights = class_weights[1:]  # Remove background weight
            dice = dice * class_weights.unsqueeze(0)
        
        # Average over batch and classes
        dice_loss = 1 - dice.mean()
        
        return dice_loss


class IoULoss(nn.Module):
    """
    Intersection over Union (IoU) Loss for segmentation
    Also known as Jaccard Loss
    """
    def __init__(self, smooth=1e-5, ignore_index=None):
        super(IoULoss, self).__init__()
        self.smooth = smooth
        self.ignore_index = ignore_index

    def forward(self, inputs, targets):
        """
        Args:
            inputs: [B, C, D, H, W] - predicted logits
            targets: [B, D, H, W] - ground truth labels
        Returns:
            iou_loss: scalar tensor
        """
        # Convert logits to probabilities
        inputs = F.softmax(inputs, dim=1)  # [B, C, D, H, W]
        
        # Convert targets to one-hot
        num_classes = inputs.size(1)
        targets_one_hot = F.one_hot(targets.long(), num_classes=num_classes)  # [B, D, H, W, C]
        targets_one_hot = targets_one_hot.permute(0, 4, 1, 2, 3).float()  # [B, C, D, H, W]
        
        # Handle ignore_index
        if self.ignore_index is not None:
            mask = (targets != self.ignore_index).float()
            mask = mask.unsqueeze(1).expand_as(inputs)
            inputs = inputs * mask
            targets_one_hot = targets_one_hot * mask
        
        # Flatten
        inputs_flat = inputs.view(inputs.size(0), inputs.size(1), -1)
        targets_flat = targets_one_hot.view(targets_one_hot.size(0), targets_one_hot.size(1), -1)
        
        # Calculate IoU
        intersection = (inputs_flat * targets_flat).sum(dim=2)  # [B, C]
        union = inputs_flat.sum(dim=2) + targets_flat.sum(dim=2) - intersection  # [B, C]
        
        iou = (intersection + self.smooth) / (union + self.smooth)  # [B, C]
        iou_loss = 1 - iou.mean()
        
        return iou_loss


class FocalLoss(nn.Module):
    """
    Focal Loss for addressing class imbalance in segmentation
    Focuses on hard examples (low probability predictions)
    """
    def __init__(self, alpha=1.0, gamma=2.0, ignore_index=None):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.ignore_index = ignore_index

    def forward(self, inputs, targets):
        """
        Args:
            inputs: [B, C, D, H, W] - predicted logits
            targets: [B, D, H, W] - ground truth labels
        Returns:
            focal_loss: scalar tensor
        """
        # Standard cross entropy
        ce_loss = F.cross_entropy(inputs, targets.long(), 
                                 ignore_index=self.ignore_index, 
                                 reduction='none')  # [B, D, H, W]
        
        # Calculate probabilities
        pt = torch.exp(-ce_loss)
        
        # Apply focal term
        focal_loss = self.alpha * (1 - pt) ** self.gamma * ce_loss
        
        return focal_loss.mean()


class BinaryFocalLoss(nn.Module):
    """
    Binary Focal Loss for region-based training with sigmoid activation.
    
    Focal Loss: FL(p_t) = -alpha * (1 - p_t)^gamma * log(p_t)
    
    Where p_t = p if y=1, else p_t = 1-p
    
    Key features:
    - Works with sigmoid activation (independent binary predictions per channel)
    - Down-weights easy examples (high confidence correct predictions)
    - Focuses training on hard examples (low confidence predictions)
    - Per-channel gamma for selective application
    
    For region-based training:
    - Easy negatives (background far from prostate) get heavily down-weighted
    - Hard examples (cancer boundaries, small lesions) get full weight
    
    Args:
        gamma: Focusing parameter. Higher = more focus on hard examples
               - gamma=0: equivalent to BCE
               - gamma=2: standard focal loss (recommended)
               Can be a single float or per-channel list [g0, g1, ...]
        alpha: Class balance weight (not used if pos_weight provided)
        pos_weight: Per-channel foreground weighting [pw0, pw1, ...]
        channel_weights: Per-channel importance weights [w0, w1, ...]
    """
    def __init__(self, gamma=2.0, alpha=1.0, pos_weight=None, channel_weights=None):
        super(BinaryFocalLoss, self).__init__()
        self.gamma = gamma if isinstance(gamma, (list, tuple)) else [gamma]
        self.alpha = alpha
        
        if pos_weight is not None:
            self.register_buffer('pos_weight', torch.tensor(pos_weight, dtype=torch.float32))
        else:
            self.pos_weight = None
            
        if channel_weights is not None:
            self.register_buffer('channel_weights', torch.tensor(channel_weights, dtype=torch.float32))
        else:
            self.channel_weights = None

    def forward(self, inputs, targets):
        """
        Args:
            inputs: [B, C, D, H, W] - predicted logits (before sigmoid)
            targets: [B, C, D, H, W] - ground truth (0 or 1)
        Returns:
            focal_loss: scalar tensor
        """
        # Get probabilities via sigmoid
        probs = torch.sigmoid(inputs)
        
        # p_t = p if y=1, else 1-p
        p_t = probs * targets + (1 - probs) * (1 - targets)
        
        # Compute focal weight: (1 - p_t)^gamma
        # For correct confident predictions: p_t ≈ 1, weight ≈ 0 (down-weighted)
        # For incorrect/uncertain predictions: p_t ≈ 0.5, weight ≈ 0.25
        num_channels = inputs.size(1)
        gamma_tensor = torch.tensor(
            self.gamma * (num_channels // len(self.gamma) + 1),  # Repeat gamma if needed
            device=inputs.device, dtype=inputs.dtype
        )[:num_channels]
        gamma_tensor = gamma_tensor.view(1, -1, 1, 1, 1)  # [1, C, 1, 1, 1]
        
        focal_weight = (1 - p_t) ** gamma_tensor
        
        # Compute BCE loss (numerically stable)
        # BCE = -[y * log(p) + (1-y) * log(1-p)]
        # Using log_sigmoid for numerical stability
        bce_loss = F.binary_cross_entropy_with_logits(inputs, targets.float(), reduction='none')
        
        # Apply pos_weight to foreground pixels if provided
        if self.pos_weight is not None:
            pos_w = self.pos_weight.to(inputs.device).view(1, -1, 1, 1, 1)
            # Weight foreground (y=1) by pos_w, background (y=0) by 1
            bce_loss = bce_loss * (pos_w * targets.float() + (1 - targets.float()))
        
        # Apply focal weighting
        focal_loss = focal_weight * bce_loss
        
        # Apply channel weights if provided
        if self.channel_weights is not None:
            weights = self.channel_weights.to(inputs.device).view(1, -1, 1, 1, 1)
            focal_loss = focal_loss * weights
            # Normalize by total weight
            return focal_loss.sum() / (weights.sum() * inputs.numel() / inputs.size(1))
        
        return focal_loss.mean()


class nnUNetDiceLoss(nn.Module):
    """
    Exact replica of nnUNet's MemoryEfficientSoftDiceLoss
    
    Key differences from standard DiceLoss:
    - batch_dice=False: Computes Dice per sample, then averages (default for 3d_fullres)
    - do_bg=False: Excludes background from Dice computation (nnUNet default)
    - smooth=1.0: nnUNet standard smoothing factor
    - Returns NEGATIVE Dice (loss = -dice)
    """
    def __init__(self, batch_dice=False, do_bg=False, smooth=1.0):
        super(nnUNetDiceLoss, self).__init__()
        self.batch_dice = batch_dice
        self.do_bg = do_bg
        self.smooth = smooth
    
    def forward(self, x, y, loss_mask=None):
        """
        Args:
            x: [B, C, D, H, W] - predicted logits (will apply softmax)
            y: [B, D, H, W] - ground truth labels (NOT one-hot)
            loss_mask: Optional mask for ignored regions
        Returns:
            -dice: scalar tensor (NEGATIVE Dice, as in nnUNet)
        """
        # Apply softmax to logits
        x = F.softmax(x, dim=1)
        
        # Convert to shape (b, c) - sum over spatial dimensions
        axes = tuple(range(2, x.ndim))
        
        with torch.no_grad():
            # Expand y to have channel dimension if needed
            if x.ndim != y.ndim:
                y = y.view((y.shape[0], 1, *y.shape[1:]))
            
            # Convert to one-hot encoding
            if x.shape == y.shape:
                y_onehot = y
            else:
                y_onehot = torch.zeros(x.shape, device=x.device, dtype=torch.bool)
                y_onehot.scatter_(1, y.long(), 1)
            
            # Exclude background if requested
            if not self.do_bg:
                y_onehot = y_onehot[:, 1:]
            
            sum_gt = y_onehot.sum(axes) if loss_mask is None else (y_onehot * loss_mask).sum(axes)
        
        # Exclude background from predictions if requested
        if not self.do_bg:
            x = x[:, 1:]
        
        # Compute intersection and sum of predictions
        if loss_mask is None:
            intersect = (x * y_onehot).sum(axes)
            sum_pred = x.sum(axes)
        else:
            intersect = (x * y_onehot * loss_mask).sum(axes)
            sum_pred = (x * loss_mask).sum(axes)
        
        # Batch dice: sum over batch dimension before computing Dice
        if self.batch_dice:
            intersect = intersect.sum(0)
            sum_pred = sum_pred.sum(0)
            sum_gt = sum_gt.sum(0)
        
        # Compute Dice coefficient
        dc = (2 * intersect + self.smooth) / (torch.clamp(sum_gt + sum_pred + self.smooth, min=1e-8))
        
        # Average over classes (and batch if not batch_dice)
        dc = dc.mean()
        
        # Return NEGATIVE Dice (as loss)
        return -dc


class DC_and_CE_loss(nn.Module):
    """
    Exact replica of nnUNet's DC_and_CE_loss (Dice + CrossEntropy combined)
    
    This is THE standard nnUNet loss for region-based training.
    
    Default configuration matches nnUNet 3d_fullres:
    - weight_ce=1, weight_dice=1 (50/50 split)
    - batch_dice=False (per-sample Dice, then average)
    - do_bg=False (exclude background from Dice)
    - smooth=1.0 (nnUNet standard)
    """
    def __init__(self, weight_ce=1, weight_dice=1, ignore_label=None, 
                 batch_dice=False, do_bg=False, smooth=1.0):
        super(DC_and_CE_loss, self).__init__()
        self.weight_dice = weight_dice
        self.weight_ce = weight_ce
        self.ignore_label = ignore_label
        
        # nnUNet Dice loss (exact replica)
        self.dc = nnUNetDiceLoss(batch_dice=batch_dice, do_bg=do_bg, smooth=smooth)
        
        # Standard CrossEntropy (nnUNet uses RobustCrossEntropyLoss, but it's just a wrapper)
        if ignore_label is not None:
            self.ce = nn.CrossEntropyLoss(ignore_index=ignore_label)
        else:
            self.ce = nn.CrossEntropyLoss()
    
    def forward(self, net_output, target):
        """
        Args:
            net_output: [B, C, D, H, W] - predicted logits
            target: [B, D, H, W] - ground truth labels
        Returns:
            combined_loss: scalar tensor
        """
        # Handle ignore_label for Dice
        if self.ignore_label is not None:
            mask = (target != self.ignore_label)
            target_dice = torch.where(mask, target, torch.zeros_like(target))
            num_fg = mask.sum()
        else:
            target_dice = target
            mask = None
        
        # Compute losses
        dc_loss = self.dc(net_output, target_dice, loss_mask=mask) if self.weight_dice != 0 else 0
        ce_loss = self.ce(net_output, target.long()) if self.weight_ce != 0 and (self.ignore_label is None or num_fg > 0) else 0
        
        result = self.weight_ce * ce_loss + self.weight_dice * dc_loss
        return result


class DC_and_BCE_loss(nn.Module):
    """
    Dice + Binary Cross Entropy loss for region-based training (TRUE nnUNet regions)
    
    For region-based training, each output channel is an independent binary prediction:
    - Channel 0: Region (1,2,3) - all prostate tissue
    - Channel 1: Region (2,3) - cancer regions
    - Channel 2: Region (3) - clinically significant cancer only
    
    Uses sigmoid activation (not softmax) and BCE loss (not CE).
    Targets are converted to multi-hot encoding where a voxel can belong to multiple regions.
    
    Default configuration matches nnUNet region-based training:
    - weight_bce=1, weight_dice=1 (50/50 split)
    - batch_dice=False (per-sample Dice, then average)
    - do_bg=True (include background channel - first region channel)
    - smooth=1e-5 (nnUNet standard)
    
    NEW: channel_weights for channel-level weighting:
    - channel_weights: [w0, w1, w2] weights for each region channel
    - Default None = equal weights
    - Recommended for imbalanced data: [1.0, 3.0, 5.0] to focus on cancer regions
    
    NEW: pos_weight for pixel-level foreground weighting:
    - pos_weight: [pw0, pw1, pw2] weight for FOREGROUND pixels (where target=1) in each channel
    - This addresses class imbalance at PIXEL level within each channel
    - BCE becomes: -pos_weight * y * log(sigmoid(x)) - (1-y) * log(1-sigmoid(x))
    - Recommended: [10, 50, 100] for prostate/cancer/cspca (based on inverse frequency)
    """
    def __init__(self, weight_bce=1, weight_dice=1, ignore_label=None, 
                 batch_dice=False, do_bg=True, smooth=1e-5, channel_weights=None,
                 pos_weight=None):
        super(DC_and_BCE_loss, self).__init__()
        self.weight_dice = weight_dice
        self.weight_bce = weight_bce
        self.ignore_label = ignore_label
        self.do_bg = do_bg
        self.batch_dice = batch_dice
        self.smooth = smooth
        
        # Per-channel weights for foreground-focused training
        # channel_weights: [prostate_weight, cancer_weight, cspca_weight]
        if channel_weights is not None:
            self.register_buffer('channel_weights', torch.tensor(channel_weights, dtype=torch.float32))
        else:
            self.channel_weights = None
        
        # Per-channel pos_weight for PIXEL-LEVEL foreground weighting
        # pos_weight: weight for positive (foreground) pixels in BCE
        # Higher pos_weight = more penalty for missing foreground pixels
        if pos_weight is not None:
            self.register_buffer('pos_weight', torch.tensor(pos_weight, dtype=torch.float32))
        else:
            self.pos_weight = None
        
        # BCE with logits - always use reduction='none' to apply weights manually
        self.bce = nn.BCEWithLogitsLoss(reduction='none')
    
    def forward(self, net_output, target):
        """
        Args:
            net_output: [B, C, D, H, W] - predicted logits (will apply sigmoid)
            target: [B, C, D, H, W] - multi-hot encoded region targets (boolean or float, already converted by transforms)
        Returns:
            combined_loss: scalar tensor
        """
        # Convert boolean targets to float if needed
        if target.dtype == torch.bool:
            target = target.float()
        
        # Apply sigmoid to get probabilities
        probs = torch.sigmoid(net_output)
        
        # Handle ignore_label for masking
        if self.ignore_label is not None:
            # Assume the last channel is the ignore label mask
            mask = target[:, -1:, ...] == 0  # Where ignore label is NOT present
            target = target[:, :-1, ...]  # Remove ignore channel
        else:
            mask = None
        
        # Compute Dice loss on probabilities
        if self.weight_dice != 0:
            axes = tuple(range(2, net_output.ndim))  # spatial dimensions
            
            if self.batch_dice:
                # Compute Dice over the entire batch
                axes = tuple([0] + list(axes))
            
            # Skip background channel if do_bg=False
            start_ch = 0 if self.do_bg else 1
            probs_for_dice = probs[:, start_ch:, ...]
            target_for_dice = target[:, start_ch:, ...].float()
            
            if mask is not None:
                mask_for_dice = mask[:, :, ...].float() if mask.shape[1] == 1 else mask[:, start_ch:, ...].float()
                intersect = ((probs_for_dice * target_for_dice * mask_for_dice).sum(axes))
                sum_pred = ((probs_for_dice * mask_for_dice).sum(axes))
                sum_gt = ((target_for_dice * mask_for_dice).sum(axes))
            else:
                intersect = (probs_for_dice * target_for_dice).sum(axes)
                sum_pred = probs_for_dice.sum(axes)
                sum_gt = target_for_dice.sum(axes)
            
            if not self.batch_dice:
                # Average over batch after computing per-sample Dice
                intersect = intersect.sum(0)
                sum_pred = sum_pred.sum(0)
                sum_gt = sum_gt.sum(0)
            
            # Compute per-channel Dice coefficient
            dc = (2 * intersect + self.smooth) / (torch.clamp(sum_gt + sum_pred + self.smooth, min=1e-8))
            
            # Apply channel weights to Dice loss if provided
            if self.channel_weights is not None:
                # Get weights for channels being used (skip background if do_bg=False)
                weights = self.channel_weights[start_ch:].to(dc.device)
                # Weighted mean: higher weight = more penalty for low Dice
                dc_loss = -(dc * weights).sum() / weights.sum()
            else:
                dc_loss = -dc.mean()  # Negative because we want to maximize Dice
        else:
            dc_loss = 0
        
        # Compute BCE loss with pos_weight for PIXEL-LEVEL foreground emphasis
        if self.weight_bce != 0:
            # Manually compute BCE with pos_weight for each channel
            # BCE formula: -pos_weight * y * log(sigmoid(x)) - (1-y) * log(1-sigmoid(x))
            # This is equivalent to: pos_weight * y * F.softplus(-x) + (1-y) * F.softplus(x)
            # Using the numerically stable form from PyTorch
            
            if self.pos_weight is not None:
                # Apply per-channel pos_weight at PIXEL level
                # pos_weight gives more weight to foreground pixels (where target=1)
                pos_w = self.pos_weight.to(net_output.device)
                # Reshape for broadcasting: [C] -> [1, C, 1, 1, 1]
                pos_w = pos_w.view(1, -1, 1, 1, 1)
                
                # Compute BCE with pos_weight manually (numerically stable)
                # log_sigmoid(x) = -softplus(-x)
                # log(1 - sigmoid(x)) = -softplus(x)
                max_val = torch.clamp(-net_output, min=0)
                bce_per_element = (1 - target.float()) * net_output + max_val + \
                    torch.log(torch.exp(-max_val) + torch.exp(-net_output - max_val))
                # Apply pos_weight to positive (foreground) pixels
                # This multiplies the foreground loss term by pos_weight
                bce_per_element = bce_per_element * (pos_w * target.float() + (1 - target.float()))
            else:
                # Standard BCE without pos_weight
                bce_per_element = self.bce(net_output, target.float())  # [B, C, D, H, W]
            
            if self.ignore_label is not None:
                bce_per_element = bce_per_element[:, :-1, ...]
                bce_per_element = bce_per_element * mask.float()
            
            if self.channel_weights is not None:
                # Apply channel weights: [B, C, D, H, W] -> weighted by channel
                weights = self.channel_weights.to(bce_per_element.device)
                # Reshape weights for broadcasting: [C] -> [1, C, 1, 1, 1]
                weights = weights.view(1, -1, 1, 1, 1)
                bce_weighted = bce_per_element * weights
                
                # Log per-channel BCE loss values (before and after channel weighting)
                import random
                if random.random() < 0.005:  # Log 0.5% of batches to avoid spam
                    with torch.no_grad():
                        # Per-channel BCE BEFORE channel weighting (but AFTER pos_weight)
                        bce_per_channel_raw = bce_per_element.mean(dim=(0, 2, 3, 4))  # [C]
                        # Per-channel BCE AFTER channel weighting
                        bce_per_channel_weighted = bce_weighted.mean(dim=(0, 2, 3, 4))  # [C]
                        # Count foreground pixels per channel
                        fg_pixels_per_channel = target.float().sum(dim=(0, 2, 3, 4))  # [C]
                        total_pixels = target.shape[0] * target.shape[2] * target.shape[3] * target.shape[4]
                        fg_ratio = fg_pixels_per_channel / total_pixels
                        print(f"[BCE DEBUG] Raw BCE per channel (after pos_w): ch0={bce_per_channel_raw[0]:.4f}, ch1={bce_per_channel_raw[1]:.4f}, ch2={bce_per_channel_raw[2]:.4f}")
                        print(f"[BCE DEBUG] Weighted BCE per channel: ch0={bce_per_channel_weighted[0]:.4f}, ch1={bce_per_channel_weighted[1]:.4f}, ch2={bce_per_channel_weighted[2]:.4f}")
                        print(f"[BCE DEBUG] FG pixel ratio: ch0={fg_ratio[0]:.4f}, ch1={fg_ratio[1]:.4f}, ch2={fg_ratio[2]:.4f}")
                
                if self.ignore_label is not None:
                    bce_loss = bce_weighted.sum() / (mask.sum() * weights.sum())
                else:
                    # Mean over all dimensions, normalized by total weight
                    bce_loss = bce_weighted.mean() * (len(self.channel_weights) / self.channel_weights.sum())
            else:
                if self.ignore_label is not None:
                    bce_loss = bce_per_element.sum() / mask.sum()
                else:
                    bce_loss = bce_per_element.mean()
        else:
            bce_loss = 0
        
        result = self.weight_bce * bce_loss + self.weight_dice * dc_loss
        return result


class DC_and_Focal_loss(nn.Module):
    """
    Dice + Binary Focal Loss for region-based training.
    
    Focal Loss: FL(p_t) = -(1 - p_t)^gamma * log(p_t)
    
    Key advantage over BCE:
    - Down-weights easy examples (confident correct predictions)
    - Focuses training on hard examples (cancer boundaries, small lesions)
    - Helps with extreme class imbalance where background dominates
    
    With gamma=2:
    - p_t=0.9 (easy): weight = 0.01 (100× down-weighted)
    - p_t=0.5 (medium): weight = 0.25 (4× down-weighted)
    - p_t=0.1 (hard): weight = 0.81 (nearly full)
    
    Args:
        weight_focal: Weight for focal loss component
        weight_dice: Weight for Dice loss component
        gamma: Focusing parameter (higher = more focus on hard examples)
        pos_weight: Per-channel foreground pixel weighting [w0, w1, ...]
        channel_weights: Per-channel importance weighting [w0, w1, ...]
        batch_dice: If True, compute Dice over entire batch
        do_bg: If True, include first channel (prostate) in Dice
        smooth: Smoothing factor for Dice computation
    """
    def __init__(self, weight_focal=1.0, weight_dice=1.0, gamma=2.0,
                 pos_weight=None, channel_weights=None,
                 batch_dice=False, do_bg=True, smooth=1e-5):
        super(DC_and_Focal_loss, self).__init__()
        self.weight_focal = weight_focal
        self.weight_dice = weight_dice
        self.gamma = gamma
        self.batch_dice = batch_dice
        self.do_bg = do_bg
        self.smooth = smooth
        
        if pos_weight is not None:
            self.register_buffer('pos_weight', torch.tensor(pos_weight, dtype=torch.float32))
        else:
            self.pos_weight = None
            
        if channel_weights is not None:
            self.register_buffer('channel_weights', torch.tensor(channel_weights, dtype=torch.float32))
        else:
            self.channel_weights = None
    
    def forward(self, net_output, target, loss_mask=None):
        """
        Args:
            net_output: [B, C, D, H, W] - predicted logits (before sigmoid)
            target: [B, C, D, H, W] - ground truth multi-hot encoding
            loss_mask: Optional mask for ignoring certain voxels
            
        Returns:
            Combined Dice + Focal loss
        """
        # ========== DICE LOSS ==========
        if self.weight_dice != 0:
            # Compute probabilities via sigmoid
            probs = torch.sigmoid(net_output)
            
            # Determine axes for reduction
            # For [B, C, D, H, W]: reduce over spatial dims (D, H, W) = axes (2, 3, 4)
            axes = tuple(range(2, net_output.ndim))
            
            if self.batch_dice:
                axes = tuple([0] + list(axes))
            
            # Skip background channel if do_bg=False
            start_ch = 0 if self.do_bg else 1
            probs_for_dice = probs[:, start_ch:, ...]
            target_for_dice = target[:, start_ch:, ...].float()
            
            intersect = (probs_for_dice * target_for_dice).sum(axes)
            sum_pred = probs_for_dice.sum(axes)
            sum_gt = target_for_dice.sum(axes)
            
            if not self.batch_dice:
                intersect = intersect.sum(0)
                sum_pred = sum_pred.sum(0)
                sum_gt = sum_gt.sum(0)
            
            dc = (2 * intersect + self.smooth) / (torch.clamp(sum_gt + sum_pred + self.smooth, min=1e-8))
            
            if self.channel_weights is not None:
                weights = self.channel_weights[start_ch:].to(dc.device)
                dc_loss = -(dc * weights).sum() / weights.sum()
            else:
                dc_loss = -dc.mean()
        else:
            dc_loss = 0
        
        # ========== FOCAL LOSS ==========
        if self.weight_focal != 0:
            # Get probabilities
            probs = torch.sigmoid(net_output)
            
            # p_t = p if y=1, else 1-p
            p_t = probs * target.float() + (1 - probs) * (1 - target.float())
            
            # Focal weight: (1 - p_t)^gamma
            focal_weight = (1 - p_t) ** self.gamma
            
            # BCE loss (numerically stable)
            bce = F.binary_cross_entropy_with_logits(net_output, target.float(), reduction='none')
            
            # Apply pos_weight to foreground pixels
            if self.pos_weight is not None:
                pos_w = self.pos_weight.to(net_output.device).view(1, -1, 1, 1, 1)
                bce = bce * (pos_w * target.float() + (1 - target.float()))
            
            # Apply focal weighting
            focal_loss_raw = focal_weight * bce
            
            # Apply channel weights
            if self.channel_weights is not None:
                weights = self.channel_weights.to(focal_loss_raw.device).view(1, -1, 1, 1, 1)
                focal_loss_weighted = focal_loss_raw * weights
                focal_loss = focal_loss_weighted.mean() * (len(self.channel_weights) / self.channel_weights.sum())
            else:
                focal_loss = focal_loss_raw.mean()
        else:
            focal_loss = 0
        
        result = self.weight_focal * focal_loss + self.weight_dice * dc_loss
        return result


class CombinedLoss(nn.Module):
    """
    Combined Dice + CrossEntropy Loss (LEGACY - use DC_and_CE_loss for nnUNet compatibility)
    Often works better than either alone for segmentation
    """
    def __init__(self, dice_weight=0.5, ce_weight=0.5, focal_weight=0.0, 
                 use_focal=False, ignore_index=None, class_weights=None, ce_class_weights=None,
                 include_background=True):
        """
        Args:
            dice_weight: Weight for Dice loss
            ce_weight: Weight for CrossEntropy loss
            focal_weight: Weight for Focal loss (if use_focal=True)
            use_focal: Use Focal loss instead of CE
            ignore_index: Index to ignore in loss calculation
            class_weights: Weights for Dice loss classes (e.g., [1.0, 3.0, 9.0])
            ce_class_weights: Weights for CE loss classes (can be different from Dice weights)
                             If None, uses class_weights
            include_background: Whether to include background (class 0) in Dice calculation
        """
        super(CombinedLoss, self).__init__()
        self.dice_weight = dice_weight
        self.ce_weight = ce_weight
        self.focal_weight = focal_weight
        self.use_focal = use_focal
        
        # Use separate weights for CE if provided, otherwise use same as Dice
        if ce_class_weights is None:
            ce_class_weights = class_weights
        
        self.dice_loss = DiceLoss(ignore_index=ignore_index, class_weights=class_weights,
                                 include_background=include_background)
        
        if use_focal:
            self.focal_loss = FocalLoss(ignore_index=ignore_index)
        else:
            # CrossEntropyLoss requires ignore_index to be int, use -100 as default
            ce_ignore_index = ignore_index if ignore_index is not None else -100
            self.ce_loss = nn.CrossEntropyLoss(ignore_index=ce_ignore_index, 
                                             weight=ce_class_weights)
    
    def forward(self, inputs, targets):
        """
        Args:
            inputs: [B, C, D, H, W] - predicted logits  
            targets: [B, D, H, W] - ground truth labels
        Returns:
            combined_loss: scalar tensor
        """
        dice = self.dice_loss(inputs, targets)
        
        if self.use_focal:
            focal = self.focal_loss(inputs, targets)
            return (self.dice_weight * dice + 
                   self.focal_weight * focal)
        else:
            ce = self.ce_loss(inputs, targets.long())
            combined = (self.dice_weight * dice + self.ce_weight * ce)
            
            # Debug: Check for NaN
            if torch.isnan(combined):
                print(f"  [LOSS DEBUG] Combined loss is NaN! Dice={dice.item():.4f}, CE={ce.item():.4f}")
            
            return combined


class TverskyLoss(nn.Module):
    """
    Tversky Loss - Generalization of Dice Loss
    Can be tuned to focus more on precision or recall
    alpha > 0.5: focuses on recall (good for small structures)
    alpha < 0.5: focuses on precision
    """
    def __init__(self, alpha=0.7, smooth=1e-5, ignore_index=None):
        super(TverskyLoss, self).__init__()
        self.alpha = alpha
        self.beta = 1 - alpha
        self.smooth = smooth
        self.ignore_index = ignore_index

    def forward(self, inputs, targets):
        """
        Args:
            inputs: [B, C, D, H, W] - predicted logits
            targets: [B, D, H, W] - ground truth labels
        Returns:
            tversky_loss: scalar tensor
        """
        # Convert logits to probabilities
        inputs = F.softmax(inputs, dim=1)
        
        # Convert targets to one-hot
        num_classes = inputs.size(1)
        targets_one_hot = F.one_hot(targets.long(), num_classes=num_classes)
        targets_one_hot = targets_one_hot.permute(0, 4, 1, 2, 3).float()
        
        # Handle ignore_index
        if self.ignore_index is not None:
            mask = (targets != self.ignore_index).float()
            mask = mask.unsqueeze(1).expand_as(inputs)
            inputs = inputs * mask
            targets_one_hot = targets_one_hot * mask
        
        # Flatten
        inputs_flat = inputs.view(inputs.size(0), inputs.size(1), -1)
        targets_flat = targets_one_hot.view(targets_one_hot.size(0), targets_one_hot.size(1), -1)
        
        # Calculate Tversky components
        true_pos = (inputs_flat * targets_flat).sum(dim=2)  # [B, C]
        false_neg = (targets_flat * (1 - inputs_flat)).sum(dim=2)  # [B, C]
        false_pos = ((1 - targets_flat) * inputs_flat).sum(dim=2)  # [B, C]
        
        # Tversky coefficient
        tversky = (true_pos + self.smooth) / (true_pos + 
                                             self.alpha * false_neg + 
                                             self.beta * false_pos + 
                                             self.smooth)
        
        tversky_loss = 1 - tversky.mean()
        
        return tversky_loss


# Factory functions for easy model creation
def get_segmentation_criterion(loss_type='combined', **kwargs):
    """
    Factory function to create segmentation losses
    
    Args:
        loss_type: str, one of ['dice', 'iou', 'focal', 'combined', 'tversky', 'ce']
        **kwargs: Additional arguments for the loss function
        
    Returns:
        loss_fn: The requested loss function
    """
    if loss_type == 'dice':
        return DiceLoss(**kwargs)
    elif loss_type == 'iou':
        return IoULoss(**kwargs)
    elif loss_type == 'focal':
        return FocalLoss(**kwargs)
    elif loss_type == 'combined':
        return CombinedLoss(**kwargs)
    elif loss_type == 'tversky':
        return TverskyLoss(**kwargs)
    elif loss_type == 'ce':
        return nn.CrossEntropyLoss(**kwargs)
    else:
        raise ValueError(f"Unknown loss type: {loss_type}")


# Recommended configurations for PIMED dataset
def get_pimed_criterion(variant='balanced'):
    """
    Get recommended loss configurations for PIMED prostate segmentation
    
    Args:
        variant: str, one of:
            - 'nnunet': EXACT nnUNet DC_and_CE_loss (50/50, batch_dice=False, do_bg=False) - RECOMMENDED FOR REGION-BASED
            - 'balanced': 70% Dice + 30% CE, includes background (default)
            - 'dice_heavy': 90% Dice + 10% CE, includes background
            - 'focal': 60% Dice + 40% Focal loss
            - 'tversky': Tversky loss with alpha=0.7 (focus on recall)
            - 'foreground_focused': 80% Dice + 20% CE, Dice excludes background (RECOMMENDED for classes 1&2)
            - 'transition_focused': 85% Dice + 15% CE, very high weight on class 2 (rare transition zone)
        
    Returns:
        criterion: Configured loss function
    """
    # Full class weights for all 3 classes (used by CE)
    # [background, peripheral zone, transition zone]
    full_class_weights = torch.tensor([1.0, 10.0, 30.0], dtype=torch.float32)
    
    if variant == 'nnunet':
        # EXACT nnUNet configuration for region-based training
        # Uses BCE (Binary Cross Entropy) with sigmoid, not CE with softmax
        # - 50/50 Dice/BCE split (weight_bce=1, weight_dice=1)
        # - batch_dice=False (per-sample Dice, default for 3d_fullres)
        # - do_bg=True (include background/first region channel in Dice)
        # - smooth=1e-5 (nnUNet standard)
        print("===> Using EXACT nnUNet DC_and_BCE_loss (50/50, batch_dice=False, do_bg=True)")
        print("     Region-based training with SIGMOID activation")
        return DC_and_BCE_loss(
            weight_bce=1.0,
            weight_dice=1.0,
            ignore_label=None,
            batch_dice=False,  # nnUNet 3d_fullres default
            do_bg=True,        # Include first region channel
            smooth=1e-5        # nnUNet standard
        )
    
    if variant == 'nnunet_foreground':
        # nnUNet with CHANNEL WEIGHTS for foreground-focused training
        # Addresses class imbalance in region-based training:
        # - Channel 0 (prostate): ~5% of image, weight=1.0
        # - Channel 1 (cancer): ~0.5% of image, weight=3.0 (3x more important)
        # - Channel 2 (csPCa): ~0.3% of image, weight=5.0 (5x more important)
        # This prevents model from collapsing to all-zeros (trivial solution)
        print("===> Using nnUNet DC_and_BCE_loss with FOREGROUND WEIGHTS [1.0, 3.0, 5.0]")
        print("     Higher weight on cancer regions to prevent collapse")
        return DC_and_BCE_loss(
            weight_bce=1.0,
            weight_dice=1.0,
            ignore_label=None,
            batch_dice=False,
            do_bg=True,
            smooth=1e-5,
            channel_weights=[1.0, 3.0, 5.0]  # Focus on cancer regions
        )
    
    if variant == 'nnunet_pos_weight':
        # nnUNet with PIXEL-LEVEL pos_weight to address foreground/background imbalance
        # This is the KEY fix for model collapse to all-zeros!
        # 
        # Problem: BCE at pixel level treats foreground (y=1) and background (y=0) equally,
        # but background is 95%+ of pixels, so model learns to predict all zeros.
        # 
        # Solution: pos_weight multiplies the loss for foreground pixels (where target=1)
        # BCE becomes: -pos_weight * y * log(sigmoid(x)) - (1-y) * log(1-sigmoid(x))
        # 
        # NOTE: Values like [20, 100, 100] cause gradient explosion when model
        # predicts high values for background pixels (log(1-0.999) explodes).
        # Use MODERATE values to balance foreground emphasis vs stability:
        # - Channel 0 (prostate): ~5% foreground -> pos_weight = 1 (largest region, least emphasis)
        # - Channel 1 (PCa): ~0.3% foreground -> pos_weight = 5 (moderate emphasis)
        # - Channel 2 (csPCa): ~0.2% foreground -> pos_weight = 10 (smallest, most emphasis)
        # 
        # ALSO use channel_weights to emphasize cancer channels at LOSS level:
        # - Channel 0 (prostate): weight = 1.0
        # - Channel 1 (PCa): weight = 3.0 (3x more important)
        # - Channel 2 (csPCa): weight = 5.0 (5x more important)
        print("===> Using nnUNet DC_and_BCE_loss with pos_weight [1, 5, 10] + channel_weights [1, 3, 5]")
        print("     Both pixel-level (pos_weight) and channel-level (channel_weights) weighting")
        return DC_and_BCE_loss(
            weight_bce=1.0,
            weight_dice=1.0,
            ignore_label=None,
            batch_dice=False,
            do_bg=True,
            smooth=1e-5,
            pos_weight=[1.0, 5.0, 10.0],       # Pixel-level: emphasize foreground pixels
            channel_weights=[1.0, 3.0, 5.0]    # Channel-level: emphasize cancer channels in Dice & BCE
        )
    
    if variant == 'brats':
        # BraTS2023 (Dataset137) region-based training
        # Regions: [WT (whole_tumor), ET (enhancing_tumor), TC (tumor_core)]
        #
        # Foreground ratios (measured from 50 training cases):
        #   WT ~5.7% of brain voxels (inverse ~17x)
        #   ET ~1.6% of brain voxels (inverse ~64x)
        #   TC ~0.5% of brain voxels (inverse ~220x)
        #
        # pos_weight: pixel-level emphasis on foreground (penalizes false negatives).
        # Analysis of 251 test cases shows systematic under-prediction:
        #   WT: recall=0.938, FP/FN=0.57 → mild under-prediction
        #   ET: recall=0.914, FP/FN=0.55 → moderate under-prediction
        #   TC: recall=0.811, FP/FN=0.48 → severe under-prediction (only 81% of TC voxels found)
        # Key failure: 11.3% of TC voxels predicted as edema (WT-only), 5.8% as ET
        # → model activates WT channel correctly but fails to activate ET+TC channels
        #
        # pos_weight boosts loss for missed foreground pixels (false negatives):
        #   WT=3  (5.7% fg, mild boost)
        #   ET=7  (1.6% fg, moderate boost - 6.5% of ET voxels misclassified)
        #   TC=15 (0.5% fg, strong boost - 18.9% of TC voxels missed)
        #
        # channel_weights: channel-level emphasis in both Dice and BCE.
        #   [1, 3, 5] - stronger emphasis on TC (was [1,2,3] but TC needs more)
        #   TC gets 5x weight because it's the hardest region (Dice=0.8526 vs WT=0.9507)
        print("===> Using nnUNet DC_and_BCE_loss (BraTS) with pos_weight [3, 7, 15] + channel_weights [1, 3, 5]")
        print("     BraTS regions: 3 channels [WT, ET, TC]")
        print("     pos_weight penalises false negatives (missing tumor voxels)")
        print("     channel_weights: TC=5x, ET=3x emphasis (TC is bottleneck region)")
        return DC_and_BCE_loss(
            weight_bce=1.0,
            weight_dice=1.0,
            ignore_label=None,
            batch_dice=False,
            do_bg=True,           # Include WT (first region) in Dice
            smooth=1e-5,
            pos_weight=[3.0, 7.0, 15.0],     # Pixel-level: aggressively penalise missed TC/ET
            channel_weights=[1.0, 3.0, 5.0]  # Channel-level: TC=5x, ET=3x (TC is bottleneck)
        )
    
    if variant == 'nnunet_pos_weight_binary':
        # Binary (2-channel) variant: prostate + cancer (PCa+csPCa combined)
        # For pred_type='binary' training where we only have 2 output channels:
        # - Channel 0 (prostate): ~5% foreground -> pos_weight = 1, channel_weight = 1
        # - Channel 1 (cancer): ~0.3% foreground -> pos_weight = 10, channel_weight = 5
        print("===> Using nnUNet DC_and_BCE_loss (BINARY) with pos_weight [1, 10] + channel_weights [1, 5]")
        print("     Binary mode: 2 channels [prostate, cancer (PCa+csPCa combined)]")
        return DC_and_BCE_loss(
            weight_bce=1.0,
            weight_dice=1.0,
            ignore_label=None,
            batch_dice=False,
            do_bg=True,
            smooth=1e-5,
            pos_weight=[1.0, 10.0],       # Pixel-level: heavier emphasis on cancer foreground
            channel_weights=[1.0, 5.0]    # Channel-level: emphasize cancer channel
        )
    
    if variant == 'focal_binary':
        # Binary (2-channel) with FOCAL LOSS instead of BCE
        # Focal loss down-weights easy examples (confident correct predictions)
        # This is especially useful when most pixels are easy background
        # 
        # gamma=2 is standard: easy examples (p_t=0.9) get 100× less weight
        # pos_weight still used for foreground emphasis
        # channel_weights for channel-level importance
        print("===> Using Dice + Binary Focal Loss (BINARY)")
        print("     Focal Loss: gamma=2 down-weights easy examples by (1-p_t)^2")
        print("     Binary mode: 2 channels [prostate, cancer]")
        print("     pos_weight [1, 10], channel_weights [1, 5]")
        return DC_and_Focal_loss(
            weight_focal=1.0,
            weight_dice=1.0,
            gamma=2.0,
            pos_weight=[1.0, 10.0],
            channel_weights=[1.0, 1.0],
            batch_dice=False,
            do_bg=True,
            smooth=1e-5
        )
    
    if variant == 'focal_binary_mild':
        # MILD focal: Balanced between reducing FP and maintaining TP
        # Use this for larger models that may overfit with sharp focal
        print("===> Using Dice + Binary Focal Loss (MILD - balanced)")
        print("     Focal Loss: gamma=2.0 (standard down-weighting)")
        print("     weight_focal=1.0, weight_dice=1.0 (balanced)")
        print("     Binary mode: 2 channels [prostate, cancer]")
        print("     pos_weight [1, 10], channel_weights [1, 1.5]")
        return DC_and_Focal_loss(
            weight_focal=1.0,           # Balanced with dice (not 2.0)
            weight_dice=1.0,            # Equal weight
            gamma=2.0,                  # Standard gamma (not 1.5)
            pos_weight=[1.0, 10.0],     # Still emphasize cancer foreground
            channel_weights=[1.0, 1.5], # Slight extra weight on cancer (not 2.0)
            batch_dice=False,
            do_bg=True,
            smooth=1e-5
        )
    
    if variant == 'focal_binary_sharp':
        # IMPROVED: Binary focal with SHARPER predictions on negative cases
        # Key changes:
        # 1. weight_focal=2.0 vs weight_dice=1.0: More emphasis on focal loss
        #    - Focal loss penalizes ALL false positives, Dice doesn't on empty GT
        # 2. gamma=1.5 (lower than 2.0): Less down-weighting of "easy" negatives
        #    - At gamma=2: pt=0.9 gets 0.01 weight (nearly ignored)
        #    - At gamma=1.5: pt=0.9 gets 0.03 weight (3x more gradient)
        # 3. neg_weight added: Extra penalty for false positives (predicting cancer on non-cancer)
        print("===> Using Dice + Binary Focal Loss (SHARP - improved for negative cases)")
        print("     Focal Loss: gamma=1.5 (less down-weighting of easy negatives)")
        print("     weight_focal=2.0, weight_dice=1.0 (2:1 focal emphasis)")
        print("     Binary mode: 2 channels [prostate, cancer]")
        print("     pos_weight [1, 10], channel_weights [1, 2]")
        return DC_and_Focal_loss(
            weight_focal=2.0,           # Increased from 1.0 - more emphasis on focal
            weight_dice=1.0,            # Keep dice for overlap
            gamma=1.5,                  # Reduced from 2.0 - less down-weighting of negatives
            pos_weight=[1.0, 10.0],     # Still emphasize cancer foreground
            channel_weights=[1.0, 2.0], # More weight on cancer channel
            batch_dice=False,
            do_bg=True,
            smooth=1e-5
        )
    
    if variant == 'focal_binary_high_sensitivity':
        # HIGH SENSITIVITY: Maximize cancer detection while keeping reasonable FP
        # Key changes vs focal_binary:
        # 1. pos_weight [1, 20]: 2× more penalty for missing cancer voxels (FN)
        #    This pushes the model to predict cancer more aggressively
        # 2. channel_weights [1, 3]: 3× more weight on cancer channel overall
        # 3. weight_dice=1.5: More Dice emphasis to improve overlap with GT cancer
        # 4. gamma=2.0: Standard focal to still down-weight easy background
        # Use with feature masking to keep FP low via masked KD while
        # the loss drives higher sensitivity
        print("===> Using Dice + Binary Focal Loss (HIGH SENSITIVITY)")
        print("     Focal Loss: gamma=2.0 (standard)")
        print("     weight_focal=1.0, weight_dice=1.5 (1.5:1 dice emphasis for overlap)")
        print("     Binary mode: 2 channels [prostate, cancer]")
        print("     pos_weight [1, 30] (3x cancer FN penalty), channel_weights [1, 3]")
        return DC_and_Focal_loss(
            weight_focal=1.0,           # Standard focal
            weight_dice=1.5,            # More dice for overlap
            gamma=2.0,                  # Standard gamma
            pos_weight=[1.0, 30.0],     # 3× more FN penalty on cancer (was 20)
            channel_weights=[1.0, 3.0], # 3× cancer channel weight (was 1)
            batch_dice=False,
            do_bg=True,
            smooth=1e-5
        )
    
    if variant == 'focal_3class':
        # 3-class with FOCAL LOSS
        print("===> Using Dice + Binary Focal Loss (3-class)")
        print("     Focal Loss: gamma=2 down-weights easy examples")
        print("     3-class mode: 3 channels [prostate, PCa, csPCa]")
        print("     pos_weight [1, 5, 10], channel_weights [1, 3, 5]")
        return DC_and_Focal_loss(
            weight_focal=1.0,
            weight_dice=1.0,
            gamma=2.0,
            pos_weight=[1.0, 5.0, 10.0],
            channel_weights=[1.0, 3.0, 5.0],
            batch_dice=False,
            do_bg=True,
            smooth=1e-5
        )
    
    if variant == 'balanced':
        # Balanced Dice + CE - good starting point, includes background
        return CombinedLoss(dice_weight=0.5, ce_weight=0.5, 
                          class_weights=full_class_weights,
                          include_background=True)
    
    elif variant == 'dice_heavy':
        # Focus on Dice for overlap - best for small structures
        return CombinedLoss(dice_weight=0.9, ce_weight=0.1, 
                          class_weights=full_class_weights,
                          include_background=True)
    
    elif variant == 'focal':
        # Use focal loss for hard examples
        return CombinedLoss(dice_weight=0.6, focal_weight=0.4, 
                          use_focal=True, class_weights=full_class_weights,
                          include_background=True)
    
    elif variant == 'tversky':
        # Tversky with focus on recall (alpha=0.7)
        return TverskyLoss(alpha=0.7)
    
    elif variant == 'foreground_focused':
        # RECOMMENDED: Dice focuses only on foreground classes (peripheral & transition zones)
        # Dice weights: [3.0, 9.0] for classes [1, 2] only (background excluded)
        # CE weights: [1.0, 3.0, 9.0] for all classes (provides overall guidance)
        # 70/30 split balances Dice overlap with CE stability
        foreground_weights = torch.tensor([10.0, 30.0], dtype=torch.float32)
        return CombinedLoss(dice_weight=0.7, ce_weight=0.3,
                          class_weights=foreground_weights,
                          ce_class_weights=full_class_weights,
                          include_background=False)
    
    elif variant == 'transition_focused':
        # ALTERNATIVE: Even higher focus on rare class 2 (transition zone)
        # Dice weights: [2.0, 15.0] - transition zone gets 7.5x more than peripheral
        # CE weights: [1.0, 2.0, 15.0] - matching emphasis
        # 85/15 split heavily favors Dice overlap for the rare structure
        foreground_weights_high = torch.tensor([2.0, 15.0], dtype=torch.float32)
        ce_weights_high = torch.tensor([1.0, 2.0, 15.0], dtype=torch.float32)
        return CombinedLoss(dice_weight=0.85, ce_weight=0.15,
                          class_weights=foreground_weights_high,
                          ce_class_weights=ce_weights_high,
                          include_background=False)
    
    else:
        raise ValueError(f"Unknown variant: {variant}")


class ChannelNorm3D(nn.Module):
    """
    Enhanced ChannelNorm that works with both 2D and 3D features
    Auto-detects input dimensions and normalizes appropriately
    """
    def __init__(self):
        super(ChannelNorm3D, self).__init__()
    
    def forward(self, featmap):
        if featmap.dim() == 4:
            # 2D case: (n, c, h, w)
            n, c, h, w = featmap.shape
            featmap = featmap.reshape((n, c, -1))
        elif featmap.dim() == 5:
            # 3D case: (n, c, d, h, w)
            n, c, d, h, w = featmap.shape
            featmap = featmap.reshape((n, c, -1))
        else:
            raise ValueError(f"Unsupported feature map dimensions: {featmap.dim()}")
        
        featmap = featmap.softmax(dim=-1)
        return featmap


class FeatureKLLoss3D(nn.Module):
    """
    Enhanced FeatureKLLoss that works with both 2D and 3D features
    Compatible with existing FeatureKLLoss but handles 3D medical images
    """
    def __init__(self, temperature=4.0):
        super(FeatureKLLoss3D, self).__init__()
        self.normalize = ChannelNorm3D()
        self.criterion = nn.KLDivLoss(reduction='none')
        self.temperature = temperature
       
    def forward(self, f_s, f_t):
        # Normalize with temperature
        norm_s = self.normalize(f_s / self.temperature)
        norm_t = self.normalize(f_t.detach() / self.temperature)
        norm_s = norm_s.log()

        # Compute KL divergence
        loss = self.criterion(norm_s, norm_t).sum(-1).mean(-1)

        return loss * (self.temperature ** 2)


class FeatureMSELoss3D(nn.Module):
    """
    Enhanced FeatureMSELoss that works with both 2D and 3D features
    Same functionality as original but with explicit 3D support
    """
    def __init__(self):
        super(FeatureMSELoss3D, self).__init__()

    def forward(self, f_s, f_t):
        # Flatten to [B, -1] - works for any number of dimensions
        f_s = f_s.view(f_s.size(0), -1)
        f_t = f_t.view(f_t.size(0), -1)
        loss = ((f_s - f_t) ** 2).mean(1)
        return loss


if __name__ == '__main__':
    """Test the segmentation losses with dummy data"""
    
    print("🧪 Testing 3D Segmentation Losses")
    print("=" * 50)
    
    # Create dummy data matching PIMED format
    batch_size = 2
    num_classes = 2  # background + prostate
    depth = 20
    height = 256
    width = 256
    
    # Dummy logits and targets
    logits = torch.randn(batch_size, num_classes, depth, height, width)
    targets = torch.randint(0, num_classes, (batch_size, depth, height, width))
    
    print(f"Input logits shape: {logits.shape}")
    print(f"Target labels shape: {targets.shape}")
    
    # Test all loss functions
    losses_to_test = {
        'DiceLoss': DiceLoss(),
        'IoULoss': IoULoss(),
        'FocalLoss': FocalLoss(),
        'CombinedLoss': CombinedLoss(),
        'TverskyLoss': TverskyLoss(),
        'CrossEntropyLoss': nn.CrossEntropyLoss()
    }
    
    print("\n📊 Loss Function Results:")
    for name, loss_fn in losses_to_test.items():
        try:
            loss_value = loss_fn(logits, targets)
            print(f"✅ {name:15s}: {loss_value.item():.4f}")
        except Exception as e:
            print(f"❌ {name:15s}: ERROR - {str(e)[:50]}...")
    
    # Test PIMED configurations
    print("\n🏥 PIMED Configuration Tests:")
    pimed_variants = ['balanced', 'dice_heavy', 'focal', 'tversky']
    
    for variant in pimed_variants:
        try:
            criterion = get_pimed_criterion(variant)
            loss_value = criterion(logits, targets)
            print(f"✅ PIMED {variant:10s}: {loss_value.item():.4f}")
        except Exception as e:
            print(f"❌ PIMED {variant:10s}: ERROR - {str(e)[:50]}...")
    
    print(f"\n🎉 All segmentation losses are ready for PIMED training!")
    print(f"💡 Recommended: Use 'get_pimed_criterion(\"balanced\")' as starting point")


class DeepSupervisionLoss(nn.Module):
    """
    Deep Supervision Loss Wrapper (nnUNet-style)
    
    When a model outputs multiple predictions at different scales (deep supervision),
    this wrapper computes weighted losses across all scales following nnUNet's scheme:
    
    Weight Calculation (for N outputs):
    1. Start with exponentially decreasing: [1.0, 0.5, 0.25, 0.125, ...]
    2. Set last weight to 0 (don't use deepest/coarsest output)
    3. Normalize remaining weights to sum to 1.0
    
    Example for 6 outputs (7-stage UNet):
    - Raw: [1.0, 0.5, 0.25, 0.125, 0.0625, 0.03125]
    - Set last to 0: [1.0, 0.5, 0.25, 0.125, 0.0625, 0.0]
    - Normalized: [0.5161, 0.2581, 0.1290, 0.0645, 0.0323, 0.0]
    
    This means:
    - Highest resolution (first output): 51.61% weight
    - Each lower resolution gets half the previous weight
    - Lowest resolution (last output): NOT USED (0% weight)
    
    Args:
        base_loss: The base loss function to apply at each scale
        weights: Optional custom weights for each scale (default: nnUNet-style)
        
    Example:
        >>> base_criterion = get_pimed_criterion('foreground_focused')
        >>> ds_criterion = DeepSupervisionLoss(base_criterion)
        >>> 
        >>> # Model with deep supervision returns list of predictions
        >>> outputs = model(x)  # [pred_full, pred_half, pred_quarter, ...]
        >>> loss = ds_criterion(outputs, targets)
    """
    def __init__(self, base_loss, weights=None):
        super(DeepSupervisionLoss, self).__init__()
        self.base_loss = base_loss
        self.weights = weights
    
    def forward(self, outputs, targets):
        """
        Args:
            outputs: list of [B, C, D, H, W] tensors at different scales
                    OR single [B, C, D, H, W] tensor (for compatibility)
            targets: [B, D, H, W] ground truth labels
        
        Returns:
            weighted_loss: scalar tensor
        """
        # Handle case where model doesn't use deep supervision (single output)
        if not isinstance(outputs, (list, tuple)):
            return self.base_loss(outputs, targets)
        
        # Compute weights if not provided (nnUNet scheme)
        if self.weights is None:
            num_outputs = len(outputs)
            
            # Step 1: Exponentially decreasing weights
            weights = torch.tensor([0.5 ** i for i in range(num_outputs)], 
                                  device=outputs[0].device, dtype=torch.float32)
            
            # Step 2: Set last weight to 0 (don't use deepest/coarsest output)
            weights[-1] = 0.0
            
            # Step 3: Normalize so they sum to 1
            weights = weights / weights.sum()
        else:
            weights = torch.tensor(self.weights, device=outputs[0].device, dtype=torch.float32)
        
        # Compute loss at each scale
        total_loss = 0.0
        for i, output in enumerate(outputs):
            # Skip if weight is 0 (saves computation)
            if weights[i] == 0.0:
                continue
                
            # Downsample targets to match output resolution if needed
            if output.shape[2:] != targets.shape[1:]:
                # Targets need downsampling
                target_resized = F.interpolate(
                    targets.float().unsqueeze(1),  # [B, 1, D, H, W]
                    size=output.shape[2:],
                    mode='nearest'
                ).squeeze(1).long()  # [B, D', H', W']
            else:
                target_resized = targets
            
            # Compute loss at this scale
            loss_at_scale = self.base_loss(output, target_resized)
            total_loss += weights[i] * loss_at_scale
        
        return total_loss


def get_pimed_criterion_with_deep_supervision(variant='foreground_focused', weights=None):
    """
    Get PIMED segmentation loss with deep supervision support.
    
    Args:
        variant: Loss variant (same as get_pimed_criterion)
        weights: Optional custom weights for deep supervision scales
    
    Returns:
        DeepSupervisionLoss wrapping the base criterion
    
    Example:
        >>> criterion = get_pimed_criterion_with_deep_supervision('foreground_focused')
        >>> # Works with both single output and multi-scale outputs
        >>> loss = criterion(model_output, targets)
    """
    base_criterion = get_pimed_criterion(variant)
    return DeepSupervisionLoss(base_criterion, weights=weights)
