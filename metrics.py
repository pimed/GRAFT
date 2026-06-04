"""
Evaluation Metrics for MTKD-RL Framework

Supports both:
- 2D Classification metrics (CIFAR-100)  
- 3D Segmentation metrics (PIMED medical imaging)
"""

import torch
import torch.nn.functional as F


def correct_num(output, target, topk=(1,)):
    """
    Computes the accuracy over the k top predictions for classification tasks
    
    Args:
        output: [B, num_classes] - predicted logits
        target: [B] - ground truth class labels
        topk: tuple of ints - compute accuracy for top-k predictions
        
    Returns:
        list of correct predictions for each k
    """
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)

        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))

        res = []
        for k in topk:
            correct_k = correct[:k].contiguous().view(-1).float().sum(0, keepdim=True)
            res.append(correct_k)
        return res


def dice_score(pred_logits, targets, num_classes=2, smooth=1.0, include_background=True, regions=None, use_sigmoid=False):
    """
    Compute Dice score for segmentation evaluation (nnUNet-compatible)
    
    Args:
        pred_logits: [B, C, D, H, W] - predicted logits
        targets: [B, D, H, W] or [B, C, D, H, W] - ground truth labels (class indices or multi-hot for regions)
        num_classes: number of classes
        smooth: smoothing factor (DEPRECATED - not used in nnUNet computation)
        include_background: whether to include class 0 (background) in the average
        regions: dict mapping region names to list of label indices
                 e.g., {'foreground': [1,2,3], 'cancer': [2,3], 'csPCa': [3]}
                 If None, computes per-class Dice
        use_sigmoid: if True, use sigmoid activation for region-based (BCE training)
                     if False, use softmax activation for class-based (CE training)
        
    Returns:
        dice: scalar tensor - mean Dice score
        
    Note:
        nnUNet uses region-based evaluation:
        - Region (1,2,3): All prostate tissue
        - Region (2,3): Cancer regions (PCa)  
        - Region 3: Clinically significant cancer (csPCa)
        
        nnUNet Dice formula (NO smooth factor):
        Dice = 2*TP / (2*TP + FP + FN)
    """
    if use_sigmoid:
        # Region-based with sigmoid (TRUE nnUNet regions)
        pred_probs = torch.sigmoid(pred_logits)  # [B, C, D, H, W]
        # Threshold at 0.5 to get binary predictions
        pred_binary = (pred_probs > 0.5).float()  # [B, C, D, H, W]
    else:
        # Class-based with softmax
        pred_probs = F.softmax(pred_logits, dim=1)  # [B, C, D, H, W]
        pred_labels = torch.argmax(pred_probs, dim=1)  # [B, D, H, W]
    
    dice_scores = []
    
    if regions is not None and use_sigmoid:
        # TRUE region-based Dice (multi-hot, used with sigmoid/BCE)
        # targets should already be multi-hot [B, C, D, H, W]
        # Use nnUNet's TP/FP/FN computation (no smooth factor)
        for region_idx, (region_name, label_list) in enumerate(regions.items()):
            # pred_binary[:, region_idx] is the binary prediction for this region
            pred_mask = pred_binary[:, region_idx, ...]  # [B, D, H, W]
            true_mask = targets[:, region_idx, ...].float() if targets.ndim == 5 else targets[:, region_idx, ...].float()  # [B, D, H, W]
            
            # nnUNet formula: TP, FP, FN computation
            # axes = [0] + list(range(2, len(pred_mask.shape))) would be batch+spatial
            # But we sum over spatial only, then average over batch
            axes = list(range(1, len(pred_mask.shape)))  # spatial dimensions only
            
            tp = (pred_mask * true_mask).sum(dim=axes)  # [B]
            fp = (pred_mask * (1 - true_mask)).sum(dim=axes)  # [B]
            fn = ((1 - pred_mask) * true_mask).sum(dim=axes)  # [B]
            
            # nnUNet Dice formula: 2*TP / (2*TP + FP + FN)
            dice = (2. * tp) / (2. * tp + fp + fn + 1e-8)  # [B], add small epsilon to avoid division by zero
            dice_scores.append(dice.mean())  # Average across batch for this region
    
    elif regions is not None:
        # Region-based Dice with softmax (combine class predictions into regions)
        for region_name, label_list in regions.items():
            # Create binary masks for this region (any of the labels in label_list)
            pred_mask = torch.zeros_like(pred_labels, dtype=torch.float32)
            true_mask = torch.zeros_like(targets, dtype=torch.float32)
            
            for label in label_list:
                pred_mask = torch.maximum(pred_mask, (pred_labels == label).float())
                true_mask = torch.maximum(true_mask, (targets == label).float())
            
            # nnUNet formula: TP, FP, FN computation
            axes = list(range(1, len(pred_mask.shape)))  # spatial dimensions only
            
            tp = (pred_mask * true_mask).sum(dim=axes)  # [B]
            fp = (pred_mask * (1 - true_mask)).sum(dim=axes)  # [B]
            fn = ((1 - pred_mask) * true_mask).sum(dim=axes)  # [B]
            
            # nnUNet Dice formula: 2*TP / (2*TP + FP + FN)
            dice = (2. * tp) / (2. * tp + fp + fn + 1e-8)  # [B]
            dice_scores.append(dice.mean())  # Average across batch for this region
    else:
        # Per-class Dice (original behavior)
        start_class = 0 if include_background else 1
        
        for class_idx in range(start_class, num_classes):
            # Binary masks for current class
            pred_mask = (pred_labels == class_idx).float()  # [B, D, H, W]
            true_mask = (targets == class_idx).float()      # [B, D, H, W]
            
            # nnUNet formula: TP, FP, FN computation
            axes = list(range(1, len(pred_mask.shape)))  # spatial dimensions only
            
            tp = (pred_mask * true_mask).sum(dim=axes)  # [B]
            fp = (pred_mask * (1 - true_mask)).sum(dim=axes)  # [B]
            fn = ((1 - pred_mask) * true_mask).sum(dim=axes)  # [B]
            
            # nnUNet Dice formula: 2*TP / (2*TP + FP + FN)
            dice = (2. * tp) / (2. * tp + fp + fn + 1e-8)  # [B]
            dice_scores.append(dice.mean())  # Average across batch for this class
    
    # Return mean across regions/classes
    return torch.stack(dice_scores).mean()


def iou_score(pred_logits, targets, num_classes=2, smooth=1.0, include_background=True, regions=None, use_sigmoid=False):
    """
    Compute IoU (Jaccard) score for segmentation evaluation (nnUNet-compatible)
    
    Args:
        pred_logits: [B, C, D, H, W] - predicted logits
        targets: [B, D, H, W] or [B, C, D, H, W] - ground truth labels  
        num_classes: number of classes
        smooth: smoothing factor (DEPRECATED - using TP/FP/FN formula like nnUNet)
        include_background: whether to include background (class 0) in average
        regions: dict mapping region names to list of label indices
                 If None, computes per-class IoU
        use_sigmoid: if True, use sigmoid activation for region-based (BCE training)
                     if False, use softmax activation for class-based (CE training)
        
    Returns:
        iou: scalar tensor - mean IoU score
        
    Note:
        IoU = TP / (TP + FP + FN)
    """
    if use_sigmoid:
        # Region-based with sigmoid (TRUE nnUNet regions)
        pred_probs = torch.sigmoid(pred_logits)  # [B, C, D, H, W]
        # Threshold at 0.5 to get binary predictions
        pred_binary = (pred_probs > 0.5).float()  # [B, C, D, H, W]
    else:
        # Class-based with softmax
        pred_probs = F.softmax(pred_logits, dim=1)  # [B, C, D, H, W]
        pred_labels = torch.argmax(pred_probs, dim=1)  # [B, D, H, W]
    
    iou_scores = []
    
    if regions is not None and use_sigmoid:
        # TRUE region-based IoU (multi-hot, used with sigmoid/BCE)
        for region_idx, (region_name, label_list) in enumerate(regions.items()):
            pred_mask = pred_binary[:, region_idx, ...]  # [B, D, H, W]
            true_mask = targets[:, region_idx, ...].float() if targets.ndim == 5 else targets[:, region_idx, ...].float()
            
            # nnUNet-style: TP, FP, FN computation
            axes = list(range(1, len(pred_mask.shape)))  # spatial dimensions only
            
            tp = (pred_mask * true_mask).sum(dim=axes)  # [B]
            fp = (pred_mask * (1 - true_mask)).sum(dim=axes)  # [B]
            fn = ((1 - pred_mask) * true_mask).sum(dim=axes)  # [B]
            
            # IoU = TP / (TP + FP + FN)
            iou = tp / (tp + fp + fn + 1e-8)  # [B]
            iou_scores.append(iou.mean())
    
    elif regions is not None:
        # Region-based IoU (nnUNet style)
        for region_name, label_list in regions.items():
            # Create binary masks for this region
            pred_mask = torch.zeros_like(pred_labels, dtype=torch.float32)
            true_mask = torch.zeros_like(targets, dtype=torch.float32)
            
            for label in label_list:
                pred_mask = torch.maximum(pred_mask, (pred_labels == label).float())
                true_mask = torch.maximum(true_mask, (targets == label).float())
            
            # nnUNet-style: TP, FP, FN computation
            axes = list(range(1, len(pred_mask.shape)))  # spatial dimensions only
            
            tp = (pred_mask * true_mask).sum(dim=axes)  # [B]
            fp = (pred_mask * (1 - true_mask)).sum(dim=axes)  # [B]
            fn = ((1 - pred_mask) * true_mask).sum(dim=axes)  # [B]
            
            # IoU = TP / (TP + FP + FN)
            iou = tp / (tp + fp + fn + 1e-8)  # [B]
            iou_scores.append(iou.mean())
    else:
        # Per-class IoU (original behavior)
        start_class = 0 if include_background else 1
        
        for class_idx in range(start_class, num_classes):
            # Binary masks for current class
            pred_mask = (pred_labels == class_idx).float()  # [B, D, H, W]
            true_mask = (targets == class_idx).float()      # [B, D, H, W]
            
            # nnUNet-style: TP, FP, FN computation
            axes = list(range(1, len(pred_mask.shape)))  # spatial dimensions only
            
            tp = (pred_mask * true_mask).sum(dim=axes)  # [B]
            fp = (pred_mask * (1 - true_mask)).sum(dim=axes)  # [B]
            fn = ((1 - pred_mask) * true_mask).sum(dim=axes)  # [B]
            
            # IoU = TP / (TP + FP + FN)
            iou = tp / (tp + fp + fn + 1e-8)  # [B]
            iou_scores.append(iou.mean())
    
    # Return mean across regions/classes
    return torch.stack(iou_scores).mean()


def get_evaluation_metrics(pred_logits, targets, num_classes=3, regions=None):
    """
    Compute comprehensive evaluation metrics (nnUNet-compatible)
    
    Args:
        pred_logits: [B, C, D, H, W] - predicted logits
        targets: [B, D, H, W] - ground truth labels
        num_classes: number of classes
        regions: dict mapping region names to list of label indices
                 e.g., {'foreground': [1,2,3], 'cancer': [2,3], 'csPCa': [3]}
        
    Returns:
        dict with 'dice' and 'iou' scores
        
    Example:
        # Region-based evaluation (nnUNet style)
        regions = {
            'foreground': [1, 2, 3],  # All prostate
            'cancer': [2, 3],          # PCa regions
            'csPCa': [3]               # Clinically significant cancer
        }
        metrics = get_evaluation_metrics(logits, targets, num_classes=3, regions=regions)
    """
    dice = dice_score(pred_logits, targets, num_classes=num_classes, 
                     smooth=1.0, include_background=False, regions=regions)
    iou = iou_score(pred_logits, targets, num_classes=num_classes,
                   smooth=1.0, include_background=False, regions=regions)
    
    return {
        'dice': dice.item(),
        'iou': iou.item()
    }


def hausdorff_distance(pred_logits, targets, num_classes=2):
    """
    Compute Hausdorff distance for segmentation evaluation (optional advanced metric)
    
    Args:
        pred_logits: [B, C, D, H, W] - predicted logits
        targets: [B, D, H, W] - ground truth labels
        num_classes: number of classes
        
    Returns:
        hd: scalar tensor - mean Hausdorff distance
        
    Note: This is a placeholder - full implementation would require 
          distance transform computations
    """
    # Placeholder - would require scipy.ndimage or custom CUDA implementation
    # for actual distance transform computations
    return torch.tensor(0.0, device=pred_logits.device)


# Metric dispatcher for automatic selection based on task type
def get_metric_functions(task_type='classification'):
    """
    Get appropriate evaluation functions based on task type (factory function)
    
    Args:
        task_type: 'classification' or 'segmentation'
        
    Returns:
        dict: metric functions and their names
    """
    if task_type == 'classification':
        return {
            'accuracy': correct_num,
            'top1': lambda output, target: correct_num(output, target, topk=(1,))[0],
            'top5': lambda output, target: correct_num(output, target, topk=(1, 5))[1],
        }
    elif task_type == 'segmentation':
        return {
            'dice': dice_score,
            'iou': iou_score,
            'jaccard': iou_score,  # IoU and Jaccard are the same
            # 'hausdorff': hausdorff_distance,  # Optional advanced metric
        }
    else:
        raise ValueError(f"Unknown task_type: {task_type}")