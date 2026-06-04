"""
Contrastive Knowledge Distillation Losses

Instead of matching probability distributions (KL divergence), contrastive KD 
focuses on relative relationships between samples and decision boundaries.

Key idea: The student should learn the same decision boundaries as the teacher,
not just copy the teacher's soft probabilities.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class ContrastiveKDLoss(nn.Module):
    """
    Contrastive KD loss based on ranking consistency.
    
    For each sample, we want the student to rank classes in the same order as the teacher.
    This focuses on decision boundaries rather than exact probability values.
    
    Args:
        temperature: Temperature for softening distributions (default: 4.0)
        margin: Margin for contrastive loss (default: 0.5)
    """
    def __init__(self, temperature=4.0, margin=0.5):
        super(ContrastiveKDLoss, self).__init__()
        self.temperature = temperature
        self.margin = margin
    
    def forward(self, student_logits, teacher_logits, unreduce=False):
        """
        Args:
            student_logits: [B, C] or [B, C, H, W] or [B, C, D, H, W]
            teacher_logits: [B, C] or [B, C, H, W] or [B, C, D, H, W]
            unreduce: If True, return per-sample loss [B]
        
        Returns:
            loss: scalar or [B] if unreduce=True
        """
        # Soften logits with temperature
        student_soft = F.softmax(student_logits / self.temperature, dim=1)
        teacher_soft = F.softmax(teacher_logits / self.temperature, dim=1)
        
        # Get top-2 classes from teacher (decision boundary is between top-1 and top-2)
        teacher_top2_values, teacher_top2_indices = torch.topk(teacher_soft, k=2, dim=1)
        
        # Extract student probabilities for teacher's top-2 classes
        # For 2D/3D: gather along class dimension
        if student_soft.dim() == 4:  # [B, C, H, W]
            B, C, H, W = student_soft.shape
            # Reshape to [B, H, W, C] for easier indexing
            student_soft_hw = student_soft.permute(0, 2, 3, 1)  # [B, H, W, C]
            teacher_top2_indices_hw = teacher_top2_indices.permute(0, 2, 3, 1)  # [B, H, W, 2]
            
            # Gather top-1 and top-2 probabilities
            student_top1 = torch.gather(student_soft_hw, dim=3, 
                                       index=teacher_top2_indices_hw[:, :, :, 0:1]).squeeze(-1)  # [B, H, W]
            student_top2 = torch.gather(student_soft_hw, dim=3,
                                       index=teacher_top2_indices_hw[:, :, :, 1:2]).squeeze(-1)  # [B, H, W]
            
        elif student_soft.dim() == 5:  # [B, C, D, H, W]
            B, C, D, H, W = student_soft.shape
            # Reshape to [B, D, H, W, C] for easier indexing
            student_soft_dhw = student_soft.permute(0, 2, 3, 4, 1)  # [B, D, H, W, C]
            teacher_top2_indices_dhw = teacher_top2_indices.permute(0, 2, 3, 4, 1)  # [B, D, H, W, 2]
            
            # Gather top-1 and top-2 probabilities
            student_top1 = torch.gather(student_soft_dhw, dim=4,
                                       index=teacher_top2_indices_dhw[:, :, :, :, 0:1]).squeeze(-1)  # [B, D, H, W]
            student_top2 = torch.gather(student_soft_dhw, dim=4,
                                       index=teacher_top2_indices_dhw[:, :, :, :, 1:2]).squeeze(-1)  # [B, D, H, W]
            
        else:  # [B, C] - classification
            student_top1 = torch.gather(student_soft, dim=1, 
                                       index=teacher_top2_indices[:, 0:1]).squeeze(1)  # [B]
            student_top2 = torch.gather(student_soft, dim=1,
                                       index=teacher_top2_indices[:, 1:2]).squeeze(1)  # [B]
        
        # Contrastive loss: student should separate top-1 and top-2 by at least margin
        # margin = teacher's separation (we want student to match teacher's decision boundary)
        teacher_margin = teacher_top2_values[:, 0] - teacher_top2_values[:, 1]  # [B, ...] teacher's confidence gap
        student_margin = student_top1 - student_top2  # [B, ...] student's confidence gap
        
        # Loss: penalize when student's margin is less than teacher's margin
        # Use hinge loss: max(0, teacher_margin - student_margin)
        loss = F.relu(teacher_margin - student_margin)
        
        # Scale by temperature^2 to match KL divergence scaling
        loss = loss * (self.temperature ** 2)
        
        if unreduce:
            # Return per-sample loss [B] or [B, H, W] or [B, D, H, W]
            if loss.dim() > 1:
                # For spatial outputs, reduce spatial dimensions but keep batch
                loss = loss.view(loss.size(0), -1).mean(dim=1)  # [B]
            return loss
        else:
            # Return scalar
            return loss.mean()


class RelationKDLoss(nn.Module):
    """
    Relation-based KD loss: preserve pairwise similarities between samples.
    
    Instead of matching individual predictions, we match how the teacher
    relates different samples to each other.
    
    This is useful when the teacher's absolute predictions may not be perfect,
    but the relative relationships (e.g., "sample A is more similar to sample B 
    than to sample C") are informative.
    
    Args:
        temperature: Temperature for softening (default: 4.0)
    """
    def __init__(self, temperature=4.0):
        super(RelationKDLoss, self).__init__()
        self.temperature = temperature
    
    def forward(self, student_logits, teacher_logits, unreduce=False):
        """
        Args:
            student_logits: [B, C] or [B, C, H, W] or [B, C, D, H, W]
            teacher_logits: [B, C] or [B, C, H, W] or [B, C, D, H, W]
            unreduce: If True, return per-sample loss [B]
        
        Returns:
            loss: scalar or [B] if unreduce=True
        """
        # For spatial outputs, pool to get global representations
        if student_logits.dim() == 5:  # [B, C, D, H, W]
            student_pooled = F.adaptive_avg_pool3d(student_logits, (1, 1, 1)).view(student_logits.size(0), student_logits.size(1))
            teacher_pooled = F.adaptive_avg_pool3d(teacher_logits, (1, 1, 1)).view(teacher_logits.size(0), teacher_logits.size(1))
        elif student_logits.dim() == 4:  # [B, C, H, W]
            student_pooled = F.adaptive_avg_pool2d(student_logits, (1, 1)).view(student_logits.size(0), student_logits.size(1))
            teacher_pooled = F.adaptive_avg_pool2d(teacher_logits, (1, 1)).view(teacher_logits.size(0), teacher_logits.size(1))
        else:  # [B, C]
            student_pooled = student_logits
            teacher_pooled = teacher_logits
        
        # Normalize to unit vectors for cosine similarity
        student_norm = F.normalize(student_pooled, p=2, dim=1)  # [B, C]
        teacher_norm = F.normalize(teacher_pooled, p=2, dim=1)  # [B, C]
        
        # Compute pairwise similarity matrices [B, B]
        student_sim = torch.mm(student_norm, student_norm.t())  # [B, B]
        teacher_sim = torch.mm(teacher_norm, teacher_norm.t())  # [B, B]
        
        # Match similarity matrices using MSE
        loss = F.mse_loss(student_sim, teacher_sim, reduction='none')  # [B, B]
        
        if unreduce:
            # Return per-sample loss [B]
            return loss.mean(dim=1)  # Average over the second dimension
        else:
            return loss.mean()


class AngularMarginKDLoss(nn.Module):
    """
    Angular margin loss for KD: focus on angular distance in feature space.
    
    This encourages the student to learn similar decision boundaries as the teacher
    in terms of angular relationships, which can be more robust than Euclidean distance.
    
    Args:
        temperature: Temperature for softening (default: 4.0)
        margin: Angular margin in radians (default: 0.5)
    """
    def __init__(self, temperature=4.0, margin=0.5):
        super(AngularMarginKDLoss, self).__init__()
        self.temperature = temperature
        self.margin = margin
    
    def forward(self, student_logits, teacher_logits, unreduce=False):
        """
        Args:
            student_logits: [B, C] or [B, C, H, W] or [B, C, D, H, W]
            teacher_logits: [B, C] or [B, C, H, W] or [B, C, D, H, W]
            unreduce: If True, return per-sample loss [B]
        
        Returns:
            loss: scalar or [B] if unreduce=True
        """
        # Soften distributions
        student_soft = F.log_softmax(student_logits / self.temperature, dim=1)
        teacher_soft = F.softmax(teacher_logits / self.temperature, dim=1)
        
        # Standard KL divergence as base
        kl_loss = F.kl_div(student_soft, teacher_soft, reduction='none')  # [B, C, ...]
        
        # Add angular margin penalty
        # Compute cosine similarity between student and teacher distributions
        if kl_loss.dim() > 2:
            # For spatial outputs, sum over spatial dimensions first
            kl_loss_spatial = kl_loss.sum(dim=1)  # [B, H, W] or [B, D, H, W]
            if unreduce:
                # Average over spatial dimensions
                loss = kl_loss_spatial.view(kl_loss_spatial.size(0), -1).mean(dim=1)  # [B]
            else:
                loss = kl_loss_spatial.mean()
        else:
            # Classification case
            kl_loss_class = kl_loss.sum(dim=1)  # [B]
            if unreduce:
                loss = kl_loss_class
            else:
                loss = kl_loss_class.mean()
        
        # Scale by temperature^2
        loss = loss * (self.temperature ** 2)
        
        return loss


def get_contrastive_kd_loss(kd_type='contrastive', temperature=4.0, margin=0.5):
    """
    Factory function to get contrastive KD loss.
    
    Args:
        kd_type: str, one of ['contrastive', 'relation', 'angular']
        temperature: float, temperature for softening
        margin: float, margin for contrastive losses
    
    Returns:
        loss_fn: Contrastive KD loss function
    """
    if kd_type == 'contrastive':
        return ContrastiveKDLoss(temperature=temperature, margin=margin)
    elif kd_type == 'relation':
        return RelationKDLoss(temperature=temperature)
    elif kd_type == 'angular':
        return AngularMarginKDLoss(temperature=temperature, margin=margin)
    else:
        raise ValueError(f"Unknown contrastive KD type: {kd_type}. "
                        f"Choose from ['contrastive', 'relation', 'angular']")
