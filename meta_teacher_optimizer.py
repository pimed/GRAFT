"""
Meta-Learning Teacher Weight Optimizer for MTKD-RL

This module implements gradient-based optimization for teacher weight selection,
replacing the neural network agent with direct convex optimization.
"""

import torch
import torch.nn.functional as F
import numpy as np


def simplex_projection(weights, eps=1e-8):
    """
    Project weights onto the probability simplex using Duchi et al. algorithm
    
    Args:
        weights: [num_teachers] tensor to project
        eps: small value for numerical stability
        
    Returns:
        projected_weights: [num_teachers] tensor on simplex
    """
    device = weights.device
    weights = weights.detach()
    
    # Sort weights in descending order
    sorted_weights, _ = torch.sort(weights, descending=True)
    
    # Find the threshold
    cumsum = torch.cumsum(sorted_weights, dim=0)
    arange = torch.arange(1, len(weights) + 1, device=device, dtype=weights.dtype)
    
    # Find largest k such that sorted_weights[k] - (cumsum[k] - 1) / (k + 1) > 0
    condition = sorted_weights - (cumsum - 1) / arange > 0
    if condition.any():
        k = torch.where(condition)[0][-1].item()  # Last True index
        threshold = (cumsum[k] - 1) / (k + 1)
    else:
        threshold = (cumsum[0] - 1) / 1
    
    # Project
    projected = torch.clamp(weights - threshold, min=0.0)
    
    # Normalize to ensure sum = 1 (handle numerical errors)
    projected = projected / (projected.sum() + eps)
    
    return projected


class MetaTeacherOptimizer:
    """
    Meta-learning optimizer for teacher weight selection using gradient descent
    on the simplex constraint manifold.
    """
    
    def __init__(self, num_teachers, inner_steps=5, lr=0.1, momentum=0.9, 
                 temperature=1.0, regularization=0.01):
        """
        Initialize meta-learning teacher optimizer
        
        Args:
            num_teachers: Number of teacher models
            inner_steps: Number of gradient steps for weight optimization
            lr: Learning rate for weight optimization
            momentum: Momentum factor for smoother updates
            temperature: Temperature for softmax (lower = more peaked weights)
            regularization: L2 regularization on weights to prevent extreme values
        """
        self.num_teachers = num_teachers
        self.inner_steps = inner_steps
        self.lr = lr
        self.momentum = momentum
        self.temperature = temperature
        self.regularization = regularization
        
        # Running averages for momentum
        self.logit_momentum = None
        self.feat_momentum = None
        
    def optimize_teacher_weights(self, student_features, teacher_features, 
                               student_logits, teacher_logits, targets, 
                               criterion_div, feat_kd_func):
        """
        Optimize teacher weights for current batch using gradient descent
        
        Args:
            student_features: [B, C, ...] student feature maps (transformed)
            teacher_features: List of [B, C, ...] teacher feature maps  
            student_logits: [B, num_classes, ...] student predictions
            teacher_logits: List of [B, num_classes, ...] teacher predictions
            targets: [B, ...] ground truth labels
            criterion_div: KL divergence criterion for logit distillation
            feat_kd_func: Feature distillation loss function
            
        Returns:
            logit_weights: [num_teachers] optimized weights for logit distillation
            feature_weights: [num_teachers] optimized weights for feature distillation
        """
        device = student_logits.device
        batch_size = student_logits.size(0)
        
        # Initialize weights uniformly
        logit_weights = torch.ones(self.num_teachers, device=device) / self.num_teachers
        feature_weights = torch.ones(self.num_teachers, device=device) / self.num_teachers
        
        # Enable gradients for optimization
        logit_weights.requires_grad_(True)
        feature_weights.requires_grad_(True)
        
        # Optimize for inner_steps iterations
        for step in range(self.inner_steps):
            # Compute weighted logit distillation loss
            logit_losses = []
            for i in range(self.num_teachers):
                kd_loss = criterion_div(student_logits, teacher_logits[i], unreduce=True)
                logit_losses.append(kd_loss.mean(dim=tuple(range(1, kd_loss.dim()))))  # Mean over spatial dims
            
            logit_losses = torch.stack(logit_losses, dim=1)  # [B, num_teachers]
            weighted_logit_loss = torch.sum(logit_weights.unsqueeze(0) * logit_losses, dim=1).mean()
            
            # Compute weighted feature distillation loss
            feature_losses = []
            for i in range(self.num_teachers):
                feat_loss = feat_kd_func(student_features[i], teacher_features[i])
                if feat_loss.dim() > 1:
                    feat_loss = feat_loss.mean(dim=tuple(range(1, feat_loss.dim())))  # Mean over spatial dims
                feature_losses.append(feat_loss)
            
            feature_losses = torch.stack(feature_losses, dim=1)  # [B, num_teachers]
            weighted_feature_loss = torch.sum(feature_weights.unsqueeze(0) * feature_losses, dim=1).mean()
            
            # Add regularization to prevent extreme weights
            logit_reg = self.regularization * torch.sum(logit_weights ** 2)
            feature_reg = self.regularization * torch.sum(feature_weights ** 2)
            
            # Total loss to minimize
            total_loss = weighted_logit_loss + weighted_feature_loss + logit_reg + feature_reg
            
            # Compute gradients
            if logit_weights.grad is not None:
                logit_weights.grad.zero_()
            if feature_weights.grad is not None:
                feature_weights.grad.zero_()
                
            total_loss.backward(retain_graph=True)
            
            # Gradient descent step with momentum
            with torch.no_grad():
                # Momentum updates
                if self.logit_momentum is None:
                    self.logit_momentum = logit_weights.grad.clone()
                    self.feat_momentum = feature_weights.grad.clone()
                else:
                    self.logit_momentum = self.momentum * self.logit_momentum + logit_weights.grad
                    self.feat_momentum = self.momentum * self.feat_momentum + feature_weights.grad
                
                # Update weights
                logit_weights_new = logit_weights - self.lr * self.logit_momentum
                feature_weights_new = feature_weights - self.lr * self.feat_momentum
                
                # Project onto simplex
                logit_weights = simplex_projection(logit_weights_new)
                feature_weights = simplex_projection(feature_weights_new)
                
                # Re-enable gradients for next iteration
                logit_weights.requires_grad_(True)
                feature_weights.requires_grad_(True)
        
        # Apply temperature scaling for more peaked distributions
        if self.temperature != 1.0:
            logit_weights = F.softmax(torch.log(logit_weights + 1e-8) / self.temperature, dim=0)
            feature_weights = F.softmax(torch.log(feature_weights + 1e-8) / self.temperature, dim=0)
        
        return logit_weights.detach(), feature_weights.detach()
    
    def get_context_adaptive_weights(self, student_features, teacher_features,
                                   student_logits, teacher_logits, targets):
        """
        Get context-adaptive weights based on current student-teacher similarity
        This provides a fast heuristic when full optimization is too expensive
        
        Args:
            student_features: [B, C, ...] student feature maps
            teacher_features: List of [B, C, ...] teacher feature maps
            student_logits: [B, num_classes, ...] student predictions  
            teacher_logits: List of [B, num_classes, ...] teacher predictions
            targets: [B, ...] ground truth labels
            
        Returns:
            logit_weights: [B, num_teachers] per-sample logit weights
            feature_weights: [B, num_teachers] per-sample feature weights
        """
        batch_size = student_logits.size(0)
        device = student_logits.device
        
        # Compute feature similarities (cosine similarity of global pooled features)
        feature_sims = []
        for i in range(self.num_teachers):
            # Global pool both student and teacher features
            if student_features[i].dim() == 5:  # 3D
                student_pool = F.adaptive_avg_pool3d(student_features[i], (1, 1, 1)).flatten(1)
                teacher_pool = F.adaptive_avg_pool3d(teacher_features[i], (1, 1, 1)).flatten(1)
            else:  # 2D
                student_pool = F.adaptive_avg_pool2d(student_features[i], (1, 1)).flatten(1)
                teacher_pool = F.adaptive_avg_pool2d(teacher_features[i], (1, 1)).flatten(1)
            
            # Cosine similarity
            sim = F.cosine_similarity(student_pool, teacher_pool, dim=1)
            feature_sims.append(sim)
        
        feature_sims = torch.stack(feature_sims, dim=1)  # [B, num_teachers]
        
        # Compute prediction similarities (negative KL divergence)
        pred_sims = []
        for i in range(self.num_teachers):
            # Pool logits to same shape for KL computation
            if student_logits.dim() == 5:  # 3D segmentation
                student_pool = F.adaptive_avg_pool3d(student_logits, (1, 1, 1)).squeeze()
                teacher_pool = F.adaptive_avg_pool3d(teacher_logits[i], (1, 1, 1)).squeeze()
            elif student_logits.dim() == 4:  # 2D segmentation  
                student_pool = F.adaptive_avg_pool2d(student_logits, (1, 1)).squeeze()
                teacher_pool = F.adaptive_avg_pool2d(teacher_logits[i], (1, 1)).squeeze()
            else:  # Classification
                student_pool = student_logits
                teacher_pool = teacher_logits[i]
                
            # Negative KL divergence (higher = more similar)
            kl_div = F.kl_div(F.log_softmax(student_pool, dim=1), 
                             F.softmax(teacher_pool, dim=1), 
                             reduction='none').sum(dim=1)
            pred_sims.append(-kl_div)  # Negate so higher is better
        
        pred_sims = torch.stack(pred_sims, dim=1)  # [B, num_teachers]
        
        # Combine similarities and convert to weights
        feature_weights = F.softmax(feature_sims / self.temperature, dim=1)
        logit_weights = F.softmax(pred_sims / self.temperature, dim=1)
        
        return logit_weights, feature_weights


class AdaptiveMetaOptimizer(MetaTeacherOptimizer):
    """
    Advanced version that adapts optimization parameters based on training progress
    """
    
    def __init__(self, num_teachers, **kwargs):
        super().__init__(num_teachers, **kwargs)
        self.step_count = 0
        self.initial_lr = self.lr
        self.initial_inner_steps = self.inner_steps
        
    def update_hyperparams(self, epoch, loss_history=None):
        """
        Adapt optimization hyperparameters based on training progress
        
        Args:
            epoch: Current training epoch
            loss_history: Recent loss values for adaptation
        """
        self.step_count += 1
        
        # Decay learning rate over time
        self.lr = self.initial_lr * (0.95 ** (epoch // 10))
        
        # Reduce inner steps as training progresses (faster convergence)
        if epoch > 50:
            self.inner_steps = max(2, self.initial_inner_steps - 1)
        if epoch > 100:
            self.inner_steps = max(1, self.initial_inner_steps - 2)
            
        # Increase temperature for more exploration early in training
        if epoch < 20:
            self.temperature = 2.0
        elif epoch < 50:
            self.temperature = 1.5
        else:
            self.temperature = 1.0


def get_meta_teacher_optimizer(num_teachers, mode='standard', **kwargs):
    """
    Factory function to create appropriate meta-learning optimizer
    
    Args:
        num_teachers: Number of teacher models
        mode: 'standard' or 'adaptive'
        **kwargs: Additional arguments for optimizer
        
    Returns:
        MetaTeacherOptimizer instance
    """
    if mode == 'adaptive':
        return AdaptiveMetaOptimizer(num_teachers, **kwargs)
    else:
        return MetaTeacherOptimizer(num_teachers, **kwargs)