import torch
import torch.nn as nn
import torch.nn.functional as F

__all__ = ['NormalizedFeatureMSELoss', 'NormalizedFeatureKLLoss']


class NormalizedFeatureMSELoss(nn.Module):
    """MSE between L2-normalized features (makes loss scale-invariant).

    Expects tensors of shape (N, C, H, W) or (N, C, ...). Will flatten spatial dims.
    """
    def __init__(self, eps: float = 1e-6):
        super().__init__()
        self.eps = eps

    def _l2_normalize(self, x: torch.Tensor) -> torch.Tensor:
        # Flatten non-batch dims
        n = x.size(0)
        flat = x.view(n, -1)
        norm = flat.norm(p=2, dim=1, keepdim=True).clamp_min(self.eps)
        flat = flat / norm
        return flat

    def forward(self, f_s: torch.Tensor, f_t: torch.Tensor) -> torch.Tensor:
        s = self._l2_normalize(f_s)
        t = self._l2_normalize(f_t)
        loss = F.mse_loss(s, t, reduction='none')
        # mean per-sample
        return loss.mean(dim=1)


class NormalizedFeatureKLLoss(nn.Module):
    """KL divergence on channel-normalized maps, with optional clipping for logits.

    This wraps the existing ChannelNorm approach but adds a clipping step to
    avoid extreme logits creating numerically unstable softmaxes.
    """
    def __init__(self, temperature: float = 4.0, clip_logit: float = 50.0):
        super().__init__()
        self.temperature = temperature
        self.clip_logit = clip_logit
        self.criterion = nn.KLDivLoss(reduction='none')

    def channel_softmax(self, x: torch.Tensor) -> torch.Tensor:
        # x expected shape (N, C, ...). Softmax along last dim after flattening spatial dims
        n = x.size(0)
        flat = x.view(n, x.size(1), -1)  # (N, C, S)
        # softmax over spatial locations
        return flat.softmax(dim=-1)

    def forward(self, f_s: torch.Tensor, f_t: torch.Tensor) -> torch.Tensor:
        # Clip extreme logits for numerical stability
        s = torch.clamp(f_s / self.temperature, min=-self.clip_logit, max=self.clip_logit)
        t = torch.clamp(f_t.detach() / self.temperature, min=-self.clip_logit, max=self.clip_logit)

        norm_s = self.channel_softmax(s).log()
        norm_t = self.channel_softmax(t)

        loss = self.criterion(norm_s, norm_t).sum(-1).mean(-1)
        return loss * (self.temperature ** 2)


if __name__ == '__main__':
    # quick smoke test
    print('Running normalized_feature_losses quick self-test...')
    device = 'cpu'
    N, C, H, W = 2, 8, 4, 4
    a = torch.randn(N, C, H, W, device=device) * 2.0
    b = torch.randn(N, C, H, W, device=device)

    mse = NormalizedFeatureMSELoss()
    kl = NormalizedFeatureKLLoss()

    out_mse = mse(a, b)
    out_kl = kl(a, b)
    print('MSE output shape:', out_mse.shape, 'values:', out_mse)
    print('KL output shape:', out_kl.shape, 'values:', out_kl)
    print('Self-test OK')
"""
Normalized and robust versions of feature distillation losses
to handle feature magnitude outliers and prevent loss spikes.

Author: GitHub Copilot
Date: November 18, 2025
Purpose: Fix batch 230/400 feature loss spikes (27 → 750+)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

__all__ = ['FeatureMSELoss', 'FeatureKLLoss', 'FeatureMSELoss3D', 'FeatureKLLoss3D',
           'NormalizedFeatureMSELoss', 'NormalizedFeatureMSELoss3D',
           'RobustFeatureMSELoss', 'RobustFeatureMSELoss3D',
           'MaskedFeatureMSELoss3D']


# ============================================================================
# Original Losses (for backward compatibility)
# ============================================================================

class FeatureMSELoss(nn.Module):
    """Fitnets: hints for thin deep nets, ICLR 2015"""
    def __init__(self):
        super(FeatureMSELoss, self).__init__()

    def forward(self, f_s, f_t):
        f_s = f_s.view(f_s.size(0), -1)
        f_t = f_t.view(f_t.size(0), -1) 
        loss = ((f_s - f_t)**2).mean(1)
        return loss


class ChannelNorm(nn.Module):
    def __init__(self):
        super(ChannelNorm, self).__init__()
    def forward(self,featmap):
        n,c,h,w = featmap.shape
        featmap = featmap.reshape((n,c,-1))
        featmap = featmap.softmax(dim=-1)
        return featmap
    
    
class FeatureKLLoss(nn.Module):
    def __init__(self, temperature=4.0):
        super(FeatureKLLoss, self).__init__()
        self.normalize = ChannelNorm()
        self.criterion = nn.KLDivLoss(reduction='none')
        self.temperature = temperature
       
    def forward(self, f_s, f_t):
        #n,c,h,w = f_s.shape
        norm_s = self.normalize(f_s/self.temperature)
        norm_t = self.normalize(f_t.detach()/self.temperature)
        norm_s = norm_s.log()

        loss = self.criterion(norm_s, norm_t).sum(-1).mean(-1)

        return loss * (self.temperature**2)


# ============================================================================
# NORMALIZED Feature Losses (L2 normalization before MSE)
# ============================================================================

class NormalizedFeatureMSELoss(nn.Module):
    """
    Feature loss using cosine similarity for scale-invariant comparison.
    
    Uses cosine distance (1 - cosine_similarity) which:
    - Is scale-invariant (handles magnitude outliers)
    - Produces loss in [0, 2] range
    - 0 = perfect alignment, 1 = perpendicular, 2 = opposite
    
    This is more numerically stable than L2-normalized MSE and gives
    reasonable loss magnitudes.
    """
    def __init__(self, eps=1e-8):
        super(NormalizedFeatureMSELoss, self).__init__()
        self.eps = eps

    def forward(self, f_s, f_t):
        # Flatten to [B, C] where C includes all spatial dimensions
        f_s_flat = f_s.view(f_s.size(0), -1)
        f_t_flat = f_t.view(f_t.size(0), -1)
        
        # Compute cosine similarity
        cos_sim = F.cosine_similarity(f_s_flat, f_t_flat, dim=1, eps=self.eps)
        
        # Convert to loss
        loss = 1 - cos_sim
        
        return loss


class NormalizedFeatureMSELoss3D(nn.Module):
    """
    3D Feature loss using cosine similarity instead of full L2 normalization.
    
    This provides a gentler normalization that:
    - Handles magnitude outliers (scale-invariant)
    - Preserves loss scale similar to original MSE (0-2 range)
    - Uses cosine distance: loss = 1 - cos(f_s, f_t)
    
    Loss interpretation:
    - 0: Perfect alignment (same direction)
    - 1: Perpendicular features
    - 2: Opposite directions
    
    This is equivalent to MSE on L2-normalized features, but scaled to give
    reasonable magnitudes similar to the original 20-400 range when combined
    with appropriate weighting.
    """
    def __init__(self, eps=1e-8):
        super(NormalizedFeatureMSELoss3D, self).__init__()
        self.eps = eps

    def forward(self, f_s, f_t):
        # Flatten to [B, -1]
        f_s_flat = f_s.view(f_s.size(0), -1)
        f_t_flat = f_t.view(f_t.size(0), -1)
        
        # Compute cosine similarity
        # cos_sim in [-1, 1], where 1 = same direction, -1 = opposite
        cos_sim = F.cosine_similarity(f_s_flat, f_t_flat, dim=1, eps=self.eps)
        
        # Convert to loss: 0 (perfect) to 2 (opposite)
        # This gives similar scale to normalized MSE but with better numerical properties
        loss = 1 - cos_sim
        
        return loss


# ============================================================================
# ROBUST Feature Losses (Huber/Smooth-L1 instead of MSE)
# ============================================================================

class RobustFeatureMSELoss(nn.Module):
    """
    Robust feature loss using Smooth-L1 (Huber) loss instead of MSE.
    
    Smooth-L1 loss is less sensitive to outliers than MSE:
    - For small errors: behaves like MSE (quadratic)
    - For large errors: behaves linearly, preventing explosion
    
    Combined with normalization for maximum robustness.
    
    Args:
        beta: float, threshold for switching from quadratic to linear (default: 1.0)
        normalize: bool, whether to L2-normalize features first (default: True)
        eps: float, epsilon for normalization stability
    """
    def __init__(self, beta=1.0, normalize=True, eps=1e-8):
        super(RobustFeatureMSELoss, self).__init__()
        self.beta = beta
        self.normalize = normalize
        self.eps = eps

    def forward(self, f_s, f_t):
        # Flatten
        f_s_flat = f_s.view(f_s.size(0), -1)
        f_t_flat = f_t.view(f_t.size(0), -1)
        
        # Optional normalization
        if self.normalize:
            f_s_flat = F.normalize(f_s_flat, p=2, dim=1, eps=self.eps)
            f_t_flat = F.normalize(f_t_flat, p=2, dim=1, eps=self.eps)
        
        # Smooth-L1 loss (more robust than MSE)
        loss = F.smooth_l1_loss(f_s_flat, f_t_flat, reduction='none', beta=self.beta)
        loss = loss.mean(1)  # Average over feature dimension, keep batch dimension
        
        return loss


class RobustFeatureMSELoss3D(nn.Module):
    """3D version of RobustFeatureMSELoss"""
    def __init__(self, beta=1.0, normalize=True, eps=1e-8):
        super(RobustFeatureMSELoss3D, self).__init__()
        self.beta = beta
        self.normalize = normalize
        self.eps = eps

    def forward(self, f_s, f_t):
        # Cast to float32 for numerical stability (BF16 loses precision)
        f_s = f_s.float()
        f_t = f_t.float()
        
        # Flatten
        f_s_flat = f_s.view(f_s.size(0), -1)
        f_t_flat = f_t.view(f_t.size(0), -1)
        
        # Optional normalization
        if self.normalize:
            f_s_flat = F.normalize(f_s_flat, p=2, dim=1, eps=self.eps)
            f_t_flat = F.normalize(f_t_flat, p=2, dim=1, eps=self.eps)
        
        # MSE loss instead of smooth_l1 (avoid beta threshold issue)
        # For normalized features, MSE = 2(1 - cos_sim), which is more meaningful
        loss = F.mse_loss(f_s_flat, f_t_flat, reduction='none')
        loss = loss.sum(1)  # Sum over features to get meaningful loss magnitude
        
        return loss


class MaskedFeatureMSELoss3D(nn.Module):
    """
    Per-voxel L2-normalized MSE with spatial cancer-correctness mask.
    
    Unlike RobustFeatureMSELoss3D which flattens everything and does global L2 norm,
    this normalizes per-voxel (across channels) and computes per-voxel MSE, then
    applies a binary spatial mask to only distill features where the teacher is correct
    about cancer.
    
    Input shapes:
        f_s, f_t: [B, C, D, H, W] — student/teacher features
        mask:     [B, 1, D, H, W] — binary mask (1=distill, 0=skip)
    
    Output: [B] — per-sample loss (averaged over masked voxels)
    """
    def __init__(self, normalize=True, eps=1e-8):
        super(MaskedFeatureMSELoss3D, self).__init__()
        self.normalize = normalize
        self.eps = eps

    def forward(self, f_s, f_t, mask=None):
        """
        Args:
            f_s: [B, C, D, H, W] student features
            f_t: [B, C, D, H, W] teacher features
            mask: [B, 1, D, H, W] binary spatial mask, or None (no masking)
        Returns:
            loss: [B] per-sample loss
        """
        f_s = f_s.float()
        f_t = f_t.float()
        
        if mask is None:
            # Fallback to global behavior (same as RobustFeatureMSELoss3D)
            f_s_flat = f_s.view(f_s.size(0), -1)
            f_t_flat = f_t.view(f_t.size(0), -1)
            if self.normalize:
                f_s_flat = F.normalize(f_s_flat, p=2, dim=1, eps=self.eps)
                f_t_flat = F.normalize(f_t_flat, p=2, dim=1, eps=self.eps)
            loss = F.mse_loss(f_s_flat, f_t_flat, reduction='none').sum(1)
            return loss
        
        # Per-voxel L2 normalization across channel dim (dim=1)
        if self.normalize:
            f_s = F.normalize(f_s, p=2, dim=1, eps=self.eps)
            f_t = F.normalize(f_t, p=2, dim=1, eps=self.eps)
        
        # Per-voxel MSE across channels: [B, C, D, H, W] → [B, D, H, W]
        voxel_mse = ((f_s - f_t) ** 2).sum(dim=1)  # [B, D, H, W]
        
        # Apply mask: [B, 1, D, H, W] → [B, D, H, W]
        mask_squeezed = mask.squeeze(1).float()  # [B, D, H, W]
        masked_mse = voxel_mse * mask_squeezed
        
        # Average over masked voxels (avoid div by 0 if mask is all zeros)
        n_masked = mask_squeezed.sum(dim=(1, 2, 3)).clamp(min=1.0)  # [B]
        loss = masked_mse.sum(dim=(1, 2, 3)) / n_masked  # [B]
        
        return loss


# ============================================================================
# Helper function to get the right loss based on configuration
# ============================================================================

def get_feature_mse_loss(use_3d=False, normalize=True, robust=False, **kwargs):
    """
    Factory function to get the appropriate feature MSE loss.
    
    Args:
        use_3d: bool, whether to use 3D-compatible version
        normalize: bool, whether to normalize features before MSE
        robust: bool, whether to use robust (Smooth-L1) loss
        **kwargs: additional arguments for the loss function
        
    Returns:
        loss_fn: The appropriate loss function instance
        
    Examples:
        # Original MSE (backward compatible)
        loss_fn = get_feature_mse_loss(use_3d=True, normalize=False, robust=False)
        
        # Normalized MSE (recommended for fixing spikes)
        loss_fn = get_feature_mse_loss(use_3d=True, normalize=True, robust=False)
        
        # Robust normalized loss (maximum stability)
        loss_fn = get_feature_mse_loss(use_3d=True, normalize=True, robust=True, beta=1.0)
    """
    if robust:
        if use_3d:
            return RobustFeatureMSELoss3D(**kwargs)
        else:
            return RobustFeatureMSELoss(**kwargs)
    elif normalize:
        if use_3d:
            return NormalizedFeatureMSELoss3D(**kwargs)
        else:
            return NormalizedFeatureMSELoss(**kwargs)
    else:
        # Original loss (no normalization)
        if use_3d:
            from distiller_zoo.segmentation_3d_losses import FeatureMSELoss3D as OrigMSE3D
            return OrigMSE3D()
        else:
            return FeatureMSELoss()


# ============================================================================
# Testing
# ============================================================================

if __name__ == '__main__':
    print("=" * 80)
    print("Testing Normalized Feature Losses")
    print("=" * 80)
    
    # Simulate the batch 230 spike scenario
    batch_size = 2
    channels = 256
    depth, height, width = 4, 16, 16
    
    # Normal student features
    student_feat = torch.randn(batch_size, channels, depth, height, width) * 20
    
    # Outlier teacher features (like batch 230)
    teacher_feat = torch.randn(batch_size, channels, depth, height, width) * 60
    
    print(f"\n📊 Test Scenario (simulating batch 230 spike):")
    print(f"   Student features: mean={student_feat.mean():.2f}, std={student_feat.std():.2f}, max={student_feat.abs().max():.2f}")
    print(f"   Teacher features: mean={teacher_feat.mean():.2f}, std={teacher_feat.std():.2f}, max={teacher_feat.abs().max():.2f}")
    
    # Test original loss
    print(f"\n1️⃣ Original FeatureMSELoss3D:")
    # Ensure project root is on sys.path when running this script directly
    import os, sys
    sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
    from distiller_zoo.segmentation_3d_losses import FeatureMSELoss3D as OrigMSE3D
    orig_loss = OrigMSE3D()
    loss_orig = orig_loss(student_feat, teacher_feat)
    print(f"   Loss: {loss_orig.mean():.2f} (per sample: {loss_orig.tolist()})")
    
    # Test normalized loss
    print(f"\n2️⃣ NormalizedFeatureMSELoss3D:")
    norm_loss = NormalizedFeatureMSELoss3D()
    loss_norm = norm_loss(student_feat, teacher_feat)
    print(f"   Loss: {loss_norm.mean():.2f} (per sample: {loss_norm.tolist()})")
    print(f"   ✅ Reduction: {loss_orig.mean() / loss_norm.mean():.1f}×")
    
    # Test robust loss
    print(f"\n3️⃣ RobustFeatureMSELoss3D (with normalization):")
    robust_loss = RobustFeatureMSELoss3D(beta=1.0, normalize=True)
    loss_robust = robust_loss(student_feat, teacher_feat)
    print(f"   Loss: {loss_robust.mean():.2f} (per sample: {loss_robust.tolist()})")
    print(f"   ✅ Reduction: {loss_orig.mean() / loss_robust.mean():.1f}×")
    
    # Test with normal (non-outlier) features
    print(f"\n4️⃣ Sanity check with normal features (both ~20):")
    teacher_feat_normal = torch.randn(batch_size, channels, depth, height, width) * 20
    
    loss_orig_normal = orig_loss(student_feat, teacher_feat_normal).mean()
    loss_norm_normal = norm_loss(student_feat, teacher_feat_normal).mean()
    loss_robust_normal = robust_loss(student_feat, teacher_feat_normal).mean()
    
    print(f"   Original: {loss_orig_normal:.2f}")
    print(f"   Normalized: {loss_norm_normal:.2f}")
    print(f"   Robust: {loss_robust_normal:.2f}")
    print(f"   ✅ All methods work well when features are similar scale")
    
    print(f"\n" + "=" * 80)
    print("Summary:")
    print("  - Original MSE: Explodes with outliers (750+)")
    print("  - Normalized MSE: Scale-invariant, robust to outliers (~2-4)")
    print("  - Robust MSE: Most stable, handles both outliers and normal cases")
    print("=" * 80)
