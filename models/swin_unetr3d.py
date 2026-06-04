"""
3D Swin-UNETR (Swin Transformer U-Net) for medical image segmentation
Transformer-based alternative to CNN-based U-Net

Based on: "Swin UNETR: Swin Transformers for Semantic Segmentation of Brain Tumors in MRI Images"
https://arxiv.org/abs/2201.01266

Input: [B, 3, 20, 256, 256] (T2/ADC/DWI volumes)
Output: [B, num_classes, 20, 256, 256] (segmentation masks)
Features: Support is_feat=True for intermediate feature extraction (KD compatible)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
from einops.layers.torch import Rearrange
import numpy as np


class PatchEmbed3D(nn.Module):
    """3D Image to Patch Embedding with hierarchical structure"""
    
    def __init__(self, img_size=(20, 256, 256), patch_size=(2, 4, 4), in_chans=3, embed_dim=96):
        super().__init__()
        self.img_size = img_size
        self.patch_size = patch_size
        self.grid_size = (img_size[0] // patch_size[0], 
                          img_size[1] // patch_size[1], 
                          img_size[2] // patch_size[2])
        self.num_patches = self.grid_size[0] * self.grid_size[1] * self.grid_size[2]
        
        self.proj = nn.Conv3d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size)
        self.norm = nn.LayerNorm(embed_dim)

    def forward(self, x):
        # x: [B, C, D, H, W]
        B, C, D, H, W = x.shape
        x = self.proj(x)  # [B, embed_dim, D', H', W']
        x = rearrange(x, 'b c d h w -> b (d h w) c')  # [B, num_patches, embed_dim]
        x = self.norm(x)
        return x


class WindowAttention3D(nn.Module):
    """Window-based Multi-head Self Attention for 3D volumes"""
    
    def __init__(self, dim, window_size, num_heads):
        super().__init__()
        self.dim = dim
        self.window_size = window_size  # (Wd, Wh, Ww)
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = head_dim ** -0.5

        self.qkv = nn.Linear(dim, dim * 3)
        self.proj = nn.Linear(dim, dim)

    def forward(self, x):
        # x: [B*num_windows, Wd*Wh*Ww, C]
        B_, N, C = x.shape
        qkv = self.qkv(x).reshape(B_, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]  # [B_, num_heads, N, head_dim]

        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)

        x = (attn @ v).transpose(1, 2).reshape(B_, N, C)
        x = self.proj(x)
        return x


class SwinTransformerBlock3D(nn.Module):
    """Swin Transformer Block with shifted windows"""
    
    def __init__(self, dim, num_heads, window_size=(2, 8, 8), shift_size=0, mlp_ratio=4.):
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.window_size = window_size
        self.shift_size = shift_size
        self.mlp_ratio = mlp_ratio

        self.norm1 = nn.LayerNorm(dim)
        self.attn = WindowAttention3D(dim, window_size, num_heads)
        
        self.norm2 = nn.LayerNorm(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = nn.Sequential(
            nn.Linear(dim, mlp_hidden_dim),
            nn.GELU(),
            nn.Linear(mlp_hidden_dim, dim),
        )

    def forward(self, x, D, H, W):
        # x: [B, D*H*W, C]
        B, L, C = x.shape
        
        shortcut = x
        x = self.norm1(x)
        x = x.view(B, D, H, W, C)
        
        # Simplified window partitioning (for basic implementation)
        # For full Swin-UNETR, implement proper window shift + partition
        x = x.view(B, -1, C)
        x = self.attn(x)
        
        # FFN
        x = shortcut + x
        x = x + self.mlp(self.norm2(x))
        
        return x


class PatchMerging3D(nn.Module):
    """Patch Merging Layer - reduces spatial dimensions"""
    
    def __init__(self, dim, norm_layer=nn.LayerNorm):
        super().__init__()
        self.dim = dim
        self.reduction = nn.Linear(8 * dim, 2 * dim, bias=False)  # 2x2x2 merging
        self.norm = norm_layer(8 * dim)

    def forward(self, x, D, H, W):
        # x: [B, D*H*W, C]
        B, L, C = x.shape
        
        x = x.view(B, D, H, W, C)
        
        # Pad if needed
        pad_d = (2 - D % 2) % 2
        pad_h = (2 - H % 2) % 2
        pad_w = (2 - W % 2) % 2
        x = F.pad(x, (0, 0, 0, pad_w, 0, pad_h, 0, pad_d))
        
        D, H, W = D + pad_d, H + pad_h, W + pad_w
        
        # Merge 2x2x2 patches
        x0 = x[:, 0::2, 0::2, 0::2, :]  # [B, D/2, H/2, W/2, C]
        x1 = x[:, 1::2, 0::2, 0::2, :]
        x2 = x[:, 0::2, 1::2, 0::2, :]
        x3 = x[:, 1::2, 1::2, 0::2, :]
        x4 = x[:, 0::2, 0::2, 1::2, :]
        x5 = x[:, 1::2, 0::2, 1::2, :]
        x6 = x[:, 0::2, 1::2, 1::2, :]
        x7 = x[:, 1::2, 1::2, 1::2, :]
        x = torch.cat([x0, x1, x2, x3, x4, x5, x6, x7], -1)  # [B, D/2, H/2, W/2, 8*C]
        
        x = x.view(B, -1, 8 * C)  # [B, D/2*H/2*W/2, 8*C]
        x = self.norm(x)
        x = self.reduction(x)  # [B, D/2*H/2*W/2, 2*C]
        
        return x, D // 2, H // 2, W // 2


class SwinEncoder3D(nn.Module):
    """Swin Transformer Encoder - hierarchical feature extraction"""
    
    def __init__(self, img_size=(20, 256, 256), patch_size=(2, 4, 4), in_chans=3, 
                 embed_dim=48, depths=[2, 2, 2, 2], num_heads=[3, 6, 12, 24]):
        super().__init__()
        
        self.patch_embed = PatchEmbed3D(img_size, patch_size, in_chans, embed_dim)
        
        D, H, W = self.patch_embed.grid_size
        
        self.layers = nn.ModuleList()
        self.down_samplers = nn.ModuleList()
        
        for i in range(len(depths)):
            layer = nn.ModuleList([
                SwinTransformerBlock3D(
                    dim=embed_dim * (2 ** i),
                    num_heads=num_heads[i],
                    window_size=(2, 8, 8),
                    shift_size=0 if (j % 2 == 0) else 1,
                )
                for j in range(depths[i])
            ])
            self.layers.append(layer)
            
            if i < len(depths) - 1:
                down_sampler = PatchMerging3D(embed_dim * (2 ** i))
                self.down_samplers.append(down_sampler)
        
        self.num_layers = len(depths)
        
    def forward(self, x):
        # x: [B, 3, D, H, W]
        x = self.patch_embed(x)  # [B, num_patches, embed_dim]
        
        D, H, W = self.patch_embed.grid_size
        features = []
        
        for i, layer in enumerate(self.layers):
            for blk in layer:
                x = blk(x, D, H, W)
            
            features.append(x.view(-1, D, H, W, x.shape[-1]).permute(0, 4, 1, 2, 3))  # [B, C, D, H, W]
            
            if i < len(self.down_samplers):
                x, D, H, W = self.down_samplers[i](x, D, H, W)
        
        return features


class Up3D(nn.Module):
    """Upscaling block for decoder"""
    
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.up = nn.ConvTranspose3d(in_channels, out_channels, kernel_size=2, stride=2)
        self.conv = nn.Sequential(
            nn.Conv3d(out_channels * 2, out_channels, kernel_size=3, padding=1, bias=False),
            nn.InstanceNorm3d(out_channels, affine=True),
            nn.ReLU(inplace=True),
            nn.Conv3d(out_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.InstanceNorm3d(out_channels, affine=True),
            nn.ReLU(inplace=True)
        )

    def forward(self, x1, x2):
        # x1: from decoder (lower resolution)
        # x2: skip connection from encoder (higher resolution)
        x1 = self.up(x1)
        
        # Handle size mismatch
        diffD = x2.size()[2] - x1.size()[2]
        diffH = x2.size()[3] - x1.size()[3]
        diffW = x2.size()[4] - x1.size()[4]
        
        x1 = F.pad(x1, [diffW // 2, diffW - diffW // 2,
                        diffH // 2, diffH - diffH // 2,
                        diffD // 2, diffD - diffD // 2])
        
        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)


class SwinUNETR3D(nn.Module):
    """
    Swin-UNETR: Transformer-based 3D U-Net with Swin Transformer encoder
    
    Compatible with MTKD-RL framework - returns features for knowledge distillation
    """
    
    def __init__(self, num_classes=3, img_size=(20, 256, 256), in_chans=3, 
                 embed_dim=48, depths=[2, 2, 2, 2], num_heads=[3, 6, 12, 24],
                 patch_size=(2, 8, 8)):
        super().__init__()
        
        # Use balanced patch size for memory vs detail trade-off
        # patch_size=(2, 8, 8) creates grid (10, 32, 32) = 10,240 patches
        # Original (2, 4, 4) created (10, 64, 64) = 40,960 patches (OOM!)
        # (2, 16, 16) created (10, 16, 16) = 2,560 patches (too coarse)
        self.encoder = SwinEncoder3D(img_size, patch_size, in_chans, embed_dim, depths, num_heads)
        self.patch_size = patch_size
        
        # Decoder - mirrors encoder hierarchy
        self.up1 = Up3D(embed_dim * 8, embed_dim * 4)  # 384 -> 192
        self.up2 = Up3D(embed_dim * 4, embed_dim * 2)  # 192 -> 96
        self.up3 = Up3D(embed_dim * 2, embed_dim)      # 96 -> 48
        
        # Final upsampling to match input size
        self.final_up = nn.Sequential(
            nn.ConvTranspose3d(embed_dim, embed_dim // 2, kernel_size=patch_size, stride=patch_size),
            nn.InstanceNorm3d(embed_dim // 2, affine=True),
            nn.ReLU(inplace=True),
        )
        
        self.out_conv = nn.Conv3d(embed_dim // 2, num_classes, kernel_size=1)
        
        # Initialize weights with Kaiming (He) initialization
        self._init_weights()
    
    def _init_weights(self):
        """
        Initialize weights with Kaiming (He) initialization for better training start.
        Focus on decoder and output layers which most affect final predictions.
        """
        # Initialize decoder upsampling layers
        for module in [self.up1, self.up2, self.up3]:
            if hasattr(module, 'up'):
                for m in module.up.modules():
                    if isinstance(m, nn.Conv3d) or isinstance(m, nn.ConvTranspose3d):
                        nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                        if m.bias is not None:
                            nn.init.constant_(m.bias, 0)
            if hasattr(module, 'conv'):
                for m in module.conv.modules():
                    if isinstance(m, nn.Conv3d):
                        nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                        if m.bias is not None:
                            nn.init.constant_(m.bias, 0)
        
        # Initialize final upsampling
        for m in self.final_up.modules():
            if isinstance(m, nn.ConvTranspose3d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
        
        # Initialize output conv with smaller weights to prevent extreme initial logits
        # Use Xavier (Glorot) for output layer since it connects to softmax (no ReLU after)
        nn.init.xavier_uniform_(self.out_conv.weight, gain=1.0)
        if self.out_conv.bias is not None:
            # Small negative bias for background class to encourage foreground predictions
            nn.init.constant_(self.out_conv.bias, 0.0)
            # Alternative: bias background class slightly negative
            # self.out_conv.bias.data[0] = -0.1  # Background class
            # self.out_conv.bias.data[1:] = 0.0  # Foreground classes

    def forward(self, x, is_feat=False):
        """
        Forward pass with optional feature extraction for knowledge distillation
        
        Args:
            x: [B, 3, D, H, W] input volume
            is_feat: bool or tuple, if True return intermediate features
        
        Returns:
            If is_feat=True: (features_list, logits)
            Otherwise: logits
        """
        # Handle DataParallel case where is_feat is passed as a tuple
        if isinstance(is_feat, tuple):
            is_feat = is_feat[0]
        
        # print(f"[DEBUG SWIN] input x: {x.shape}, x std:{x.std().item():.6f}")
        # Encoder
        enc_features = self.encoder(x)  # List of [B, C, D, H, W] at different scales
        # enc_features[0]: [B, 48, 10, 64, 64]
        # enc_features[1]: [B, 96, 5, 32, 32]
        # enc_features[2]: [B, 192, 2, 16, 16]  <- layer -2 (for KD)
        # enc_features[3]: [B, 384, 1, 8, 8]    <- layer -1 (for KD)
        
        # Decoder with skip connections
        d1 = self.up1(enc_features[3], enc_features[2])  # [B, 192, 2, 16, 16]
        d2 = self.up2(d1, enc_features[1])                # [B, 96, 5, 32, 32]
        d3 = self.up3(d2, enc_features[0])                # [B, 48, 10, 64, 64]
        
        d4 = self.final_up(d3)  # [B, 24, 20, 256, 256]
        logits = self.out_conv(d4)  # [B, num_classes, 20, 256, 256]
        
        if is_feat:
            # DEBUG: Check where constant outputs come from
            print(f"[DEBUG SWIN] Encoder bottleneck std: {enc_features[-1].std().item():.6f}")
            print(f"[DEBUG SWIN] Decoder d1 std: {d1.std().item():.6f}")
            print(f"[DEBUG SWIN] Decoder d2 std: {d2.std().item():.6f}")
            print(f"[DEBUG SWIN] Decoder d3 std: {d3.std().item():.6f}")
            print(f"[DEBUG SWIN] Decoder d4 std: {d4.std().item():.6f}")
            print(f"[DEBUG SWIN] Final logits std: {logits.std().item():.6f}")
            
            # Return features compatible with MTKD-RL framework
            # Use d3 and d4 (both decoder features) for better spatial coverage
            # features[-2] = d3 [B, 48, 10, 64, 64] - mid-decoder with good spatial resolution
            # features[-1] = d4 [B, 24, 20, 256, 256] - final decoder before output conv
            # This avoids tiny spatial dimensions (1×8×8) that cause InstanceNorm collapse
            return [d3, d4], logits
        else:
            return logits


class SwinUNETR3D_Small(nn.Module):
    """Smaller Swin-UNETR for faster training"""
    
    def __init__(self, num_classes=3, img_size=(20, 256, 256), in_chans=3, patch_size=(2, 8, 8)):
        super().__init__()
        self.model = SwinUNETR3D(
            num_classes=num_classes,
            img_size=img_size,
            in_chans=in_chans,
            embed_dim=24,           # Reduced from 48
            depths=[1, 1, 1, 1],    # Shallow for speed
            num_heads=[2, 4, 8, 16], # Reduced from [3, 6, 12, 24]
            patch_size=patch_size   # Balanced patch size (2, 8, 8)
        )
    
    def forward(self, x, is_feat=False):
        if isinstance(is_feat, tuple):
            is_feat = is_feat[0]
        return self.model(x, is_feat)


class SwinUNETR3D_Medium(nn.Module):
    """Medium Swin-UNETR - balanced between Small and Standard"""
    
    def __init__(self, num_classes=3, img_size=(20, 256, 256), in_chans=3, patch_size=(2, 8, 8)):
        super().__init__()
        self.model = SwinUNETR3D(
            num_classes=num_classes,
            img_size=img_size,
            in_chans=in_chans,
            embed_dim=32,           # Between Small (24) and Standard (48)
            depths=[2, 2, 2, 2],    # Uniform depth like original paper
            num_heads=[2, 4, 8, 16], # Same head configuration
            patch_size=patch_size   # Balanced patch size (2, 8, 8)
        )
    
    def forward(self, x, is_feat=False):
        if isinstance(is_feat, tuple):
            is_feat = is_feat[0]
        return self.model(x, is_feat)


class SwinUNETR3D_Small_Deep(nn.Module):
    """Small Swin-UNETR with deeper stages for better capacity"""
    
    def __init__(self, num_classes=3, img_size=(20, 256, 256), in_chans=3, patch_size=(2, 8, 8)):
        super().__init__()
        self.model = SwinUNETR3D(
            num_classes=num_classes,
            img_size=img_size,
            in_chans=in_chans,
            embed_dim=24,           # Same as Small
            depths=[2, 2, 6, 2],    # Deep! Especially in middle stages
            num_heads=[2, 4, 8, 16], # Same as Small
            patch_size=patch_size   # Balanced patch size (2, 8, 8)
        )
    
    def forward(self, x, is_feat=False):
        if isinstance(is_feat, tuple):
            is_feat = is_feat[0]
        return self.model(x, is_feat)


class SwinUNETR3D_Tiny(nn.Module):
    """Tiny Swin-UNETR for very limited resources"""
    
    def __init__(self, num_classes=3, img_size=(20, 256, 256), in_chans=3, patch_size=(2, 8, 8)):
        super().__init__()
        self.model = SwinUNETR3D(
            num_classes=num_classes,
            img_size=img_size,
            in_chans=in_chans,
            embed_dim=16,           # Further reduced
            depths=[1, 1, 1, 1],    # Shallow for minimal resources
            num_heads=[2, 4, 8, 16],
            patch_size=patch_size   # Balanced patch size (2, 8, 8)
        )
    
    def forward(self, x, is_feat=False):
        if isinstance(is_feat, tuple):
            is_feat = is_feat[0]
        return self.model(x, is_feat)


# Factory functions
def swin_unetr3d(num_classes=3, **kwargs):
    """Standard Swin-UNETR"""
    return SwinUNETR3D(num_classes=num_classes, **kwargs)

def swin_unetr3d_small(num_classes=3, **kwargs):
    """Small Swin-UNETR (faster, less memory)"""
    return SwinUNETR3D_Small(num_classes=num_classes, **kwargs)

def swin_unetr3d_medium(num_classes=3, **kwargs):
    """Medium Swin-UNETR (balanced capacity and speed)"""
    return SwinUNETR3D_Medium(num_classes=num_classes, **kwargs)

def swin_unetr3d_small_deep(num_classes=3, **kwargs):
    """Small Swin-UNETR with deeper stages (better capacity, moderate speed)"""
    return SwinUNETR3D_Small_Deep(num_classes=num_classes, **kwargs)

def swin_unetr3d_tiny(num_classes=3, **kwargs):
    """Tiny Swin-UNETR (fastest, minimal memory)"""
    return SwinUNETR3D_Tiny(num_classes=num_classes, **kwargs)


if __name__ == '__main__':
    # Test the model
    model = swin_unetr3d_small(num_classes=3)
    x = torch.randn(2, 3, 20, 256, 256)
    
    # Test without features
    logits = model(x, is_feat=False)
    print(f"Logits shape: {logits.shape}")  # Should be [2, 3, 20, 256, 256]
    
    # Test with features
    features, logits = model(x, is_feat=True)
    print(f"Features: {[f.shape for f in features]}")
    print(f"Logits shape: {logits.shape}")
    
    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Total parameters: {total_params:,}")
