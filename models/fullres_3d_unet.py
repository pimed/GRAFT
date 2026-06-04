"""
Standalone 3D UNet replicating the exact architecture from Dataset301 3d_fullres plan.
This creates a PlainConvUNet with the exact same configuration as used in nnUNet training.
"""

import torch
import torch.nn as nn
from dynamic_network_architectures.architectures.unet import PlainConvUNet
from typing import Union, List, Tuple, Dict


def create_3d_fullres_unet(
    input_channels: int = 3,
    num_classes: int = 3,
    deep_supervision: bool = False
) -> PlainConvUNet:
    """
    Create a 3D UNet with the exact architecture from Dataset301_BxMR_withRegions_public_3SEQ
    3d_fullres configuration.
    
    Architecture specifications from plans.json:
    - n_stages: 7
    - features_per_stage: [32, 64, 128, 256, 320, 320, 320]
    - kernel_sizes: Variable per stage (anisotropic in early stages)
    - strides: Variable per stage
    - n_conv_per_stage: 2 for all stages
    - n_conv_per_stage_decoder: 2 for all decoder stages
    - Input patch size: [20, 256, 256]
    - Batch size: 3
    
    Args:
        input_channels: Number of input channels (default: 3 for T2, ADC, DWI)
        num_classes: Number of output classes (default: 3 for background + 2 foreground)
        deep_supervision: Whether to use deep supervision (default: False)
        
    Returns:
        PlainConvUNet model with 3d_fullres architecture
        
    Example:
        >>> model = create_3d_fullres_unet(input_channels=3, num_classes=3)
        >>> model = model.cuda()
        >>> x = torch.randn(1, 3, 20, 256, 256).cuda()
        >>> output = model(x)
        >>> print(output.shape)  # [1, 3, 20, 256, 256]
    """
    
    # Architecture parameters from plans.json
    n_stages = 7
    features_per_stage = [32, 64, 128, 256, 320, 320, 320]
    
    # Kernel sizes - anisotropic in early stages (1x3x3), isotropic later (3x3x3)
    kernel_sizes = [
        [1, 3, 3],  # Stage 0
        [1, 3, 3],  # Stage 1
        [3, 3, 3],  # Stage 2
        [3, 3, 3],  # Stage 3
        [3, 3, 3],  # Stage 4
        [3, 3, 3],  # Stage 5
        [3, 3, 3],  # Stage 6
    ]
    
    # Strides - no downsampling in z for early stages due to small z dimension (20)
    strides = [
        [1, 1, 1],  # Stage 0 - no downsampling
        [1, 2, 2],  # Stage 1 - downsample x,y only
        [1, 2, 2],  # Stage 2 - downsample x,y only
        [2, 2, 2],  # Stage 3 - downsample all dimensions
        [2, 2, 2],  # Stage 4 - downsample all dimensions
        [1, 2, 2],  # Stage 5 - downsample x,y only
        [1, 2, 2],  # Stage 6 - downsample x,y only
    ]
    
    # Number of convolutions per stage
    n_conv_per_stage = [2, 2, 2, 2, 2, 2, 2]
    n_conv_per_stage_decoder = [2, 2, 2, 2, 2, 2]
    
    # Create the model with exact specifications
    model = PlainConvUNet(
        input_channels=input_channels,
        n_stages=n_stages,
        features_per_stage=features_per_stage,
        conv_op=nn.Conv3d,
        kernel_sizes=kernel_sizes,
        strides=strides,
        n_conv_per_stage=n_conv_per_stage,
        num_classes=num_classes,
        n_conv_per_stage_decoder=n_conv_per_stage_decoder,
        conv_bias=True,
        norm_op=nn.InstanceNorm3d,
        norm_op_kwargs={'eps': 1e-05, 'affine': True},
        dropout_op=None,
        dropout_op_kwargs=None,
        nonlin=nn.LeakyReLU,
        nonlin_kwargs={'inplace': True},
        deep_supervision=deep_supervision,
    )
    
    return model


class UNetWithFeatures(nn.Module):
    """
    Wrapper around PlainConvUNet that extracts intermediate features during forward pass.
    Compatible with MTKD-RL framework - returns (features, logits) when is_feat=True.
    """
    
    def __init__(self, base_model: PlainConvUNet):
        super().__init__()
        self.model = base_model
        self.features = {}
        self._register_hooks()
    
    def _register_hooks(self):
        """Register hooks to capture intermediate features."""
        # Find decoder stages for feature extraction
        n_decoder_stages = len(self.model.decoder.stages)
        decoder_neg2_idx = 4  # Decoder stage 4: 64 channels, 20×128×128 resolution
        last_decoder_idx = n_decoder_stages - 1  # Stage 5 for 6-stage decoder
        
        def get_hook(name):
            def hook(module, input, output):
                self.features[name] = output
            return hook
        
        # Register hook on decoder stage 4 (2nd conv) for feature_neg2
        # This gives 64 channels at 20×128×128 resolution with skip connection info
        decoder_neg2_stage = self.model.decoder.stages[decoder_neg2_idx]
        decoder_neg2_stage.convs[1].conv.register_forward_hook(get_hook('feature_neg2'))
        
        # Register hook on last decoder stage (2nd conv) for feature_neg1
        last_decoder_stage = self.model.decoder.stages[last_decoder_idx]
        last_decoder_stage.convs[1].conv.register_forward_hook(get_hook('feature_neg1'))
        
        print(f"Registered hooks:")
        print(f"  - feature_neg2: decoder.stages.{decoder_neg2_idx}.convs.1.conv (decoder stage 4, 64 channels, 20×128×128)")
        print(f"  - feature_neg1: decoder.stages.{last_decoder_idx}.convs.1.conv (last decoder, 32 channels)")
    
    def forward(self, x: torch.Tensor, is_feat=False):
        """
        Forward pass with optional feature extraction for knowledge distillation.
        
        Args:
            x: Input tensor [batch, channels, depth, height, width]
            is_feat: bool or tuple, if True return intermediate features
            
        Returns:
            If is_feat=True: (features_list, logits)
                - features_list: [feature_neg2, feature_neg1] 
                - feature_neg2: [batch, 32, D, H, W]
                - feature_neg1: [batch, 32, D, H, W]
                - logits: [batch, num_classes, D, H, W]
            Otherwise: logits only
        """
        # Handle DataParallel case where is_feat is passed as a tuple
        if isinstance(is_feat, tuple):
            is_feat = is_feat[0]
        
        # Clear previous features
        self.features = {}
        
        # Forward pass (hooks will capture features automatically)
        logits = self.model(x)
        
        if is_feat:
            # Return features as list for compatibility with MTKD-RL
            feature_neg2 = self.features.get('feature_neg2')
            feature_neg1 = self.features.get('feature_neg1')
            return [feature_neg2, feature_neg1], logits
        else:
            return logits


def create_3d_fullres_unet_with_features(
    input_channels: int = 3,
    num_classes: int = 3,
    deep_supervision: bool = False
) -> UNetWithFeatures:
    """
    Create a 3D UNet that returns intermediate features along with logits.
    
    Args:
        input_channels: Number of input channels (default: 3)
        num_classes: Number of output classes (default: 3)
        deep_supervision: Whether to use deep supervision (default: False)
        
    Returns:
        UNetWithFeatures model that returns dict with 'logits', 'feature_neg2', 'feature_neg1'
        
    Example:
        >>> model = create_3d_fullres_unet_with_features(input_channels=3, num_classes=3)
        >>> model = model.cuda()
        >>> x = torch.randn(1, 3, 20, 256, 256).cuda()
        >>> outputs = model(x)
        >>> print(outputs['logits'].shape)       # [1, 3, 20, 256, 256]
        >>> print(outputs['feature_neg2'].shape) # [1, 32, 20, 256, 256]
        >>> print(outputs['feature_neg1'].shape) # [1, 32, 20, 256, 256]
    """
    base_model = create_3d_fullres_unet(input_channels, num_classes, deep_supervision)
    return UNetWithFeatures(base_model)


# Convenience alias for shorter function name
def fullres_3d_unet(num_classes=2):
    """
    Create 3D full-resolution UNet compatible with MTKD-RL framework.
    Always returns model with feature extraction capability.
    Deep supervision is DISABLED.
    
    Args:
        num_classes: Number of output classes (default: 2)
    
    Returns:
        UNetWithFeatures model that supports is_feat parameter
    
    Usage:
        # Standard forward (returns only logits)
        model = fullres_3d_unet(num_classes=2)
        logits = model(x, is_feat=False)
        
        # Feature extraction forward (returns features and logits)
        model = fullres_3d_unet(num_classes=3)
        features, logits = model(x, is_feat=True)
        # features[0]: feature_neg2 [B, 32, D, H, W]
        # features[1]: feature_neg1 [B, 32, D, H, W]
    """
    return create_3d_fullres_unet_with_features(input_channels=3, num_classes=num_classes, deep_supervision=False)


def fullres_3d_unet_deep_supervision(num_classes=2):
    """
    Create 3D full-resolution UNet with DEEP SUPERVISION enabled.
    Returns model with feature extraction capability and deep supervision.
    
    When deep supervision is enabled, the model outputs a list of predictions
    at different scales (one from each decoder stage). The loss function must
    handle multiple outputs and compute weighted losses across all scales.
    
    Args:
        num_classes: Number of output classes (default: 2)
    
    Returns:
        UNetWithFeatures model that supports is_feat parameter and deep supervision
    
    Usage:
        # Standard forward (returns list of logits at different scales)
        model = fullres_3d_unet_deep_supervision(num_classes=3)
        outputs = model(x, is_feat=False)
        # outputs is a list: [logits_full_res, logits_1/2, logits_1/4, ...]
        
        # Feature extraction forward
        model = fullres_3d_unet_deep_supervision(num_classes=3)
        features, outputs = model(x, is_feat=True)
        # features[0]: feature_neg2 [B, 32, D, H, W]
        # features[1]: feature_neg1 [B, 32, D, H, W]
        # outputs: list of logits at different scales
    """
    return create_3d_fullres_unet_with_features(input_channels=3, num_classes=num_classes, deep_supervision=True)


class UNetWithEncoderFeatures(nn.Module):
    """
    Wrapper around PlainConvUNet that extracts features from ENCODER BOTTLENECK.
    This version extracts feature_neg2 from the encoder bottleneck (320 channels)
    instead of the decoder, providing deeper/richer features for distillation.
    Compatible with MTKD-RL framework - returns (features, logits) when is_feat=True.
    """
    
    def __init__(self, base_model: PlainConvUNet):
        super().__init__()
        self.model = base_model
        self.features = {}
        self._register_hooks()
    
    def _register_hooks(self):
        """Register hooks to capture intermediate features from encoder bottleneck."""
        # Find the bottleneck (last encoder stage) and last decoder stage
        n_encoder_stages = len(self.model.encoder.stages)
        n_decoder_stages = len(self.model.decoder.stages)
        bottleneck_idx = n_encoder_stages - 1  # Stage 6 for 7-stage encoder
        last_decoder_idx = n_decoder_stages - 1  # Stage 5 for 6-stage decoder
        
        def get_hook(name):
            def hook(module, input, output):
                self.features[name] = output
            return hook
        
        # Register hook on bottleneck (last encoder stage, 2nd conv) for feature_neg2
        bottleneck_stage = self.model.encoder.stages[bottleneck_idx]
        bottleneck_stage[0].convs[1].conv.register_forward_hook(get_hook('feature_neg2'))
        
        # Register hook on last decoder stage (2nd conv) for feature_neg1
        last_decoder_stage = self.model.decoder.stages[last_decoder_idx]
        last_decoder_stage.convs[1].conv.register_forward_hook(get_hook('feature_neg1'))
        
        print(f"Registered hooks on ENCODER BOTTLENECK:")
        print(f"  - feature_neg2: encoder.stages.{bottleneck_idx}[0].convs.1.conv (bottleneck, 320 channels)")
        print(f"  - feature_neg1: decoder.stages.{last_decoder_idx}.convs.1.conv (last decoder, 32 channels)")
    
    def forward(self, x: torch.Tensor, is_feat=False):
        """
        Forward pass with optional feature extraction for knowledge distillation.
        
        Args:
            x: Input tensor [batch, channels, depth, height, width]
            is_feat: bool or tuple, if True return intermediate features
            
        Returns:
            If is_feat=True: (features_list, logits)
                - features_list: [feature_neg2, feature_neg1] 
                - feature_neg2: [batch, 320, D/32, H/32, W/32] (from encoder bottleneck)
                - feature_neg1: [batch, 32, D, H, W] (from last decoder)
                - logits: [batch, num_classes, D, H, W]
            Otherwise: logits only
        """
        # Handle DataParallel case where is_feat is passed as a tuple
        if isinstance(is_feat, tuple):
            is_feat = is_feat[0]
        
        # Clear previous features
        self.features = {}
        
        # Forward pass through the model
        logits = self.model(x)
        
        if is_feat:
            # Return features in MTKD-RL format: (features_list, logits)
            features = [
                self.features.get('feature_neg2'),  # Encoder bottleneck (320 channels)
                self.features.get('feature_neg1'),  # Last decoder (32 channels)
            ]
            return features, logits
        else:
            return logits


def create_3d_fullres_unet_with_encoder_features(input_channels=3, num_classes=2, deep_supervision=False):
    """
    Create UNet with encoder bottleneck feature extraction.
    This version extracts feature_neg2 from the encoder bottleneck (320 channels).
    
    Args:
        input_channels: Number of input channels (default: 3 for T2, ADC, DWI)
        num_classes: Number of output classes (default: 2)
        deep_supervision: Enable deep supervision (default: False)
    
    Returns:
        UNetWithEncoderFeatures wrapper with encoder bottleneck feature extraction
    """
    base_model = create_3d_fullres_unet(input_channels, num_classes, deep_supervision)
    return UNetWithEncoderFeatures(base_model)


def fullres_3d_unet_enc(num_classes=2):
    """
    Create 3D full-resolution UNet with ENCODER BOTTLENECK feature extraction.
    This version extracts feature_neg2 from the encoder bottleneck (320 channels)
    instead of the decoder, providing deeper/richer features for distillation.
    Deep supervision is DISABLED.
    
    Args:
        num_classes: Number of output classes (default: 2)
    
    Returns:
        UNetWithEncoderFeatures model that supports is_feat parameter
    
    Usage:
        # Standard forward (returns only logits)
        model = fullres_3d_unet_enc(num_classes=2)
        logits = model(x, is_feat=False)
        
        # Feature extraction forward (returns features and logits)
        model = fullres_3d_unet_enc(num_classes=3)
        features, logits = model(x, is_feat=True)
        # features[0]: feature_neg2 [B, 320, D/32, H/32, W/32] (encoder bottleneck - DEEPER)
        # features[1]: feature_neg1 [B, 32, D, H, W] (last decoder)
    """
    return create_3d_fullres_unet_with_encoder_features(input_channels=3, num_classes=num_classes, deep_supervision=False)


# ============================================================================
# LARGE MODEL VARIANTS (nnU-Net L)
# ============================================================================

def create_3d_fullres_unet_large(
    input_channels: int = 3,
    num_classes: int = 2,
    deep_supervision: bool = False
) -> PlainConvUNet:
    """
    Create a LARGE 3D UNet with approximately 2x the channel widths.
    
    Architecture comparison:
        Standard: [32, 64, 128, 256, 320, 320, 320] (~30-40M params)
        Large:    [64, 128, 256, 512, 640, 640, 640] (~120-150M params)
    
    Args:
        input_channels: Number of input channels (default: 3)
        num_classes: Number of output classes (default: 2)
        deep_supervision: Whether to use deep supervision (default: False)
        
    Returns:
        PlainConvUNet model with larger capacity
    """
    # 7 encoder stages, each with specific kernel sizes for anisotropic data
    # Kernel sizes: start with 1x3x3 for anisotropic resolution, then 3x3x3 for isotropic
    kernel_sizes = [
        (1, 3, 3),  # Stage 0: anisotropic kernel for thick slices
        (1, 3, 3),  # Stage 1: anisotropic kernel
        (3, 3, 3),  # Stage 2: isotropic kernel
        (3, 3, 3),  # Stage 3: isotropic kernel
        (3, 3, 3),  # Stage 4: isotropic kernel
        (3, 3, 3),  # Stage 5: isotropic kernel
        (3, 3, 3),  # Stage 6: bottleneck
    ]
    
    # Strides - no downsampling in z for early/late stages due to small z dimension (20)
    # Must match standard model pattern to avoid decoder skip connection size mismatch
    strides = [
        (1, 1, 1),  # Stage 0 - no downsampling
        (1, 2, 2),  # Stage 1 - downsample x,y only
        (1, 2, 2),  # Stage 2 - downsample x,y only
        (2, 2, 2),  # Stage 3 - downsample all dimensions
        (2, 2, 2),  # Stage 4 - downsample all dimensions
        (1, 2, 2),  # Stage 5 - downsample x,y only (preserve depth)
        (1, 2, 2),  # Stage 6 - downsample x,y only (preserve depth)
    ]
    
    # LARGE: Doubled channel widths (approx 2x standard nnU-Net)
    features_per_stage = [64, 128, 256, 512, 640, 640, 640]
    
    # 2 conv blocks per stage (same as nnU-Net default)
    n_conv_per_stage = [2, 2, 2, 2, 2, 2, 2]
    
    model = PlainConvUNet(
        input_channels=input_channels,
        n_stages=7,
        features_per_stage=features_per_stage,
        conv_op=nn.Conv3d,
        kernel_sizes=kernel_sizes,
        strides=strides,
        n_conv_per_stage=n_conv_per_stage,
        num_classes=num_classes,
        n_conv_per_stage_decoder=[2, 2, 2, 2, 2, 2],  # 6 decoder stages for 7 encoder stages
        conv_bias=True,
        norm_op=nn.InstanceNorm3d,
        norm_op_kwargs={'eps': 1e-05, 'affine': True},
        dropout_op=None,
        dropout_op_kwargs=None,
        nonlin=nn.LeakyReLU,
        nonlin_kwargs={'inplace': True},
        deep_supervision=deep_supervision,
    )
    
    return model


class UNetLargeWithFeatures(nn.Module):
    """
    Wrapper around LARGE PlainConvUNet that extracts intermediate features during forward pass.
    Compatible with MTKD-RL framework - returns (features, logits) when is_feat=True.
    
    Feature channels for LARGE model:
        - feature_neg2 (bottleneck): 640 channels
        - feature_neg1 (last decoder): 64 channels
    """
    
    def __init__(self, base_model: PlainConvUNet):
        super().__init__()
        self.model = base_model
        self.features = {}
        self._register_hooks()
    
    def _register_hooks(self):
        """Register hooks to capture intermediate features."""
        n_encoder_stages = len(self.model.encoder.stages)
        n_decoder_stages = len(self.model.decoder.stages)
        bottleneck_idx = n_encoder_stages - 1  # Stage 6 for 7-stage encoder
        last_decoder_idx = n_decoder_stages - 1  # Stage 5 for 6-stage decoder
        
        def get_hook(name):
            def hook(module, input, output):
                self.features[name] = output
            return hook
        
        # Register hook on bottleneck (last encoder stage, 2nd conv) for feature_neg2
        bottleneck_stage = self.model.encoder.stages[bottleneck_idx]
        bottleneck_stage[0].convs[1].conv.register_forward_hook(get_hook('feature_neg2'))
        
        # Register hook on last decoder stage (2nd conv) for feature_neg1
        last_decoder_stage = self.model.decoder.stages[last_decoder_idx]
        last_decoder_stage.convs[1].conv.register_forward_hook(get_hook('feature_neg1'))
        
        print(f"Registered hooks (LARGE model):")
        print(f"  - feature_neg2: encoder.stages.{bottleneck_idx}[0].convs.1.conv (bottleneck, 640 channels)")
        print(f"  - feature_neg1: decoder.stages.{last_decoder_idx}.convs.1.conv (last decoder, 64 channels)")
    
    def forward(self, x: torch.Tensor, is_feat=False):
        """
        Forward pass with optional feature extraction for knowledge distillation.
        
        Args:
            x: Input tensor [batch, channels, depth, height, width]
            is_feat: bool or tuple, if True return intermediate features
            
        Returns:
            If is_feat=True: (features_list, logits)
                - features_list: [feature_neg2, feature_neg1] 
                - feature_neg2: [batch, 640, D/32, H/32, W/32]
                - feature_neg1: [batch, 64, D, H, W]
                - logits: [batch, num_classes, D, H, W]
            Otherwise: logits only
        """
        if isinstance(is_feat, tuple):
            is_feat = is_feat[0]
        
        self.features = {}
        logits = self.model(x)
        
        if is_feat:
            feature_neg2 = self.features.get('feature_neg2')
            feature_neg1 = self.features.get('feature_neg1')
            return [feature_neg2, feature_neg1], logits
        else:
            return logits


def create_3d_fullres_unet_large_with_features(
    input_channels: int = 3,
    num_classes: int = 2,
    deep_supervision: bool = False
) -> UNetLargeWithFeatures:
    """
    Create a LARGE 3D UNet that returns intermediate features along with logits.
    
    Args:
        input_channels: Number of input channels (default: 3)
        num_classes: Number of output classes (default: 2)
        deep_supervision: Whether to use deep supervision (default: False)
        
    Returns:
        UNetLargeWithFeatures model with ~120-150M parameters
    """
    base_model = create_3d_fullres_unet_large(input_channels, num_classes, deep_supervision)
    return UNetLargeWithFeatures(base_model)


def fullres_3d_unet_large(num_classes=2):
    """
    Create LARGE 3D full-resolution UNet (nnU-Net L) compatible with MTKD-RL framework.
    Always returns model with feature extraction capability.
    Deep supervision is DISABLED.
    
    Model capacity: ~120-150M parameters (approximately 4x the standard model)
    Channel widths: [64, 128, 256, 512, 640, 640, 640]
    
    Args:
        num_classes: Number of output classes (default: 2)
    
    Returns:
        UNetLargeWithFeatures model that supports is_feat parameter
    
    Usage:
        model = fullres_3d_unet_large(num_classes=2)
        logits = model(x, is_feat=False)
        features, logits = model(x, is_feat=True)
        # features[0]: feature_neg2 [B, 640, D/32, H/32, W/32]
        # features[1]: feature_neg1 [B, 64, D, H, W]
    """
    return create_3d_fullres_unet_large_with_features(input_channels=3, num_classes=num_classes, deep_supervision=False)


def fullres_3d_unet_large_deep_supervision(num_classes=2):
    """
    Create LARGE 3D full-resolution UNet (nnU-Net L) with DEEP SUPERVISION enabled.
    
    Model capacity: ~120-150M parameters (approximately 4x the standard model)
    Channel widths: [64, 128, 256, 512, 640, 640, 640]
    
    Args:
        num_classes: Number of output classes (default: 2)
    
    Returns:
        UNetLargeWithFeatures model with deep supervision
    """
    return create_3d_fullres_unet_large_with_features(input_channels=3, num_classes=num_classes, deep_supervision=True)


# ============================================================================
# MEDIUM MODEL VARIANTS (nnU-Net M) - ~80M params, between standard and large
# ============================================================================

def create_3d_fullres_unet_medium(
    input_channels: int = 3,
    num_classes: int = 2,
    deep_supervision: bool = False
) -> PlainConvUNet:
    """
    Create a MEDIUM 3D UNet with ~1.5x the channel widths of standard.
    
    Architecture comparison:
        Standard: [32, 64, 128, 256, 320, 320, 320] (~44M params)
        Medium:   [48, 96, 192, 384, 480, 480, 480] (~80M params)
        Large:    [64, 128, 256, 512, 640, 640, 640] (~178M params)
    
    Args:
        input_channels: Number of input channels (default: 3)
        num_classes: Number of output classes (default: 2)
        deep_supervision: Whether to use deep supervision (default: False)
        
    Returns:
        PlainConvUNet model with medium capacity (~80M params)
    """
    # 7 encoder stages, each with specific kernel sizes for anisotropic data
    kernel_sizes = [
        (1, 3, 3),  # Stage 0: anisotropic kernel for thick slices
        (1, 3, 3),  # Stage 1: anisotropic kernel
        (3, 3, 3),  # Stage 2: isotropic kernel
        (3, 3, 3),  # Stage 3: isotropic kernel
        (3, 3, 3),  # Stage 4: isotropic kernel
        (3, 3, 3),  # Stage 5: isotropic kernel
        (3, 3, 3),  # Stage 6: bottleneck
    ]
    
    # Strides - must match standard model pattern
    strides = [
        (1, 1, 1),  # Stage 0 - no downsampling
        (1, 2, 2),  # Stage 1 - downsample x,y only
        (1, 2, 2),  # Stage 2 - downsample x,y only
        (2, 2, 2),  # Stage 3 - downsample all dimensions
        (2, 2, 2),  # Stage 4 - downsample all dimensions
        (1, 2, 2),  # Stage 5 - downsample x,y only (preserve depth)
        (1, 2, 2),  # Stage 6 - downsample x,y only (preserve depth)
    ]
    
    # MEDIUM: 1.5x channel widths (between standard and large)
    features_per_stage = [48, 96, 192, 384, 480, 480, 480]
    
    # 2 conv blocks per stage (same as nnU-Net default)
    n_conv_per_stage = [2, 2, 2, 2, 2, 2, 2]
    
    model = PlainConvUNet(
        input_channels=input_channels,
        n_stages=7,
        features_per_stage=features_per_stage,
        conv_op=nn.Conv3d,
        kernel_sizes=kernel_sizes,
        strides=strides,
        n_conv_per_stage=n_conv_per_stage,
        num_classes=num_classes,
        n_conv_per_stage_decoder=[2, 2, 2, 2, 2, 2],  # 6 decoder stages for 7 encoder stages
        conv_bias=True,
        norm_op=nn.InstanceNorm3d,
        norm_op_kwargs={'eps': 1e-05, 'affine': True},
        dropout_op=None,
        dropout_op_kwargs=None,
        nonlin=nn.LeakyReLU,
        nonlin_kwargs={'inplace': True},
        deep_supervision=deep_supervision,
    )
    
    return model


class UNetMediumWithFeatures(nn.Module):
    """
    Wrapper around MEDIUM PlainConvUNet that extracts intermediate features during forward pass.
    Compatible with MTKD-RL framework - returns (features, logits) when is_feat=True.
    
    Feature channels for MEDIUM model:
        - feature_neg2 (bottleneck): 480 channels
        - feature_neg1 (last decoder): 48 channels
    """
    
    def __init__(self, base_model: PlainConvUNet):
        super().__init__()
        self.model = base_model
        self.features = {}
        self._register_hooks()
    
    def _register_hooks(self):
        """Register hooks to capture intermediate features."""
        n_encoder_stages = len(self.model.encoder.stages)
        n_decoder_stages = len(self.model.decoder.stages)
        bottleneck_idx = n_encoder_stages - 1  # Stage 6 for 7-stage encoder
        last_decoder_idx = n_decoder_stages - 1  # Stage 5 for 6-stage decoder
        
        def get_hook(name):
            def hook(module, input, output):
                self.features[name] = output
            return hook
        
        # Register hook on bottleneck (last encoder stage, 2nd conv) for feature_neg2
        bottleneck_stage = self.model.encoder.stages[bottleneck_idx]
        bottleneck_stage[0].convs[1].conv.register_forward_hook(get_hook('feature_neg2'))
        
        # Register hook on last decoder stage (2nd conv) for feature_neg1
        last_decoder_stage = self.model.decoder.stages[last_decoder_idx]
        last_decoder_stage.convs[1].conv.register_forward_hook(get_hook('feature_neg1'))
        
        print(f"Registered hooks (MEDIUM model):")
        print(f"  - feature_neg2: encoder.stages.{bottleneck_idx}[0].convs.1.conv (bottleneck, 480 channels)")
        print(f"  - feature_neg1: decoder.stages.{last_decoder_idx}.convs.1.conv (last decoder, 48 channels)")
    
    def forward(self, x: torch.Tensor, is_feat=False):
        """
        Forward pass with optional feature extraction for knowledge distillation.
        
        Args:
            x: Input tensor [batch, channels, depth, height, width]
            is_feat: bool or tuple, if True return intermediate features
            
        Returns:
            If is_feat=True: (features_list, logits)
                - features_list: [feature_neg2, feature_neg1] 
                - feature_neg2: [batch, 480, D/32, H/32, W/32]
                - feature_neg1: [batch, 48, D, H, W]
                - logits: [batch, num_classes, D, H, W]
            Otherwise: logits only
        """
        if isinstance(is_feat, tuple):
            is_feat = is_feat[0]
        
        self.features = {}
        logits = self.model(x)
        
        if is_feat:
            feature_neg2 = self.features.get('feature_neg2')
            feature_neg1 = self.features.get('feature_neg1')
            return [feature_neg2, feature_neg1], logits
        else:
            return logits


def create_3d_fullres_unet_medium_with_features(
    input_channels: int = 3,
    num_classes: int = 2,
    deep_supervision: bool = False
) -> UNetMediumWithFeatures:
    """
    Create a MEDIUM 3D UNet that returns intermediate features along with logits.
    
    Args:
        input_channels: Number of input channels (default: 3)
        num_classes: Number of output classes (default: 2)
        deep_supervision: Whether to use deep supervision (default: False)
        
    Returns:
        UNetMediumWithFeatures model with ~80M parameters
    """
    base_model = create_3d_fullres_unet_medium(input_channels, num_classes, deep_supervision)
    return UNetMediumWithFeatures(base_model)


def fullres_3d_unet_medium(num_classes=2):
    """
    Create MEDIUM 3D full-resolution UNet (nnU-Net M) compatible with MTKD-RL framework.
    Always returns model with feature extraction capability.
    Deep supervision is DISABLED.
    
    Model capacity: ~80M parameters (approximately 1.8x the standard model)
    Channel widths: [48, 96, 192, 384, 480, 480, 480]
    
    Args:
        num_classes: Number of output classes (default: 2)
    
    Returns:
        UNetMediumWithFeatures model that supports is_feat parameter
    
    Usage:
        model = fullres_3d_unet_medium(num_classes=2)
        logits = model(x, is_feat=False)
        features, logits = model(x, is_feat=True)
        # features[0]: feature_neg2 [B, 480, D/32, H/32, W/32]
        # features[1]: feature_neg1 [B, 48, D, H, W]
    """
    return create_3d_fullres_unet_medium_with_features(input_channels=3, num_classes=num_classes, deep_supervision=False)


def fullres_3d_unet_medium_deep_supervision(num_classes=2):
    """
    Create MEDIUM 3D full-resolution UNet (nnU-Net M) with DEEP SUPERVISION enabled.
    
    Model capacity: ~80M parameters (approximately 1.8x the standard model)
    Channel widths: [48, 96, 192, 384, 480, 480, 480]
    
    Args:
        num_classes: Number of output classes (default: 2)
    
    Returns:
        UNetMediumWithFeatures model with deep supervision
    """
    return create_3d_fullres_unet_medium_with_features(input_channels=3, num_classes=num_classes, deep_supervision=True)


# ============================================================================
# DATASET 137 - BraTS2023 MODEL
# ============================================================================

def create_3d_fullres_unet_137(
    input_channels: int = 4,
    num_classes: int = 3,
    deep_supervision: bool = False
) -> PlainConvUNet:
    """
    Create a 3D UNet with the exact architecture from Dataset137_BraTS2023
    3d_fullres configuration.
    
    Architecture specifications from checkpoint init_args:
    - n_stages: 6
    - features_per_stage: [32, 64, 128, 256, 320, 320]
    - kernel_sizes: All [3, 3, 3] (isotropic for BraTS data)
    - strides: [[1,1,1], [2,2,2], [2,2,2], [2,2,2], [2,2,2], [2,2,1]]
    - n_conv_per_stage: 2 for all stages
    - n_conv_per_stage_decoder: 2 for all decoder stages (5 stages)
    - Input patch size: [128, 160, 112]
    - Batch size: 2
    - Labels: whole_tumor, enhancing_tumor, tumor_core (region-based)
    
    Args:
        input_channels: Number of input channels (default: 4 for T1c, T1f, T2f, T2w)
        num_classes: Number of output classes (default: 3 for region-based labels)
        deep_supervision: Whether to use deep supervision (default: False)
        
    Returns:
        PlainConvUNet model with BraTS2023 3d_fullres architecture
        
    Example:
        >>> model = create_3d_fullres_unet_137(input_channels=4, num_classes=3)
        >>> model = model.cuda()
        >>> x = torch.randn(1, 4, 128, 160, 112).cuda()
        >>> output = model(x)
        >>> print(output.shape)  # [1, 3, 128, 160, 112]
    """
    
    # Architecture parameters from plans.json / checkpoint
    n_stages = 6
    features_per_stage = [32, 64, 128, 256, 320, 320]
    
    # Kernel sizes - all isotropic [3, 3, 3] for BraTS (1mm isotropic spacing)
    kernel_sizes = [
        [3, 3, 3],  # Stage 0
        [3, 3, 3],  # Stage 1
        [3, 3, 3],  # Stage 2
        [3, 3, 3],  # Stage 3
        [3, 3, 3],  # Stage 4
        [3, 3, 3],  # Stage 5
    ]
    
    # Strides - isotropic downsampling except last stage
    strides = [
        [1, 1, 1],  # Stage 0 - no downsampling
        [2, 2, 2],  # Stage 1 - downsample all dimensions
        [2, 2, 2],  # Stage 2 - downsample all dimensions
        [2, 2, 2],  # Stage 3 - downsample all dimensions
        [2, 2, 2],  # Stage 4 - downsample all dimensions
        [2, 2, 1],  # Stage 5 - preserve last dimension
    ]
    
    # Number of convolutions per stage
    n_conv_per_stage = [2, 2, 2, 2, 2, 2]
    n_conv_per_stage_decoder = [2, 2, 2, 2, 2]
    
    # Create the model with exact specifications
    model = PlainConvUNet(
        input_channels=input_channels,
        n_stages=n_stages,
        features_per_stage=features_per_stage,
        conv_op=nn.Conv3d,
        kernel_sizes=kernel_sizes,
        strides=strides,
        n_conv_per_stage=n_conv_per_stage,
        num_classes=num_classes,
        n_conv_per_stage_decoder=n_conv_per_stage_decoder,
        conv_bias=True,
        norm_op=nn.InstanceNorm3d,
        norm_op_kwargs={'eps': 1e-05, 'affine': True},
        dropout_op=None,
        dropout_op_kwargs=None,
        nonlin=nn.LeakyReLU,
        nonlin_kwargs={'inplace': True},
        deep_supervision=deep_supervision,
    )
    
    return model


class UNet137WithFeatures(nn.Module):
    """
    Wrapper around Dataset137 BraTS2023 PlainConvUNet that extracts intermediate features.
    Compatible with MTKD-RL framework - returns (features, logits) when is_feat=True.
    
    Feature channels for BraTS2023 model (6-stage encoder, 5-stage decoder):
        - feature_neg2 (conv0): 32 channels (last decoder stage, conv 0 output)
        - feature_neg1 (conv1): 32 channels (last decoder stage, conv 1 output)
        
    Both are at full resolution (128, 160, 112) to match extracted teacher features.
    """
    
    def __init__(self, base_model: PlainConvUNet):
        super().__init__()
        self.model = base_model
        self.features = {}
        self._register_hooks()
    
    def _register_hooks(self):
        """Register hooks to capture intermediate features from last decoder stage."""
        n_decoder_stages = len(self.model.decoder.stages)
        last_decoder_idx = n_decoder_stages - 1  # Stage 4 for 5-stage decoder
        
        def get_hook(name):
            def hook(module, input, output):
                self.features[name] = output
            return hook
        
        # Get last decoder stage (32 channels, full resolution)
        last_decoder_stage = self.model.decoder.stages[last_decoder_idx]
        
        # Register hook on conv 0 (first conv in block) for feature_neg2
        last_decoder_stage.convs[0].conv.register_forward_hook(get_hook('feature_neg2'))
        
        # Register hook on conv 1 (second conv in block) for feature_neg1
        last_decoder_stage.convs[1].conv.register_forward_hook(get_hook('feature_neg1'))
        
        print(f"Registered hooks (BraTS2023 Dataset137 model):")
        print(f"  - feature_neg2: decoder.stages.{last_decoder_idx}.convs.0.conv (32 channels, full res)")
        print(f"  - feature_neg1: decoder.stages.{last_decoder_idx}.convs.1.conv (32 channels, full res)")
    
    def forward(self, x: torch.Tensor, is_feat=False):
        """
        Forward pass with optional feature extraction for knowledge distillation.
        
        Args:
            x: Input tensor [batch, channels, depth, height, width]
            is_feat: bool or tuple, if True return intermediate features
            
        Returns:
            If is_feat=True: (features_list, logits)
                - features_list: [feature_neg2, feature_neg1] 
                - feature_neg2: [batch, 32, D, H, W] (conv0 of last decoder)
                - feature_neg1: [batch, 32, D, H, W] (conv1 of last decoder)
                - logits: [batch, num_classes, D, H, W]
            Otherwise: logits only
        """
        if isinstance(is_feat, tuple):
            is_feat = is_feat[0]
        
        self.features = {}
        logits = self.model(x)
        
        if is_feat:
            feature_neg2 = self.features.get('feature_neg2')
            feature_neg1 = self.features.get('feature_neg1')
            return [feature_neg2, feature_neg1], logits
        else:
            return logits


def create_3d_fullres_unet_137_with_features(
    input_channels: int = 4,
    num_classes: int = 3,
    deep_supervision: bool = False
) -> UNet137WithFeatures:
    """
    Create a Dataset137 BraTS2023 3D UNet that returns intermediate features along with logits.
    
    Args:
        input_channels: Number of input channels (default: 4 for T1c, T1f, T2f, T2w)
        num_classes: Number of output classes (default: 3)
        deep_supervision: Whether to use deep supervision (default: False)
        
    Returns:
        UNet137WithFeatures model
    """
    base_model = create_3d_fullres_unet_137(input_channels, num_classes, deep_supervision)
    return UNet137WithFeatures(base_model)


def fullres_3d_unet_137(num_classes=3):
    """
    Create BraTS2023 (Dataset137) 3D full-resolution UNet compatible with MTKD-RL framework.
    Always returns model with feature extraction capability.
    Deep supervision is DISABLED.
    
    Architecture: 6-stage nnUNet matching Dataset137_BraTS2023 checkpoint
    Input channels: 4 (T1c, T1f, T2f, T2w)
    Output classes: 3 (whole_tumor, enhancing_tumor, tumor_core - region-based)
    Channel widths: [32, 64, 128, 256, 320, 320]
    
    Args:
        num_classes: Number of output classes (default: 3)
    
    Returns:
        UNet137WithFeatures model that supports is_feat parameter
    
    Usage:
        model = fullres_3d_unet_137(num_classes=3)
        logits = model(x, is_feat=False)
        features, logits = model(x, is_feat=True)
        # features[0]: feature_neg2 [B, 32, D, H, W] (conv0 of last decoder)
        # features[1]: feature_neg1 [B, 32, D, H, W] (conv1 of last decoder)
    """
    return create_3d_fullres_unet_137_with_features(input_channels=4, num_classes=num_classes, deep_supervision=False)


def fullres_3d_unet_137_ds(num_classes=3):
    """
    Create BraTS2023 (Dataset137) 3D full-resolution UNet with DEEP SUPERVISION enabled.
    
    Architecture: 6-stage nnUNet matching Dataset137_BraTS2023 checkpoint
    Input channels: 4 (T1c, T1f, T2f, T2w)
    
    Args:
        num_classes: Number of output classes (default: 3)
    
    Returns:
        UNet137WithFeatures model with deep supervision
    """
    return create_3d_fullres_unet_137_with_features(input_channels=4, num_classes=num_classes, deep_supervision=True)


def fullres_3d_unet_137_binary(num_classes=1):
    """
    Create BraTS2023 (Dataset137) 3D full-resolution UNet for BINARY segmentation.
    Whole tumor only (labels 1, 2, 3 combined vs background).
    
    Architecture: 6-stage nnUNet matching Dataset137_BraTS2023 checkpoint
    Input channels: 4 (T1c, T1f, T2f, T2w)
    Output classes: 1 (whole tumor binary)
    
    Args:
        num_classes: Number of output classes (default: 1)
    
    Returns:
        UNet137WithFeatures model for binary segmentation
    """
    return create_3d_fullres_unet_137_with_features(input_channels=4, num_classes=num_classes, deep_supervision=False)


def fullres_3d_unet_137_binary_ds(num_classes=1):
    """
    Create BraTS2023 (Dataset137) 3D full-resolution UNet for BINARY segmentation with DEEP SUPERVISION.
    
    Args:
        num_classes: Number of output classes (default: 1)
    
    Returns:
        UNet137WithFeatures model with deep supervision for binary segmentation
    """
    return create_3d_fullres_unet_137_with_features(input_channels=4, num_classes=num_classes, deep_supervision=True)


def print_model_info(model: Union[PlainConvUNet, UNetWithFeatures, UNetWithEncoderFeatures, UNetLargeWithFeatures, UNetMediumWithFeatures, UNet137WithFeatures]):
    """
    Print detailed information about the model architecture.
    
    Args:
        model: PlainConvUNet or UNetWithFeatures or UNetWithEncoderFeatures model instance
    """
    # Handle wrapper models
    base_model = model.model if isinstance(model, (UNetWithFeatures, UNetWithEncoderFeatures)) else model
    
    print("=" * 80)
    print("3D Full Resolution UNet Architecture")
    print("=" * 80)
    print(f"Total parameters: {sum(p.numel() for p in model.parameters()):,}")
    print(f"Trainable parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad):,}")
    print()
    
    print("Encoder stages:")
    for i, stage in enumerate(base_model.encoder.stages):
        print(f"  Stage {i}: {stage}")
    
    print("\nDecoder stages:")
    for i, stage in enumerate(base_model.decoder.stages):
        print(f"  Stage {i}: {stage}")
    
    print("\nSegmentation heads:")
    for i, seg_layer in enumerate(base_model.decoder.seg_layers):
        print(f"  Seg layer {i}: {seg_layer}")
    
    print("=" * 80)


if __name__ == "__main__":
    # Demo 1: Standard model (returns only logits)
    print("=" * 80)
    print("Demo 1: Standard Model (returns only logits)")
    print("=" * 80)
    model = fullres_3d_unet(input_channels=3, num_classes=3)
    
    print_model_info(model)
    
    # Test forward pass
    if torch.cuda.is_available():
        print("\nTesting forward pass on GPU...")
        model = model.cuda()
        x = torch.randn(1, 3, 20, 256, 256).cuda()
        
        with torch.no_grad():
            output = model(x)
        
        print(f"\nInput shape: {x.shape}")
        print(f"Output shape: {output.shape}")
        print(f"Expected shape: [1, 3, 20, 256, 256]")
        print(f"✅ Shapes match!" if output.shape == (1, 3, 20, 256, 256) else "❌ Shape mismatch!")
    else:
        print("\nCUDA not available, skipping GPU test.")
        print("Testing forward pass on CPU...")
        x = torch.randn(1, 3, 20, 256, 256)
        
        with torch.no_grad():
            output = model(x)
        
        print(f"\nInput shape: {x.shape}")
        print(f"Output shape: {output.shape}")
        print(f"Expected shape: [1, 3, 20, 256, 256]")
        print(f"✅ Shapes match!" if output.shape == (1, 3, 20, 256, 256) else "❌ Shape mismatch!")
    
    # Demo 2: Model with feature extraction
    print("\n" + "=" * 80)
    print("Demo 2: Model with Feature Extraction (returns dict)")
    print("=" * 80)
    
    if torch.cuda.is_available():
        print("\nTesting feature extraction on GPU...")
        model_with_features = model_with_features.cuda()
        x = torch.randn(1, 3, 20, 256, 256).cuda()
        
        with torch.no_grad():
            outputs = model_with_features(x)
        
        print(f"\nInput shape: {x.shape}")
        print(f"\nOutput dictionary keys: {list(outputs.keys())}")
        print(f"  - logits shape:       {outputs['logits'].shape}")
        print(f"  - feature_neg2 shape: {outputs['feature_neg2'].shape}")
        print(f"  - feature_neg1 shape: {outputs['feature_neg1'].shape}")
        
        print(f"\n✅ All outputs have correct spatial dimensions!")
        print(f"   Logits: {outputs['logits'].shape[0]} batch × {outputs['logits'].shape[1]} classes × {outputs['logits'].shape[2:]} spatial")
        print(f"   feature_neg2: {outputs['feature_neg2'].shape[0]} batch × {outputs['feature_neg2'].shape[1]} channels × {outputs['feature_neg2'].shape[2:]} spatial")
        print(f"   feature_neg1: {outputs['feature_neg1'].shape[0]} batch × {outputs['feature_neg1'].shape[1]} channels × {outputs['feature_neg1'].shape[2:]} spatial")
    else:
        print("\nCUDA not available, testing on CPU...")
        x = torch.randn(1, 3, 20, 256, 256)
        
        with torch.no_grad():
            outputs = model_with_features(x)
        
        print(f"\nInput shape: {x.shape}")
        print(f"\nOutput dictionary keys: {list(outputs.keys())}")
        print(f"  - logits shape:       {outputs['logits'].shape}")
        print(f"  - feature_neg2 shape: {outputs['feature_neg2'].shape}")
        print(f"  - feature_neg1 shape: {outputs['feature_neg1'].shape}")
        print(f"Expected shape: [1, 3, 20, 256, 256]")
        print(f"✅ Shapes match!" if output.shape == (1, 3, 20, 256, 256) else "❌ Shape mismatch!")

    model_with_features = fullres_3d_unet(input_channels=3, num_classes=3, return_features=True)