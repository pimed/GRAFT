"""
3D U-Net implementation for medical image segmentation
Adapted for MTKD-RL framework with knowledge distillation support

Input: [B, 3, 20, 256, 256] (T2/ADC/DWI volumes)
Output: [B, 2, 20, 256, 256] (segmentation masks)
Features: Support is_feat=True for intermediate feature extraction
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class DoubleConv3D(nn.Module):
    """(convolution => [IN] => ReLU) * 2
    
    Note: Using InstanceNorm3d instead of BatchNorm3d for better performance with small batch sizes.
    InstanceNorm normalizes each sample independently, avoiding issues with running statistics
    in batch sizes of 2. This is standard practice in medical image segmentation (e.g., nnU-Net).
    """
    
    def __init__(self, in_channels, out_channels, mid_channels=None):
        super().__init__()
        if not mid_channels:
            mid_channels = out_channels
        self.double_conv = nn.Sequential(
            nn.Conv3d(in_channels, mid_channels, kernel_size=3, padding=1, bias=False),
            nn.InstanceNorm3d(mid_channels, affine=True),
            nn.ReLU(inplace=True),
            nn.Conv3d(mid_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.InstanceNorm3d(out_channels, affine=True),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.double_conv(x)


class Down3D(nn.Module):
    """Downscaling with maxpool then double conv"""

    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.maxpool_conv = nn.Sequential(
            nn.MaxPool3d(2),
            DoubleConv3D(in_channels, out_channels)
        )

    def forward(self, x):
        return self.maxpool_conv(x)


class Up3D(nn.Module):
    """Upscaling then double conv"""

    def __init__(self, in_channels, out_channels, bilinear=True):
        super().__init__()

        # if bilinear, use the normal convolutions to reduce the number of channels
        if bilinear:
            self.up = nn.Upsample(scale_factor=2, mode='trilinear', align_corners=True)
            self.conv = DoubleConv3D(in_channels, out_channels, in_channels // 2)
        else:
            self.up = nn.ConvTranspose3d(in_channels, in_channels // 2, kernel_size=2, stride=2)
            self.conv = DoubleConv3D(in_channels, out_channels)

    def forward(self, x1, x2):
        x1 = self.up(x1)
        # input is CHW
        diffZ = x2.size()[2] - x1.size()[2]
        diffY = x2.size()[3] - x1.size()[3]
        diffX = x2.size()[4] - x1.size()[4]

        x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2,
                        diffY // 2, diffY - diffY // 2,
                        diffZ // 2, diffZ - diffZ // 2])
        
        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)


class OutConv3D(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(OutConv3D, self).__init__()
        self.conv = nn.Conv3d(in_channels, out_channels, kernel_size=1)

    def forward(self, x):
        return self.conv(x)


class UNet3D(nn.Module):
    """
    3D U-Net for medical image segmentation
    
    Args:
        n_channels: Number of input channels (3 for T2/ADC/DWI)
        n_classes: Number of output classes (2 for background/prostate)
        bilinear: Use bilinear upsampling instead of transpose convolutions
    """
    
    def __init__(self, n_channels=3, n_classes=2, bilinear=False):
        super(UNet3D, self).__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.bilinear = bilinear

        # Encoder (Downsampling path)
        self.inc = DoubleConv3D(n_channels, 64)
        self.down1 = Down3D(64, 128)
        self.down2 = Down3D(128, 256)
        self.down3 = Down3D(256, 512)
        factor = 2 if bilinear else 1
        self.down4 = Down3D(512, 1024 // factor)
        
        # Decoder (Upsampling path)
        self.up1 = Up3D(1024, 512 // factor, bilinear)
        self.up2 = Up3D(512, 256 // factor, bilinear)
        self.up3 = Up3D(256, 128 // factor, bilinear)
        self.up4 = Up3D(128, 64, bilinear)
        
        # Output layer
        self.outc = OutConv3D(64, n_classes)

    def forward(self, x, is_feat=False):
        """
        Forward pass with optional feature extraction for knowledge distillation
        
        Args:
            x: Input tensor [B, 3, 20, 256, 256] OR tuple (tensor, is_feat) for DataParallel
            is_feat: If True, return intermediate features for KD
            
        Returns:
            If is_feat=False: logits [B, 2, 20, 256, 256]
            If is_feat=True: (features_list, logits)
                features_list: List of intermediate features for KD
                logits: Output segmentation [B, 2, 20, 256, 256]
        """
        # Handle DataParallel case where is_feat is passed as a tuple
        if isinstance(is_feat, tuple):
            is_feat = is_feat[0]
        
        # Encoder
        x1 = self.inc(x)     # [B, 64, 20, 256, 256]
        x2 = self.down1(x1)  # [B, 128, 10, 128, 128] 
        x3 = self.down2(x2)  # [B, 256, 5, 64, 64]
        x4 = self.down3(x3)  # [B, 512, 2, 32, 32] (rounded down)
        x5 = self.down4(x4)  # [B, 512, 1, 16, 16] (bottleneck)

        # Decoder
        x = self.up1(x5, x4) # [B, 256, 2, 32, 32]
        x = self.up2(x, x3)  # [B, 128, 5, 64, 64]  
        x = self.up3(x, x2)  # [B, 64, 10, 128, 128]
        x = self.up4(x, x1)  # [B, 64, 20, 256, 256]
        
        logits = self.outc(x) # [B, 2, 20, 256, 256]

        if is_feat:
            # Return features for knowledge distillation
            # Following the pattern from 2D models: return features at different scales
            features = [
                x1,  # Early features [B, 64, 20, 256, 256]
                x2,  # [B, 128, 10, 128, 128]  
                x3,  # [B, 256, 5, 64, 64]
                x4,  # [B, 512, 2, 32, 32]
                x5,  # [B, 512, 1, 16, 16] (bottleneck - penultimate layer)
                x    # [B, 64, 20, 256, 256] (final features before classification)
            ]
            return features, logits
        else:
            return logits

class UNet3D_dec(nn.Module):
    """
    3D U-Net for medical image segmentation
    
    Args:
        n_channels: Number of input channels (3 for T2/ADC/DWI)
        n_classes: Number of output classes (2 for background/prostate)
        bilinear: Use bilinear upsampling instead of transpose convolutions
    """
    
    def __init__(self, n_channels=3, n_classes=2, bilinear=False):
        super(UNet3D_dec, self).__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.bilinear = bilinear

        # Encoder (Downsampling path)
        self.inc = DoubleConv3D(n_channels, 64)
        self.down1 = Down3D(64, 128)
        self.down2 = Down3D(128, 256)
        self.down3 = Down3D(256, 512)
        factor = 2 if bilinear else 1
        self.down4 = Down3D(512, 1024 // factor)
        
        # Decoder (Upsampling path)
        self.up1 = Up3D(1024, 512 // factor, bilinear)
        self.up2 = Up3D(512, 256 // factor, bilinear)
        self.up3 = Up3D(256, 128 // factor, bilinear)
        self.up4 = Up3D(128, 64, bilinear)
        
        # Output layer
        self.outc = OutConv3D(64, n_classes)

    def forward(self, x, is_feat=False):
        """
        Forward pass with optional feature extraction for knowledge distillation
        
        Args:
            x: Input tensor [B, 3, 20, 256, 256] OR tuple (tensor, is_feat) for DataParallel
            is_feat: If True, return intermediate features for KD
            
        Returns:
            If is_feat=False: logits [B, 2, 20, 256, 256]
            If is_feat=True: (features_list, logits)
                features_list: List of intermediate features for KD
                logits: Output segmentation [B, 2, 20, 256, 256]
        """
        # Handle DataParallel case where is_feat is passed as a tuple
        if isinstance(is_feat, tuple):
            is_feat = is_feat[0]
        
        # Encoder
        x1 = self.inc(x)     # [B, 64, 20, 256, 256]
        x2 = self.down1(x1)  # [B, 128, 10, 128, 128] 
        x3 = self.down2(x2)  # [B, 256, 5, 64, 64]
        x4 = self.down3(x3)  # [B, 512, 2, 32, 32] (rounded down)
        x5 = self.down4(x4)  # [B, 512, 1, 16, 16] (bottleneck)

        # Decoder
        d1 = self.up1(x5, x4) # [B, 256, 2, 32, 32]
        d2 = self.up2(d1, x3)  # [B, 128, 5, 64, 64]  
        d3 = self.up3(d2, x2)  # [B, 64, 10, 128, 128]
        d4 = self.up4(d3, x1)  # [B, 64, 20, 256, 256]
        
        logits = self.outc(d4) # [B, 2, 20, 256, 256]

        if is_feat:
            # Return features for knowledge distillation
            # Following the pattern from 2D models: return features at different scales
            features = [
                x1,  # Early features [B, 64, 20, 256, 256]
                x2,  # [B, 128, 10, 128, 128]  
                d1,  # [B, 256, 5, 64, 64]
                d2,  # [B, 512, 2, 32, 32]
                d3,  # [B, 512, 1, 16, 16] (bottleneck - penultimate layer)
                d4    # [B, 64, 20, 256, 256] (final features before classification)
            ]
            return features, logits
        else:
            return logits


class UNet3D_enc(nn.Module):
    """
    3D U-Net for medical image segmentation
    
    Args:
        n_channels: Number of input channels (3 for T2/ADC/DWI)
        n_classes: Number of output classes (2 for background/prostate)
        bilinear: Use bilinear upsampling instead of transpose convolutions
    """
    
    def __init__(self, n_channels=3, n_classes=2, bilinear=False):
        super(UNet3D_enc, self).__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.bilinear = bilinear

        # Encoder (Downsampling path)
        self.inc = DoubleConv3D(n_channels, 64)
        self.down1 = Down3D(64, 128)
        self.down2 = Down3D(128, 256)
        self.down3 = Down3D(256, 512)
        factor = 2 if bilinear else 1
        self.down4 = Down3D(512, 1024 // factor)
        
        # Decoder (Upsampling path)
        self.up1 = Up3D(1024, 512 // factor, bilinear)
        self.up2 = Up3D(512, 256 // factor, bilinear)
        self.up3 = Up3D(256, 128 // factor, bilinear)
        self.up4 = Up3D(128, 64, bilinear)
        
        # Output layer
        self.outc = OutConv3D(64, n_classes)

    def forward(self, x, is_feat=False):
        """
        Forward pass with optional feature extraction for knowledge distillation
        
        Args:
            x: Input tensor [B, 3, 20, 256, 256] OR tuple (tensor, is_feat) for DataParallel
            is_feat: If True, return intermediate features for KD
            
        Returns:
            If is_feat=False: logits [B, 2, 20, 256, 256]
            If is_feat=True: (features_list, logits)
                features_list: List of intermediate features for KD
                logits: Output segmentation [B, 2, 20, 256, 256]
        """
        # Handle DataParallel case where is_feat is passed as a tuple
        if isinstance(is_feat, tuple):
            is_feat = is_feat[0]
        
        # Encoder
        x1 = self.inc(x)     # [B, 64, 20, 256, 256]
        x2 = self.down1(x1)  # [B, 128, 10, 128, 128] 
        x3 = self.down2(x2)  # [B, 256, 5, 64, 64]
        x4 = self.down3(x3)  # [B, 512, 2, 32, 32] (rounded down)
        x5 = self.down4(x4)  # [B, 512, 1, 16, 16] (bottleneck)

        # Decoder
        d1 = self.up1(x5, x4) # [B, 256, 2, 32, 32]
        d2 = self.up2(d1, x3)  # [B, 128, 5, 64, 64]  
        d3 = self.up3(d2, x2)  # [B, 64, 10, 128, 128]
        d4 = self.up4(d3, x1)  # [B, 64, 20, 256, 256]
        
        logits = self.outc(d4) # [B, 2, 20, 256, 256]

        if is_feat:
            # Return features for knowledge distillation
            # Following the pattern from 2D models: return features at different scales
            features = [
                x1,  # Early features [B, 64, 20, 256, 256]
                x2,  # [B, 128, 10, 128, 128]  
                # d1,  # [B, 256, 5, 64, 64]
                # d2,  # [B, 512, 2, 32, 32]
                # d3,  # [B, 512, 1, 16, 16] (bottleneck - penultimate layer)
                # d4    # [B, 64, 20, 256, 256] (final features before classification)
            ]
            return features, logits
        else:
            return logits
        
class UNet3DLight(nn.Module):
    """
    Lightweight 3D U-Net for faster training/testing
    Fewer channels to reduce computational cost
    """
    
    def __init__(self, n_channels=3, n_classes=2, bilinear=True):
        super(UNet3DLight, self).__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.bilinear = bilinear

        # Encoder (Downsampling path) - Reduced channels
        self.inc = DoubleConv3D(n_channels, 32)
        self.down1 = Down3D(32, 64)
        self.down2 = Down3D(64, 128)
        self.down3 = Down3D(128, 256)
        factor = 2 if bilinear else 1
        self.down4 = Down3D(256, 512 // factor)
        
        # Decoder (Upsampling path)
        self.up1 = Up3D(512, 256 // factor, bilinear)
        self.up2 = Up3D(256, 128 // factor, bilinear)  
        self.up3 = Up3D(128, 64 // factor, bilinear)
        self.up4 = Up3D(64, 32, bilinear)
        
        # Output layer
        self.outc = OutConv3D(32, n_classes)

    def forward(self, x, is_feat=False):
        """
        Forward pass with optional feature extraction
        
        Args:
            x: Input tensor [B, 3, 20, 256, 256] OR tuple (tensor, is_feat) for DataParallel
            is_feat: If True, return intermediate features for KD
        """
        # Handle DataParallel case where is_feat is passed as a tuple
        if isinstance(is_feat, tuple):
            is_feat = is_feat[0]
        
        # Encoder
        x1 = self.inc(x)     # [B, 32, 20, 256, 256]
        x2 = self.down1(x1)  # [B, 64, 10, 128, 128]
        x3 = self.down2(x2)  # [B, 128, 5, 64, 64]
        x4 = self.down3(x3)  # [B, 256, 2, 32, 32]
        x5 = self.down4(x4)  # [B, 256, 1, 16, 16] (bottleneck)

        # Decoder
        x = self.up1(x5, x4) # [B, 128, 2, 32, 32]
        x = self.up2(x, x3)  # [B, 64, 5, 64, 64]
        x = self.up3(x, x2)  # [B, 32, 10, 128, 128] 
        x = self.up4(x, x1)  # [B, 32, 20, 256, 256]
        
        logits = self.outc(x) # [B, 2, 20, 256, 256]

        if is_feat:
            # Return features for knowledge distillation
            features = [
                x1,  # [B, 32, 20, 256, 256]
                x2,  # [B, 64, 10, 128, 128]
                x3,  # [B, 128, 5, 64, 64]  
                x4,  # [B, 256, 2, 32, 32]
                x5,  # [B, 256, 1, 16, 16] (penultimate layer)
                x    # [B, 32, 20, 256, 256] (final features)
            ]
            return features, logits
        else:
            return logits


# Factory functions for easy model creation
def unet3d(num_classes=2):
    """Create standard 3D U-Net"""
    return UNet3D(n_channels=3, n_classes=num_classes, bilinear=False)

def unet3d_enc(num_classes=2):
    """Create standard 3D U-Net"""
    return UNet3D_enc(n_channels=3, n_classes=num_classes, bilinear=False)  

def unet3d_dec(num_classes=2):
    """Create standard 3D U-Net"""
    return UNet3D_dec(n_channels=3, n_classes=num_classes, bilinear=False)

def unet3d_light(num_classes=2):
    """Create lightweight 3D U-Net"""  
    return UNet3DLight(n_channels=3, n_classes=num_classes, bilinear=True)


def unet3d_bilinear(num_classes=2):
    """Create 3D U-Net with bilinear upsampling"""
    return UNet3D(n_channels=3, n_classes=num_classes, bilinear=True)


if __name__ == '__main__':
    # Test the models
    device = torch.device('cpu')  # Use CPU for testing without GPU
    
    # Test input: [B=2, C=3, D=20, H=256, W=256]
    test_input = torch.randn(2, 3, 20, 256, 256).to(device)
    
    print("Testing 3D U-Net models...")
    print(f"Input shape: {test_input.shape}")
    
    # Test standard U-Net
    model = unet3d(num_classes=2).to(device)
    print(f"\nStandard U-Net parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    # Test forward pass without features
    with torch.no_grad():
        output = model(test_input, is_feat=False)
        print(f"Output shape (is_feat=False): {output.shape}")
        
        # Test forward pass with features  
        features, output = model(test_input, is_feat=True)
        print(f"Output shape (is_feat=True): {output.shape}")
        print("Feature shapes:")
        for i, feat in enumerate(features):
            print(f"  Feature {i}: {feat.shape}")
    
    # Test lightweight U-Net
    model_light = unet3d_light(num_classes=2).to(device)
    print(f"\nLightweight U-Net parameters: {sum(p.numel() for p in model_light.parameters()):,}")
    
    with torch.no_grad():
        output_light = model_light(test_input, is_feat=False)
        print(f"Light output shape: {output_light.shape}")
        
        features_light, output_light = model_light(test_input, is_feat=True)
        print("Light feature shapes:")
        for i, feat in enumerate(features_light):
            print(f"  Feature {i}: {feat.shape}")
    
    print("\n✅ All tests passed!")