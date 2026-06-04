from .resnet import resnet8, resnet14, resnet20, resnet32, resnet44, resnet56, resnet110, resnet8x4, resnet110x2, resnet32x4, resnet20x4
from .resnet import resnet8x4_double
from .wrn import wrn_16_1, wrn_16_2, wrn_40_1, wrn_40_2, wrn_28_4, wrn_10_2, wrn_10_1 
from .vgg import vgg19_bn, vgg16_bn, vgg13_bn, vgg11_bn, vgg8_bn
from .mobilenetv2 import mobile_half, mobilenet
from .ShuffleNetv1 import ShuffleV1
from .ShuffleNetv2 import ShuffleV2, ShuffleV2_0_5
from .regnet import RegNetY_400MF, RegNetX_400MF,  RegNetX_200MF
from .policy import Policy, PolicyTrans, PolicyTrans_Simple
from .unet3d import unet3d, unet3d_dec, unet3d_enc, unet3d_light, unet3d_bilinear
from .fullres_3d_unet import fullres_3d_unet, fullres_3d_unet_deep_supervision, fullres_3d_unet_enc, fullres_3d_unet_large, fullres_3d_unet_large_deep_supervision, fullres_3d_unet_medium, fullres_3d_unet_medium_deep_supervision, fullres_3d_unet_137, fullres_3d_unet_137_ds, fullres_3d_unet_137_binary, fullres_3d_unet_137_binary_ds
from .swin_unetr3d import swin_unetr3d, swin_unetr3d_small, swin_unetr3d_medium, swin_unetr3d_small_deep, swin_unetr3d_tiny

model_dict = {
    'resnet8': resnet8,
    'resnet14': resnet14,
    'resnet20': resnet20,
    'resnet32': resnet32,
    'resnet44': resnet44,
    'resnet56': resnet56,
    'resnet110': resnet110,
    'resnet8x4': resnet8x4,
    'resnet8x4_double': resnet8x4_double,
    'resnet32x4': resnet32x4,
    'resnet110x2': resnet110x2, 
    'resnet20x4': resnet20x4,
    'wrn_10_2': wrn_10_2,
    'wrn_10_1': wrn_10_1,
    'wrn_16_1': wrn_16_1,
    'wrn_16_2': wrn_16_2,
    'wrn_40_1': wrn_40_1,
    'wrn_40_2': wrn_40_2,
    'wrn_28_4': wrn_28_4,
    'vgg8': vgg8_bn,
    'vgg11': vgg11_bn,
    'vgg13': vgg13_bn,
    'vgg16': vgg16_bn,
    'vgg19': vgg19_bn,
    'MobileNetV2': mobilenet,
    'ShuffleV1': ShuffleV1,
    'ShuffleV2': ShuffleV2,
    'ShuffleV2_0_5': ShuffleV2_0_5,
    'RegNetY_400MF': RegNetY_400MF, 
    'RegNetX_400MF': RegNetX_400MF,  
    'RegNetX_200MF': RegNetX_200MF,
    'Policy': Policy,
    'PolicyTrans': PolicyTrans,
    'PolicyTrans_Simple': PolicyTrans_Simple,
    # 3D models for medical image segmentation
    'unet3d': unet3d,
    'unet3d_dec': unet3d_dec,
    'unet3d_enc': unet3d_enc,
    'unet3d_light': unet3d_light, 
    'unet3d_bilinear': unet3d_bilinear,
    'fullres_3d_unet': fullres_3d_unet,
    'fullres_3d_unet_ds': fullres_3d_unet_deep_supervision,  # With deep supervision
    'fullres_3d_unet_enc': fullres_3d_unet_enc,  # With encoder bottleneck features
    'fullres_3d_unet_large': fullres_3d_unet_large,  # Large model (~4x params)
    'fullres_3d_unet_large_ds': fullres_3d_unet_large_deep_supervision,  # Large + deep supervision
    'fullres_3d_unet_medium': fullres_3d_unet_medium,  # Medium model (~2.2x params)
    'fullres_3d_unet_medium_ds': fullres_3d_unet_medium_deep_supervision,  # Medium + deep supervision
    # BraTS2023 (Dataset137) models
    'fullres_3d_unet_137': fullres_3d_unet_137,  # BraTS2023 model (4 input channels, 3 output classes)
    'fullres_3d_unet_137_ds': fullres_3d_unet_137_ds,  # BraTS2023 + deep supervision
    'fullres_3d_unet_137_binary': fullres_3d_unet_137_binary,  # BraTS2023 binary (1 output class)
    'fullres_3d_unet_137_binary_ds': fullres_3d_unet_137_binary_ds,  # BraTS2023 binary + deep supervision
    # 3D Transformer-based models
    'swin_unetr3d': swin_unetr3d,
    'swin_unetr3d_small': swin_unetr3d_small,
    'swin_unetr3d_medium': swin_unetr3d_medium,
    'swin_unetr3d_small_deep': swin_unetr3d_small_deep,
    'swin_unetr3d_tiny': swin_unetr3d_tiny,
}
