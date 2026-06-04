from .feature_mse_mtkd_rl import FeatureKLLoss, FeatureMSELoss
from .segmentation_3d_losses import (
    DiceLoss, IoULoss, FocalLoss, CombinedLoss, TverskyLoss,
    nnUNetDiceLoss, DC_and_CE_loss,  # nnUNet-compatible losses
    get_segmentation_criterion, get_pimed_criterion,
    FeatureKLLoss3D, FeatureMSELoss3D
)
from .normalized_feature_losses import (
    NormalizedFeatureMSELoss, NormalizedFeatureMSELoss3D,
    RobustFeatureMSELoss, RobustFeatureMSELoss3D,
    MaskedFeatureMSELoss3D,
    get_feature_mse_loss
)
from .contrastive_kd import (
    ContrastiveKDLoss, RelationKDLoss, AngularMarginKDLoss,
    get_contrastive_kd_loss
)
