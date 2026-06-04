import torch
import torch.nn as nn
import torch.nn.functional as F

__all__ = ['Policy', 'PolicyTrans']

class Policy(nn.Module):
    def __init__(self, input_size, output_size):
        super(Policy, self).__init__()

        self.head = nn.Sequential(
            nn.Linear(input_size, 128, bias=False),
            nn.ReLU(True),
            nn.Linear(128, output_size, bias=False),
            nn.Sigmoid()
        )

    def forward(self, input):
        output = self.head(input)
        return output 


class PolicyTrans(nn.Module):
    def __init__(self, input_size, teacher_num, dynamic=False, use_3d=False, enable_logits_actions=False, num_scalars_per_teacher=3, enable_neg1_actions=True, enable_neg2_actions=False, use_accuracy_gated_diversity=True, quality_temperature=0.1):
        super(PolicyTrans, self).__init__()
        self.teacher_num = teacher_num
        self.use_3d = use_3d
        self.enable_logits_actions = enable_logits_actions
        self.enable_neg1_actions = enable_neg1_actions
        self.enable_neg2_actions = enable_neg2_actions
        self.use_accuracy_gated_diversity = use_accuracy_gated_diversity
        self.num_scalars_per_teacher = num_scalars_per_teacher  # 2 (cos_sim, dice) or 3 (cos_sim, dice, disagreement)
        self.quality_temperature = quality_temperature  # Temperature for sharpening Dice-based quality weights
        
        # Convolutional encoders for each teacher's spatial features
        # These will encode teacher embeddings and logits before pooling
        # NOTE: input_size is now ignored - we use dynamic conv layer creation
        self.teacher_encoders = nn.ModuleList()
        
        for idx in range(teacher_num):
            # Create placeholder encoders - first conv layer will be initialized dynamically
            if use_3d:
                # 3D conv encoder with LESS AGGRESSIVE pooling to preserve spatial info
                # Pipeline: [B, C, 20, 256, 256] -> [B, 16, 20, 256, 256] -> [B, 16, 5, 32, 32]
                #        -> [B, 32, 5, 32, 32] -> [B, 32, 2, 8, 8] -> [B, 64, 2, 8, 8] -> [B, 64, 1, 1, 1]
                # This preserves more spatial info (5×32×32 = 5120 voxels vs 128 before)
                encoder = nn.ModuleList([
                    None,  # Placeholder for first conv (will be created dynamically)
                    nn.ReLU(),
                    nn.AdaptiveAvgPool3d((5, 32, 32)),  # Less aggressive: keep 5×32×32 = 5120 voxels
                    nn.Conv3d(16, 32, kernel_size=3, padding=1),
                    nn.ReLU(),
                    nn.AdaptiveAvgPool3d((2, 8, 8)),  # Further reduce to 2×8×8 = 128
                    nn.Conv3d(32, 64, kernel_size=3, padding=1),
                    nn.ReLU(),
                    nn.AdaptiveAvgPool3d((1, 1, 1))  # Final pooling
                ])
            else:
                # 2D conv encoder (also less aggressive)
                encoder = nn.ModuleList([
                    None,  # Placeholder for first conv
                    nn.ReLU(),
                    nn.AdaptiveAvgPool2d((16, 16)),  # Less aggressive
                    nn.Conv2d(16, 32, kernel_size=3, padding=1),
                    nn.ReLU(),
                    nn.AdaptiveAvgPool2d((4, 4)),
                    nn.Conv2d(32, 64, kernel_size=3, padding=1),
                    nn.ReLU(),
                    nn.AdaptiveAvgPool2d((1, 1))
                ])
            
            self.teacher_encoders.append(encoder)
        
        # Calculate total input size for the MLP
        # Each teacher contributes: 64 (encoded features) + num_scalars_per_teacher (2 or 3 scalars)
        all_input_size = teacher_num * (64 + num_scalars_per_teacher)
        
        self.steam = nn.Sequential(
                nn.Linear(all_input_size, 128, bias=False),
                nn.ReLU())
        # Feature head for neg1 layer distillation weights (only if neg1 is enabled)
        if enable_neg1_actions:
            self.feature_head = nn.Linear(128, teacher_num, bias=True)
        else:
            self.feature_head = None
        
        # Optional neg2 feature head for separate neg2 actions
        if enable_neg2_actions:
            self.neg2_feature_head = nn.Linear(128, teacher_num, bias=True)
        else:
            self.neg2_feature_head = None
        
        # Optional logit head for logits distillation weights
        if enable_logits_actions:
            self.logit_head = nn.Linear(128, teacher_num, bias=True)
        else:
            self.logit_head = None
                
        self.sigmoid = nn.Sigmoid()
        self.softmax = nn.Softmax(dim=1)
        
        # Support both 2D and 3D pooling
        if use_3d:
            self.avgpool = nn.AdaptiveAvgPool3d((1, 1, 1))
        else:
            self.avgpool = nn.AdaptiveAvgPool2d((1, 1))

        self.dynamic = dynamic
        if dynamic:
            # Only need feature weight factor now (no logit distillation)
            self.feature_weight_factor = torch.nn.Parameter(torch.tensor([1., 1., 1.]), requires_grad=True)

    def forward(self, agent_state):
        # agent_state now contains: (teacher_embeddings_neg1, teacher_features_neg2, teacher_logits_list, scalars_list)
        # teacher_embeddings_neg1: list of [B, C, D, H, W] tensors (neg1 layer features) - can be None if not enabled
        # teacher_features_neg2: list of [B, C, D, H, W] tensors (neg2 layer features) - can be None if not enabled
        # teacher_logits_list: list of [B, num_cls, D, H, W] tensors (teacher predictions)
        # scalars_list: list of [B, 3] tensors (feat_cos_sim, dice_mean, disagreement per teacher)
        
        teacher_embeddings_neg1, teacher_features_neg2, teacher_logits, teacher_scalars = agent_state
        
        # Determine device and batch_size from whichever features are available
        if teacher_embeddings_neg1 is not None and len(teacher_embeddings_neg1) > 0:
            device = teacher_embeddings_neg1[0].device
            batch_size = teacher_embeddings_neg1[0].size(0)
        elif teacher_features_neg2 is not None and len(teacher_features_neg2) > 0:
            device = teacher_features_neg2[0].device
            batch_size = teacher_features_neg2[0].size(0)
        else:
            device = teacher_logits[0].device
            batch_size = teacher_logits[0].size(0)
        
        # Encode each teacher's spatial information
        encoded_teachers = []
        
        for idx in range(self.teacher_num):
            # Get teacher logits (always available)
            logits = teacher_logits[idx]   # [B, C_logits, D, H, W]
            
            # Build feature list based on which features are enabled
            features_to_cat = []
            reference_spatial = None
            
            # Add neg1 features if enabled
            if self.enable_neg1_actions and teacher_embeddings_neg1 is not None:
                neg1_feat = teacher_embeddings_neg1[idx]  # [B, C_neg1, D, H, W]
                if neg1_feat is not None:
                    features_to_cat.append(neg1_feat)
                    reference_spatial = neg1_feat.shape[2:]  # Use neg1 spatial dims as reference
            
            # Add neg2 features if enabled
            if self.enable_neg2_actions and teacher_features_neg2 is not None:
                neg2_feat = teacher_features_neg2[idx]  # [B, C_neg2, D, H, W]
                if neg2_feat is not None:
                    # If we have neg1, resize neg2 to match neg1 spatial dims
                    if reference_spatial is not None and neg2_feat.shape[2:] != reference_spatial:
                        if self.use_3d:
                            neg2_feat = F.interpolate(neg2_feat, size=reference_spatial, mode='trilinear', align_corners=True)
                        else:
                            neg2_feat = F.interpolate(neg2_feat, size=reference_spatial, mode='bilinear', align_corners=True)
                    features_to_cat.append(neg2_feat)
                    if reference_spatial is None:
                        reference_spatial = neg2_feat.shape[2:]  # Use neg2 spatial dims as reference
            
            # Resize logits to match feature spatial dimensions
            logits_spatial = logits.shape[2:]
            if reference_spatial is not None and logits_spatial != reference_spatial:
                if self.use_3d:
                    logits = F.interpolate(logits, size=reference_spatial, mode='trilinear', align_corners=True)
                else:
                    logits = F.interpolate(logits, size=reference_spatial, mode='bilinear', align_corners=True)
            
            # Add logits to features
            features_to_cat.append(logits)
            
            # Concatenate all features along channel dimension
            # Result: [B, C_neg1 + C_neg2 + C_logits, D, H, W] (if both enabled)
            #      or [B, C_neg1 + C_logits, D, H, W] (if neg1 only)
            #      or [B, C_neg2 + C_logits, D, H, W] (if neg2 only)
            teacher_spatial = torch.cat(features_to_cat, dim=1)
            
            # Apply conv encoder: [B, C_total, D, H, W] -> [B, 32, 1, 1, 1] or [B, 32, 1, 1]
            # We need to adjust the encoder's first layer to match input channels
            in_channels = teacher_spatial.size(1)
            
            # Dynamically create first conv layer if needed (first forward pass)
            if self.teacher_encoders[idx][0] is None:
                if self.use_3d:
                    self.teacher_encoders[idx][0] = nn.Conv3d(
                        in_channels=in_channels, out_channels=16, kernel_size=3, padding=1
                    ).to(device=device, dtype=teacher_spatial.dtype)
                else:
                    self.teacher_encoders[idx][0] = nn.Conv2d(
                        in_channels=in_channels, out_channels=16, kernel_size=3, padding=1
                    ).to(device=device, dtype=teacher_spatial.dtype)
            
            # Apply encoder layers sequentially
            x = teacher_spatial
            for layer_idx, layer in enumerate(self.teacher_encoders[idx]):
                if layer is not None:
                    # Ensure layer dtype matches input dtype (for mixed precision training)
                    # Only for layers with parameters (conv layers, not ReLU or pooling)
                    if hasattr(layer, 'weight') and layer.weight.dtype != x.dtype:
                        layer.to(dtype=x.dtype)
                    x = layer(x)
            
            # Flatten: [B, 64, 1, 1, 1] or [B, 64, 1, 1] -> [B, 64]
            encoded_spatial = x.view(batch_size, -1)
            
            # Concatenate encoded spatial features with scalars
            # [B, 64] + [B, num_scalars_per_teacher] -> [B, 64 + num_scalars_per_teacher]
            teacher_repr = torch.cat([encoded_spatial, teacher_scalars[idx]], dim=1)
            encoded_teachers.append(teacher_repr)
        
        # Concatenate all teacher representations
        all_teacher_infos = torch.cat(encoded_teachers, dim=1)  # [B, teacher_num * 66]
        
        # MLP to predict feature weights (and optionally logit weights)
        out1 = self.steam(all_teacher_infos)
        
        # neg1 feature weights (only if neg1 is enabled)
        if self.enable_neg1_actions:
            feature_weights = self.softmax(self.feature_head(out1))  # neg1 layer weights
        else:
            feature_weights = None
        
        # Optional neg2 feature weights (separate from neg1)
        if self.enable_neg2_actions:
            neg2_feature_weights = self.softmax(self.neg2_feature_head(out1))
        else:
            neg2_feature_weights = None
        
        # Optional logit weights
        if self.enable_logits_actions:
            logit_weights = self.softmax(self.logit_head(out1))
        else:
            logit_weights = None

        # Compute final weights using heuristics
        # Scalar format: teacher_scalars[i] = [feat_cos_sim, dice_mean, (opt) disagreement]
        # - feat_cos_sim: how well student features match teacher features (higher = better match)
        # - dice_mean: per-sample cancer Dice of this teacher against GT (higher = better teacher)
        if self.dynamic:
            # Extract scalars from teacher_scalars for dynamic weighting
            t_dice = torch.stack([teacher_scalars[i][:, 1] for i in range(self.teacher_num)], dim=1)  # [B, teacher_num]
            t_s_feat_div = torch.stack([teacher_scalars[i][:, 0] for i in range(self.teacher_num)], dim=1)  # [B, teacher_num]
            
            # Weight by teacher quality: higher Dice = better teacher
            # Temperature sharpening: τ=0.1 makes Dice [0.7, 0.3, 0.2] → weights [0.88, 0.09, 0.03]
            weight_loss_t = F.softmax(t_dice / self.quality_temperature, dim=1)
            # Diversity weight: INVERT similarity - favor teachers with DIFFERENT features (more to learn)
            diversity_weight = F.softmax(-t_s_feat_div, dim=1)
            
            # Accuracy-gated diversity: only favor diverse teachers if they're also accurate
            if self.use_accuracy_gated_diversity:
                accuracy_gated_diversity = weight_loss_t * diversity_weight
                accuracy_gated_diversity = accuracy_gated_diversity / (accuracy_gated_diversity.sum(dim=1, keepdim=True) + 1e-8)
            
            f_f = F.softmax(self.feature_weight_factor, dim=0)
            
            # For features (neg1): use learned weights + teacher quality + (optionally) accuracy-gated diversity
            if self.enable_neg1_actions:
                if self.use_accuracy_gated_diversity:
                    all_feature_weights = (f_f[0]*feature_weights + f_f[1]*weight_loss_t + f_f[2]*accuracy_gated_diversity)
                else:
                    all_feature_weights = (f_f[0]*feature_weights + f_f[1]*weight_loss_t) / (f_f[0] + f_f[1])
            else:
                all_feature_weights = None
            
            # For neg2 features: use similar weighting if enabled
            if self.enable_neg2_actions:
                if self.use_accuracy_gated_diversity:
                    all_neg2_feature_weights = (f_f[0]*neg2_feature_weights + f_f[1]*weight_loss_t + f_f[2]*accuracy_gated_diversity)
                else:
                    all_neg2_feature_weights = (f_f[0]*neg2_feature_weights + f_f[1]*weight_loss_t) / (f_f[0] + f_f[1])
            else:
                all_neg2_feature_weights = None
            
            # For logits: use similar weighting if enabled
            if self.enable_logits_actions:
                if self.use_accuracy_gated_diversity:
                    all_logit_weights = (f_f[0]*logit_weights + f_f[1]*weight_loss_t + f_f[2]*accuracy_gated_diversity)
                else:
                    all_logit_weights = (f_f[0]*logit_weights + f_f[1]*weight_loss_t) / (f_f[0] + f_f[1])
            else:
                all_logit_weights = None
        else:
            # Extract scalars for non-dynamic weighting
            t_dice = torch.stack([teacher_scalars[i][:, 1] for i in range(self.teacher_num)], dim=1)
            t_s_feat_div = torch.stack([teacher_scalars[i][:, 0] for i in range(self.teacher_num)], dim=1)
            
            # Weight by teacher quality: higher Dice = better teacher
            # Temperature sharpening: τ=0.1 makes Dice [0.7, 0.3, 0.2] → weights [0.88, 0.09, 0.03]
            weight_loss_t = F.softmax(t_dice / self.quality_temperature, dim=1)
            # Diversity weight: INVERT similarity - favor teachers with DIFFERENT features (more to learn)
            diversity_weight = F.softmax(-t_s_feat_div, dim=1)
            
            # Accuracy-gated diversity: only favor diverse teachers if they're also accurate
            # High accuracy + high diversity = valuable unique knowledge
            # Low accuracy + high diversity = different but wrong - avoid!
            if self.use_accuracy_gated_diversity:
                accuracy_gated_diversity = weight_loss_t * diversity_weight
                accuracy_gated_diversity = accuracy_gated_diversity / (accuracy_gated_diversity.sum(dim=1, keepdim=True) + 1e-8)
            
            # For features (neg1): use learned weights + teacher quality + (optionally) accuracy-gated diversity
            if self.enable_neg1_actions:
                if self.use_accuracy_gated_diversity:
                    all_feature_weights = (feature_weights + weight_loss_t + accuracy_gated_diversity) / 3.
                else:
                    all_feature_weights = (feature_weights + weight_loss_t) / 2.
                print('[PolicyTrans] Computed neg1 feature weights:', feature_weights)
            else:
                all_feature_weights = None
            
            # For neg2 features: use similar weighting if enabled
            if self.enable_neg2_actions:
                if self.use_accuracy_gated_diversity:
                    all_neg2_feature_weights = (neg2_feature_weights + weight_loss_t + accuracy_gated_diversity) / 3.
                else:
                    all_neg2_feature_weights = (neg2_feature_weights + weight_loss_t) / 2.
                print('[PolicyTrans] Computed neg2 feature weights:', neg2_feature_weights)
            else:
                all_neg2_feature_weights = None
            
            if self.enable_neg1_actions or self.enable_neg2_actions:
                print(f'[PolicyTrans] Weight from Dice (quality, τ={self.quality_temperature}):', weight_loss_t)
                if self.use_accuracy_gated_diversity:
                    print('[PolicyTrans] Accuracy-gated diversity:', accuracy_gated_diversity)
            
            # For logits: use similar weighting if enabled
            if self.enable_logits_actions:
                if self.use_accuracy_gated_diversity:
                    all_logit_weights = (logit_weights + weight_loss_t + accuracy_gated_diversity) / 3.
                else:
                    all_logit_weights = (logit_weights + weight_loss_t) / 2.
            else:
                all_logit_weights = None

        # Return weights based on which heads are enabled
        # Build return tuple dynamically based on enabled actions
        result = []
        if self.enable_neg1_actions:
            result.append(all_feature_weights)
        if self.enable_neg2_actions:
            result.append(all_neg2_feature_weights)
        if self.enable_logits_actions:
            result.append(all_logit_weights)
        
        # Return single tensor if only one action type, else tuple
        if len(result) == 1:
            return result[0]
        else:
            return tuple(result)



class PolicyTrans_Simple(nn.Module):
    def __init__(self, input_size, teacher_num, dynamic=False, use_3d=False):
        super(PolicyTrans_Simple, self).__init__()
        self.teacher_num = teacher_num
        self.use_3d = use_3d
        
        self.sim_trans = nn.ModuleList([])
        all_input_size = 0
        for idx in range(teacher_num):
            all_input_size = all_input_size + input_size[idx]
        
        self.steam = nn.Sequential(
                nn.Linear(all_input_size, 128, bias=False),
                nn.ReLU())
        self.logit_head = nn.Linear(128, teacher_num, bias=True)
        self.feature_head = nn.Linear(128, teacher_num, bias=True)
                
        self.sigmoid = nn.Sigmoid()
        self.softmax = nn.Softmax(dim=1)
        
        # Support both 2D and 3D pooling
        if use_3d:
            self.avgpool = nn.AdaptiveAvgPool3d((1, 1, 1))
        else:
            self.avgpool = nn.AdaptiveAvgPool2d((1, 1))

        self.dynamic = dynamic
        if dynamic:
            self.logit_weight_factor = torch.nn.Parameter(torch.tensor([1., 1., 1.]), requires_grad=True)
            self.feature_weight_factor = torch.nn.Parameter(torch.tensor([1., 1., 1.]), requires_grad=True)

    def forward(self, agent_state):
        teacher_infos, t_ces, t_s_logit_div, t_s_feat_div = agent_state

        # Note: Device is already set correctly by train_agent before calling forward
        # These checks are defensive to ensure compatibility
        device = teacher_infos[0].device if len(teacher_infos) > 0 else t_ces.device
        
        # Ensure all inputs are on the same device
        t_ces = t_ces.to(device) if t_ces.device != device else t_ces
        t_s_logit_div = t_s_logit_div.to(device) if t_s_logit_div.device != device else t_s_logit_div
        t_s_feat_div = t_s_feat_div.to(device) if t_s_feat_div.device != device else t_s_feat_div
        teacher_infos = [t.to(device) if t.device != device else t for t in teacher_infos]

        weight_loss_t = (1. - F.softmax(t_ces, dim=1)) / (self.teacher_num - 1)
        weight_loss_t_s_logit_div = F.softmax(t_s_logit_div, dim=1)
        weight_loss_t_s_feat_div = F.softmax(t_s_feat_div, dim=1)

        all_teacher_infos = torch.cat(teacher_infos, dim=1)
        out1 = self.steam(all_teacher_infos)
        logit_weights = self.softmax(self.logit_head(out1))
        feature_weights = self.softmax(self.feature_head(out1))

        if self.dynamic:
            l_f = F.softmax(self.logit_weight_factor, dim=0)
            f_f = F.softmax(self.feature_weight_factor, dim=0)
            all_logit_weights = (l_f[0]*logit_weights + l_f[1]*weight_loss_t + l_f[2]*weight_loss_t_s_logit_div)
            all_feature_weights = (f_f[0] * feature_weights + f_f[1] * weight_loss_t + f_f[2] * weight_loss_t_s_feat_div)
        else:
            all_logit_weights = (logit_weights + weight_loss_t + weight_loss_t_s_logit_div)/ 3.
            all_feature_weights = (feature_weights + weight_loss_t + weight_loss_t_s_feat_div) / 3.


        return all_logit_weights,  all_feature_weights