"""
3D Medical Image Dataloader with Pre-Extracted Teacher Features
Extends the base PIMEDDataset_2D to load pre-computed teacher features and logits
"""

import os
import torch
import numpy as np
from torch.utils.data import Dataset
from collections import OrderedDict
from .pimedloader_3d import PIMEDDataset_3D
from .region_converter import ConvertSegmentationToRegions


class LRUCache:
    """
    Least Recently Used (LRU) cache with size limit.
    
    When the cache reaches max_size, the least recently used item is automatically
    evicted to make room for new items. This prevents unbounded memory growth.
    
    Args:
        max_size (int): Maximum number of items to cache (default: 300)
                       With 0.66GB per case: 300 cases × 8 GPUs = ~1.6TB total
    """
    def __init__(self, max_size=300):
        self.cache = OrderedDict()
        self.max_size = max_size
        self.eviction_count = 0
    
    def get(self, key):
        """Get item from cache, marking it as recently used"""
        if key not in self.cache:
            return None
        # Move to end (most recently used)
        self.cache.move_to_end(key)
        return self.cache[key]
    
    def put(self, key, value):
        """Add item to cache, evicting oldest if at capacity"""
        if key in self.cache:
            # Update existing item and mark as recently used
            self.cache.move_to_end(key)
            self.cache[key] = value
        else:
            # Add new item
            self.cache[key] = value
            # Evict oldest if we exceeded capacity
            if len(self.cache) > self.max_size:
                evicted_key = self.cache.popitem(last=False)[0]  # Remove oldest (first item)
                self.eviction_count += 1
    
    def __contains__(self, key):
        return key in self.cache
    
    def __len__(self):
        return len(self.cache)


class PIMEDDataset_WithFeatures(PIMEDDataset_3D):
    """
    Extended dataloader that loads both raw 3D volumes and pre-extracted teacher features.
    
    This is designed for MTKD-RL framework where teacher features are pre-extracted 
    offline to save GPU memory during student training.
    
    Args:
        features_dir (str): Root directory containing pre-extracted features
                           Structure: features_dir/teacher_name/case_id_features.pt
        teacher_names (list): List of teacher model names (e.g., ['teacher1', 'teacher2', 'teacher3'])
        load_features (bool): Whether to load pre-extracted features (default: True)
                             Set to False if you want to use this as normal dataloader
        **kwargs: All other arguments passed to parent PIMEDDataset_2D class
    
    Directory Structure Expected:
        features_dir/
        ├── teacher1/
        │   ├── features_10000_1000000.npz
        │   ├── features_100001_0001.npz
        │   ├── features_100002_10001.npz
        │   └── ...
        ├── teacher2/
        │   ├── features_10000_1000000.npz
        │   └── ...
        └── teacher3/
            └── ...
    
    Feature File Format (.npz):
        features_{case_id}.npz contains:
        {
            'feature_neg2': np.ndarray([1, C, D, H, W]) or ([C, D, H, W]),  # Second-to-last layer
            'feature_neg1': np.ndarray([1, C, D, H, W]) or ([C, D, H, W]),  # Last layer
            'logits': np.ndarray([1, num_classes, D, H, W]) or ([num_classes, D, H, W])  # Segmentation logits
        }
        Note: Batch dimension (if present) will be automatically removed
    
    Returns:
        If load_features=True:
            img: [3, D, H, W] - Stacked T2/ADC/DWI volumes
            label: [D, H, W] - Segmentation label
            teacher_features: List of dicts, one per teacher
            teacher_logits: List of tensors, one per teacher
            case_id: str - Case identifier
            
        If load_features=False:
            img: [3, D, H, W] - Stacked T2/ADC/DWI volumes
            label: [D, H, W] - Segmentation label
    """
    
    def __init__(self, 
                 path_dict,
                 fold_cases,
                 case_id2cohort,
                 features_path_dict=None,
                 teacher_names=None,
                 load_features=True,
                 features_format_dict=None,  # 'npz' or dict mapping teacher->format
                 cache_features=False,  # Cache features in memory (use only if enough RAM)
                 cache_size=300,  # Max number of cases to cache per GPU (for LRU cache)
                 load_neg2=False,  # Whether to load layer_minus_2 features (default: False to save memory)
                 **kwargs):
        
        # Extract and store config if provided (not passed to parent)
        self.config = kwargs.pop('config', None)
        
        # Store load_neg2 flag
        self.load_neg2 = load_neg2
        
        # Initialize parent class
        super().__init__(path_dict=path_dict, fold_cases=fold_cases, case_id2cohort=case_id2cohort, **kwargs)
        
        # Normalize features_path_dict: accept either a root dir (str) or a dict mapping teacher->dir
        if isinstance(features_path_dict, str):
            # Build mapping teacher_name -> os.path.join(root, teacher_name)
            if teacher_names is None:
                raise ValueError("teacher_names must be provided when features_path_dict is a root directory string")
            self.features_path_dict = {t: os.path.join(features_path_dict, t) for t in teacher_names}
        else:
            self.features_path_dict = features_path_dict or {}

        self.teacher_names = teacher_names if teacher_names is not None else []
        self.load_features = load_features

        # Normalize features_format_dict: accept a string (apply to all) or a dict per teacher
        if features_format_dict is None:
            # default to 'npz' for all teachers
            self.features_format_dict = {t: 'npz' for t in self.teacher_names}
        elif isinstance(features_format_dict, str):
            self.features_format_dict = {t: features_format_dict for t in self.teacher_names}
        else:
            # assume dict-like
            self.features_format_dict = features_format_dict
        self.cache_features = cache_features
        self.cache_size = cache_size
        
        # Validate inputs if loading features
        if self.load_features:
            self._validate_feature_setup()
            
        # Cache for features - use LRU cache if size limit specified, otherwise unlimited dict
        if cache_features:
            if cache_size > 0:
                # LRU cache with size limit (recommended for DDP with shuffling)
                self.feature_cache = LRUCache(max_size=cache_size)
            else:
                # Unlimited cache (will grow unbounded - only use if you have enough RAM!)
                self.feature_cache = {}
        else:
            self.feature_cache = None
        
        self._cache_hit_count = 0  # Track cache hits for debugging
        self._cache_miss_count = 0  # Track cache misses for debugging
        self._failed_cases = set()  # Track cases that failed to load
        self._max_retries = 10  # Maximum retries per __getitem__ call
        
        # Set up caching info for DDP
        if self.cache_features and self.load_features:
            import torch.distributed as dist
            if dist.is_available() and dist.is_initialized():
                rank = dist.get_rank()
                world_size = dist.get_world_size()
                
                if self.cache_size > 0:
                    # LRU cache - bounded memory
                    memory_per_gpu = self.cache_size * 0.66  # GB
                    total_memory = world_size * memory_per_gpu
                    print(f"🔄 [GPU {rank}] LRU cache enabled (max {self.cache_size} cases)")
                    if rank == 0:
                        print(f"   ✅ Memory bounded: {world_size} GPUs × {memory_per_gpu:.1f}GB = ~{total_memory:.1f}GB total")
                        print(f"   Expected hit rate: ~18% after warmup (trade-off for bounded memory)")
                else:
                    # Unlimited cache - will cause OOM!
                    print(f"⚠️  [GPU {rank}] Unlimited cache enabled - THIS WILL CAUSE OOM!")
                    if rank == 0:
                        print(f"   ❌ WARNING: With shuffling, each GPU will eventually cache ALL cases")
                        print(f"   Total memory needed: {world_size} GPUs × ~1124GB = ~{world_size * 1.124:.0f}TB")
                        print(f"   Consider:")
                        print(f"   1. Set cache_size=300 for LRU cache (bounded memory)")
                        print(f"   2. Disable caching (remove --cache-features)")
                        print(f"   3. Disable shuffling (lose training quality)")
            else:
                # Single GPU mode
                if self.cache_size > 0:
                    memory_estimate = self.cache_size * 0.66
                    print(f"🔄 LRU cache enabled (max {self.cache_size} cases, ~{memory_estimate:.1f}GB)")
                else:
                    print(f"🔄 Unlimited cache enabled - will cache all {len(self.case_ids)} cases")
    
    def _validate_feature_setup(self):
        """Validate that feature directories and teacher names are properly set"""
        if len(self.teacher_names) == 0:
            raise ValueError(
                "teacher_names list cannot be empty when load_features=True. "
                "Provide a list like ['teacher1', 'teacher2', 'teacher3']"
            )

        # Ensure we have a path for each teacher
        missing = [t for t in self.teacher_names if t not in self.features_path_dict]
        if len(missing) > 0:
            raise ValueError(f"features_path_dict missing entries for teachers: {missing}")

        # Verify directories exist for each teacher
        for teacher_name in self.teacher_names:
            teacher_dir_or_list = self.features_path_dict[teacher_name]
            
            # Handle both single path and list of paths
            if isinstance(teacher_dir_or_list, list):
                teacher_dirs = teacher_dir_or_list
                # Check that at least one directory exists
                if not any(os.path.exists(d) for d in teacher_dirs):
                    raise FileNotFoundError(
                        f"No valid directories found for teacher '{teacher_name}'\n"
                        f"Checked: {teacher_dirs}"
                    )
            else:
                if not os.path.exists(teacher_dir_or_list):
                    raise FileNotFoundError(
                        f"Teacher directory not found: {teacher_dir_or_list}\n"
                        f"Expected: features_path_dict['{teacher_name}'] to point to a folder containing feature files"
                )

        # Verify formats mapping contains all teachers
        for teacher_name in self.teacher_names:
            if teacher_name not in self.features_format_dict:
                # default to 'npz' if missing
                self.features_format_dict[teacher_name] = 'npz'
    
    def _load_teacher_features(self, case_id):
        """
        Load pre-extracted features and logits for all teachers for a given case
        
        Args:
            case_id: Case identifier (e.g., 'case_001' or '10000_1000000')
            
        Returns:
            teacher_features: List of dicts, one per teacher
            teacher_logits: List of tensors, one per teacher
        """
        teacher_features_list = []
        teacher_logits_list = []
        
        for teacher_name in self.teacher_names:
            teacher_dir_or_list = self.features_path_dict[teacher_name]
            
            # Handle both single path (str) and multiple paths (list)
            if isinstance(teacher_dir_or_list, list):
                teacher_dirs = teacher_dir_or_list
            else:
                teacher_dirs = [teacher_dir_or_list]
            
            # Build filename based on teacher-specific naming convention
            if 'provicnet' in teacher_name:
                feature_filename = f'features_{case_id}.npz'
            elif 'prostatlasdiff' in teacher_name:
                feature_filename = f'diffusion_features_{case_id}.npz'
            else:  # nnunet or any other teacher
                feature_filename = f'features_{case_id}.npz'
                
            fmt = self.features_format_dict.get(teacher_name, 'npz')
            
            # Try to find the file (support both .npz and .pt)
            file_path = None
            actual_fmt = fmt
            
            for teacher_dir in teacher_dirs:
                # Try the specified format first
                if fmt == 'npz':
                    candidate_path = os.path.join(teacher_dir, feature_filename)
                    if os.path.exists(candidate_path):
                        file_path = candidate_path
                        actual_fmt = 'npz'
                        break
                elif fmt == 'pt':
                    # For .pt format, use same filename but with .pt extension
                    pt_filename = feature_filename.replace('.npz', '.pt')
                    candidate_path = os.path.join(teacher_dir, pt_filename)
                    if os.path.exists(candidate_path):
                        file_path = candidate_path
                        actual_fmt = 'pt'
                        break
                
                # Fallback: try both formats if specified format not found
                for ext, try_fmt in [('.npz', 'npz'), ('.pt', 'pt')]:
                    test_filename = feature_filename.replace('.npz', ext)
                    candidate_path = os.path.join(teacher_dir, test_filename)
                    if os.path.exists(candidate_path):
                        file_path = candidate_path
                        actual_fmt = try_fmt
                        break
                
                if file_path:
                    break
            
            if file_path is None:
                raise FileNotFoundError(
                    f"Feature file not found in any of the directories: {teacher_dirs}\n"
                    f"Looking for: {feature_filename} (or .pt variant)\n"
                    f"Case: {case_id}, Teacher: {teacher_name}"
                )
            
            # Load the file based on its actual format
            if actual_fmt == 'npz':
                # Load .npz file
                npz_data = np.load(file_path)
                
                # Required keys: 'feature_neg1', 'logits' (feature_neg2 is optional)
                if 'feature_neg1' not in npz_data or 'logits' not in npz_data:
                    raise KeyError(
                        f"NPZ file missing required keys. Found: {list(npz_data.keys())}\n"
                        f"Expected: ['feature_neg1', 'logits'] (feature_neg2 is optional)\n"
                        f"File: {file_path}"
                    )
                
                # Convert to torch tensors
                # Only load neg2 if requested AND available
                if self.load_neg2 and 'feature_neg2' in npz_data:
                    feat_neg2 = torch.from_numpy(npz_data['feature_neg2'])
                else:
                    feat_neg2 = None
                feat_neg1 = torch.from_numpy(npz_data['feature_neg1'])
                logits_data = torch.from_numpy(npz_data['logits'])
            
            elif actual_fmt == 'pt':
                # Load PyTorch .pt file with same structure as .npz
                try:
                    pt_data = torch.load(file_path, map_location='cpu', weights_only=False)
                except Exception as e:
                    raise RuntimeError(
                        f"Failed to load PT file (possibly corrupted): {file_path}\n"
                        f"Error: {e}\n"
                        f"Case: {case_id}, Teacher: {teacher_name}"
                    )
                
                # Handle provicnet's nested structure: features are under 'features' key
                # provicnet format: {'features': {'feature_neg1': ..., 'logits': ...}, ...}
                if 'features' in pt_data and isinstance(pt_data['features'], dict):
                    # Provicnet nested format - extract the inner features dict
                    pt_data = pt_data['features']
                
                # Required keys: 'feature_neg1', 'logits' (feature_neg2 is optional)
                if 'feature_neg1' not in pt_data or 'logits' not in pt_data:
                    raise KeyError(
                        f"PT file missing required keys. Found: {list(pt_data.keys())}\n"
                        f"Expected: ['feature_neg1', 'logits'] (feature_neg2 is optional)\n"
                        f"File: {file_path}"
                    )
                
                # Data is already in torch tensors
                # Only load neg2 if requested AND available
                if self.load_neg2 and 'feature_neg2' in pt_data:
                    feat_neg2 = pt_data['feature_neg2']
                else:
                    feat_neg2 = None
                feat_neg1 = pt_data['feature_neg1']
                logits_data = pt_data['logits']
            
            else:
                raise ValueError(f"Unsupported format: {actual_fmt}")

            # Handle teacher-specific formats and class remapping (common for both .npz and .pt)
            # Note: provicnet has 4 channels [background, prostate, ciPCA, csPCA] with softmax output
            # We keep it as-is here; the training code handles conversion via normalize_provicnet_logits()
            # Only warn if unexpected channel count for non-provicnet teachers
            if 'provicnet' not in teacher_name:
                if logits_data.shape[0] == 4 or (logits_data.ndim > 1 and logits_data.shape[1] == 4):
                    import warnings
                    warnings.warn(
                        f"WARNING: Teacher '{teacher_name}' has 4-channel logits for case {case_id}! "
                        f"Expected 3 channels for non-provicnet teachers. Shape: {logits_data.shape}.",
                        UserWarning
                    )
                
            if 'prostatlasdiff' in teacher_name:
                # ProstAtlasDiff format: [D, C, modalities, H, W]
                # Trained in 2.5D fashion: 3 slices in, center slice out
                # For features: treat 3 slices as additional channels: [C*3, D, H, W]
                if feat_neg2 is not None:
                    D, C, modalities, H, W = feat_neg2.shape
                    # Reshape features: [D, C, 3, H, W] -> [D, C*3, H, W] -> [C*3, D, H, W]
                    feat_neg2 = feat_neg2.reshape(D, C * modalities, H, W).permute(1, 0, 2, 3)
                
                D_neg1, C_neg1, modalities_neg1, H_neg1, W_neg1 = feat_neg1.shape
                feat_neg1 = feat_neg1.reshape(D_neg1, C_neg1 * modalities_neg1, H_neg1, W_neg1).permute(1, 0, 2, 3)
                
                # Logits: [D, num_classes, modalities, H, W]
                # Keep all 3 slices as channels: [modalities, D, H, W] = [3, D, H, W]
                D, num_classes, modalities, H, W = logits_data.shape
                # Reshape: [D, num_classes, 3, H, W] -> [D, 3, H, W] -> [3, D, H, W]
                # Assuming num_classes=1, we take that dimension
                logits_data = logits_data[:, 0, :, :, :]  # [D, 3, H, W] - all 3 slices
                logits_data = logits_data.permute(1, 0, 2, 3)  # [3, D, H, W]
            elif 'provicnet' in teacher_name:
                # ProViCNet format: [D, C, H, W] - needs permutation to [C, D, H, W]
                # Features: [D, C, H, W] -> [C, D, H, W]
                # Logits: [D, 4, H, W] -> [4, D, H, W] (4 classes: bg, prostate, ciPCA, csPCA)
                if feat_neg2 is not None:
                    feat_neg2 = feat_neg2.permute(1, 0, 2, 3)  # [D, C, H, W] -> [C, D, H, W]
                feat_neg1 = feat_neg1.permute(1, 0, 2, 3)  # [D, C, H, W] -> [C, D, H, W]
                logits_data = logits_data.permute(1, 0, 2, 3)  # [D, 4, H, W] -> [4, D, H, W]
            else:
                # Standard format: handle batch dimension if present
                if feat_neg2 is not None and feat_neg2.shape[0] == 1:
                    feat_neg2 = feat_neg2.squeeze(0)  # [C, D, H, W]
                if feat_neg1.shape[0] == 1:
                    feat_neg1 = feat_neg1.squeeze(0)  # [C, D, H, W]
                if logits_data.shape[0] == 1:
                    logits_data = logits_data.squeeze(0)  # [num_classes, D, H, W]
            
            # Create features dict and logits
            features_dict = {
                'layer_minus_1': feat_neg1   # [C, D, H, W]
            }
            # Only add layer_minus_2 if it was loaded
            if feat_neg2 is not None:
                features_dict['layer_minus_2'] = feat_neg2  # [C, D, H, W]
            logits = logits_data  # [num_classes, D, H, W]
            
            # Convert logits to 2-class for binary mode
            # Note: provicnet has 4 channels [bg, prostate, ciPCA, csPCA] - keep as-is for training code
            # Other teachers (nnunet) have 3 channels [prostate, PCa, csPCa]
            if self.pred_type == 'binary':
                if 'provicnet' in teacher_name:
                    # Keep provicnet's 4 channels - training code handles via normalize_provicnet_logits
                    pass
                elif 'prostatlasdiff' in teacher_name:
                    # ProstAtlasDiff logits are [modalities=3, D, H, W] where the leading 3 is 2.5D slices,
                    # NOT [prostate, PCa, csPCa] classes. Do NOT combine as if they were classes.
                    pass
                elif logits.shape[0] == 3:
                    # 3-class teachers (nnunet): combine PCa and csPCa
                    # logits: [3, D, H, W] -> [2, D, H, W]
                    prostate_logit = logits[0:1]  # [1, D, H, W]
                    cancer_logit = torch.max(logits[1:2], logits[2:3])  # max(PCa, csPCa) -> [1, D, H, W]
                    logits = torch.cat([prostate_logit, cancer_logit], dim=0)  # [2, D, H, W]
            
            teacher_features_list.append(features_dict)
            teacher_logits_list.append(logits)
        
        return teacher_features_list, teacher_logits_list
    
    def _pad_or_crop_features(self, teacher_features, teacher_logits, mask_start_idx, mask_end_idx, target_depth=20):
        """
        Process teacher features to match target depth (20 slices)
        
        Strategy:
        1. If prostate_depth >= 20: Center crop to 20 slices
        2. If prostate_depth < 20: Crop to prostate region [mask_start_idx:mask_end_idx+1], then pad to 20
        
        Args:
            teacher_features: List of dicts with 'layer_minus_1' (and optionally 'layer_minus_2') [C, D, H, W]
            teacher_logits: List of tensors [num_classes, D, H, W]
            mask_start_idx: Starting index of prostate mask bounding box
            mask_end_idx: Ending index of prostate mask bounding box (inclusive)
            target_depth: Target depth dimension (default: 20)
            
        Returns:
            Processed teacher_features and teacher_logits - all with depth=20
        """
        processed_features = []
        processed_logits = []
        
        prostate_depth = mask_end_idx - mask_start_idx + 1
        
        for teacher_idx, (feat_dict, logits) in enumerate(zip(teacher_features, teacher_logits)):
            # Features shape: [C, D, H, W]
            feat_minus_2 = feat_dict.get('layer_minus_2', None)  # May be None if not loaded
            feat_minus_1 = feat_dict['layer_minus_1']
            
            current_depth = feat_minus_1.shape[1]  # Use feat_minus_1 for depth since it's always present
            
            # ALWAYS crop or pad to exactly target_depth, regardless of current depth
            if current_depth == target_depth:
                # Already the right size
                processed_feat_minus_2 = feat_minus_2
                processed_feat_minus_1 = feat_minus_1
                processed_logit = logits
            elif current_depth > target_depth:
                # Crop: center crop to target_depth
                crop_start = (current_depth - target_depth) // 2
                crop_end = crop_start + target_depth
                
                processed_feat_minus_2 = feat_minus_2[:, crop_start:crop_end, :, :] if feat_minus_2 is not None else None
                processed_feat_minus_1 = feat_minus_1[:, crop_start:crop_end, :, :]
                processed_logit = logits[:, crop_start:crop_end, :, :]
            else:
                # Pad: pad to target_depth
                padding_need = target_depth - current_depth
                pad_before = padding_need // 2
                pad_after = padding_need - pad_before
                
                # Pad on depth dimension (dimension 1): [C, D, H, W]
                # torch.nn.functional.pad format: (W_left, W_right, H_top, H_bottom, D_front, D_back)
                if feat_minus_2 is not None:
                    processed_feat_minus_2 = torch.nn.functional.pad(
                        feat_minus_2, 
                        (0, 0, 0, 0, pad_before, pad_after),
                        mode='constant', value=0
                    )
                else:
                    processed_feat_minus_2 = None
                processed_feat_minus_1 = torch.nn.functional.pad(
                    feat_minus_1,
                    (0, 0, 0, 0, pad_before, pad_after),
                    mode='constant', value=0
                )
                processed_logit = torch.nn.functional.pad(
                    logits,
                    (0, 0, 0, 0, pad_before, pad_after),
                    mode='constant', value=0
                )
            
            # Store processed features - all should be [C, 20, 256, 256] now
            processed_feat_dict = {'layer_minus_1': processed_feat_minus_1}
            if processed_feat_minus_2 is not None:
                processed_feat_dict['layer_minus_2'] = processed_feat_minus_2
            processed_features.append(processed_feat_dict)
            processed_logits.append(processed_logit)
        
        return processed_features, processed_logits
    
    def __getitem__(self, idx):
        """
        Get a single data sample with optional pre-extracted teacher features
        
        Returns:
            If load_features=True:
                img, label, teacher_features, teacher_logits, case_id
            If load_features=False:
                img, label, case_id
        """
        # Get case_id 
        case_id = self.case_ids[idx]
        
        # If not loading features, use parent class behavior (validation mode)
        if not self.load_features:
            # Retry logic for validation: skip cases with missing files
            retry_count = 0
            current_idx = idx
            while retry_count < self._max_retries:
                case_id = self.case_ids[current_idx]
                if case_id in self._failed_cases:
                    retry_count += 1
                    current_idx = (current_idx + 1) % len(self)
                    continue
                try:
                    img, label, prostate_mask, mask_start_idx, mask_end_idx = super().__getitem__(current_idx)
                    
                    # Convert labels to region-based multi-hot encoding if needed
                    label_original = label  # Keep original for visualization
                    if self.region_based:
                        import torch
                        import numpy as np
                        
                        # Ensure label is torch tensor
                        if not torch.is_tensor(label):
                            label = torch.from_numpy(label)
                        
                        # Initialize transform if not already done
                        if not hasattr(self, '_region_converter'):
                            from dataset.region_converter import get_regions_for_pred_type
                            regions = get_regions_for_pred_type(self.pred_type)
                            self._region_converter = ConvertSegmentationToRegions(regions)
                        
                        # Convert: [D, H, W] → [C, D, H, W] boolean
                        label = self._region_converter(label)  # Returns boolean tensor [C, D, H, W]
                        label = label.float()  # Convert to float for BCE loss
                
                    return img, label, case_id, label_original, prostate_mask
                except Exception as e:
                    self._failed_cases.add(case_id)
                    if len(self._failed_cases) <= 20:
                        print(f"❌ Error loading val case {case_id} (index {current_idx}): {str(e)[:100]}")
                    retry_count += 1
                    current_idx = (current_idx + 1) % len(self)
            
            raise RuntimeError(
                f"Failed to load valid val data after {self._max_retries} attempts from index {idx}. "
                f"Failed cases: {sorted(list(self._failed_cases))[:10]}..."
            )
        
        # Track retry attempts to avoid infinite loops
        retry_count = 0
        attempted_indices = set()
        current_idx = idx
        
        while retry_count < self._max_retries:
            case_id = self.case_ids[current_idx]
            
            # Skip cases we've already tried and failed
            if current_idx in attempted_indices:
                retry_count += 1
                current_idx = (current_idx + 1) % len(self)
                continue
            
            attempted_indices.add(current_idx)
            
            # Skip cases that have previously failed
            if case_id in self._failed_cases:
                retry_count += 1
                current_idx = (current_idx + 1) % len(self)
                continue
            
            try:
                # Call parent's __getitem__ to get the processed img, label, prostate_mask, and mask indices
                # Parent returns: img, label, prostate_mask, mask_start_idx, mask_end_idx
                img, label, prostate_mask, mask_start_idx, mask_end_idx = super().__getitem__(current_idx)
                
                # Load teacher features and logits
                # Check cache first
                if self.cache_features:
                    if isinstance(self.feature_cache, LRUCache):
                        # LRU cache
                        cached_data = self.feature_cache.get(case_id)
                        if cached_data is not None:
                            self._cache_hit_count += 1
                            teacher_features, teacher_logits = cached_data
                        else:
                            # Cache miss - load from disk
                            self._cache_miss_count += 1
                            teacher_features, teacher_logits = self._load_teacher_features(case_id)
                            self.feature_cache.put(case_id, (teacher_features, teacher_logits))
                            
                            # Print progress every 50 misses
                            if self._cache_miss_count % 50 == 0:
                                import torch.distributed as dist
                                if dist.is_available() and dist.is_initialized():
                                    rank = dist.get_rank()
                                    hit_rate = 100 * self._cache_hit_count / (self._cache_hit_count + self._cache_miss_count)
                                    print(f"  [GPU {rank}] LRU cache: {len(self.feature_cache)} cases, "
                                          f"hits: {self._cache_hit_count}, misses: {self._cache_miss_count}, "
                                          f"hit rate: {hit_rate:.1f}%, evictions: {self.feature_cache.eviction_count}")
                    else:
                        # Unlimited dict cache
                        if case_id in self.feature_cache:
                            self._cache_hit_count += 1
                            teacher_features, teacher_logits = self.feature_cache[case_id]
                        else:
                            self._cache_miss_count += 1
                            teacher_features, teacher_logits = self._load_teacher_features(case_id)
                            self.feature_cache[case_id] = (teacher_features, teacher_logits)
                            
                            if self._cache_miss_count % 50 == 0:
                                import torch.distributed as dist
                                if dist.is_available() and dist.is_initialized():
                                    rank = dist.get_rank()
                                    print(f"  [GPU {rank}] Cached {len(self.feature_cache)} cases "
                                          f"(hits: {self._cache_hit_count}, misses: {self._cache_miss_count})")
                else:
                    # No caching - always load from disk
                    teacher_features, teacher_logits = self._load_teacher_features(case_id)
                
                # Crop to prostate region, then pad/crop to match the image depth (20 slices)
                teacher_features, teacher_logits = self._pad_or_crop_features(
                    teacher_features, teacher_logits, mask_start_idx, mask_end_idx, target_depth=20
                )
                
                # Verify depth dimension is correct (must be 20 to match image preprocessing)
                # Note: H/W can differ per teacher (e.g., provicnet=128x128, nnunet=256x256)
                # The feature transformation will project student features to match each teacher's dimensions
                for i, (feat_dict, logits) in enumerate(zip(teacher_features, teacher_logits)):
                    # Use layer_minus_1 for verification (always present, layer_minus_2 may be absent)
                    D_feat = feat_dict['layer_minus_1'].shape[1]
                    D_logit = logits.shape[1]
                    
                    if D_feat != 20:
                        raise ValueError(f"Teacher {i} feature depth mismatch: expected D=20, got D={D_feat} for case {case_id}")
                    if D_logit != 20:
                        raise ValueError(f"Teacher {i} logit depth mismatch: expected D=20, got D={D_logit} for case {case_id}")
                
                # Convert labels to region-based multi-hot encoding if needed
                label_original = label  # Keep original for visualization
                if self.region_based:
                    # Convert class labels {0,1,2,3} to multi-hot regions [C, D, H, W]
                    # Regions depend on pred_type:
                    #   '3class': [1,2,3] (prostate), [2,3] (PCa), [3] (csPCa)
                    #   'binary': [1,2,3] (prostate), [2,3] (cancer combined)
                    import torch
                    import numpy as np
                    
                    # Ensure label is torch tensor
                    if not torch.is_tensor(label):
                        label = torch.from_numpy(label)
                    
                    # Initialize transform if not already done
                    if not hasattr(self, '_region_converter'):
                        from dataset.region_converter import get_regions_for_pred_type
                        regions = get_regions_for_pred_type(self.pred_type)
                        self._region_converter = ConvertSegmentationToRegions(regions)
                    
                    # Convert: [D, H, W] → [C, D, H, W] boolean
                    label = self._region_converter(label)  # Returns boolean tensor [C, D, H, W]
                    label = label.float()  # Convert to float for BCE loss
                
                # Success! Return the data (including original label and prostate_mask for visualization/masking)
                return img, label, teacher_features, teacher_logits, case_id, label_original, prostate_mask
                
            except Exception as e:
                # Mark this case as failed
                self._failed_cases.add(case_id)
                
                # Print error (but limit spam)
                if len(self._failed_cases) <= 20:  # Only print first 20 errors
                    print(f"❌ Error loading features for {case_id} (index {current_idx}): {str(e)[:100]}")
                    if len(self._failed_cases) == 20:
                        print(f"   (Suppressing further error messages - {len(self._failed_cases)} cases failed)")
                
                # Try next index
                retry_count += 1
                current_idx = (current_idx + 1) % len(self)
        
        # If we've exhausted retries, raise an error
        raise RuntimeError(
            f"Failed to load valid data after {self._max_retries} attempts. "
            f"Started at index {idx} (case {self.case_ids[idx]}). "
            f"Total failed cases so far: {len(self._failed_cases)}. "
            f"Failed cases: {sorted(list(self._failed_cases))[:10]}..."
        )
    
    def get_feature_info(self, teacher_idx=0, case_idx=0):
        """
        Get information about feature dimensions for a specific teacher and case
        Useful for debugging and verifying feature extraction
        
        Args:
            teacher_idx: Index of teacher to check (default: 0)
            case_idx: Index of case to check (default: 0)
            
        Returns:
            dict with feature dimension information
        """
        if not self.load_features:
            return {"error": "load_features is False"}
        
        case_id = self.case_ids[case_idx]
        teacher_features, teacher_logits = self._load_teacher_features(case_id)
        
        if teacher_idx >= len(teacher_features):
            return {"error": f"teacher_idx {teacher_idx} out of range"}
        
        feat_dict = teacher_features[teacher_idx]
        logits = teacher_logits[teacher_idx]
        
        info = {
            'case_id': case_id,
            'teacher_name': self.teacher_names[teacher_idx],
            'features_layer_minus_1_shape': feat_dict['layer_minus_1'].shape,
            'logits_shape': logits.shape,
            'features_layer_minus_1_dtype': feat_dict['layer_minus_1'].dtype,
            'logits_dtype': logits.dtype,
        }
        
        # Only add layer_minus_2 info if it exists
        if 'layer_minus_2' in feat_dict:
            info['features_layer_minus_2_shape'] = feat_dict['layer_minus_2'].shape
            info['features_layer_minus_2_dtype'] = feat_dict['layer_minus_2'].dtype
        
        return info


def collate_fn_with_features(batch):
    """
    Custom collate function for batches with teacher features.
    Defined at module level to be picklable for Windows multiprocessing.
    """
    # Unpack: img, label, teacher_features, teacher_logits, case_id, label_original, prostate_mask
    if len(batch[0]) == 7:
        imgs, labels, teacher_feats, teacher_logs, case_ids, labels_original, prostate_masks = zip(*batch)
    elif len(batch[0]) == 6:
        # Backward compatibility without prostate_mask
        imgs, labels, teacher_feats, teacher_logs, case_ids, labels_original = zip(*batch)
        prostate_masks = None
    else:
        # Backward compatibility if dataloader doesn't return original labels
        imgs, labels, teacher_feats, teacher_logs, case_ids = zip(*batch)
        labels_original = labels
        prostate_masks = None
    
    # Stack images and labels
    imgs = torch.stack([torch.from_numpy(img) if isinstance(img, np.ndarray) else img 
                       for img in imgs])
    labels = torch.stack([torch.from_numpy(lbl) if isinstance(lbl, np.ndarray) else lbl 
                         for lbl in labels])
    labels_original = torch.stack([torch.from_numpy(lbl) if isinstance(lbl, np.ndarray) else lbl 
                                   for lbl in labels_original])
    
    # Stack prostate masks if available
    if prostate_masks is not None:
        prostate_masks = torch.stack([torch.from_numpy(m) if isinstance(m, np.ndarray) else m 
                                      for m in prostate_masks])
    
    # Organize teacher features and logits
    # teacher_feats is list of lists: batch_size x num_teachers
    num_teachers = len(teacher_feats[0])
    batch_teacher_features = []
    batch_teacher_logits = []
    
    for t_idx in range(num_teachers):
        # Collect features for this teacher across the batch
        t_features_batch = [teacher_feats[b_idx][t_idx] for b_idx in range(len(batch))]
        t_logits_batch = [teacher_logs[b_idx][t_idx] for b_idx in range(len(batch))]
        
        # Teacher features should already have consistent dimensions
        # If they don't, it indicates a problem with feature extraction
        
        # Validate and stack features - they should already have consistent dimensions
        # Check if layer_minus_2 exists (may be None if not loaded)
        has_neg2 = 'layer_minus_2' in t_features_batch[0]
        feat_neg2_list = [f['layer_minus_2'] for f in t_features_batch] if has_neg2 else None
        feat_neg1_list = [f['layer_minus_1'] for f in t_features_batch]
        logits_list = t_logits_batch
        
        # Check dimensions are consistent (they should be if feature extraction was done correctly)
        if has_neg2 and len(set(f.shape for f in feat_neg2_list)) > 1:
            shapes = [f.shape for f in feat_neg2_list]
            print(f"❌ ERROR: Teacher {t_idx} layer_minus_2 features have inconsistent shapes: {shapes}")
            print(f"   This indicates feature extraction was done on different input sizes!")
            print(f"   All teachers should extract features from the same preprocessed volumes.")
            # For now, use the first item only
            feat_neg2_list = [feat_neg2_list[0]]
            feat_neg1_list = [feat_neg1_list[0]]
            logits_list = [logits_list[0]]
        
        if len(set(f.shape for f in feat_neg1_list)) > 1:
            shapes = [f.shape for f in feat_neg1_list]
            print(f"❌ ERROR: Teacher {t_idx} layer_minus_1 features have inconsistent shapes: {shapes}")
        
        if len(set(f.shape for f in logits_list)) > 1:
            shapes = [f.shape for f in logits_list]
            print(f"❌ ERROR: Teacher {t_idx} logits have inconsistent shapes: {shapes}")
        
        # Stack features (should work if dimensions are consistent)
        try:
            stacked_features = {
                'layer_minus_1': torch.stack(feat_neg1_list)
            }
            # Only stack layer_minus_2 if it exists
            if has_neg2:
                stacked_features['layer_minus_2'] = torch.stack(feat_neg2_list)
            stacked_logits = torch.stack(logits_list)
        except RuntimeError as e:
            print(f"❌ ERROR: Cannot stack teacher {t_idx} features: {e}")
            if has_neg2:
                print(f"   Shapes: feat_neg2={[f.shape for f in feat_neg2_list]}")
            print(f"           feat_neg1={[f.shape for f in feat_neg1_list]}")
            print(f"           logits={[f.shape for f in logits_list]}")
            # Fallback: use only first item
            stacked_features = {
                'layer_minus_1': feat_neg1_list[0].unsqueeze(0)
            }
            if has_neg2:
                stacked_features['layer_minus_2'] = feat_neg2_list[0].unsqueeze(0)
            stacked_logits = logits_list[0].unsqueeze(0)
        
        batch_teacher_features.append(stacked_features)
        batch_teacher_logits.append(stacked_logits)
    
    return imgs, labels, batch_teacher_features, batch_teacher_logits, case_ids, labels_original, prostate_masks


def collate_fn_without_features(batch):
    """
    Standard collate function for batches without teacher features.
    Defined at module level to be picklable for Windows multiprocessing.
    Handles batches with format: (img, label, case_id, label_original, prostate_mask)
    """
    # Unpack batch - now includes case_id, label_original, and prostate_mask
    if len(batch[0]) == 5:
        imgs, labels, case_ids, labels_original, prostate_masks = zip(*batch)
    elif len(batch[0]) == 4:
        # Backward compatibility without prostate_mask
        imgs, labels, case_ids, labels_original = zip(*batch)
        prostate_masks = None
    else:
        # Backward compatibility
        imgs, labels, case_ids = zip(*batch)
        labels_original = labels
        prostate_masks = None
    
    # Convert to tensors first
    imgs = [torch.from_numpy(img) if isinstance(img, np.ndarray) else img for img in imgs]
    labels = [torch.from_numpy(lbl) if isinstance(lbl, np.ndarray) else lbl for lbl in labels]
    labels_original = [torch.from_numpy(lbl) if isinstance(lbl, np.ndarray) else lbl for lbl in labels_original]
    
    # Handle variable volume sizes by padding to max size in batch
    def pad_to_max_size_basic(tensors):
        """Pad 3D volumes to same size within batch"""
        if len(tensors) == 1:
            return torch.stack(tensors)
        
        # Find max dimensions
        max_d = max(t.shape[-3] for t in tensors)  # depth
        max_h = max(t.shape[-2] for t in tensors)  # height  
        max_w = max(t.shape[-1] for t in tensors)  # width
        
        padded_tensors = []
        for tensor in tensors:
            # Calculate padding needed for each dimension
            pad_d = max_d - tensor.shape[-3]
            pad_h = max_h - tensor.shape[-2]
            pad_w = max_w - tensor.shape[-1]
            # PyTorch pad format: (pad_left, pad_right, pad_top, pad_bottom, pad_front, pad_back)
            padding = (0, pad_w, 0, pad_h, 0, pad_d)
            
            padded = torch.nn.functional.pad(tensor, padding, mode='constant', value=0)
            padded_tensors.append(padded)
        
        return torch.stack(padded_tensors)
    
    imgs = pad_to_max_size_basic(imgs)
    labels = pad_to_max_size_basic(labels)
    labels_original = pad_to_max_size_basic(labels_original)
    
    # Pad prostate masks if available
    if prostate_masks is not None:
        prostate_masks = [torch.from_numpy(m) if isinstance(m, np.ndarray) else m for m in prostate_masks]
        prostate_masks = pad_to_max_size_basic(prostate_masks)
    
    return imgs, labels, case_ids, labels_original, prostate_masks


def get_dataloader_with_features(config,
                                  batch_size=2,
                                  num_workers=4,
                                  load_features=True,
                                  cache_features=False,
                                  cache_size=300,
                                  load_neg2=False):
    """
    Convenience function to create train and val DataLoaders with feature loading.
    Mimics the CIFAR-100 dataloader API: just pass config and get both loaders back.
    
    Args:
        config: Configuration dict (loaded from YAML file)
        batch_size: Batch size for DataLoader (default: 2)
        num_workers: Number of worker processes (default: 4)
        load_features: Whether to load pre-extracted features (default: True)
        cache_features: Whether to cache all features in memory (default: False)
        load_neg2: Whether to load layer_minus_2 features (default: False to save memory)
        cache_features: Whether to cache all features in memory (default: False)
        
    Returns:
        train_loader, val_loader (tuple of DataLoader instances)
        
    Example:
        import yaml
        with open('./dataset/pimed_dataset_configs.yaml', 'r') as f:
            config = yaml.safe_load(f)
        
        train_loader, val_loader = get_dataloader_with_features(
            config=config,
            batch_size=2,
            num_workers=4
        )
    """
    import json
    from torch.utils.data import DataLoader
    
    # Build path_dict and case lists from config
    case_id2cohort = {}
    
    # Initialize all case ID lists to empty (will be populated based on cohort flags)
    stanford_train_ids = []
    stanford_val_ids = []
    picai_train_ids = []
    picai_val_ids = []
    ucla_train_ids = []
    ucla_val_ids = []
    
    # Check if using dynamic case loading
    if config.get('dynamic_case_loading', False):
        print("Using dynamic case loading...")
        from .dynamic_case_loader import load_dynamic_cases_for_config
        
        case_lists = load_dynamic_cases_for_config(config)
        stanford_train_ids = [c for c in case_lists['train'] if len(c) == 11]
        stanford_val_ids = [c for c in case_lists['val'] if len(c) == 11]
        picai_train_ids = [c for c in case_lists['train'] if len(c) == 13]
        picai_val_ids = [c for c in case_lists['val'] if len(c) == 13]
        ucla_train_ids = [c for c in case_lists['train'] if len(c) == 12]
        ucla_val_ids = [c for c in case_lists['val'] if len(c) == 12]
    else:
        # Load from fold JSON files - ensures same splits as teacher training
        # Load Stanford data (only if enabled)
        if config.get('train_stanford', True):
            with open(config['stanford_fold_json']) as json_file:
                stanford_fold_data = json.load(json_file)
                
                # Special case: fold_num=-1 means use ALL data for training
                if config['fold_num'] == -1:
                    # For 5-fold ensemble: use val from folds 1-4 (each case exactly once)
                    # Stanford fold JSON uses standard train/val keys where fold i's train
                    # contains ALL data except fold i's partition. Using train sets would
                    # cause ~4x duplication and leak fold 0's val into training.
                    stanford_train_ids = []
                    for fold_idx in range(1, 5):
                        stanford_train_ids.extend(stanford_fold_data[fold_idx]['val'])
                    
                    # Also add fold 0's train cases that are NOT in folds 1-4 val
                    # (fold 0's train = folds 1-4 data, which are exactly folds 1-4 val combined)
                    # So folds 1-4 val already covers all non-fold-0 data. We just need fold 0's
                    # train partition minus fold 0's val = the rest. But actually:
                    # fold 0 val = fold 0 partition (281 cases)
                    # folds 1-4 val = folds 1-4 partitions (1123 cases)
                    # Total = 1404 = all Stanford cases, with fold 0 val held out
                    # So we DON'T need fold 0's train at all - folds 1-4 val covers it.
                    
                    stanford_val_ids = stanford_fold_data[0]['val']
                    print(f"==> [fold_num=-1] Using Stanford data for training: {len(stanford_train_ids)} samples")
                    print(f"    - Using val from folds 1-4 (each case exactly once)")
                    print(f"    - Fold 0 val: RESERVED for validation (not in training)")
                    print(f"==> [fold_num=-1] Validation set: fold 0 val ({len(stanford_val_ids)} samples)")
                else:
                    stanford_train_ids = stanford_fold_data[config['fold_num']]['train']
                    stanford_val_ids = stanford_fold_data[config['fold_num']]['val']
        if config.get('train_picai', True):
            with open(config['picai_fold_json']) as json_file:
                picai_fold_data = json.load(json_file)
            # Special case: fold_num=-1 means use ALL data for training (for ensemble distillation)
            if config['fold_num'] == -1:
                # For 5-fold ensemble: use inner_val from all folds (each case appears exactly once)
                # This avoids the duplication issue where each case would appear ~4 times if using inner_train
                picai_train_ids = []
                # Use inner_val from folds 1-4 for training (each case appears in exactly one fold's val)
                for fold_idx in range(1, 5):
                    picai_train_ids.extend(picai_fold_data[fold_idx]['inner_val'])
                
                # Use fold 0's val set for true validation (not included in training)
                picai_val_ids = picai_fold_data[0]['inner_val']
                print(f"==> [fold_num=-1] Using PICAI data for training: {len(picai_train_ids)} samples")
                print(f"    - Using inner_val from folds 1-4 (each case exactly once)")
                print(f"    - Fold 0 val: RESERVED for validation (not in training)")
                print(f"==> [fold_num=-1] Validation set: fold 0 val ({len(picai_val_ids)} samples)")
            else:
                picai_train_ids = picai_fold_data[config['fold_num']]['inner_train']
                picai_val_ids = picai_fold_data[config['fold_num']]['inner_val']
            
        if config.get('train_ucla', True):
            with open(config['ucla_fold_json']) as json_file:
                ucla_fold_data = json.load(json_file)
            
            # Special case: fold_num=-1 means use ALL data for training
            if config['fold_num'] == -1:
                # For 5-fold ensemble: each case appears in exactly ONE fold's inner_val
                # Use inner_val from folds 1-4 for training, fold 0's inner_val for validation
                ucla_train_ids = []
                for fold_idx in range(1, 5):
                    ucla_train_ids.extend(ucla_fold_data[fold_idx]['inner_val'])
                
                ucla_val_ids = ucla_fold_data[0]['inner_val']
                print(f"==> [fold_num=-1] Using UCLA data for training: {len(ucla_train_ids)} samples")
                print(f"    - Using inner_val from folds 1-4 (each case exactly once)")
                print(f"==> [fold_num=-1] Validation set: fold 0 val ({len(ucla_val_ids)} samples)")
            else:
                ucla_train_ids = ucla_fold_data[config['fold_num']]['inner_train']
                ucla_val_ids = ucla_fold_data[config['fold_num']]['inner_val']


    if config.get('region_based', False):
        file_paths = config['image_dirs']['file_paths']
        path_dict = {'t2':file_paths['T2_dir'],
                     'adc':file_paths['ADC_dir'],
                     'dwi':file_paths['DWI_dir'],
                     'prostate':file_paths['mask_dir'],
                     'cancer':file_paths['lesions_dir']}
    else:
        # Only set up Stanford paths if Stanford training is enabled
        if config.get('train_stanford', True) and "stanford_paths" in config["image_dirs"]:
            stanford_paths = config["image_dirs"]["stanford_paths"]
            path_dict = {
                "stanford": {
                    "t2": stanford_paths["T2_dir"],
                    "adc": stanford_paths["ADC_dir"],
                    "dwi": stanford_paths["DWI_dir"],
                    "prostate": stanford_paths["mask_dir"],
                    "cancer": stanford_paths["lesions_dir"],
                }
            }
        else:
            path_dict = {}
    
        # Load PICAI data (only if enabled)
        if config.get('train_picai', True):
            # Set up paths
            file_paths = config["image_dirs"]["file_paths"]
            path_dict["picai"] = {
                "t2": file_paths["T2_dir"],
                "adc": file_paths["ADC_dir"],
                "dwi": file_paths["DWI_dir"],
                "prostate": file_paths["mask_dir"],
                "cancer": file_paths["lesions_dir"],
            }
            
    
    # Load UCLA data (only if enabled)
        if config.get('train_ucla', True):
            # Set up paths
            ucla_paths = config["image_dirs"]["ucla_paths"]
            path_dict["ucla"] = {
                "t2": ucla_paths["T2_dir"],
                "adc": ucla_paths["ADC_dir"],
                "dwi": ucla_paths["DWI_dir"],
                "prostate": ucla_paths["mask_dir"],
                "cancer": ucla_paths["lesions_dir"],
            }
            
    
    if config.get('train_stanford', True):
            for case in stanford_train_ids:
                case_id2cohort[str(case)] = "stanford"
            for case in stanford_val_ids:
                case_id2cohort[str(case)] = "stanford"
    if config.get('train_picai', True):
        for case in picai_train_ids:
            case_id2cohort[str(case)] = "picai"
        for case in picai_val_ids:
            case_id2cohort[str(case)] = "picai"
    if config.get('train_ucla', True):
        for case in ucla_train_ids:
            case_id2cohort[str(case)] = "ucla"
        for case in ucla_val_ids:
            case_id2cohort[str(case)] = "ucla"
    # Combine case lists (only include enabled cohorts)
    train_cases = stanford_train_ids + picai_train_ids + ucla_train_ids
    val_cases = stanford_val_ids + picai_val_ids + ucla_val_ids
    
    print(f'Dataset statistics:')
    print(f'  Stanford: {len(stanford_train_ids)} train, {len(stanford_val_ids)} val')
    print(f'  PICAI: {len(picai_train_ids)} train, {len(picai_val_ids)} val')
    print(f'  UCLA: {len(ucla_train_ids)} train, {len(ucla_val_ids)} val')
    print(f'  Total: {len(train_cases)} train, {len(val_cases)} val')
        
    # Get teacher info from config
    teacher_names = config['teacher_names']
    features_path_dict = config['image_dirs']['teachers_features_paths']

    # ------------------------------------------------------------------
    # Pre-flight: drop training cases whose teacher feature files are
    # missing or truncated. Real .pt feature files are tens of MB; anything
    # < 1 MB is a failed/truncated write. We scan once up-front on rank 0
    # to avoid mid-epoch crashes when the dataloader's retry budget is hit.
    # Only applies to train_cases when load_features=True. Val loader sets
    # load_features=False so doesn't need teacher features.
    # ------------------------------------------------------------------
    if load_features and len(train_cases) > 0:
        MIN_FEATURE_BYTES = 1_000_000  # 1 MB
        # Build per-teacher directory list (handles str or list)
        def _teacher_dirs(teacher_name):
            entry = features_path_dict.get(teacher_name)
            if entry is None:
                return []
            if isinstance(entry, list):
                return entry
            return [entry]

        def _feature_filename(teacher_name, case_id):
            if 'provicnet' in teacher_name:
                return f'features_{case_id}'
            elif 'prostatlasdiff' in teacher_name:
                return f'diffusion_features_{case_id}'
            else:
                return f'features_{case_id}'

        def _case_has_valid_features(case_id):
            for teacher_name in teacher_names:
                dirs = _teacher_dirs(teacher_name)
                stem = _feature_filename(teacher_name, case_id)
                found_ok = False
                for d in dirs:
                    for ext in ('.pt', '.npz'):
                        p = os.path.join(d, stem + ext)
                        if os.path.exists(p) and os.path.getsize(p) >= MIN_FEATURE_BYTES:
                            found_ok = True
                            break
                    if found_ok:
                        break
                if not found_ok:
                    return False
            return True

        original_n = len(train_cases)
        kept, dropped = [], []
        for c in train_cases:
            if _case_has_valid_features(str(c)):
                kept.append(c)
            else:
                dropped.append(c)
        if dropped:
            # Only rank 0 prints, but safe for single-process too
            try:
                import torch.distributed as _dist
                is_rank0 = (not _dist.is_available()) or (not _dist.is_initialized()) or _dist.get_rank() == 0
            except Exception:
                is_rank0 = True
            if is_rank0:
                print(f"==> [feature pre-flight] Dropping {len(dropped)} / {original_n} train cases with missing or truncated teacher feature files")
                print(f"    Sample dropped cases: {dropped[:10]}")
                print(f"    Remaining train cases: {len(kept)}")
        train_cases = kept
    
    # Get features format from config (default to 'npz' for backward compatibility)
    features_format = config.get('features_format', 'npz')
    if isinstance(features_format, str):
        # Apply same format to all teachers
        features_format_dict = {t: features_format for t in teacher_names}
    else:
        # features_format is already a dict mapping teacher -> format
        features_format_dict = features_format
    
    stats_per_case_file = config.get('stats_per_case_file', None)
    
    # Get pred_type from config (default to '3class' for backward compatibility)
    pred_type = config.get('pred_type', '3class')
    print(f"  Using pred_type: '{pred_type}' ({'2 channels' if pred_type == 'binary' else '3 channels'})")
    print(f"  Loading layer_minus_2 (neg2) features: {load_neg2}")
    
    # Create train dataset
    train_dataset = PIMEDDataset_WithFeatures(
        path_dict=path_dict,
        fold_cases=train_cases,
        case_id2cohort=case_id2cohort,
        features_path_dict=features_path_dict,
        teacher_names=teacher_names,
        features_format_dict=features_format_dict,
        load_features=load_features,
        cache_features=cache_features,
        cache_size=cache_size,
        load_neg2=load_neg2,  # Whether to load layer_minus_2 features
        stats_per_case_file=stats_per_case_file,
        pred_type=pred_type,  # Use pred_type from config ('3class' or 'binary')
        region_based=config.get('region_based', False),
        config=config
    )
    
    # Create val dataset (load_features=False since teacher features aren't used during validation)
    val_dataset = PIMEDDataset_WithFeatures(
        path_dict=path_dict,
        fold_cases=val_cases,
        case_id2cohort=case_id2cohort,
        features_path_dict=features_path_dict,
        teacher_names=teacher_names,
        features_format_dict=features_format_dict,
        load_features=False,
        cache_features=False,
        cache_size=0,
        load_neg2=False,  # Not needed for validation
        stats_per_case_file=stats_per_case_file,
        pred_type=pred_type,  # Use pred_type from config ('3class' or 'binary')
        region_based=config.get('region_based', False),
        config=config
    )
    
    # Select the appropriate collate functions
    # Train: with features (for KD), Val: without features (only need images+labels)
    train_collate_fn = collate_fn_with_features if load_features else collate_fn_without_features
    val_collate_fn = collate_fn_without_features  # Val never needs teacher features
    
    # Check if we're in distributed training mode
    import torch.distributed as dist
    is_distributed = dist.is_available() and dist.is_initialized()
    
    if is_distributed:
        # Use DistributedSampler for DDP - splits data across GPUs
        from torch.utils.data.distributed import DistributedSampler
        train_sampler = DistributedSampler(
            train_dataset,
            shuffle=True,  # Shuffle within each GPU's subset
            drop_last=False
        )
        val_sampler = DistributedSampler(
            val_dataset,
            shuffle=False,
            drop_last=False
        )
        # When using sampler, shuffle must be False
        train_shuffle = False
        val_shuffle = False
        print(f"==> Using DistributedSampler: {len(train_dataset)} train samples split across {dist.get_world_size()} GPUs")
    else:
        # Single GPU or DataParallel mode
        train_sampler = None
        val_sampler = None
        train_shuffle = True
        val_shuffle = False
    
    # Create train DataLoader
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=train_shuffle,
        sampler=train_sampler,
        num_workers=num_workers,
        collate_fn=train_collate_fn,
        pin_memory=True
    )
    
    # Create val DataLoader
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=val_shuffle,
        sampler=val_sampler,
        num_workers=num_workers,
        collate_fn=val_collate_fn,
        pin_memory=True
    )
    
    return train_loader, val_loader


def get_dataloader_with_features_test(config,
                                  batch_size=2,
                                  num_workers=4,
                                  load_features=True,
                                  cache_features=False):
    """
    Convenience function to create train and val DataLoaders with feature loading.
    Mimics the CIFAR-100 dataloader API: just pass config and get both loaders back.
    
    Args:
        config: Configuration dict (loaded from YAML file)
        batch_size: Batch size for DataLoader (default: 2)
        num_workers: Number of worker processes (default: 4)
        load_features: Whether to load pre-extracted features (default: True)
        cache_features: Whether to cache all features in memory (default: False)
        
    Returns:
        train_loader, val_loader (tuple of DataLoader instances)
        
    Example:
        import yaml
        with open('./dataset/pimed_dataset_configs.yaml', 'r') as f:
            config = yaml.safe_load(f)
        
        train_loader, val_loader = get_dataloader_with_features(
            config=config,
            batch_size=2,
            num_workers=4
        )
    """
    import json
    from torch.utils.data import DataLoader
    
    # Build path_dict and case lists from config
    case_id2cohort = {}
    with open(config['stanford_test_json']) as json_file:
        stanford_test_json = json.load(json_file)

    stanford_test_list = stanford_test_json['bx_test']
    stanford_test_ids = [test['Anon_ID'] for test in stanford_test_list]
    
    stanford_paths = config["image_dirs"]["stanford_paths"]
    path_dict = {
        "stanford": {
            "t2": stanford_paths["T2_dir"],
            "adc": stanford_paths["ADC_dir"],
            "dwi": stanford_paths["DWI_dir"],
            "prostate": stanford_paths["mask_dir"],
            "cancer": stanford_paths["lesions_dir"],
        }
    }
    
    for case in stanford_test_ids:
        case_id2cohort[str(case)] = "stanford"

    # Load PICAI data
    file_paths = config["image_dirs"]["file_paths"]
    path_dict["picai"] = {
        "t2": file_paths["T2_dir"],
        "adc": file_paths["ADC_dir"],
        "dwi": file_paths["DWI_dir"],
        "prostate": file_paths["mask_dir"],
        "cancer": file_paths["lesions_dir"],
    }
    
    with open(config["picai_test_json"]) as json_file:
        picai_test_ids = json.load(json_file)
    for case in picai_test_ids:
        case_id2cohort[case] = "picai"
    
    # Load UCLA data
    with open(config['ucla_test_json']) as json_file:
        ucla_test_case_ids = json.load(json_file)
    # ucla_test_case_ids = ucla_split_data['test']
    
    ucla_paths = config["image_dirs"]["ucla_paths"]
    path_dict["ucla"] = {
        "t2": ucla_paths["T2_dir"],
        "adc": ucla_paths["ADC_dir"],
        "dwi": ucla_paths["DWI_dir"],
        "prostate": ucla_paths["mask_dir"],
        "cancer": ucla_paths["lesions_dir"],
    }
    
    for case in ucla_test_case_ids:
        case_id2cohort[str(case)] = "ucla"
    
    # Combine case lists
    test_cases = stanford_test_ids + picai_test_ids + ucla_test_case_ids
    
    print(f'Test dataset statistics:')
    print(f'  Stanford: {len(stanford_test_ids)} test')
    print(f'  PICAI: {len(picai_test_ids)} test')
    print(f'  UCLA: {len(ucla_test_case_ids)} test')
    print(f'  Total: {len(test_cases)} test')
    
    # Create test dataset (no teacher features needed at test time)
    test_dataset = PIMEDDataset_WithFeatures(
        path_dict=path_dict,
        fold_cases=test_cases,
        case_id2cohort=case_id2cohort,
        features_path_dict=None,  # No teacher features at test time
        teacher_names=[],  # No teachers at test time
        features_format_dict={},  # No features format needed
        load_features=False,  # Don't load teacher features
        cache_features=False,
        stats_per_case_file=None,
        pred_type='3class',  # Use 3-class prediction (0=background, 1=peripheral, 2=transition)
        region_based=config.get('region_based', False),
        config=config
    )
    
    # Create DataLoader
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,  # Don't shuffle test data
        num_workers=num_workers,
        pin_memory=True,
        drop_last=False  # Keep all test samples
    )
    
    return test_loader

# Example usage and testing
if __name__ == '__main__':
    """
    Example usage:
    
    # OPTION 1: Use config file (easiest - mimics CIFAR-100 style)
    import yaml
    with open('./dataset/pimed_dataset_configs.yaml', 'r') as f:
        config = yaml.safe_load(f)
    
    train_loader, val_loader = get_dataloader_with_features(
        config=config,
        batch_size=2,
        num_workers=4,
        load_features=True,
        cache_features=False
    )
    
    # OPTION 2: Manual setup (more control)
    features_path_dict = {
        'nnunet': '/path/to/nnunet_features',
        'provicnet': '/path/to/provicnet_features',
        'prostatlasdiff': '/path/to/prostatlasdiff_features'
    }
    teacher_names = ['nnunet', 'provicnet', 'prostatlasdiff']
    
    train_loader, val_loader = get_dataloader_with_features(
        path_dict=your_path_dict,
        train_cases=train_case_ids,
        val_cases=val_case_ids,
        case_id2cohort=case_id2cohort,
        features_path_dict=features_path_dict,
        teacher_names=teacher_names,
        batch_size=2,
        num_workers=4,
        load_features=True
    )
    
    # Test loading
    print("Training loader:")
    for batch_idx, (imgs, labels, teacher_features, teacher_logits, case_ids) in enumerate(train_loader):
        print(f"Batch {batch_idx}:")
        print(f"  Images: {imgs.shape}")  # [B, 3, 20, 256, 256]
        print(f"  Labels: {labels.shape}")  # [B, 20, 256, 256]
        print(f"  Num teachers: {len(teacher_features)}")
        print(f"  Teacher 0 features[-2]: {teacher_features[0]['layer_minus_2'].shape}")
        print(f"  Teacher 0 features[-1]: {teacher_features[0]['layer_minus_1'].shape}")
        print(f"  Teacher 0 logits: {teacher_logits[0].shape}")
        print(f"  Case IDs: {case_ids}")
        break
    
    print("\nValidation loader:")
    for batch_idx, (imgs, labels, teacher_features, teacher_logits, case_ids) in enumerate(val_loader):
        print(f"Batch {batch_idx}:")
        print(f"  Images: {imgs.shape}")
        break
    """
    print("Import this module to use PIMEDDataset_WithFeatures")
    print("See docstring and __main__ section for usage examples")
