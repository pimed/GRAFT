"""
Validation Visualization Module
Saves 3D NIfTI files and 2D PNG center slice visualizations during validation
"""

import os
import torch
import numpy as np
import matplotlib.pyplot as plt
import SimpleITK as sitk
from matplotlib.colors import ListedColormap


class ValidationVisualizer:
    """
    Visualizer for validation results - saves both 3D NIfTI and 2D PNG visualizations
    
    Args:
        save_dir: Directory to save visualizations (can be checkpoint_dir or specific subdir)
        num_cases_to_visualize: Number of cases to visualize per epoch (default: 10)
        spacing: Tuple of (x, y, z) spacing for NIfTI files (default: (0.5, 0.5, 3.0))
        save_nifti: Whether to save full 3D NIfTI files (default: False for memory savings)
        region_based: Whether using region-based training (sigmoid + threshold) or class-based (softmax + argmax)
    """
    
    def __init__(self, save_dir, num_cases_to_visualize=10, spacing=(0.5, 0.5, 3.0), 
                 skip_first_n_cases=0, save_nifti=False, region_based=False):
        # Handle both old (checkpoint_dir) and new (save_dir) parameter names
        if os.path.basename(save_dir) == 'validation_vis':
            self.vis_dir = save_dir
        else:
            self.vis_dir = os.path.join(save_dir, 'validation_vis')
        os.makedirs(self.vis_dir, exist_ok=True)
        self.num_cases_to_visualize = num_cases_to_visualize
        self.skip_first_n_cases = skip_first_n_cases
        self.spacing = spacing
        self.save_nifti_flag = save_nifti  # Flag to control NIfTI saving (renamed to avoid conflict)
        self.region_based = region_based  # Flag for region-based vs class-based training
        
        # Track visualized cases separately for train and val modes
        self.visualized_cases_train = set()
        self.visualized_cases_val = set()
        self.skipped_cases_train = set()
        self.skipped_cases_val = set()
        
        # Define colormap for segmentation (4-class for region-based: background, prostate-only, PCa, csPCa)
        # Region-based: 0=background, 1=prostate (no cancer), 2=PCa (indolent), 3=csPCa (aggressive)
        colors = ['black', 'blue', 'green', 'yellow']
        self.seg_cmap = ListedColormap(colors)
        
    def _sigmoid(self, x):
        """Compute sigmoid activation"""
        return 1 / (1 + np.exp(-np.clip(x, -500, 500)))  # Clip to avoid overflow
    
    def _regions_to_classes(self, region_probs, threshold=0.5):
        """
        Convert region-based probabilities to class labels.
        Follows nnUNet's convert_probabilities_to_segmentation() logic.
        
        Region-based format (3 channels):
            Channel 0: Prostate region (labels 1, 2, 3)
            Channel 1: PCa region (labels 2, 3)  
            Channel 2: csPCa region (label 3)
            
        Output uses regions_class_order = [1, 2, 3]:
            0: Background (no prostate predicted)
            1: Prostate (label 1, may include benign tissue)
            2: PCa (label 2, indolent cancer)
            3: csPCa (label 3, aggressive cancer)
        
        Args:
            region_probs: [3, D, H, W] - sigmoid probabilities for each region
            threshold: threshold for binary prediction (default 0.5)
            
        Returns:
            class_labels: [D, H, W] - class labels 0-3
        """
        # nnUNet regions_class_order for PIMED: [1, 2, 3]
        regions_class_order = [1, 2, 3]
        
        # Initialize with zeros (background)
        class_labels = np.zeros(region_probs.shape[1:], dtype=np.uint8)
        
        # Apply each region in order (later regions overwrite earlier ones)
        for i, c in enumerate(regions_class_order):
            class_labels[region_probs[i] > threshold] = c
        
        return class_labels
    
    def _softmax(self, x, axis=0):
        """
        Compute softmax along specified axis
        
        Args:
            x: numpy array
            axis: axis along which to compute softmax
            
        Returns:
            softmax probabilities
        """
        e_x = np.exp(x - np.max(x, axis=axis, keepdims=True))
        return e_x / e_x.sum(axis=axis, keepdims=True)
        
    def save_nifti(self, img_array, filepath, spacing=None):
        """
        Save numpy array as NIfTI file with specified spacing
        
        Args:
            img_array: numpy array [D, H, W] or [C, D, H, W]
            filepath: output .nii.gz path
            spacing: (x, y, z) spacing tuple (default: self.spacing)
        """
        if spacing is None:
            spacing = self.spacing
            
        # Handle multi-channel images (e.g., 3 modalities)
        if img_array.ndim == 4:
            # Save each channel separately
            base_path = filepath.replace('.nii.gz', '')
            for c in range(img_array.shape[0]):
                channel_path = f"{base_path}_ch{c}.nii.gz"
                sitk_img = sitk.GetImageFromArray(img_array[c])
                sitk_img.SetSpacing(spacing)
                sitk.WriteImage(sitk_img, channel_path)
        else:
            # Single channel image [D, H, W]
            sitk_img = sitk.GetImageFromArray(img_array)
            sitk_img.SetSpacing(spacing)
            sitk.WriteImage(sitk_img, filepath)
    
    def visualize_case(self, inputs, labels, student_logits, teacher_logits, case_id, epoch, teacher_names, mode='val'):
        """
        Visualize a single case: save 3D NIfTI files and 2D PNG center slice
        
        Args:
            inputs: [3, D, H, W] - T2/ADC/DWI stacked volumes
            labels: [D, H, W] - Ground truth segmentation
            student_logits: [num_classes, D, H, W] - Student predictions
            teacher_logits: List of [num_classes, D, H, W] - Teacher predictions
            case_id: Case identifier string
            epoch: Current epoch number
            teacher_names: List of teacher names
            mode: str - 'val' or 'train' to distinguish visualization type
        """
        # Create epoch directory with mode prefix
        epoch_dir = os.path.join(self.vis_dir, f'epoch_{epoch:03d}_{mode}')
        os.makedirs(epoch_dir, exist_ok=True)
        
        # Create case subdirectory
        case_dir = os.path.join(epoch_dir, f'case_{case_id}')
        os.makedirs(case_dir, exist_ok=True)
        
        # Convert tensors to numpy if needed (handle BFloat16 from FP16 training)
        if torch.is_tensor(inputs):
            inputs = inputs.float().cpu().numpy()  # Convert to float32 first
        if torch.is_tensor(labels):
            labels = labels.cpu().numpy()
        if torch.is_tensor(student_logits):
            student_logits = student_logits.float().cpu().numpy()  # Convert to float32 first
        teacher_logits = [t.float().cpu().numpy() if torch.is_tensor(t) else t for t in teacher_logits]
        
        # Separate input modalities
        t2_np = inputs[0]  # [D, H, W]
        adc_np = inputs[1]  # [D, H, W]
        dwi_np = inputs[2]  # [D, H, W]
        
        # Convert logits to probabilities and predictions based on training mode
        if self.region_based:
            # Region-based: use sigmoid + threshold
            # Student logits: [3, D, H, W] (prostate, PCa, csPCa regions)
            student_probs = self._sigmoid(student_logits)  # [3, D, H, W]
            
            # Teacher logits: List of [3, D, H, W]
            teacher_probs = [self._sigmoid(t_logits) for t_logits in teacher_logits]
            
            # Convert region probabilities to class labels for visualization
            student_pred = self._regions_to_classes(student_probs)  # [D, H, W]
            teacher_preds = [self._regions_to_classes(t_probs) for t_probs in teacher_probs]
            
            # Handle labels format:
            # - If labels is [3, D, H, W] (region format), convert to class labels
            # - If labels is [D, H, W] (already class labels from targets_original), use directly
            if labels.ndim == 4 and labels.shape[0] == 3:
                # Region-based labels [3, D, H, W] -> class labels [D, H, W]
                labels = self._regions_to_classes(labels.astype(np.float32))
            # else: labels is already class-based [D, H, W] with values 0, 1, 2, 3
            
            # Update class names for region-based
            class_names = ['Prostate', 'PCa', 'csPCa']
        else:
            # Class-based: use softmax + argmax
            # Student logits: [num_classes, D, H, W]
            student_probs = self._softmax(student_logits, axis=0)  # [num_classes, D, H, W]
            
            # Teacher logits: List of [num_classes, D, H, W]
            teacher_probs = [self._softmax(t_logits, axis=0) for t_logits in teacher_logits]
            
            # Convert logits to predictions (argmax over class dimension)
            student_pred = np.argmax(student_logits, axis=0)  # [D, H, W]
            teacher_preds = [np.argmax(t_logits, axis=0) for t_logits in teacher_logits]  # List of [D, H, W]
            
            class_names = ['Background', 'Indolent', 'Aggressive']
        
        # --- Save full 3D NIfTI volumes (optional - can be disabled to save disk space) ---
        if self.save_nifti_flag:
            print(f"  Saving 3D NIfTI files for case {case_id}...")
            self.save_nifti(t2_np, os.path.join(case_dir, 't2.nii.gz'))
            self.save_nifti(adc_np, os.path.join(case_dir, 'adc.nii.gz'))
            self.save_nifti(dwi_np, os.path.join(case_dir, 'dwi.nii.gz'))
            self.save_nifti(labels.astype(np.uint8), os.path.join(case_dir, 'ground_truth.nii.gz'))
            for class_idx in range(student_probs.shape[0]):
                class_names = ['background', 'indolent', 'aggressive']
                self.save_nifti(student_probs[class_idx].astype(np.float32), 
                              os.path.join(case_dir, f'student_prob_{class_names[class_idx]}.nii.gz'))
            self.save_nifti(student_pred.astype(np.uint8), os.path.join(case_dir, 'student_pred.nii.gz'))
            for teacher_idx, (teacher_name, teacher_prob, teacher_pred) in enumerate(zip(teacher_names, teacher_probs, teacher_preds)):
                for class_idx in range(teacher_prob.shape[0]):
                    class_names = ['background', 'indolent', 'aggressive']
                    self.save_nifti(teacher_prob[class_idx].astype(np.float32), 
                                  os.path.join(case_dir, f'{teacher_name}_prob_{class_names[class_idx]}.nii.gz'))
                self.save_nifti(teacher_pred.astype(np.uint8), 
                              os.path.join(case_dir, f'{teacher_name}_pred.nii.gz'))
        
        # --- Create 2D center slice visualization ---
        print(f"  Creating 2D center slice visualization for case {case_id}...")
        
        # Debug: print unique values in labels
        print(f"  [DEBUG] Labels shape: {labels.shape}, unique values: {np.unique(labels)}")
        
        # Select the best slice to visualize
        # Cancer is defined as labels 2 (PCa) or 3 (csPCa) - NOT label 1 (prostate only)
        has_cancer = np.any(labels >= 2)  # Labels 2 or 3 are cancer
        
        if has_cancer:
            # Count cancer voxels per slice (labels >= 2, i.e., PCa or csPCa)
            cancer_per_slice = np.sum(labels >= 2, axis=(1, 2))  # [D]
            # Select slice with maximum cancer
            center_slice_idx = int(np.argmax(cancer_per_slice))
            slice_type = "max cancer"
        else:
            # No cancer - check if has prostate (label 1)
            has_prostate = np.any(labels >= 1)
            if has_prostate:
                # Select slice with most prostate tissue
                prostate_per_slice = np.sum(labels >= 1, axis=(1, 2))  # [D]
                center_slice_idx = int(np.argmax(prostate_per_slice))
                slice_type = "max prostate"
            else:
                # No prostate - use center slice
                center_slice_idx = labels.shape[0] // 2
                slice_type = "center"
        
        print(f"  Selected slice {center_slice_idx}/{labels.shape[0]} ({slice_type}) for visualization")
        
        # Layout: 
        # Row 1: T2, ADC, DWI, Ground Truth
        # Row 2: Student Background, Student Indolent, Student Aggressive, (empty or combined)
        # Row 3+: Teacher probs (one row per teacher with 3 probability maps)
        
        num_rows = 2 + len(teacher_names)  # First row (inputs+GT), Student row, Teacher rows
        num_cols = 4  # T2/ADC/DWI/GT or Background/Indolent/Aggressive/Combined
        
        fig, axes = plt.subplots(num_rows, num_cols, figsize=(16, num_rows * 4))
        
        # Ensure axes is 2D
        if num_rows == 1:
            axes = axes.reshape(1, -1)
        
        row_idx = 0
        
        # ===== ROW 1: Input modalities and Ground Truth =====
        # Column 0: T2-weighted
        axes[row_idx, 0].imshow(t2_np[center_slice_idx], cmap='gray')
        axes[row_idx, 0].set_title('T2-weighted', fontsize=14, fontweight='bold')
        axes[row_idx, 0].axis('off')
        
        # Column 1: ADC
        axes[row_idx, 1].imshow(adc_np[center_slice_idx], cmap='gray')
        axes[row_idx, 1].set_title('ADC', fontsize=14, fontweight='bold')
        axes[row_idx, 1].axis('off')
        
        # Column 2: DWI
        axes[row_idx, 2].imshow(dwi_np[center_slice_idx], cmap='gray')
        axes[row_idx, 2].set_title('DWI', fontsize=14, fontweight='bold')
        axes[row_idx, 2].axis('off')
        
        # Column 3: Ground Truth (vmax depends on region-based or class-based)
        gt_vmax = 3 if self.region_based else 2
        axes[row_idx, 3].imshow(labels[center_slice_idx], cmap=self.seg_cmap, vmin=0, vmax=gt_vmax)
        axes[row_idx, 3].set_title('Ground Truth', fontsize=14, fontweight='bold')
        axes[row_idx, 3].axis('off')
        
        row_idx += 1
        
        # ===== ROW 2: Student Probability Maps =====
        # class_names already set based on region_based flag above
        for col_idx in range(min(3, student_probs.shape[0])):
            im = axes[row_idx, col_idx].imshow(student_probs[col_idx, center_slice_idx], 
                                              cmap='hot', vmin=0, vmax=1)
            axes[row_idx, col_idx].set_title(f'Student: {class_names[col_idx]}', 
                                            fontsize=14, fontweight='bold')
            axes[row_idx, col_idx].axis('off')
            plt.colorbar(im, ax=axes[row_idx, col_idx], fraction=0.046, pad=0.04)
        
        # Column 3: Student prediction (threshold for region-based, argmax for class-based)
        vmax = 3 if self.region_based else 2
        axes[row_idx, 3].imshow(student_pred[center_slice_idx], cmap=self.seg_cmap, vmin=0, vmax=vmax)
        axes[row_idx, 3].set_title('Student Prediction', fontsize=14, fontweight='bold')
        axes[row_idx, 3].axis('off')
        
        row_idx += 1
        
        # ===== ROWS 3+: Teacher Probability Maps (one row per teacher) =====
        for teacher_name, teacher_prob, teacher_pred in zip(teacher_names, teacher_probs, teacher_preds):
            # Columns 0-2: Probability maps for each class
            for col_idx in range(min(3, teacher_prob.shape[0])):
                im = axes[row_idx, col_idx].imshow(teacher_prob[col_idx, center_slice_idx], 
                                                  cmap='hot', vmin=0, vmax=1)
                axes[row_idx, col_idx].set_title(f'{teacher_name}: {class_names[col_idx]}', 
                                                fontsize=14, fontweight='bold')
                axes[row_idx, col_idx].axis('off')
                plt.colorbar(im, ax=axes[row_idx, col_idx], fraction=0.046, pad=0.04)
            
            # Column 3: Teacher prediction
            axes[row_idx, 3].imshow(teacher_pred[center_slice_idx], cmap=self.seg_cmap, vmin=0, vmax=vmax)
            axes[row_idx, 3].set_title(f'{teacher_name} Prediction', fontsize=14, fontweight='bold')
            axes[row_idx, 3].axis('off')
            
            row_idx += 1
        
        # Add legend for segmentation colors (for ground truth and predictions)
        from matplotlib.patches import Patch
        if self.region_based:
            legend_elements = [
                Patch(facecolor='black', label='Background (0)'),
                Patch(facecolor='blue', label='Prostate (1)'),
                Patch(facecolor='green', label='PCa (2)'),
                Patch(facecolor='yellow', label='csPCa (3)')
            ]
            ncol = 4
        else:
            legend_elements = [
                Patch(facecolor='black', label='Background (0)'),
                Patch(facecolor='green', label='Indolent (1)'),
                Patch(facecolor='yellow', label='Aggressive (2)')
            ]
            ncol = 3
        fig.legend(handles=legend_elements, loc='lower center', ncol=ncol, frameon=True, 
                  fontsize=12, fancybox=True, shadow=True)
        
        # Create title with slice information
        # More detailed cancer status
        if has_cancer:
            has_cspca = np.any(labels == 3)
            cancer_status = "WITH csPCa" if has_cspca else "WITH PCa"
        else:
            cancer_status = "NO CANCER (prostate only)" if np.any(labels >= 1) else "NO PROSTATE"
        plt.suptitle(f'Case {case_id} ({cancer_status}) - Epoch {epoch} - Slice {center_slice_idx}/{labels.shape[0]} ({slice_type})', 
                    fontsize=18, fontweight='bold', y=0.995)
        plt.tight_layout(rect=[0, 0.02, 1, 0.99])
        
        # Save PNG
        png_path = os.path.join(case_dir, 'center_slice_comparison.png')
        plt.savefig(png_path, dpi=150, bbox_inches='tight')  # Reduced DPI from 200 to 150
        plt.close(fig)  # Explicitly close the figure
        
        # Clear matplotlib cache to free memory
        import gc
        gc.collect()
        
        print(f"  ✅ Saved 2D visualization for case {case_id}")
    
    def visualize_batch(self, inputs, labels, student_logits, teacher_logits, case_ids, epoch, teacher_names, mode='val'):
        """
        Visualize a batch of validation/training samples
        
        Args:
            inputs: [B, 3, D, H, W] - Batch of stacked T2/ADC/DWI volumes
            labels: [B, D, H, W] - Batch of ground truth segmentations
            student_logits: [B, num_classes, D, H, W] - Student predictions
            teacher_logits: List of [B, num_classes, D, H, W] - Teacher predictions
            case_ids: List of case identifier strings
            epoch: Current epoch number
            teacher_names: List of teacher names
            mode: str - 'val' or 'train' to distinguish visualization type
        """
        batch_size = inputs.shape[0]
        
        # Select the appropriate tracking sets based on mode
        if mode == 'train':
            visualized_cases = self.visualized_cases_train
            skipped_cases = self.skipped_cases_train
        else:  # mode == 'val'
            visualized_cases = self.visualized_cases_val
            skipped_cases = self.skipped_cases_val
        
        # Determine how many cases to visualize from this batch
        remaining_slots = self.num_cases_to_visualize - len(visualized_cases)
        if remaining_slots <= 0:
            return  # Already visualized enough cases
        
        num_to_visualize = min(batch_size, remaining_slots)
        
        for i in range(num_to_visualize):
            case_id = case_ids[i]
            
            # Skip if already visualized this case in this epoch (for this mode)
            if case_id in visualized_cases:
                continue
            
            # Skip first N cases if configured
            if len(skipped_cases) < self.skip_first_n_cases:
                skipped_cases.add(case_id)
                continue
            
            # Extract single sample from batch
            input_i = inputs[i]  # [3, D, H, W]
            label_i = labels[i]  # [D, H, W]
            student_logits_i = student_logits[i]  # [num_classes, D, H, W]
            teacher_logits_i = [t[i] for t in teacher_logits]  # List of [num_classes, D, H, W]
            
            # Visualize this case
            self.visualize_case(input_i, label_i, student_logits_i, teacher_logits_i, 
                              case_id, epoch, teacher_names, mode)
            
            # Clean up memory after visualizing each case
            del input_i, label_i, student_logits_i, teacher_logits_i
            import gc
            gc.collect()
            
            # Mark as visualized for this mode
            visualized_cases.add(case_id)
    
    def reset_for_new_epoch(self):
        """Reset the visualized cases counter for a new epoch"""
        self.visualized_cases_train.clear()
        self.visualized_cases_val.clear()
        self.skipped_cases_train.clear()
        self.skipped_cases_val.clear()
    
    def visualize_case_2d(self, case_id, t2, adc, dwi, ground_truth, prediction, epoch, phase='val'):
        """
        Simplified 2D visualization for standalone training (no teachers)
        
        Args:
            case_id: Case identifier
            t2: [D, H, W] - T2 volume
            adc: [D, H, W] - ADC volume (can be None)
            dwi: [D, H, W] - DWI volume (can be None)
            ground_truth: [D, H, W] - Ground truth labels
            prediction: [C, D, H, W] - Prediction probabilities (softmax)
            epoch: Current epoch number
            phase: 'train' or 'val'
        """
        # Select appropriate tracking sets based on phase
        if phase == 'train':
            visualized_cases = self.visualized_cases_train
            skipped_cases = self.skipped_cases_train
        else:
            visualized_cases = self.visualized_cases_val
            skipped_cases = self.skipped_cases_val
        
        # For training, limit visualizations
        if phase == 'train' and len(visualized_cases) >= self.num_cases_to_visualize:
            return
        
        # Skip if already visualized in this epoch
        if case_id in visualized_cases:
            return
        
        # Create directory structure
        phase_dir = os.path.join(self.vis_dir, phase, f'epoch_{epoch:03d}')
        os.makedirs(phase_dir, exist_ok=True)
        
        # Select slice: Use slice with most cancer if present, otherwise center slice
        depth = t2.shape[0]
        
        # Count cancer pixels per slice (class 1 and 2 are cancer)
        cancer_per_slice = np.sum(ground_truth > 0, axis=(1, 2))
        
        if cancer_per_slice.max() > 0:
            # Use slice with most cancer annotations
            center_slice = np.argmax(cancer_per_slice)
        else:
            # No cancer in this case, use center slice
            center_slice = depth // 2
        
        # Create figure
        num_cols = 3  # Input, GT, Prediction
        fig, axes = plt.subplots(1, num_cols, figsize=(15, 5))
        
        # Plot T2 input (center modality)
        axes[0].imshow(t2[center_slice], cmap='gray')
        axes[0].set_title(f'T2 Input\n{case_id}')
        axes[0].axis('off')
        
        # Plot ground truth
        axes[1].imshow(t2[center_slice], cmap='gray', alpha=0.7)
        gt_overlay = axes[1].imshow(ground_truth[center_slice], cmap=self.seg_cmap, 
                                     alpha=0.5, vmin=0, vmax=2)
        axes[1].set_title('Ground Truth')
        axes[1].axis('off')
        
        # Plot prediction
        pred_labels = np.argmax(prediction, axis=0)
        axes[2].imshow(t2[center_slice], cmap='gray', alpha=0.7)
        pred_overlay = axes[2].imshow(pred_labels[center_slice], cmap=self.seg_cmap,
                                      alpha=0.5, vmin=0, vmax=2)
        axes[2].set_title('Student Prediction')
        axes[2].axis('off')
        
        plt.tight_layout()
        
        # Save PNG
        png_path = os.path.join(phase_dir, f'{case_id}_center_slice.png')
        plt.savefig(png_path, dpi=150, bbox_inches='tight')
        plt.close(fig)
        
        print(f"✅ Saved {phase} visualization for {case_id} (epoch {epoch})")
        
        # Mark as visualized
        visualized_cases.add(case_id)
        
        # Save NIfTI if enabled
        if self.save_nifti:
            nifti_dir = os.path.join(phase_dir, case_id)
            os.makedirs(nifti_dir, exist_ok=True)
            
            self.save_nifti(t2, os.path.join(nifti_dir, 't2.nii.gz'))
            if adc is not None:
                self.save_nifti(adc, os.path.join(nifti_dir, 'adc.nii.gz'))
            if dwi is not None:
                self.save_nifti(dwi, os.path.join(nifti_dir, 'dwi.nii.gz'))
            self.save_nifti(ground_truth.astype(np.uint8), os.path.join(nifti_dir, 'ground_truth.nii.gz'))
            self.save_nifti(pred_labels.astype(np.uint8), os.path.join(nifti_dir, 'prediction.nii.gz'))
            print(f"  Saved NIfTI files for {case_id}")


def add_visualization_to_validation(visualizer, inputs, labels, logits, teacher_logits, 
                                    case_ids, epoch, teacher_names):
    """
    Helper function to add visualization during validation loop
    
    Args:
        visualizer: ValidationVisualizer instance
        inputs: [B, 3, D, H, W] - Batch of input volumes
        labels: [B, D, H, W] - Batch of labels
        logits: [B, num_classes, D, H, W] - Student predictions
        teacher_logits: List of [B, num_classes, D, H, W] - Teacher predictions (if available)
        case_ids: List of case IDs
        epoch: Current epoch
        teacher_names: List of teacher names
    """
    if visualizer is not None:
        try:
            visualizer.visualize_batch(inputs, labels, logits, teacher_logits, 
                                      case_ids, epoch, teacher_names)
        except Exception as e:
            print(f"⚠️ Warning: Visualization failed: {e}")


# Example usage
if __name__ == '__main__':
    """
    Example usage in training script:
    
    # In train_student_rl.py, after creating checkpoint_dir:
    from validation_visualizer import ValidationVisualizer
    
    # Create visualizer
    visualizer = ValidationVisualizer(
        checkpoint_dir=args.checkpoint_dir,
        num_cases_to_visualize=10,
        spacing=(0.5, 0.5, 3.0)
    )
    
    # In validation loop:
    for epoch in range(epochs):
        # Reset for new epoch
        visualizer.reset_for_new_epoch()
        
        # During validation
        for batch_idx, (inputs, labels, teacher_features, teacher_logits, case_ids) in enumerate(val_loader):
            # ... run student model ...
            features, logits = model(inputs, is_feat=True)
            
            # Visualize (only first 10 cases)
            visualizer.visualize_batch(
                inputs=inputs,
                labels=labels,
                student_logits=logits,
                teacher_logits=teacher_logits,
                case_ids=case_ids,
                epoch=epoch,
                teacher_names=args.teacher_name_list
            )
    """
    print("Import this module to use ValidationVisualizer")
    print("See docstring and __main__ section for usage examples")
