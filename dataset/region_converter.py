"""
Region converter for BCE-based region training.
Converts class labels {0,1,2,3} to multi-hot region encoding.

Supports two modes:
- '3class': 3 output channels [prostate, PCa, csPCa]
- 'binary': 2 output channels [prostate, cancer (PCa+csPCa combined)]
"""
import torch
from typing import List, Tuple, Union


def get_regions_for_pred_type(pred_type: str) -> List[List[int]]:
    """
    Get the region definitions based on prediction type.
    
    Args:
        pred_type: '3class' or 'binary'
        
    Returns:
        List of label lists defining each region
        
    For '3class' (3 output channels):
        - Channel 0: Prostate (labels 1,2,3 - includes all foreground)
        - Channel 1: PCa (labels 2,3 - any cancer)
        - Channel 2: csPCa (label 3 only - clinically significant)
        
    For 'binary' (2 output channels):
        - Channel 0: Prostate (labels 1,2,3 - includes all foreground)
        - Channel 1: Cancer (labels 2,3 - PCa + csPCa combined)
    """
    if pred_type == '3class':
        return [[1, 2, 3], [2, 3], [3]]  # prostate, PCa, csPCa
    elif pred_type == 'binary':
        return [[1, 2, 3], [2, 3]]  # prostate, cancer (combined)
    else:
        raise ValueError(f"Unknown pred_type: {pred_type}. Expected '3class' or 'binary'.")


class ConvertSegmentationToRegions:
    """
    Convert segmentation labels to region-based multi-hot encoding.
    
    Compatible with nnUNet's ConvertSegmentationToRegionsTransform but works with numpy/torch.
    """
    def __init__(self, regions: Union[List, Tuple]):
        """
        Args:
            regions: List of label lists for each region
                     e.g., [[1,2,3], [2,3], [3]] for prostate, PCa, csPCa
        """
        self.regions = regions
    
    def __call__(self, segmentation: torch.Tensor) -> torch.Tensor:
        """
        Convert segmentation to regions.
        
        Args:
            segmentation: [D, H, W] or [1, D, H, W] - class labels {0,1,2,3}
        
        Returns:
            region_output: [C, D, H, W] - boolean tensor with C regions
        """
        # Handle both [D,H,W] and [1,D,H,W] inputs
        if segmentation.ndim == 4:
            segmentation = segmentation[0]  # Remove channel dim
        
        num_regions = len(self.regions)
        region_output = torch.zeros((num_regions, *segmentation.shape), 
                                    dtype=torch.bool, device=segmentation.device)
        
        for region_id, region_labels in enumerate(self.regions):
            if len(region_labels) == 1:
                # Single label region
                region_output[region_id] = (segmentation == region_labels[0])
            else:
                # Multi-label region - check if label is in region_labels
                mask = torch.zeros_like(segmentation, dtype=torch.bool)
                for label in region_labels:
                    mask |= (segmentation == label)
                region_output[region_id] = mask
        
        return region_output


def test_region_converter():
    """Test the region converter"""
    # Create test segmentation [1, 20, 256, 256]
    seg = torch.zeros(1, 20, 256, 256, dtype=torch.long)
    seg[:, 5:15, 100:150, 100:150] = 1  # Benign prostate
    seg[:, 7:12, 120:140, 120:140] = 2  # PCa
    seg[:, 8:11, 125:135, 125:135] = 3  # csPCa
    
    # Define regions like nnUNet
    regions = [
        [1, 2, 3],  # Prostate (all foreground)
        [2, 3],     # PCa (cancer)
        [3]         # csPCa
    ]
    
    converter = ConvertSegmentationToRegions(regions)
    region_output = converter(seg)
    
    print(f"Input shape: {seg.shape}")
    print(f"Input labels: {torch.unique(seg)}")
    print(f"Output shape: {region_output.shape}")
    print(f"Output dtype: {region_output.dtype}")
    
    # Check each region
    for i, region_labels in enumerate(regions):
        count = region_output[i].sum().item()
        print(f"Region {i} (labels {region_labels}): {count} voxels = {count/(20*256*256)*100:.2f}%")
    
    # Verify correctness
    # Region 0 should include all labels [1,2,3]
    expected_0 = (seg == 1) | (seg == 2) | (seg == 3)
    assert torch.all(region_output[0] == expected_0[0]), "Region 0 mismatch!"
    
    # Region 1 should include labels [2,3]
    expected_1 = (seg == 2) | (seg == 3)
    assert torch.all(region_output[1] == expected_1[0]), "Region 1 mismatch!"
    
    # Region 2 should include only label [3]
    expected_2 = (seg == 3)
    assert torch.all(region_output[2] == expected_2[0]), "Region 2 mismatch!"
    
    print("\n✅ All tests passed!")


if __name__ == '__main__':
    test_region_converter()
