"""
Fixed Subset Shuffling Sampler for DDP with Feature Caching

Each GPU gets a FIXED subset of cases that never changes across epochs.
Within each epoch, the GPU shuffles only its own subset.

Benefits:
- Perfect for LRU caching: each GPU caches only its ~211 cases (for 1689 total / 8 GPUs)
- ~100% cache hit rate after epoch 1
- Much faster training (minimal disk I/O after warmup)
- Still has shuffling (within each GPU's subset)

Trade-offs:
- Less randomness than global shuffling (each GPU sees same cases)
- But with 211 cases per GPU and batch_size=2, still 105 diverse batches per epoch
"""

import math
import torch
import torch.distributed as dist
from torch.utils.data import Sampler


class FixedSubsetShufflingSampler(Sampler):
    """
    Sampler that divides dataset into fixed subsets per GPU rank.
    Each GPU shuffles only within its own subset.
    
    This is optimal for feature caching with DDP:
    - Each GPU caches only its subset (e.g., 211 cases for 1689 total / 8 GPUs)
    - Cache hit rate ~100% after epoch 1
    - No cache evictions due to seeing different cases
    
    Args:
        dataset: Dataset to sample from
        shuffle: Whether to shuffle within each GPU's subset (default: True)
        seed: Random seed for reproducibility (default: 0)
    
    Example:
        # 1689 cases, 8 GPUs
        # GPU 0 gets cases [0:211]
        # GPU 1 gets cases [211:422]
        # ...
        # GPU 7 gets cases [1478:1689]
        
        # Epoch 1: Each GPU shuffles its subset
        # GPU 0: [5, 120, 34, 198, ...]  (shuffled indices from [0:211])
        # GPU 1: [250, 305, 411, ...]    (shuffled indices from [211:422])
        
        # Epoch 2: Each GPU shuffles its subset AGAIN (different order)
        # GPU 0: [67, 8, 145, 23, ...]   (different shuffle of [0:211])
        # GPU 1: [389, 222, 301, ...]    (different shuffle of [211:422])
    """
    
    def __init__(self, dataset, shuffle=True, seed=0):
        self.dataset = dataset
        self.shuffle = shuffle
        self.seed = seed
        self.epoch = 0
        
        # Get DDP info
        if not dist.is_available():
            raise RuntimeError("Requires distributed package to be available")
        
        self.num_replicas = dist.get_world_size()
        self.rank = dist.get_rank()
        
        # Divide dataset into fixed chunks per GPU
        self.total_size = len(dataset)
        self.num_samples_per_rank = math.ceil(self.total_size / self.num_replicas)
        
        # Calculate this rank's slice of the dataset
        self.start_idx = self.rank * self.num_samples_per_rank
        self.end_idx = min(self.start_idx + self.num_samples_per_rank, self.total_size)
        self.num_samples = self.end_idx - self.start_idx
        
        if self.rank == 0:
            print(f"\n{'='*80}")
            print(f"FixedSubsetShufflingSampler Configuration")
            print(f"{'='*80}")
            print(f"Total dataset size: {self.total_size} cases")
            print(f"Number of GPUs: {self.num_replicas}")
            print(f"Cases per GPU: ~{self.num_samples_per_rank}")
            print(f"\nGPU subset assignments:")
            for rank in range(self.num_replicas):
                start = rank * self.num_samples_per_rank
                end = min(start + self.num_samples_per_rank, self.total_size)
                print(f"  GPU {rank}: cases [{start}:{end}] ({end-start} cases)")
            print(f"\n{'='*80}")
            print(f"Cache Strategy:")
            print(f"  - Each GPU will cache only its {self.num_samples} cases")
            print(f"  - Expected cache hit rate: ~100% after epoch 1")
            print(f"  - No cache evictions (same cases every epoch)")
            print(f"{'='*80}\n")
    
    def __iter__(self):
        """
        Generate indices for this epoch.
        Returns shuffled indices from this GPU's fixed subset.
        """
        # Generate this GPU's fixed subset of indices
        indices = list(range(self.start_idx, self.end_idx))
        
        if self.shuffle:
            # Create generator with seed based on epoch for reproducibility
            # Different epochs will have different shuffles
            g = torch.Generator()
            g.manual_seed(self.seed + self.epoch)
            
            # Shuffle indices within this GPU's subset
            shuffled_order = torch.randperm(len(indices), generator=g).tolist()
            indices = [indices[i] for i in shuffled_order]
        
        return iter(indices)
    
    def __len__(self):
        return self.num_samples
    
    def set_epoch(self, epoch):
        """
        Set the epoch for this sampler.
        This ensures different shuffling across epochs.
        
        Must be called at the start of each epoch in training loop:
            for epoch in range(epochs):
                sampler.set_epoch(epoch)
                for batch in dataloader:
                    ...
        """
        self.epoch = epoch


class FixedSubsetSequentialSampler(Sampler):
    """
    Non-shuffling variant of FixedSubsetShufflingSampler.
    Each GPU gets a fixed subset but returns indices in sequential order.
    
    Useful for validation/testing where you want deterministic ordering.
    """
    
    def __init__(self, dataset):
        self.dataset = dataset
        
        if not dist.is_available():
            raise RuntimeError("Requires distributed package to be available")
        
        self.num_replicas = dist.get_world_size()
        self.rank = dist.get_rank()
        
        self.total_size = len(dataset)
        self.num_samples_per_rank = math.ceil(self.total_size / self.num_replicas)
        
        self.start_idx = self.rank * self.num_samples_per_rank
        self.end_idx = min(self.start_idx + self.num_samples_per_rank, self.total_size)
        self.num_samples = self.end_idx - self.start_idx
    
    def __iter__(self):
        indices = list(range(self.start_idx, self.end_idx))
        return iter(indices)
    
    def __len__(self):
        return self.num_samples
    
    def set_epoch(self, epoch):
        """No-op for sequential sampler, but kept for API consistency"""
        pass
