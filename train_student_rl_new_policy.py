import argparse
import os

# Limit to GPUs 0, 1, 2 (set before importing torch!)
# Comment out or modify this line to use different GPUs
# COMMENTED OUT: Let the shell script control CUDA_VISIBLE_DEVICES instead
# os.environ['CUDA_VISIBLE_DEVICES'] = '0,1,2'

import random
import shutil
import time
import warnings
from enum import Enum
import datetime
import torch
import torch.backends.cudnn as cudnn
import torch.distributed as dist
import torch.multiprocessing as mp
import torch.nn as nn
import torch.nn.parallel
import torch.optim as optim
import torch.utils.data
import torch.utils.data.distributed
import torchvision.datasets as datasets
import torchvision.models as models
import torchvision.transforms as transforms
import sys
sys.stdout.reconfigure(encoding='utf-8', errors='replace')
sys.stderr.reconfigure(encoding='utf-8', errors='replace')


from train_loops_new_policy import train, test
from torch.utils.tensorboard import SummaryWriter
from validation_visualizer import ValidationVisualizer
from models import model_dict
from setting import  teacher_model_path_dict
from dataset.cifar100 import get_cifar100_dataloaders
from dataset.pimedloader_3d_with_features import get_dataloader_with_features
from utils import set_logger, cal_param_size, cal_multi_adds, AverageMeter, adjust_lr, DistillKL, correct_num
from distiller_zoo import get_contrastive_kd_loss
from models.util import Regress, TransFeat
from distiller_zoo import get_pimed_criterion
import models
from models.swin_unetr3d import swin_unetr3d_small


parser = argparse.ArgumentParser(description='PyTorch ImageNet Training')
parser.add_argument('--data', metavar='DIR', nargs='?', default='imagenet',
                    help='path to dataset (default: imagenet)')
parser.add_argument('--data-configs', metavar='PATH', type=str, default='./dataset/pimed_dataset_configs_local.yaml',
                    help='path to data config file for PIMED dataset (default: ./dataset/pimed_dataset_configs_local.yaml)')
parser.add_argument('-a', '--arch', metavar='ARCH', default='resnet18_imagenet')
parser.add_argument('-j', '--workers', default=8, type=int, metavar='N',
                    help='number of data loading workers (default: 4)')
parser.add_argument('--epochs', default=240, type=int, metavar='N',
                    help='number of total epochs to run')
parser.add_argument('--start-epoch', default=0, type=int, metavar='N',
                    help='manual epoch number (useful on restarts)')
parser.add_argument('-b', '--batch-size', default=64, type=int,
                    metavar='N',
                    help='mini-batch size (default: 256), this is the total '
                         'batch size of all GPUs on the current node when '
                         'using Data Parallel or Distributed Data Parallel')  # 32*2
                         
parser.add_argument('--lr', '--learning-rate', default=0.1, type=float,
                    metavar='LR', help='initial learning rate', dest='lr')
parser.add_argument('--momentum', default=0.9, type=float, metavar='M',
                    help='momentum')
parser.add_argument('--wd', '--weight-decay', default=1e-4, type=float,
                    metavar='W', help='weight decay (default: 1e-4)',
                    dest='weight_decay')
parser.add_argument('-p', '--print-freq', default=10, type=int,
                    metavar='N', help='print frequency (default: 10)')
parser.add_argument('--resume', default='', type=str, metavar='PATH',
                    help='path to latest checkpoint (default: none)')
parser.add_argument('--use-deep-supervision', action='store_true',
                    help='use deep supervision for classification loss (nnUNet-style multi-scale)')
parser.add_argument('--use-region-based-training', action='store_true',
                    help='use nnUNet-style region-based training and evaluation (exact nnUNet DC_and_CE_loss)')
parser.add_argument('--use-sharp-focal', action='store_true',
                    help='use sharper focal loss to reduce false positives on negative cases (gamma=1.5, focal weight=2)')
parser.add_argument('--use-mild-focal', action='store_true',
                    help='use mild focal loss (balanced gamma=2.0, equal weights) - better for larger models')
parser.add_argument('--use-high-sensitivity', action='store_true',
                    help='use high-sensitivity focal loss (pos_weight=20, channel_weight=3) to boost cancer detection')
parser.add_argument('--early-stopping', action='store_true',
                    help='enable early stopping based on validation cancer dice')
parser.add_argument('--early-stopping-patience', type=int, default=30,
                    help='number of epochs to wait for improvement before stopping (default: 30)')
parser.add_argument('--early-stopping-min-delta', type=float, default=0.001,
                    help='minimum improvement in cancer dice to reset patience (default: 0.001)')
parser.add_argument('-e', '--evaluate', dest='evaluate', action='store_true',
                    help='evaluate model on validation set')
parser.add_argument('--pretrained', dest='pretrained', action='store_true',
                    help='use pre-trained model')
parser.add_argument('--world-size', default=-1, type=int,
                    help='number of nodes for distributed training')
parser.add_argument('--rank', default=-1, type=int,
                    help='node rank for distributed training')
parser.add_argument('--dist-url', default='tcp://224.66.41.62:23456', type=str,
                    help='url used to set up distributed training')
parser.add_argument('--dist-backend', default='nccl', type=str,
                    help='distributed backend')
parser.add_argument('--seed', default=None, type=int,
                    help='seed for initializing training. ')
parser.add_argument('--gpu', default=None, type=int,
                    help='GPU id to use.')
parser.add_argument('--local-rank', default=-1, type=int,
                    help='local rank for distributed training (set by torch.distributed.launch)')
parser.add_argument('--multiprocessing-distributed', action='store_true',
                    help='Use multi-processing distributed training to launch '
                         'N processes per node, which has N GPUs. This is the '
                         'fastest way to use PyTorch for either single node or '
                         'multi node data parallel training')
parser.add_argument('--dummy', action='store_true', help="use fake data to benchmark")
parser.add_argument('--dynamic', action='store_true', help="use dynamic weight aggregation strategy")
parser.add_argument('--ce-weight', type=float, default=1, help='ce loss coefficient')
parser.add_argument('--kd-weight', type=float, default=1, help='kd loss coefficient')
parser.add_argument('--feat-weight', type=float, default=5, help='kd loss coefficient')
parser.add_argument('--milestones', default=[150,180,210], type=int, nargs='+', help='milestones for lr-multistep')
parser.add_argument('--init-lr', default=0.05, type=float, help='learning rate')
parser.add_argument('--lr-type', default='multistep', type=str, help='learning rate strategy')
parser.add_argument('--warmup-epochs', default=0, type=int, help='number of epochs for learning rate warmup (starts at 10% and increases to init-lr)')
parser.add_argument('--feat-kd', default='mse', type=str, help='feature kd loss')
parser.add_argument('--use-orthogonal', action='store_true', 
                    help='use orthogonal projection (VkD-style) for feature embedding layer')
parser.add_argument('--use-robust-loss', action='store_true',
                    help='use robust (Smooth-L1) loss for feature distillation (VkD-style, auto-enabled with --use-orthogonal)')
parser.add_argument('--kd-T', type=int, default=4, help='temperature')
parser.add_argument('--kd-loss-type', type=str, default='kl', choices=['kl', 'contrastive', 'relation', 'angular'],
                    help='KD loss type: "kl"=standard KL divergence, "contrastive"=margin-based, "relation"=pairwise similarity, "angular"=angular distance')
parser.add_argument('--contrastive-margin', type=float, default=0.5, help='margin for contrastive KD loss')
parser.add_argument('--agent-step', type=int, default=1000, help='agent optimization step (ignored if --use-episode-reward is True)')
parser.add_argument('--agent-lr', type=float, default=0.001, help='learning rate for the policy agent (default: 0.001)')
parser.add_argument('--agent-warmup-epochs', type=int, default=5, help='number of epochs to gradually blend uniform weights with agent weights (0=no warmup, use agent from start)')
parser.add_argument('--use-episode-reward', action='store_true', help='use episode-based returns with validation feedback instead of per-step immediate rewards')
parser.add_argument('--reward-gamma', type=float, default=0.95, help='discount factor for episode returns (only used with --use-episode-reward)')
parser.add_argument('--agent-update-interval', type=int, default=50, help='number of batches between agent updates to prevent memory explosion (only used with --use-episode-reward)')
parser.add_argument('--batches-per-epoch', type=int, default=None, help='limit number of batches per epoch (for memory-constrained episode buffer). None = use all batches')
parser.add_argument('--logits-actions', action='store_true', default=False, help='enable logits distillation with logits actions from policy model (in addition to feature actions)')
parser.add_argument('--reward-alpha', type=float, default=0.7, help='blend factor for ensemble-based reward: 1.0=pure dice improvement, 0.0=pure loss minimization (default: 0.7)')
parser.add_argument('--use-disagreement-reward', action='store_true', default=False, help='add per-teacher disagreement as 3rd scalar in agent state (in addition to cos_sim and bce). Helps agent identify which teachers have unique perspectives.')
parser.add_argument('--no-accuracy-gated-diversity', action='store_true', default=False, help='disable accuracy-gated diversity in teacher weighting (use only learned weights + BCE accuracy)')
parser.add_argument('--checkpoint-dir', default='./checkpoint', type=str, help='checkpoint directory')
parser.add_argument('--teacher-name-list', default=['resnet32x4', 'wrn_28_4'], type=str, nargs='+', help='teacher models')
parser.add_argument('--dataset', type=str, default='cifar100', choices=['cifar100', 'imagenet', 'tinyimagenet', 'dogs', 'cub_200_2011', 'mit67', 'pimed'], help='dataset')
parser.add_argument('--fp16', action='store_true', help='use FP16 mixed precision training to reduce memory usage')
parser.add_argument('--bf16', action='store_true', help='use BF16 mixed precision training (better for models with BF16 features, same memory as FP16)')
parser.add_argument('--gradient-accumulation-steps', type=int, default=1, help='number of gradient accumulation steps (simulates larger batch size)')
parser.add_argument('--trial', type=str, default='1', help='trial id')
parser.add_argument('--overfit-batches', type=int, default=None, help='number of batches to overfit on for testing (None = use full dataset)')
parser.add_argument('--val-mode', type=str, default='full', choices=['full', 'train', 'subset'], 
                    help='validation mode: "full"=use full val set, "train"=validate on training data (best for overfitting test), "subset"=use limited val cases')
parser.add_argument('--val-subset-size', type=int, default=20, help='number of validation cases to use when val-mode=subset')
parser.add_argument('--selected-cases', type=str, default=None, help='JSON file with pre-selected case IDs for training (e.g., from select_cases_simple.py)')
parser.add_argument('--use-loss-balancing', action='store_true', help='use EMA-based adaptive loss normalization to balance cls/kd/feat contributions')
parser.add_argument('--loss-balance-momentum', type=float, default=0.9, help='EMA momentum for loss balancing (default: 0.9, ~10-batch memory)')
parser.add_argument('--loss-balance-warmup', type=int, default=10, help='number of warmup steps before loss balancing starts (default: 10)')
parser.add_argument('--cache-features', action='store_true', help='cache teacher features in RAM for faster training')
parser.add_argument('--cache-size', type=int, default=300, help='LRU cache size (max cases per GPU). Use 0 for unlimited cache (WARNING: will OOM with shuffling!). Default: 300 (~1.6TB for 8 GPUs)')
parser.add_argument('--use-fixed-subset-sampler', action='store_true', 
                    help='use FixedSubsetShufflingSampler for DDP: each GPU gets fixed subset of cases (optimal for caching, ~100%% hit rate after epoch 1). If disabled, uses standard DistributedSampler with global shuffling (lower cache hit rate ~18%%, but more randomness)')
parser.add_argument('--innovation-suffix', type=str, default='', 
                    help='custom prefix for innovation suffix (e.g., "5fold_CV"). Will be prepended to innovation list like: customPrefix_contrastKD_episodeReward')
parser.add_argument('--load-neg2', action='store_true', default=False,
                    help='load layer_minus_2 (neg2) features from pre-extracted files. Default: False (only load neg1 and logits to save memory)')
parser.add_argument('--distill-features', nargs='+', default=['neg1'], choices=['neg1', 'neg2'],
                    help='Feature layers to distill. Options: neg1 (layer -1), neg2 (layer -2). '
                         'Examples: --distill-features neg1 (default), --distill-features neg2, '
                         '--distill-features neg1 neg2 (both). At least one must be specified. '
                         'neg2 requires --load-neg2.')
parser.add_argument('--neg2-weight', type=float, default=1.0,
                    help='Weight multiplier for neg2 feature loss when using both neg1 and neg2 distillation. '
                         'Default: 1.0 (equal weight). Use >1 to emphasize neg2 features.')
parser.add_argument('--use-feature-masking', action='store_true', default=False,
                    help='Apply cancer-correctness masking to feature distillation: only distill where teacher is correct about cancer (TP+TN), skip FP/FN')
parser.add_argument('--mask-dilation-radius', type=int, default=3,
                    help='3D dilation radius around cancer TP voxels for feature masking (default: 3, kernel=7x7x7)')



def get_tensorboard_path(path):
    time_stamp = datetime.datetime.now().strftime("%d-%m-%Y_%H-%M-%S")
    write_path = path + '/' + time_stamp
    os.makedirs(write_path)
    return write_path

def main():
    args = parser.parse_args()
    args.teacher_name_str = "_".join(args.teacher_name_list)
    print('args.teacher_name_str', args.teacher_name_str)
    args.teacher_num = len(args.teacher_name_list)

    # Build model name with overfit indicator and innovations suffix
    innovations_suffix = ''
    innovations_list = []
    
    # Prepend custom suffix if provided
    if args.innovation_suffix:
        innovations_list.append(args.innovation_suffix)
    
    # Check each innovation individually
    if hasattr(args, 'kd_loss_type') and args.kd_loss_type == 'contrastive':
        innovations_list.append('contrastKD')
    if hasattr(args, 'use_episode_reward') and args.use_episode_reward:
        innovations_list.append('episodeReward')
    if hasattr(args, 'use_loss_balancing') and args.use_loss_balancing:
        innovations_list.append('lossBalance')
    
    if innovations_list:
        innovations_suffix = '_' + '_'.join(innovations_list)
    
    if args.selected_cases is not None:
        # Using selected balanced cases
        args.model_name = args.arch + '_'+ args.dataset+ '_'+ 'rl'+'_'+ 'select20'+'_'+str(args.teacher_num)+'_'+args.teacher_name_str + innovations_suffix
    elif args.overfit_batches is not None:
        args.model_name = args.arch + '_'+ args.dataset+ '_'+ 'rl'+'_'+ 'overfit'+'_'+str(args.teacher_num)+'_'+args.teacher_name_str + innovations_suffix
    else:
        args.model_name = args.arch + '_'+ args.dataset+ '_'+ 'rl'+'_'+ args.trial+'_'+str(args.teacher_num)+'_'+args.teacher_name_str + innovations_suffix

    # Determine checkpoint directory - reuse existing if resuming, otherwise create new
    if len(args.resume) != 0:
        # Resuming from checkpoint - extract directory from checkpoint path
        # Example: ./checkpoint/unet3d_pimed_rl_1_2_nnunet_prostatlasdiff_15-12-2025_10-30-45/unet3d_best.pth.tar
        # Extract: ./checkpoint/unet3d_pimed_rl_1_2_nnunet_prostatlasdiff_15-12-2025_10-30-45
        args.checkpoint_dir = os.path.dirname(args.resume)
        # Extract info from existing checkpoint directory name (last part of path)
        info = os.path.basename(args.checkpoint_dir)
        print(f'===> Resuming training - using existing checkpoint dir: {args.checkpoint_dir}')
    else:
        # Starting fresh - create new timestamped directory
        info_time = datetime.datetime.now().strftime("%d-%m-%Y_%H-%M-%S")
        info = args.model_name + info_time
        print(f'===> info is : {info}')
        args.checkpoint_dir = os.path.join(args.checkpoint_dir, info)
        print(f'===> Starting new training - checkpoint dir: {args.checkpoint_dir}')

    # In DDP, check environment variable; otherwise check args.rank
    is_main_process = True
    if "RANK" in os.environ:
        is_main_process = (int(os.environ["RANK"]) == 0)
    elif args.rank is not None and args.rank != -1:
        is_main_process = (args.rank == 0)
    
    if is_main_process:
        # Only main process (rank 0) creates directories to avoid race condition in DDP
        if not os.path.isdir(args.checkpoint_dir):
            os.makedirs(args.checkpoint_dir, exist_ok=True)
        args.log_txt = os.path.join(args.checkpoint_dir, info + '.txt')
        args.logger = set_logger(args.log_txt)
        args.logger.info("==========\nArgs:{}\n==========".format(args))
        
        # Copy selected cases file to checkpoint directory if used
        if args.selected_cases is not None:
            import json
            dest_path = os.path.join(args.checkpoint_dir, 'selected_cases.json')
            # Read and write to avoid permission issues with shutil.copy on WSL
            with open(args.selected_cases, 'r') as src:
                with open(dest_path, 'w') as dst:
                    dst.write(src.read())
            args.logger.info(f"Copied selected cases file to: {dest_path}")
        
        # Setup TensorBoard writer
        tensorboard_dir = os.path.join(args.checkpoint_dir, 'tensorboard')
        if not os.path.isdir(tensorboard_dir):
            os.makedirs(tensorboard_dir, exist_ok=True)
        args.writer = SummaryWriter(tensorboard_dir)
        args.logger.info(f"TensorBoard logging to: {tensorboard_dir}")
        
        # Setup validation visualizer (only for PIMED dataset)
        if args.dataset == 'pimed':
            vis_dir = os.path.join(args.checkpoint_dir, 'visualizations')
            # Pass region_based flag to visualizer for correct prediction conversion
            region_based = hasattr(args, 'use_region_based_training') and args.use_region_based_training
            args.visualizer = ValidationVisualizer(save_dir=vis_dir, save_nifti=False, region_based=region_based)
            args.logger.info(f"Validation visualizations will be saved to: {vis_dir}")
            args.logger.info(f"NIfTI saving DISABLED (PNG only) to save disk space")
            args.logger.info(f"Visualizer using {'region-based (sigmoid+threshold)' if region_based else 'class-based (softmax+argmax)'} prediction conversion")
        else:
            args.visualizer = None
        
        # Setup CSV logging
        args.csv_path = os.path.join(args.checkpoint_dir, 'training_log.csv')
        import csv
        with open(args.csv_path, 'w', newline='') as f:
            csv_writer = csv.writer(f)
            # Add val_dice and val_iou columns for segmentation tasks
            if args.dataset == 'pimed':
                csv_writer.writerow(['epoch', 'train_loss', 'train_loss_cls', 'train_loss_kd', 
                                    'train_loss_feat', 'train_metric', 'val_loss', 'val_dice', 'val_iou', 'learning_rate'])
            else:
                csv_writer.writerow(['epoch', 'train_loss', 'train_loss_cls', 'train_loss_kd', 
                                    'train_loss_feat', 'train_metric', 'val_loss', 'val_metric', 'learning_rate'])
        args.logger.info(f"CSV logging to: {args.csv_path}")

    if args.seed is not None :
        random.seed(args.seed)
        torch.manual_seed(args.seed)
        cudnn.deterministic = True
        cudnn.benchmark = False
        warnings.warn('You have chosen to seed training. '
                      'This will turn on the CUDNN deterministic setting, '
                      'which can slow down your training considerably! '
                      'You may see unexpected behavior when restarting '
                      'from checkpoints.')
    
    if args.gpu is not None :
        warnings.warn('You have chosen a specific GPU. This will completely '
                      'disable data parallelism.')
    
    if args.dist_url == "env://" and args.world_size == -1:
        args.world_size = int(os.environ["WORLD_SIZE"])

    args.distributed = args.world_size > 1 or args.multiprocessing_distributed # True
    print(f'======> args.distributed is {args.distributed}')

    if torch.cuda.is_available():
        ngpus_per_node = torch.cuda.device_count()
    else:
        ngpus_per_node = 1
    print(f'======> ngpus_per_node is {ngpus_per_node}') 
    
    # Check if we're already in a torchrun/torch.distributed.run process
    if "RANK" in os.environ and "LOCAL_RANK" in os.environ:
        # torchrun already spawned processes, don't use mp.spawn
        print(f'======> Using torchrun-spawned process (RANK={os.environ["RANK"]}, LOCAL_RANK={os.environ["LOCAL_RANK"]})')
        # Don't set args.rank here - let main_worker read it from environment
        args.gpu = int(os.environ["LOCAL_RANK"])
        main_worker(args.gpu, ngpus_per_node, args)
    elif args.multiprocessing_distributed:
        # Manual multiprocessing with mp.spawn
        args.world_size = ngpus_per_node * args.world_size 
        mp.spawn(main_worker, nprocs=ngpus_per_node, args=(ngpus_per_node, args))
    else:
        main_worker(args.gpu, ngpus_per_node, args)  


def get_agent(teacher_models, args, teacher_feature_shapes=None):
    """
    Create agent for teacher weight prediction using PolicyTrans with conv encoders.
    
    PolicyTrans uses spatial conv encoding for teacher embeddings and logits.
    The first conv layer is created dynamically based on actual input channels,
    so we don't need to pass input_size.
    
    Args:
        teacher_models: List of teacher models (for CIFAR), or None for PIMED
        args: Training arguments
        teacher_feature_shapes: For PIMED dataset with pre-extracted features, 
            a list of dicts for each teacher containing:
            {'layer_minus_2_shape': (B, C, D, H, W), 'layer_minus_1_shape': (B, C, D, H, W), 'logits_shape': (B, num_cls, D, H, W)}
    
    Returns:
        agent: PolicyTrans model
        Sets appropriate args.t_feat_dims* based on distill_features list
    """
    use_3d = False
    distill_features = getattr(args, 'distill_features', ['neg1'])
    distill_neg1 = 'neg1' in distill_features
    distill_neg2 = 'neg2' in distill_features
    
    if teacher_feature_shapes is not None:
        # PIMED case: Use pre-extracted feature shapes
        teacher_num = len(teacher_feature_shapes)
        feature_dims_neg1 = []
        feature_dims_neg2 = []
        
        for t_shapes in teacher_feature_shapes:
            # feature_dims stores the shape of layer_minus_1 (for TransFeat embedding)
            feat_neg1_shape = t_shapes['layer_minus_1_shape']  # [B, C, D, H, W]
            feature_dims_neg1.append(feat_neg1_shape)
            
            # Also get neg2 shape if available and distill_neg2 is enabled
            if distill_neg2 and 'layer_minus_2_shape' in t_shapes:
                feat_neg2_shape = t_shapes['layer_minus_2_shape']
                feature_dims_neg2.append(feat_neg2_shape)
        
        # Store feature dimensions in args for get_feat_trans
        # Always store what we have for consistency
        if distill_neg1:
            args.t_feat_dims_neg1 = feature_dims_neg1
            args.t_feat_dims = feature_dims_neg1  # Backward compatibility
        if distill_neg2:
            args.t_feat_dims_neg2 = feature_dims_neg2
        
        # Detect if we're using 3D features (check if shape has 5 dimensions: B, C, D, H, W)
        feat_neg1_shape = teacher_feature_shapes[0]['layer_minus_1_shape']
        use_3d = len(feat_neg1_shape) == 5
    else:
        # CIFAR case: Run teacher models to get dimensions
        teacher_num = len(teacher_models)
        x = torch.rand(args.res).cuda()
        feature_dims_neg1 = []
        feature_dims_neg2 = []
        for t in teacher_models:
            feature, logits = t(x, is_feat=True)
            feature_dims_neg1.append(feature[-1].size())
            feature_dims_neg2.append(feature[-2].size())
        
        # Store feature dimensions in args for get_feat_trans
        if distill_neg1:
            args.t_feat_dims_neg1 = feature_dims_neg1
            args.t_feat_dims = feature_dims_neg1  # Backward compatibility
        if distill_neg2:
            args.t_feat_dims_neg2 = feature_dims_neg2
        
        # Detect if we're using 3D features (check if args.res has 5 dimensions)
        use_3d = len(args.res) == 5
    
    # PolicyTrans creates first conv layer dynamically based on actual input channels
    # input_size is ignored but kept for backward compatibility
    enable_logits_actions = getattr(args, 'logits_actions', False)
    use_disagreement = getattr(args, 'use_disagreement_reward', False)
    
    # Calculate number of scalars per teacher based on:
    # - neg1 only: 2 scalars (cos_sim, bce) or 3 (cos_sim, bce, disagreement)
    # - neg2 only: 2 scalars (cos_sim, bce) or 3 (cos_sim, bce, disagreement)
    # - both: 3 scalars (neg1_cos_sim, neg2_cos_sim, bce) or 4 (neg1_cos_sim, neg2_cos_sim, bce, disagreement)
    distill_both = distill_neg1 and distill_neg2
    if distill_both:
        num_scalars_per_teacher = 4 if use_disagreement else 3
    else:
        num_scalars_per_teacher = 3 if use_disagreement else 2
    
    # Enable separate neg1/neg2 actions in PolicyTrans based on distill_features
    use_accuracy_gated_diversity = not getattr(args, 'no_accuracy_gated_diversity', False)
    agent = model_dict['PolicyTrans'](
        input_size=None, 
        teacher_num=teacher_num, 
        dynamic=args.dynamic, 
        use_3d=use_3d, 
        enable_logits_actions=enable_logits_actions, 
        num_scalars_per_teacher=num_scalars_per_teacher,
        enable_neg1_actions=distill_neg1,
        enable_neg2_actions=distill_neg2,
        use_accuracy_gated_diversity=use_accuracy_gated_diversity
    ).cuda()
    return agent, feature_dims_neg1 if distill_neg1 else feature_dims_neg2


def get_feat_trans(model, args):
    """
    Extract student feature dimensions by running a forward pass.
    
    Model should be on CPU when this is called. We temporarily move it to a GPU,
    run the forward pass, then move it back to CPU. This avoids any device conflicts
    before DDP initialization.
    
    Returns:
        Based on distill_features list:
        - ['neg1'] (default): feat_trans_neg1
        - ['neg2']: feat_trans_neg2
        - ['neg1', 'neg2'] or ['neg2', 'neg1']: (feat_trans_neg2, feat_trans_neg1)
    """
    # Determine which GPU to use for the forward pass
    gpu_id = args.gpu if args.gpu is not None else 0
    
    # Temporarily move model to GPU
    model = model.cuda(gpu_id)
    model.eval()
    
    with torch.no_grad():
        s_feat, s_logits = model(torch.rand(args.res).cuda(gpu_id), is_feat=True)
    
    # Get dimensions for both layers
    args.s_feat_dim_neg1 = s_feat[-1].size()  # Layer -1 (embeddings)
    args.s_feat_dim_neg2 = s_feat[-2].size()  # Layer -2 (features)
    
    # For backward compatibility
    args.s_feat_dim = args.s_feat_dim_neg1
    
    # Move model back to CPU
    model = model.cpu()
    
    # Check flags from distill_features list
    use_orthogonal = getattr(args, 'use_orthogonal', False)
    distill_features = getattr(args, 'distill_features', ['neg1'])
    distill_neg1 = 'neg1' in distill_features
    distill_neg2 = 'neg2' in distill_features
    
    if distill_neg1 and distill_neg2:
        # Create two separate TransFeat modules for neg2 and neg1
        feat_trans_neg2 = TransFeat(args.s_feat_dim_neg2, args.t_feat_dims_neg2, use_orthogonal=use_orthogonal)
        feat_trans_neg1 = TransFeat(args.s_feat_dim_neg1, args.t_feat_dims_neg1, use_orthogonal=use_orthogonal)
        print(f"===> Created two TransFeat modules (both neg1 and neg2):")
        print(f"     feat_trans_neg2: student {args.s_feat_dim_neg2} -> teachers {[t for t in args.t_feat_dims_neg2]}")
        print(f"     feat_trans_neg1: student {args.s_feat_dim_neg1} -> teachers {[t for t in args.t_feat_dims_neg1]}")
        return feat_trans_neg2, feat_trans_neg1
    elif distill_neg2:
        # Only neg2: single TransFeat for layer -2
        feat_trans_neg2 = TransFeat(args.s_feat_dim_neg2, args.t_feat_dims_neg2, use_orthogonal=use_orthogonal)
        print(f"===> Created TransFeat for neg2 only:")
        print(f"     feat_trans_neg2: student {args.s_feat_dim_neg2} -> teachers {[t for t in args.t_feat_dims_neg2]}")
        return feat_trans_neg2
    else:
        # Only neg1 (default): single TransFeat for layer -1
        feat_trans_neg1 = TransFeat(args.s_feat_dim_neg1, args.t_feat_dims_neg1, use_orthogonal=use_orthogonal)
        print(f"===> Created TransFeat for neg1 only (default):")
        print(f"     feat_trans_neg1: student {args.s_feat_dim_neg1} -> teachers {[t for t in args.t_feat_dims_neg1]}")
        return feat_trans_neg1
        
def main_worker(gpu, ngpus_per_node, args):
    
    args.gpu = gpu

    if args.gpu is not None:
        print("Use GPU: {} for training".format(args.gpu))

    if args.distributed:
        if args.dist_url == "env://" and args.rank == -1:
            args.rank = int(os.environ["RANK"])
            args.world_size = int(os.environ["WORLD_SIZE"])
            # torchrun sets world_size and rank correctly, don't modify them
        elif args.multiprocessing_distributed:
            # Only modify rank if using manual mp.spawn (not torchrun)
            args.rank = args.rank * ngpus_per_node + gpu 
        dist.init_process_group(backend=args.dist_backend, init_method=args.dist_url,
                                world_size=args.world_size, rank=args.rank)
        
        
    def load_teacher(model_path, n_cls, model_t, opt=None):
        model = model_dict[model_t](num_classes=n_cls).cuda()
        map_location = None if opt.gpu is None else {'cuda:0': 'cuda:%d' % (opt.gpu if opt.multiprocessing_distributed else 0)}
        model.load_state_dict(torch.load(model_path, map_location=map_location)['model'])
        model.eval()
        for t_n, t_p in model.named_parameters():
            t_p.requires_grad = False
        return model


    # def load_teacher_list(opt):
    #     print('==> loading teacher model list')
    #     teacher_model_list = [load_teacher(teacher_model_path_dict[model_name], args.n_cls, model_name, opt)
    #                         for model_name in opt.teacher_name_list]
    #     print('==> done')
    #     return teacher_model_list

    if args.dataset.startswith('cifar100'):
        args.n_cls = 100
        args.res = (1, 3, 32, 32)
    elif args.dataset.startswith('imagenet'):
        args.n_cls = 1000
        args.res = (1, 3, 224, 224)
    elif args.dataset == 'pimed':
        # Load config first to determine pred_type
        import yaml
        with open(args.data_configs, 'r') as f:
            config = yaml.safe_load(f)
        
        # Determine number of output classes based on pred_type
        pred_type = config.get('pred_type', '3class')
        if pred_type == 'binary':
            args.n_cls = 2  # 2 channels: prostate, cancer (PCa+csPCa combined)
            print(f"==> Binary mode (pred_type='binary'): Using {args.n_cls} output channels [prostate, cancer]")
        else:
            args.n_cls = 3  # 3 channels: prostate, PCa, csPCa
            print(f"==> 3-class mode (pred_type='3class'): Using {args.n_cls} output channels [prostate, PCa, csPCa]")
        args.res = (1, 3, 20, 256, 256)  # [B, C=3 (T2/ADC/DWI), D=20, H=256, W=256]
        args.pred_type = pred_type  # Store for later use
    
    ################### load data FIRST (for PIMED to extract feature shapes) ###################
    if args.dataset == 'cifar100':
        train_loader, val_loader = get_cifar100_dataloaders(data_folder=args.data,
                                                            batch_size=args.batch_size,
                                                            num_workers=args.workers)
        # For CIFAR, load teachers the traditional way
        teacher_models = [load_teacher(teacher_model_path_dict[model_name], args.n_cls, model_name, args)
                         for model_name in args.teacher_name_list]
        
        # Handle overfit mode for CIFAR-100
        if args.overfit_batches is not None:
            print(f'==> OVERFITTING MODE: Using only {args.overfit_batches} training batches')
            from torch.utils.data import Subset
            dataset = train_loader.dataset
            subset_size = min(args.overfit_batches * args.batch_size, len(dataset))
            subset_indices = list(range(subset_size))
            train_subset = Subset(dataset, subset_indices)
            train_loader = torch.utils.data.DataLoader(
                train_subset,
                batch_size=args.batch_size,
                shuffle=False,
                num_workers=args.workers,
                pin_memory=True,
                drop_last=False
            )
            print(f'==> Training on {len(train_subset)} samples ({len(train_loader)} batches)')
        
        # Handle validation mode for CIFAR-100
        if args.val_mode == 'train':
            print(f'==> VALIDATION MODE: Using training data for validation (best for overfitting test)')
            val_loader = train_loader
        elif args.val_mode == 'subset':
            print(f'==> VALIDATION MODE: Using subset of {args.val_subset_size} validation samples')
            from torch.utils.data import Subset
            val_dataset = val_loader.dataset
            subset_size = min(args.val_subset_size, len(val_dataset))
            subset_indices = list(range(subset_size))
            val_subset = Subset(val_dataset, subset_indices)
            val_loader = torch.utils.data.DataLoader(
                val_subset,
                batch_size=args.batch_size,
                shuffle=False,
                num_workers=args.workers,
                pin_memory=True,
                drop_last=False
            )
            print(f'==> Validating on {len(val_subset)} samples ({len(val_loader)} batches)')
        else:
            print(f'==> VALIDATION MODE: Using full validation set')
            
    elif args.dataset == 'pimed':
        # Config already loaded above when determining n_cls
        train_loader, val_loader = get_dataloader_with_features(
            config=config,
            batch_size=args.batch_size,
            num_workers=args.workers,
            load_features=True,
            cache_features=args.cache_features,  # Use command-line argument
            cache_size=args.cache_size,  # LRU cache size (0 = unlimited)
            load_neg2=args.load_neg2  # Whether to load layer_minus_2 features
        )
        
        # Filter by selected cases if JSON file provided
        if args.selected_cases is not None:
            print(f'==> SELECTED CASES MODE: Loading cases from {args.selected_cases}')
            import json
            with open(args.selected_cases, 'r') as f:
                selected_data = json.load(f)
            selected_case_ids = set(selected_data['case_ids'])
            
            # Filter training dataset
            from torch.utils.data import Subset
            dataset = train_loader.dataset
            selected_indices = [i for i, case_id in enumerate(dataset.case_ids) 
                              if case_id in selected_case_ids]
            
            if len(selected_indices) == 0:
                raise ValueError(f"No matching cases found! Check that case IDs in {args.selected_cases} match dataset.")
            
            train_subset = Subset(dataset, selected_indices)
            
            # Recreate dataloader with selected cases
            from dataset.pimedloader_3d_with_features import collate_fn_with_features
            train_loader = torch.utils.data.DataLoader(
                train_subset,
                batch_size=args.batch_size,
                shuffle=False,  # Don't shuffle for overfitting (want same cases each epoch)
                num_workers=args.workers,
                collate_fn=collate_fn_with_features,
                pin_memory=True,
                drop_last=False
            )
            print(f'==> Training on {len(train_subset)} selected cases ({len(train_loader)} batches)')
            print(f'    {selected_data["description"]}')
        
        # Overfit on a small subset if requested (for testing)
        elif args.overfit_batches is not None:
            print(f'==> OVERFITTING MODE: Using only {args.overfit_batches} training batches')
            from torch.utils.data import Subset
            
            # Calculate indices for subset
            dataset = train_loader.dataset
            subset_size = min(args.overfit_batches * args.batch_size, len(dataset))
            subset_indices = list(range(subset_size))
            
            # Create subset dataset
            train_subset = Subset(dataset, subset_indices)
            
            # Recreate dataloader with subset
            train_loader = torch.utils.data.DataLoader(
                train_subset,
                batch_size=args.batch_size,
                shuffle=False,  # Don't shuffle for overfitting (want same batches each epoch)
                num_workers=args.workers,
                pin_memory=True,
                drop_last=False
            )
            print(f'==> Training on {len(train_subset)} samples ({len(train_loader)} batches)')
        
        # Handle validation mode
        if args.val_mode == 'train':
            print(f'==> VALIDATION MODE: Using training data for validation (best for overfitting test)')
            val_loader = train_loader
        elif args.val_mode == 'subset':
            print(f'==> VALIDATION MODE: Using subset of {args.val_subset_size} validation cases')
            from torch.utils.data import Subset
            val_dataset = val_loader.dataset
            subset_size = min(args.val_subset_size, len(val_dataset))
            subset_indices = list(range(subset_size))
            val_subset = Subset(val_dataset, subset_indices)
            
            # Recreate validation dataloader with subset
            from dataset.pimedloader_3d_with_features import collate_fn_with_features
            val_loader = torch.utils.data.DataLoader(
                val_subset,
                batch_size=args.batch_size,
                shuffle=False,
                num_workers=args.workers,
                collate_fn=collate_fn_with_features,
                pin_memory=True,
                drop_last=False
            )
            print(f'==> Validating on {len(val_subset)} samples ({len(val_loader)} batches)')
        else:
            print(f'==> VALIDATION MODE: Using full validation set ({len(val_loader.dataset)} samples)')
        
        # For PIMED: Extract feature shapes from a sample batch
        print('==> Extracting teacher feature shapes from sample batch...')
        sample_batch = next(iter(train_loader))
        if len(sample_batch) == 7:
            img, label, teacher_features, teacher_logits, case_ids, label_original, prostate_mask = sample_batch
        elif len(sample_batch) == 6:
            img, label, teacher_features, teacher_logits, case_ids, label_original = sample_batch
            prostate_mask = None
        else:
            # Backward compatibility
            img, label, teacher_features, teacher_logits, case_ids = sample_batch
            label_original = label
            prostate_mask = None
        
        # Build teacher_feature_shapes list
        teacher_feature_shapes = []
        for i in range(len(teacher_features)):  # For each teacher
            t_feat = teacher_features[i]  # Dict with 'layer_minus_1' (and optionally 'layer_minus_2')
            t_logits = teacher_logits[i]  # Tensor [B, num_cls, D, H, W]
            
            # Keep full tensor shapes (including batch dimension for consistency)
            shape_dict = {
                'layer_minus_1_shape': t_feat['layer_minus_1'].shape,  # [B, C, D, H, W] 
                'logits_shape': t_logits.shape  # [B, num_cls, D, H, W]
            }
            # Only add layer_minus_2 shape if it exists
            if 'layer_minus_2' in t_feat:
                shape_dict['layer_minus_2_shape'] = t_feat['layer_minus_2'].shape  # [B, C, D, H, W]
            teacher_feature_shapes.append(shape_dict)
        
        print(f'==> Found {len(teacher_feature_shapes)} teachers with shapes:')
        for i, shapes in enumerate(teacher_feature_shapes):
            neg2_info = f'feat_neg2={shapes["layer_minus_2_shape"]}, ' if 'layer_minus_2_shape' in shapes else ''
            print(f'    Teacher {i}: {neg2_info}'
                  f'feat_neg1={shapes["layer_minus_1_shape"]}, logits={shapes["logits_shape"]}')
        
        teacher_models = None  # No teacher models for PIMED
    else:
        raise ValueError(f"Unsupported dataset: {args.dataset}")
    
    ##### load student model #####
    # Modify architecture name if deep supervision is requested
    model_arch = args.arch
    if hasattr(args, 'use_deep_supervision') and args.use_deep_supervision:
        # Check if we need to switch to deep supervision variant
        if args.arch == 'fullres_3d_unet':
            model_arch = 'fullres_3d_unet_ds'
            print(f"===> Deep supervision enabled: Using {model_arch} instead of {args.arch}")
        elif args.arch == 'fullres_3d_unet_enc':
            # fullres_3d_unet_enc doesn't have a _ds variant yet, but supports deep_supervision parameter
            print(f"===> Deep supervision enabled for {args.arch} (uses deep_supervision parameter)")
        else:
            print(f"===> Warning: Deep supervision requested but not supported for {args.arch}")
    
    # Initialize model on CPU first - will be moved to GPU after get_feat_trans
    model = model_dict[model_arch](num_classes=args.n_cls)
    args.start_epoch = 0
    resume_checkpoint = None  # Store checkpoint for later use (agent, optimizer)
    
    if len(args.resume) != 0:
        if not os.path.isfile(args.resume):
            raise FileNotFoundError(f"No checkpoint found at '{args.resume}'")
        
        print(f'======> Loading checkpoint from {args.resume}')
        map_location = None if args.gpu is None else 'cuda:%d' % (args.gpu if args.multiprocessing_distributed else 0)
        resume_checkpoint = torch.load(args.resume, map_location=map_location)
        
        # Handle DataParallel checkpoint (keys start with "module.")
        state_dict = resume_checkpoint['model']
        if list(state_dict.keys())[0].startswith('module.'):
            # Remove "module." prefix
            from collections import OrderedDict
            new_state_dict = OrderedDict()
            for k, v in state_dict.items():
                name = k[7:]  # remove 'module.' prefix
                new_state_dict[name] = v
            state_dict = new_state_dict
        
        model.load_state_dict(state_dict)
        args.start_epoch = resume_checkpoint['epoch'] + 1  # Start from next epoch
        print(f'======> Resumed model from epoch {resume_checkpoint["epoch"]}')
        print(f'======> Will start training from epoch {args.start_epoch}')
        
    print('======> load student model finish')
    
    ##### Create agent and feat_trans #####
    if args.dataset == 'pimed':
        # Use pre-extracted feature shapes
        agent, args.t_feat_dims = get_agent(None, args, teacher_feature_shapes=teacher_feature_shapes)
    else:
        # Use teacher models
        agent, args.t_feat_dims = get_agent(teacher_models, args)
    print("===> get agent finish......")

    # Call get_feat_trans while model is on CPU, it will temporarily move to GPU and back
    feat_trans = get_feat_trans(model, args)
    # Parse distill_features to determine mode
    distill_features = getattr(args, 'distill_features', ['neg1'])
    distill_neg1 = 'neg1' in distill_features
    distill_neg2 = 'neg2' in distill_features
    distill_both = distill_neg1 and distill_neg2
    print(f"===> get feat_trans finish (distill_features={distill_features})......")
        
    if args.distributed:
        if torch.cuda.is_available():
            if args.gpu is not None:
                torch.cuda.set_device(args.gpu)
                # Move all modules from CPU to target GPU (single move each)
                model.cuda(args.gpu)
                agent.cuda(args.gpu)
                # Handle feat_trans as tuple if both neg1 and neg2 are enabled
                if distill_both and isinstance(feat_trans, tuple):
                    feat_trans = (feat_trans[0].cuda(args.gpu), feat_trans[1].cuda(args.gpu))
                else:
                    feat_trans.cuda(args.gpu)
                # In DDP, batch_size is already per-GPU, don't divide it
                # args.batch_size = int(args.batch_size / ngpus_per_node)
                args.workers = int((args.workers + ngpus_per_node - 1) / ngpus_per_node)
                # Wrap in DDP after moving to correct device
                # find_unused_parameters=True needed for fullres_3d_unet when is_feat=True (seg_layers unused)
                model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.gpu], find_unused_parameters=True)
                agent = torch.nn.parallel.DistributedDataParallel(agent, device_ids=[args.gpu])
                
            else:
                # This branch: DDP without explicit GPU ID (fallback to GPU 0)
                # WARNING: This may not work correctly with multi-GPU training!
                # Prefer using torchrun or setting args.gpu explicitly
                model.cuda()
                agent.cuda()
                # Handle feat_trans as tuple if both neg1 and neg2 are enabled
                if distill_both and isinstance(feat_trans, tuple):
                    feat_trans = (feat_trans[0].cuda(), feat_trans[1].cuda())
                else:
                    feat_trans.cuda()
                # find_unused_parameters=True needed for fullres_3d_unet when is_feat=True (seg_layers unused)
                model = torch.nn.parallel.DistributedDataParallel(model, find_unused_parameters=True)
                agent = torch.nn.parallel.DistributedDataParallel(agent)  
                
    elif args.gpu is not None and torch.cuda.is_available():
        torch.cuda.set_device(args.gpu)
        # Move all modules from CPU to target GPU (single move each)
        model.cuda(args.gpu)
        agent.cuda(args.gpu)
        # Handle feat_trans as tuple if both neg1 and neg2 are enabled
        if distill_both and isinstance(feat_trans, tuple):
            feat_trans = (feat_trans[0].cuda(args.gpu), feat_trans[1].cuda(args.gpu))
        else:
            feat_trans.cuda(args.gpu)
    else:
        # Explicitly specify all available GPUs for DataParallel
        num_gpus = torch.cuda.device_count()
        device_ids = list(range(num_gpus))
        print(f"===> Using DataParallel on {num_gpus} GPUs: {device_ids}")
        model = torch.nn.DataParallel(model, device_ids=device_ids).cuda()
    
    if torch.cuda.is_available():
        if args.gpu:
            device = torch.device('cuda:{}'.format(args.gpu))
        else:
            device = torch.device("cuda")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")
    
    ################### loss function and optimizer ###################
    
    # Create criterion based on dataset type
    criterion_list = nn.ModuleList([])
    
    if args.dataset == 'pimed':
        # Choose loss based on training mode
        if args.use_region_based_training:
            # TRUE nnUNet region-based loss with BCE and sigmoid
            # Select loss variant based on pred_type (2 or 3 channels)
            if hasattr(args, 'pred_type') and args.pred_type == 'binary':
                # Binary mode: 2 channels [prostate, cancer]
                if getattr(args, 'use_high_sensitivity', False):
                    # HIGH SENSITIVITY: Boost cancer detection (pos_weight=20, channel_weight=3)
                    criterion_ce = get_pimed_criterion('focal_binary_high_sensitivity').to(device)
                    print("===> Using Dice + Binary Focal Loss (HIGH SENSITIVITY)")
                    print("     pos_weight=20 (2x FN penalty), channel_weight=3, dice_weight=1.5")
                elif getattr(args, 'use_mild_focal', False):
                    # MILD: Better for larger models that may overfit with sharp focal
                    criterion_ce = get_pimed_criterion('focal_binary_mild').to(device)
                    print("===> Using Dice + Binary Focal Loss (MILD - balanced)")
                    print("     gamma=2.0 (standard), focal_weight=1.0 (balanced)")
                elif getattr(args, 'use_sharp_focal', False):
                    # SHARP: Better for reducing false positives on negative cases
                    criterion_ce = get_pimed_criterion('focal_binary_sharp').to(device)
                    print("===> Using Dice + Binary Focal Loss (SHARP - reduced FP)")
                    print("     gamma=1.5 (less down-weighting), focal_weight=2.0")
                else:
                    # Use standard FOCAL loss to focus on hard examples (cancer boundaries)
                    criterion_ce = get_pimed_criterion('focal_binary').to(device)
                    print("===> Using Dice + Binary Focal Loss (gamma=2)")
                print("     Binary mode: 2 channels [prostate, cancer (PCa+csPCa combined)]")
                print("     Focal loss down-weights easy examples by (1-p_t)^gamma")
            else:
                # 3-class mode: 3 channels [prostate, PCa, csPCa]
                criterion_ce = get_pimed_criterion('nnunet_pos_weight').to(device)
                print("===> Using nnUNet DC_and_BCE_loss with pos_weight [1, 5, 10] + channel_weights [1, 3, 5]")
                print("     3-class mode: 3 channels [prostate, PCa, csPCa]")
            print("     Region-based training: outputs are multi-hot regions")
            print("     Loss uses SIGMOID (not softmax)")
            print("     pos_weight heavily penalizes missing foreground pixels to prevent collapse")
        else:
            # Custom loss optimized for imbalanced data
            criterion_ce = get_pimed_criterion('foreground_focused').to(device)
            print("===> Using Custom Foreground-Focused Loss (70% Dice + 30% CE, class weights [10.0, 30.0])")
            print("     Per-class training: standard Dice/IoU metrics")
    else:
        # Use classification losses for CIFAR-100 and other datasets
        criterion_ce = nn.CrossEntropyLoss().to(device)
        print("===> Using CrossEntropy Loss for classification")
        
    # Initialize KD divergence loss (standard KL or contrastive variants)
    if args.kd_loss_type == 'kl':
        criterion_div = DistillKL(args.kd_T).to(device)
        print(f"===> Using standard KL divergence loss (T={args.kd_T})")
    else:
        criterion_div = get_contrastive_kd_loss(
            kd_type=args.kd_loss_type,
            temperature=args.kd_T,
            margin=args.contrastive_margin
        ).to(device)
        print(f"===> Using {args.kd_loss_type} contrastive KD loss (T={args.kd_T}, margin={args.contrastive_margin})")
    
    criterion_list.append(criterion_ce)
    criterion_list.append(criterion_div)

    trainable_list = nn.ModuleList([])
    trainable_list.append(model)
    # feat_trans can be a tuple (feat_trans_neg2, feat_trans_neg1) when using both layers
    if isinstance(feat_trans, tuple):
        for ft in feat_trans:
            trainable_list.append(ft)
    else:
        trainable_list.append(feat_trans)

    optimizer = optim.SGD(trainable_list.parameters(),
                        lr=0.1, momentum=0.9, weight_decay=args.weight_decay, nesterov=True)
    agent_optimizer = optim.SGD(agent.parameters(), lr=args.agent_lr, )
    
    # Resume optimizer and agent states if checkpoint was loaded
    if resume_checkpoint is not None:
        if 'optimizer' in resume_checkpoint:
            optimizer.load_state_dict(resume_checkpoint['optimizer'])
            print(f'======> Resumed optimizer state')
        if 'agent_optimizer' in resume_checkpoint:
            agent_optimizer.load_state_dict(resume_checkpoint['agent_optimizer'])
            print(f'======> Resumed agent optimizer state')
        if 'agent' in resume_checkpoint:
            # Load agent state dict (handle DDP wrapper if needed)
            agent_state = resume_checkpoint['agent']
            if args.distributed:
                # Agent is wrapped in DDP, load into module
                if list(agent_state.keys())[0].startswith('module.'):
                    agent.module.load_state_dict(agent_state)
                else:
                    # Add module prefix if checkpoint doesn't have it
                    from collections import OrderedDict
                    new_agent_state = OrderedDict()
                    for k, v in agent_state.items():
                        new_agent_state['module.' + k] = v
                    agent.load_state_dict(new_agent_state)
            else:
                # Remove module prefix if checkpoint has it but we're not using DDP
                if list(agent_state.keys())[0].startswith('module.'):
                    from collections import OrderedDict
                    new_agent_state = OrderedDict()
                    for k, v in agent_state.items():
                        name = k[7:]  # remove 'module.' prefix
                        new_agent_state[name] = v
                    agent.load_state_dict(new_agent_state)
                else:
                    agent.load_state_dict(agent_state)
            print(f'======> Resumed agent state')
        if 'best_acc' in resume_checkpoint:
            best_acc = resume_checkpoint.get('best_acc', 0.)
            print(f'======> Previous best accuracy: {best_acc:.4f}')
    
    # Initialize GradScaler for mixed precision training
    # Note: BF16 doesn't need gradient scaling, but we use it for FP16
    if args.fp16 and args.bf16:
        raise ValueError("Cannot use both --fp16 and --bf16. Choose one.")
    
    use_amp = args.fp16 or args.bf16
    amp_dtype = torch.bfloat16 if args.bf16 else torch.float16
    
    # GradScaler is only needed for FP16 (not for BF16)
    scaler = torch.amp.GradScaler('cuda', enabled=args.fp16)
    
    if args.bf16:
        print("===> Using BF16 (bfloat16) mixed precision training")
        print("     BF16 matches teacher feature precision - no precision loss")
        print("     BF16 uses same memory as FP16 but with better numerical stability")
    elif args.fp16:
        print("===> Using FP16 (float16) mixed precision training")
        print("     Note: Teacher features are BF16, some precision conversion occurs")
    
    # Note: Data loaders were already created earlier in this function
    
    ################### train model ###################
    best_acc = 0.  # best test accuracy
    
    # Early stopping tracking
    early_stopping_counter = 0
    best_cancer_dice = 0.0
    early_stopped = False
    
    # Test teacher models (skip for PIMED since no teacher models)
    if teacher_models is not None:
        t_results = []
        for t_model in teacher_models:
            acc, _ = test(0, t_model, device, val_loader, criterion_ce, args, verbose=False)
            t_results.append(round(acc, 2))
        args.logger.info('Teacher accruacy: '+ str(t_results))
    else:
        print('==> Skipping teacher testing (using pre-extracted features)')

    for epoch in range(args.start_epoch, args.epochs) :
        # Set epoch for sampler (required for FixedSubsetShufflingSampler to shuffle differently each epoch)
        if hasattr(train_loader, 'sampler') and train_loader.sampler is not None:
            if hasattr(train_loader.sampler, 'set_epoch'):
                train_loader.sampler.set_epoch(epoch)
        if hasattr(val_loader, 'sampler') and val_loader.sampler is not None:
            if hasattr(val_loader.sampler, 'set_epoch'):
                val_loader.sampler.set_epoch(epoch)
        
        # Train and get training metrics - pass visualizer and teacher names for PIMED
        train_metrics = train(train_loader, model, criterion_list, optimizer, epoch, device, args, agent, feat_trans, teacher_models, agent_optimizer, scaler, args.gradient_accumulation_steps,
                             visualizer=args.visualizer if hasattr(args, 'visualizer') else None,
                             teacher_names=args.teacher_name_list if args.dataset == 'pimed' else None,
                             amp_dtype=amp_dtype,
                             distill_neg1=distill_neg1,
                             distill_neg2=distill_neg2)
        
        # Validate and get validation metrics - pass visualizer and teacher names for PIMED
        val_results = test(epoch, model, device, val_loader, criterion_ce, args,
                          verbose=True,
                          visualizer=args.visualizer if hasattr(args, 'visualizer') else None,
                          teacher_names=args.teacher_name_list if args.dataset == 'pimed' else None,
                          amp_dtype=amp_dtype)
        
        # Handle different return formats based on dataset type
        if args.dataset == 'pimed':
            # For PIMED, val_results is a dict with 'dice', 'iou', 'loss'
            val_metric = val_results['dice']  # Use Dice as primary metric
            val_loss = val_results['loss']
            val_iou = val_results['iou']
        else:
            # For classification, val_results is (accuracy, loss)
            val_metric, val_loss = val_results
            val_iou = None  # Not applicable for classification
        
        # Episode-based RL agent update using validation feedback
        if args.use_episode_reward and 'episode_buffer' in train_metrics:
            from train_loops_new_policy import train_agent_episode
            
            # Synchronize all ranks before episode agent training
            if torch.distributed.is_initialized():
                torch.distributed.barrier()
            
            # Only rank 0 trains the agent (use .module to bypass DDP wrapper)
            if args.rank == 0:
                # Unwrap agent from DDP for single-rank training (prevents collective mismatch)
                agent_model = agent.module if hasattr(agent, 'module') else agent
                train_agent_episode(args, epoch, train_metrics['episode_buffer'], val_metric, 
                                   agent_model, agent_optimizer)
            
            # Synchronize after training and broadcast updated weights
            if torch.distributed.is_initialized():
                torch.distributed.barrier()
                # Broadcast from agent.module (the actual model, not DDP wrapper)
                agent_params = agent.module.parameters() if hasattr(agent, 'module') else agent.parameters()
                for param in agent_params:
                    torch.distributed.broadcast(param.data, src=0)
        
        # Get current learning rate
        current_lr = optimizer.param_groups[0]['lr']

        if args.rank <= 0 :
            # Log to TensorBoard
            args.writer.add_scalar('Loss/train_total', train_metrics['train_loss'], epoch)
            args.writer.add_scalar('Loss/train_cls', train_metrics['train_loss_cls'], epoch)
            args.writer.add_scalar('Loss/train_kd', train_metrics['train_loss_kd'], epoch)
            args.writer.add_scalar('Loss/train_feat', train_metrics['train_loss_feat'], epoch)
            args.writer.add_scalar('Loss/val', val_loss, epoch)
            args.writer.add_scalar('Metric/train', train_metrics['train_acc'], epoch)
            args.writer.add_scalar('Metric/val', val_metric, epoch)
            
            # Log Dice and IoU for segmentation tasks
            if args.dataset == 'pimed':
                args.writer.add_scalar('Segmentation/val_dice', val_results['dice'], epoch)
                args.writer.add_scalar('Segmentation/val_iou', val_results['iou'], epoch)
            
            args.writer.add_scalar('Learning_Rate', current_lr, epoch)
            
            # Log to CSV
            import csv
            with open(args.csv_path, 'a', newline='') as f:
                csv_writer = csv.writer(f)
                if args.dataset == 'pimed':
                    # Log Dice and IoU for segmentation
                    csv_writer.writerow([
                        epoch,
                        train_metrics['train_loss'],
                        train_metrics['train_loss_cls'],
                        train_metrics['train_loss_kd'],
                        train_metrics['train_loss_feat'],
                        train_metrics['train_acc'],
                        val_loss,
                        val_results['dice'],
                        val_results['iou'],
                        current_lr
                    ])
                else:
                    # Log accuracy for classification
                    csv_writer.writerow([
                        epoch,
                        train_metrics['train_loss'],
                        train_metrics['train_loss_cls'],
                        train_metrics['train_loss_kd'],
                        train_metrics['train_loss_feat'],
                        train_metrics['train_acc'],
                        val_loss,
                        val_metric,
                        current_lr
                    ])
            
            # Save checkpoint
            state = {
                    'epoch': epoch,
                    'arch': args.arch, 
                    'model': model.module.state_dict() if args.distributed else model.state_dict(),
                    'agent': agent.module.state_dict() if args.distributed else agent.state_dict(),
                    'optimizer': optimizer.state_dict(),
                    'agent_optimizer': agent_optimizer.state_dict(),
                    'acc': val_metric,
                    'best_acc': best_acc,
                    'val_loss': val_loss,
                    'train_metrics': train_metrics,
            }

            torch.save(state, os.path.join(args.checkpoint_dir, args.arch+'.pth.tar'))

            is_best = False
            if best_acc < val_metric:
                best_acc = val_metric
                is_best = True
            if is_best:
                shutil.copyfile(os.path.join(args.checkpoint_dir, args.arch + '.pth.tar'),
                                    os.path.join(args.checkpoint_dir, args.arch + '_best.pth.tar'))

            # Early stopping logic based on cancer dice
            if getattr(args, 'early_stopping', False) and args.dataset == 'pimed':
                # Get cancer dice (use dice_cancer for binary mode, dice_PCa for 3-class)
                current_cancer_dice = val_results.get('dice_cancer', val_results.get('dice_PCa', 0.0))
                
                if current_cancer_dice > best_cancer_dice + args.early_stopping_min_delta:
                    # Improvement found - reset counter
                    best_cancer_dice = current_cancer_dice
                    early_stopping_counter = 0
                    if args.rank <= 0:
                        args.logger.info(f'Early stopping: New best cancer dice {best_cancer_dice:.4f}')
                else:
                    # No improvement
                    early_stopping_counter += 1
                    if args.rank <= 0:
                        args.logger.info(f'Early stopping: No improvement for {early_stopping_counter}/{args.early_stopping_patience} epochs '
                                        f'(current: {current_cancer_dice:.4f}, best: {best_cancer_dice:.4f})')
                
                if early_stopping_counter >= args.early_stopping_patience:
                    if args.rank <= 0:
                        args.logger.info(f'\n{"="*60}')
                        args.logger.info(f'EARLY STOPPING TRIGGERED at epoch {epoch}')
                        args.logger.info(f'No improvement in cancer dice for {args.early_stopping_patience} epochs')
                        args.logger.info(f'Best cancer dice: {best_cancer_dice:.4f}')
                        args.logger.info(f'{"="*60}\n')
                    early_stopped = True
                    break  # Exit the training loop

    
    if args.rank <= 0 :
        args.logger.info('Evaluate the best model:')
        args.evaluate = True
        checkpoint = torch.load(os.path.join(args.checkpoint_dir, args.arch + '_best.pth.tar'),
                                    map_location=torch.device('cpu'))
        
        # Handle DDP checkpoint: adjust state_dict keys if needed
        state_dict = checkpoint['model']
        model_is_ddp = isinstance(model, torch.nn.parallel.DistributedDataParallel)
        checkpoint_is_ddp = list(state_dict.keys())[0].startswith('module.')
        
        if model_is_ddp and not checkpoint_is_ddp:
            # Model is DDP but checkpoint is not: add "module." prefix
            from collections import OrderedDict
            new_state_dict = OrderedDict()
            for k, v in state_dict.items():
                new_state_dict['module.' + k] = v
            state_dict = new_state_dict
        elif not model_is_ddp and checkpoint_is_ddp:
            # Model is not DDP but checkpoint is: remove "module." prefix
            from collections import OrderedDict
            new_state_dict = OrderedDict()
            for k, v in state_dict.items():
                name = k[7:]  # remove 'module.' prefix
                new_state_dict[name] = v
            state_dict = new_state_dict
        
        model.load_state_dict(state_dict)
        best_results = test(epoch, model, device, val_loader, criterion_ce, args,
                           verbose=True,
                           visualizer=None,  # Don't visualize during final evaluation
                           teacher_names=args.teacher_name_list if args.dataset == 'pimed' else None,
                           amp_dtype=amp_dtype)
        
        # Handle different return formats
        if args.dataset == 'pimed':
            best_metric = best_results['dice']
            best_loss = best_results['loss']
            best_iou = best_results['iou']
            args.logger.info('Test best Dice: {:.4f}'.format(best_metric))
            args.logger.info('Test best IoU: {:.4f}'.format(best_iou))
            args.logger.info('Test best loss: {:.4f}'.format(best_loss))
        else:
            best_metric, best_loss = best_results
            args.logger.info('Test best metric (Acc): {}'.format(best_metric))
            args.logger.info('Test best loss: {}'.format(best_loss))
        
        args.logger.info('load pre-trained weights from: {}'.format(os.path.join(args.checkpoint_dir,  args.arch + '_best.pth.tar')))
        
        # Close TensorBoard writer
        args.writer.close()
        args.logger.info('TensorBoard logging closed')

if __name__ == '__main__' :
     main()





    


