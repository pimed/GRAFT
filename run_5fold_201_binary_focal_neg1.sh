#!/bin/bash
# 5-Fold Ensemble Training Script — Dataset201
# BINARY mode with FOCAL LOSS, neg1 features
# Uses 5-fold nnUNet teachers from Dataset201_BxMR_withRegions_3SEQ
# Trains on all 3 cohorts: PICAI + UCLA + Stanford

# Activate virtual environment (edit this path to your venv/conda env)
# Example: source /path/to/your/venv/bin/activate
# Example: conda activate <env_name>
# source <PATH_TO_YOUR_VENV>/bin/activate

# Change to this script's directory so relative paths work
cd "$(dirname "$0")"
echo "Working directory: $(pwd)"

echo "=========================================="
echo "5-Fold Ensemble Training — Dataset201 (Binary + Focal Loss + neg1)"
echo "Starting at: $(date)"
echo "=========================================="

# Print environment info
echo "Python: $(which python)"
echo "PyTorch version:"
python -c "import torch; print(f'  {torch.__version__}')"
python -c "import torch; print(f'  CUDA available: {torch.cuda.is_available()}')"
python -c "import torch; print(f'  GPUs: {torch.cuda.device_count()}')"
echo "=========================================="

# ========================================
# GPU Configuration
# ========================================
export CUDA_VISIBLE_DEVICES=0,1
GPU_IDS="0,1"
NUM_GPUS=2

# ========================================
# Training Parameters
# ========================================
BATCH_SIZE=2
GRADIENT_ACCUMULATION=2
WORKERS=4
EPOCHS=200
ARCH="fullres_3d_unet"
DATASET="pimed"

# ========================================
# 5-Fold Teacher Configuration
# ========================================
TEACHERS="fold0 fold1 fold2 fold3 fold4"
# Dataset201 binary config
DATA_CONFIG="dataset/pimed_dataset_configs_local_region_5fold_binary_201.yaml"

# ========================================
# Output Configuration
# ========================================
CHECKPOINT_DIR="./checkpoint_5fold_201"
INNOVATION_SUFFIX="5fold_ensemble_binary_focal_201_neg1"

# Learning Rate Configuration
INIT_LR=0.001
LR_TYPE="cosine"
MILESTONES="50 75"

# Batches per epoch (for episode reward with memory constraints)
BATCHES_PER_EPOCH=9999

# Loss Weights
CE_WEIGHT=1
KD_WEIGHT=0      # No logit KD
FEAT_WEIGHT=1
KD_T=4

# RL/Reward Configuration
KD_LOSS_TYPE="kl"
AGENT_STEP=200
AGENT_WARMUP_EPOCHS=1  # Original behavior: epoch 0 uniform, epoch 1+ agent
REWARD_GAMMA=0.95
REWARD_ALPHA=0.7  # 70% Dice-based, 30% loss-based

# ========================================
# DDP Configuration
# ========================================
echo "==> Finding free port for distributed training..."
MASTER_PORT=""
for attempt in {1..20}; do
    CANDIDATE_PORT=$(shuf -i 20000-65000 -n 1)
    if ! ss -ltn | grep -q ":${CANDIDATE_PORT} "; then
        MASTER_PORT=$CANDIDATE_PORT
        echo "==> Found free port: $MASTER_PORT (attempt $attempt)"
        break
    fi
done

if [ -z "$MASTER_PORT" ]; then
    echo "WARNING: Could not find free port after 20 attempts, using default"
    MASTER_PORT=29600
fi

export MASTER_PORT

# ========================================
# Create checkpoint directory
# ========================================
mkdir -p ${CHECKPOINT_DIR}
mkdir -p logs

# Create detailed log file
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
DETAILED_LOG="logs/training_5fold_201_${ARCH}_binary_focal_neg1_${TIMESTAMP}.log"

# ========================================
# Launch Training
# ========================================
echo "=========================================="
echo "Starting 5-Fold Ensemble Training (Dataset201, Binary + Focal + neg1)..."
echo "GPUs: ${GPU_IDS}"
echo "Teachers: ${TEACHERS}"
echo "Checkpoint Dir: ${CHECKPOINT_DIR}"
echo "Data Config: ${DATA_CONFIG}"
echo "Master Port: $MASTER_PORT"
echo ""
echo "BINARY MODE FEATURES:"
echo "  - 2 output channels: [prostate, cancer]"
echo "  - Dice + Focal Loss (gamma=2)"
echo "  - neg1 feature distillation"
echo "  - All 3 cohorts: PICAI + UCLA + Stanford"
echo "  - Dataset201 5-fold nnUNet teachers"
echo "=========================================="

python -u -m torch.distributed.run \
    --nproc_per_node=${NUM_GPUS} \
    --nnodes=1 \
    --master_port=${MASTER_PORT} \
    train_student_rl_new_policy.py \
    --arch ${ARCH} \
    --dataset ${DATASET} \
    --data-configs ${DATA_CONFIG} \
    --checkpoint-dir ${CHECKPOINT_DIR} \
    --batch-size ${BATCH_SIZE} \
    --gradient-accumulation-steps ${GRADIENT_ACCUMULATION} \
    --teacher-name-list ${TEACHERS} \
    --workers ${WORKERS} \
    --epochs ${EPOCHS} \
    --kd-loss-type ${KD_LOSS_TYPE} \
    --use-episode-reward \
    --reward-gamma ${REWARD_GAMMA} \
    --reward-alpha ${REWARD_ALPHA} \
    --use-disagreement-reward \
    --agent-step ${AGENT_STEP} \
    --agent-warmup-epochs ${AGENT_WARMUP_EPOCHS} \
    --agent-update-interval 50 \
    --batches-per-epoch ${BATCHES_PER_EPOCH} \
    --bf16 \
    --init-lr ${INIT_LR} \
    --lr-type ${LR_TYPE} \
    --milestones ${MILESTONES} \
    --ce-weight ${CE_WEIGHT} \
    --kd-weight ${KD_WEIGHT} \
    --feat-weight ${FEAT_WEIGHT} \
    --innovation-suffix ${INNOVATION_SUFFIX} \
    --kd-T ${KD_T} \
    --print-freq 50 \
    --multiprocessing-distributed \
    --dist-backend nccl \
    --dist-url "env://" \
    --world-size 1 \
    --use-orthogonal \
    --use-deep-supervision \
    --use-region-based-training \
    --distill-features neg1 \
    2>&1 | tee ${DETAILED_LOG}

echo "=========================================="
echo "Training finished at: $(date)"
echo "Log saved to: ${DETAILED_LOG}"
echo "=========================================="
