#!/bin/bash
#SBATCH --job-name=grpo_training
#SBATCH --account=pli                # Use PLI account
#SBATCH --nodes=1                    # Number of nodes (matches trainer.nnodes=1)
#SBATCH --ntasks-per-node=1          # Tasks per node
#SBATCH --cpus-per-task=16           # CPUs per task
#SBATCH --gres=gpu:4                 # Request 4 GPUs (matches trainer.n_gpus_per_node=4)
#SBATCH --mem=200G                   # Memory per node
#SBATCH --time=1:00:00              # Time limit (1 hour)
#SBATCH --partition=pli              # Use PLI partition
#SBATCH --qos=pli-short              # QoS for short jobs
#SBATCH --output=logs/slurm-%j.out   # Standard output log
#SBATCH --error=logs/slurm-%j.err    # Standard error log
#SBATCH --mail-type=BEGIN,END,FAIL   # Email notifications
#SBATCH --mail-user=juventia2014@gmail.com

# Create logs directory if it doesn't exist
mkdir -p logs

# Set Singularity/Apptainer cache to scratch to avoid home quota issues
export SINGULARITY_CACHEDIR=/scratch/gpfs/DANQIC/jz4391/.singularity_cache
export APPTAINER_CACHEDIR=/scratch/gpfs/DANQIC/jz4391/.singularity_cache
mkdir -p $SINGULARITY_CACHEDIR

# Set HuggingFace cache to scratch to use pre-downloaded models
export HF_HOME=/scratch/gpfs/DANQIC/jz4391/.cache/huggingface
export TRANSFORMERS_CACHE=/scratch/gpfs/DANQIC/jz4391/.cache/huggingface
export HF_DATASETS_CACHE=/scratch/gpfs/DANQIC/jz4391/.cache/huggingface/datasets
export HF_HUB_CACHE=/scratch/gpfs/DANQIC/jz4391/.cache/huggingface/hub
export HUGGINGFACE_HUB_CACHE=/scratch/gpfs/DANQIC/jz4391/.cache/huggingface/hub
# Force offline mode - prevent any internet access attempts
export HF_HUB_OFFLINE=1
# Set timeout to fail fast if trying to access internet (which won't work on compute nodes)
export HF_HUB_DOWNLOAD_TIMEOUT=1
mkdir -p $HF_HOME

# Set Weights & Biases to offline mode (compute nodes don't have internet)
export WANDB_MODE=offline
export WANDB_DIR=/scratch/gpfs/DANQIC/jz4391/wandb
mkdir -p $WANDB_DIR

# Prevent Python from loading packages from home directory that conflict with container packages
export PYTHONNOUSERSITE=1

# Print job information
echo "=========================================="
echo "Job started at: $(date)"
echo "Running on node: $(hostname)"
echo "Job ID: $SLURM_JOB_ID"
echo "Working directory: $(pwd)"
echo "Number of GPUs: $SLURM_GPUS_ON_NODE"
echo "=========================================="

# Print GPU information
nvidia-smi

# Print HuggingFace cache information
echo "=========================================="
echo "HuggingFace Cache Configuration:"
echo "HF_HOME: $HF_HOME"
echo "TRANSFORMERS_CACHE: $TRANSFORMERS_CACHE"
echo "HF_HUB_CACHE: $HF_HUB_CACHE"
echo "HF_HUB_OFFLINE: $HF_HUB_OFFLINE"
echo "Checking if model is cached..."
if [ -d "$HF_HOME/models--deepseek-ai--DeepSeek-R1-Distill-Qwen-1.5B" ]; then
    echo "✓ Model found in cache"
    ls -lh "$HF_HOME/models--deepseek-ai--DeepSeek-R1-Distill-Qwen-1.5B/snapshots/"
else
    echo "✗ Model NOT found in cache!"
    echo "Please run ./download_model.sh first"
fi
echo ""
echo "Weights & Biases Configuration:"
echo "WANDB_MODE: $WANDB_MODE"
echo "WANDB_DIR: $WANDB_DIR"
echo "=========================================="

# Set the Singularity image (use local .sif file to avoid network issues)
SINGULARITY_IMAGE="verl_app-verl0.4-vllm0.8.5-mcore0.12.1.sif"

# Run the training script inside the Singularity container
# The container will have access to GPUs and the current directory
singularity exec --nv \
    --bind $(pwd):$(pwd) \
    --bind /scratch/gpfs/DANQIC/jz4391/.cache:/scratch/gpfs/DANQIC/jz4391/.cache \
    --bind /scratch/gpfs/DANQIC/jz4391/wandb:/scratch/gpfs/DANQIC/jz4391/wandb \
    --pwd $(pwd) \
    --env HF_HOME=${HF_HOME} \
    --env TRANSFORMERS_CACHE=${TRANSFORMERS_CACHE} \
    --env HF_DATASETS_CACHE=${HF_DATASETS_CACHE} \
    --env HF_HUB_CACHE=${HF_HUB_CACHE} \
    --env HUGGINGFACE_HUB_CACHE=${HUGGINGFACE_HUB_CACHE} \
    --env HF_HUB_OFFLINE=${HF_HUB_OFFLINE} \
    --env HF_HUB_DOWNLOAD_TIMEOUT=${HF_HUB_DOWNLOAD_TIMEOUT} \
    --env WANDB_MODE=${WANDB_MODE} \
    --env WANDB_DIR=${WANDB_DIR} \
    --env PYTHONNOUSERSITE=${PYTHONNOUSERSITE} \
    ${SINGULARITY_IMAGE} \
    ./run_grpo.sh "$@"

echo "=========================================="
echo "Job finished at: $(date)"
echo "=========================================="
