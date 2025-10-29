#!/bin/bash
# Download the DeepSeek model to local cache
# Run this on a LOGIN NODE (which has internet access) before submitting SLURM jobs

set -e

echo "=========================================="
echo "Downloading DeepSeek-R1-Distill-Qwen-1.5B model"
echo "=========================================="
echo ""
echo "IMPORTANT: This script must be run on a LOGIN NODE with internet access,"
echo "NOT on a compute node."
echo ""

# Set cache directory
export HF_HOME=/scratch/gpfs/DANQIC/jz4391/.cache/huggingface
export TRANSFORMERS_CACHE=/scratch/gpfs/DANQIC/jz4391/.cache/huggingface
export HF_DATASETS_CACHE=/scratch/gpfs/DANQIC/jz4391/.cache/huggingface/datasets

# Prevent Python from loading packages from home directory that conflict with container packages
export PYTHONNOUSERSITE=1

mkdir -p ${HF_HOME}

echo "Cache directory: ${HF_HOME}"
echo ""

# Check if we can reach huggingface.co
echo "Checking internet connectivity..."
if ! curl -s --connect-timeout 5 https://huggingface.co > /dev/null; then
    echo "ERROR: Cannot reach huggingface.co"
    echo "Please ensure you are on a login node with internet access."
    exit 1
fi
echo "Internet connection OK"
echo ""

# Run inside the Singularity container to use the same environment
if [ -f "verl_app-verl0.4-vllm0.8.5-mcore0.12.1.sif" ]; then
    echo "Using Singularity container to download model..."
    singularity exec \
        --bind $(pwd):$(pwd) \
        --bind /scratch/gpfs/DANQIC/jz4391/.cache:/scratch/gpfs/DANQIC/jz4391/.cache \
        --pwd $(pwd) \
        --env HF_HOME=${HF_HOME} \
        --env TRANSFORMERS_CACHE=${TRANSFORMERS_CACHE} \
        --env HF_DATASETS_CACHE=${HF_DATASETS_CACHE} \
        --env PYTHONNOUSERSITE=${PYTHONNOUSERSITE} \
        verl_app-verl0.4-vllm0.8.5-mcore0.12.1.sif \
        python3 download_model.py
else
    echo "ERROR: verl_app-verl0.4-vllm0.8.5-mcore0.12.1.sif not found"
    echo "Please ensure the SIF file is in the current directory"
    exit 1
fi

echo ""
echo "=========================================="
echo "Model download complete!"
echo "=========================================="
echo ""
echo "You can now submit training jobs using ./submit_job.sh"
