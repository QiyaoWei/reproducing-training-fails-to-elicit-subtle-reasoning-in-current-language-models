# VERL Training on HPC Cluster - Complete Setup Guide

**Last Updated:** 2025-10-29
**Status:** ✅ All network errors fixed, ready to run

---

## Table of Contents
1. [Quick Start](#quick-start)
2. [What Was Fixed](#what-was-fixed)
3. [Environment Configuration](#environment-configuration)
4. [Running Training](#running-training)
5. [Troubleshooting](#troubleshooting)
6. [File Reference](#file-reference)

---

## Quick Start

### First Time Setup

```bash
# 1. Verify model is downloaded (should already be there)
ls -lh /scratch/gpfs/DANQIC/jz4391/.cache/huggingface/models--deepseek-ai--DeepSeek-R1-Distill-Qwen-1.5B/

# 2. If model is missing, download it (run on login node)
./download_model.sh

# 3. Submit training job
sbatch submit_grpo.slurm --dataset diamonds-seed0

# 4. Monitor job
squeue -u $USER
tail -f logs/slurm-<jobid>.err
```

### After Training (Optional)

Sync Weights & Biases logs from login node:
```bash
wandb login  # Only needed once
wandb sync /scratch/gpfs/DANQIC/jz4391/wandb
```

---

## What Was Fixed

### Timeline of Issues and Solutions

#### Issue 1: HuggingFace DNS Error ✅ FIXED
**Error:**
```
socket.gaierror: [Errno -3] Temporary failure in name resolution
Failed to resolve 'huggingface.co'
```

**Root Cause:** vLLM trying to access HuggingFace servers from compute nodes (no internet)

**Solution:**
- Added `export HF_HUB_OFFLINE=1` in `submit_grpo.slurm`
- Forces all HuggingFace libraries to work offline

#### Issue 2: Model Config Not Found ✅ FIXED
**Error:**
```
ValueError: Could not detect config format for no config file found.
```

**Root Cause:** vLLM needs direct filesystem path in offline mode, not HuggingFace model ID

**Solution:**
- Updated `run_grpo.sh` to use local path when offline:
  ```bash
  MODEL_PATH="/scratch/gpfs/DANQIC/jz4391/.cache/huggingface/models--deepseek-ai--DeepSeek-R1-Distill-Qwen-1.5B/snapshots/ad9f0ae0864d7fbcd1cd905e3c6c5b069cc8b562"
  ```

#### Issue 3: Weights & Biases Timeout ✅ FIXED
**Error:**
```
wandb: Network error (ConnectionError)
Failed to resolve 'o151352.ingest.sentry.io'
CommError: Run initialization has timed out after 90.0 sec.
```

**Root Cause:** W&B trying to sync to cloud servers from compute nodes (no internet)

**Solution:**
- Added `export WANDB_MODE=offline` in `submit_grpo.slurm`
- W&B logs saved locally, can sync manually later

#### Issue 4: Triton Import Error ✅ FIXED
**Error:**
```
ImportError: cannot import name 'AttrsDescriptor' from 'triton.compiler.compiler'
```

**Root Cause:** Triton in `~/.local/` conflicts with container's PyTorch

**Solution:**
- Added `export PYTHONNOUSERSITE=1` to prevent loading user packages

---

## Environment Configuration

### All Environment Variables

The following are automatically set by `submit_grpo.slurm`:

| Variable | Value | Purpose |
|----------|-------|---------|
| `HF_HOME` | `/scratch/gpfs/DANQIC/jz4391/.cache/huggingface` | Main HuggingFace cache |
| `TRANSFORMERS_CACHE` | `/scratch/gpfs/DANQIC/jz4391/.cache/huggingface` | Transformers library cache |
| `HF_HUB_CACHE` | `/scratch/gpfs/DANQIC/jz4391/.cache/huggingface/hub` | Model hub cache |
| `HF_HUB_OFFLINE` | `1` | **Force offline mode (no internet)** |
| `HF_HUB_DOWNLOAD_TIMEOUT` | `1` | Fail fast if internet attempted |
| `WANDB_MODE` | `offline` | **W&B offline mode** |
| `WANDB_DIR` | `/scratch/gpfs/DANQIC/jz4391/wandb` | W&B logs directory |
| `PYTHONNOUSERSITE` | `1` | Prevent package conflicts |
| `SINGULARITY_CACHEDIR` | `/scratch/gpfs/DANQIC/jz4391/.singularity_cache` | Container cache |

### Model Location

The model is cached at:
```
/scratch/gpfs/DANQIC/jz4391/.cache/huggingface/models--deepseek-ai--DeepSeek-R1-Distill-Qwen-1.5B/snapshots/ad9f0ae0864d7fbcd1cd905e3c6c5b069cc8b562/
```

This directory contains:
- `config.json` - Model configuration
- `*.safetensors` - Model weights
- `tokenizer.json` - Tokenizer files
- All other required files

### Container Configuration

Singularity container launch command:
```bash
singularity exec --nv \
    --bind $(pwd):$(pwd) \
    --bind /scratch/gpfs/DANQIC/jz4391/.cache:/scratch/gpfs/DANQIC/jz4391/.cache \
    --bind /scratch/gpfs/DANQIC/jz4391/wandb:/scratch/gpfs/DANQIC/jz4391/wandb \
    --pwd $(pwd) \
    --env HF_HOME=${HF_HOME} \
    --env HF_HUB_OFFLINE=${HF_HUB_OFFLINE} \
    --env WANDB_MODE=${WANDB_MODE} \
    --env WANDB_DIR=${WANDB_DIR} \
    --env PYTHONNOUSERSITE=${PYTHONNOUSERSITE} \
    verl_app-verl0.4-vllm0.8.5-mcore0.12.1.sif \
    ./run_grpo.sh "$@"
```

---

## Running Training

### Submit a Job

Basic usage:
```bash
sbatch submit_grpo.slurm --dataset diamonds-seed0
```

With custom parameters:
```bash
sbatch submit_grpo.slurm --dataset diamonds-seed0 --epochs 20 --lr 1e-5 --kl-coef 0.2
```

Available options (from `run_grpo.sh --help`):
- `--dataset NAME` - Dataset name (diamonds-seed0 to diamonds-seed7, function_correctness)
- `--epochs NUM` - Number of training epochs
- `--lr RATE` - Learning rate
- `--kl-coef COEF` - KL divergence coefficient
- `--k COEF` - Verbosity reward coefficient
- `--experiment-name NAME` - Custom experiment name
- `--resume-from-path PATH` - Resume from checkpoint

### Monitor Jobs

Check job status:
```bash
squeue -u $USER
```

View logs in real-time:
```bash
tail -f logs/slurm-<jobid>.out
tail -f logs/slurm-<jobid>.err
```

Check completed logs:
```bash
cat logs/slurm-<jobid>.out
cat logs/slurm-<jobid>.err
```

### Expected Output

On successful startup, you should see:
```
✓ Model found in cache
Using local model path (offline mode): /scratch/gpfs/DANQIC/jz4391/.cache/huggingface/...
WANDB_MODE: offline
Filtering prompts longer than 2048 tokens: 100%
Starting GRPO training...
```

No network errors should appear!

---

## Troubleshooting

### Model Not Found Error

**Error:** `ValueError: Invalid repository ID or local directory specified`

**Check:**
```bash
ls -lh /scratch/gpfs/DANQIC/jz4391/.cache/huggingface/models--deepseek-ai--DeepSeek-R1-Distill-Qwen-1.5B/
```

**Fix:**
```bash
./download_model.sh  # Run on login node
```

### Network Connection Errors

**Error:** `Failed to resolve 'huggingface.co'` or `Failed to resolve 'o151352.ingest.sentry.io'`

**Check:** Verify offline mode is enabled:
```bash
grep -E "HF_HUB_OFFLINE|WANDB_MODE" submit_grpo.slurm
```

Should show:
```
export HF_HUB_OFFLINE=1
export WANDB_MODE=offline
```

**Fix:** Already applied in latest version - resubmit job

### Import Errors (Triton/PyTorch)

**Error:** `ImportError: cannot import name 'AttrsDescriptor'`

**Check:**
```bash
grep PYTHONNOUSERSITE submit_grpo.slurm
```

Should show:
```
export PYTHONNOUSERSITE=1
```

**Fix:** Already applied - resubmit job

### Disk Space Issues

**Error:** `OSError: [Errno 28] No space left on device`

**Check disk usage:**
```bash
df -h /scratch/gpfs/DANQIC/jz4391
```

**Free up space:**
```bash
# Remove old checkpoints
rm -rf checkpoints/old_experiment_*

# Check largest directories
du -sh /scratch/gpfs/DANQIC/jz4391/* | sort -h
```

### Job Fails Immediately

**Check error logs:**
```bash
cat logs/slurm-<jobid>.err
```

**Common causes:**
- Missing data files → Check `./data/diamonds-seed0/` exists
- Wrong dataset name → Use `--dataset diamonds-seed0` (not `diamonds-seed1` unless data exists)
- Configuration errors → Check SLURM script settings

### W&B Authentication (Optional)

You do **NOT** need to login before running jobs in offline mode.

Only login when syncing logs **after** training:
```bash
# On login node
wandb login
wandb sync /scratch/gpfs/DANQIC/jz4391/wandb
```

---

## File Reference

### Script Files

| File | Purpose | When to Use |
|------|---------|-------------|
| `submit_grpo.slurm` | SLURM job submission script | Modified with all fixes |
| `run_grpo.sh` | Training script runner | Modified to use local model path |
| `submit_job.sh` | Wrapper for submitting jobs | Use this to submit jobs |
| `download_model.sh` | Download models from HuggingFace | Run once on login node |

### Utility Files

| File | Purpose |
|------|---------|
| `download_model.py` | Python helper for model download |
| `test_model_load.py` | Test if model loads correctly |

### Container

| File | Purpose |
|------|---------|
| `verl_app-verl0.4-vllm0.8.5-mcore0.12.1.sif` | Singularity container with all dependencies |

### Documentation (This File)

| File | Status |
|------|--------|
| `HPC_SETUP_README.md` | ✅ **READ THIS ONE** - Complete guide |
| ~~`FINAL_FIX_SUMMARY.md`~~ | Superseded by this file |
| ~~`FIX_NETWORK_ERROR.md`~~ | Superseded by this file |
| ~~`MODEL_DOWNLOAD_README.md`~~ | Superseded by this file |
| ~~`OFFLINE_MODE_FIX_SUMMARY.md`~~ | Superseded by this file |
| ~~`QUICK_FIX_REFERENCE.md`~~ | Superseded by this file |
| ~~`SETUP_SUMMARY.md`~~ | Superseded by this file |
| ~~`TROUBLESHOOTING.md`~~ | Superseded by this file |

---

## Key Takeaways

### For Running on HPC

1. ✅ **Pre-download models** on login nodes (have internet)
2. ✅ **Force offline mode** with `HF_HUB_OFFLINE=1` and `WANDB_MODE=offline`
3. ✅ **Use local paths** to cached models, not HuggingFace IDs
4. ✅ **Isolate packages** with `PYTHONNOUSERSITE=1` to avoid conflicts
5. ✅ **Store everything in /scratch** to avoid home quota issues

### Current Status

- **HuggingFace:** ✅ Working offline
- **Model loading:** ✅ Using local path
- **Weights & Biases:** ✅ Working offline
- **Training:** ✅ Ready to run
- **All network errors:** ✅ RESOLVED

---

## Getting Help

If you encounter issues not covered here:

1. Check the error logs: `cat logs/slurm-<jobid>.err`
2. Verify environment: `grep -E "OFFLINE|WANDB|PYTHONNOUSERSITE" submit_grpo.slurm`
3. Test model loading: `singularity exec ... python3 test_model_load.py`
4. Check disk space: `df -h /scratch/gpfs/DANQIC/jz4391`

---

**Ready to run!** 🚀

```bash
sbatch submit_grpo.slurm --dataset diamonds-seed0
```


# File Organization Summary

## What Changed

All documentation has been consolidated into **one comprehensive file**: `HPC_SETUP_README.md`

## Documentation Files

### ✅ ACTIVE - Read This One
- **`HPC_SETUP_README.md`** - Complete setup and troubleshooting guide
  - Contains everything from all other docs
  - Quick start guide
  - All fixes and solutions
  - Troubleshooting section
  - File reference

### 📦 OLD - Can Be Archived
These files contain information now in `HPC_SETUP_README.md`:

- `FINAL_FIX_SUMMARY.md` - Timeline of fixes
- `FIX_NETWORK_ERROR.md` - Network error fixes
- `MODEL_DOWNLOAD_README.md` - Model download instructions
- `OFFLINE_MODE_FIX_SUMMARY.md` - Offline mode setup
- `QUICK_FIX_REFERENCE.md` - Quick reference
- `SETUP_SUMMARY.md` - Initial setup summary
- `TROUBLESHOOTING.md` - Troubleshooting guide

## Script Files (Keep These)

### Training Scripts
- `submit_grpo.slurm` - SLURM job script (**modified with all fixes**)
- `run_grpo.sh` - Training runner (**modified for offline mode**)
- `submit_job.sh` - Job submission wrapper (unchanged)

### Utility Scripts
- `download_model.sh` - Download models on login node
- `download_model.py` - Python helper for downloads
- `test_model_load.py` - Test model loading
- `cleanup_old_docs.sh` - Archive old documentation (**NEW**)

### Container
- `verl_app-verl0.4-vllm0.8.5-mcore0.12.1.sif` - Singularity container

## How to Clean Up

To archive the old documentation files:

```bash
./cleanup_old_docs.sh
```

This will:
1. Move old `.md` files to `./archived_docs/`
2. Keep the essential `HPC_SETUP_README.md`
3. Preserve all script files

## Quick Reference

**To read:** `HPC_SETUP_README.md`

**To run training:**
```bash
sbatch submit_grpo.slurm --dataset diamonds-seed0
```

**To download model (if needed):**
```bash
./download_model.sh  # On login node
```

**To clean up docs:**
```bash
./cleanup_old_docs.sh
```

---

Everything you need is in `HPC_SETUP_README.md` 📖
