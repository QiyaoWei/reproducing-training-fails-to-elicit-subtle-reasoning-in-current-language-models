# GPU Out of Memory (OOM) Fix

## Problem

Training started successfully (all network errors fixed!) but crashed with:

```
torch.OutOfMemoryError: CUDA out of memory. Tried to allocate 17.66 GiB.
GPU 0 has a total capacity of 79.18 GiB of which 4.97 GiB is free.
Of the allocated memory 100.77 GiB is allocated by PyTorch
```

## Root Cause

The batch sizes were too large for the available GPU memory:
- Training batch size: 1024
- Mini-batch size: 256
- Micro-batch size per GPU: 32

With gradient checkpointing enabled but no memory offloading, the model consumed over 100GB per GPU.

## Solution

Reduced batch sizes and enabled memory offloading:

### Changes Made

| Parameter | Old Value | New Value | Impact |
|-----------|-----------|-----------|--------|
| `train_batch_size` | 1024 | **512** | 2x less data per step |
| `ppo_mini_batch_size` | 256 | **128** | 2x fewer mini-batches |
| `ppo_micro_batch_size_per_gpu` | 32 | **16** | 2x smaller gradient accumulation |
| `fsdp_config.param_offload` | False | **True** | Offload model params to CPU |
| `fsdp_config.optimizer_offload` | False | **True** | Offload optimizer state to CPU |

### Memory Savings

**Before:**
- Model parameters on GPU
- Optimizer state on GPU
- Gradients for 32 samples per GPU
- Total: ~100GB+ per GPU → **OOM**

**After:**
- Model parameters on CPU (offloaded)
- Optimizer state on CPU (offloaded)
- Gradients for 16 samples per GPU
- Total: ~40-50GB per GPU → **Should fit**

## Trade-offs

### Slower Training
- Offloading adds CPU↔GPU transfer overhead
- Smaller batches = more gradient updates needed
- Expect ~30-50% slower per epoch

### Same Convergence
- Batch size reduction is gradual (512 still reasonable)
- Learning rate should still work well
- Model quality should be unaffected

## Resubmit

```bash
sbatch submit_grpo.slurm --dataset diamonds-seed0
```

## If Still OOM

If you still get OOM errors, try these additional reductions:

### Option 1: Further reduce batch sizes
```bash
# Edit run_grpo.sh:
data.train_batch_size=256
ppo_mini_batch_size=64
ppo_micro_batch_size_per_gpu=8
```

### Option 2: Reduce vLLM memory
```bash
# Edit run_grpo.sh:
actor_rollout_ref.rollout.gpu_memory_utilization=0.4  # Currently 0.6
```

### Option 3: Reduce max response length
```bash
# For diamonds dataset:
data.max_response_length=512  # Currently 1024
```

### Option 4: Use more GPUs
```bash
# Edit submit_grpo.slurm:
#SBATCH --gres=gpu:8  # Request 8 GPUs instead of 4

# Edit run_grpo.sh:
trainer.n_gpus_per_node=8
```

## Monitoring

After resubmitting, monitor GPU memory:

```bash
# In the job output, look for memory usage
tail -f logs/slurm-<jobid>.err | grep -i "memory\|oom"

# Or check with nvidia-smi (if you can ssh to the compute node)
watch -n 1 nvidia-smi
```

Expected memory usage should be 40-60GB per GPU (down from 100GB+).

## Status

✅ Network errors: FIXED
✅ Model loading: FIXED
✅ W&B offline: FIXED
🔧 Memory optimization: APPLIED

**Ready to resubmit!**
