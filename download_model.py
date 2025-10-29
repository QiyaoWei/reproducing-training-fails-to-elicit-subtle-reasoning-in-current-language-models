#!/usr/bin/env python3
"""
Download the DeepSeek model to local cache before running training.
Run this on a login node (which has internet access) before submitting SLURM jobs.
"""

import os
from transformers import AutoTokenizer, AutoModelForCausalLM

# Model to download
MODEL_NAME = "deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B"

# Set cache directory to scratch to avoid home quota issues
CACHE_DIR = "/scratch/gpfs/DANQIC/jz4391/.cache/huggingface"
os.makedirs(CACHE_DIR, exist_ok=True)
os.environ['HF_HOME'] = CACHE_DIR
os.environ['TRANSFORMERS_CACHE'] = CACHE_DIR

print(f"Downloading model: {MODEL_NAME}")
print(f"Cache directory: {CACHE_DIR}")

# Download tokenizer
print("\nDownloading tokenizer...")
tokenizer = AutoTokenizer.from_pretrained(
    MODEL_NAME,
    cache_dir=CACHE_DIR,
    trust_remote_code=True
)
print("Tokenizer downloaded successfully!")

# Download model
print("\nDownloading model (this may take several minutes)...")
model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME,
    cache_dir=CACHE_DIR,
    trust_remote_code=True,
    torch_dtype="auto",
    device_map="cpu"  # Don't load to GPU, just download
)
print("Model downloaded successfully!")

print(f"\nModel cached at: {CACHE_DIR}")
print("You can now run training jobs using this model.")
