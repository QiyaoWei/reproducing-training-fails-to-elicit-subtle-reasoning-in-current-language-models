#!/usr/bin/env python3
"""
Quick test to verify the DeepSeek model can be loaded from cache.
Run this inside the container to test before submitting training jobs.
"""

import os
import sys

# Print environment
print("=" * 50)
print("Environment Variables:")
print(f"HF_HOME: {os.getenv('HF_HOME', 'NOT SET')}")
print(f"TRANSFORMERS_CACHE: {os.getenv('TRANSFORMERS_CACHE', 'NOT SET')}")
print(f"HF_HUB_CACHE: {os.getenv('HF_HUB_CACHE', 'NOT SET')}")
print(f"HUGGINGFACE_HUB_CACHE: {os.getenv('HUGGINGFACE_HUB_CACHE', 'NOT SET')}")
print(f"PYTHONNOUSERSITE: {os.getenv('PYTHONNOUSERSITE', 'NOT SET')}")
print("=" * 50)

MODEL_NAME = "deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B"

try:
    print(f"\nAttempting to load tokenizer for: {MODEL_NAME}")
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(
        MODEL_NAME,
        trust_remote_code=True,
        local_files_only=False  # Allow cache lookups
    )
    print("✓ Tokenizer loaded successfully!")
    print(f"  Vocab size: {tokenizer.vocab_size}")

except Exception as e:
    print(f"✗ Failed to load tokenizer: {e}")
    sys.exit(1)

try:
    print(f"\nAttempting to load model config for: {MODEL_NAME}")
    from transformers import AutoConfig

    config = AutoConfig.from_pretrained(
        MODEL_NAME,
        trust_remote_code=True,
        local_files_only=False  # Allow cache lookups
    )
    print("✓ Model config loaded successfully!")
    print(f"  Model type: {config.model_type}")
    print(f"  Hidden size: {config.hidden_size}")

except Exception as e:
    print(f"✗ Failed to load model config: {e}")
    sys.exit(1)

print("\n" + "=" * 50)
print("SUCCESS: Model can be loaded from cache!")
print("=" * 50)
