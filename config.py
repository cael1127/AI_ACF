"""
Configuration file for GPT model training and inference.
"""
from dataclasses import dataclass
from typing import Dict

# Model Architecture Configuration
MODEL_CONFIG = {
    'vocab_size': 50257,
    'context_length': 512,  # Context window size
    'emb_dim': 512,  # Embedding dimension (increased from 256)
    'n_heads': 8,   # Number of attention heads
    'n_layers': 6,  # Number of transformer layers (increased from 4)
    'dropout': 0.1,
}

# Training Configuration
TRAIN_CONFIG = {
    'batch_size': 4,  # Batch size
    'grad_accum_steps': 1,
    'learning_rate': 1e-4,  # Lower LR for stability
    'warmup_steps': 100,
    'max_steps': 2000,  # More steps for better learning
    'weight_decay': 0.1,
    'beta1': 0.9,
    'beta2': 0.99,  # Higher beta2
    'grad_clip': 1.0,
    'checkpoint_dir': 'checkpoints',
    'save_every': 200,  # Save every 200 steps
    'eval_every': 500,
    'eval_steps': 10,
    'use_fp16': False,  # CPU doesn't support FP16
}

# Generation Configuration
GEN_CONFIG = {
    'temperature': 0.8,
    'top_k': 50,
    'top_p': 0.9,
    'max_new_tokens': 200,
    'do_sample': True,
}

# Chat Configuration
CHAT_CONFIG = {
    'keep_history': True,
    'max_history_tokens': 512,
    'user_prefix': "User: ",
    'assistant_prefix': "Assistant: ",
    'system_prompt': "You are a helpful AI assistant.",
}


