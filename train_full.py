"""
Full-featured training script with proper learning dynamics.
"""
import os
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm
import tiktoken
import json
import time
from pathlib import Path

from model import GPT
from config import MODEL_CONFIG, TRAIN_CONFIG
from dataset import load_text_from_directory, TextDataset


def save_checkpoint(model, optimizer, step, loss, metrics, checkpoint_dir, is_best=False):
    """Save model checkpoint."""
    os.makedirs(checkpoint_dir, exist_ok=True)
    
    checkpoint = {
        'step': step,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'loss': loss,
        'metrics': metrics,
        'config': MODEL_CONFIG,
    }
    
    # Save checkpoint
    checkpoint_path = os.path.join(checkpoint_dir, f'checkpoint_step_{step}.pt')
    torch.save(checkpoint, checkpoint_path)
    
    # Save as latest
    latest_path = os.path.join(checkpoint_dir, 'latest_checkpoint.pt')
    torch.save(checkpoint, latest_path)
    
    # Save best if applicable
    if is_best:
        best_path = os.path.join(checkpoint_dir, 'best_checkpoint.pt')
        torch.save(checkpoint, best_path)
        print(f"Saved best model with loss: {loss:.4f}")


def load_checkpoint(checkpoint_path, model, optimizer=None):
    """Load checkpoint."""
    checkpoint = torch.load(checkpoint_path, map_location='cpu')
    model.load_state_dict(checkpoint['model_state_dict'])
    if optimizer and 'optimizer_state_dict' in checkpoint:
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    return checkpoint.get('step', 0), checkpoint.get('loss', float('inf'))


def compute_loss(model, input_ids, target_ids):
    """Compute cross-entropy loss."""
    logits = model(input_ids)
    B, T, V = logits.shape
    logits = logits.view(B * T, V)
    target_ids = target_ids.view(B * T)
    loss = F.cross_entropy(logits, target_ids)
    return loss


def evaluate(model, dataloader, device):
    """Evaluate model."""
    model.eval()
    total_loss = 0
    n_samples = 0
    
    with torch.no_grad():
        for input_ids, target_ids in dataloader:
            input_ids = input_ids.to(device)
            target_ids = target_ids.to(device)
            
            loss = compute_loss(model, input_ids, target_ids)
            
            batch_size = input_ids.size(0)
            total_loss += loss.item() * batch_size
            n_samples += batch_size
    
    model.train()
    return total_loss / n_samples


def get_lr_schedule(optimizer, max_steps, warmup_steps, min_lr=1e-6):
    """Create learning rate schedule."""
    def lr_lambda(step):
        if step < warmup_steps:
            return step / warmup_steps
        else:
            # Cosine decay
            progress = (step - warmup_steps) / (max_steps - warmup_steps)
            return 0.5 * (1 + math.cos(math.pi * progress)) * (1 - min_lr / TRAIN_CONFIG['learning_rate']) + min_lr / TRAIN_CONFIG['learning_rate']
    
    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)


def main():
    import argparse
    import math
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', type=str, required=True, help='Directory containing training texts')
    parser.add_argument('--resume', type=str, default=None, help='Path to checkpoint to resume from')
    parser.add_argument('--epochs', type=int, default=1, help='Number of epochs')
    parser.add_argument('--device', type=str, default='cpu', help='Device to use')
    args = parser.parse_args()
    
    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Load data
    print("\nLoading data...")
    if os.path.isfile(args.data_dir):
        # Single file
        with open(args.data_dir, 'r', encoding='utf-8') as f:
            texts = [f.read()]
    else:
        # Directory
        texts = load_text_from_directory(args.data_dir)
    print(f"Loaded {len(texts)} text file(s) with {sum(len(t) for t in texts)} total characters")
    
    # Tokenize and create dataset
    print("\nCreating dataset...")
    tokenizer = tiktoken.get_encoding("gpt2")
    dataset = TextDataset(
        texts, tokenizer, 
        context_length=MODEL_CONFIG['context_length'],
        stride=MODEL_CONFIG['context_length'] // 2  # 50% overlap
    )
    
    dataloader = DataLoader(
        dataset,
        batch_size=TRAIN_CONFIG['batch_size'],
        shuffle=True,
        num_workers=0,
        pin_memory=False
    )
    
    print(f"Dataset size: {len(dataset)} chunks")
    print(f"Batch size: {TRAIN_CONFIG['batch_size']}")
    print(f"Steps per epoch: {len(dataloader)}")
    
    # Create model
    print("\nInitializing model...")
    model = GPT(**MODEL_CONFIG).to(device)
    n_params = model.get_num_params()
    print(f"Model parameters: {n_params / 1e6:.2f}M")
    
    # Setup optimizer
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=TRAIN_CONFIG['learning_rate'],
        weight_decay=TRAIN_CONFIG['weight_decay'],
        betas=(TRAIN_CONFIG['beta1'], TRAIN_CONFIG['beta2'])
    )
    
    # Setup scheduler
    scheduler = get_lr_schedule(
        optimizer, 
        TRAIN_CONFIG['max_steps'], 
        TRAIN_CONFIG['warmup_steps']
    )
    
    # Resume from checkpoint
    start_step = 0
    best_loss = float('inf')
    global_step = 0
    
    if args.resume:
        print(f"\nResuming from checkpoint: {args.resume}")
        start_step, best_loss = load_checkpoint(args.resume, model, optimizer)
        global_step = start_step
        print(f"Resumed from step {start_step}")
    
    # Training
    print("\nStarting training...")
    model.train()
    
    total_loss = 0
    n_batches = 0
    save_every = TRAIN_CONFIG['save_every']
    
    for epoch in range(args.epochs):
        progress_bar = tqdm(dataloader, desc=f"Epoch {epoch+1}")
        
        for batch_idx, (input_ids, target_ids) in enumerate(progress_bar):
            input_ids = input_ids.to(device)
            target_ids = target_ids.to(device)
            
            # Forward
            optimizer.zero_grad()
            loss = compute_loss(model, input_ids, target_ids)
            
            # Skip if NaN
            if torch.isnan(loss):
                print(f"NaN detected at step {global_step}, skipping...")
                continue
            
            # Backward
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), TRAIN_CONFIG['grad_clip'])
            optimizer.step()
            scheduler.step()
            
            global_step += 1
            current_lr = scheduler.get_last_lr()[0]
            
            # Accumulate for metrics
            total_loss += loss.item()
            n_batches += 1
            avg_loss = total_loss / n_batches
            
            # Update progress
            progress_bar.set_postfix({
                'loss': f'{loss.item():.4f}',
                'avg': f'{avg_loss:.4f}',
                'lr': f'{current_lr:.2e}',
                'step': global_step
            })
            
            # Save checkpoint
            if global_step % save_every == 0:
                metrics = {
                    'loss': avg_loss,
                    'current_loss': loss.item(),
                    'learning_rate': current_lr
                }
                save_checkpoint(model, optimizer, global_step, avg_loss, metrics, TRAIN_CONFIG['checkpoint_dir'])
            
            # Stop if max steps reached
            if global_step >= TRAIN_CONFIG['max_steps']:
                print(f"\nReached max_steps: {TRAIN_CONFIG['max_steps']}")
                break
        
        print(f"\nEpoch {epoch+1} complete. Average loss: {avg_loss:.4f}")
    
    # Final save
    metrics = {
        'loss': avg_loss,
        'steps': global_step
    }
    save_checkpoint(model, optimizer, global_step, avg_loss, metrics, TRAIN_CONFIG['checkpoint_dir'], is_best=True)
    print(f"\nTraining complete! Final loss: {avg_loss:.4f}")
    print(f"Checkpoints saved in: {TRAIN_CONFIG['checkpoint_dir']}")


if __name__ == '__main__':
    main()

