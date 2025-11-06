"""
Training script for GPT model.
"""
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.cuda.amp import GradScaler, autocast
from torch.utils.tensorboard import SummaryWriter
import math
import json
from pathlib import Path
from tqdm import tqdm
from model import GPT
from config import MODEL_CONFIG, TRAIN_CONFIG
from dataset import load_text_from_directory, create_dataloader
import tiktoken


def compute_loss(model, input_ids, target_ids):
    """Compute cross-entropy loss."""
    logits = model(input_ids)
    B, T, V = logits.shape
    logits = logits.view(B * T, V)
    target_ids = target_ids.view(B * T)
    
    # Check for NaN in logits
    if torch.isnan(logits).any():
        print("Warning: NaN in logits!")
        logits = torch.nan_to_num(logits, nan=0.0)
    
    loss = F.cross_entropy(logits, target_ids)
    
    # Clip loss to prevent overflow
    if torch.isnan(loss):
        print("Warning: NaN loss detected!")
        loss = torch.tensor(0.0, requires_grad=True)
    
    return loss


def train_step(model, optimizer, input_ids, target_ids, scaler=None, use_fp16=False):
    """Single training step with optional mixed precision."""
    if use_fp16 and scaler is not None:
        with autocast():
            loss = compute_loss(model, input_ids, target_ids)
        scaler.scale(loss).backward()
    else:
        loss = compute_loss(model, input_ids, target_ids)
        loss.backward()
    return loss.item()


def validate(model, dataloader, device, use_fp16=False):
    """Validate the model."""
    model.eval()
    total_loss = 0
    n_samples = 0
    
    with torch.no_grad():
        for input_ids, target_ids in dataloader:
            input_ids = input_ids.to(device)
            target_ids = target_ids.to(device)
            
            with autocast(enabled=use_fp16):
                loss = compute_loss(model, input_ids, target_ids)
            
            total_loss += loss.item() * input_ids.size(0)
            n_samples += input_ids.size(0)
    
    model.train()
    return total_loss / n_samples


def save_checkpoint(model, optimizer, step, loss, checkpoint_dir, is_best=False):
    """Save model checkpoint."""
    os.makedirs(checkpoint_dir, exist_ok=True)
    
    checkpoint = {
        'step': step,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'loss': loss,
    }
    
    # Save regular checkpoint
    checkpoint_path = os.path.join(checkpoint_dir, f'checkpoint_step_{step}.pt')
    torch.save(checkpoint, checkpoint_path)
    
    # Save as latest
    latest_path = os.path.join(checkpoint_dir, 'latest_checkpoint.pt')
    torch.save(checkpoint, latest_path)
    
    # Save best if applicable
    if is_best:
        best_path = os.path.join(checkpoint_dir, 'best_checkpoint.pt')
        torch.save(checkpoint, best_path)


def load_checkpoint(checkpoint_path, model, optimizer=None):
    """Load model checkpoint."""
    checkpoint = torch.load(checkpoint_path, map_location='cpu')
    
    model.load_state_dict(checkpoint['model_state_dict'])
    
    if optimizer is not None and 'optimizer_state_dict' in checkpoint:
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    
    start_step = checkpoint.get('step', 0)
    best_loss = checkpoint.get('loss', float('inf'))
    
    return start_step, best_loss


def train(
    model,
    train_dataloader,
    val_dataloader=None,
    epochs=None,
    device='cuda',
    checkpoint_dir='checkpoints',
    resume_from=None,
    writer=None
):
    """Main training loop."""
    
    # Setup optimizer
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=TRAIN_CONFIG['learning_rate'],
        betas=(TRAIN_CONFIG['beta1'], TRAIN_CONFIG['beta2']),
        weight_decay=TRAIN_CONFIG['weight_decay']
    )
    
    # Setup mixed precision training
    scaler = GradScaler() if TRAIN_CONFIG['use_fp16'] else None
    use_fp16 = TRAIN_CONFIG['use_fp16']
    
    # Learning rate scheduler
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=TRAIN_CONFIG['max_steps'],
        eta_min=1e-6
    )
    
    # Load checkpoint if resuming
    start_step = 0
    best_val_loss = float('inf')
    global_step = 0
    
    if resume_from:
        print(f"Resuming from checkpoint: {resume_from}")
        start_step, best_val_loss = load_checkpoint(resume_from, model, optimizer)
        global_step = start_step
    
    # Training loop
    model.train()
    
    for epoch in range(epochs if epochs else 1):
        progress_bar = tqdm(train_dataloader, desc=f"Epoch {epoch+1}")
        
        for batch_idx, (input_ids, target_ids) in enumerate(progress_bar):
            input_ids = input_ids.to(device)
            target_ids = target_ids.to(device)
            
            # Forward pass
            optimizer.zero_grad()
            loss = compute_loss(model, input_ids, target_ids)
            loss.backward()
            
            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(model.parameters(), TRAIN_CONFIG['grad_clip'])
            optimizer.step()
            scheduler.step()
            global_step += 1
            
            # Update progress bar
            avg_loss = loss.item()
            progress_bar.set_postfix({
                'loss': f'{avg_loss:.4f}',
                'lr': f'{scheduler.get_last_lr()[0]:.2e}',
                'step': global_step
            })
            
            # Logging
            if global_step % 10 == 0 and writer is not None:
                writer.add_scalar('train/loss', avg_loss, global_step)
                writer.add_scalar('train/learning_rate', scheduler.get_last_lr()[0], global_step)
            
            # Save checkpoint
            if global_step % TRAIN_CONFIG['save_every'] == 0:
                save_checkpoint(model, optimizer, global_step, avg_loss, checkpoint_dir)
            
            # Validation
            if val_dataloader and global_step % TRAIN_CONFIG['eval_every'] == 0:
                val_loss = validate(model, val_dataloader, device, use_fp16)
                print(f"\nValidation loss: {val_loss:.4f}")
                
                if writer is not None:
                    writer.add_scalar('val/loss', val_loss, global_step)
                    writer.add_scalar('val/perplexity', math.exp(val_loss), global_step)
                
                # Save best model
                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    save_checkpoint(model, optimizer, global_step, val_loss, checkpoint_dir, is_best=True)
            
            # Stop if max steps reached
            if global_step >= TRAIN_CONFIG['max_steps']:
                print(f"\nReached max_steps: {TRAIN_CONFIG['max_steps']}")
                return


def main():
    """Main entry point for training."""
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', type=str, required=True, help='Directory containing training texts')
    parser.add_argument('--resume', type=str, default=None, help='Path to checkpoint to resume from')
    parser.add_argument('--epochs', type=int, default=1, help='Number of epochs')
    parser.add_argument('--device', type=str, default='cuda', help='Device to use (cuda/cpu)')
    args = parser.parse_args()
    
    # Setup device
    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Load data
    print("Loading data...")
    from dataset import load_text_from_directory
    texts = load_text_from_directory(args.data_dir)
    print(f"Loaded {len(texts)} text files")
    
    # Create dataloader
    train_dataloader, tokenizer = create_dataloader(
        texts=texts,
        context_length=MODEL_CONFIG['context_length'],
        batch_size=TRAIN_CONFIG['batch_size'],
        stride=MODEL_CONFIG['context_length']
    )
    
    # Create model
    print("\nInitializing model...")
    model = GPT(**MODEL_CONFIG).to(device)
    n_params = model.get_num_params()
    print(f"Model parameters: {n_params / 1e6:.2f}M")
    
    # Create writer
    writer = SummaryWriter(log_dir='runs')
    
    # Train
    print("\nStarting training...")
    train(
        model=model,
        train_dataloader=train_dataloader,
        epochs=args.epochs,
        device=device,
        checkpoint_dir=TRAIN_CONFIG['checkpoint_dir'],
        resume_from=args.resume,
        writer=writer
    )
    
    print("\nTraining complete!")
    writer.close()


if __name__ == '__main__':
    import torch.nn.functional as F
    main()


