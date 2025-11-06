"""
Train the GPT model on enhanced multi-domain dataset.
"""
import os
import argparse
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm
import tiktoken
import math

from model import GPT
from config import MODEL_CONFIG, TRAIN_CONFIG
from dataset import TextDataset


def compute_loss(model, input_ids, target_ids):
    """Compute cross-entropy loss."""
    logits = model(input_ids)
    B, T, V = logits.shape
    logits = logits.view(B * T, V)
    target_ids = target_ids.view(B * T)
    loss = F.cross_entropy(logits, target_ids)
    return loss


def load_file(filename):
    """Load text from file."""
    with open(filename, 'r', encoding='utf-8') as f:
        return f.read()


def main():
    parser = argparse.ArgumentParser(description="Train enhanced GPT on combined dataset")
    parser.add_argument('--device', type=str, default='cpu', help='cpu or cuda')
    parser.add_argument('--resume', type=str, default='', help='Path to checkpoint to resume from')
    parser.add_argument('--max_steps', type=int, default=None, help='Override max training steps')
    parser.add_argument('--epochs', type=int, default=3, help='Epochs to train if max_steps not hit')
    args = parser.parse_args()

    device = torch.device(args.device if (args.device == 'cuda' and torch.cuda.is_available()) else 'cpu')
    print(f"Using device: {device}")
    
    # Load enhanced data
    print("\nLoading enhanced dataset...")
    data_file = 'data/enhanced/combined.txt'
    
    if not os.path.exists(data_file):
        print(f"Creating combined dataset...")
        from prepare_enhanced_data import combine_text_files
        combine_text_files(data_file)
    
    text = load_file(data_file)
    print(f"Loaded {len(text)} characters")
    
    # Create dataset
    print("\nCreating dataset...")
    tokenizer = tiktoken.get_encoding("gpt2")
    texts = [text]  # Single large text
    
    dataset = TextDataset(
        texts, tokenizer,
        context_length=MODEL_CONFIG['context_length'],
        stride=MODEL_CONFIG['context_length'] // 4  # 25% overlap for better learning
    )
    
    dataloader = DataLoader(
        dataset,
        batch_size=TRAIN_CONFIG['batch_size'],
        shuffle=True,
        num_workers=0,
        pin_memory=False
    )
    
    print(f"Dataset size: {len(dataset)} chunks")
    
    # Create model
    print("\nInitializing model...")
    model = GPT(**MODEL_CONFIG).to(device)
    n_params = model.get_num_params()
    print(f"Model parameters: {n_params / 1e6:.2f}M")
    
    # Setup optimizer with better schedule
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=TRAIN_CONFIG['learning_rate'],
        weight_decay=TRAIN_CONFIG['weight_decay'],
        betas=(TRAIN_CONFIG['beta1'], TRAIN_CONFIG['beta2'])
    )
    
    # Cosine schedule with warmup
    def lr_schedule(step):
        warmup = TRAIN_CONFIG['warmup_steps']
        max_steps = TRAIN_CONFIG['max_steps']
        if step < warmup:
            return step / warmup
        progress = (step - warmup) / max(max_steps - warmup, 1)
        return 0.5 * (1 + math.cos(math.pi * progress))
    
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_schedule)

    # Optionally resume
    global_step = 0
    best_loss = float('inf')
    if args.resume and os.path.exists(args.resume):
        print(f"\nResuming from checkpoint: {args.resume}")
        ckpt = torch.load(args.resume, map_location=device)
        model.load_state_dict(ckpt['model_state_dict'])
        if 'optimizer_state_dict' in ckpt:
            optimizer.load_state_dict(ckpt['optimizer_state_dict'])
        if 'step' in ckpt:
            global_step = ckpt['step']
        if 'loss' in ckpt:
            best_loss = ckpt['loss']
        print(f"Resumed at step {global_step}, best_loss {best_loss}")
    
    # Training loop
    print("\nStarting enhanced training...")
    model.train()
    
    total_loss = 0
    n_batches = 0
    
    os.makedirs(TRAIN_CONFIG['checkpoint_dir'], exist_ok=True)
    
    # Determine max steps
    configured_max_steps = TRAIN_CONFIG['max_steps']
    if args.max_steps is not None:
        configured_max_steps = args.max_steps
        print(f"Overriding max_steps to {configured_max_steps}")

    epochs = args.epochs
    for epoch in range(epochs):
        progress_bar = tqdm(dataloader, desc=f"Epoch {epoch+1}")
        
        for batch_idx, (input_ids, target_ids) in enumerate(progress_bar):
            input_ids = input_ids.to(device)
            target_ids = target_ids.to(device)
            
            # Forward
            optimizer.zero_grad()
            loss = compute_loss(model, input_ids, target_ids)
            
            if torch.isnan(loss):
                continue
            
            # Backward
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), TRAIN_CONFIG['grad_clip'])
            optimizer.step()
            scheduler.step()
            
            global_step += 1
            current_lr = scheduler.get_last_lr()[0]
            
            # Metrics
            total_loss += loss.item()
            n_batches += 1
            avg_loss = total_loss / n_batches
            
            progress_bar.set_postfix({
                'loss': f'{loss.item():.4f}',
                'avg': f'{avg_loss:.4f}',
                'lr': f'{current_lr:.2e}',
                'step': global_step
            })
            
            # Save checkpoint
            if global_step % TRAIN_CONFIG['save_every'] == 0:
                checkpoint = {
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'step': global_step,
                    'loss': avg_loss,
                }
                torch.save(checkpoint, f"{TRAIN_CONFIG['checkpoint_dir']}/step_{global_step}.pt")
                print(f"\nSaved checkpoint at step {global_step}")
            
            if global_step >= configured_max_steps:
                break
        
        # Epoch complete
        if avg_loss < best_loss:
            best_loss = avg_loss
            checkpoint = {
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'step': global_step,
                'loss': avg_loss,
                'epoch': epoch,
            }
            torch.save(checkpoint, f"{TRAIN_CONFIG['checkpoint_dir']}/best_enhanced.pt")
            print(f"\nNew best model! Loss: {avg_loss:.4f}")
    
    print(f"\nEnhanced training complete! Final loss: {avg_loss:.4f}")
    print(f"Best checkpoint saved: {TRAIN_CONFIG['checkpoint_dir']}/best_enhanced.pt")


if __name__ == '__main__':
    main()

