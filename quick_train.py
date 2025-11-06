"""
Quick training script with fixed issues.
"""
import torch
import torch.nn.functional as F
from model import GPT
from config import MODEL_CONFIG
from dataset import load_text_from_directory, create_dataloader
import os
import tiktoken

device = torch.device('cpu')
print(f"Using device: {device}")

# Load data
print("Loading data...")
texts = load_text_from_directory('data/cleaned')
print(f"Loaded {len(texts)} files")

# Create dataloader
print("Creating dataset...")
tokenizer = tiktoken.get_encoding("gpt2")
dataloader, _ = create_dataloader(
    texts=texts,
    tokenizer_name="gpt2",
    context_length=MODEL_CONFIG['context_length'],
    batch_size=1,  # Very small batch
    stride=MODEL_CONFIG['context_length'],
    shuffle=True,
    num_workers=0
)

print(f"Dataset size: {len(dataloader.dataset)} chunks")

# Create model
print("Initializing model...")
model = GPT(**MODEL_CONFIG).to(device)
n_params = model.get_num_params()
print(f"Model has {n_params / 1e6:.2f}M parameters")

# Setup training
optimizer = torch.optim.AdamW(
    model.parameters(),
    lr=5e-4,
    weight_decay=0.1
)

# Training loop
model.train()
print("\nStarting training...")
for epoch in range(1):
    total_loss = 0
    n_batches = 0
    
    for batch_idx, (input_ids, target_ids) in enumerate(dataloader):
        input_ids = input_ids.to(device)
        target_ids = target_ids.to(device)
        
        # Forward
        optimizer.zero_grad()
        logits = model(input_ids)
        
        # Loss
        logits_flat = logits.view(-1, logits.size(-1))
        targets_flat = target_ids.view(-1)
        loss = F.cross_entropy(logits_flat, targets_flat)
        
        # Check for NaN
        if torch.isnan(loss):
            print(f"NaN detected at batch {batch_idx}, skipping...")
            continue
        
        # Backward
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        
        total_loss += loss.item()
        n_batches += 1
        
        if batch_idx % 10 == 0:
            print(f"Batch {batch_idx}, Loss: {loss.item():.4f}")
    
    print(f"\nEpoch complete. Average loss: {total_loss / n_batches:.4f}")

# Save
os.makedirs('checkpoints', exist_ok=True)
checkpoint = {
    'model_state_dict': model.state_dict(),
    'optimizer_state_dict': optimizer.state_dict(),
}
torch.save(checkpoint, 'checkpoints/quick_trained.pt')
print("\nModel saved to checkpoints/quick_trained.pt")

