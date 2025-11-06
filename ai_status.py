"""
Check the status of the AI training and show information.
"""
import torch
import os

print("="*60)
print("GPT Project Status")
print("="*60)

# Check available checkpoints
checkpoints_dir = 'checkpoints'
if os.path.exists(checkpoints_dir):
    pt_files = [f for f in os.listdir(checkpoints_dir) if f.endswith('.pt')]
    print(f"\nFound {len(pt_files)} checkpoint(s):")
    
    # Check best checkpoint
    best_paths = ['best_enhanced.pt', 'best_checkpoint.pt', 'latest_checkpoint.pt']
    for best_path in best_paths:
        full_path = os.path.join(checkpoints_dir, best_path)
        if os.path.exists(full_path):
            ckpt = torch.load(full_path, map_location='cpu')
            step = ckpt.get('step', 'unknown')
            loss = ckpt.get('loss', 'unknown')
            print(f"\n✓ {best_path}")
            print(f"  Step: {step}")
            print(f"  Loss: {loss:.4f}")
    
    # Check latest numbered checkpoint
    step_files = [f for f in pt_files if f.startswith('step_')]
    if step_files:
        latest_step_file = sorted(step_files, key=lambda x: int(x.split('_')[1].split('.')[0]), reverse=True)[0]
        latest_path = os.path.join(checkpoints_dir, latest_step_file)
        ckpt = torch.load(latest_path, map_location='cpu')
        step = ckpt.get('step', 'unknown')
        loss = ckpt.get('loss', 'unknown')
        print(f"\n✓ Latest: {latest_step_file}")
        print(f"  Step: {step}")
        print(f"  Loss: {loss:.4f}")

# Check training data
print("\n" + "="*60)
print("Training Data Status")
print("="*60)

data_dirs = ['data/enhanced', 'data/cleaned', 'data']
for data_dir in data_dirs:
    if os.path.exists(data_dir):
        txt_files = [f for f in os.listdir(data_dir) if f.endswith('.txt')]
        if txt_files:
            print(f"\n✓ {data_dir}: {len(txt_files)} file(s)")
            total_size = 0
            for txt_file in txt_files:
                txt_path = os.path.join(data_dir, txt_file)
                size = os.path.getsize(txt_path)
                total_size += size
                print(f"  - {txt_file}: {size:,} bytes")
            print(f"  Total: {total_size:,} bytes ({total_size/1024:.2f} KB)")

print("\n" + "="*60)
print("To continue the AI:")
print("="*60)
print("1. Chat with current model: py chat_now.py")
print("2. Continue training: py continue_training.py")
print("3. Test model: py test_enhanced.py")
print("="*60)


