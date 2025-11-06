"""
Monitor training progress by watching checkpoints.
"""
import os
import time
import torch

print("Watching training progress...")
print("Press Ctrl+C to stop\n")

# Get initial checkpoint
initial_checkpoint = 'checkpoints/best_enhanced.pt'
if os.path.exists(initial_checkpoint):
    ckpt = torch.load(initial_checkpoint, map_location='cpu')
    last_step = ckpt.get('step', 0)
    last_loss = ckpt.get('loss', 0)
    print(f"Initial: Step {last_step}, Loss {last_loss:.4f}\n")
else:
    last_step = 0
    last_loss = float('inf')

checkpoints_dir = 'checkpoints'

try:
    while True:
        time.sleep(5)  # Check every 5 seconds
        
        # Find latest checkpoint
        if os.path.exists(checkpoints_dir):
            step_files = [f for f in os.listdir(checkpoints_dir) 
                         if f.startswith('step_') and f.endswith('.pt')]
            
            if step_files:
                # Get latest step file
                latest_file = sorted(step_files, 
                                   key=lambda x: int(x.split('_')[1].split('.')[0]), 
                                   reverse=True)[0]
                latest_path = os.path.join(checkpoints_dir, latest_file)
                
                ckpt = torch.load(latest_path, map_location='cpu')
                current_step = ckpt.get('step', 0)
                current_loss = ckpt.get('loss', 0)
                
                # Only print if something changed
                if current_step > last_step:
                    loss_change = current_loss - last_loss
                    print(f"Step {current_step}: Loss {current_loss:.4f} "
                          f"({loss_change:+.4f})")
                    last_step = current_step
                    last_loss = current_loss
        
except KeyboardInterrupt:
    print("\nMonitoring stopped.")


