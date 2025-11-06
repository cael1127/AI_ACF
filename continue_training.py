"""
Continue training the GPT model from the latest checkpoint.
"""
import subprocess
import sys

if __name__ == '__main__':
    print("Continuing GPT training from step 2800...")
    print("This will train for additional steps to improve quality.")
    print()
    
    subprocess.run([
        sys.executable,
        'train_enhanced.py',
        '--resume',
        'checkpoints/best_enhanced.pt',
        '--max_steps',
        '5000',
        '--epochs',
        '10'
    ])


