"""
Full-featured chat interface with trained model.
"""
import torch
from model import GPT
from config import MODEL_CONFIG, GEN_CONFIG
import tiktoken
import sys

device = torch.device('cpu')
tokenizer = tiktoken.get_encoding("gpt2")

class ChatBot:
    def __init__(self, checkpoint_path, device='cpu'):
        print("Loading model...")
        self.model = GPT(**MODEL_CONFIG)
        checkpoint = torch.load(checkpoint_path, map_location=device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.model.to(device)
        self.model.eval()
        self.device = device
        self.tokenizer = tokenizer
        
    def generate(self, prompt, max_tokens=100, temperature=0.8):
        """Generate text from prompt."""
        tokens = self.tokenizer.encode(prompt)
        
        # Truncate to context length
        if len(tokens) > MODEL_CONFIG['context_length']:
            tokens = tokens[-MODEL_CONFIG['context_length']:]
        
        input_ids = torch.tensor([tokens], dtype=torch.long).to(self.device)
        
        with torch.no_grad():
            generated = self.model.generate(
                input_ids,
                max_new_tokens=max_tokens,
                temperature=temperature,
                do_sample=True
            )
        
        # Decode only new tokens
        new_tokens = generated[0, input_ids.size(1):].cpu().tolist()
        response = self.tokenizer.decode(new_tokens)
        
        return response.strip()
    
    def chat(self):
        """Interactive chat loop."""
        print("\n" + "="*60)
        print("GPT Chat Interface - Full Training")
        print("="*60)
        print("Model loaded successfully!")
        print(f"Parameters: {self.model.get_num_params() / 1e6:.2f}M")
        print("\nCommands:")
        print("  /temp <value> - Set temperature (0.1-2.0)")
        print("  /exit - Quit")
        print("="*60 + "\n")
        
        temperature = GEN_CONFIG['temperature']
        
        while True:
            try:
                user_input = input("You: ")
                
                if user_input.startswith('/'):
                    cmd = user_input.strip()
                    if cmd == '/exit':
                        break
                    elif cmd.startswith('/temp'):
                        try:
                            temp = float(cmd.split()[1])
                            if 0.1 <= temp <= 2.0:
                                temperature = temp
                                print(f"Temperature set to {temperature}")
                            else:
                                print("Temperature must be between 0.1 and 2.0")
                        except:
                            print("Usage: /temp <value>")
                    else:
                        print("Unknown command")
                    continue
                
                if not user_input.strip():
                    continue
                
                response = self.generate(user_input, max_tokens=100, temperature=temperature)
                print(f"\nGPT: {response}\n")
                
            except KeyboardInterrupt:
                print("\n\nExiting...")
                break
            except Exception as e:
                print(f"Error: {e}")
        
        print("\nGoodbye!")


if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint', type=str, default='checkpoints/best_checkpoint.pt',
                       help='Path to model checkpoint')
    args = parser.parse_args()
    
    chatbot = ChatBot(args.checkpoint, device='cpu')
    chatbot.chat()

