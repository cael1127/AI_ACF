"""
Interactive chat interface for GPT model.
"""
import torch
import tiktoken
import sys
from typing import List, Tuple
from model import GPT
from config import MODEL_CONFIG, GEN_CONFIG, CHAT_CONFIG


class ChatInterface:
    """Interactive chat interface with conversation history."""
    
    def __init__(self, model: GPT, tokenizer, device='cuda'):
        self.model = model
        self.tokenizer = tokenizer
        self.device = device
        self.history: List[Tuple[str, str]] = []
        self.config = GEN_CONFIG.copy()
        
    def add_to_history(self, user_msg: str, assistant_msg: str):
        """Add message to conversation history."""
        self.history.append((user_msg, assistant_msg))
    
    def truncate_history(self, max_tokens: int = CHAT_CONFIG['max_history_tokens']):
        """Truncate history if it gets too long."""
        total_tokens = 0
        for user_msg, assistant_msg in reversed(self.history):
            tokens = len(self.tokenizer.encode(user_msg + assistant_msg))
            total_tokens += tokens
            if total_tokens > max_tokens:
                cutoff = len(self.history) - list(reversed(self.history)).index((user_msg, assistant_msg))
                self.history = self.history[cutoff:]
                break
    
    def format_prompt(self, user_input: str) -> str:
        """Format prompt with conversation history."""
        if not CHAT_CONFIG['keep_history'] or not self.history:
            prompt = f"{CHAT_CONFIG['system_prompt']}\n\n{CHAT_CONFIG['user_prefix']}{user_input}"
        else:
            # Build conversation context
            context = CHAT_CONFIG['system_prompt'] + "\n\n"
            for user_msg, assistant_msg in self.history[-5:]:  # Last 5 exchanges
                context += f"{CHAT_CONFIG['user_prefix']}{user_msg}\n"
                context += f"{CHAT_CONFIG['assistant_prefix']}{assistant_msg}\n\n"
            context += f"{CHAT_CONFIG['user_prefix']}{user_input}"
            prompt = context
        
        # Truncate if too long
        tokens = self.tokenizer.encode(prompt)
        if len(tokens) > MODEL_CONFIG['context_length'] // 2:
            # Keep system prompt and recent history
            cutoff = (MODEL_CONFIG['context_length'] // 2) - len(self.tokenizer.encode(CHAT_CONFIG['system_prompt']))
            prompt = self.tokenizer.decode(tokens[-cutoff:])
        
        return prompt
    
    def generate_response(self, user_input: str) -> str:
        """Generate response from user input."""
        # Format prompt with history
        prompt = self.format_prompt(user_input)
        
        # Encode
        tokens = self.tokenizer.encode(prompt)
        
        # Truncate to context length
        if len(tokens) > MODEL_CONFIG['context_length']:
            tokens = tokens[-MODEL_CONFIG['context_length']:]
        
        # Convert to tensor
        input_ids = torch.tensor([tokens], dtype=torch.long, device=self.device)
        
        # Generate
        with torch.no_grad():
            generated_ids = self.model.generate(
                input_ids,
                max_new_tokens=self.config['max_new_tokens'],
                temperature=self.config['temperature'],
                top_k=self.config['top_k'],
                top_p=self.config['top_p'],
                do_sample=self.config['do_sample']
            )
        
        # Decode only the new tokens
        new_tokens = generated_ids[0, input_ids.size(1):].cpu().tolist()
        response = self.tokenizer.decode(new_tokens)
        
        return response.strip()
    
    def chat(self):
        """Main chat loop."""
        print("=" * 60)
        print("GPT Chat Interface")
        print("=" * 60)
        print("Commands:")
        print("  /temp <value>  - Set temperature (0.1 - 2.0)")
        print("  /clear         - Clear conversation history")
        print("  /history       - Show conversation history")
        print("  /config        - Show current configuration")
        print("  /exit          - Exit chat")
        print("=" * 60)
        print()
        
        while True:
            try:
                # Get user input
                user_input = input(f"\n{CHAT_CONFIG['user_prefix']}")
                
                # Handle commands
                if user_input.startswith('/'):
                    self.handle_command(user_input)
                    continue
                
                # Skip empty input
                if not user_input.strip():
                    continue
                
                # Generate response
                print(f"\n{CHAT_CONFIG['assistant_prefix']}", end='', flush=True)
                response = self.generate_response(user_input)
                print(response)
                
                # Update history
                if CHAT_CONFIG['keep_history']:
                    self.add_to_history(user_input, response)
                    self.truncate_history()
            
            except KeyboardInterrupt:
                print("\n\nExiting...")
                break
            except Exception as e:
                print(f"\nError: {e}")
    
    def handle_command(self, cmd: str):
        """Handle chat commands."""
        cmd = cmd.strip()
        
        if cmd == '/clear':
            self.history.clear()
            print("Conversation history cleared.")
        
        elif cmd == '/history':
            print("\nConversation History:")
            for i, (user, assistant) in enumerate(self.history, 1):
                print(f"\n{i}. User: {user}")
                print(f"   Assistant: {assistant}")
        
        elif cmd == '/config':
            print("\nCurrent Configuration:")
            print(f"  Temperature: {self.config['temperature']}")
            print(f"  Top-k: {self.config['top_k']}")
            print(f"  Top-p: {self.config['top_p']}")
            print(f"  Max tokens: {self.config['max_new_tokens']}")
            print(f"  History enabled: {CHAT_CONFIG['keep_history']}")
        
        elif cmd.startswith('/temp'):
            try:
                temp = float(cmd.split()[1])
                if 0.1 <= temp <= 2.0:
                    self.config['temperature'] = temp
                    print(f"Temperature set to {temp}")
                else:
                    print("Temperature must be between 0.1 and 2.0")
            except:
                print("Usage: /temp <value>")
        
        elif cmd == '/exit':
            sys.exit(0)
        
        else:
            print(f"Unknown command: {cmd}")


def load_model(checkpoint_path: str, device='cuda'):
    """Load model from checkpoint."""
    print(f"Loading model from {checkpoint_path}...")
    
    # Initialize model
    model = GPT(**MODEL_CONFIG).to(device)
    
    # Load checkpoint
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    
    print("Model loaded successfully!")
    return model


def main():
    """Main entry point for chat."""
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint', type=str, required=True, help='Path to model checkpoint')
    parser.add_argument('--device', type=str, default='cuda', help='Device to use (cuda/cpu)')
    args = parser.parse_args()
    
    # Setup device
    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Load model
    model = load_model(args.checkpoint, device)
    model.eval()
    
    # Initialize tokenizer
    tokenizer = tiktoken.get_encoding("gpt2")
    
    # Create chat interface
    chat = ChatInterface(model, tokenizer, device)
    
    # Start chat
    chat.chat()


if __name__ == '__main__':
    main()


