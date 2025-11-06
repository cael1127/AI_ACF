"""
Enhanced chat interface for the trained GPT model with conversation, math, and coding capabilities.
"""
import torch
import argparse
import os
from model import GPT
from config import MODEL_CONFIG, GEN_CONFIG, CHAT_CONFIG
import tiktoken


class EnhancedChat:
    def __init__(self, model_path, device='cpu'):
        self.device = torch.device(device)
        self.tokenizer = tiktoken.get_encoding("gpt2")
        
        # Load model
        print("Loading enhanced model...")
        self.model = GPT(**MODEL_CONFIG).to(self.device)
        
        checkpoint = torch.load(model_path, map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.model.eval()
        
        print(f"Model loaded! Training loss: {checkpoint.get('loss', 'unknown')}")
        
        # Chat state
        self.conversation_history = []
        self.temperature = GEN_CONFIG['temperature']
        self.max_tokens = GEN_CONFIG['max_new_tokens']
        
    def generate_response(self, prompt, temperature=None, max_tokens=None):
        """Generate response from the model."""
        if temperature is None:
            temperature = self.temperature
        if max_tokens is None:
            max_tokens = self.max_tokens
            
        # Prepare input
        context = self._build_context(prompt)
        input_ids = self.tokenizer.encode(context)
        
        if len(input_ids) > MODEL_CONFIG['context_length'] - max_tokens:
            # Truncate context if too long
            input_ids = input_ids[-(MODEL_CONFIG['context_length'] - max_tokens):]
        
        input_tensor = torch.tensor([input_ids], device=self.device)
        
        # Generate
        with torch.no_grad():
            generated = self._generate_tokens(input_tensor, temperature, max_tokens)
        
        # Decode response
        response_tokens = generated[0][len(input_ids):].tolist()
        response = self.tokenizer.decode(response_tokens)
        
        return response.strip()
    
    def _build_context(self, prompt):
        """Build conversation context."""
        if not self.conversation_history:
            return f"{CHAT_CONFIG['system_prompt']}\n\nUser: {prompt}\nAssistant:"
        
        # Add recent conversation history
        context = CHAT_CONFIG['system_prompt'] + "\n\n"
        for user_msg, assistant_msg in self.conversation_history[-3:]:  # Last 3 exchanges
            context += f"User: {user_msg}\nAssistant: {assistant_msg}\n\n"
        
        context += f"User: {prompt}\nAssistant:"
        return context
    
    def _generate_tokens(self, input_ids, temperature, max_tokens):
        """Generate tokens using the model."""
        generated = input_ids.clone()
        repetition_penalty = 1.1  # light penalty to reduce repeats
        stop_sequences = [
            "\nUser:",
            "\nAssistant:",
        ]
        
        for _ in range(max_tokens):
            # Get logits for next token
            with torch.no_grad():
                logits = self.model(generated)
                next_token_logits = logits[0, -1, :]

            # Repetition penalty on previously generated tokens
            if generated.numel() > input_ids.numel():
                prev_tokens = generated[0, input_ids.shape[1]:]
                next_token_logits[prev_tokens] /= repetition_penalty

            # Temperature scaling
            next_token_logits = next_token_logits / max(temperature, 1e-5)
            
            # Apply top-k and top-p filtering
            if GEN_CONFIG['top_k'] > 0:
                top_k = min(GEN_CONFIG['top_k'], next_token_logits.size(-1))
                top_k_logits, top_k_indices = torch.topk(next_token_logits, top_k)
                next_token_logits = torch.full_like(next_token_logits, float('-inf'))
                next_token_logits[top_k_indices] = top_k_logits
            
            if GEN_CONFIG['top_p'] < 1.0:
                sorted_logits, sorted_indices = torch.sort(next_token_logits, descending=True)
                cumulative_probs = torch.cumsum(torch.softmax(sorted_logits, dim=-1), dim=-1)
                sorted_indices_to_remove = cumulative_probs > GEN_CONFIG['top_p']
                sorted_indices_to_remove[1:] = sorted_indices_to_remove[:-1].clone()
                sorted_indices_to_remove[0] = 0
                
                indices_to_remove = sorted_indices[sorted_indices_to_remove]
                next_token_logits[indices_to_remove] = float('-inf')
            
            # Sample next token
            probs = torch.softmax(next_token_logits, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)
            
            # Stop if EOS token
            if next_token.item() == self.tokenizer.eot_token:
                break
            
            # Append to generated sequence
            generated = torch.cat([generated, next_token.unsqueeze(0)], dim=1)

            # Early stop on stop sequences
            decoded = self.tokenizer.decode(generated[0].tolist())
            for stop in stop_sequences:
                if stop in decoded[len(self.tokenizer.decode(input_ids[0].tolist())):]:
                    return generated
        
        return generated
    
    def chat(self):
        """Main chat loop."""
        print("\n" + "="*60)
        print("🤖 Enhanced GPT Chat - Math, Coding & Conversation")
        print("="*60)
        print("Commands:")
        print("  /temp <value>  - Set temperature (0.1-2.0)")
        print("  /clear         - Clear conversation history")
        print("  /history       - Show conversation history")
        print("  /config        - Show current settings")
        print("  /exit          - Exit chat")
        print("="*60)
        
        while True:
            try:
                user_input = input("\n👤 You: ").strip()
                
                if not user_input:
                    continue
                
                # Handle commands
                if user_input.startswith('/'):
                    self._handle_command(user_input)
                    continue
                
                # Generate response
                print("🤖 Assistant: ", end="", flush=True)
                response = self.generate_response(user_input)
                print(response)
                
                # Update history
                self.conversation_history.append((user_input, response))
                
                # Keep history manageable
                if len(self.conversation_history) > 10:
                    self.conversation_history = self.conversation_history[-10:]
                    
            except KeyboardInterrupt:
                print("\n\nGoodbye! 👋")
                break
            except Exception as e:
                print(f"\nError: {e}")
    
    def _handle_command(self, command):
        """Handle chat commands."""
        parts = command.split()
        cmd = parts[0].lower()
        
        if cmd == '/temp' and len(parts) > 1:
            try:
                temp = float(parts[1])
                if 0.1 <= temp <= 2.0:
                    self.temperature = temp
                    print(f"Temperature set to {temp}")
                else:
                    print("Temperature must be between 0.1 and 2.0")
            except ValueError:
                print("Invalid temperature value")
        
        elif cmd == '/clear':
            self.conversation_history = []
            print("Conversation history cleared")
        
        elif cmd == '/history':
            if not self.conversation_history:
                print("No conversation history")
            else:
                print("\nConversation History:")
                for i, (user, assistant) in enumerate(self.conversation_history, 1):
                    print(f"{i}. User: {user}")
                    print(f"   Assistant: {assistant}")
        
        elif cmd == '/config':
            print(f"\nCurrent Settings:")
            print(f"  Temperature: {self.temperature}")
            print(f"  Max tokens: {self.max_tokens}")
            print(f"  History length: {len(self.conversation_history)}")
        
        elif cmd == '/exit':
            print("Goodbye! 👋")
            exit()
        
        else:
            print("Unknown command. Type /help for available commands.")


def main():
    parser = argparse.ArgumentParser(description="Enhanced GPT Chat Interface")
    parser.add_argument('--checkpoint', type=str, default='checkpoints/best_enhanced.pt',
                        help='Path to model checkpoint')
    parser.add_argument('--device', type=str, default='cpu',
                        help='Device to use (cpu/cuda)')
    
    args = parser.parse_args()
    
    if not os.path.exists(args.checkpoint):
        print(f"Checkpoint not found: {args.checkpoint}")
        print("Available checkpoints:")
        if os.path.exists('checkpoints'):
            for f in os.listdir('checkpoints'):
                if f.endswith('.pt'):
                    print(f"  - checkpoints/{f}")
        return
    
    chat = EnhancedChat(args.checkpoint, args.device)
    chat.chat()


if __name__ == '__main__':
    main()
