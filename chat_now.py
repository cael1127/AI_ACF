"""
Simple chat interface to interact with the AI right now.
"""
import torch
from model import GPT
from config import MODEL_CONFIG
import tiktoken

def chat():
    device = torch.device('cpu')
    tokenizer = tiktoken.get_encoding("gpt2")
    
    # Load model
    print("Loading GPT model...")
    model = GPT(**MODEL_CONFIG).to(device)
    
    checkpoint_path = 'checkpoints/best_enhanced.pt'
    print(f"Loading checkpoint: {checkpoint_path}")
    
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    
    print(f"Model loaded! Training loss: {checkpoint.get('loss', 'unknown')}")
    print("\n" + "="*60)
    print("🤖 GPT Chat Interface")
    print("="*60)
    print("Type your messages below.")
    print("Commands: 'exit' to quit, 'clear' to clear context")
    print("="*60)
    print()
    
    # Chat loop
    while True:
        try:
            user_input = input("👤 You: ").strip()
            
            if user_input.lower() == 'exit':
                print("\n👋 Goodbye!")
                break
            
            if user_input.lower() == 'clear':
                print("\n✨ Context cleared!\n")
                continue
            
            if not user_input:
                continue
            
            # Encode prompt
            prompt = f"User: {user_input}\nAssistant:"
            tokens = tokenizer.encode(prompt)
            
            # Truncate if too long
            max_input = MODEL_CONFIG['context_length'] - 100
            if len(tokens) > max_input:
                tokens = tokens[-max_input:]
            
            input_ids = torch.tensor([tokens], device=device)
            
            # Generate
            print("🤖 Assistant: ", end="", flush=True)
            
            with torch.no_grad():
                generated = model.generate(
                    input_ids,
                    max_new_tokens=100,
                    temperature=0.8,
                    do_sample=True
                )
            
            # Decode only new tokens
            new_tokens = generated[0, input_ids.size(1):].cpu().tolist()
            response = tokenizer.decode(new_tokens)
            
            print(response.strip())
            print()
            
        except KeyboardInterrupt:
            print("\n\n👋 Goodbye!")
            break
        except Exception as e:
            print(f"\n❌ Error: {e}\n")

if __name__ == '__main__':
    chat()


