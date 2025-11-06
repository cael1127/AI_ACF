"""
Quick chat interface test.
"""
import torch
from model import GPT
from config import MODEL_CONFIG
import tiktoken

device = torch.device('cpu')
tokenizer = tiktoken.get_encoding("gpt2")

# Load model
print("Loading model...")
model = GPT(**MODEL_CONFIG)
checkpoint = torch.load('checkpoints/quick_trained.pt', map_location='cpu')
model.load_state_dict(checkpoint['model_state_dict'])
model.to(device)
model.eval()

print("\n=== GPT Chat Interface ===")
print("Type your messages. Type 'exit' to quit.\n")

prompt = input("You: ")
while prompt.lower() != 'exit':
    # Encode prompt
    tokens = tokenizer.encode(prompt)
    if len(tokens) > MODEL_CONFIG['context_length']:
        tokens = tokens[-MODEL_CONFIG['context_length']:]
    
    input_ids = torch.tensor([tokens], dtype=torch.long)
    
    # Generate
    with torch.no_grad():
        generated = model.generate(
            input_ids,
            max_new_tokens=50,
            temperature=0.8,
            do_sample=True
        )
    
    # Decode only new tokens
    new_tokens = generated[0, input_ids.size(1):].cpu().tolist()
    response = tokenizer.decode(new_tokens)
    
    print(f"GPT: {response}\n")
    
    prompt = input("You: ")

print("\nGoodbye!")

