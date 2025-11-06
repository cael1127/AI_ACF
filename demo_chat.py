"""
Demo script showing the chat interface without a trained model.
"""
import torch
from model import GPT
from config import MODEL_CONFIG
import tiktoken

def demo_chat():
    print("=" * 60)
    print("GPT Chat Interface Demo")
    print("=" * 60)
    print("This demo shows the chat interface structure.")
    print("Note: The model is untrained, so responses will be random.")
    print("For real use, train the model first with your scraped data.")
    print("=" * 60)
    print()
    
    # Initialize model and tokenizer
    print("Loading model...")
    model = GPT(**MODEL_CONFIG)
    tokenizer = tiktoken.get_encoding("gpt2")
    print(f"Model loaded with {model.get_num_params() / 1e6:.2f}M parameters")
    print()
    
    # Simulate a chat
    history = []
    print("Commands: /temp, /clear, /history, /config, /exit")
    print()
    
    print(f"User: Hello! Can you tell me about artificial intelligence?")
    user_input = "Hello! Can you tell me about artificial intelligence?"
    
    # Generate response
    prompt = f"User: {user_input}"
    tokens = tokenizer.encode(prompt)
    input_ids = torch.tensor([tokens[-MODEL_CONFIG['context_length']:] if len(tokens) > MODEL_CONFIG['context_length'] else tokens], dtype=torch.long)
    
    with torch.no_grad():
        generated = model.generate(input_ids, max_new_tokens=50, temperature=0.8, do_sample=True)
    
    response_tokens = generated[0, input_ids.size(1):].cpu().tolist()
    response = tokenizer.decode(response_tokens)
    
    print(f"Assistant: {response}")
    print()
    
    print("=" * 60)
    print("To use the full chat interface:")
    print("1. Scrape training data: python ../Web_Scraper/scraper.py ...")
    print("2. Train model: python train.py --data_dir <cleaned_data>")
    print("3. Chat: python chat.py --checkpoint checkpoints/best_checkpoint.pt")
    print("=" * 60)

if __name__ == '__main__':
    demo_chat()

