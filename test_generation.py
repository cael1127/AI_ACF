"""
Quick test script for text generation.
"""
import torch
from model import GPT
from config import MODEL_CONFIG
import tiktoken


def test_generation():
    """Test text generation with a simple prompt."""
    
    # Initialize tokenizer
    tokenizer = tiktoken.get_encoding("gpt2")
    
    # Create model (not trained, just for testing architecture)
    print("Initializing model...")
    model = GPT(**MODEL_CONFIG)
    n_params = model.get_num_params()
    print(f"Model has {n_params / 1e6:.2f}M parameters")
    
    # Test prompt
    prompt = "The future of artificial intelligence"
    
    # Encode
    tokens = tokenizer.encode(prompt)
    print(f"Prompt: {prompt}")
    print(f"Tokens: {len(tokens)}")
    
    # Generate
    print("\nGenerating text...")
    input_ids = torch.tensor([tokens], dtype=torch.long)
    
    generated = model.generate(
        input_ids,
        max_new_tokens=50,
        temperature=0.8,
        do_sample=True
    )
    
    # Decode
    generated_tokens = generated[0].cpu().tolist()
    generated_text = tokenizer.decode(generated_tokens)
    
    print(f"\nGenerated: {generated_text}")
    print("\nNote: This is untrained, so output will be random.")


if __name__ == '__main__':
    test_generation()


