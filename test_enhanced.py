"""
Test the enhanced model with specific prompts for math, coding, and conversation.
"""
import torch
from model import GPT
from config import MODEL_CONFIG, GEN_CONFIG
import tiktoken


def test_model(checkpoint_path='checkpoints/best_enhanced.pt'):
    """Test the model with various prompts."""
    device = torch.device('cpu')
    tokenizer = tiktoken.get_encoding("gpt2")
    
    # Load model
    print("Loading model...")
    model = GPT(**MODEL_CONFIG).to(device)
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    
    print(f"Model loaded! Loss: {checkpoint.get('loss', 'unknown')}")
    
    # Test prompts
    test_prompts = [
        # Math
        "What is 5 plus 3?",
        "Solve: 2x + 5 = 15",
        "What is the square of 9?",
        
        # Coding
        "How do you define a function in Python?",
        "What is a variable in programming?",
        "How do you create a list in Python?",
        
        # Conversation
        "Hello, how are you?",
        "What is machine learning?",
        "Tell me about artificial intelligence.",
    ]
    
    print("\n" + "="*60)
    print("TESTING ENHANCED MODEL")
    print("="*60)
    
    for i, prompt in enumerate(test_prompts, 1):
        print(f"\n{i}. Prompt: {prompt}")
        print("-" * 40)
        
        # Generate response
        response = generate_response(model, tokenizer, prompt, device)
        print(f"Response: {response}")
        print()


def generate_response(model, tokenizer, prompt, device, max_tokens=50, temperature=0.8):
    """Generate a response to a prompt."""
    # Encode prompt
    input_ids = tokenizer.encode(prompt)
    
    # Truncate if too long
    max_input = MODEL_CONFIG['context_length'] - max_tokens
    if len(input_ids) > max_input:
        input_ids = input_ids[-max_input:]
    
    input_tensor = torch.tensor([input_ids], device=device)
    
    # Generate tokens
    generated = input_tensor.clone()
    
    with torch.no_grad():
        for _ in range(max_tokens):
            # Get logits
            logits = model(generated)
            next_token_logits = logits[0, -1, :] / temperature
            
            # Sample next token
            probs = torch.softmax(next_token_logits, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)
            
            # Stop if EOS
            if next_token.item() == tokenizer.eot_token:
                break
            
            # Append
            generated = torch.cat([generated, next_token.unsqueeze(0)], dim=1)
    
    # Decode response
    response_tokens = generated[0][len(input_ids):].tolist()
    response = tokenizer.decode(response_tokens)
    
    return response.strip()


if __name__ == '__main__':
    test_model()
