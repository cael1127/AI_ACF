# GPT Model for Text Generation

A clean, implement-from-scratch GPT model optimized for text generation with web scraping capabilities.

## Features

- **Clean Architecture**: Modular GPT implementation from scratch
- **Modern Training**: Mixed precision, gradient accumulation, learning rate scheduling
- **Flexible Generation**: Multiple sampling strategies (greedy, top-k, top-p, temperature)
- **Interactive Chat**: Terminal-based chat interface with conversation history
- **Web Scraping**: Built-in utilities for collecting training data
- **Efficient**: Supports medium-scale models (100M-500M parameters)

## Project Structure

```
GPT_Project/
  ├── model.py          # GPT architecture implementation
  ├── train.py          # Training loop with checkpointing
  ├── chat.py           # Interactive chat interface
  ├── dataset.py        # Dataset handling for text data
  ├── config.py         # Model and training configuration
  ├── requirements.txt  # Python dependencies
  └── README.md         # This file
```

## Quick Start

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

### 2. Scrape Training Data

```bash
cd ../Web_Scraper

# Scrape Wikipedia articles
python scraper.py --topic "artificial_intelligence" --max-pages 20 --output ../AI/GPT_Project/data/scraped

# Or scrape specific URLs
python scraper.py --urls "https://example.com/article1" "https://example.com/article2" --output ../AI/GPT_Project/data/scraped

# Clean the scraped data
python clean_data.py --input ../AI/GPT_Project/data/scraped --output ../AI/GPT_Project/data/cleaned
```

### 3. Train the Model

```bash
cd ../AI/GPT_Project

python train.py --data_dir data/cleaned --epochs 1
```

### 4. Chat with the Model

```bash
python chat.py --checkpoint checkpoints/best_checkpoint.pt
```

## Usage

### Training

```bash
python train.py \
    --data_dir <path_to_text_files> \
    --epochs <num_epochs> \
    --resume <checkpoint_path>  # Optional: resume from checkpoint
```

**Configuration**:
- Model architecture and hyperparameters are in `config.py`
- Adjust `MODEL_CONFIG` for model size
- Adjust `TRAIN_CONFIG` for training parameters

### Chat Interface

```bash
python chat.py --checkpoint <path_to_checkpoint>
```

**Chat Commands**:
- `/temp <value>` - Set temperature (0.1-2.0)
- `/clear` - Clear conversation history
- `/history` - Show conversation history
- `/config` - Show current configuration
- `/exit` - Exit chat

### Generation Parameters

Edit `GEN_CONFIG` in `config.py`:
- `temperature`: Controls randomness (0.1 = deterministic, 2.0 = very random)
- `top_k`: Keep only top k tokens
- `top_p`: Nucleus sampling threshold
- `max_new_tokens`: Maximum tokens to generate

## Model Architecture

- **Type**: Decoder-only transformer (GPT-style)
- **Context Length**: 1024 tokens
- **Vocabulary**: GPT-2 (50,257 tokens)
- **Default Size**: ~125M parameters
- **Attention**: Multi-head causal self-attention
- **Feedforward**: 4x embedding dimension
- **Normalization**: LayerNorm (pre-norm)
- **Activation**: GELU

## Configuration

### Model Configuration (`config.py`)

```python
MODEL_CONFIG = {
    'vocab_size': 50257,
    'context_length': 1024,
    'emb_dim': 768,
    'n_heads': 12,
    'n_layers': 12,
    'dropout': 0.1,
}
```

### Training Configuration

```python
TRAIN_CONFIG = {
    'batch_size': 8,
    'grad_accum_steps': 4,
    'learning_rate': 3e-4,
    'warmup_steps': 1000,
    'max_steps': 100000,
    'use_fp16': True,  # Mixed precision training
}
```

## Web Scraping

Located in `../Web_Scraper/`:

### scraper.py
- Web scraping utilities using BeautifulSoup
- Automatic crawling with depth limits
- Respectful rate limiting
- Save scraped text to files

### clean_data.py
- Text cleaning and preprocessing
- Remove URLs, emails, special characters
- Deduplicate sentences
- Filter short paragraphs
- Merge multiple files

## Tips

1. **Dataset**: Start with at least 10MB of text data for meaningful training
2. **Memory**: Use gradient accumulation for effective large batch sizes
3. **Checkpoints**: Model saves checkpoints every 1000 steps
4. **Monitoring**: TensorBoard logs available in `runs/` directory
5. **GPU**: CUDA recommended for training (CPU is very slow)

## Troubleshooting

**Out of Memory**:
- Reduce `batch_size` in `TRAIN_CONFIG`
- Increase `grad_accum_steps` to maintain effective batch size
- Use `use_fp16: True` for mixed precision

**Slow Training**:
- Use GPU (CUDA)
- Reduce `context_length` (requires retraining)
- Reduce model size (`n_layers`, `emb_dim`)

**Poor Generation**:
- Train for more steps/epochs
- Increase dataset size
- Adjust generation parameters (temperature, top-k, top-p)

## Implementation Notes

- Built from scratch with PyTorch
- Uses GPT-2 tokenizer (tiktoken)
- Pre-norm transformer architecture
- Causal masking for autoregressive generation
- Proper gradient clipping and mixed precision support

## License

This project is for educational purposes.

