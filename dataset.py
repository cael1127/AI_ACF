"""
Dataset handling for text data with GPT-2 tokenizer.
"""
import os
import torch
from torch.utils.data import Dataset, DataLoader
from typing import List, Tuple, Optional
import tiktoken


class TextDataset(Dataset):
    """Dataset for text data with tokenization and chunking."""
    
    def __init__(self, texts: List[str], tokenizer, context_length: int, stride: int):
        """
        Initialize dataset.
        
        Args:
            texts: List of text strings
            tokenizer: Tokenizer instance (e.g., tiktoken)
            context_length: Length of context window
            stride: Step size for sliding window
        """
        self.tokenizer = tokenizer
        self.context_length = context_length
        self.stride = stride
        
        # Tokenize all texts
        self.tokenized_texts = []
        for text in texts:
            tokens = tokenizer.encode(text)
            if len(tokens) > 0:
                self.tokenized_texts.append(tokens)
        
        # Create chunks
        self.chunks = []
        for tokens in self.tokenized_texts:
            # Use sliding window to create overlapping chunks
            for i in range(0, len(tokens) - context_length, stride):
                chunk = tokens[i:i + context_length + 1]  # +1 for target
                if len(chunk) == context_length + 1:
                    self.chunks.append(chunk)
    
    def __len__(self) -> int:
        return len(self.chunks)
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Get a sample from the dataset.
        
        Returns:
            input_ids: [context_length]
            target_ids: [context_length]
        """
        chunk = self.chunks[idx]
        input_ids = torch.tensor(chunk[:-1], dtype=torch.long)
        target_ids = torch.tensor(chunk[1:], dtype=torch.long)
        return input_ids, target_ids


def load_text_from_file(file_path: str) -> str:
    """Load text from a file."""
    with open(file_path, 'r', encoding='utf-8') as f:
        return f.read()


def load_text_from_directory(dir_path: str, extensions: Optional[List[str]] = None) -> List[str]:
    """
    Load text from all files in a directory.
    
    Args:
        dir_path: Path to directory
        extensions: List of file extensions to load (default: ['.txt'])
    
    Returns:
        List of text strings
    """
    if extensions is None:
        extensions = ['.txt']
    
    texts = []
    for root, dirs, files in os.walk(dir_path):
        for file in files:
            if any(file.endswith(ext) for ext in extensions):
                file_path = os.path.join(root, file)
                try:
                    text = load_text_from_file(file_path)
                    if len(text.strip()) > 0:
                        texts.append(text)
                except Exception as e:
                    print(f"Error loading {file_path}: {e}")
    
    return texts


def create_dataloader(
    texts: List[str],
    tokenizer_name: str = "gpt2",
    context_length: int = 1024,
    batch_size: int = 8,
    stride: Optional[int] = None,
    shuffle: bool = True,
    num_workers: int = 0
) -> DataLoader:
    """
    Create a DataLoader for training.
    
    Args:
        texts: List of text strings
        tokenizer_name: Name of tiktoken encoding
        context_length: Length of context window
        batch_size: Batch size
        stride: Step size for sliding window (default: context_length)
        shuffle: Whether to shuffle data
        num_workers: Number of worker processes
    
    Returns:
        DataLoader instance
    """
    if stride is None:
        stride = context_length
    
    # Initialize tokenizer
    tokenizer = tiktoken.get_encoding(tokenizer_name)
    
    # Create dataset
    dataset = TextDataset(texts, tokenizer, context_length, stride)
    
    # Create DataLoader
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=True
    )
    
    return dataloader, tokenizer


