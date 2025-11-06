"""
GPT Model Architecture - Decoder-Only Transformer
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import Optional, Tuple


class CausalSelfAttention(nn.Module):
    """Multi-head causal self-attention mechanism."""
    
    def __init__(self, emb_dim: int, n_heads: int, dropout: float = 0.1):
        super().__init__()
        assert emb_dim % n_heads == 0, "emb_dim must be divisible by n_heads"
        
        self.emb_dim = emb_dim
        self.n_heads = n_heads
        self.head_dim = emb_dim // n_heads
        self.dropout = dropout
        
        # Linear projections for Q, K, V
        self.qkv = nn.Linear(emb_dim, 3 * emb_dim, bias=False)
        self.out_proj = nn.Linear(emb_dim, emb_dim)
        self.dropout_layer = nn.Dropout(dropout)
        
        # Causal mask to prevent attending to future tokens
        self.register_buffer("causal_mask", None)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch_size, seq_len, _ = x.shape
        
        # Compute Q, K, V
        qkv = self.qkv(x)
        q, k, v = qkv.chunk(3, dim=-1)
        
        # Reshape for multi-head attention
        q = q.view(batch_size, seq_len, self.n_heads, self.head_dim).transpose(1, 2)
        k = k.view(batch_size, seq_len, self.n_heads, self.head_dim).transpose(1, 2)
        v = v.view(batch_size, seq_len, self.n_heads, self.head_dim).transpose(1, 2)
        
        # Compute attention scores with causal masking
        scale = 1.0 / math.sqrt(self.head_dim)
        attn_scores = torch.matmul(q, k.transpose(-2, -1)) * scale
        
        # Apply causal mask
        causal_mask = torch.triu(torch.ones(seq_len, seq_len, device=x.device), diagonal=1).bool()
        attn_scores = attn_scores.masked_fill(causal_mask, float('-inf'))
        
        attn_probs = F.softmax(attn_scores, dim=-1)
        attn_probs = self.dropout_layer(attn_probs)
        
        # Apply attention to values
        attn_output = torch.matmul(attn_probs, v)
        
        # Concatenate heads
        attn_output = attn_output.transpose(1, 2).contiguous()
        attn_output = attn_output.view(batch_size, seq_len, self.emb_dim)
        
        # Final projection
        output = self.out_proj(attn_output)
        return output


class FeedForward(nn.Module):
    """Position-wise feedforward network."""
    
    def __init__(self, emb_dim: int, dropout: float = 0.1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(emb_dim, 4 * emb_dim),
            nn.GELU(),
            nn.Linear(4 * emb_dim, emb_dim),
            nn.Dropout(dropout)
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class TransformerBlock(nn.Module):
    """Single transformer decoder block with pre-norm architecture."""
    
    def __init__(self, emb_dim: int, n_heads: int, dropout: float = 0.1):
        super().__init__()
        self.attn = CausalSelfAttention(emb_dim, n_heads, dropout)
        self.ff = FeedForward(emb_dim, dropout)
        self.norm1 = nn.LayerNorm(emb_dim)
        self.norm2 = nn.LayerNorm(emb_dim)
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Pre-norm architecture with residual connections
        x = x + self.dropout(self.attn(self.norm1(x)))
        x = x + self.dropout(self.ff(self.norm2(x)))
        return x


class GPT(nn.Module):
    """GPT Model - Decoder-only transformer for text generation."""
    
    def __init__(
        self,
        vocab_size: int,
        context_length: int,
        emb_dim: int = 768,
        n_heads: int = 12,
        n_layers: int = 12,
        dropout: float = 0.1
    ):
        super().__init__()
        
        self.vocab_size = vocab_size
        self.context_length = context_length
        self.emb_dim = emb_dim
        
        # Token embeddings with smaller initialization
        self.token_embedding = nn.Embedding(vocab_size, emb_dim)
        nn.init.normal_(self.token_embedding.weight, std=0.02)
        
        # Positional embeddings
        self.pos_embedding = nn.Embedding(context_length, emb_dim)
        nn.init.normal_(self.pos_embedding.weight, std=0.02)
        
        # Dropout
        self.dropout = nn.Dropout(dropout)
        
        # Transformer blocks
        self.blocks = nn.Sequential(
            *[TransformerBlock(emb_dim, n_heads, dropout) for _ in range(n_layers)]
        )
        
        # Layer norm before output
        self.final_norm = nn.LayerNorm(emb_dim)
        
        # Output projection
        self.lm_head = nn.Linear(emb_dim, vocab_size, bias=False)
        nn.init.normal_(self.lm_head.weight, std=0.02)
        
        # Tie weights: token embedding and output projection
        self.lm_head.weight = self.token_embedding.weight
        
    def forward(self, idx: torch.Tensor) -> torch.Tensor:
        """
        Forward pass for training.
        
        Args:
            idx: Token indices [batch_size, seq_len]
        
        Returns:
            logits: [batch_size, seq_len, vocab_size]
        """
        batch_size, seq_len = idx.shape
        device = idx.device
        
        # Create positional indices
        pos = torch.arange(0, seq_len, dtype=torch.long, device=device).unsqueeze(0)
        
        # Get embeddings
        tok_emb = self.token_embedding(idx)
        pos_emb = self.pos_embedding(pos)
        
        # Combine and apply dropout
        x = self.dropout(tok_emb + pos_emb)
        
        # Pass through transformer blocks
        x = self.blocks(x)
        
        # Final layer norm
        x = self.final_norm(x)
        
        # Output projection
        logits = self.lm_head(x)
        
        return logits
    
    @torch.no_grad()
    def generate(
        self,
        idx: torch.Tensor,
        max_new_tokens: int,
        temperature: float = 1.0,
        top_k: Optional[int] = None,
        top_p: Optional[float] = None,
        do_sample: bool = True,
    ) -> torch.Tensor:
        """
        Generate text using the model.
        
        Args:
            idx: Starting token indices [batch_size, seq_len]
            max_new_tokens: Number of new tokens to generate
            temperature: Sampling temperature (higher = more random)
            top_k: Keep only top k tokens
            top_p: Nucleus sampling threshold
            do_sample: Whether to sample or take argmax
        
        Returns:
            Generated token indices [batch_size, seq_len + max_new_tokens]
        """
        self.eval()
        
        for _ in range(max_new_tokens):
            # Crop context to context_length
            idx_cond = idx[:, -self.context_length:]
            
            # Get predictions
            logits = self(idx_cond)
            
            # Focus on last time step
            logits = logits[:, -1, :] / temperature
            
            # Apply top-k filtering if specified
            if top_k is not None:
                v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < v[:, [-1]]] = -float('inf')
            
            # Apply nucleus (top-p) filtering if specified
            if top_p is not None and top_p < 1.0:
                sorted_logits, sorted_indices = torch.sort(logits, descending=True)
                cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
                
                # Remove tokens with cumulative probability above threshold
                sorted_indices_to_remove = cumulative_probs > top_p
                sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
                sorted_indices_to_remove[..., 0] = 0
                
                indices_to_remove = sorted_indices_to_remove.scatter(1, sorted_indices, sorted_indices_to_remove)
                logits[indices_to_remove] = -float('inf')
            
            # Sample from distribution
            if do_sample:
                probs = F.softmax(logits, dim=-1)
                next_token = torch.multinomial(probs, num_samples=1)
            else:
                next_token = torch.argmax(logits, dim=-1, keepdim=True)
            
            # Append to sequence
            idx = torch.cat([idx, next_token], dim=1)
        
        return idx
    
    def get_num_params(self, non_embedding: bool = True) -> int:
        """Return the number of parameters in the model."""
        n_params = sum(p.numel() for p in self.parameters())
        
        if non_embedding:
            # Subtract token embeddings (often tied with output head)
            n_params -= self.token_embedding.weight.numel()
        
        return n_params


