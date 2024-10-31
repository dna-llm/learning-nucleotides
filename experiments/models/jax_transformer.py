import jax
import optax
from typing import Tuple
from flax import linen as nn
from jax import (
    Array, 
    numpy as jnp, 
    lax
)

class PositionalEmbedding:
    """```
    Sinusoidal Fixed Positional Embeddings
    Args:
        maxlen:int
        dim:int
    sinusoidal_embeddings: 
        pos_emb: (1, maxlen, dim)
    get_freqs:
        get_freqs: sin_freqs(1, maxlen, 1, dim), cos_freqs(1, maxlen, 1, dim)
    ```"""
    def __init__(self, maxlen:int, dim:int):
        p, i = jnp.meshgrid(jnp.arange(float(maxlen)), jnp.arange(dim/2)*2)
        theta = (p/1e4**(i/dim)).T

        self.pos_emb = jnp.stack([jnp.sin(theta), jnp.cos(theta)], axis=-1)
        self.pos_emb = self.pos_emb.reshape((maxlen, dim))[None] # (1, maxlen, dim)

    def sinusoidal_embeddings(self):
        return self.pos_emb # (1, maxlen, dim)
    
    def get_freqs(self):
        sin_freqs = jnp.repeat(self.pos_emb[..., None, ::2], repeats=2, axis=-1)
        cos_freqs = jnp.repeat(self.pos_emb[..., None, 1::2], repeats=2, axis=-1)
        return sin_freqs, cos_freqs # (1, maxlen, 1, dim), (1, maxlen, 1, dim)
    

class DecoderBlock(nn.Module):
    hidden_dim: int
    num_heads: int
    dropout_rate: float = 0.1

    @nn.compact
    def __call__(self, x, training: bool = True):
        # Self-attention
        norm_x = nn.LayerNorm()(x)
        attention_output = nn.MultiHeadDotProductAttention(
            num_heads=self.num_heads, dropout_rate=self.dropout_rate
        )(norm_x, norm_x, norm_x, deterministic=not training)
        x += attention_output

        # Feed-forward network
        norm_x = nn.LayerNorm()(x)
        ff_output = nn.Dense(self.hidden_dim * 4)(norm_x)
        ff_output = nn.gelu(ff_output)
        ff_output = nn.Dense(self.hidden_dim)(ff_output)
        ff_output = nn.Dropout(rate=self.dropout_rate, deterministic=not training)(ff_output)

        return x + ff_output


class TransformerDecoder(nn.Module):
    vocab_size: int
    hidden_dim: int
    num_layers: int
    num_heads: int
    max_len: int
    dropout_rate: float = 0.1

    @nn.compact
    def __call__(self, x, training: bool = True):
        # Token embedding
        token_emb = nn.Embed(num_embeddings=self.vocab_size, features=self.hidden_dim)(x)

        # Positional embedding
        pos_emb = PositionalEmbedding(maxlen=self.max_len, dim=self.hidden_dim).sinusoidal_embeddings()
        # Ensureing pos_emb matches token_emb in sequence length
        pos_emb = pos_emb[:, :x.shape[1], :]

        x = token_emb + pos_emb
        x = nn.Dropout(rate=self.dropout_rate, deterministic=not training)(x)

        # Decoder blocks
        for _ in range(self.num_layers):
            x = DecoderBlock(
                hidden_dim=self.hidden_dim, num_heads=self.num_heads, dropout_rate=self.dropout_rate
            )(x, training)

        x = nn.LayerNorm()(x)
        logits = nn.Dense(self.vocab_size)(x)
        return logits



