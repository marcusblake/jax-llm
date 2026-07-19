from flax import nnx
import jax
import transformer_blocks as tb
import chex
import jax.numpy as jnp
import dataclasses
from typing import Callable, Dict

class MaskedSelfAttentionBlock(nnx.Module):
    def __init__(self, num_heads: int,
                 hidden_dim: int,
                 *,
                 rngs: nnx.Rngs = nnx.Rngs(0),
                 qk_dim: int = -1,
                 value_dim: int = -1):
        self.multihead_self_attention = tb.MultiHeadAttention(num_heads=num_heads, hidden_dim=hidden_dim)
        self.linear = nnx.Linear(hidden_dim, hidden_dim)

    def __call__(self, input_tokens: jax.Array) -> jax.Array:
        pass

class MiniGPT(nnx.Module):

    def __init__(self, vocab_size: int, hidden_dim: int, attention_heads: int,
                 num_attention_layers: int):
        self.vocab_size = vocab_size
        self.word_embeddings = tb.WordEmbeddings(
            vocab_size, hidden_dim)
        self.positional_embeddings = tb.PositionalEmbeddings()
        self.attention_blocks = [
            MaskedSelfAttentionBlock(attention_heads, hidden_dim) for _ in range(num_attention_layers)
        ]
        self.text_classifier = nnx.Linear(hidden_dim, vocab_size)

    def __call__(self, input_sequence: jax.Array) -> jax.Array:
        batch_size = input_sequence.shape[0]
        seq = self.word_embeddings(input_sequence)
        seq = seq + self.positional_embeddings()
        for attn_block in self.attention_blocks:
            output = attn_block(seq)

        # Use last output to predict the next token. vocab_logits should be of
        vocab_logits = self.text_classifier(output[:, :, -1])
        chex.assert_shape(vocab_logits, (batch_size, self.vocab_size))
        return nnx.softmax(vocab_logits, axis=-1)


