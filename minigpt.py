from flax import nnx
import jax
import transformer_blocks as tb
import chex
import jax.numpy as jnp
import dataclasses
from typing import Callable, Dict
from train_utils import TrainingConfig
import data
import optax


class Feedforward(nnx.Module):

    def __init__(self,
                 hidden_dim: int,
                 *,
                 expansion: int = 4,
                 rngs: nnx.Rngs = nnx.Rngs(0)):
        self.linear1 = nnx.Linear(hidden_dim, hidden_dim * expansion, rngs=rngs)
        self.linear2 = nnx.Linear(hidden_dim * expansion, hidden_dim, rngs=rngs)

    def __call__(self, inputs: jax.Array) -> jax.Array:
        inputs = self.linear1(inputs)
        inputs = nnx.relu(inputs)
        return self.linear2(inputs)


class MaskedSelfAttentionBlock(nnx.Module):

    def __init__(self,
                 num_heads: int,
                 hidden_dim: int,
                 *,
                 rngs: nnx.Rngs = nnx.Rngs(0)):
        self.multihead_self_attention = tb.MultiHeadAttention(
            num_heads=num_heads, hidden_dim=hidden_dim, rngs=rngs)
        self.ffn = Feedforward(hidden_dim, rngs=rngs)
        self.layer_norm1 = tb.LayerNorm(hidden_dim)
        self.layer_norm2 = tb.LayerNorm(hidden_dim)

    def __call__(self, input_tokens: jax.Array) -> jax.Array:
        """Forward pass in self attention block consistent with GPT architecture.

        Args:
            input_tokens: Input tokens of dimension [B, T, D]

        Returns:
            Activations after passing through attention block with causal attention mask.
        """
        attention_output = self.multihead_self_attention(
            input_tokens,
            input_tokens,
            input_tokens,
            attn_mask=tb.causal_attn_mask(input_tokens.shape[1]))
        chex.assert_equal_shape([attention_output.activations, input_tokens])
        # Add skip connection.
        out = self.layer_norm1(input_tokens + attention_output.activations)
        # Add another skip connection.
        out = out + self.ffn(out)
        return self.layer_norm2(out)


class MiniGPT(nnx.Module):

    def __init__(self, vocab_size: int, max_seq_length: int, hidden_dim: int,
                 attention_heads: int, num_attention_layers: int):
        self.vocab_size = vocab_size
        self.word_embeddings = tb.WordEmbeddings(vocab_size, hidden_dim)
        self.positional_embeddings = tb.SinusoidalPositionalEmbeddings(
            max_seq_length, hidden_dim)
        self.attention_blocks = [
            MaskedSelfAttentionBlock(attention_heads, hidden_dim)
            for _ in range(num_attention_layers)
        ]
        self.next_token_prediction = nnx.Linear(hidden_dim, vocab_size)

    def __call__(self, input_sequence: jax.Array) -> jax.Array:
        batch_size = input_sequence.shape[0]
        seq_length = input_sequence.shape[1]
        seq = self.word_embeddings(input_sequence)
        seq = self.positional_embeddings(seq)
        for attn_block in self.attention_blocks:
            seq = attn_block(seq)

        vocab_logits = self.next_token_prediction(seq)
        chex.assert_shape(vocab_logits,
                          (batch_size, seq_length, self.vocab_size))
        return vocab_logits


def mini_gpt_training_config() -> TrainingConfig:
    return TrainingConfig(
        model=MiniGPT(),
        train_data_loader=data.get_tiny_stories_data_loader('train',
                                                            num_epochs=10,
                                                            batch_size=64),
        eval_data_loader=data.get_tiny_stories_data_loader('test',
                                                           num_epochs=10,
                                                           batch_size=32),
        loss_fn=optax.softmax_cross_entropy_with_integer_labels)
