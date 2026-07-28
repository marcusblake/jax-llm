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

    def __init__(self,
                 vocab_size: int,
                 max_seq_length: int,
                 hidden_dim: int,
                 attention_heads: int,
                 num_attention_layers: int,
                 *,
                 rngs: nnx.Rngs = nnx.Rngs(0)):
        self.vocab_size = vocab_size
        self.word_embeddings = tb.WordEmbeddings(vocab_size, hidden_dim)
        self.positional_embeddings = tb.SinusoidalPositionalEmbeddings(
            max_seq_length, hidden_dim)
        self.attention_blocks = nnx.List([
            MaskedSelfAttentionBlock(attention_heads, hidden_dim, rngs=rngs)
            for _ in range(num_attention_layers)
        ])
        self.next_token_prediction = nnx.Linear(hidden_dim,
                                                vocab_size,
                                                rngs=rngs)

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
    data_config = data.get_tiny_stories_config(num_epochs=100, batch_size=64)
    gpt_model = MiniGPT(data_config.vocab_size,
                        data_config.max_seq_length,
                        hidden_dim=256,
                        attention_heads=4,
                        num_attention_layers=2)
    print('model', gpt_model)
    return TrainingConfig(
        model=gpt_model,
        data_config=data_config,
        optimizer=nnx.Optimizer(gpt_model,
                                optax.adam(learning_rate=1e-3),
                                wrt=nnx.Param),
        loss_fn=optax.softmax_cross_entropy_with_integer_labels)
