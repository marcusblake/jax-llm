from flax import nnx, struct
import jax
import jax.numpy as jnp
from typing import Dict
import chex
import dataclasses

# Needed to make logits very small so that they don't contribute to probabilities
# when computing softmax.
SMALL_LOGIT = -jnp.inf


def causal_attn_mask(seq_length: int) -> jax.Array:
    """Returns a causal attention mask to prevent model from attending to context in the past.
    """
    return jnp.tril(jnp.ones(shape=(seq_length, seq_length)))


class PositionalEmbeddings(nnx.Module):
    """
    """

    def __init__(self,
                 max_seq_length: int,
                 hidden_dim: int,
                 *,
                 rngs: nnx.Rngs = nnx.Rngs(0)):
        self.positional_embeds = nnx.Embed(max_seq_length,
                                           hidden_dim,
                                           rngs=rngs)

    def __call__(self, x: jax.Array, seq_length: int) -> jax.Array:
        seq_length = x.shape[0]
        return x + self.positional_embeds(jnp.arange(0, seq_length))


class SinusoidalPositionalEmbeddings(nnx.Module):

    def __init__(self, max_seq_length: int, hidden_dim: int):
        self.max_seq_length = max_seq_length
        self.hidden_dim = hidden_dim
        if hidden_dim % 2 != 0:
            raise ValueError("hidden_dim must be even")

        self.positional_embeds = jnp.zeros(shape=(max_seq_length, hidden_dim))
        positions = jnp.expand_dims(jnp.arange(0, max_seq_length), axis=1)
        dim_values = jnp.log(10000) * jnp.arange(0, hidden_dim, 2) / hidden_dim
        values = positions / jnp.exp(dim_values)
        # pos / 10000 ^ ((2 * i)/d_model)
        self.positional_embeds = self.positional_embeds.at[:, ::2].set(
            jnp.sin(values))
        self.positional_embeds = self.positional_embeds.at[:, 1::2].set(
            jnp.cos(values))

    def __call__(self, x: jax.Array) -> jax.Array:
        """Forward pass on sequence x [B, T, D]
        """
        return x + self.positional_embeds


class WordEmbeddings(nnx.Module):

    def __init__(self,
                 vocab_size: int,
                 hidden_dim: int,
                 *,
                 rngs: nnx.Rngs = nnx.Rngs(0)):
        self.word_embeds = nnx.Embed(vocab_size, hidden_dim, rngs=rngs)

    def __call__(self, token_ids: jax.Array):
        """Array of token_ids.

        Args:
            token_ids: Token ids of size [B, T]

        Returns:
            Learned word embeddings of size [B, T, D]
        """
        return self.word_embeds(token_ids)


@struct.dataclass
class AttentionOutput:
    activations: jax.Array
    attn_scores: jax.Array | None


class Attention(nnx.Module):

    def __init__(self,
                 hidden_dim: int,
                 *,
                 rngs: nnx.Rngs = nnx.Rngs(0),
                 qk_dim: int = -1,
                 value_dim: int = -1):
        if qk_dim < 0:
            qk_dim = hidden_dim
        if value_dim < 0:
            value_dim = hidden_dim

        self.W_q = nnx.Linear(qk_dim, hidden_dim, rngs=rngs)
        self.W_k = nnx.Linear(qk_dim, hidden_dim, rngs=rngs)
        self.W_v = nnx.Linear(value_dim, hidden_dim, rngs=rngs)

    def __call__(self,
                 query: jax.Array,
                 key: jax.Array,
                 value: jax.Array,
                 attn_mask: jax.Array | None = None) -> AttentionOutput:
        """Forward pass for calculating self attention.

        Args:
            query - Query vectors [B, T, D]
            key - Key vector [B, T, D]
            value - value vectors [B, T, D]
            attn_mask - Attention mask to apply on output. Espcially for causal attention. [T, T]

        Performs dot product self attention softmax(QK^T / sqrt(d)) * V

        Returns:
            Attention scores, 
        """
        h_q = self.W_q(query)
        h_k = self.W_k(key)
        d_k = h_k.shape[-1]
        h_k = jnp.transpose(h_k, axes=[0, 2, 1])
        # Get dimension of the keys.
        logits = jnp.matmul(h_q, h_k) / jnp.sqrt(d_k)
        if attn_mask is not None:
            logits = jax.vmap(
                lambda logits, mask: jnp.where(mask, logits, SMALL_LOGIT),
                in_axes=[0, None])(logits, attn_mask)
        # Calculates attention scores.
        scores = nnx.softmax(logits, axis=-1)
        h_v = self.W_v(value)

        output = jnp.matmul(scores, h_v)
        return AttentionOutput(activations=output, attn_scores=scores)


class LayerNorm(nnx.Module):

    def __init__(self, hidden_dim: int, eps: int = 1e-9):
        self.gain_param = nnx.Param(jnp.ones(hidden_dim))
        self.bias_param = nnx.Param(jnp.zeros(hidden_dim))
        self.eps = eps

    def __call__(self, input_tokens: jax.Array) -> jax.Array:
        """Normalizes the activations across the batch.

        https://arxiv.org/pdf/1607.06450

        input_tokens: [B, T, D]
        """
        # Normalize across hidden dimension activations.
        mu = jnp.mean(input_tokens, axis=-1, keepdims=True)  # Outputs [B, T, 1]
        sigma = jnp.std(input_tokens, axis=-1, keepdims=True)
        res = (self.gain_param *
                (input_tokens - mu)) / (sigma + self.eps) + self.bias_param
        return res


class MultiHeadAttention(nnx.Module):

    def __init__(self,
                 num_heads: int,
                 hidden_dim: int,
                 *,
                 rngs: nnx.Rngs = nnx.Rngs(0),
                 qk_dim: int = -1,
                 value_dim: int = -1):
        head_dim = hidden_dim // num_heads
        def make_head(rngs: nnx.Rngs):
            return Attention(hidden_dim=head_dim,
                             rngs=rngs,
                             qk_dim=hidden_dim,
                             value_dim=hidden_dim)

        self.attention_heads = nnx.vmap(make_head, in_axes=0,
                                        out_axes=0)(rngs.split(num_heads))
        self.W_o = nnx.Linear(hidden_dim, hidden_dim, rngs=rngs)
        self.value_dim = value_dim
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        self.dim_per_head = head_dim

    def __call__(self,
                 query: jax.Array,
                 key: jax.Array,
                 value: jax.Array,
                 attn_mask: jax.Array | None = None):
        """Forward pass for calculating self attention.

        Args:
            query - Query vectors [B, T, D]
            key - Key vector [B, T, D]
            value - value vectors [B, T, D]
            attn_mask - Attention mask to apply on output. Espcially for causal attention. [T, T]

        Performs dot product self attention softmax(QK^T / sqrt(d)) * V

        Returns:
            Attention scores, 
        """
        b, seq_len, _ = query.shape

        def run_head(attn, q, k, v, mask):
            return attn(q, k, v, mask)
        outputs = nnx.vmap(run_head,
                           in_axes=(0, None, None, None, None),
                           out_axes=0)(self.attention_heads, query, key, value,
                                       attn_mask)
        h = jnp.transpose(outputs.activations, axes=[1, 2, 3, 0])
        h = jnp.reshape(h, shape=(b, seq_len, -1))
        chex.assert_shape(h, (b, seq_len, self.hidden_dim))
        h = self.W_o(h)
        chex.assert_shape(h, (b, seq_len, self.hidden_dim))
        return AttentionOutput(activations=h, attn_scores=outputs.attn_scores)
