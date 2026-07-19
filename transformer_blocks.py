from flax import nnx
import jax
import jax.numpy as jnp
from typing import Dict
import chex

# Needed to make logits very small so that they don't contribute to probabilities
# when computing softmax.
SMALL_LOGIT = -jnp.inf


def causal_attn_mask(seq_length: int) -> jax.Array:
    """Returns a causal attention mask to prevent model from attending to context in the past.
    """
    return jnp.tril(jnp.ones(shape=(seq_length, seq_length)))


class PositionalEmbeddings(nnx.Module):

    def __init__(self):
        pass

    def __call__(self):
        pass


class WordEmbeddings(nnx.Module):

    def __init__(self, vocab_size: int, hidden_dim: int):
        self.word_embeds = nnx.Embed(vocab_size, hidden_dim)

    def __call__(self, token_id: jax.Array):
        return self.word_embeds(token_id)


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
                 attn_mask: jax.Array | None = None) -> Dict[str, jax.Array]:
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
        h_k = jnp.transpose(self.W_k(key), axes=[0, 2, 1])
        # Get dimension of the keys.
        d_k = h_k.shape[-1]
        logits = jnp.matmul(h_q, h_k) / jnp.sqrt(d_k)
        if attn_mask is not None:
            logits = jax.vmap(
                lambda logits, mask: jnp.where(mask, logits, SMALL_LOGIT),
                in_axes=[0, None])(logits, attn_mask)
        # Calculates attention scores.
        scores = nnx.softmax(logits, axis=-1)
        h_v = self.W_v(value)

        output = jnp.matmul(scores, h_v)
        return {'output': output, 'attn_scores': scores}
    
class LayerNorm(nnx.Module):
    def __init__(self, hidden_dim: int):
        self.gain_param = nnx.Param(jnp.ones(hidden_dim))
        self.bias_param = nnx.Param(jnp.zeros(hidden_dim))

    def __call__(self, input_tokens: jax.Array) -> jax.Array:
        """Normalizes the activations across the batch.

        https://arxiv.org/pdf/1607.06450

        input_tokens: [B, T, D]
        """
        mu = jnp.mean(input_tokens, axis=-1)
        sigma = jnp.std(input_tokens, axis=-1)
        return (self.gain_param / sigma) * (input_tokens - mu) + self.bias_param


class MultiHeadAttention(nnx.Module):

    def __init__(self,
                 num_heads: int,
                 hidden_dim: int,
                 *,
                 rngs: nnx.Rngs = nnx.Rngs(0),
                 qk_dim: int = -1,
                 value_dim: int = -1):
        if qk_dim < 0:
            qk_dim = hidden_dim
        if value_dim < 0:
            value_dim = hidden_dim
        self.attention_heads = [
            Attention(hidden_dim,
                      rngs=rngs,
                      qk_dim=qk_dim,
                      value_dim=value_dim) for _ in range(num_heads)
        ]
        self.W_o = nnx.Linear(num_heads * value_dim, hidden_dim)

    def __call__(self, query: jax.Array, key: jax.Array, value: jax.Array,
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
        outputs = []
        attn_scores = []
        for attn in self.attention_heads:
            output = attn(query, key, value, attn_mask)
            outputs.append(output['output'])
            attn_scores.append(output['attn_scores'])
        outputs = jnp.stack(outputs)
        attn_scores = jnp.stack(attn_scores)
        chex.assert_shape(outputs, ())


