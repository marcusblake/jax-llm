import unittest
import transformer_blocks as tb
import numpy as np
import jax.numpy as jnp
import jax


class AttentionTest(unittest.TestCase):

    def test_vanilla_self_attn(self):
        split_keys = jax.random.split(jax.random.PRNGKey(0), 3)
        batch_size, seq_length, hidden_dim = 10, 5, 20
        queries = jax.random.uniform(split_keys[0],
                                     shape=(batch_size, seq_length,
                                            hidden_dim),
                                     dtype=jnp.float32)
        keys = jax.random.uniform(split_keys[1],
                                  shape=(batch_size, seq_length, hidden_dim),
                                  dtype=jnp.float32)
        values = jax.random.uniform(split_keys[2],
                                    shape=(batch_size, seq_length, hidden_dim),
                                    dtype=jnp.float32)

        attn = tb.Attention(hidden_dim)
        output = attn(queries, keys, values)
        output_tokens = output['output']
        attn_scores = output['attn_scores']
        np.testing.assert_array_equal(output_tokens.shape,
                                      (batch_size, seq_length, hidden_dim))
        np.testing.assert_array_equal(attn_scores.shape,
                                      (batch_size, seq_length, seq_length))
        np.testing.assert_allclose(jnp.sum(attn_scores, axis=-1),
                                   jnp.ones(shape=(batch_size, seq_length)),
                                   rtol=1e-5)

    def test_self_attention_causal_mask(self):
        split_keys = jax.random.split(jax.random.PRNGKey(0), 3)
        batch_size, seq_length, hidden_dim = 10, 5, 20
        queries = jax.random.uniform(split_keys[0],
                                     shape=(batch_size, seq_length,
                                            hidden_dim),
                                     dtype=jnp.float32)
        keys = jax.random.uniform(split_keys[1],
                                  shape=(batch_size, seq_length, hidden_dim),
                                  dtype=jnp.float32)
        values = jax.random.uniform(split_keys[2],
                                    shape=(batch_size, seq_length, hidden_dim),
                                    dtype=jnp.float32)

        attn = tb.Attention(hidden_dim)
        mask = tb.causal_attn_mask(seq_length)
        expected_mask = np.array([[1, 0, 0, 0, 0], [1, 1, 0, 0, 0],
                                  [1, 1, 1, 0, 0], [1, 1, 1, 1, 0],
                                  [1, 1, 1, 1, 1]])
        np.testing.assert_array_equal(mask, expected_mask)
        output = attn(queries, keys, values, mask=mask)
        attn_scores = output['attn_scores']
        total_probabilities = jax.vmap(
            lambda scores, mask: jnp.sum(mask * scores, axis=-1),
            in_axes=[0, None])(attn_scores, mask)
        np.testing.assert_allclose(total_probabilities,
                                   jnp.ones(shape=(batch_size, seq_length)),
                                   rtol=1e-5)

    def test_cross_attention(self):
        split_keys = jax.random.split(jax.random.PRNGKey(0), 3)
        batch_size, seq_length, hidden_dim, qk_dim, value_dim = 10, 5, 20, 8, 16
        queries = jax.random.uniform(split_keys[0],
                                     shape=(batch_size, seq_length, qk_dim),
                                     dtype=jnp.float32)
        keys = jax.random.uniform(split_keys[1],
                                  shape=(batch_size, seq_length, qk_dim),
                                  dtype=jnp.float32)
        values = jax.random.uniform(split_keys[2],
                                    shape=(batch_size, seq_length, value_dim),
                                    dtype=jnp.float32)

        attn = tb.Attention(hidden_dim, qk_dim=qk_dim, value_dim=value_dim)
        output = attn(queries, keys, values)
        output_tokens = output['output']
        attn_scores = output['attn_scores']
        np.testing.assert_array_equal(output_tokens.shape,
                                      (batch_size, seq_length, hidden_dim))
        np.testing.assert_array_equal(attn_scores.shape,
                                      (batch_size, seq_length, seq_length))


class LayerNormTest(unittest.TestCase):
    def test_vanilla_layer_norm(self):
        layer_norm = tb.LayerNorm(10)

        

# class PositionalEmbeddingTest(unittest.TestCase):
#     def test_positional_embedding(self):
#         pass

if __name__ == '__main__':
    unittest.main()
