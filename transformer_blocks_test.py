import unittest
import transformer_blocks as tb
import numpy as np
import jax.numpy as jnp
import jax


class SinusoidalPositionalEmbeddingsTest(unittest.TestCase):

    def test_positional_embedding_dimension(self):
        batch_size, seq_length, hidden_dim = 4, 10, 20

        pe = tb.SinusoidalPositionalEmbeddings(seq_length, hidden_dim)
        sequence = jax.random.uniform(jax.random.PRNGKey(0),
                                      shape=(batch_size, seq_length,
                                             hidden_dim))

        output = pe(sequence)
        np.testing.assert_array_equal(output.shape,
                                      (batch_size, seq_length, hidden_dim))

    def test_positional_embedding_values(self):
        batch_size, seq_length, hidden_dim = 2, 2, 2

        pe = tb.SinusoidalPositionalEmbeddings(seq_length, hidden_dim)
        sequence = jnp.zeros(shape=(batch_size, seq_length, hidden_dim))

        output = pe(sequence)
        np.testing.assert_array_almost_equal(
            output,
            jnp.array([[[0.0, 1.0], [np.sin(1.0), np.cos(1.0)]],
                       [[0.0, 1.0], [np.sin(1.0), np.cos(1.0)]]]))

    def test_positional_embedding_values_two(self):
        batch_size, seq_length, hidden_dim = 2, 3, 4

        # correct formula: PE(pos, 2i)   = sin(pos / 10000^(2i/d))
        #                  PE(pos, 2i+1) = cos(pos / 10000^(2i/d))
        freq0 = 10000**(0 / hidden_dim)  # i=0
        freq1 = 10000**(2 / hidden_dim)  # i=1

        expected_row0 = [0.0, 1.0, 0.0, 1.0]
        expected_row1 = [
            np.sin(1 / freq0),
            np.cos(1 / freq0),
            np.sin(1 / freq1),
            np.cos(1 / freq1)
        ]
        expected_row2 = [
            np.sin(2 / freq0),
            np.cos(2 / freq0),
            np.sin(2 / freq1),
            np.cos(2 / freq1)
        ]

        expected = jnp.array([[expected_row0, expected_row1, expected_row2],
                              [expected_row0, expected_row1, expected_row2]])

        pe = tb.SinusoidalPositionalEmbeddings(seq_length, hidden_dim)
        sequence = jnp.zeros(shape=(batch_size, seq_length, hidden_dim))

        output = pe(sequence)

        np.testing.assert_array_almost_equal(output, expected)


class AttentionTest(unittest.TestCase):

    def test_vanilla_self_attn(self):
        split_keys = jax.random.split(jax.random.PRNGKey(0), 3)
        batch_size, seq_length, hidden_dim = 10, 5, 20
        queries = jax.random.uniform(split_keys[0],
                                     shape=(batch_size, seq_length, hidden_dim),
                                     dtype=jnp.float32)
        keys = jax.random.uniform(split_keys[1],
                                  shape=(batch_size, seq_length, hidden_dim),
                                  dtype=jnp.float32)
        values = jax.random.uniform(split_keys[2],
                                    shape=(batch_size, seq_length, hidden_dim),
                                    dtype=jnp.float32)

        attn = tb.Attention(hidden_dim)
        attention_output = attn(queries, keys, values)
        output_tokens = attention_output.activations
        attn_scores = attention_output.attn_scores
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
                                     shape=(batch_size, seq_length, hidden_dim),
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
        attention_output = attn(queries, keys, values, attn_mask=mask)
        attn_scores = attention_output.attn_scores
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
        attn_output = attn(queries, keys, values)
        output_tokens = attn_output.activations
        attn_scores = attn_output.attn_scores
        np.testing.assert_array_equal(output_tokens.shape,
                                      (batch_size, seq_length, hidden_dim))
        np.testing.assert_array_equal(attn_scores.shape,
                                      (batch_size, seq_length, seq_length))


class LayerNormTest(unittest.TestCase):

    def test_layer_norm_returns_correct_dimensions(self):
        batch_size, seq_length, hidden_dim = 10, 6, 5
        layer_norm = tb.LayerNorm(hidden_dim)
        input_tokens = jax.random.uniform(key=jax.random.PRNGKey(0),
                                          shape=(batch_size, seq_length,
                                                 hidden_dim))
        output = layer_norm(input_tokens)
        np.testing.assert_array_equal(output.shape,
                                      (batch_size, seq_length, hidden_dim))

    def test_layer_norm_returns_correct_result(self):
        hidden_dim = 3
        layer_norm = tb.LayerNorm(hidden_dim, eps=0.0)
        # output = g / std(activations) * (activations - mean(activations)) + b
        input_tokens = jnp.array([[[1.0, 0.0, -1.0], [1.0, 1.5, 2.0]]])
        output = layer_norm(input_tokens)
        np.testing.assert_array_almost_equal(
            output,
            jnp.array([[[1.0 / np.sqrt(2 / 3), 0.0, -1.0 / np.sqrt(2 / 3)],
                        [-0.5 / np.sqrt(1 / 6), 0.0, 0.5 / np.sqrt(1 / 6)]]]))


class MultiHeadAttentionTest(unittest.TestCase):

    def test_simple_multihead_attention(self):
        num_heads, batch_size, seq_length, hidden_dim = 4, 10, 5, 20
        split_keys = jax.random.split(jax.random.PRNGKey(0), 3)
        queries = jax.random.uniform(split_keys[0],
                                     shape=(batch_size, seq_length, hidden_dim),
                                     dtype=jnp.float32)
        keys = jax.random.uniform(split_keys[1],
                                  shape=(batch_size, seq_length, hidden_dim),
                                  dtype=jnp.float32)
        values = jax.random.uniform(split_keys[2],
                                    shape=(batch_size, seq_length, hidden_dim),
                                    dtype=jnp.float32)

        mha = tb.MultiHeadAttention(num_heads, hidden_dim)
        output = mha(queries, keys, values)

        np.testing.assert_array_equal(output.activations.shape,
                                      (batch_size, seq_length, hidden_dim))

    def test_multihead_attention_causal_mask(self):
        num_heads, batch_size, seq_length, hidden_dim = 4, 10, 5, 20
        split_keys = jax.random.split(jax.random.PRNGKey(0), 3)
        queries = jax.random.uniform(split_keys[0],
                                     shape=(batch_size, seq_length, hidden_dim),
                                     dtype=jnp.float32)
        keys = jax.random.uniform(split_keys[1],
                                  shape=(batch_size, seq_length, hidden_dim),
                                  dtype=jnp.float32)
        values = jax.random.uniform(split_keys[2],
                                    shape=(batch_size, seq_length, hidden_dim),
                                    dtype=jnp.float32)

        mha = tb.MultiHeadAttention(num_heads, hidden_dim)
        mask = tb.causal_attn_mask(seq_length)
        output = mha(queries, keys, values, attn_mask=mask)

        np.testing.assert_array_equal(output.activations.shape,
                                      (batch_size, seq_length, hidden_dim))

    def test_multihead_attention_single_head_matches_attention(self):
        # Sanity check: num_heads=1 should behave like plain Attention
        # in terms of output shape (not necessarily identical values,
        # since params are initialized independently).
        batch_size, seq_length, hidden_dim = 10, 5, 20
        split_keys = jax.random.split(jax.random.PRNGKey(0), 3)
        queries = jax.random.uniform(split_keys[0],
                                     shape=(batch_size, seq_length, hidden_dim),
                                     dtype=jnp.float32)
        keys = jax.random.uniform(split_keys[1],
                                  shape=(batch_size, seq_length, hidden_dim),
                                  dtype=jnp.float32)
        values = jax.random.uniform(split_keys[2],
                                    shape=(batch_size, seq_length, hidden_dim),
                                    dtype=jnp.float32)

        mha = tb.MultiHeadAttention(1, hidden_dim)
        output = mha(queries, keys, values)

        np.testing.assert_array_equal(output.activations.shape,
                                      (batch_size, seq_length, hidden_dim))

    def test_multihead_attention_qk_value_dims(self):
        num_heads, batch_size, seq_length = 4, 10, 5
        hidden_dim, qk_dim, value_dim = 20, 8, 16
        split_keys = jax.random.split(jax.random.PRNGKey(0), 3)
        queries = jax.random.uniform(split_keys[0],
                                     shape=(batch_size, seq_length, qk_dim),
                                     dtype=jnp.float32)
        keys = jax.random.uniform(split_keys[1],
                                  shape=(batch_size, seq_length, qk_dim),
                                  dtype=jnp.float32)
        values = jax.random.uniform(split_keys[2],
                                    shape=(batch_size, seq_length, value_dim),
                                    dtype=jnp.float32)

        mha = tb.MultiHeadAttention(num_heads,
                                    hidden_dim,
                                    qk_dim=qk_dim,
                                    value_dim=value_dim)
        output = mha(queries, keys, values)

        np.testing.assert_array_equal(output.activations.shape,
                                      (batch_size, seq_length, hidden_dim))


if __name__ == '__main__':
    unittest.main()
