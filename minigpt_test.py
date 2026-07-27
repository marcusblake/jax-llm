import unittest
import numpy as np
import jax.numpy as jnp
import jax
from minigpt import MiniGPT, MaskedSelfAttentionBlock

# class MiniGptTest(unittest.TestCase):

#     def test_full_forward_pass(self):
#         mini_gpt = MiniGPT()


class MaskedSelfAttentionBlockTest(unittest.TestCase):

    def test_forward_pass(self):
        batch_size, seq_length, hidden_dim = 5, 100, 5
        num_heads = 4
        input_sequences = jax.random.uniform(jax.random.PRNGKey(0),
                                             shape=(batch_size, seq_length,
                                                    hidden_dim))

        mse = MaskedSelfAttentionBlock(num_heads, hidden_dim)

        outputs = mse(input_sequences)
        np.testing.assert_array_equal(outputs.shape,
                                      (batch_size, seq_length, hidden_dim))

    def test_causal_masking(self):
        hidden_dim, num_heads, seq_length = 4, 2, 4
        block = MaskedSelfAttentionBlock(num_heads=num_heads,
                                         hidden_dim=hidden_dim)

        key = jax.random.PRNGKey(0)
        tokens = jax.random.normal(key, (1, seq_length, hidden_dim))

        # Perturb only the final position.
        tokens_modified = tokens.at[:, -1, :].set(
            jax.random.normal(jax.random.PRNGKey(1), (hidden_dim,)))

        output_original = block(tokens)
        output_modified = block(tokens_modified)
        print('org', output_original)
        print('mod', output_modified)

        # Every position except the last must be unaffected by the change.
        np.testing.assert_array_almost_equal(output_original[:, :-1, :],
                                             output_modified[:, :-1, :])

        # The last position SHOULD change, since it attends to itself.
        self.assertFalse(
            jnp.allclose(output_original[:, -1, :], output_modified[:, -1, :]))


if __name__ == '__main__':
    unittest.main()
