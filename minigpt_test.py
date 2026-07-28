import unittest
import numpy as np
import jax.numpy as jnp
import jax
from flax import nnx
from minigpt import MiniGPT, MaskedSelfAttentionBlock
from train_utils import train_step
import optax


class MiniGptTest(unittest.TestCase):

    def test_train_step(self):
        batch_size, seq_length, hidden_dim = 5, 10, 156
        vocab_size = 10
        mini_gpt = MiniGPT(vocab_size=vocab_size,
                           max_seq_length=seq_length,
                           hidden_dim=hidden_dim,
                           attention_heads=2,
                           num_attention_layers=4)
        input_sequences = jax.random.randint(jax.random.PRNGKey(0),
                                             minval=0,
                                             maxval=vocab_size - 1,
                                             shape=(batch_size, seq_length))
        vocab_logits = jax.random.uniform(jax.random.PRNGKey(1),
                                          shape=vocab_size)
        vocab_prob = nnx.softmax(vocab_logits)
        labels = jax.random.multinomial(jax.random.PRNGKey(2),
                                        n=1,
                                        p=vocab_prob,
                                        shape=(batch_size, seq_length,
                                               vocab_size))
        outputs = mini_gpt(input_sequences)
        np.testing.assert_array_equal(outputs.shape,
                                      (batch_size, seq_length, vocab_size))
        train_step(
            mini_gpt,
            nnx.Optimizer(mini_gpt,
                          optax.adam(learning_rate=1e-3),
                          wrt=nnx.Param), input_sequences, labels,
            optax.safe_softmax_cross_entropy)


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

        # Every position except the last must be unaffected by the change.
        np.testing.assert_array_almost_equal(output_original[:, :-1, :],
                                             output_modified[:, :-1, :])

        # The last position SHOULD change, since it attends to itself.
        self.assertFalse(
            jnp.allclose(output_original[:, -1, :], output_modified[:, -1, :]))


if __name__ == '__main__':
    unittest.main()
