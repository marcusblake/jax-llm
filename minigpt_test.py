import unittest
import numpy as np
import jax.numpy as jnp
import jax
from minigpt import MiniGPT


class MiniGptTest(unittest.TestCase):

    def test_full_forward_pass(self):
        mini_gpt = MiniGPT()


if __name__ == '__main__':
    unittest.main()
