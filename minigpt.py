import flax.nnx as nnx
import jax


class PositionEmbeddings(nnx.Module):
    def __init__(self):
        pass

    def __call__(self):
        pass


class WordEmbeddings(nnx.Module):
    def __init__(self, vocab_size: int, hidden_dim: int):
        self.word_embeds = nnx.Embed(vocab_size, hidden_dim)

    def __call__(self, token_id: jax.Array):
        return self.word_embeds(token_id)
    

class MultiHeadAttention(nnx.Module):
    def __init__(self):
        pass

    def __call__(self, query: jax.Array, key: jax.Array, value: jax.Array, attention_mask: jax.Array):
        pass


class MiniGPT(nnx.Module):
    def __init__(self, vocab_size: int, attention_heads: int,):
        pass

    def __call__(self):
        pass