import tiktoken
import numpy as np
import os
import grain
import jax

_TINY_STORIES_SPECIAL_CHARACTER = '<|endoftext|>'
_TINY_STORIES_TRAIN_DATA = 'data/TinyStories/TinyStories-train.txt'
_TINY_STORIES_TEST_DATA = 'data/TinyStories/TinyStories-valid.txt'
_DEFAULT_TOKENIZER = tiktoken.encoding_for_model('gpt-4')

_MAX_SEQUENCE_LENGTH = 516


class Tokenize(grain.transforms.Map):

    def map(self, input_text: list[str]) -> np.array:
        return _DEFAULT_TOKENIZER.encode_to_numpy(input_text)


class TruncateOrPad(grain.transforms.Map):

    def map(self, input_sequence: np.array) -> np.array:
        n_tokens = len(input_sequence)
        padding_length = max(0, _MAX_SEQUENCE_LENGTH - n_tokens)
        output_sequence = np.pad(input_sequence, (0, padding_length))
        return output_sequence[:_MAX_SEQUENCE_LENGTH]


def _read_tinystories_from_file(data_split):
    filename = _TINY_STORIES_TRAIN_DATA if data_split == 'train' else _TINY_STORIES_TEST_DATA
    full_path = os.path.join(os.curdir, filename)
    with open(full_path) as f:
        text = f.read()
        stories = text.split(_TINY_STORIES_SPECIAL_CHARACTER)
    return stories


class TinyStoriesDataSource(grain.sources.RandomAccessDataSource):

    def __init__(self, data_split: str):
        if data_split not in ("train", "test"):
            raise ValueError("data_split must be train or test")
        self._stories = _read_tinystories_from_file(data_split)

    def __getitem__(self, index: int) -> str:
        """Given an index, returns story at index within the dataset."""
        return self._stories[index]

    def __len__(self) -> int:
        return len(self._stories)

    def __repr__(self) -> str:
        return f'TinyStoriesDataSource(len={len(self)})'


def get_tiny_stories_data_loader(data_split: str,
                                 num_epochs: int) -> grain.DataLoader:
    data_source = TinyStoriesDataSource(data_split)
    sampler = grain.samplers.IndexSampler(num_records=len(data_source),
                                          shuffle=True,
                                          num_epochs=num_epochs,
                                          seed=10)
    return grain.DataLoader(data_source=data_source,
                            sampler=sampler,
                            operations=[Tokenize(),
                                        TruncateOrPad()])
