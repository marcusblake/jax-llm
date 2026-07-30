import unittest
import numpy as np
import data


class TestCreateExample(unittest.TestCase):
    def test_create_example(self):
        input_tokens = np.array(
            [1, 2, 3, 4, 5], dtype=np.uint32
        )
        expected_labels = np.array([2, 3, 4, 5, 0])
        output_pairs = data.CreateExample().map(input_tokens)
        np.testing.assert_array_equal(output_pairs.data, input_tokens)
        np.testing.assert_array_equal(output_pairs.labels, expected_labels)

if __name__ == '__main__':
    unittest.main()