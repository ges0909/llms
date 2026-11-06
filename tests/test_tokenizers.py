import unittest

from topic_01_tokenization.basic_tokenizers import CharacterTokenizer
from topic_01_tokenization.bpe_tokenizer import BPETokenizer


class CharacterTokenizerTests(unittest.TestCase):
    def test_encode_rejects_characters_not_seen_during_training(self) -> None:
        tokenizer = CharacterTokenizer()
        tokenizer.train("abc")

        with self.assertRaisesRegex(ValueError, "not seen during training"):
            tokenizer.encode("☃")


class BPETokenizerTests(unittest.TestCase):
    def test_round_trip_preserves_whitespace(self) -> None:
        tokenizer = BPETokenizer()
        tokenizer.train("a b\nc\t", num_merges=3, verbose=False)
        text = " a  b\nc\t"

        self.assertEqual(tokenizer.decode(tokenizer.encode(text)), text)

    def test_encode_rejects_characters_not_seen_during_training(self) -> None:
        tokenizer = BPETokenizer()
        tokenizer.train("hello world", num_merges=3, verbose=False)

        with self.assertRaisesRegex(ValueError, "not seen during training"):
            tokenizer.encode("☃")

    def test_training_resets_previous_merges(self) -> None:
        tokenizer = BPETokenizer()
        tokenizer.train("a b", num_merges=1, verbose=False)
        tokenizer.train("a b", num_merges=0, verbose=False)

        self.assertEqual(tokenizer.merges, {})
        self.assertEqual(tokenizer.decode(tokenizer.encode("a b")), "a b")


if __name__ == "__main__":
    unittest.main()
