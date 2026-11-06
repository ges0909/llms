import math
import unittest

from topic_02_embeddings.embedding_lookup import EmbeddingTable, one_hot_encode
from topic_02_embeddings.embedding_training import NextTokenModel
from topic_02_embeddings.transformer_embeddings import TransformerEmbedding
from topic_02_embeddings.vector_similarity import (
    VectorSpace,
    cosine_similarity,
    euclidean_distance,
)


class EmbeddingLookupTests(unittest.TestCase):
    def test_one_hot_encode(self) -> None:
        vec = one_hot_encode(2, 5)
        self.assertEqual(vec, [0.0, 0.0, 1.0, 0.0, 0.0])

    def test_one_hot_encode_out_of_bounds(self) -> None:
        with self.assertRaises(ValueError):
            one_hot_encode(5, 5)
        with self.assertRaises(ValueError):
            one_hot_encode(-1, 5)

    def test_lookup_matches_one_hot_multiplication(self) -> None:
        table = EmbeddingTable(vocab_size=10, embedding_dim=4, seed=123)
        for tid in range(10):
            direct = table.lookup(tid)
            via_one_hot = table.lookup_via_one_hot(tid)
            self.assertEqual(direct, via_one_hot)

    def test_lookup_sequence(self) -> None:
        table = EmbeddingTable(vocab_size=10, embedding_dim=6, seed=123)
        seq = table.lookup_sequence([0, 2, 4])
        self.assertEqual(len(seq), 3)
        self.assertEqual(len(seq[0]), 6)

    def test_lookup_out_of_bounds(self) -> None:
        table = EmbeddingTable(vocab_size=5, embedding_dim=3)
        with self.assertRaises(ValueError):
            table.lookup(10)


class EmbeddingTrainingTests(unittest.TestCase):
    def test_training_reduces_loss_and_updates_embeddings(self) -> None:
        corpus = [
            "the cat sleeps",
            "the kitten sleeps",
            "the dog barks",
            "the puppy barks",
        ]
        model = NextTokenModel(corpus, embedding_dim=4, seed=7)
        initial_loss = model.loss()
        initial_embeddings = [vector.copy() for vector in model.embeddings]

        model.train(epochs=200, learning_rate=0.1)

        self.assertLess(model.loss(), initial_loss)
        self.assertNotEqual(model.embeddings, initial_embeddings)
        self.assertEqual(model.predict_next("cat", top_k=1)[0][0], "sleeps")


class VectorSimilarityTests(unittest.TestCase):
    def test_cosine_similarity_properties(self) -> None:
        v1 = [1.0, 0.0, 0.0]
        v2 = [1.0, 0.0, 0.0]
        v3 = [0.0, 1.0, 0.0]
        v4 = [-1.0, 0.0, 0.0]

        # Identisch
        self.assertAlmostEqual(cosine_similarity(v1, v2), 1.0)
        # Orthogonal
        self.assertAlmostEqual(cosine_similarity(v1, v3), 0.0)
        # Entgegengesetzt
        self.assertAlmostEqual(cosine_similarity(v1, v4), -1.0)

    def test_euclidean_distance(self) -> None:
        v1 = [0.0, 0.0]
        v2 = [3.0, 4.0]
        self.assertAlmostEqual(euclidean_distance(v1, v2), 5.0)

    def test_analogy_solving(self) -> None:
        space = VectorSpace()
        space.add("mann", [-1.0, 0.0])
        space.add("frau", [1.0, 0.0])
        space.add("koenig", [-1.0, 1.0])
        space.add("koenigin", [1.0, 1.0])

        # koenig - mann + frau -> koenigin
        results = space.analogy("mann", "koenig", "frau")
        self.assertEqual(results[0][0], "koenigin")
        self.assertAlmostEqual(results[0][1], 1.0)


class TransformerEmbeddingTests(unittest.TestCase):
    def test_forward_shape(self) -> None:
        layer = TransformerEmbedding(vocab_size=100, d_model=16, scale_by_sqrt=False)
        output = layer.forward([1, 5, 20])
        self.assertEqual(len(output), 3)
        self.assertEqual(len(output[0]), 16)

    def test_scaling_factor(self) -> None:
        d_model = 16
        layer_unscaled = TransformerEmbedding(
            vocab_size=10, d_model=d_model, scale_by_sqrt=False, seed=42
        )
        layer_scaled = TransformerEmbedding(
            vocab_size=10, d_model=d_model, scale_by_sqrt=True, seed=42
        )

        unscaled_vec = layer_unscaled.forward([0])[0]
        scaled_vec = layer_scaled.forward([0])[0]

        expected_scale = math.sqrt(d_model)
        for u, s in zip(unscaled_vec, scaled_vec):
            self.assertAlmostEqual(s, round(u * expected_scale, 4), places=3)


if __name__ == "__main__":
    unittest.main()
