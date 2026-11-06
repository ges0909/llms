"""=== LESSON 4: Wie Embeddings durch Training entstehen ===

Ein kleines Next-Token-Modell lernt aus Beispielen, nach einem Wort das nächste
Wort vorherzusagen. Der Vorhersagefehler aktualisiert dabei auch die Embeddings.

Das Modell ist ein Lernbeispiel, kein vollständiger Transformer.
"""

import math
import random


class NextTokenModel:
    """Kleines Bigram-Modell mit trainierbarer Embedding-Tabelle."""

    def __init__(self, corpus: list[str], embedding_dim: int = 4, seed: int = 42) -> None:
        if embedding_dim <= 0:
            raise ValueError("embedding_dim muss groesser als 0 sein.")

        self.vocabulary = sorted({word for sentence in corpus for word in sentence.split()})
        if not self.vocabulary:
            raise ValueError("Das Korpus muss mindestens ein Wort enthalten.")

        self.token_to_id = {word: index for index, word in enumerate(self.vocabulary)}
        self.training_pairs: list[tuple[int, int]] = []
        for sentence in corpus:
            token_ids = [self.token_to_id[word] for word in sentence.split()]
            self.training_pairs.extend(zip(token_ids, token_ids[1:]))
        if not self.training_pairs:
            raise ValueError("Das Korpus muss mindestens ein Wortpaar enthalten.")

        self.embedding_dim = embedding_dim
        rng = random.Random(seed)
        self.embeddings = [
            [rng.uniform(-0.1, 0.1) for _ in range(embedding_dim)] for _ in self.vocabulary
        ]
        self.output_weights = [
            [rng.uniform(-0.1, 0.1) for _ in range(embedding_dim)] for _ in self.vocabulary
        ]
        self.output_biases = [0.0 for _ in self.vocabulary]

    def _logits(self, token_id: int) -> list[float]:
        """Berechnet Vorhersagewerte fuer alle Woerter im Vokabular."""
        embedding = self.embeddings[token_id]
        return [
            sum(value * weight for value, weight in zip(embedding, output_vector)) + bias
            for output_vector, bias in zip(self.output_weights, self.output_biases)
        ]

    def loss(self) -> float:
        """Berechnet den mittleren Cross-Entropy-Verlust des Trainingskorpus."""
        total_loss = 0.0
        for input_id, target_id in self.training_pairs:
            logits = self._logits(input_id)
            max_logit = max(logits)
            log_normalizer = max_logit + math.log(
                sum(math.exp(logit - max_logit) for logit in logits)
            )
            total_loss += log_normalizer - logits[target_id]
        return total_loss / len(self.training_pairs)

    def train(self, epochs: int = 200, learning_rate: float = 0.1, seed: int = 42) -> None:
        """Aktualisiert Embeddings und Ausgabeschicht mit einfachem SGD."""
        if epochs <= 0:
            raise ValueError("epochs muss groesser als 0 sein.")
        if learning_rate <= 0:
            raise ValueError("learning_rate muss groesser als 0 sein.")

        rng = random.Random(seed)
        pairs = self.training_pairs.copy()
        for _ in range(epochs):
            rng.shuffle(pairs)
            for input_id, target_id in pairs:
                input_vector = self.embeddings[input_id]
                logits = self._logits(input_id)
                max_logit = max(logits)
                exponentials = [math.exp(logit - max_logit) for logit in logits]
                normalizer = sum(exponentials)
                gradients = [value / normalizer for value in exponentials]
                gradients[target_id] -= 1.0

                input_gradient = [0.0] * self.embedding_dim
                for output_id, gradient in enumerate(gradients):
                    output_vector = self.output_weights[output_id]
                    for dimension, value in enumerate(input_vector):
                        input_gradient[dimension] += gradient * output_vector[dimension]
                        output_vector[dimension] -= learning_rate * gradient * value
                    self.output_biases[output_id] -= learning_rate * gradient

                for dimension, gradient in enumerate(input_gradient):
                    input_vector[dimension] -= learning_rate * gradient

    def predict_next(self, word: str, top_k: int = 3) -> list[tuple[str, float]]:
        """Gibt die wahrscheinlichsten Folgewörter und ihre Wahrscheinlichkeiten zurueck."""
        if word not in self.token_to_id:
            raise ValueError(f"Wort '{word}' ist nicht im Vokabular.")
        if not 1 <= top_k <= len(self.vocabulary):
            raise ValueError("top_k muss zwischen 1 und der Vokabulargroesse liegen.")

        logits = self._logits(self.token_to_id[word])
        max_logit = max(logits)
        exponentials = [math.exp(logit - max_logit) for logit in logits]
        normalizer = sum(exponentials)
        probabilities = [value / normalizer for value in exponentials]
        ranked = sorted(zip(self.vocabulary, probabilities), key=lambda item: item[1], reverse=True)
        return ranked[:top_k]


def main() -> None:
    corpus = [
        "the cat sleeps",
        "the kitten sleeps",
        "the dog barks",
        "the puppy barks",
    ]
    model = NextTokenModel(corpus, embedding_dim=4, seed=7)
    cat_id = model.token_to_id["cat"]
    initial_embedding = model.embeddings[cat_id].copy()
    initial_loss = model.loss()

    model.train(epochs=200, learning_rate=0.1)

    print("TOPIC 02 - LESSON 4: LEARNING EMBEDDINGS FROM PREDICTION")
    print(f"Loss vorher: {initial_loss:.3f}")
    print(f"Loss nachher: {model.loss():.3f}")
    print(f"Embedding 'cat' vorher: {[round(value, 3) for value in initial_embedding]}")
    print(f"Embedding 'cat' nachher: {[round(value, 3) for value in model.embeddings[cat_id]]}")

    for word in ("cat", "kitten", "dog", "puppy"):
        predictions = model.predict_next(word, top_k=2)
        formatted = ", ".join(
            f"{next_word}: {probability:.2f}" for next_word, probability in predictions
        )
        print(f"Naechstes Wort nach '{word}': {formatted}")

    print("\nDie Vorhersageaufgabe passt Embeddings an das Trainingskorpus an.")
    print("Das Toy-Modell zeigt das Prinzip, nicht das Training eines vollstaendigen LLMs.")


if __name__ == "__main__":
    main()
