"""
=== LESSON 1: From One-Hot to Dense Embeddings ===

Nach der Tokenisierung liegt Text als Sequenz von Zahlen (Token-IDs) vor:
"Hallo Welt" -> [15496, 16896]

Problem:
- Token-IDs sind diskrete Kategorien, keine kontinuierlichen Werte.
- ID 10 ist mathematisch nicht "doppelt so viel" oder "näher" an ID 5 als an ID 1000.
- Neuronale Netze benötigen kontinuierliche Vektoren, um Gradienten zu berechnen.

Die zwei Repräsentationsformen:
1. One-Hot-Encoding:
   - Ein Vektor der Größe V (Vokabular), der überall 0 ist und nur an Index token_id eine 1 hat.
   - Nachteil 1: Extrem groß und ineffizient (V = 50.000 -> 49.999 Nullen pro Token).
   - Nachteil 2: Orthogonalität. Jedes Wortpaar hat Distanz sqrt(2) und Cosine-Similarity 0.
   - Keine Semantik darstellbar.

2. Dense Embeddings (Dichte Vektoren):
   - Jedes Token wird auf einen Vektor mit d Dimensionen (z. B. d = 768) abgebildet.
   - Alle Einträge sind kontinuierliche Fließkommazahlen (Floats).
   - Ähnliche Bedeutungen erhalten ähnliche Vektoren im Raum.

Mathematisch vs. Praktisch:
- Mathematisch: x_one_hot @ W_embedding (Matrixmultiplikation)
- Praktisch: W_embedding[token_id] (direkter Tabellen-Lookup in O(1))
"""

import random


def one_hot_encode(token_id: int, vocab_size: int) -> list[float]:
    """
    Erstellt einen One-Hot-Vektor fuer eine gegebene Token-ID.

    Args:
        token_id: Index des Tokens im Vokabular.
        vocab_size: Gesamtzahl aller Tokens im Vokabular.

    Returns:
        Liste der Laenge vocab_size mit genau einer 1.0 an Position token_id.
    """
    if not 0 <= token_id < vocab_size:
        raise ValueError(
            f"token_id ({token_id}) muss im Bereich 0 <= id < vocab_size ({vocab_size}) liegen."
        )

    vector = [0.0] * vocab_size
    vector[token_id] = 1.0
    return vector


class EmbeddingTable:
    """
    Einfache Embedding-Tabelle (Gewichtsmatrix) von Grund auf implementiert.

    Speichert fuer jedes Token des Vokabulars einen dichten Vektor der Dimension embedding_dim.
    """

    def __init__(self, vocab_size: int, embedding_dim: int, seed: int = 42) -> None:
        """
        Initialisiert die Embedding-Matrix mit deterministischen Zufallswerten.

        Args:
            vocab_size: Anzahl der zulaessigen Tokens (Zeilen der Matrix).
            embedding_dim: Groesse jedes Embedding-Vektors (Spalten der Matrix).
            seed: Zufalls-Seed fuer reproduzierbare Gewichte.
        """
        self.vocab_size = vocab_size
        self.embedding_dim = embedding_dim

        # Initialisierung mit kleinen Zufallszahlen um 0 herum
        rng = random.Random(seed)
        self.weights: list[list[float]] = [
            [round(rng.gauss(0.0, 0.1), 4) for _ in range(embedding_dim)] for _ in range(vocab_size)
        ]

    def lookup(self, token_id: int) -> list[float]:
        """
        Schlaegt den Embedding-Vektor fuer ein Token nach (O(1)-Zugriff).

        Args:
            token_id: Index des gesuchten Tokens.

        Returns:
            Dichter Vektor mit embedding_dim Werten.
        """
        if not 0 <= token_id < self.vocab_size:
            raise ValueError(
                f"token_id ({token_id}) liegt ausserhalb des Vokabulars ({self.vocab_size})."
            )
        return self.weights[token_id]

    def lookup_sequence(self, token_ids: list[int]) -> list[list[float]]:
        """
        Wandelt eine Liste von Token-IDs in eine Liste von Embedding-Vektoren um.

        Args:
            token_ids: Sequenz von Token-IDs.

        Returns:
            Matrix der Form (len(token_ids), embedding_dim).
        """
        return [self.lookup(tid) for tid in token_ids]

    def lookup_via_one_hot(self, token_id: int) -> list[float]:
        """
        Berechnet das Embedding ueber Matrixmultiplikation: one_hot @ weights.

        Beweist, dass Lookup und Matrixmultiplikation mathematisch identisch sind.
        """
        one_hot = one_hot_encode(token_id, self.vocab_size)
        result = [0.0] * self.embedding_dim

        # Zeilenweise Multiplikation und Aufsummieren
        for row_idx, factor in enumerate(one_hot):
            if factor != 0.0:
                row = self.weights[row_idx]
                for col_idx in range(self.embedding_dim):
                    result[col_idx] += factor * row[col_idx]

        return [round(val, 4) for val in result]

    @property
    def num_parameters(self) -> int:
        """Gibt die Gesamtanzahl an lernbaren Parametern in der Tabelle zurueck."""
        return self.vocab_size * self.embedding_dim


def main() -> None:
    # 1. Kleines Vokabular definieren
    vocab = ["der", "hund", "sieht", "die", "katze"]
    vocab_size = len(vocab)
    embedding_dim = 4

    print("=" * 70)
    print("TOPIC 02 - LESSON 1: EMBEDDING LOOKUP & ONE-HOT ENCODING")
    print("=" * 70)
    print(f"Vokabular ({vocab_size} Wörter): {vocab}")
    print(f"Embedding-Dimension (d_model): {embedding_dim}\n")

    # 2. One-Hot-Encoding demonstrieren
    print(">>> 1. ONE-HOT ENCODING <<<")
    for tid, word in enumerate(vocab):
        oh = one_hot_encode(tid, vocab_size)
        print(f"ID {tid} ('{word:>5}'): {oh}")

    print("\nProblem bei One-Hot:")
    print("- Kein Begriff von Ähnlichkeit: Alle Vektoren stehen orthogonal aufeinander.")
    print("- Skaliert katastrophal: Bei 50.000 Tokens hätte jeder Vektor 50.000 Dimensionen.")

    # 3. Dense Embedding Table
    print("\n" + "=" * 70)
    print(">>> 2. DENSE EMBEDDING LOOKUP TABLE <<<")
    table = EmbeddingTable(vocab_size=vocab_size, embedding_dim=embedding_dim, seed=42)

    for tid, word in enumerate(vocab):
        vec = table.lookup(tid)
        formatted = ", ".join(f"{v:+.4f}" for v in vec)
        print(f"ID {tid} ('{word:>5}') -> [{formatted}]")

    # 4. Beweis: Lookup == One-Hot @ Weights
    print("\n" + "=" * 70)
    print(">>> 3. MATHEMATISCHER BEWEIS: LOOKUP == ONE-HOT @ WEIGHTS <<<")
    test_id = 1
    direct = table.lookup(test_id)
    multiplied = table.lookup_via_one_hot(test_id)
    print(f"Wort: '{vocab[test_id]}' (ID {test_id})")
    print(f"Direkter Tabellen-Lookup (O(1)):         {direct}")
    print(f"Matrix-Multiplikation (One-Hot @ W):     {multiplied}")
    print(f"Identisch? {'Ja!' if direct == multiplied else 'Nein'}")
    print("-> LLMs nutzen den direkten Lookup, weil er dramatisch schneller und sparsamer ist.")

    # 5. Sequenz-Verarbeitung
    print("\n" + "=" * 70)
    print(">>> 4. SEQUENZ VON TOKENS EINBETTEN <<<")
    sentence_tokens = [0, 1, 2, 3, 4]  # "der hund sieht die katze"
    embedded_sequence = table.lookup_sequence(sentence_tokens)
    print(f"Input Token-IDs: {sentence_tokens}")
    print(f"Form des Output-Tensors: ({len(embedded_sequence)}, {embedding_dim})")
    print("Ergebnis (Tokens x Dimensionen):")
    for tid, vec in zip(sentence_tokens, embedded_sequence):
        formatted = ", ".join(f"{v:+.4f}" for v in vec)
        print(f"  Token {tid} ('{vocab[tid]}'): [{formatted}]")

    # 6. Reale Dimensionen & Parameteranzahl
    print("\n" + "=" * 70)
    print(">>> 5. PARAMETER REALER MODELLE <<<")
    models = [
        ("Unser Mini-Beispiel", vocab_size, embedding_dim),
        ("GPT-2 (Small)", 50257, 768),
        ("GPT-3", 50257, 12288),
        ("Llama 3 (8B)", 128256, 4096),
    ]

    for name, v, d in models:
        params = v * d
        mb = (params * 4) / (1024 * 1024)  # 4 Bytes pro Float32
        print(f"{name:<22}: V={v:>7,} | d={d:>5,} | {params:>12,} Params (~{mb:>6.1f} MB)")

    print("\n" + "=" * 70)
    print("TAKEAWAYS:")
    print("1. One-Hot ist speicherhungrig und semantisch blind (orthogonale Vektoren).")
    print("2. Dense Embeddings bilden Tokens auf kontinuierliche Fließkomma-Vektoren ab.")
    print("3. Embedding-Lookup E[token_id] ist mathematisch identisch zu One-Hot @ E.")
    print("4. Die Embedding-Matrix macht bei großen Vokabularen viele Gewichte aus.")
    print("=" * 70)


if __name__ == "__main__":
    main()
