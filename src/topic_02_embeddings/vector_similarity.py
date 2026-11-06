"""
=== LESSON 2: Semantic Space and Vector Geometry ===

Wie drückt sich Bedeutung in Vektoren aus?
- Wörter mit ähnlicher Bedeutung zeigen im Raum in eine ähnliche Richtung.
- Beziehungen zwischen Konzepten bilden konsistente Richtungsvektoren.

Die wichtigsten Maße für Vektor-Ähnlichkeit:
1. Skalarprodukt (Dot Product):
   dot(a, b) = sum(a_i * b_i)
   Berücksichtigt Winkel UND Vektorlänge.

2. Euklidische Distanz (L2-Abstand):
   dist(a, b) = sqrt(sum((a_i - b_i)^2))
   Misst den geometrischen Abstand. Identisch = 0.

3. Cosine-Similarity (Kosinus-Ähnlichkeit):
   cos(theta) = dot(a, b) / (norm(a) * norm(b))
   Misst nur den Winkel, ignoriert die Vektorlänge.
   Bereich: -1 (entgegengesetzt), 0 (orthogonal/unabhängig), +1 (identische Richtung).

Vektorarithmetik ("Word2Vec-Analogie"):
- Vektoren codieren relationale Eigenschaften.
- König - Mann + Frau ≈ Königin
"""

import math


def dot_product(vec_a: list[float], vec_b: list[float]) -> float:
    """Berechnet das Skalarprodukt zweier gleich langer Vektoren."""
    if len(vec_a) != len(vec_b):
        raise ValueError("Vektoren müssen die gleiche Dimension besitzen.")
    return sum(a * b for a, b in zip(vec_a, vec_b))


def norm(vec: list[float]) -> float:
    """Berechnet die euklidische Norm (Länge) eines Vektors."""
    return math.sqrt(sum(x * x for x in vec))


def euclidean_distance(vec_a: list[float], vec_b: list[float]) -> float:
    """Berechnet die euklidische Distanz (L2-Abstand) zwischen zwei Vektoren."""
    if len(vec_a) != len(vec_b):
        raise ValueError("Vektoren müssen die gleiche Dimension besitzen.")
    return math.sqrt(sum((a - b) ** 2 for a, b in zip(vec_a, vec_b)))


def cosine_similarity(vec_a: list[float], vec_b: list[float]) -> float:
    """
    Berechnet die Cosine-Similarity zwischen zwei Vektoren.

    Wertebereich: [-1.0, 1.0].
    """
    norm_a = norm(vec_a)
    norm_b = norm(vec_b)
    if norm_a == 0.0 or norm_b == 0.0:
        raise ValueError("Cosine-Similarity kann fuer Nullvektoren nicht berechnet werden.")
    return dot_product(vec_a, vec_b) / (norm_a * norm_b)


def vector_add(vec_a: list[float], vec_b: list[float]) -> list[float]:
    """Addiert zwei Vektoren elementweise: a + b."""
    if len(vec_a) != len(vec_b):
        raise ValueError("Vektoren müssen die gleiche Dimension besitzen.")
    return [a + b for a, b in zip(vec_a, vec_b)]


def vector_sub(vec_a: list[float], vec_b: list[float]) -> list[float]:
    """Subtrahiert zwei Vektoren elementweise: a - b."""
    if len(vec_a) != len(vec_b):
        raise ValueError("Vektoren müssen die gleiche Dimension besitzen.")
    return [a - b for a, b in zip(vec_a, vec_b)]


class VectorSpace:
    """Verwaltet einen semantischen Vektorraum und erlaubt Aehnlichkeitssuchen."""

    def __init__(self) -> None:
        self.embeddings: dict[str, list[float]] = {}

    def add(self, word: str, vector: list[float]) -> None:
        """Fuegt ein Wort mit seinem Vektor zum Vektorraum hinzu."""
        self.embeddings[word] = vector

    def get(self, word: str) -> list[float]:
        """Gibt den Vektor fuer ein Wort zurueck."""
        if word not in self.embeddings:
            raise KeyError(f"Wort '{word}' nicht im Vektorraum vorhanden.")
        return self.embeddings[word]

    def similarity(self, word_a: str, word_b: str) -> float:
        """Gibt die Cosine-Similarity zweier Woerter zurueck."""
        return cosine_similarity(self.get(word_a), self.get(word_b))

    def most_similar(
        self,
        query_vec: list[float],
        top_k: int = 3,
        exclude: set[str] | None = None,
    ) -> list[tuple[str, float]]:
        """
        Findet die top_k aehnlichsten Woerter zu einem gegebenen Vektor.

        Args:
            query_vec: Zielvektor fuer die Suche.
            top_k: Anzahl der gewuenschten Ergebnisse.
            exclude: Woerter, die aus dem Ergebnis gefiltert werden sollen.

        Returns:
            Liste von Tupeln (Wort, Cosine-Similarity), absteigend sortiert.
        """
        excluded = exclude or set()
        scores: list[tuple[str, float]] = []

        for word, vec in self.embeddings.items():
            if word in excluded:
                continue
            sim = cosine_similarity(query_vec, vec)
            scores.append((word, sim))

        scores.sort(key=lambda item: item[1], reverse=True)
        return scores[:top_k]

    def analogy(self, word_a: str, word_b: str, word_c: str) -> list[tuple[str, float]]:
        """
        Loest Wort-Analogien nach dem Schema: 'a verhaelt sich zu b wie c zu ?'
        Formel: ziel = b - a + c
        """
        vec_a = self.get(word_a)
        vec_b = self.get(word_b)
        vec_c = self.get(word_c)

        # ziel = (b - a) + c
        target_vec = vector_add(vector_sub(vec_b, vec_a), vec_c)

        # Ursprungswörter ausschließen
        return self.most_similar(target_vec, top_k=3, exclude={word_a, word_b, word_c})


def main() -> None:
    print("=" * 70)
    print("TOPIC 02 - LESSON 2: VECTOR SIMILARITY & WORD ARITHMETIC")
    print("=" * 70)

    # Wir konstruieren einen interpretierbaren 4D-Vektorraum:
    # Dim 0: Geschlecht (negativ = männlich, positiv = weiblich)
    # Dim 1: Status (negativ = gewöhnlich, positiv = königlich)
    # Dim 2: Biologie (negativ = Mensch, positiv = Tier)
    # Dim 3: Alter/Reife (negativ = jung/klein, positiv = ausgewachsen)
    space = VectorSpace()
    space.add("mann", [-0.9, -0.4, -0.9, +0.8])
    space.add("frau", [+0.9, -0.4, -0.9, +0.8])
    space.add("koenig", [-0.9, +0.9, -0.9, +0.8])
    space.add("koenigin", [+0.9, +0.9, -0.9, +0.8])
    space.add("hund", [0.0, -0.8, +0.9, +0.7])
    space.add("welpe", [0.0, -0.8, +0.9, -0.7])
    space.add("katze", [+0.1, -0.8, +0.9, +0.6])
    space.add("kaetzchen", [+0.1, -0.8, +0.9, -0.7])

    print("Definierter semantischer Raum (Dimensionen: Geschlecht, Status, Spezies, Reife):")
    for word, vec in space.embeddings.items():
        formatted = ", ".join(f"{x:+.1f}" for x in vec)
        print(f"  {word:<10} -> [{formatted}]")

    # 1. Cosine-Similarity vergleichen
    print("\n" + "=" * 70)
    print(">>> 1. COSINE-SIMILARITY IM VERGLEICH <<<")
    pairs = [
        ("mann", "frau"),
        ("koenig", "koenigin"),
        ("hund", "katze"),
        ("hund", "welpe"),
        ("koenig", "welpe"),
    ]

    for w1, w2 in pairs:
        sim = space.similarity(w1, w2)
        dist = euclidean_distance(space.get(w1), space.get(w2))
        print(f"'{w1}' <-> '{w2}': Cosine={sim:+.3f} | Euklidische Distanz={dist:.3f}")

    print("\nErkenntnis:")
    print("- 'koenig' und 'koenigin' teilen fast alle Dimensionen (Status, Spezies, Reife).")
    print("- 'koenig' und 'welpe' haben kaum semantische Schnittmengen -> Cosine ist negativ.")

    # 2. Vektorarithmetik (Analogie)
    print("\n" + "=" * 70)
    print(">>> 2. VEKTORARITHMETIK (KÖNIG - MANN + FRAU) <<<")
    print("Berechne: v_ziel = v('koenig') - v('mann') + v('frau')")
    results = space.analogy(word_a="mann", word_b="koenig", word_c="frau")

    print("\nTop Treffer im Raum:")
    for rank, (word, score) in enumerate(results, start=1):
        print(f"  {rank}. '{word}' mit Cosine-Similarity {score:.4f}")

    print("\n>>> 3. WEITERE ANALOGIE: HUND ZU WELPE WIE KATZE ZU ? <<<")
    print("Berechne: v_ziel = v('welpe') - v('hund') + v('katze')")
    animal_results = space.analogy(word_a="hund", word_b="welpe", word_c="katze")
    for rank, (word, score) in enumerate(animal_results, start=1):
        print(f"  {rank}. '{word}' mit Cosine-Similarity {score:.4f}")

    # 3. Nearest-Neighbor-Suche
    print("\n" + "=" * 70)
    print(">>> 4. NEAREST NEIGHBOR SEARCH <<<")
    query_word = "katze"
    neighbors = space.most_similar(space.get(query_word), top_k=3, exclude={query_word})
    print(f"Die naechsten Nachbarn von '{query_word}':")
    for word, score in neighbors:
        print(f"  - '{word}' ({score:+.3f})")

    print("\n" + "=" * 70)
    print("TAKEAWAYS:")
    print("1. Semantik ist Geometrie: Ähnliche Begriffe liegen im Raum nah beieinander.")
    print("2. Cosine-Similarity misst die Richtung und ignoriert Vektorlängen.")
    print("3. Lineare Vektorbeziehungen spiegeln Bedeutungsbeziehungen wider.")
    print("4. Nearest-Neighbor-Suche ist das Fundament moderner Vektordatenbanken.")
    print("=" * 70)


if __name__ == "__main__":
    main()
