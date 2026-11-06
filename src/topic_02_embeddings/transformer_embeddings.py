"""
=== LESSON 3: Embeddings in Modern LLMs (GPT & Co.) ===

Wie Transformer-Modelle Embeddings verarbeiten:
1. Token Embedding Matrix (W_E):
   - Jedes Token aus dem Vokabular V wird auf einen Vektor mit d_model Dimensionen abgebildet.
   - Form der Matrix: (vocab_size, d_model).

2. Skalierung mit sqrt(d_model):
   - Im ursprünglichen Transformer ("Attention Is All You Need") werden die Embeddings
     mit sqrt(d_model) multipliziert.
   - Grund: Gleicht die Skalierung gegenueber den folgenden Positional Encodings aus.

3. Das Polysemie-Problem (Kontextfreiheit):
   - Reine Token-Embeddings sind STATISCH:
     "Ich sitze auf der Bank" vs. "Ich bringe Geld zur Bank".
   - In beiden Sätzen hat das Token "Bank" VOR dem Transformer exakt denselben Vektor!
   - Erst die Attention-Schichten (Topic 4 & 5) machen daraus kontextuelle Embeddings.

4. Weight Tying:
   - Viele Modelle (GPT-2, PaLM, Gemma) teilen sich die Gewichtsmatrix W_E:
     Eingang: Token-IDs -> Vektoren (Lookup in W_E)
     Ausgang: Finale Vektoren -> Token-Logits (Multiplikation mit W_E^T)
   - Spart Millionen Parameter.
"""

import math
import random

import tiktoken


class TransformerEmbedding:
    """
    Simuliert den Embedding-Layer eines modernen Decoders (wie GPT).
    """

    def __init__(
        self,
        vocab_size: int,
        d_model: int,
        scale_by_sqrt: bool = True,
        seed: int = 42,
    ) -> None:
        self.vocab_size = vocab_size
        self.d_model = d_model
        self.scale_by_sqrt = scale_by_sqrt
        self.scale_factor = math.sqrt(d_model) if scale_by_sqrt else 1.0

        # Initialisierung: Gauss-Verteilung mit Varianz 1 / d_model
        rng = random.Random(seed)
        std_dev = 1.0 / math.sqrt(d_model)
        self.weights: list[list[float]] = [
            [round(rng.gauss(0.0, std_dev), 4) for _ in range(d_model)] for _ in range(vocab_size)
        ]

    def forward(self, token_ids: list[int]) -> list[list[float]]:
        """
        Wandelt eine Token-ID-Sequenz in skalierte Embedding-Vektoren um.

        Input:  Sequenz der Laenge T (Token-IDs)
        Output: Tensor der Form (T, d_model)
        """
        embedded_sequence: list[list[float]] = []

        for tid in token_ids:
            if not 0 <= tid < self.vocab_size:
                raise ValueError(
                    f"Token-ID {tid} liegt ausserhalb des Vokabulars (0..{self.vocab_size - 1})."
                )
            base_vec = self.weights[tid]
            scaled_vec = [round(val * self.scale_factor, 4) for val in base_vec]
            embedded_sequence.append(scaled_vec)

        return embedded_sequence


def compare_model_dimensions() -> None:
    """Vergleicht Embedding-Dimensionen und Parameter-Anteile bekannter LLMs."""
    # (Modell, Vocab Size V, d_model, Gesamtparameter ca.)
    models: list[tuple[str, int, int, int]] = [
        ("GPT-2 Small", 50257, 768, 124_000_000),
        ("GPT-2 XL", 50257, 1600, 1_558_000_000),
        ("Llama 2 (7B)", 32000, 4096, 6_738_000_000),
        ("Llama 3 (8B)", 128256, 4096, 8_030_000_000),
        ("GPT-3 (175B)", 50257, 12288, 175_000_000_000),
    ]

    print("\n" + "=" * 70)
    print(">>> EMBEDDING-DIMENSIONEN BEKANNTER LLMS <<<")
    hdr = f"{'Modell':<15} | {'Vocab V':>8} | {'d_model':>7} | {'Emb Params':>12} | {'Anteil':>8}"
    print(hdr)
    print("-" * 70)

    for name, v, d, total_params in models:
        emb_params = v * d
        ratio = (emb_params / total_params) * 100
        print(f"{name:<15} | {v:>8,} | {d:>7,} | {emb_params:>12,} | {ratio:>7.1f}%")

    print("\nBeobachtung:")
    print("- Bei Llama 3 machen die Embeddings durch das 128k-Vokabular ca. 6.5% des Modells aus.")
    print("- Bei kleineren Modellen wie GPT-2 Small liegt der Anteil sogar bei fast 31%!")


def main() -> None:
    print("=" * 70)
    print("TOPIC 02 - LESSON 3: TRANSFORMER EMBEDDINGS (GPT & CO.)")
    print("=" * 70)

    # 1. Reale Tokenisierung mit tiktoken aus Topic 01
    enc = tiktoken.get_encoding("cl100k_base")
    sample_text = "Die Bank im Park."
    token_ids = enc.encode(sample_text)
    token_strings = [enc.decode([tid]) for tid in token_ids]

    print("Schritt 1: Tokenisierung (aus Topic 01):")
    print(f"Text: '{sample_text}'")
    print(f"Token-IDs:     {token_ids}")
    print(f"Token-Strings: {token_strings}")

    # 2. Embedding-Layer instanziieren
    d_model = 8  # Für Lehrzwecke kompakt gewählt
    emb_layer = TransformerEmbedding(
        vocab_size=enc.n_vocab,
        d_model=d_model,
        scale_by_sqrt=True,
        seed=1337,
    )

    embedded = emb_layer.forward(token_ids)

    scale = math.sqrt(d_model)
    print(f"\nSchritt 2: Lookup + Skalierung (sqrt({d_model}) = {scale:.3f}):")
    print(f"Shape: ({len(embedded)}, {d_model}) [Sequenzlänge x d_model]")
    for tid, t_str, vec in zip(token_ids, token_strings, embedded):
        formatted = ", ".join(f"{x:+.3f}" for x in vec)
        print(f"  ID {tid:>6} ('{t_str:<5}') -> [{formatted}]")

    # 3. Das Polysemie-Problem demonstrieren
    print("\n" + "=" * 70)
    print(">>> 3. DAS PROBLEM STATISCHER EMBEDDINGS: POLYSEMIE <<<")
    satz_1 = "Die Bank im Park"
    satz_2 = "Geld zur Bank bringen"

    ids_1 = enc.encode(satz_1)
    ids_2 = enc.encode(satz_2)
    print(f"Token-IDs Satz 1: {ids_1}")
    print(f"Token-IDs Satz 2: {ids_2}")

    # Finde das Token fuer ' Bank'
    bank_id = enc.encode(" Bank")[0]

    vec_bank_1 = emb_layer.forward([bank_id])[0]
    vec_bank_2 = emb_layer.forward([bank_id])[0]

    print(f"Satz 1: '{satz_1}' -> Token ' Bank' hat ID {bank_id}")
    print(f"Satz 2: '{satz_2}' -> Token ' Bank' hat ID {bank_id}")
    print(f"Identischer Input-Vektor? {vec_bank_1 == vec_bank_2}")
    print("-> Embedding-Layer weiss NICHT, ob Parkbank oder Geldinstitut gemeint ist!")
    print("-> Erst nachfolgende Attention-Layer (Topic 4 & 5) loesen den Kontext auf.")

    # 4. Modellvergleich
    compare_model_dimensions()

    print("\n" + "=" * 70)
    print("TAKEAWAYS:")
    print("1. Transformer-Input ist eine Matrix der Form (Sequenzlänge, d_model).")
    print("2. Die Multiplikation mit sqrt(d_model) stabilisiert nachfolgende Layer.")
    print("3. Reine Token-Embeddings sind kontextblind; Kontext entsteht erst durch Attention.")
    print("4. Große Vokabulare (z. B. 128k bei Llama 3) erhöhen den Speicherbedarf für Embeddings.")
    print("=" * 70)


if __name__ == "__main__":
    main()
