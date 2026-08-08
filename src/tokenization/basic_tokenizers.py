"""
=== LESSON 1: Understanding Basic Tokenizers ===

Was ist Tokenizing?
- LLMs können keinen Text direkt verarbeiten - nur Zahlen
- Tokenizing wandelt Text in eine Sequenz von Zahlen (Token-IDs) um
- Der Tokenizer hat ein "Vokabular" (Mapping: Wort/Zeichen -> Zahl)

Wir bauen hier zwei einfache Tokenizer:
1. Character-Level: Jedes Zeichen ist ein Token
2. Word-Level: Jedes Wort ist ein Token
"""


class CharacterTokenizer:
    """
    Tokenisiert Text auf Zeichen-Ebene.

    Vorteil: Kann jeden Text verarbeiten (kleine Vokabular-Größe)
    Nachteil: Sequenzen werden sehr lang, Modell muss Wörter "lernen"
    """

    def __init__(self) -> None:
        self.char_to_id: dict[str, int] = {}
        self.id_to_char: dict[int, str] = {}

    def train(self, text: str) -> None:
        """Baut das Vokabular aus dem Text auf."""
        chars = sorted(set(text))
        self.char_to_id = {char: idx for idx, char in enumerate(chars)}
        self.id_to_char = {idx: char for idx, char in enumerate(chars)}

        print(f"Vocabulary size: {len(chars)} characters")
        print(f"Vocabulary: {chars}")

    def encode(self, text: str) -> list[int]:
        """Text -> List of token IDs."""
        return [self.char_to_id[char] for char in text]

    def decode(self, token_ids: list[int]) -> str:
        """List of token IDs -> Text."""
        return "".join(self.id_to_char[idx] for idx in token_ids)


class WordTokenizer:
    """
    Tokenisiert Text auf Wort-Ebene.

    Vorteil: Sequenzen sind kürzer, semantisch sinnvoller
    Nachteil: Riesiges Vokabular nötig, unbekannte Wörter problematisch
    """

    UNK_TOKEN = "<UNK>"

    def __init__(self) -> None:
        self.word_to_id: dict[str, int] = {}
        self.id_to_word: dict[int, str] = {}

    def train(self, text: str) -> None:
        """Baut das Vokabular aus dem Text auf."""
        words = text.split()
        unique_words = sorted(set(words))
        vocab = [self.UNK_TOKEN] + unique_words

        self.word_to_id = {word: idx for idx, word in enumerate(vocab)}
        self.id_to_word = {idx: word for idx, word in enumerate(vocab)}

        print(f"Vocabulary size: {len(vocab)} words")
        print(f"First 10 words: {vocab[:10]}")

    def encode(self, text: str) -> list[int]:
        """Text -> List of token IDs."""
        words = text.split()
        return [self.word_to_id.get(word, 0) for word in words]

    def decode(self, token_ids: list[int]) -> str:
        """List of token IDs -> Text."""
        return " ".join(self.id_to_word[idx] for idx in token_ids)


def main() -> None:
    text = "Hallo Welt! Das ist ein Test. Hallo nochmal."

    print("=" * 60)
    print("ORIGINAL TEXT:")
    print(text)
    print("=" * 60)

    # Character-Level
    print("\n>>> CHARACTER-LEVEL TOKENIZER <<<\n")

    char_tok = CharacterTokenizer()
    char_tok.train(text)

    encoded = char_tok.encode("Hallo")
    print(f"\n'Hallo' encoded: {encoded}")
    print(f"Decoded back: '{char_tok.decode(encoded)}'")

    full_encoded = char_tok.encode(text)
    print(f"\nFull text ({len(text)} chars) -> {len(full_encoded)} tokens")
    print(f"Token IDs: {full_encoded}")

    # Word-Level
    print("\n" + "=" * 60)
    print(">>> WORD-LEVEL TOKENIZER <<<\n")

    word_tok = WordTokenizer()
    word_tok.train(text)

    encoded = word_tok.encode("Hallo Welt!")
    print(f"\n'Hallo Welt!' encoded: {encoded}")
    print(f"Decoded back: '{word_tok.decode(encoded)}'")

    full_encoded = word_tok.encode(text)
    print(f"\nFull text -> {len(full_encoded)} tokens")
    print(f"Token IDs: {full_encoded}")

    # Unknown words
    print("\n--- Unknown words ---")
    unknown = word_tok.encode("Python ist toll")
    print(f"'Python ist toll' encoded: {unknown}")
    print(f"Decoded: '{word_tok.decode(unknown)}'")
    print("(Python, ist, toll were not in training -> become <UNK>)")

    print("\n" + "=" * 60)
    print("TAKEAWAYS:")
    print("- Character-Level: Flexible, but long sequences")
    print("- Word-Level: Short sequences, but unknown word problem")
    print("- Solution: BPE (Byte Pair Encoding) -> see next lesson!")
    print("=" * 60)


if __name__ == "__main__":
    main()
