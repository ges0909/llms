"""
=== LESSON 2: BPE (Byte Pair Encoding) ===

Das Problem mit Character- und Word-Level Tokenizern:
- Character: Zu lange Sequenzen, Modell muss Wörter selbst lernen
- Word: Zu großes Vokabular, kann neue Wörter nicht verarbeiten

Die Lösung: BPE (Byte Pair Encoding)
- Startet mit Zeichen als Basis-Tokens
- Findet die häufigsten Zeichenpaare und merged sie zu neuen Tokens
- Wiederholt das, bis gewünschte Vokabular-Größe erreicht
- Ergebnis: Subword-Tokens (z.B. "un", "belieb", "able")

GPT-2, GPT-3, GPT-4 nutzen alle BPE (mit kleinen Variationen)!
"""

from collections import Counter


class BPETokenizer:
    """
    Byte Pair Encoding Tokenizer - wie GPT tokenisiert.

    Der Algorithmus:
    1. Starte mit allen Zeichen als einzelne Tokens
    2. Zähle alle benachbarten Token-Paare
    3. Merge das häufigste Paar zu einem neuen Token
    4. Wiederhole bis gewünschte Vokabular-Größe erreicht
    """

    def __init__(self) -> None:
        # Note: merges dict preserves insertion order (Python 3.7+),
        # which is required for correct BPE encoding
        self.merges: dict[tuple[str, str], str] = {}
        self.vocab: dict[str, int] = {}
        self.inverse_vocab: dict[int, str] = {}

    def _get_pairs(self, tokens: list[str]) -> Counter[tuple[str, str]]:
        """Count all adjacent pairs in the token list."""
        pairs: Counter[tuple[str, str]] = Counter()
        for i in range(len(tokens) - 1):
            pair = (tokens[i], tokens[i + 1])
            pairs[pair] += 1
        return pairs

    def _merge_pair(
        self, tokens: list[str], pair: tuple[str, str], new_token: str
    ) -> list[str]:
        """Replace all occurrences of a pair with the new token."""
        result: list[str] = []
        i = 0
        while i < len(tokens):
            if (
                i < len(tokens) - 1
                and tokens[i] == pair[0]
                and tokens[i + 1] == pair[1]
            ):
                result.append(new_token)
                i += 2
            else:
                result.append(tokens[i])
                i += 1
        return result

    def train(self, text: str, num_merges: int = 20, *, verbose: bool = True) -> None:
        """
        Trainiert den BPE Tokenizer.

        Args:
            text: Training text
            num_merges: How many merge operations (more = larger vocabulary)
            verbose: Show each merge step
        """
        words = text.split()
        tokenized_words = [list(word) + ["</w>"] for word in words]

        all_tokens: list[str] = []
        for word_tokens in tokenized_words:
            all_tokens.extend(word_tokens)
            all_tokens.append(" ")
        all_tokens = all_tokens[:-1]

        unique_tokens = sorted(set(all_tokens))
        self.vocab = {token: idx for idx, token in enumerate(unique_tokens)}

        if verbose:
            print(f"Initial vocabulary ({len(self.vocab)} tokens):")
            print(f"  {unique_tokens}\n")
            print(f"Text as tokens: {all_tokens[:50]}...")
            print(f"\nStarting {num_merges} merge operations:\n")

        for i in range(num_merges):
            pairs = self._get_pairs(all_tokens)
            if not pairs:
                print("No more pairs to merge!")
                break

            most_common_pair = pairs.most_common(1)[0]
            pair, count = most_common_pair
            new_token = pair[0] + pair[1]

            if verbose:
                print(
                    f"Merge #{i+1}: '{pair[0]}' + '{pair[1]}' -> '{new_token}' "
                    f"(occurred {count}x)"
                )

            all_tokens = self._merge_pair(all_tokens, pair, new_token)

            self.merges[pair] = new_token
            if new_token not in self.vocab:
                self.vocab[new_token] = len(self.vocab)

        self.inverse_vocab = {idx: token for token, idx in self.vocab.items()}

        if verbose:
            print(f"\nFinal vocabulary ({len(self.vocab)} tokens):")
            print(f"  {list(self.vocab.keys())}")

    def encode(self, text: str) -> list[int]:
        """Text -> Token IDs using learned merges."""
        words = text.split()
        all_token_ids: list[int] = []

        for word in words:
            tokens: list[str] = list(word) + ["</w>"]

            # Apply merges in training order (dict preserves insertion order)
            for pair, new_token in self.merges.items():
                tokens = self._merge_pair(tokens, pair, new_token)

            for token in tokens:
                if token in self.vocab:
                    all_token_ids.append(self.vocab[token])
                else:
                    for char in token:
                        if char in self.vocab:
                            all_token_ids.append(self.vocab[char])

        return all_token_ids

    def decode(self, token_ids: list[int]) -> str:
        """Token IDs -> Text."""
        tokens = [self.inverse_vocab[idx] for idx in token_ids]
        text = "".join(tokens)
        text = text.replace("</w>", " ").strip()
        return text

    def tokenize(self, text: str) -> list[str]:
        """Returns token strings (not IDs) - useful for understanding."""
        words = text.split()
        all_tokens: list[str] = []

        for word in words:
            tokens: list[str] = list(word) + ["</w>"]
            for pair, new_token in self.merges.items():
                tokens = self._merge_pair(tokens, pair, new_token)
            all_tokens.extend(tokens)

        return all_tokens


def main() -> None:
    training_text = """
    der die das ein eine einer einem einen
    lernen lernt gelernt lernend
    programmieren programmiert programmierer
    tokenisieren tokenisiert tokenisierer tokenisierung
    verstehen versteht verstanden
    """

    print("=" * 70)
    print("BPE (BYTE PAIR ENCODING) - How GPT tokenizes!")
    print("=" * 70)

    tokenizer = BPETokenizer()
    tokenizer.train(training_text.strip(), num_merges=15, verbose=True)

    print("\n" + "=" * 70)
    print("ENCODING TESTS:")
    print("=" * 70)

    test_texts = [
        "lernen",
        "gelernt",
        "programmieren",
        "tokenisierung",
        "der die das",
    ]

    for text in test_texts:
        tokens = tokenizer.tokenize(text)
        ids = tokenizer.encode(text)
        decoded = tokenizer.decode(ids)

        print(f"\n'{text}':")
        print(f"  Tokens:  {tokens}")
        print(f"  IDs:     {ids}")
        print(f"  Decoded: '{decoded}'")

    print("\n" + "=" * 70)
    print("WHY BPE IS CLEVER:")
    print("=" * 70)
    print("""
    1. Common words/subwords become single tokens
       -> "der", "die" could be single tokens

    2. Rare words are split into known parts
       -> "Tokenisierung" = "Token" + "is" + "ierung"

    3. Vocabulary stays manageable (~50,000 tokens for GPT)
       while ANY text can be processed

    4. Morphology is often learned:
       -> "-ung", "-er", "-iert" become their own tokens
    """)

    print("=" * 70)
    print("Next lesson: Comparison with the real GPT tokenizer (tiktoken)!")
    print("=" * 70)


if __name__ == "__main__":
    main()
