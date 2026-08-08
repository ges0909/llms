"""
=== LESSON 3: The Real GPT Tokenizer (tiktoken) ===

tiktoken ist OpenAIs offizieller Tokenizer für GPT-3.5, GPT-4, etc.
Hier vergleichen wir unseren selbstgebauten Tokenizer mit dem echten.

Wichtig: Jedes LLM hat seinen eigenen Tokenizer!
- GPT-4/4o: tiktoken (cl100k_base, o200k_base) ~100-200k Tokens
- Llama 2/3: SentencePiece (BPE) ~32k Tokens
- Claude: Eigener (BPE-basiert) ~100k Tokens
- Mistral: SentencePiece (BPE) ~32k Tokens

tiktoken Encodings:
- "cl100k_base": GPT-4, GPT-3.5-turbo, text-embedding-ada-002
- "p50k_base": Codex, text-davinci-002/003
- "r50k_base": GPT-3 (davinci, curie, etc.)
- "o200k_base": GPT-4o (neuestes Modell)
"""

import tiktoken


def show_tokenization(text: str, encoding_name: str = "cl100k_base") -> list[int]:
    """Shows in detail how tiktoken tokenizes a text."""
    enc = tiktoken.get_encoding(encoding_name)

    token_ids = enc.encode(text)
    tokens = [enc.decode([tid]) for tid in token_ids]

    print(f"Text: '{text}'")
    print(f"Encoding: {encoding_name}")
    print(f"Token count: {len(token_ids)}")
    print("\nToken-by-token breakdown:")
    print("-" * 50)

    for i, (tid, token) in enumerate(zip(token_ids, tokens)):
        display_token = repr(token)[1:-1]
        print(f"  {i:3d}: ID={tid:6d}  ->  '{display_token}'")

    print("-" * 50)
    print(f"Decoded: '{enc.decode(token_ids)}'")
    return token_ids


def compare_encodings(text: str) -> None:
    """Compares different tokenizer encodings."""
    encodings = ["cl100k_base", "p50k_base", "o200k_base"]

    print(f"\nText: '{text}'")
    print("=" * 60)

    for enc_name in encodings:
        try:
            enc = tiktoken.get_encoding(enc_name)
            tokens = enc.encode(text)
            print(
                f"{enc_name:15s}: {len(tokens):3d} tokens  ->  "
                f"{tokens[:10]}{'...' if len(tokens) > 10 else ''}"
            )
        except Exception as e:
            print(f"{enc_name:15s}: Error - {e}")


def count_tokens(text: str, model: str = "gpt-4") -> int:
    """
    Counts tokens for a specific model.
    Useful for estimating API costs!
    """
    try:
        enc = tiktoken.encoding_for_model(model)
    except KeyError:
        enc = tiktoken.get_encoding("cl100k_base")

    return len(enc.encode(text))


def visualize_subwords() -> None:
    """Shows how BPE splits words into subwords."""
    enc = tiktoken.get_encoding("cl100k_base")

    examples = [
        # German words
        ("Tokenisierung", "German compound"),
        ("Programmierung", "Another compound"),
        ("unglaublich", "With prefix"),
        # English words
        ("tokenization", "English comparison"),
        ("programming", "English"),
        # Interesting cases
        ("ChatGPT", "Brand name"),
        ("GPT-4", "With hyphen"),
        ("🚀", "Emoji"),
        ("日本語", "Japanese"),
        # Code
        ("def hello_world():", "Python code"),
        ("console.log('hello')", "JavaScript"),
        # Numbers
        ("123456789", "Long number"),
        ("3.14159", "Decimal"),
    ]

    print("\n" + "=" * 70)
    print("HOW GPT TOKENIZES DIFFERENT TEXTS:")
    print("=" * 70)

    for text, description in examples:
        tokens = enc.encode(text)
        decoded_tokens = [enc.decode([t]) for t in tokens]

        token_display = " | ".join(f"'{t}'" for t in decoded_tokens)

        print(f"\n{description}:")
        print(f"  '{text}' -> {len(tokens)} token(s)")
        print(f"  Tokens: [{token_display}]")


def token_cost_estimator() -> None:
    """Estimates API costs based on token count."""
    enc = tiktoken.get_encoding("cl100k_base")

    prompts = [
        "Was ist Python?",
        "Erkläre mir die Funktionsweise von Large Language Models in einfachen Worten.",
        """Schreibe eine Python-Funktion, die:
        1. Eine Liste von Zahlen entgegennimmt
        2. Alle geraden Zahlen filtert
        3. Diese quadriert
        4. Die Summe zurückgibt
        Füge Docstrings und Type Hints hinzu.""",
    ]

    print("\n" + "=" * 70)
    print("ESTIMATING TOKEN COSTS (GPT-4 prices, 2024):")
    print("=" * 70)

    input_price_per_1k = 0.01

    for i, prompt in enumerate(prompts, 1):
        token_count = len(enc.encode(prompt))
        input_cost = (token_count / 1000) * input_price_per_1k

        print(f"\nPrompt #{i}:")
        print(f"  Text: '{prompt[:50]}{'...' if len(prompt) > 50 else ''}'")
        print(f"  Tokens: {token_count}")
        print(f"  Estimated input cost: ${input_cost:.6f}")


def main() -> None:
    print("=" * 70)
    print("TIKTOKEN - OpenAI's Official GPT Tokenizer")
    print("=" * 70)

    print("\n>>> EXAMPLE 1: Basic Tokenization <<<\n")
    show_tokenization("Hallo, ich lerne wie LLMs funktionieren!")

    print("\n>>> EXAMPLE 2: Compare Different Encodings <<<")
    compare_encodings("The quick brown fox jumps over the lazy dog.")
    compare_encodings("Donaudampfschifffahrtsgesellschaftskapitän")

    print("\n>>> EXAMPLE 3: Subword Splitting <<<")
    visualize_subwords()

    print("\n>>> EXAMPLE 4: Estimate API Costs <<<")
    token_cost_estimator()

    print("\n" + "=" * 70)
    print("KEY INSIGHTS:")
    print("=" * 70)
    print("""
    1. LANGUAGE EFFICIENCY:
       Tokenizer sind sprachunabhängig im Design, aber abhängig in der Effizienz:

       | Sprache   | Tokens/Wort | Grund                    |
       |-----------|-------------|--------------------------|
       | Englisch  | ~1.3        | Meiste Trainingsdaten    |
       | Deutsch   | ~1.8        | Weniger Daten, Komposita |
       | Chinesisch| ~2-3        | Zeichen-basiert          |
       | Arabisch  | ~3-5        | Wenig Trainingsdaten     |

       -> Deutscher Text kostet ~30-40% mehr Tokens als englischer!

    2. CODE is handled specially:
       - Common patterns like 'def', 'function', '()' are single tokens
       - Indentation (spaces) can be multiple tokens!

    3. NUMBERS are expensive:
       - Each digit can be its own token
       - "123456789" = 9 tokens!

    4. EMOJIS and UNICODE:
       - Split into multiple tokens
       - One emoji can cost 2-4 tokens

    5. WHITESPACE matters:
       - Leading space often part of the token
       - ' hello' and 'hello' are different tokens!

    6. DIFFERENT MODELS, DIFFERENT TOKENIZERS:
       - GPT-4 Tokenizer gibt falsche Ergebnisse für Llama-Text
       - Für Token-Zählung immer den passenden Tokenizer verwenden!
    """)

    print("=" * 70)
    print("Run with: uv run tokenize-tiktoken")
    print("=" * 70)


if __name__ == "__main__":
    main()
