# Tokenization

## Was ist Tokenization?

Tokenization wandelt Text in Zahlen um – der erste Schritt bei der Verarbeitung durch ein LLM.

```
"Hallo Welt" → [15496, 16896] → Modell → [284, 612] → "to you"
```

| Begriff        | Erklärung                                     |
| -------------- | --------------------------------------------- |
| **Token**      | Kleinste Einheit (Zeichen, Subwort oder Wort) |
| **Vocabulary** | Menge aller bekannten Tokens mit IDs          |
| **Encoding**   | Text → Token-IDs                              |
| **Decoding**   | Token-IDs → Text                              |

## Tokenizer-Arten

### Character-Level

```
"Hello" → ["H", "e", "l", "l", "o"] → [7, 4, 11, 11, 14]
```

- ✅ Kleines Vokabular (~100-300), verarbeitet jedes Zeichen aus dem Trainingsvokabular
- ❌ Sehr lange Sequenzen, Modell muss Wörter selbst lernen

### Word-Level

```
"Hello World" → ["Hello", "World"] → [142, 89]
```

- ✅ Kurze Sequenzen, semantisch sinnvoll
- ❌ Riesiges Vokabular nötig, unbekannte Wörter problematisch (OOV)

### Subword-Level (BPE)

```
"tokenization" → ["token", "ization"]
"unbelievable" → ["un", "believ", "able"]
```

- ✅ Balancierte Vokabulargröße (~50k), zerlegt unbekannte Wörter aus bekannten Zeichen
- ❌ Unbekannte Zeichen werden mit `ValueError` abgelehnt
- ❌ Tokenisierung nicht immer intuitiv

## BPE-Algorithmus

Der Algorithmus hinter GPT-2/3/4:

1. Starte mit allen Zeichen als Tokens
2. Finde das häufigste benachbarte Paar
3. Merge es zu einem neuen Token
4. Wiederhole bis gewünschte Vokabulargröße erreicht

```
"l" + "o" → "lo" → "lo" + "w" → "low" → ...
```

## Wichtige Erkenntnisse

### Spracheffizienz

BPE ist sprachunabhängig im Design, aber abhängig in der Effizienz:

| Sprache    | Tokens/Wort | Grund                    |
| ---------- | ----------- | ------------------------ |
| Englisch   | ~1.3        | Meiste Trainingsdaten    |
| Deutsch    | ~1.8        | Weniger Daten, Komposita |
| Chinesisch | ~2–3        | Zeichen-basiert          |
| Arabisch   | ~3–5        | Wenig Trainingsdaten     |

Deutscher Text kostet ~30–40% mehr Tokens als englischer.

### Whitespace, Zahlen, Code

```
" hello" ≠ "hello"      # Leerzeichen ist Teil des Tokens
"123456789" → 9 Tokens  # Jede Ziffer einzeln
"def " → 1 Token        # Häufige Code-Patterns zusammengefasst
```

### Groß-/Kleinschreibung

```
"Hello" → 1 Token       "HELLO" → 2-3 Tokens (selten, wird zerlegt)
"hello" → 1 Token       "HeLLo" → 3-4 Tokens
```

### Interpunktion

```
"Hello!" → ["Hello", "!"]    "..." → ["..."]    "don't" → ["don", "'t"]
```

### Rechtschreibfehler

Der Tokenizer korrigiert nichts – unbekannte Wörter werden zerlegt:

```
"Hallo"   → 1 Token          "Halllo" → ["Hall", "lo"]
```

Funktioniert trotzdem: Subword-Teile bleiben erkennbar, LLMs trainieren auf verrauschten Daten.

## Tokenizer in der Praxis

Jedes LLM hat seinen eigenen Tokenizer – es gibt keinen Standard.

| Modell-Familie | Tokenizer                | Vokabular |
| -------------- | ------------------------ | --------- |
| GPT-4/4o       | tiktoken (cl100k, o200k) | 100k–200k |
| Llama 2/3      | SentencePiece            | 32k       |
| Mistral        | SentencePiece            | 32k       |
| Claude         | Eigener (BPE-basiert)    | ~100k     |
| Gemini         | SentencePiece            | ~256k     |

**Warum?** Der Tokenizer wird vor dem Training erstellt. Das Modell lernt Embeddings für genau dieses Vokabular – ein anderer Tokenizer macht es unbrauchbar.

**Konsequenz:** Für Token-Zählung den richtigen Tokenizer verwenden!

## Lektionen & Ausführen

| Datei                    | Inhalt                   | Befehl                     |
| ------------------------ | ------------------------ | -------------------------- |
| `basic_tokenizers.py`    | Character- & Word-Level  | `uv run tokenize-basic`    |
| `bpe_tokenizer.py`       | BPE selbst implementiert | `uv run tokenize-bpe`      |
| `tiktoken_comparison.py` | OpenAIs GPT-Tokenizer    | `uv run tokenize-tiktoken` |

## Ressourcen

- [Andrej Karpathy: Let's build the GPT Tokenizer](https://www.youtube.com/watch?v=zduSFxRajkE)
- [OpenAI Tokenizer Tool](https://platform.openai.com/tokenizer)
- [Hugging Face Tokenizers](https://huggingface.co/docs/tokenizers)
