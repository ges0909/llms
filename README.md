# LLM Learning

Hands-on learning of LLM concepts with Python.

## Topics

| Module         | Topic                      | Status  |
| -------------- | -------------------------- | ------- |
| `tokenization` | Understanding tokenization | done    |
| `embeddings`   | Word embeddings & vectors  | planned |
| `attention`    | Attention mechanism        | planned |
| `transformer`  | Transformer architecture   | planned |

## Setup

```bash
uv sync              # Nur Runtime-Dependencies
uv sync --group dev  # Mit Ruff & Pyrefly
```

## Running Lessons

```bash
uv run tokenize-basic    # Character & Word Tokenizer
uv run tokenize-bpe      # BPE Tokenizer
uv run tokenize-tiktoken # OpenAI tiktoken Comparison
```

Alternativ direkt:

```bash
uv run python src/tokenization/basic_tokenizers.py
```

## Structure

```
src/
├── tokenization/
├── embeddings/
├── attention/
└── transformer/
```

## Code Quality

```bash
uv run ruff check src/   # Linting
uv run ruff format src/  # Formatting
uv run pyrefly check     # Type Checking
```
