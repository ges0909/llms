# Agents Instructions

This is a Python learning project for understanding LLM concepts hands-on.

## Project Structure

```
src/
├── tokenization/   # Tokenization concepts
├── embeddings/     # (planned) Word embeddings
├── attention/      # (planned) Attention mechanism
└── transformer/    # (planned) Transformer architecture
```

Each topic has its own directory with:

- Python files demonstrating concepts
- A `README.md` explaining the theory

## Code Style

- **Language:** Code, variables, and file names in English
- **Documentation:** German docstrings and comments (for learning purposes)
- **Type hints:** Use Python 3.12+ type hints
- **Formatting:** Follow ruff defaults (line-length 100)

## Running Code

```bash
# Via script commands (recommended)
uv run tokenize-basic
uv run tokenize-bpe
uv run tokenize-tiktoken

# Or directly
uv run python src/tokenization/basic_tokenizers.py
```

## Adding New Topics

1. Create directory: `src/<topic>/`
2. Add `__init__.py`
3. Add `README.md` with theory explanation
4. Add Python files with working examples

## Dependencies

Managed via `uv`. Add new dependencies with:

```bash
uv add <package>
```

## Guidelines for Code

- Each file should be runnable standalone (`if __name__ == "__main__"`)
- Include clear print output showing what's happening
- Build concepts from scratch before using libraries
- Compare self-built implementations with real libraries

## Consistency Rules

- When updating a topic's `README.md`, check if Python files need matching updates
- When updating Python files, check if the `README.md` reflects the changes
- Keep docstrings and README content aligned but not duplicated
