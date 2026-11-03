# Agents Instructions

This is a Python learning project for understanding LLM concepts hands-on.

The goal is to make modern language model internals understandable through small, runnable examples. The project focuses on building intuition by combining theory, implementation, and comparison against real libraries.

## Project rules

### Caveman communication rule

Apply the Caveman style to responses and explanations:

- Answer first. No greeting, no fluff, no recap.
- One idea per sentence. Keep it short and direct.
- Preserve code, commands, filenames, and paths verbatim.
- Do not rewrite the user's prompt or add unnecessary abstraction.
- Prefer the smallest correct answer, then the next step.
- If a change is needed, explain what changed and why in 1-3 sentences.
- When code is involved, keep the technical meaning exact and avoid filler.

### Ponytail engineering rule

Apply the Ponytail style to implementation work:

- Do the minimum that solves the problem.
- Reuse what's already in the codebase before adding new abstractions.
- Read the touched code and the real data flow before patching.
- Prefer the simplest correct solution over cleverness.
- Do not refactor broadly for style alone.
- Keep validation, error handling, and correctness intact.
- If logic has branches, parsing, data loss risk, or security implications, add a small validation or test.
- Explain the change in plain language, without extra ceremony.

When communication and engineering rules conflict, correctness wins first, then simplicity, then brevity.

The project combines both: speak like Caveman, code like Ponytail.

## Learning Focus

This repository covers the foundational concepts behind modern LLMs:

- Tokenization: converting text into model-readable units
- Embeddings: mapping tokens to vectors
- Positional encoding: preserving sequence order
- Attention: identifying which parts of a sequence matter
- Transformer architecture: composing the full model stack
- Training objective: predicting the next token
- Decoding and inference: generating text from a trained model

The project starts with tokenization because it is the most practical and accessible entry point into LLM internals.

## Project Structure

```text
src/
├── tokenization/         # Tokenization concepts
├── embeddings/           # Word embeddings and vector representations
├── positional-encoding/   # Sequence order information
├── attention/            # Attention mechanism
├── transformer/          # Transformer architecture
├── training/             # Next-token prediction and training objective
├── inference/            # Decoding and generation strategies
├── applications/         # RAG, prompting and practical LLM usage
└── README.md
```

Each topic should contain:

- Python files demonstrating the concept
- A README.md explaining the theory
- runnable examples with clear output
- a practical comparison to real-world implementations when useful

## Code Style

- Language: code, variables, and filenames in English
- Documentation: German docstrings and comments for learning purposes
- Type hints: use Python 3.12+ syntax
- Formatting: follow Ruff defaults and keep line length around 100

## Running Code

```bash
# preferred via script commands
uv run tokenize-basic
uv run tokenize-bpe
uv run tokenize-tiktoken

# or directly
uv run python src/tokenization/basic_tokenizers.py
```

## Adding New Topics

1. Create a new directory under `src/<topic>/`
2. Add an `__init__.py`
3. Add a `README.md` with a theory explanation
4. Add executable Python examples
5. Update the top-level README if the learning roadmap changes

## Dependencies

Dependencies are managed with `uv`.

```bash
uv add <package>
```

## Guidelines for Code

- Each file should be runnable standalone via `if __name__ == "__main__":`
- Include clear print output showing what is happening
- Build concepts from scratch before using external libraries
- Prefer educational clarity over production abstraction
- Compare handcrafted implementations with real libraries when relevant
- Keep examples small and understandable

## Consistency Rules

- When changing a topic's README.md, check whether related Python files also need updates
- When updating Python files, ensure the README still reflects the content accurately
- Keep docstrings and README content aligned but not duplicated
- Keep the project documentation consistent with the top-level README and the current roadmap

## Recommended Roadmap

1. Tokenization
2. Embeddings
3. Positional encoding
4. Attention
5. Transformer block
6. Training objective
7. Decoding and inference
8. Fine-tuning and alignment
9. RAG and practical applications

This order keeps the learning path incremental: from the simplest data transformation to the full architecture and practical usage of LLMs.
