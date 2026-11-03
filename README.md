# LLM Learning

Hands-on learning of LLM concepts with Python.

This project is designed to make the internals of modern language models understandable through small, runnable examples. Instead of only reading theory, the goal is to implement core ideas in Python, compare them with real libraries, and build intuition step by step.

## Project goal

Large language models are built from a small set of foundational ideas:

- Tokenization: turning text into model-readable units
- Embeddings: mapping tokens to vectors
- Positional encoding: preserving token order
- Attention: deciding which tokens matter at a given moment
- Transformer blocks: composing these ideas into a scalable architecture
- Training objective: predicting the next token
- Decoding: generating text during inference

The project starts with tokenization because it is the easiest point of entry and the most practical place to understand how text becomes numbers for a model.

## Core concepts covered

### 1. Tokenization

Models do not work with raw text. Before any reasoning happens, text is converted into tokens and then token IDs.

Examples:

- character-level tokenization
- word-level tokenization
- subword tokenization with BPE
- comparison with real tokenizers such as OpenAI's tiktoken

This is the first and essential step of any LLM pipeline.

### 2. Embeddings

Once a token is represented as an ID, the model maps it to a learned numerical vector. These vectors carry semantic information and represent the initial meaning of each token.

This is where language starts becoming geometry: similar words or concepts become nearby points in vector space.

### 3. Positional encoding

A transformer does not inherently know the order of tokens. Positional encoding gives each token information about its position in a sequence.

This is essential for understanding syntax, word order, and contextual meaning.

### 4. Attention

Attention is the central mechanism behind Transformers. It allows a model to focus on relevant parts of a sequence while processing each token.

Key ideas:

- self-attention
- query, key, value vectors
- multi-head attention
- context-dependent relevance

### 5. Transformer architecture

The Transformer combines embeddings, positional encoding, attention, and feed-forward layers in stacked blocks.

Once these components are understood, the overall structure of modern LLMs becomes much easier to reason about.

### 6. Training objective and generation

A model is usually trained to predict the next token. That objective is simple but powerful.

After training, the model generates text through decoding strategies like:

- greedy decoding
- sampling
- top-k / top-p filtering
- temperature tuning

### 7. Practical extensions

Beyond the core architecture, real-world LLM systems include additional important concepts:

- fine-tuning
- instruction tuning
- RLHF / alignment
- context windows
- retrieval-augmented generation (RAG)
- evaluation and benchmarking

These are not strictly required to understand the basics, but they are the next layer of understanding for practical LLM work.

## Topics

| Module                | Topic                              | Status  |
| --------------------- | ---------------------------------- | ------- |
| `tokenization`        | Understanding tokenization         | done    |
| `embeddings`          | Word embeddings and vectors        | planned |
| `positional-encoding` | Order information in sequences     | planned |
| `attention`           | Attention mechanism                | planned |
| `transformer`         | Transformer architecture           | planned |
| `training`            | Next-token prediction and training | planned |
| `inference`           | Decoding and generation            | planned |

## Recommended learning path

1. Tokenization
2. Embeddings
3. Positional encoding
4. Attention
5. Transformer block
6. Training objective
7. Decoding and inference
8. Fine-tuning and alignment
9. RAG and application patterns

This sequence moves from the simplest conceptual entry point to the full architecture and practical use cases.

## Setup

```bash
uv sync --no-dev     # runtime dependencies only
uv sync              # includes the default dev group (Ruff and Pyrefly)
```

## Running lessons

```bash
uv run tokenize-basic    # character and word tokenizers
uv run tokenize-bpe      # BPE tokenizer
uv run tokenize-tiktoken # OpenAI tiktoken comparison
```

Alternatively, run the scripts directly:

```bash
uv run python src/tokenization/basic_tokenizers.py
```

## Project structure

```text
src/
├── tokenization/
├── embeddings/
├── positional-encoding/
├── attention/
├── transformer/
├── training/
├── inference/
└── applications/
```

## Code quality

```bash
uv run ruff check src/
uv run ruff format src/
uv run pyrefly check
```

## Why this project matters

Modern LLMs can seem like black boxes because they are large, complex, and often hidden behind APIs. This project aims to make the internal logic visible and teachable by breaking it into understandable building blocks.

The goal is not to reimplement production-scale models, but to build intuition through small, transparent examples.

## Notes

This repository follows the idea that learning is strongest when theory and code are connected. Each module should use runnable Python examples and should be paired with an explanation of the underlying concept.
