# LLM Learning

Hands-on learning of LLM concepts with Python.

This project makes modern language model internals understandable through small, runnable Python examples. It builds intuition by connecting theory, from-scratch implementations, and comparisons with real libraries. The learning path starts with tokenization, the step that turns text into model-readable units.

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

| Module                  | Topic                              | Status  |
| ----------------------- | ---------------------------------- | ------- |
| `topic_01_tokenization` | Understanding tokenization         | done    |
| `topic_02_embeddings`   | Word embeddings and vectors        | done    |
| `positional-encoding`   | Order information in sequences     | planned |
| `attention`             | Attention mechanism                | planned |
| `transformer`           | Transformer architecture           | planned |
| `training`              | Next-token prediction and training | planned |
| `inference`             | Decoding and generation            | planned |
| `applications`          | RAG and practical LLM usage        | planned |

## Setup

```bash
uv sync --no-dev     # runtime dependencies only
uv sync              # includes the default dev group (Ruff, Pyrefly, and pytest)
```

## Running lessons

```bash
# Topic 01: Tokenization
uv run tokenize-basic    # character and word tokenizers
uv run tokenize-bpe      # BPE tokenizer
uv run tokenize-tiktoken # OpenAI tiktoken comparison

# Topic 02: Embeddings
uv run embed-lookup      # One-Hot vs. Dense Embedding lookup
uv run embed-similarity  # Cosine-Similarity and vector analogies
uv run embed-transformer # Transformer embeddings and scaling
uv run embed-training    # Train embeddings through next-token prediction
```

Alternatively, run the scripts directly:

```bash
uv run python src/topic_01_tokenization/basic_tokenizers.py
```

## Tests

```bash
uv run pytest
```

## Code quality

```bash
uv run ruff check src/
uv run ruff format src/
uv run pyrefly check
```
