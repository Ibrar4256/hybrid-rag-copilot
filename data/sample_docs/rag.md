# Retrieval-Augmented Generation

## What is RAG?

Retrieval-Augmented Generation (RAG) is a technique that combines a large language model
with an external retrieval system. Instead of relying solely on knowledge baked into the
model's parameters during training, the system retrieves relevant documents or passages
from a corpus at query time and provides them to the model as context.

## Why RAG matters

Large language models can hallucinate facts, especially about information outside their
training data or events after their training cutoff. RAG reduces hallucination by grounding
the model's answer in retrieved evidence, and it allows updating the knowledge base without
retraining the model.

## Hybrid retrieval

Modern RAG systems often combine dense vector search, which captures semantic similarity,
with sparse keyword search such as BM25, which captures exact term matches. Combining both
with a fusion method like Reciprocal Rank Fusion (RRF) tends to outperform either method
alone, because dense and sparse retrieval fail in different, complementary ways.

## Reranking

After an initial retrieval step returns a broad set of candidate passages, a reranker such
as a cross-encoder can re-score each candidate against the query with much higher precision
than the original retrieval score, since it processes the query and passage jointly instead
of comparing precomputed embeddings.
