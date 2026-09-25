# Vector Databases

## What they store

A vector database stores high-dimensional embeddings produced by a machine learning model
and allows approximate nearest neighbor (ANN) search over them, typically using an index
structure like HNSW (Hierarchical Navigable Small World graphs).

## Qdrant

Qdrant is an open-source vector database written in Rust. It supports named vectors per
point, sparse vectors for BM25-style search, payload filtering, and native fusion queries
that combine dense and sparse results using methods like Reciprocal Rank Fusion.

## Scaling considerations

As a collection grows past tens of millions of vectors, teams typically introduce scalar or
product quantization to reduce memory footprint, and shard collections across multiple
nodes. These techniques trade a small amount of recall for large gains in memory efficiency
and query throughput.
