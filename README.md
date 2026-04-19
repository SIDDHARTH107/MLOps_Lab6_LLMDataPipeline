# Fintech App Review — LLM Data Pipeline

## Problem Statement
US fintech apps (Robinhood, Cash App, Venmo, Chime, etc.) receive thousands of user reviews daily. To train an LLM that can automatically classify reviews, extract feature requests, and detect sentiment, we need a scalable data pipeline that can process millions of reviews efficiently.

This project demonstrates three progressively complex approaches to building an LLM data pipeline for fintech app review text:

1. **In-Memory Pipeline** — Load all data into RAM, tokenize, chunk, and batch.
2. **Streaming Pipeline** — Process data without loading it all into memory using a rolling buffer.
3. **Multi-Process Streaming Pipeline** — Shard the data stream across multiple CPU workers for parallel processing.

## Dataset
- **`sealuzh/app_reviews`** from HuggingFace — Google Play app reviews with star ratings and review text.

## Files

| File | Description | Lab Concept |
|------|-------------|-------------|
| `inmemory_pipeline.py` | Full dataset in RAM → tokenize → concat → chunk → DataLoader | Baseline approach |
| `streaming_pipeline.py` | Streaming dataset → rolling buffer → fixed blocks → DataLoader | O(1) memory |
| `streaming_shard_reviews.py` | Multi-process streaming with per-app sharding | Parallel data prep |

## How to Run

```bash
pip install datasets transformers torch

# Step 1: In-memory pipeline
python inmemory_pipeline.py

# Step 2: Streaming pipeline
python streaming_pipeline.py

# Step 3: Multi-process sharded pipeline
python streaming_shard_reviews.py
```

## Key Concepts Demonstrated
- **Streaming**: Process web-scale data without loading into RAM
- **Rolling Buffer**: Concatenate variable-length reviews into uniform token blocks
- **Multi-Process Sharding**: Split data across CPU workers for parallel tokenization
- **Causal LM Data Prep**: Labels = input_ids (next-token prediction)

## Author
Siddharth Mohapatra | Northeastern University | IE 7374 MLOps
