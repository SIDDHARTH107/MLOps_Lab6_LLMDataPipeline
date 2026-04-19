"""
Streaming Multi-Process Sharded Pipeline - Fintech App Reviews
Goal:
    Build a PRODUCTION-SCALE pipeline that:
    1. Streams reviews without loading everything into RAM
    2. SHARDS the data across multiple CPU processes (parallel processing)
    3. Each worker tokenizes its shard independently
    4. Uses rolling buffer to produce fixed-length LM training blocks
    5. All workers run SIMULTANEOUSLY — visible via interleaved timestamps

WHY MULTI-PROCESS:
    Lab 2 streams data efficiently but uses ONE CPU core.
    Real fintech companies process millions of reviews.
    To speed up data prep, we split the stream across 4 workers:
    
    Worker 0 gets reviews 0, 4, 8, 12, ...
    Worker 1 gets reviews 1, 5, 9, 13, ...
    Worker 2 gets reviews 2, 6, 10, 14, ...
    Worker 3 gets reviews 3, 7, 11, 15, ...
    
    Each worker independently tokenizes + buffers + chunks its shard.
    Total throughput = 4x a single process.

SHARDING STRATEGY:
    We use round-robin sharding (review index % num_workers).
    In a production system, you might shard by APP NAME instead:
      Worker 0 → Robinhood reviews
      Worker 1 → Cash App reviews
      Worker 2 → Venmo reviews
      Worker 3 → Chime reviews

Dataset: sealuzh/app_reviews (streamed from HuggingFace)

Usage:
    python streaming_shard_reviews.py
"""

import os
import sys
import time
import torch
from datetime import datetime
from torch.utils.data import IterableDataset, DataLoader
from datasets import load_dataset
from transformers import AutoTokenizer
import multiprocessing as mp

# Rolling buffer - produces fixed-length token blocks
def rolling_token_blocks(token_iter, block_size, pad_token_id):
    """
    Collects tokens into a buffer, yields blocks of exactly block_size.
    Pads the final leftover block if the stream ends mid-block.
    """
    buffer = []
    for tokens in token_iter:
        buffer.extend(tokens)
        while len(buffer) >= block_size:
            chunk = buffer[:block_size]
            buffer = buffer[block_size:]
            yield {
                "input_ids": torch.tensor(chunk, dtype=torch.long),
                "attention_mask": torch.ones(block_size, dtype=torch.long),
            }
    # Pad leftover tokens
    if buffer:
        pad_len = block_size - len(buffer)
        yield {
            "input_ids": torch.tensor(buffer + [pad_token_id] * pad_len, dtype=torch.long),
            "attention_mask": torch.tensor([1] * len(buffer) + [0] * pad_len, dtype=torch.long),
        }

# Manual sharding - each worker gets every Nth example
def manual_shard(dataset_iter, num_shards, process_index):
    """
    Round-robin sharding: worker K gets examples K, K+num_shards, K+2*num_shards, ...
    This ensures every review is processed by EXACTLY one worker.
    """
    for idx, example in enumerate(dataset_iter):
        if idx % num_shards == process_index:
            yield example

# IterableDataset wrapper - combines tokenization + rolling buffer
class LMStreamingDataset(IterableDataset):
    def __init__(self, dataset_iter, tokenizer, block_size):
        self.dataset_iter = dataset_iter
        self.tokenizer = tokenizer
        self.block_size = block_size

    def __iter__(self):
        # Tokenize each review's text on-the-fly, then feed into rolling buffer
        token_stream = (
            self.tokenizer(ex["review"], add_special_tokens=False)["input_ids"]
            for ex in self.dataset_iter
            if ex.get("review") and len(str(ex["review"]).strip()) > 0  # skip empty reviews
        )
        yield from rolling_token_blocks(token_stream, self.block_size, self.tokenizer.pad_token_id)

# Collate function
def collate_fn(batch):
    return {
        "input_ids": torch.stack([ex["input_ids"] for ex in batch]),
        "attention_mask": torch.stack([ex["attention_mask"] for ex in batch]),
    }

# Worker entry - each process runs this independently
def worker_entry(rank, world_size, model_name, block_size, batch_size, batches_to_show):
    """
    Each worker:
    1. Loads its own copy of the streaming dataset
    2. Takes only its shard (every Nth review)
    3. Tokenizes + chunks independently
    4. Prints batch info with timestamps (to show parallelism)
    """
    # Each worker loads the stream independently
    stream_ds = load_dataset("sealuzh/app_reviews", split="train", streaming=True)
    sharded_iter = manual_shard(stream_ds, world_size, rank)

    # Each worker has its own tokenizer instance
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    tokenizer.pad_token = tokenizer.eos_token

    # Build the pipeline: shard → tokenize → rolling buffer → blocks
    lm_dataset = LMStreamingDataset(sharded_iter, tokenizer, block_size)
    loader = DataLoader(lm_dataset, batch_size=batch_size, collate_fn=collate_fn, num_workers=0)

    print(f"{datetime.now()} [PID {os.getpid()} | Worker {rank}] Starting...", flush=True)
    for i, batch in enumerate(loader):
        print(
            f"{datetime.now()} [Worker {rank}] Batch {i} → "
            f"input_ids shape: {batch['input_ids'].shape}",
            flush=True,
        )
        time.sleep(0.3)  # Slow down so parallel execution is visible in logs
        if i + 1 >= batches_to_show:
            break
    print(f"{datetime.now()} [Worker {rank}] Done.", flush=True)

# Launcher - spawns N parallel processes
def launch_multi_proc(num_procs, model_name, block_size, batch_size, batches_to_show):
    ctx = mp.get_context("spawn")  # "spawn" is safe on all platforms (Windows, Mac, Linux)
    procs = []
    for rank in range(num_procs):
        p = ctx.Process(
            target=worker_entry,
            args=(rank, num_procs, model_name, block_size, batch_size, batches_to_show),
        )
        p.start()
        procs.append(p)
    for p in procs:
        p.join()

# Main
if __name__ == "__main__":
    print("=" * 65)
    print("Fintech App Review — Multi-Process Streaming LLM Data Pipeline")
    print("=" * 65)
    print(f"Workers: 4 | Model: GPT-2 | Block size: 128 | Batch size: 4")
    print(f"Each worker processes 1/4 of the review stream in parallel.\n")

    launch_multi_proc(
        num_procs=4,          # 4 parallel CPU workers
        model_name="gpt2",    # GPT-2 tokenizer
        block_size=128,       # Each training sample = 128 tokens
        batch_size=4,         # 4 samples per batch per worker
        batches_to_show=3,    # Show 3 batches per worker then stop
    )

print("\n✅ Multi-process pipeline complete")
print("All 4 workers processed their shards in parallel!")
