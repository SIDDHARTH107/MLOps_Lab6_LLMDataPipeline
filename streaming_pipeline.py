"""
Streaming LLM Data Pipeline — Fintech App Reviews
Goal:
    This builds a STREAMING pipeline that:
    1. Loads reviews ONE AT A TIME (never the full dataset in RAM)
    2. Tokenizes each review on-the-fly
    3. Uses a ROLLING BUFFER to collect tokens and chunk into 128-token blocks
    4. Wraps in a PyTorch DataLoader

WHY STREAMING:
    Lab 1 loaded everything into memory. That works for 10K reviews.
    But real fintech apps get MILLIONS of reviews. Robinhood alone has
    500K+ reviews on Google Play. We can't hold all of that in RAM.

    Streaming processes one review at a time — O(1) memory usage
    regardless of dataset size. This is how production LLM pipelines work.

THE ROLLING BUFFER:
    Reviews are different lengths. Some are 3 words, others are 300.
    We can't just pad each review to 128 tokens (wasteful).
    Instead, we use a rolling buffer:
    
    Buffer: []
    → Read Review1 (5 tokens) → Buffer: [t1,t2,t3,t4,t5]  (not enough for 128)
    → Read Review2 (200 tokens) → Buffer: [t1,...,t205]       (enough!)
    → Yield Block1: first 128 tokens → Buffer: [remaining 77 tokens]
    → Read Review3 (60 tokens) → Buffer: [77 + 60 = 137 tokens] (enough!)
    → Yield Block2: first 128 tokens → Buffer: [remaining 9 tokens]
    → ...and so on

Dataset: sealuzh/app_reviews (streamed from HuggingFace)
"""

from datasets import load_dataset
from transformers import AutoTokenizer
from torch.utils.data import IterableDataset, DataLoader
import torch

# 1. Load dataset in STREAMING mode
# streaming=True means: don't download everything.
# Instead, give me an iterator that fetches one example at a time.
print("Loading dataset in streaming mode...")
stream_dataset = load_dataset("sealuzh/app_reviews", split="train", streaming=True)

# 2. Initialize GPT-2 tokenizer
tokenizer = AutoTokenizer.from_pretrained("gpt2")
tokenizer.pad_token = tokenizer.eos_token

# 3. Tokenize lazily (on-the-fly)
# .map() on a streaming dataset doesn't process everything upfront.
# It creates a LAZY pipeline — tokenization happens only when
# we actually iterate over the data.
def tokenize_function(examples):
    return tokenizer(examples["review"])

tokenized_stream = stream_dataset.map(tokenize_function, batched=True)

# 4. Rolling buffer — the core of streaming LM data prep
block_size = 128

def rolling_buffer_chunk(dataset_iter, block_size):
    """
    Collects tokens from the stream into a buffer.
    Whenever the buffer has >= block_size tokens, yields a block.
    Leftover tokens carry over to the next iteration.
    
    Think of it like filling a glass from a faucet:
    - The faucet drips tokens (variable amounts per review)
    - When the glass is full (128 tokens), pour it out (yield a block)
    - Whatever overflowed stays in the glass for the next fill
    """
    buffer = []
    for example in dataset_iter:
        # Add this review's tokens to the buffer
        buffer.extend(example["input_ids"])
        
        # Keep yielding full blocks while we have enough tokens
        while len(buffer) >= block_size:
            chunk = buffer[:block_size]
            buffer = buffer[block_size:]
            yield {
                "input_ids": chunk,
                "attention_mask": [1] * block_size,
            }
    
    # OPTIONAL: handle leftover tokens at the end of the stream
    # We pad them so they form a complete block
    if buffer:
        pad_length = block_size - len(buffer)
        yield {
            "input_ids": buffer + [tokenizer.pad_token_id] * pad_length,
            "attention_mask": [1] * len(buffer) + [0] * pad_length,
        }

# 5. Wrap in PyTorch IterableDataset
# PyTorch's DataLoader needs an IterableDataset to work with
# generators/streams (as opposed to a regular map-style Dataset).
class StreamingLMDataset(IterableDataset):
    def __init__(self, hf_stream, block_size):
        self.stream = hf_stream
        self.block_size = block_size

    def __iter__(self):
        return rolling_buffer_chunk(self.stream, self.block_size)

streaming_lm_dataset = StreamingLMDataset(tokenized_stream, block_size)

# 6. Collate function for batching
def collate_fn(batch):
    input_ids = torch.tensor([ex["input_ids"] for ex in batch], dtype=torch.long)
    attention_mask = torch.tensor([ex["attention_mask"] for ex in batch], dtype=torch.long)
    return {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "labels": input_ids.clone(),
    }

# 7. DataLoader
train_loader = DataLoader(streaming_lm_dataset, batch_size=8, collate_fn=collate_fn)

# 8. Verify — iterate a few batches
print("\nStreaming batches (O(1) memory):")
for i, batch in enumerate(train_loader):
    print(f"  Batch {i} → input_ids: {batch['input_ids'].shape}")
    if i == 2:
        break

print("\n✅Streaming pipeline with rolling buffer ready!")
print("Memory usage stays constant regardless of dataset size.")
