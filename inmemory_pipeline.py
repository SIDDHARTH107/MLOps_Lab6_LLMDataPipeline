"""
Lab 1: In-Memory LLM Data Pipeline - Fintech App Reviews
Goal:
    This builds a basic data pipeline that:
    1. Loads the FULL app review dataset into RAM
    2. Tokenizes all reviews using GPT-2 tokenizer
    3. Concatenates all tokens and chunks into fixed-length blocks (128 tokens)
    4. Wraps in a PyTorch DataLoader ready for LLM training

This is one of the simplest approach. It works for small datasets but
does not scale to millions of reviews (runs out of memory).

Dataset: sealuzh/app_reviews (Google Play app reviews)
"""

from datasets import load_dataset
from transformers import AutoTokenizer
from torch.utils.data import DataLoader
import torch

# 1. Load the full dataset into memory
# This downloads ALL reviews at once. Fine for ~10K reviews,
# but would crash if we had millions.
print("Loading dataset...")
dataset = load_dataset("sealuzh/app_reviews", split="train")
print(f"Total reviews loaded: {len(dataset)}")
print(f"Columns: {dataset.column_names}")
print(f"Sample: {dataset[0]}")

# 2. Initialize GPT-2 tokenizer
# GPT-2 has no pad token by default, so we set pad = eos.
tokenizer = AutoTokenizer.from_pretrained("gpt2")
tokenizer.pad_token = tokenizer.eos_token

# 3. Tokenize all reviews
# .map() with batched=True processes chunks of rows at once (faster).
# We use the "review" column which contains the review text.
def tokenize_function(examples):
    return tokenizer(examples["review"], return_special_tokens_mask=False)

print("Tokenizing...")
tokenized_ds = dataset.map(tokenize_function, batched=True, remove_columns=dataset.column_names)
print(f"Tokenized dataset columns: {tokenized_ds.column_names}")
print(f"First review token IDs (first 20): {tokenized_ds[0]['input_ids'][:20]}")

# 4. Concatenate all tokens → chunk into fixed-length blocks
# WHY: Individual reviews are different lengths (3 words to 300 words).
#      LLM training needs UNIFORM blocks of tokens.
#      So we flatten ALL tokens into one long sequence, then slice
#      into blocks of exactly 128 tokens.
# Example: Review1=[5 tokens] + Review2=[200 tokens] + Review3=[10 tokens]
#          = [215 tokens total] → Block1=[128] + Block2=[128] (drop leftover)

block_size = 128

def group_texts(examples):
    # Flatten all token lists into one long list
    concatenated_ids = sum(examples["input_ids"], [])
    concatenated_masks = sum(examples["attention_mask"], [])

    # Drop the remainder that doesn't fill a full block
    total_length = (len(concatenated_ids) // block_size) * block_size
    concatenated_ids = concatenated_ids[:total_length]
    concatenated_masks = concatenated_masks[:total_length]

    # Slice into blocks of block_size
    result_ids = [concatenated_ids[i : i + block_size] for i in range(0, total_length, block_size)]
    result_masks = [concatenated_masks[i : i + block_size] for i in range(0, total_length, block_size)]

    return {"input_ids": result_ids, "attention_mask": result_masks}

print("Grouping into fixed-length blocks...")
lm_dataset = tokenized_ds.map(group_texts, batched=True, batch_size=1000)
print(f"Total training sequences: {len(lm_dataset)} blocks of {block_size} tokens each")

# 5. DataLoader with collate function
# For causal LM training: labels = input_ids
# The model internally shifts them to predict next token.
def collate_fn(batch):
    input_ids = torch.tensor([ex["input_ids"] for ex in batch], dtype=torch.long)
    attention_mask = torch.tensor([ex["attention_mask"] for ex in batch], dtype=torch.long)
    return {"input_ids": input_ids, "attention_mask": attention_mask, "labels": input_ids.clone()}

train_loader = DataLoader(lm_dataset, batch_size=8, shuffle=True, collate_fn=collate_fn)

# 6. Verify — iterate a few batches
print("\nSample batches:")
for i, batch in enumerate(train_loader):
    print(f"  Batch {i} → input_ids: {batch['input_ids'].shape}, labels: {batch['labels'].shape}")
    if i == 2:
        break

print("\n✅ In-memory pipeline ready for training!")
