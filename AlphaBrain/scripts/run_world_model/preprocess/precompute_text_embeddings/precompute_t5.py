"""
Precompute Cosmos-Predict2 T5-XXL text embeddings for all LIBERO instructions.

Output: {instruction_str: tensor[512, 1024]} saved as pickle.
- T5-XXL (1024 d_model) outputs last_hidden_state -> [512, 1024]
- Zero-mask padding positions (after attention_mask end)
- Save as float16 to reduce file size

Usage (from project root):
    python scripts/run_world_model/preprocess/precompute_text_embeddings/precompute_t5.py

Override paths via env vars:
    TEXT_ENCODER_DIR=... TOKENIZER_DIR=... DATA_ROOT=... OUTPUT_PATH=... python ...
"""
import os
import pickle
import torch
import json

TEXT_ENCODER_PATH = os.environ.get(
    "TEXT_ENCODER_DIR",
    "data/pretrained_models/Cosmos-Predict2-2B-Video2World/text_encoder",
)
TOKENIZER_PATH = os.environ.get(
    "TOKENIZER_DIR",
    "data/pretrained_models/Cosmos-Predict2-2B-Video2World/tokenizer",
)
DATA_ROOT = os.environ.get("DATA_ROOT", "data/datasets/libero_datasets")
OUTPUT_PATH = os.environ.get("OUTPUT_PATH", os.path.join(DATA_ROOT, "t5_text_embeddings.pkl"))

# Collect all unique LIBERO instructions
instructions = set()
for suite_dir in os.listdir(DATA_ROOT):
    tasks_file = os.path.join(DATA_ROOT, suite_dir, "meta", "tasks.jsonl")
    if os.path.isfile(tasks_file):
        with open(tasks_file) as f:
            for line in f:
                task = json.loads(line.strip())
                if "task" in task:
                    instructions.add(task["task"])

instructions = sorted(instructions)
print(f"Found {len(instructions)} unique instructions")
for i, inst in enumerate(instructions[:5]):
    print(f"  [{i}] {inst}")
print("  ...")

# Load T5EncoderModel
print(f"\nLoading T5 text encoder from {TEXT_ENCODER_PATH}...")
from transformers import T5EncoderModel, T5Tokenizer

tokenizer = T5Tokenizer.from_pretrained(TOKENIZER_PATH)
model = T5EncoderModel.from_pretrained(
    TEXT_ENCODER_PATH,
    torch_dtype=torch.bfloat16,
)
model = model.eval().cuda()
model.requires_grad_(False)

d_model = model.config.d_model
print(f"T5 loaded on GPU, d_model={d_model}, {sum(p.numel() for p in model.parameters())/1e6:.0f}M params")

# Precompute embeddings
embeddings = {}

print(f"\nPrecomputing embeddings for {len(instructions)} instructions...")
with torch.no_grad():
    for i, instruction in enumerate(instructions):
        tokens = tokenizer(
            instruction,
            return_tensors="pt",
            padding="max_length",
            max_length=512,
            truncation=True,
        ).to("cuda")

        outputs = model(
            input_ids=tokens.input_ids,
            attention_mask=tokens.attention_mask,
        )

        emb = outputs.last_hidden_state[0]  # [512, 1024]

        # Zero-mask positions beyond attention mask
        attn_mask = tokens.attention_mask[0]  # [512]
        length = int(attn_mask.sum().item())
        emb[length:] = 0.0

        embeddings[instruction] = emb.cpu().to(torch.float16)

        if (i + 1) % 10 == 0 or i == 0:
            print(f"  [{i+1}/{len(instructions)}] {instruction[:70]}  shape={emb.shape}")

# Save
os.makedirs(os.path.dirname(OUTPUT_PATH) or ".", exist_ok=True)
print(f"\nSaving to {OUTPUT_PATH}...")
with open(OUTPUT_PATH, "wb") as f:
    pickle.dump(embeddings, f)

file_size = os.path.getsize(OUTPUT_PATH) / 1e6
print("\nDone!")
print(f"  {len(embeddings)} instructions saved")
print(f"  File size: {file_size:.1f} MB")
print(f"  Each embedding shape: [512, {d_model}], dtype: float16")
