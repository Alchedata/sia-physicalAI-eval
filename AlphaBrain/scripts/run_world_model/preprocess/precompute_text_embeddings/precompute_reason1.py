"""
Precompute Cosmos-Reason1 28-layer concat text embeddings for LIBERO instructions.

This matches NVIDIA Cosmos-Predict2.5 official text conditioning:
  - Reason1 (Qwen2.5-VL-7B) text-only forward with output_hidden_states=True
  - Take hidden_states[1:] (28 transformer layer outputs, skip input embed at idx 0)
  - Per-layer LayerNorm (elementwise_affine=False, i.e. F.layer_norm without weights)
  - Concat along last dim: [B, 512, 3584*28] == [B, 512, 100352]
  - This is the raw pre-projection tensor; the model applies a trainable
    Linear(100352, 1024) -> [B, 512, 1024] to feed DiT cross-attention.

Output file: data/pretrained_models/text_embeddings/reason1_28layer_text_embeddings.pkl
  dict[instruction_str -> torch.Tensor shape [512, 100352] fp16]

SIZE WARNING: Each fp16 embedding = 512 * 100352 * 2 bytes ~= 103 MB.
For N instructions the pickle is ~N * 103 MB. Recommended: deduplicate.

Usage (from project root):
    python scripts/run_world_model/preprocess/precompute_text_embeddings/precompute_reason1.py

Override paths via env vars:
    REASON1_PATH=... DATA_ROOT=... OUTPUT_PATH=... python ...
"""
import os
import pickle
import torch
import torch.nn.functional as F
import json

REASON1_PATH = os.environ.get(
    "REASON1_PATH",
    "data/pretrained_models/Cosmos-Reason1-7B",
)
OUTPUT_DIR = os.environ.get(
    "OUTPUT_DIR",
    "data/pretrained_models/text_embeddings",
)
OUTPUT_PATH = os.environ.get(
    "OUTPUT_PATH",
    os.path.join(OUTPUT_DIR, "reason1_28layer_text_embeddings.pkl"),
)
DATA_ROOT = os.environ.get("DATA_ROOT", "data/datasets/libero_datasets")
MAX_LEN = 512
HIDDEN_SIZE = 3584
NUM_LAYERS = 28  # NVIDIA uses 28 layers concat (100352 / 3584 = 28)

os.makedirs(OUTPUT_DIR, exist_ok=True)

# ---------------------------------------------------------------------------
# Collect all unique LIBERO instructions
# ---------------------------------------------------------------------------
instructions = set()
for suite_dir in os.listdir(DATA_ROOT):
    tasks_file = os.path.join(DATA_ROOT, suite_dir, "meta", "tasks.jsonl")
    if os.path.isfile(tasks_file):
        with open(tasks_file) as f:
            for line in f:
                task = json.loads(line)
                if "task" in task:
                    instructions.add(task["task"])

instructions = sorted(instructions)
print(f"Found {len(instructions)} unique instructions")
for i, inst in enumerate(instructions[:5]):
    print(f"  [{i}] {inst}")
print("  ...")

# Size estimate
bytes_per = MAX_LEN * HIDDEN_SIZE * NUM_LAYERS * 2  # fp16
total_gb = len(instructions) * bytes_per / 1e9
print(f"\nEstimated pickle size: {total_gb:.2f} GB "
      f"({bytes_per / 1e6:.1f} MB/instruction * {len(instructions)} instructions)")
if total_gb > 10:
    print(f"WARNING: pickle will exceed 10GB. Consider fp8 or sharding by suite.")

# ---------------------------------------------------------------------------
# Load Reason1 (Qwen2.5-VL-7B)
# ---------------------------------------------------------------------------
print(f"\nLoading Cosmos-Reason1 from {REASON1_PATH}...")
from transformers import Qwen2_5_VLForConditionalGeneration, AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained(REASON1_PATH)
model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
    REASON1_PATH,
    torch_dtype=torch.bfloat16,
    attn_implementation="eager",
)
model = model.eval().cuda()
print(f"Reason1 loaded on GPU, {sum(p.numel() for p in model.parameters())/1e6:.0f}M params")

# ---------------------------------------------------------------------------
# Precompute 28-layer LN+concat embeddings
# ---------------------------------------------------------------------------
embeddings = {}

print(f"\nPrecomputing 28-layer concat embeddings for {len(instructions)} instructions...")
with torch.no_grad():
    for i, instruction in enumerate(instructions):
        tokens = tokenizer(
            instruction,
            return_tensors="pt",
            padding="max_length",
            truncation=True,
            max_length=MAX_LEN,
        ).to("cuda")

        # Text-only forward through the underlying LM (bypasses vision tower)
        outputs = model.model(
            input_ids=tokens.input_ids,
            attention_mask=tokens.attention_mask,
            output_hidden_states=True,
        )

        # hidden_states: tuple length = num_hidden_layers + 1
        # Index 0 = input embeddings; index 1..N = output of each transformer layer.
        # NVIDIA uses the 28 transformer layer outputs: hidden_states[1:29]
        hs = outputs.hidden_states
        assert len(hs) >= NUM_LAYERS + 1, (
            f"Reason1 produced only {len(hs)} hidden states; need >= {NUM_LAYERS + 1}"
        )
        layers = hs[1:1 + NUM_LAYERS]  # tuple of 28 tensors, each [1, 512, 3584]

        # Per-layer LayerNorm with elementwise_affine=False (no learnable weight/bias).
        # F.layer_norm(x, normalized_shape=[D]) computes over last dim only.
        normed = [
            F.layer_norm(h.float(), normalized_shape=[HIDDEN_SIZE])
            for h in layers
        ]  # each [1, 512, 3584]

        # Concat along feature dim -> [1, 512, 28*3584] = [1, 512, 100352]
        concat = torch.cat(normed, dim=-1).squeeze(0)  # [512, 100352]

        # Zero out padding positions (keeps cross-attn mask-agnostic-safe)
        attn_mask = tokens.attention_mask[0]
        length = int(attn_mask.sum().item())
        if length < MAX_LEN:
            concat[length:] = 0

        embeddings[instruction] = concat.cpu().to(torch.float16)

        if (i + 1) % 10 == 0 or i == 0:
            print(
                f"  [{i+1}/{len(instructions)}] {instruction[:60]}... "
                f"shape={tuple(concat.shape)}"
            )

# ---------------------------------------------------------------------------
# Save
# ---------------------------------------------------------------------------
print(f"\nSaving to {OUTPUT_PATH}...")
with open(OUTPUT_PATH, "wb") as f:
    pickle.dump(embeddings, f, protocol=pickle.HIGHEST_PROTOCOL)

file_size_gb = os.path.getsize(OUTPUT_PATH) / 1e9
print(f"Done! {len(embeddings)} embeddings saved, file size: {file_size_gb:.2f} GB")
print(f"Each embedding shape: [512, 100352] ({NUM_LAYERS}x{HIDDEN_SIZE}), dtype: float16")
