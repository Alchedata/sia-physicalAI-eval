"""
Precompute UMT5-XXL text embeddings for all LIBERO instructions.
Output: data/datasets/libero_datasets/umt5_text_embeddings.pkl
Format: Dict[str, Tensor[512, 4096]], dtype=bfloat16

Usage (from project root):
    python scripts/run_world_model/preprocess/precompute_text_embeddings/precompute_umt5.py

Override paths via env vars:
    WAN_DIR=... DATA_ROOT=... OUTPUT_PATH=... python ...
"""

import os
import json
import pickle
import logging

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

import torch

DATASETS_DIR = os.environ.get("DATA_ROOT", "data/datasets/libero_datasets")
SPLITS = [
    "libero_spatial_no_noops_1.0.0_lerobot",
    "libero_object_no_noops_1.0.0_lerobot",
    "libero_goal_no_noops_1.0.0_lerobot",
    "libero_10_no_noops_1.0.0_lerobot",
    "libero_90_no_noops_lerobot",
]
OUTPUT_PKL = os.environ.get("OUTPUT_PATH", os.path.join(DATASETS_DIR, "umt5_text_embeddings.pkl"))

BASE_DIR = os.environ.get("WAN_DIR", "data/pretrained_models/Wan2.2-TI2V-5B")
T5_PTH = os.path.join(BASE_DIR, "models_t5_umt5-xxl-enc-bf16.pth")
TOK_DIR = os.path.join(BASE_DIR, "google", "umt5-xxl")
TEXT_LEN = 512
DIM = 4096

# -- 1. Collect all unique instructions -----------------------------------
all_instructions = set()
for split in SPLITS:
    meta = os.path.join(DATASETS_DIR, split, "meta", "tasks.jsonl")
    if os.path.exists(meta):
        with open(meta) as f:
            for line in f:
                task = json.loads(line.strip())
                all_instructions.add(task["task"])
    else:
        logger.warning("tasks.jsonl not found for split: %s", split)

all_instructions = sorted(all_instructions)
logger.info("Total unique instructions: %d", len(all_instructions))

# -- 2. Load UMT5-XXL -----------------------------------------------------
device = torch.device("cuda")
logger.info("Loading UMT5-XXL from %s ...", T5_PTH)

from AlphaBrain.model.modules.world_model.wan.t5 import T5EncoderModel

t5_model = T5EncoderModel(
    text_len=TEXT_LEN,
    dtype=torch.bfloat16,
    device=device,
    checkpoint_path=T5_PTH,
    tokenizer_path=TOK_DIR,
)
logger.info(
    "UMT5-XXL loaded. params: %.2fM",
    sum(p.numel() for p in t5_model.model.parameters()) / 1e6,
)

# -- 3. Encode and pad ----------------------------------------------------
embeddings = {}
BATCH_SIZE = 8  # process in small batches to avoid OOM

for i in range(0, len(all_instructions), BATCH_SIZE):
    batch = all_instructions[i : i + BATCH_SIZE]
    logger.info("Encoding %d/%d: %s ...", i, len(all_instructions), batch[0][:40])
    with torch.no_grad():
        raw = t5_model(batch, device)  # List[Tensor[L_i, 4096]]

    for instr, emb in zip(batch, raw):
        # emb: [L_i, 4096] -- pad to [512, 4096]
        padded = torch.zeros(TEXT_LEN, DIM, dtype=torch.bfloat16, device="cpu")
        L = min(emb.shape[0], TEXT_LEN)
        padded[:L] = emb[:L].cpu()
        embeddings[instr] = padded

logger.info("Encoded %d instructions.", len(embeddings))

# -- 4. Save --------------------------------------------------------------
os.makedirs(os.path.dirname(OUTPUT_PKL) or ".", exist_ok=True)
with open(OUTPUT_PKL, "wb") as f:
    pickle.dump(embeddings, f, protocol=4)

size_mb = os.path.getsize(OUTPUT_PKL) / 1024**2
logger.info("Saved to %s  (%.1f MB)", OUTPUT_PKL, size_mb)

# Quick sanity check
sample_key = all_instructions[0]
sample_val = embeddings[sample_key]
logger.info("Sample: key=%r  shape=%s  dtype=%s", sample_key, sample_val.shape, sample_val.dtype)
