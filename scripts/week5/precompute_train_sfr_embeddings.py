"""Precompute and cache SFR embeddings for all 2,088 QASPER TRAIN questions.

Encodes unique passage texts using Salesforce/SFR-Embedding-Mistral on GPU,
assembles per-question (10, 4096) tensors in bfloat16, saves to disk,
and strictly unloads SFR from GPU memory.
"""

from __future__ import annotations

import gc
import json
import logging
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Set, Tuple

import torch
from sentence_transformers import SentenceTransformer
from tqdm import tqdm

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(line_buffering=True)
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(line_buffering=True)

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s", force=True)
logger = logging.getLogger(__name__)

REPO_ROOT = Path(__file__).resolve().parents[2]


def precompute_train_sfr_embeddings():
    retrieval_file = REPO_ROOT / "results/week5/train_response_surface/train_retrieval.jsonl"
    out_dir = REPO_ROOT / "results/week5/train_response_surface"
    out_file = out_dir / "train_sfr_embeddings.pt"

    if out_file.exists():
        logger.info(f"Checking existing SFR embedding cache at {out_file}...")
        existing_cache = torch.load(out_file, map_location="cpu")
        embeds = existing_cache.get("embeddings", {})
        if len(embeds) == 2088:
            logger.info("SFR cache is already 100% complete (2,088 questions)!")
            return existing_cache

    logger.info(f"Loading retrieval records from {retrieval_file}...")
    records = []
    with open(retrieval_file, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                records.append(json.loads(line))

    assert len(records) == 2088, f"Expected 2088 records, got {len(records)}"

    # 1. Collect unique texts to avoid redundant forward passes
    unique_texts: List[str] = []
    text_to_idx: Dict[str, int] = {}
    for r in records:
        for p in r["passages"]:
            t = p["text"]
            if t not in text_to_idx:
                text_to_idx[t] = len(unique_texts)
                unique_texts.append(t)

    logger.info(f"Total passages: {len(records) * 10}, Unique passage texts: {len(unique_texts)}")

    # 3. Check for intermediate checkpoint
    checkpoint_file = out_dir / "train_sfr_checkpoint.pt"
    encoded_dict: Dict[int, torch.Tensor] = {}
    if checkpoint_file.exists():
        logger.info(f"Loading partial SFR checkpoint from {checkpoint_file}...")
        try:
            ckpt_data = torch.load(checkpoint_file, map_location="cpu")
            encoded_dict = ckpt_data.get("encoded_dict", {})
            logger.info(f"Resumed {len(encoded_dict)} / {len(unique_texts)} already encoded texts.")
        except Exception as e:
            logger.warning(f"Could not load checkpoint: {e}. Starting fresh.")

    remaining_indices = [i for i in range(len(unique_texts)) if i not in encoded_dict]
    logger.info(f"Remaining texts to encode: {len(remaining_indices)} / {len(unique_texts)}")

    if remaining_indices:
        # Load SFR on GPU only if there is work remaining
        device = "cuda" if torch.cuda.is_available() else "cpu"
        logger.info(f"Loading SentenceTransformer('Salesforce/SFR-Embedding-Mistral') on {device}...")
        t0_load = time.time()
        sfr_model = SentenceTransformer(
            "Salesforce/SFR-Embedding-Mistral",
            device=device,
            model_kwargs={"torch_dtype": torch.bfloat16},
        )
        logger.info(f"SFR model loaded in {time.time() - t0_load:.2f}s.")

        chunk_size = 16
        save_interval = 128  # checkpoint every 8 chunks (~128 texts)
        t0_enc = time.time()
        n_encoded_since_save = 0

        for chunk_start in range(0, len(remaining_indices), chunk_size):
            chunk_idxs = remaining_indices[chunk_start : chunk_start + chunk_size]
            chunk_texts = [unique_texts[i] for i in chunk_idxs]

            chunk_embeds = sfr_model.encode(
                chunk_texts,
                batch_size=len(chunk_texts),
                convert_to_tensor=True,
                show_progress_bar=False,
                device=device,
            )
            chunk_embeds_cpu = chunk_embeds.detach().to(device="cpu", dtype=torch.bfloat16)

            for i, emb in zip(chunk_idxs, chunk_embeds_cpu):
                encoded_dict[i] = emb

            n_encoded_since_save += len(chunk_idxs)
            if n_encoded_since_save >= save_interval:
                torch.save({"encoded_dict": encoded_dict}, checkpoint_file)
                n_encoded_since_save = 0
                logger.info(f"Progress checkpoint: {len(encoded_dict)} / {len(unique_texts)} texts encoded ({len(encoded_dict)/len(unique_texts)*100:.1f}%).")

        # Final checkpoint save
        torch.save({"encoded_dict": encoded_dict}, checkpoint_file)
        enc_time = time.time() - t0_enc
        logger.info(f"Completed encoding remaining texts in {enc_time:.2f}s.")

        # 4. Strictly release SFR model from GPU
        del sfr_model
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            mem_after = torch.cuda.memory_allocated() / (1024**3)
            logger.info(f"SFR unloaded from GPU. Current allocated VRAM: {mem_after:.2f} GB")
    else:
        enc_time = 0.0
        logger.info("All unique texts are already encoded in checkpoint!")

    # Assemble unique embeddings tensor in original index order
    unique_embeds_cpu = torch.stack([encoded_dict[i] for i in range(len(unique_texts))])

    # 5. Assemble per-question (10, 4096) tensors
    logger.info("Assembling per-question (10, 4096) embedding tensors...")
    embeddings_dict: Dict[str, torch.Tensor] = {}
    keys_dict: Dict[str, List[Tuple[str, str, int]]] = {}

    for r in records:
        qid = str(r["question_id"])
        pid = str(r["paper_id"])
        q_text_indices = [text_to_idx[p["text"]] for p in r["passages"]]
        q_embeds = unique_embeds_cpu[q_text_indices]  # Shape: (10, 4096)
        embeddings_dict[qid] = q_embeds
        keys_dict[qid] = [(pid, p["passage_id"], p["rank"]) for p in r["passages"]]

    cache_data = {
        "metadata": {
            "model_name": "Salesforce/SFR-Embedding-Mistral",
            "dtype": "bfloat16",
            "dim": int(unique_embeds_cpu.shape[-1]),
            "num_questions": len(records),
            "passages_per_question": 10,
            "unique_texts_encoded": len(unique_texts),
            "encoding_time_seconds": round(enc_time, 2),
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        },
        "embeddings": embeddings_dict,
        "passage_keys": keys_dict,
    }

    torch.save(cache_data, out_file)
    file_size_mb = round(out_file.stat().st_size / (1024**2), 2)
    logger.info(f"Saved complete SFR embedding cache to {out_file} ({file_size_mb} MB).")

    # Also sync to supervision dir
    sup_file = REPO_ROOT / "results/week5/supervision/train_sfr_embeddings.pt"
    torch.save(cache_data, sup_file)
    logger.info(f"Synced cache to {sup_file}.")

    # Audit cache
    nan_count = 0
    shape_errors = 0
    for qid, t in embeddings_dict.items():
        if torch.isnan(t).any():
            nan_count += 1
        if t.shape != (10, 4096):
            shape_errors += 1

    audit = {
        "expected_unique_passages": len(unique_texts),
        "cached_unique_passages": len(encoded_dict),
        "missing_passages": len(unique_texts) - len(encoded_dict),
        "duplicates": 0,
        "invalid_or_nan_embeddings": nan_count + shape_errors,
        "cache_file_size_mb": file_size_mb,
        "num_questions_covered": len(embeddings_dict),
        "expected_questions": 2088,
        "embedding_dim": int(unique_embeds_cpu.shape[-1]),
        "dtype": "bfloat16",
        "cache_coverage_percentage": round(100.0 * len(embeddings_dict) / 2088, 2),
        "status": "COMPLETE" if len(embeddings_dict) == 2088 and nan_count == 0 else "INCOMPLETE",
    }

    audit_path = out_dir / "sfr_cache_audit.json"
    with open(audit_path, "w", encoding="utf-8") as f:
        json.dump(audit, f, indent=2)

    logger.info(f"SFR cache audit: {json.dumps(audit, indent=2)}")
    return cache_data


if __name__ == "__main__":
    precompute_train_sfr_embeddings()
