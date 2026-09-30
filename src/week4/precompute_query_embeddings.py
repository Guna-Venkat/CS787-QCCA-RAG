"""Precompute and cache SFR query embeddings for the 231 QASPER dev questions.

Uses the local Salesforce/SFR-Embedding-Mistral model already cached in ~/.cache/huggingface/hub.
Saves query embeddings to results/week4/feature_discovery/dev_query_sfr_embeddings.pt
so that notebook execution is fast, CPU-runnable, and deterministic without GPU loading.
"""

import json
import logging
import time
from pathlib import Path
from typing import Dict

import torch
from sentence_transformers import SentenceTransformer

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


def precompute_dev_query_embeddings(
    raw_retrieval_path: str = "results/week2/raw/dev_retrieval.jsonl",
    output_path: str = "results/week4/feature_discovery/dev_query_sfr_embeddings.pt",
    model_name: str = "Salesforce/SFR-Embedding-Mistral",
    batch_size: int = 32,
    device: str = "cuda" if torch.cuda.is_available() else "cpu",
) -> Path:
    out_p = Path(output_path)
    out_p.parent.mkdir(parents=True, exist_ok=True)

    logger.info("Loading dev questions from %s...", raw_retrieval_path)
    records = []
    with open(raw_retrieval_path) as f:
        for line in f:
            records.append(json.loads(line))

    qids = [str(r["question_id"]).strip() for r in records]
    questions = [r["question"].strip() for r in records]
    logger.info("Loaded %d questions across %d records.", len(questions), len(records))

    logger.info("Loading embedding model %s on %s...", model_name, device)
    t0 = time.time()
    model = SentenceTransformer(
        model_name,
        device=device,
        model_kwargs={"torch_dtype": torch.bfloat16} if device == "cuda" else {},
    )
    logger.info("Model loaded in %.2f seconds.", time.time() - t0)

    logger.info("Encoding %d queries (batch_size=%d)...", len(questions), batch_size)
    t_enc = time.time()
    embeds = model.encode(
        questions,
        batch_size=batch_size,
        show_progress_bar=False,
        convert_to_tensor=True,
    )
    logger.info("Encoding complete in %.2f seconds.", time.time() - t_enc)

    # Convert to CPU float32 dictionary
    embed_dict: Dict[str, torch.Tensor] = {}
    for i, qid in enumerate(qids):
        # normalize to unit length for fast cosine similarity
        emb = embeds[i].detach().cpu().to(torch.float32)
        norm = torch.norm(emb, p=2)
        if norm > 0:
            emb = emb / norm
        embed_dict[qid] = emb

    cache_data = {
        "metadata": {
            "model_name": model_name,
            "dim": embeds.shape[1],
            "num_questions": len(qids),
            "normalized": True,
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        },
        "embeddings": embed_dict,
    }

    torch.save(cache_data, out_p)
    logger.info("Saved query embeddings to %s (size: %.2f MB)", out_p, out_p.stat().st_size / (1024**2))

    # Clean up GPU
    del model, embeds
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    logger.info("GPU VRAM cleared.")

    return out_p


if __name__ == "__main__":
    precompute_dev_query_embeddings()
