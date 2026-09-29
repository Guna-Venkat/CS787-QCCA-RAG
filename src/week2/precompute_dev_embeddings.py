"""Stage 1: Precompute & cache BM25 top-10 retrieval and SFR passage embeddings.

This module retrieves top-10 BM25 passages for all development questions,
saves the retrieval metadata to results/week2/raw/dev_retrieval.jsonl,
encodes each passage using Salesforce/SFR-Embedding-Mistral on GPU,
saves the traceable embeddings to results/week2/raw/dev_sfr_embeddings.pt,
and strictly unloads SFR from GPU memory.
"""

from __future__ import annotations

import argparse
import gc
import json
import os
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import torch
import yaml
from tqdm import tqdm

# Ensure SARA-main is in sys.path and extend src.__path__
REPO_ROOT = Path(__file__).resolve().parents[2]
SARA_ROOT = REPO_ROOT / "Baselines" / "SARA-main"
if str(SARA_ROOT) not in sys.path:
    sys.path.insert(0, str(SARA_ROOT))
import src
if hasattr(src, "__path__") and str(SARA_ROOT / "src") not in src.__path__:
    src.__path__.append(str(SARA_ROOT / "src"))

from llama_index.core import Document
from src.model.retriever import getBM25Retriever
from sentence_transformers import SentenceTransformer


def load_config(config_path: str | Path) -> Dict[str, Any]:
    with open(config_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def retrieve_dev_passages(
    dev_file: Path,
    chunk_size: int = 256,
    num_passages: int = 10,
    limit: Optional[int] = None,
) -> List[Dict[str, Any]]:
    """Retrieve top-N passages for each development question using BM25.

    Ensures every question has exactly num_passages passages, padding with empty
    passages if a paper has fewer than num_passages chunks.
    """
    if not dev_file.exists():
        raise FileNotFoundError(f"Development file not found: {dev_file}")

    rows: List[Dict[str, Any]] = []
    with open(dev_file, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))

    if limit is not None and limit > 0:
        rows = rows[:limit]

    retrieval_records: List[Dict[str, Any]] = []

    # Cache paper retrievers by paper_id / example_id to avoid redundant indexing
    paper_retrievers: Dict[str, Any] = {}

    print(f"Retrieving top-{num_passages} BM25 passages for {len(rows)} development questions...")
    for item in tqdm(rows, desc="BM25 Retrieval"):
        qid = str(item.get("id"))
        pid = str(item.get("example_id", ""))
        question = item.get("question", "")
        gold_answer = item.get("answer")
        gold_reformatted = item.get("answer_reformatted", gold_answer)
        qtype = item.get("question_type", "free_form")

        # Get or build BM25 retriever for this paper
        if pid not in paper_retrievers:
            raw_context = item.get("context", [])
            if isinstance(raw_context, list):
                docs = [Document(text=str(p)) for p in raw_context if str(p).strip()]
            else:
                docs = [Document(text=str(raw_context))] if str(raw_context).strip() else []

            if not docs:
                docs = [Document(text="Empty document.")]

            retriever, _ = getBM25Retriever(docs, similarity_top_k=num_passages, chunk_size=chunk_size)
            paper_retrievers[pid] = retriever

        retriever = paper_retrievers[pid]
        nodes = retriever.retrieve(question)

        passages: List[Dict[str, Any]] = []
        for rank_idx, node in enumerate(nodes[:num_passages]):
            passages.append({
                "passage_id": f"{pid}_chunk_{getattr(node, 'node_id', rank_idx)}",
                "rank": rank_idx + 1,
                "score": float(node.score) if getattr(node, "score", None) is not None else 0.0,
                "text": str(node.text),
            })

        # Pad if paper has fewer than num_passages chunks
        while len(passages) < num_passages:
            pad_rank = len(passages) + 1
            passages.append({
                "passage_id": f"{pid}_pad_{pad_rank}",
                "rank": pad_rank,
                "score": 0.0,
                "text": "",
            })

        record = {
            "question_id": qid,
            "paper_id": pid,
            "question": question,
            "gold_answers": gold_answer if isinstance(gold_answer, list) else [gold_answer],
            "gold_answers_reformatted": gold_reformatted if isinstance(gold_reformatted, list) else [gold_reformatted],
            "question_type": qtype,
            "passages": passages,
        }
        retrieval_records.append(record)

    return retrieval_records


def encode_and_cache_embeddings(
    retrieval_records: List[Dict[str, Any]],
    model_name: str = "Salesforce/SFR-Embedding-Mistral",
    device: str = "cuda",
    dtype_str: str = "bfloat16",
    batch_size: int = 32,
    output_embeddings_path: Optional[Path] = None,
) -> Dict[str, Any]:
    """Encode all retrieved passages with SFR and save the traceable cache to disk.

    Strictly manages GPU memory: deletes model and cleans cache after encoding.
    """
    if torch.cuda.is_available() and device.startswith("cuda"):
        torch.cuda.reset_peak_memory_stats()
        mem_before = torch.cuda.memory_allocated() / (1024**3)
        print(f"[STAGE 1] Initial GPU VRAM allocated: {mem_before:.2f} GB")

    target_dtype = torch.bfloat16 if dtype_str == "bfloat16" else torch.float16
    print(f"Loading SFR compressor {model_name} on {device} (dtype: {dtype_str})...")
    t0 = time.time()
    sfr_model = SentenceTransformer(
        model_name,
        device=device,
        model_kwargs={"torch_dtype": target_dtype}
    )
    load_time = time.time() - t0
    print(f"SFR Compressor loaded in {load_time:.2f}s.")

    # Flatten all passages preserving order: (qid, pid, passage_id, rank, text)
    flat_keys: List[Tuple[str, str, str, int]] = []
    flat_texts: List[str] = []
    for item in retrieval_records:
        qid = item["question_id"]
        pid = item["paper_id"]
        for p in item["passages"]:
            flat_keys.append((qid, pid, p["passage_id"], p["rank"]))
            flat_texts.append(p["text"])

    print(f"Encoding {len(flat_texts)} passages across {len(retrieval_records)} questions...")
    t_enc_start = time.time()
    all_embeds = sfr_model.encode(
        flat_texts,
        batch_size=batch_size,
        convert_to_tensor=True,
        show_progress_bar=True,
        device=device,
    )
    # Convert to CPU and target dtype to save RAM and prepare for disk storage
    all_embeds_cpu = all_embeds.detach().to(device="cpu", dtype=target_dtype)
    enc_time = time.time() - t_enc_start
    print(f"Encoded {len(flat_texts)} passages in {enc_time:.2f}s ({len(flat_texts)/enc_time:.1f} pass/s).")

    # Group embeddings and keys by question_id
    embeddings_dict: Dict[str, torch.Tensor] = {}
    keys_dict: Dict[str, List[Tuple[str, str, int]]] = {}

    idx = 0
    for item in retrieval_records:
        qid = item["question_id"]
        n_pass = len(item["passages"])
        q_embeds = all_embeds_cpu[idx : idx + n_pass]
        embeddings_dict[qid] = q_embeds
        keys_dict[qid] = [
            (pid, pass_id, rank) for (_, pid, pass_id, rank) in flat_keys[idx : idx + n_pass]
        ]
        idx += n_pass

    cache_data = {
        "metadata": {
            "model_name": model_name,
            "dtype": dtype_str,
            "dim": int(all_embeds_cpu.shape[-1]),
            "num_questions": len(retrieval_records),
            "passages_per_question": 10,
            "encoding_time_seconds": enc_time,
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        },
        "embeddings": embeddings_dict,
        "passage_keys": keys_dict,
    }

    if output_embeddings_path is not None:
        output_embeddings_path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(cache_data, output_embeddings_path)
        print(f"Saved SFR embedding cache to: {output_embeddings_path}")

    # Strict GPU VRAM cleanup: delete SFR model, collect garbage, empty cache
    print("Releasing SFR model and clearing GPU VRAM...")
    del sfr_model
    del all_embeds
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        mem_after = torch.cuda.memory_allocated() / (1024**3)
        peak_vram = torch.cuda.max_memory_allocated() / (1024**3)
        print(f"[STAGE 1 COMPLETE] GPU VRAM allocated after release: {mem_after:.2f} GB (Peak: {peak_vram:.2f} GB)")
        assert mem_after < 0.5, f"Warning: GPU memory not fully freed (allocated: {mem_after:.2f} GB)"

    return cache_data


def main():
    parser = argparse.ArgumentParser(description="Precompute BM25 retrieval and SFR embeddings.")
    parser.add_argument(
        "--config",
        default=str(REPO_ROOT / "configs/week2/sweep_config.yaml"),
        help="Path to sweep_config.yaml",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Optional limit on number of dev questions for smoke testing",
    )
    args = parser.parse_args()

    cfg = load_config(args.config)
    paths_cfg = cfg["paths"]

    dev_file = REPO_ROOT / paths_cfg["dev_file"]
    dev_retrieval_file = REPO_ROOT / paths_cfg["dev_retrieval_file"]
    dev_sfr_embeddings_file = REPO_ROOT / paths_cfg["dev_sfr_embeddings_file"]

    chunk_size = cfg["retrieval"].get("chunk_size", 256)
    num_passages = cfg.get("num_retrieved_passages", 10)
    sfr_model = cfg["models"].get("sfr_model", "Salesforce/SFR-Embedding-Mistral")
    sfr_dtype = cfg["models"].get("sfr_dtype", "bfloat16")

    # 1. Retrieve passages using BM25
    records = retrieve_dev_passages(
        dev_file=dev_file,
        chunk_size=chunk_size,
        num_passages=num_passages,
        limit=args.limit,
    )

    # Save retrieval records
    dev_retrieval_file.parent.mkdir(parents=True, exist_ok=True)
    with open(dev_retrieval_file, "w", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r) + "\n")
    print(f"Saved {len(records)} retrieval records to: {dev_retrieval_file}")

    # 2. Encode and cache embeddings with SFR
    device = "cuda" if torch.cuda.is_available() else "cpu"
    encode_and_cache_embeddings(
        retrieval_records=records,
        model_name=sfr_model,
        device=device,
        dtype_str=sfr_dtype,
        output_embeddings_path=dev_sfr_embeddings_file,
    )


if __name__ == "__main__":
    main()
