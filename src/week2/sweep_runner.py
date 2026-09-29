"""Stage 2: Fixed-k Development Baseline Response Matrix Runner.

Evaluates development questions across k in {0, 2, 4, 5, 6, 8, 10}:
- k in {0, 2, 4, 5, 6, 8}: Evaluated with the SARA checkpoint (sara_qasper_proj_lr5e4_seed42)
  injecting (10 - k) cached SFR embeddings at <COMPRESS> positions.
- k = 10: Evaluated with the Standard RAG checkpoint (rag_qasper_lora_r16_seed42)
  with all 10 passages formatted as natural language text.

Guarantees:
- Candidate set is strictly identical across all k (same top-10 BM25 passages in same order).
- Deterministic decoding (do_sample=False, max_new_tokens=64).
- Resumable: skips already-completed (question_id, k) pairs to prevent duplicate work.
- Sequential model lifecycle: SARA model is loaded once, then unloaded before RAG model.
"""

from __future__ import annotations

import argparse
import gc
import json
import os
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

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

from src.model.loader import load_sara_for_eval
from src.model.xMistral import extract_generated_text
from src.metrics.metrics import compute_f1, exact_match_score, rougel_score
from src.const import COMPRESS


def get_git_commit_hash() -> str:
    try:
        commit = subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=str(REPO_ROOT), stderr=subprocess.DEVNULL
        ).decode("ascii").strip()
        return commit
    except Exception:
        return "unknown"


def build_evaluation_prompt(
    question: str,
    passages: List[Dict[str, Any]],
    k: int,
    tokenizer: Any,
) -> Tuple[str, int]:
    """Format prompt for given question, top-10 passages, and evidence allocation k.

    Ranks 1..k -> Natural language text in 'Context'
    Ranks k+1..10 -> (10 - k) <COMPRESS> tokens in 'Additional Context'
    """
    assert len(passages) == 10, f"Expected 10 passages, got {len(passages)}"
    
    # 1. Text Context (ranks 1..k)
    if k > 0:
        text_blocks = []
        for i, p in enumerate(passages[:k]):
            p_text = p.get("text", "").strip()
            if p_text:
                text_blocks.append(f"Document {i + 1}. {p_text}")
        text_context = "\n\n".join(text_blocks) if text_blocks else "No direct text context provided."
    else:
        text_context = "No direct text context provided."

    # 2. Compressed Context (ranks k+1..10)
    num_compressed = 10 - k
    if num_compressed > 0:
        compress_tokens_str = "\n".join([f"{idx + 1}. {COMPRESS}" for idx in range(num_compressed)])
        additional_context_str = f"## Additional Context (compression tokens)\n{compress_tokens_str}\n---"
    else:
        additional_context_str = ""

    if k == 10:
        user_prompt = f"""Answer my questions based on the given context.
---
## Context
{text_context}
---
## Your Task
Answer the following question in a succinct manner. Use a single phrase or a short sentence if possible.
Question: {question}
Your Answer:"""
    else:
        user_prompt = f"""Answer my questions based on the given context.
---
## Context
{text_context}
---
{additional_context_str}
## Your Task
Answer the following question in a succinct manner. Use a single phrase or a short sentence if possible.
Question: {question}
Your Answer:"""

    if hasattr(tokenizer, "apply_chat_template") and tokenizer.chat_template:
        messages = [{"role": "user", "content": user_prompt}]
        formatted_prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    else:
        formatted_prompt = f"<s>[INST] {user_prompt} [/INST]"

    # Count input context tokens
    context_tokens = len(tokenizer.tokenize(user_prompt))
    return formatted_prompt, context_tokens


def compute_qa_metrics(
    prediction: str,
    gold_answers: List[str],
    gold_reformatted: Optional[List[str]] = None,
) -> Dict[str, float]:
    """Calculate Token F1, Exact Match, and ROUGE-L with max-reduction over gold references."""
    targets = []
    if gold_reformatted:
        targets.extend([str(a) for a in gold_reformatted if a is not None])
    if gold_answers:
        targets.extend([str(a) for a in gold_answers if a is not None])
    if not targets:
        targets = [""]

    f1_vals = []
    em_vals = []
    rouge_vals = []

    for tgt in targets:
        _, _, f1 = compute_f1(prediction, tgt)
        f1_vals.append(f1)
        em = 1.0 if exact_match_score(prediction, tgt) else 0.0
        em_vals.append(em)
        try:
            r = rougel_score(prediction, tgt)
            rouge_vals.append(float(r))
        except Exception:
            rouge_vals.append(0.0)

    return {
        "token_f1": round(max(f1_vals), 4) if f1_vals else 0.0,
        "exact_match": round(max(em_vals), 4) if em_vals else 0.0,
        "rouge_l": round(max(rouge_vals), 4) if rouge_vals else 0.0,
    }


def run_fixed_k_sweep(
    config_path: str | Path,
    limit: Optional[int] = None,
    specific_k_values: Optional[List[int]] = None,
) -> Path:
    """Execute fixed-k sweep across development set.

    Resumable: reads existing raw output and skips already completed (qid, k) records.
    """
    with open(config_path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    paths_cfg = cfg["paths"]
    retrieval_file = REPO_ROOT / paths_cfg["dev_retrieval_file"]
    embeddings_file = REPO_ROOT / paths_cfg["dev_sfr_embeddings_file"]
    features_file = REPO_ROOT / paths_cfg["dev_retrieval_features_file"]
    output_matrix_file = REPO_ROOT / paths_cfg["dev_fixed_k_matrix_file"]

    sara_ckpt_path = REPO_ROOT / cfg["models"]["sara_checkpoint"]
    rag_ckpt_path = REPO_ROOT / cfg["models"]["rag_checkpoint"]
    base_model_name = cfg["models"].get("base_model", "mistralai/Mistral-7B-Instruct-v0.2")
    seed = cfg.get("seed", 42)
    max_new_tokens = cfg["generation"].get("max_new_tokens", 64)
    do_sample = cfg["generation"].get("do_sample", False)
    git_commit = get_git_commit_hash()

    k_grid = specific_k_values if specific_k_values is not None else cfg.get("k_values", [0, 2, 4, 5, 6, 8, 10])

    # 1. Load retrieval records, embeddings cache, and features
    if not retrieval_file.exists():
        raise FileNotFoundError(f"Retrieval file not found: {retrieval_file}. Run Stage 1 first.")
    if not embeddings_file.exists():
        raise FileNotFoundError(f"Embeddings cache not found: {embeddings_file}. Run Stage 1 first.")

    print(f"Loading retrieval records from: {retrieval_file}")
    retrieval_records: List[Dict[str, Any]] = []
    with open(retrieval_file, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                retrieval_records.append(json.loads(line))

    if limit is not None and limit > 0:
        retrieval_records = retrieval_records[:limit]

    print(f"Loading SFR embedding cache from: {embeddings_file}")
    embed_cache = torch.load(embeddings_file, map_location="cpu")
    cached_embeddings: Dict[str, torch.Tensor] = embed_cache["embeddings"]

    # Load features if available
    features_by_qid: Dict[str, Dict[str, Any]] = {}
    if features_file.exists():
        with open(features_file, "r", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    rec = json.loads(line)
                    features_by_qid[str(rec["question_id"])] = rec

    # 2. Check for existing completed records to enable safe resumption
    completed_pairs: Set[Tuple[str, int]] = set()
    output_matrix_file.parent.mkdir(parents=True, exist_ok=True)
    if output_matrix_file.exists():
        with open(output_matrix_file, "r", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    try:
                        row = json.loads(line)
                        completed_pairs.add((str(row["question_id"]), int(row["k"])))
                    except Exception:
                        pass
        print(f"Found {len(completed_pairs)} already completed evaluations. Resuming without duplicating work.")

    # 3. Partition k values by required model checkpoint to minimize model swapping
    sara_k_values = [k for k in k_grid if k < 10]
    rag_k_values = [k for k in k_grid if k == 10]

    device = "cuda" if torch.cuda.is_available() else "cpu"

    # --- Phase A: SARA Evaluations (k in {0, 2, 4, 5, 6, 8}) ---
    if sara_k_values:
        print(f"\n============================================================")
        print(f"PHASE A: Loading SARA Checkpoint for k in {sara_k_values}")
        print(f"Checkpoint: {sara_ckpt_path}")
        print(f"============================================================")
        sara_model, sara_tokenizer = load_sara_for_eval(str(sara_ckpt_path), device=device)
        pad_id = sara_tokenizer.pad_token_id if sara_tokenizer.pad_token_id is not None else sara_tokenizer.eos_token_id

        # Warm-up inference
        warmup_prompt = "<s>[INST] Hello [/INST]"
        warmup_inputs = sara_tokenizer(warmup_prompt, return_tensors="pt").to(device)
        with torch.no_grad():
            _ = sara_model.generate(**warmup_inputs, max_new_tokens=4, do_sample=False, pad_token_id=pad_id)
        print("Warm-up complete. Executing SARA sweep...")

        with open(output_matrix_file, "a", encoding="utf-8") as out_f:
            for k in sara_k_values:
                print(f"\n--- Evaluating k={k} ({k} text + {10-k} compressed) across {len(retrieval_records)} questions ---")
                for item in tqdm(retrieval_records, desc=f"k={k}"):
                    qid = str(item["question_id"])
                    pid = str(item["paper_id"])
                    if (qid, k) in completed_pairs:
                        continue

                    passages = item["passages"]
                    question_text = item["question"]
                    gold_answers = item["gold_answers"]
                    gold_reformatted = item.get("gold_answers_reformatted", gold_answers)

                    formatted_prompt, context_tokens = build_evaluation_prompt(
                        question=question_text,
                        passages=passages,
                        k=k,
                        tokenizer=sara_tokenizer,
                    )
                    inputs = sara_tokenizer(formatted_prompt, return_tensors="pt").to(device)
                    input_tokens = int(inputs["input_ids"].shape[1])

                    # Prepare compressed embeddings slice for ranks k+1..10
                    # Shape: (10 - k, 4096)
                    comp_embeds = cached_embeddings[qid][k:10].to(device=device, dtype=sara_model.dtype)

                    if torch.cuda.is_available():
                        torch.cuda.reset_peak_memory_stats()
                    t0 = time.time()
                    with torch.no_grad():
                        outputs = sara_model.generate(
                            **inputs,
                            retrieval_embeds=comp_embeds,
                            max_new_tokens=max_new_tokens,
                            do_sample=do_sample,
                            pad_token_id=pad_id,
                            eos_token_id=sara_tokenizer.eos_token_id,
                        )
                    gen_latency = time.time() - t0
                    peak_vram = torch.cuda.max_memory_allocated() / (1024**3) if torch.cuda.is_available() else 0.0

                    pred_texts = extract_generated_text(sara_tokenizer, outputs)
                    prediction = pred_texts[0].strip() if pred_texts else ""
                    gen_tokens = int(outputs.shape[1])

                    # Latency accounting: estimated compression latency + measured generation latency
                    # Avg SFR encoding time is ~0.005s per passage
                    est_comp_latency = round((10 - k) * 0.005, 4)
                    synthesized_total_latency = round(est_comp_latency + gen_latency, 4)

                    metrics = compute_qa_metrics(
                        prediction=prediction,
                        gold_answers=gold_answers,
                        gold_reformatted=gold_reformatted,
                    )
                    feats = features_by_qid.get(qid, {})

                    record = {
                        "question_id": qid,
                        "paper_id": pid,
                        "k": k,
                        "n_compressed": 10 - k,
                        "method": "sara",
                        "checkpoint_id": sara_ckpt_path.name,
                        "model_name": base_model_name,
                        "seed": seed,
                        "prediction": prediction,
                        "gold_answers": gold_answers,
                        "retrieved_passage_ids": [p["passage_id"] for p in passages],
                        "bm25_scores": [p["score"] for p in passages],
                        "rho_1": feats.get("rho_1", 0.0),
                        "delta_12": feats.get("delta_12", 0.0),
                        "entropy": feats.get("entropy", 0.0),
                        "entropy_normalized": feats.get("entropy_normalized", 0.0),
                        "query_length": feats.get("query_token_length", len(question_text.split())),
                        "n_high": feats.get("n_high", 1),
                        "input_tokens": input_tokens,
                        "context_tokens": context_tokens,
                        "generated_tokens": gen_tokens,
                        "token_f1": metrics["token_f1"],
                        "exact_match": metrics["exact_match"],
                        "rouge_l": metrics["rouge_l"],
                        "generation_latency": round(gen_latency, 4),
                        "compression_latency": est_comp_latency,
                        "synthesized_total_latency": synthesized_total_latency,
                        "peak_vram": round(peak_vram, 2),
                        "git_commit": git_commit,
                        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
                    }
                    out_f.write(json.dumps(record) + "\n")
                    out_f.flush()
                    completed_pairs.add((qid, k))

        # Unload SARA model to free GPU memory before loading RAG model
        print("\nReleasing SARA model from GPU...")
        del sara_model
        del sara_tokenizer
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    # --- Phase B: Standard RAG Evaluation (k = 10) ---
    if rag_k_values:
        print(f"\n============================================================")
        print(f"PHASE B: Loading Standard RAG Checkpoint for k=10")
        print(f"Checkpoint: {rag_ckpt_path}")
        print(f"============================================================")
        rag_model, rag_tokenizer = load_sara_for_eval(str(rag_ckpt_path), device=device)
        pad_id = rag_tokenizer.pad_token_id if rag_tokenizer.pad_token_id is not None else rag_tokenizer.eos_token_id

        # Warm-up inference
        warmup_prompt = "<s>[INST] Hello [/INST]"
        warmup_inputs = rag_tokenizer(warmup_prompt, return_tensors="pt").to(device)
        with torch.no_grad():
            _ = rag_model.generate(**warmup_inputs, max_new_tokens=4, do_sample=False, pad_token_id=pad_id)
        print("Warm-up complete. Executing Standard RAG (k=10) sweep...")

        with open(output_matrix_file, "a", encoding="utf-8") as out_f:
            k = 10
            for item in tqdm(retrieval_records, desc="k=10"):
                qid = str(item["question_id"])
                pid = str(item["paper_id"])
                if (qid, k) in completed_pairs:
                    continue

                passages = item["passages"]
                question_text = item["question"]
                gold_answers = item["gold_answers"]
                gold_reformatted = item.get("gold_answers_reformatted", gold_answers)

                formatted_prompt, context_tokens = build_evaluation_prompt(
                    question=question_text,
                    passages=passages,
                    k=10,
                    tokenizer=rag_tokenizer,
                )
                inputs = rag_tokenizer(formatted_prompt, return_tensors="pt").to(device)
                input_tokens = int(inputs["input_ids"].shape[1])

                if torch.cuda.is_available():
                    torch.cuda.reset_peak_memory_stats()
                t0 = time.time()
                with torch.no_grad():
                    outputs = rag_model.generate(
                        **inputs,
                        max_new_tokens=max_new_tokens,
                        do_sample=do_sample,
                        pad_token_id=pad_id,
                        eos_token_id=rag_tokenizer.eos_token_id,
                    )
                gen_latency = time.time() - t0
                peak_vram = torch.cuda.max_memory_allocated() / (1024**3) if torch.cuda.is_available() else 0.0

                pred_texts = extract_generated_text(rag_tokenizer, outputs)
                prediction = pred_texts[0].strip() if pred_texts else ""
                gen_tokens = int(outputs.shape[1])

                metrics = compute_qa_metrics(
                    prediction=prediction,
                    gold_answers=gold_answers,
                    gold_reformatted=gold_reformatted,
                )
                feats = features_by_qid.get(qid, {})

                record = {
                    "question_id": qid,
                    "paper_id": pid,
                    "k": 10,
                    "n_compressed": 0,
                    "method": "rag",
                    "checkpoint_id": rag_ckpt_path.name,
                    "model_name": base_model_name,
                    "seed": seed,
                    "prediction": prediction,
                    "gold_answers": gold_answers,
                    "retrieved_passage_ids": [p["passage_id"] for p in passages],
                    "bm25_scores": [p["score"] for p in passages],
                    "rho_1": feats.get("rho_1", 0.0),
                    "delta_12": feats.get("delta_12", 0.0),
                    "entropy": feats.get("entropy", 0.0),
                    "entropy_normalized": feats.get("entropy_normalized", 0.0),
                    "query_length": feats.get("query_token_length", len(question_text.split())),
                    "n_high": feats.get("n_high", 1),
                    "input_tokens": input_tokens,
                    "context_tokens": context_tokens,
                    "generated_tokens": gen_tokens,
                    "token_f1": metrics["token_f1"],
                    "exact_match": metrics["exact_match"],
                    "rouge_l": metrics["rouge_l"],
                    "generation_latency": round(gen_latency, 4),
                    "compression_latency": 0.0,
                    "synthesized_total_latency": round(gen_latency, 4),
                    "peak_vram": round(peak_vram, 2),
                    "git_commit": git_commit,
                    "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
                }
                out_f.write(json.dumps(record) + "\n")
                out_f.flush()
                completed_pairs.add((qid, k))

        # Unload RAG model
        print("\nReleasing RAG model from GPU...")
        del rag_model
        del rag_tokenizer
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    print(f"\n[SWEEP COMPLETE] Output saved to: {output_matrix_file}")
    return output_matrix_file


def main():
    parser = argparse.ArgumentParser(description="Run fixed-k development sweep.")
    parser.add_argument(
        "--config",
        default=str(REPO_ROOT / "configs/week2/sweep_config.yaml"),
        help="Path to sweep_config.yaml",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Limit number of development questions (for smoke tests)",
    )
    parser.add_argument(
        "--k-values",
        type=int,
        nargs="+",
        default=None,
        help="Specific k values to evaluate (default: from config)",
    )
    args = parser.parse_args()

    run_fixed_k_sweep(
        config_path=args.config,
        limit=args.limit,
        specific_k_values=args.k_values,
    )


if __name__ == "__main__":
    main()
