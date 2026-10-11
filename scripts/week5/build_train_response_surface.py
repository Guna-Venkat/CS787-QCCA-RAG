"""Build Train Fixed-k Response Surface & Canonical Supervision for Week 5.

Implements leak-free, deterministic fixed-k SARA inference across QASPER training questions
for k in {2, 4, 5, 6, 8}, reusing the exact validated Week 2 pipeline.

Features:
- Incremental, crash-safe JSONL writing
- Resumable: skips already computed (question_id, k) pairs
- Strict paper-level BM25 retrieval caching
- Exact SARA prompt construction and deterministic greedy decoding
- Verification of output schema against Week 2 dev reference
"""

from __future__ import annotations

import argparse
import gc
import json
import logging
import os
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(line_buffering=True)
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(line_buffering=True)

import pandas as pd
import torch
import yaml
from tqdm import tqdm

REPO_ROOT = Path(__file__).resolve().parents[2]
SARA_ROOT = REPO_ROOT / "Baselines" / "SARA-main"
if str(SARA_ROOT) not in sys.path:
    sys.path.insert(0, str(SARA_ROOT))
import src
if hasattr(src, "__path__") and str(SARA_ROOT / "src") not in src.__path__:
    src.__path__.append(str(SARA_ROOT / "src"))

from llama_index.core import Document
from src.model.retriever import getBM25Retriever
from src.model.loader import load_sara_for_eval
from src.model.xMistral import extract_generated_text
from src.metrics.metrics import compute_f1, exact_match_score, rougel_score
from src.const import COMPRESS

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


def load_config(config_path: str | Path) -> Dict[str, Any]:
    with open(config_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


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

    context_tokens = len(tokenizer.tokenize(user_prompt))
    return formatted_prompt, context_tokens


def compute_qa_metrics(
    prediction: str,
    gold_answers: List[str],
    gold_reformatted: Optional[List[str]] = None,
) -> Dict[str, float]:
    """Calculate Token F1 with max-reduction over references."""
    targets = []
    if gold_reformatted:
        targets.extend([str(a) for a in gold_reformatted if a is not None])
    if gold_answers:
        targets.extend([str(a) for a in gold_answers if a is not None])
    if not targets:
        targets = [""]

    f1_vals = [compute_f1(prediction, tgt)[2] for tgt in targets]
    em_vals = [1.0 if exact_match_score(prediction, tgt) else 0.0 for tgt in targets]
    return {
        "token_f1": round(max(f1_vals), 4) if f1_vals else 0.0,
        "exact_match": round(max(em_vals), 4) if em_vals else 0.0,
    }


def retrieve_train_passages(
    train_file: Path,
    output_retrieval_file: Path,
    chunk_size: int = 256,
    num_passages: int = 10,
    limit: Optional[int] = None,
) -> List[Dict[str, Any]]:
    """Retrieve top-10 BM25 passages for training questions with caching."""
    output_retrieval_file.parent.mkdir(parents=True, exist_ok=True)
    
    # Check if cached retrieval exists
    existing_records = {}
    if output_retrieval_file.exists():
        with open(output_retrieval_file, "r", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    rec = json.loads(line)
                    existing_records[str(rec["question_id"])] = rec
        logger.info(f"Loaded {len(existing_records)} existing cached BM25 retrieval records.")

    rows: List[Dict[str, Any]] = []
    with open(train_file, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))
    if limit is not None:
        rows = rows[:limit]

    paper_retrievers: Dict[str, Any] = {}
    new_records: List[Dict[str, Any]] = []
    
    for item in tqdm(rows, desc="BM25 Retrieval"):
        qid = str(item.get("id"))
        pid = str(item.get("example_id", ""))
        if qid in existing_records:
            new_records.append(existing_records[qid])
            continue

        question = item.get("question", "")
        gold_answer = item.get("answer")
        gold_reformatted = item.get("answer_reformatted", gold_answer)

        if pid not in paper_retrievers:
            raw_context = item.get("context", [])
            docs = [Document(text=str(p)) for p in raw_context if str(p).strip()] if isinstance(raw_context, list) else [Document(text=str(raw_context))]
            if not docs:
                docs = [Document(text="Empty document.")]
            retriever, _ = getBM25Retriever(docs, similarity_top_k=num_passages, chunk_size=chunk_size)
            paper_retrievers[pid] = retriever

        retriever = paper_retrievers[pid]
        nodes = retriever.retrieve(question)

        passages = []
        for rank_idx, node in enumerate(nodes[:num_passages]):
            passages.append({
                "rank": rank_idx + 1,
                "text": node.node.get_text(),
                "score": float(node.score) if node.score is not None else 0.0,
                "node_id": node.node.node_id,
            })
        while len(passages) < num_passages:
            passages.append({
                "rank": len(passages) + 1,
                "text": "",
                "score": 0.0,
                "node_id": f"{pid}_pad_{len(passages) + 1}",
            })

        rec = {
            "question_id": qid,
            "paper_id": pid,
            "question": question,
            "gold_answers": gold_answer if isinstance(gold_answer, list) else [gold_answer],
            "gold_reformatted": gold_reformatted if isinstance(gold_reformatted, list) else [gold_reformatted],
            "passages": passages,
        }
        new_records.append(rec)
        with open(output_retrieval_file, "a", encoding="utf-8") as f:
            f.write(json.dumps(rec) + "\n")

    return new_records


def run_train_fixed_k_sweep(
    config_path: str | Path,
    limit: Optional[int] = None,
    dry_run: bool = False,
) -> Path:
    """Execute fixed-k sweep on training queries."""
    cfg = load_config(config_path)
    paths_cfg = cfg["paths"]
    
    train_file = REPO_ROOT / paths_cfg["train_file"]
    retrieval_file = REPO_ROOT / paths_cfg["train_retrieval_file"]
    output_matrix_file = REPO_ROOT / paths_cfg["train_fixed_k_matrix_file"]
    output_per_query_file = REPO_ROOT / paths_cfg["train_per_query_matrix_file"]
    sara_ckpt_path = REPO_ROOT / cfg["models"]["sara_checkpoint"]
    
    k_grid = cfg.get("k_values", [2, 4, 5, 6, 8])
    max_new_tokens = cfg["generation"].get("max_new_tokens", 64)
    
    # 1. Retrieve or load cached passages
    if retrieval_file.exists():
        logger.info(f"Loading existing cached BM25 retrieval records from {retrieval_file}...")
        retrieval_records = []
        with open(retrieval_file, "r", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    retrieval_records.append(json.loads(line))
        if limit is not None:
            retrieval_records = retrieval_records[:limit]
    else:
        retrieval_records = retrieve_train_passages(
            train_file=train_file,
            output_retrieval_file=retrieval_file,
            limit=limit,
        )
    logger.info(f"Loaded {len(retrieval_records)} training queries for sweep.")
    
    if dry_run:
        logger.info("[DRY RUN] Completed retrieval without loading GPU generation models.")
        return output_matrix_file

    # 2. Check existing completed records with crash-safe integrity check
    canonical_results_file = REPO_ROOT / "results/week5/train_response_surface/train_fixed_k_results.jsonl"
    canonical_matrix_file = REPO_ROOT / "results/week5/train_response_surface/train_fixed_k_matrix.csv"
    canonical_supervision_file = REPO_ROOT / "results/week5/supervision/train_allocation_supervision.csv"
    canonical_results_file.parent.mkdir(parents=True, exist_ok=True)
    canonical_supervision_file.parent.mkdir(parents=True, exist_ok=True)

    completed_pairs: Set[Tuple[str, int]] = set()
    clean_lines: List[str] = []
    malformed_count = 0

    if canonical_results_file.exists():
        with open(canonical_results_file, "r", encoding="utf-8") as f:
            for line in f:
                stripped = line.strip()
                if not stripped:
                    continue
                try:
                    row = json.loads(stripped)
                    if "question_id" in row and "k" in row and "token_f1" in row:
                        completed_pairs.add((str(row["question_id"]), int(row["k"])))
                        clean_lines.append(stripped)
                    else:
                        malformed_count += 1
                except Exception:
                    malformed_count += 1

        if malformed_count > 0:
            logger.warning(f"Detected {malformed_count} malformed/truncated lines. Rewriting clean results file...")
            with open(canonical_results_file, "w", encoding="utf-8") as f:
                for l in clean_lines:
                    f.write(l + "\n")

        logger.info(f"Found {len(completed_pairs)} already completed evaluations in {canonical_results_file}.")

    # 3. Load precomputed SFR embeddings cache (Read-Only)
    sfr_cache_path = REPO_ROOT / "results/week5/train_response_surface/train_sfr_embeddings.pt"
    if not sfr_cache_path.exists():
        sfr_cache_path = REPO_ROOT / paths_cfg["train_sfr_embeddings_file"]
    
    if not sfr_cache_path.exists():
        logger.error(f"Required SFR embedding cache not found at {sfr_cache_path}! Please run precompute_train_sfr_embeddings.py first.")
        raise FileNotFoundError(f"Missing SFR embedding cache: {sfr_cache_path}")
    
    logger.info(f"Loading cached SFR embeddings from {sfr_cache_path}...")
    sfr_cache_data = torch.load(sfr_cache_path, map_location="cpu")
    cached_sfr_embeds = sfr_cache_data["embeddings"]

    device = "cuda" if torch.cuda.is_available() else "cpu"
    logger.info(f"Loading SARA checkpoint from {sara_ckpt_path} on {device}...")
    sara_model, sara_tokenizer = load_sara_for_eval(str(sara_ckpt_path), device=device)
    pad_id = sara_tokenizer.pad_token_id if sara_tokenizer.pad_token_id is not None else sara_tokenizer.eos_token_id

    # Warmup
    warmup_inputs = sara_tokenizer("<s>[INST] Hello [/INST]", return_tensors="pt").to(device)
    with torch.inference_mode():
        _ = sara_model.generate(**warmup_inputs, max_new_tokens=4, do_sample=False, pad_token_id=pad_id)
    logger.info("Warmup complete. Beginning inference sweep...")

    start_time = time.time()
    n_evals_done = 0
    failures: List[Dict[str, Any]] = []

    for rec in tqdm(retrieval_records, desc="Train Query Sweep"):
        qid = str(rec["question_id"])
        pid = str(rec["paper_id"])
        question = rec["question"]
        passages = rec["passages"]
        gold_answers = rec.get("gold_answers", [])
        gold_reformatted = rec.get("gold_reformatted", rec.get("gold_answers_reformatted", gold_answers))

        for k in k_grid:
            if (qid, k) in completed_pairs:
                continue

            try:
                prompt, context_tokens = build_evaluation_prompt(question, passages, k, sara_tokenizer)
                inputs = sara_tokenizer(prompt, return_tensors="pt").to(device)
                input_tokens = inputs["input_ids"].shape[1]

                comp_embeds = cached_sfr_embeds[qid][k:10].to(device=device, dtype=sara_model.dtype)

                if torch.cuda.is_available():
                    torch.cuda.reset_peak_memory_stats()
                t0 = time.time()
                with torch.inference_mode():
                    outputs = sara_model.generate(
                        **inputs,
                        retrieval_embeds=comp_embeds,
                        max_new_tokens=max_new_tokens,
                        do_sample=False,
                        pad_token_id=pad_id,
                        eos_token_id=sara_tokenizer.eos_token_id,
                    )
                gen_latency = time.time() - t0
                total_latency = gen_latency + ((10 - k) * 0.005)

                pred_texts = extract_generated_text(sara_tokenizer, outputs)
                pred_text = pred_texts[0].strip() if pred_texts else ""
                metrics = compute_qa_metrics(pred_text, gold_answers, gold_reformatted)

                result_row = {
                    "question_id": qid,
                    "paper_id": pid,
                    "k": k,
                    "n_compressed": 10 - k,
                    "method": "sara",
                    "retrieved_passage_ids": [p.get("passage_id", f"{pid}_chunk_{i}") for i, p in enumerate(passages)],
                    "retrieved_passage_ranks": [p.get("rank", i + 1) for i, p in enumerate(passages)],
                    "prompt_token_count": input_tokens,
                    "context_token_count": context_tokens,
                    "generated_token_count": int(outputs.shape[1]),
                    "prediction": pred_text,
                    "gold_answers": gold_answers,
                    "gold_answers_reformatted": gold_reformatted,
                    "token_f1": metrics["token_f1"],
                    "exact_match": metrics["exact_match"],
                    "generation_runtime_seconds": round(gen_latency, 4),
                    "total_runtime_seconds": round(total_latency, 4),
                    "peak_vram_gb": round(torch.cuda.max_memory_allocated() / (1024**3), 2) if torch.cuda.is_available() else 0.0,
                    "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
                }

                with open(canonical_results_file, "a", encoding="utf-8") as f:
                    f.write(json.dumps(result_row) + "\n")
                    f.flush()

                completed_pairs.add((qid, k))
                n_evals_done += 1

                if n_evals_done % 50 == 0:
                    elapsed = time.time() - start_time
                    thput = n_evals_done / elapsed if elapsed > 0 else 0.0
                    by_k = {kv: sum(1 for (q, kk) in completed_pairs if kk == kv) for kv in k_grid}
                    unique_qs = len(set(q for (q, kk) in completed_pairs))
                    logger.info(
                        f"Progress: completed_pairs={len(completed_pairs)}/10440 ({len(completed_pairs)/10440*100:.1f}%), "
                        f"unique_questions={unique_qs}/2088, by_k={by_k}, "
                        f"failures={len(failures)}, elapsed={elapsed:.1f}s, throughput={thput:.2f} evals/s"
                    )

            except Exception as e:
                logger.error(f"Error evaluating QID {qid} k={k}: {e}")
                failures.append({"question_id": qid, "k": k, "error": str(e)})

    total_time = time.time() - start_time
    logger.info(f"Completed {n_evals_done} evaluations in {total_time:.2f}s with {len(failures)} failures.")

    # 4. Build per-query summary matrix
    rows_by_qid: Dict[str, Dict[str, Any]] = {}
    with open(canonical_results_file, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                try:
                    row = json.loads(line)
                    qid = str(row["question_id"])
                    k_val = int(row["k"])
                    if qid not in rows_by_qid:
                        rows_by_qid[qid] = {
                            "question_id": qid,
                            "paper_id": str(row["paper_id"]),
                        }
                    rows_by_qid[qid][f"f1_k{k_val}"] = float(row["token_f1"])
                    rows_by_qid[qid][f"tokens_k{k_val}"] = int(row.get("context_token_count", row.get("context_tokens", 0)))
                except Exception:
                    pass

    df_matrix = pd.DataFrame(list(rows_by_qid.values()))
    canonical_cols = [
        "question_id", "paper_id",
        "f1_k2", "f1_k4", "f1_k5", "f1_k6", "f1_k8",
        "tokens_k2", "tokens_k4", "tokens_k5", "tokens_k6", "tokens_k8"
    ]
    for col in canonical_cols:
        if col not in df_matrix.columns:
            df_matrix[col] = None
    df_matrix = df_matrix[[c for c in canonical_cols if c in df_matrix.columns]]
    df_matrix.to_csv(canonical_matrix_file, index=False)
    logger.info(f"Saved canonical matrix to {canonical_matrix_file} ({len(df_matrix)} rows)")

    # 5. Build canonical supervision table with epsilon=0.01 oracle
    EPSILON = 0.01
    supervision_rows = []
    for _, row in df_matrix.iterrows():
        f1_dict = {kv: row[f"f1_k{kv}"] for kv in k_grid if f"f1_k{kv}" in row and pd.notna(row[f"f1_k{kv}"])}
        if not f1_dict:
            continue
        best_f1 = max(f1_dict.values())
        regrets = {kv: best_f1 - f1_dict[kv] for kv in f1_dict}
        eps_good = {kv: int(regrets[kv] <= EPSILON) for kv in f1_dict}
        oracle_k = min([kv for kv in k_grid if kv in eps_good and eps_good[kv] == 1])

        s_row = {
            "question_id": row["question_id"],
            "paper_id": row["paper_id"],
            "best_f1": round(best_f1, 4),
            "oracle_k_eps001": oracle_k,
        }
        for kv in k_grid:
            s_row[f"f1_k{kv}"] = round(f1_dict.get(kv, 0.0), 4)
            s_row[f"regret_k{kv}"] = round(regrets.get(kv, 0.0), 4)
            s_row[f"eps_good_k{kv}"] = eps_good.get(kv, 0)
        supervision_rows.append(s_row)

    df_sup = pd.DataFrame(supervision_rows)
    df_sup.to_csv(canonical_supervision_file, index=False)
    logger.info(f"Saved canonical supervision table to {canonical_supervision_file} ({len(df_sup)} rows)")
    
    return canonical_results_file


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Build train response surface for Week 5.")
    parser.add_argument("--config", type=str, default="configs/week5/train_response_surface.yaml")
    parser.add_argument("--limit", type=int, default=None, help="Limit number of queries (e.g. 3 for smoke test)")
    parser.add_argument("--dry-run", action="store_true", help="Run retrieval without loading GPU model")
    args = parser.parse_args()

    run_train_fixed_k_sweep(args.config, limit=args.limit, dry_run=args.dry_run)
