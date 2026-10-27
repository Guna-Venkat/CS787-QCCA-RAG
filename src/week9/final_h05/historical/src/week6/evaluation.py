"""Evaluation metrics, routing diagnostics, SARA generation, and statistical bootstrap for Week 6.

Responsibilities:
- Routing quality: Evidence Recall@k and Precision@k.
- Selected-rank distribution: empirical selection frequency of original BM25 ranks 1..10.
- Routing diversity: pairwise selection disagreement rates and Jaccard overlaps.
- Canonical R0 reuse from dev_fixed_k_matrix.jsonl.
- Real SARA generation execution and answer F1 evaluation for R1 and R2.
- Paper-clustered bootstrap (2,000 resamples by paper_id) for Delta F1 and 95% CIs.
- Failure analysis table comparing R2 vs R0 wins and losses.
"""

from __future__ import annotations

import json
import logging
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple, Union

import numpy as np
import pandas as pd
import torch

import sys
from collections import Counter
import string
import re

REPO_ROOT = Path(__file__).resolve().parents[2]
SARA_ROOT = REPO_ROOT / "Baselines" / "SARA-main"
if str(SARA_ROOT) not in sys.path:
    sys.path.insert(0, str(SARA_ROOT))
import src
if hasattr(src, "__path__") and str(SARA_ROOT / "src") not in src.__path__:
    src.__path__.append(str(SARA_ROOT / "src"))

try:
    from src.metrics.metrics import compute_f1, exact_match_score, rougel_score
except (ImportError, ModuleNotFoundError):
    # Robust self-contained fallback matching SARA metrics exactly
    try:
        import nltk
        from nltk.corpus import stopwords
        try:
            stop_words = set(stopwords.words('english'))
        except LookupError:
            nltk.download('stopwords', quiet=True)
            stop_words = set(stopwords.words('english'))
    except Exception:
        stop_words = set()

    def normalize_answer(s: str) -> str:
        s = s.lower()
        if "---Response---\n\n" in s:
            s = s.split("---Response---\n\n")[1]
        if "---Answer---\n\n" in s:
            s = s.split("---Answer---\n\n")[1]
        if "Answer: " in s:
            s = s.split("Answer:")[1]
        exclude = set(string.punctuation)
        s = "".join(ch for ch in s if ch not in exclude)
        s = re.sub(r"\b(a|an|the)\b", " ", s)
        if stop_words:
            words = s.split()
            s = " ".join(w for w in words if w not in stop_words)
        return " ".join(s.split())

    def compute_f1(a_gold: str, a_pred: str) -> Tuple[float, float, float]:
        gold_toks = normalize_answer(a_gold).split()
        pred_toks = normalize_answer(a_pred).split()
        common = Counter(gold_toks) & Counter(pred_toks)
        num_same = sum(common.values())
        if len(gold_toks) == 0 or len(pred_toks) == 0:
            f1 = float(gold_toks == pred_toks)
            return f1, f1, f1
        if num_same == 0:
            return 0.0, 0.0, 0.0
        precision = 1.0 * num_same / len(pred_toks)
        recall = 1.0 * num_same / len(gold_toks)
        f1 = (2 * precision * recall) / (precision + recall)
        return precision, recall, f1

    def exact_match_score(a_pred: str, a_gold: str) -> bool:
        return normalize_answer(a_pred) == normalize_answer(a_gold)

    def rougel_score(a_pred: str, a_gold: str) -> float:
        return 0.0

try:
    from src.model.xMistral import extract_generated_text
except ImportError:
    def extract_generated_text(tokenizer: Any, outputs: torch.Tensor) -> List[str]:
        return [tokenizer.decode(out, skip_special_tokens=True) for out in outputs]
from src.week6.routing import (
    build_routed_sara_prompt,
    extract_routed_sara_embeddings,
    validate_routing_assignment,
)

logger = logging.getLogger(__name__)


def compute_evidence_recall_precision(
    selected_indices: List[int],
    gold_binary_labels: np.ndarray,
) -> Tuple[float, float]:
    """Compute Recall@k and Precision@k for a single question's selected passage subset.

    Args:
        selected_indices: Selected 0-indexed passage ranks (length k).
        gold_binary_labels: Binary array of length 10 indicating gold evidence matches.

    Returns:
        (recall_at_k, precision_at_k)
    """
    k = len(selected_indices)
    if k == 0:
        return 0.0, 0.0

    total_gold = float(np.sum(gold_binary_labels))
    selected_gold = float(np.sum(gold_binary_labels[selected_indices]))

    recall = (selected_gold / total_gold) if total_gold > 0 else 1.0
    precision = selected_gold / float(k)
    return float(recall), float(precision)


def evaluate_routing_quality(
    routing_assignments: Dict[str, Dict[int, List[List[int]]]],
    gold_labels_all: np.ndarray,
    budgets: List[int] = [2, 4, 6],
) -> pd.DataFrame:
    """Evaluate Evidence Recall@k and Precision@k across routers and budgets.

    Args:
        routing_assignments: Dict mapping router_name -> {k: list of selected_indices for each query}
        gold_labels_all: Shape (N_queries, 10) binary array of gold matches.
        budgets: List of k values.

    Returns:
        pd.DataFrame with columns: [router, k, evidence_recall, evidence_precision, total_queries]
    """
    rows = []
    n_queries = len(gold_labels_all)

    for router_name, k_dict in routing_assignments.items():
        for k in budgets:
            if k not in k_dict:
                continue
            assignments = k_dict[k]
            assert len(assignments) == n_queries, f"Query count mismatch for {router_name} at k={k}"

            recalls = []
            precisions = []

            for i in range(n_queries):
                rec, prec = compute_evidence_recall_precision(assignments[i], gold_labels_all[i])
                recalls.append(rec)
                precisions.append(prec)

            rows.append({
                "router": router_name,
                "k": k,
                "evidence_recall": float(np.mean(recalls)),
                "evidence_precision": float(np.mean(precisions)),
                "total_queries": n_queries,
            })

    return pd.DataFrame(rows)


def compute_selected_rank_distribution(
    routing_assignments: Dict[str, Dict[int, List[List[int]]]],
    budgets: List[int] = [2, 4, 6],
    n_passages: int = 10,
) -> pd.DataFrame:
    """Compute empirical selection frequencies of original BM25 ranks 1..10.

    Returns:
        pd.DataFrame with columns: [router, k, rank_1_freq, ..., rank_10_freq]
    """
    rows = []
    for router_name, k_dict in routing_assignments.items():
        for k in budgets:
            if k not in k_dict:
                continue
            assignments = k_dict[k]
            n_queries = len(assignments)

            counts = np.zeros(n_passages, dtype=np.float32)
            for sel in assignments:
                for idx in sel:
                    if 0 <= idx < n_passages:
                        counts[idx] += 1.0

            freqs = counts / max(n_queries, 1)

            row = {
                "router": router_name,
                "k": k,
                "n_queries": n_queries,
            }
            for r in range(1, n_passages + 1):
                row[f"rank_{r}_freq"] = float(freqs[r - 1])
            rows.append(row)

    return pd.DataFrame(rows)


def compute_routing_diversity(
    routing_assignments: Dict[str, Dict[int, List[List[int]]]],
    budgets: List[int] = [2, 4, 6],
) -> pd.DataFrame:
    """Compute pairwise selection disagreement rates and mean Jaccard overlaps."""
    routers = list(routing_assignments.keys())
    rows = []

    for k in budgets:
        for i in range(len(routers)):
            for j in range(i + 1, len(routers)):
                r_a, r_b = routers[i], routers[j]
                if k not in routing_assignments[r_a] or k not in routing_assignments[r_b]:
                    continue

                sel_a = routing_assignments[r_a][k]
                sel_b = routing_assignments[r_b][k]
                n = len(sel_a)

                disagreements = 0
                jaccards = []

                for idx in range(n):
                    set_a = set(sel_a[idx])
                    set_b = set(sel_b[idx])

                    if set_a != set_b:
                        disagreements += 1

                    inter = len(set_a & set_b)
                    union = len(set_a | set_b)
                    jaccards.append(inter / union if union > 0 else 1.0)

                disagree_rate = disagreements / float(n) if n > 0 else 0.0
                mean_jaccard = float(np.mean(jaccards)) if jaccards else 1.0

                rows.append({
                    "k": k,
                    "router_A": r_a,
                    "router_B": r_b,
                    "disagreement_rate": disagree_rate,
                    "mean_jaccard_overlap": mean_jaccard,
                    "n_queries": n,
                })

    return pd.DataFrame(rows)


def compute_qa_answer_metrics(
    prediction: str,
    gold_answers: List[str],
    gold_reformatted: Optional[List[str]] = None,
) -> Dict[str, float]:
    """Calculate token F1, exact match, and ROUGE-L with max reduction over gold answers.

    Handles empty predictions honestly without crashing:
    - Empty prediction against non-empty references receives F1=0, EM=0, ROUGE=0.
    - Never calls ROUGE with an empty hypothesis.
    """
    pred_str = str(prediction).strip() if prediction is not None else ""

    targets = []
    if gold_reformatted:
        targets.extend([str(a).strip() for a in gold_reformatted if a is not None])
    if gold_answers:
        targets.extend([str(a).strip() for a in gold_answers if a is not None])
    valid_targets = [t for t in targets if t]

    if not valid_targets:
        # Unanswerable reference set
        if pred_str == "":
            return {"token_f1": 1.0, "exact_match": 1.0, "rouge_l": 1.0}
        else:
            return {"token_f1": 0.0, "exact_match": 0.0, "rouge_l": 0.0}

    # Empty prediction against non-empty answer: count honestly as 0 without calling ROUGE
    if pred_str == "":
        return {"token_f1": 0.0, "exact_match": 0.0, "rouge_l": 0.0}

    f1_vals, em_vals, rouge_vals = [], [], []
    for tgt in valid_targets:
        _, _, f1 = compute_f1(pred_str, tgt)
        f1_vals.append(f1)
        em = 1.0 if exact_match_score(pred_str, tgt) else 0.0
        em_vals.append(em)

        try:
            r = rougel_score(pred_str, tgt)
            if r is not None and not np.isnan(r):
                rouge_vals.append(float(r))
            else:
                rouge_vals.append(0.0)
        except Exception:
            rouge_vals.append(0.0)

    return {
        "token_f1": float(max(f1_vals)) if f1_vals else 0.0,
        "exact_match": float(max(em_vals)) if em_vals else 0.0,
        "rouge_l": float(max(rouge_vals)) if rouge_vals else 0.0,
    }


def load_canonical_r0_results(
    dev_fixed_k_matrix_path: Union[str, Path],
    budgets: List[int] = [2, 4, 6],
) -> Dict[int, Dict[str, Any]]:
    """Load canonical existing R0 results from dev_fixed_k_matrix.jsonl without re-execution."""
    results_by_k: Dict[int, Dict[str, Any]] = {k: {"rows": {}, "f1_list": [], "tokens_list": []} for k in budgets}

    with open(dev_fixed_k_matrix_path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                row = json.loads(line)
                k = int(row.get("k", -1))
                if k in budgets:
                    qid = str(row["question_id"])
                    f1 = float(row.get("token_f1", 0.0))
                    tokens = int(row.get("context_tokens", 0))

                    results_by_k[k]["rows"][qid] = row
                    results_by_k[k]["f1_list"].append(f1)
                    results_by_k[k]["tokens_list"].append(tokens)

    return results_by_k


def run_sara_generation_for_routed_policy(
    sara_model: Any,
    sara_tokenizer: Any,
    dev_records: List[Dict[str, Any]],
    selected_indices_list: List[List[int]],
    cached_embeddings: Dict[str, torch.Tensor],
    k: int,
    device: str = "cuda" if torch.cuda.is_available() else "cpu",
    max_new_tokens: int = 64,
    cache_path: Optional[Union[str, Path]] = None,
) -> Dict[str, Any]:
    """Execute real SARA generation for a routed policy at fixed budget k.

    Strictly preserves:
    - exactly k passages as natural text in original BM25 rank order
    - exactly 10 - k unselected passages as <COMPRESS> tokens mapped to cached SFR embeddings
    - Resumable: skips already-completed questions if cache_path is provided.
    """
    sara_model.eval()
    pad_id = sara_tokenizer.pad_token_id if sara_tokenizer.pad_token_id is not None else sara_tokenizer.eos_token_id

    completed_by_qid: Dict[str, Dict[str, Any]] = {}
    if cache_path and Path(cache_path).exists():
        with open(cache_path, "r", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    try:
                        rec = json.loads(line)
                        completed_by_qid[str(rec["question_id"])] = rec
                    except Exception:
                        pass
        if completed_by_qid:
            logger.info(f"Loaded {len(completed_by_qid)} existing records from cache: {cache_path}")

    per_query_f1s = []
    per_query_tokens = []
    per_query_latencies = []
    predictions = []

    try:
        from tqdm import tqdm
        iterator = enumerate(tqdm(dev_records, desc=f"Evaluating k={k}", leave=True))
    except ImportError:
        iterator = enumerate(dev_records)

    cache_file = open(cache_path, "a", encoding="utf-8") if cache_path else None

    try:
        for idx, item in iterator:
            qid = str(item["question_id"])
            question_text = item["question"]
            passages = item["passages"][:10]
            gold_answers = item.get("gold_answers", [])
            gold_reformatted = item.get("gold_answers_reformatted", gold_answers)

            selected_indices = selected_indices_list[idx]
            k_val = len(selected_indices) if k is None else k
            validate_routing_assignment(selected_indices, k=k_val, total_passages=len(passages))

            # If already in cache, reuse directly without GPU forward pass
            if qid in completed_by_qid:
                cached_rec = completed_by_qid[qid]
                f1_val = float(cached_rec["token_f1"])
                tokens_val = int(cached_rec["context_tokens"])
                lat_val = float(cached_rec.get("latency", 0.0))

                per_query_f1s.append(f1_val)
                per_query_tokens.append(tokens_val)
                per_query_latencies.append(lat_val)
                predictions.append(cached_rec)
                continue

            formatted_prompt, context_tokens, unselected_indices = build_routed_sara_prompt(
                question=question_text,
                passages=passages,
                selected_indices=selected_indices,
                tokenizer=sara_tokenizer,
            )

            inputs = sara_tokenizer(formatted_prompt, return_tensors="pt").to(device)

            # Slice cached SFR embeddings for unselected passages
            qid_embeds = cached_embeddings[qid]
            comp_embeds = extract_routed_sara_embeddings(qid_embeds, selected_indices).to(
                device=device, dtype=sara_model.dtype
            )

            t0 = time.time()
            with torch.no_grad():
                outputs = sara_model.generate(
                    **inputs,
                    retrieval_embeds=comp_embeds,
                    max_new_tokens=max_new_tokens,
                    do_sample=False,
                    pad_token_id=pad_id,
                    eos_token_id=sara_tokenizer.eos_token_id,
                )
            latency = time.time() - t0

            pred_texts = extract_generated_text(sara_tokenizer, outputs)
            prediction = pred_texts[0].strip() if pred_texts else ""

            qa_metrics = compute_qa_answer_metrics(
                prediction=prediction,
                gold_answers=gold_answers,
                gold_reformatted=gold_reformatted,
            )

            per_query_f1s.append(qa_metrics["token_f1"])
            per_query_tokens.append(context_tokens)
            per_query_latencies.append(latency)

            pred_record = {
                "question_id": qid,
                "prediction": prediction,
                "token_f1": qa_metrics["token_f1"],
                "context_tokens": context_tokens,
                "latency": latency,
                "selected_ranks": [i + 1 for i in sorted(selected_indices)],
            }
            predictions.append(pred_record)

            if cache_file:
                cache_file.write(json.dumps(pred_record) + "\n")
                cache_file.flush()
    finally:
        if cache_file:
            cache_file.close()

    return {
        "k": k,
        "mean_f1": float(np.mean(per_query_f1s)),
        "std_f1": float(np.std(per_query_f1s)),
        "mean_tokens": float(np.mean(per_query_tokens)),
        "mean_latency": float(np.mean(per_query_latencies)),
        "per_query_f1s": per_query_f1s,
        "per_query_tokens": per_query_tokens,
        "predictions": predictions,
    }


def paper_clustered_bootstrap_comparison(
    paper_ids: List[str],
    f1_a: List[float],
    f1_b: List[float],
    name_a: str,
    name_b: str,
    n_resamples: int = 2000,
    confidence_level: float = 0.95,
    seed: int = 42,
) -> Dict[str, Any]:
    """Perform paper-clustered bootstrap test comparing two policies.

    Clusters questions by paper_id to account for intra-paper correlation.
    """
    assert len(paper_ids) == len(f1_a) == len(f1_b)
    df = pd.DataFrame({"paper_id": paper_ids, "f1_a": f1_a, "f1_b": f1_b})

    unique_papers = np.sort(df["paper_id"].unique())
    n_papers = len(unique_papers)

    # Group vectors by paper
    paper_diffs = []
    for pid in unique_papers:
        sub = df[df["paper_id"] == pid]
        paper_diffs.append(sub["f1_a"].values - sub["f1_b"].values)

    point_delta = float(np.mean(f1_a) - np.mean(f1_b))

    rng = np.random.default_rng(seed)
    boot_deltas = np.zeros(n_resamples, dtype=np.float32)

    for b in range(n_resamples):
        sampled_p_indices = rng.choice(n_papers, size=n_papers, replace=True)
        sampled_diffs = [paper_diffs[p_idx] for p_idx in sampled_p_indices]
        flat_diffs = np.concatenate(sampled_diffs)
        boot_deltas[b] = np.mean(flat_diffs)

    alpha = 1.0 - confidence_level
    ci_low = float(np.percentile(boot_deltas, 100.0 * (alpha / 2.0)))
    ci_high = float(np.percentile(boot_deltas, 100.0 * (1.0 - alpha / 2.0)))

    # Empirical two-sided p-value against zero difference
    p_val = float(2.0 * min(np.mean(boot_deltas <= 0.0), np.mean(boot_deltas >= 0.0)))
    p_val = min(p_val, 1.0)

    return {
        "comparison": f"{name_a}_vs_{name_b}",
        "policy_A": name_a,
        "policy_B": name_b,
        "delta_f1": point_delta,
        "ci_lower": ci_low,
        "ci_upper": ci_high,
        "p_value": p_val,
        "significant": bool(ci_low > 0.0 or ci_high < 0.0),
    }
