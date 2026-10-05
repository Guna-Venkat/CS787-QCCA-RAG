"""Deterministic Stratified Sampling of 40 QASPER Dev Questions for Final Week 0 Study.

Constraints:
- 40 questions from QASPER dev split.
- Approximately 35-40 distinct papers.
- Stratified across:
  * 1 annotated evidence paragraph
  * 2 annotated evidence paragraphs
  * 3+ annotated evidence paragraphs
  * incomplete/failed BM25 evidence retrieval
  * answer types where practical
- Selection reason recorded for every question.
- Deterministic and paper-aware.
- Output: results/week0/final40/sample_manifest.csv
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
import sys
from typing import Any, Dict, List, Set, Tuple
import numpy as np
import pandas as pd
import yaml

sys.stdout.reconfigure(line_buffering=True)
sys.stderr.reconfigure(line_buffering=True)

PROJECT_ROOT = Path("/home/gunavenkat/Downloads/CS787-RAG-Project")
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.week0.dataset_adapter import QASPERAdapter
from src.week0.dataset_analyzer import classify_wh_type, extract_content_words

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)



def check_graded_evidence_match(
    p_text: str,
    unique_ev: List[str],
    flat_hl: List[str],
) -> Tuple[str, float, int]:
    """Compute graded evidence match for a passage against gold evidence.
    
    Returns:
        (category, max_score, best_gold_idx)
        where category is one of: STRONG, PARTIAL, WEAK, NONE.
    """
    p_lower = p_text.lower()
    p_words = set(extract_content_words(p_text))
    if not p_words or not unique_ev:
        return "NONE", 0.0, -1

    best_cat = "NONE"
    best_score = 0.0
    best_idx = -1

    for g_idx, g_para in enumerate(unique_ev):
        g_words = set(extract_content_words(g_para))
        if not g_words:
            continue

        # Check highlight exact substring (>15 chars)
        exact_sub = False
        for hl in flat_hl:
            hl_c = hl.strip().lower()
            if len(hl_c) > 15 and hl_c in p_lower:
                exact_sub = True
                break

        overlap_ev = len(p_words & g_words) / len(g_words) if g_words else 0.0
        overlap_p = len(p_words & g_words) / len(p_words) if p_words else 0.0
        jaccard = len(p_words & g_words) / len(p_words | g_words) if (p_words | g_words) else 0.0

        score = max(overlap_ev, overlap_p, jaccard)

        # Categorize
        if exact_sub or overlap_ev >= 0.70 or jaccard >= 0.50:
            cat = "STRONG"
            score = 1.0 if exact_sub else max(score, 0.85)
        elif overlap_ev >= 0.35 or (overlap_p >= 0.40 and len(p_words & g_words) >= 10):
            cat = "PARTIAL"
        elif overlap_ev >= 0.15 or overlap_p >= 0.15:
            cat = "WEAK"
        else:
            cat = "NONE"

        # Priority: STRONG > PARTIAL > WEAK > NONE
        cat_order = {"STRONG": 3, "PARTIAL": 2, "WEAK": 1, "NONE": 0}
        if cat_order[cat] > cat_order[best_cat] or (cat_order[cat] == cat_order[best_cat] and score > best_score):
            best_cat = cat
            best_score = score
            best_idx = g_idx

    return best_cat, round(best_score, 4), best_idx


def select_final40_sample() -> pd.DataFrame:
    with open(PROJECT_ROOT / "configs/week0/week0.yaml", "r") as f:
        cfg = yaml.safe_load(f)

    adapter = QASPERAdapter(
        train_split_path=PROJECT_ROOT / cfg["dataset"]["train_split"],
        dev_split_path=PROJECT_ROOT / cfg["dataset"]["dev_split"],
        raw_arrow_train_path=PROJECT_ROOT / cfg["dataset"]["raw_arrow_train"],
        manifest_path=PROJECT_ROOT / cfg["dataset"]["manifest"],
        dev_retrieval_path=PROJECT_ROOT / cfg["dataset"]["dev_retrieval_file"],
    )

    dev_records = adapter.load_split("dev")
    logger.info(f"Loaded {len(dev_records)} QASPER dev records.")

    candidates = []
    for r_idx, r in enumerate(dev_records):
        if (r_idx + 1) % 50 == 0 or r_idx == 0:
            print(f"Processing dev record {r_idx + 1}/{len(dev_records)}...", flush=True)
        qid = str(r.get("id", r.get("question_id", "")))
        pid = str(r.get("example_id", r.get("paper_id", "")))
        query = adapter.get_query(r)
        wh_type = classify_wh_type(query)

        answers = adapter.get_gold_answer(r)
        gold_ans = answers[0] if answers else ""
        meta = adapter.get_metadata(r)
        ans_type = meta.get("answer_type", "extractive")

        ev_entries = adapter.get_gold_evidence(r)
        flat_ev = []
        flat_hl = []
        for e in ev_entries:
            flat_ev.extend(e.get("evidence_paragraphs", []))
            flat_hl.extend(e.get("highlighted_spans", []))
        unique_ev = list(dict.fromkeys(flat_ev))
        ev_count = len(unique_ev)

        passages = adapter.get_retrieved_passages(r)[:10]

        # Graded matching against retrieved passages
        covered_gold_indices = set()
        strong_match_count = 0
        partial_match_count = 0

        for p in passages:
            p_text = p.get("text", "")
            cat, score, best_idx = check_graded_evidence_match(p_text, unique_ev, flat_hl)
            if cat == "STRONG":
                strong_match_count += 1
                if best_idx >= 0:
                    covered_gold_indices.add(best_idx)
            elif cat == "PARTIAL":
                partial_match_count += 1
                if best_idx >= 0:
                    covered_gold_indices.add(best_idx)

        # Retrieval recall metrics
        recall_any = 1 if len(covered_gold_indices) > 0 else 0
        recall_all = 1 if (ev_count > 0 and len(covered_gold_indices) >= ev_count) else 0

        # Classify retrieval status
        if ev_count == 0:
            retrieval_status = "no_evidence_unanswerable"
        elif recall_all == 1:
            retrieval_status = "complete_retrieval"
        elif recall_any == 1:
            retrieval_status = "partial_retrieval"
        else:
            retrieval_status = "failed_retrieval"

        candidates.append({
            "question_id": qid,
            "paper_id": pid,
            "question": query,
            "gold_answer": gold_ans,
            "answer_type": ans_type,
            "wh_type": wh_type,
            "evidence_count": ev_count,
            "strong_match_count": strong_match_count,
            "partial_match_count": partial_match_count,
            "unique_gold_covered": len(covered_gold_indices),
            "recall_any": recall_any,
            "recall_all": recall_all,
            "retrieval_status": retrieval_status,
        })

    df_cand = pd.DataFrame(candidates)
    logger.info(f"Dev questions profile:\n{df_cand['evidence_count'].value_counts().sort_index()}")
    logger.info(f"Retrieval status:\n{df_cand['retrieval_status'].value_counts()}")

    # Stratified Selection Strategy:
    # 40 questions total, aim for ~38-40 distinct papers.
    # Strata target:
    # 1. 1 evidence para: 12 questions (complete or high recall)
    # 2. 2 evidence paras: 11 questions (complete or partial)
    # 3. 3+ evidence paras: 9 questions (multi-paragraph evidence)
    # 4. Incomplete/Failed BM25 retrieval: 8 questions (recall_any == 0 or recall_all == 0 with evidence_count >= 1)
    # Diversity across answer types and wh_types.

    # Group by strata
    stratum_1 = df_cand[(df_cand["evidence_count"] == 1) & (df_cand["recall_any"] == 1)].copy()
    stratum_2 = df_cand[(df_cand["evidence_count"] == 2) & (df_cand["recall_any"] == 1)].copy()
    stratum_3plus = df_cand[(df_cand["evidence_count"] >= 3) & (df_cand["recall_any"] == 1)].copy()
    stratum_failed = df_cand[(df_cand["evidence_count"] >= 1) & (df_cand["retrieval_status"].isin(["failed_retrieval", "partial_retrieval"]))].copy()

    # Prioritize 1 question per paper
    selected_rows = []
    used_papers: Set[str] = set()
    used_qids: Set[str] = set()

    # Reproducible RNG
    rng = np.random.RandomState(42)

    def select_from_stratum(df_pool: pd.DataFrame, target_count: int, reason_label: str) -> None:
        # Shuffle pool deterministically
        pool_indices = rng.permutation(df_pool.index)
        added = 0
        # Pass 1: Try new papers
        for idx in pool_indices:
            if added >= target_count:
                break
            row = df_pool.loc[idx]
            qid = str(row["question_id"])
            pid = str(row["paper_id"])
            if qid in used_qids:
                continue
            if pid not in used_papers:
                used_papers.add(pid)
                used_qids.add(qid)
                row_dict = dict(row)
                row_dict["stratum"] = reason_label
                row_dict["selection_reason"] = (
                    f"{reason_label} | ev_count={row['evidence_count']} | "
                    f"retrieval={row['retrieval_status']} | type={row['wh_type']}/{row['answer_type']}"
                )
                selected_rows.append(row_dict)
                added += 1

        # Pass 2: Allow duplicate paper if needed to hit target count
        if added < target_count:
            for idx in pool_indices:
                if added >= target_count:
                    break
                row = df_pool.loc[idx]
                qid = str(row["question_id"])
                if qid in used_qids:
                    continue
                used_qids.add(qid)
                row_dict = dict(row)
                row_dict["stratum"] = reason_label
                row_dict["selection_reason"] = (
                    f"{reason_label} (multi-question paper) | ev_count={row['evidence_count']} | "
                    f"retrieval={row['retrieval_status']} | type={row['wh_type']}/{row['answer_type']}"
                )
                selected_rows.append(row_dict)
                added += 1

    # Select across the 4 strata
    # 1. Failed/Incomplete retrieval (8 questions: 4 failed, 4 partial)
    df_pure_failed = stratum_failed[stratum_failed["retrieval_status"] == "failed_retrieval"]
    df_partial = stratum_failed[stratum_failed["retrieval_status"] == "partial_retrieval"]
    select_from_stratum(df_pure_failed, 4, "failed_bm25_retrieval")
    select_from_stratum(df_partial, 4, "incomplete_bm25_retrieval")

    # 2. 3+ evidence paragraphs (9 questions)
    select_from_stratum(stratum_3plus, 9, "multi_evidence_3plus")

    # 3. 2 evidence paragraphs (11 questions)
    select_from_stratum(stratum_2, 11, "dual_evidence_2para")

    # 4. 1 evidence paragraph (12 questions)
    select_from_stratum(stratum_1, 12, "single_evidence_1para")

    df_selected = pd.DataFrame(selected_rows)
    logger.info(f"Total selected: {len(df_selected)} questions across {df_selected['paper_id'].nunique()} distinct papers.")

    # Save to results/week0/final40/sample_manifest.csv
    out_dir = PROJECT_ROOT / "results/week0/final40"
    out_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = out_dir / "sample_manifest.csv"

    # Sort deterministically by question_id
    df_selected = df_selected.sort_values("question_id").reset_index(drop=True)
    df_selected["sample_index"] = range(1, len(df_selected) + 1)

    df_selected.to_csv(manifest_path, index=False)
    logger.info(f"Saved sample manifest to: {manifest_path}")

    # Display summary
    print("\n--- SAMPLE COMPOSITION SUMMARY ---")
    print(f"Total questions: {len(df_selected)}")
    print(f"Distinct papers: {df_selected['paper_id'].nunique()}")
    print("\nEvidence Count Distribution:")
    print(df_selected["evidence_count"].value_counts().sort_index())
    print("\nStratum Distribution:")
    print(df_selected["stratum"].value_counts())
    print("\nRetrieval Status Distribution:")
    print(df_selected["retrieval_status"].value_counts())
    print("\nWH Type Distribution:")
    print(df_selected["wh_type"].value_counts())
    print("\nAnswer Type Distribution:")
    print(df_selected["answer_type"].value_counts())

    return df_selected


if __name__ == "__main__":
    select_final40_sample()
