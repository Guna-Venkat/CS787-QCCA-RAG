from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd
import torch

REPO_ROOT = Path(__file__).resolve().parents[2]
SARA_ROOT = REPO_ROOT / "Baselines" / "SARA-main"
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
import src
if hasattr(src, "__path__") and str(SARA_ROOT / "src") not in src.__path__:
    src.__path__.append(str(SARA_ROOT / "src"))

from src.model.loader import load_sara_for_eval
from src.week8.semantic import SemanticScorer
from src.week8.data import load_config, load_week8_split
from src.week8.promotion_oracle import (
    compute_best_rank_distribution,
    compute_oracle_1,
    compute_oracle_selective_1,
    plot_oracle_results,
)
from src.week8.promotion_surface import run_single_promotion_surface
from src.week8.evidence_enrichment import (
    enrich_promotion_record,
    compute_evidence_utilization_summary,
    load_qasper_gold_evidence_map,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


def load_and_verify_canonical_base_k4(
    val_records: List[Dict[str, Any]],
    canonical_results_path: Path,
) -> Dict[str, Dict[str, Any]]:
    """Load canonical k=4 base generation results and verify exact baseline F1 ≈ 0.38884."""
    val_qids = {str(r["question_id"]) for r in val_records}
    k4_by_qid: Dict[str, Dict[str, Any]] = {}

    if not canonical_results_path.is_file():
        raise FileNotFoundError(f"Canonical fixed-k results file not found at: {canonical_results_path}")

    with open(canonical_results_path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                rec = json.loads(line)
                qid = str(rec["question_id"])
                if qid in val_qids and int(rec.get("k", 0)) == 4:
                    k4_by_qid[qid] = rec

    missing_qids = val_qids - set(k4_by_qid.keys())
    if missing_qids:
        raise ValueError(f"Missing canonical base k=4 records for {len(missing_qids)} queries: {list(missing_qids)[:5]}...")

    mean_f1 = sum(float(r["token_f1"]) for r in k4_by_qid.values()) / len(k4_by_qid)
    target_f1 = 0.38884486987682965
    logger.info("Loaded canonical base k=4 for %d queries. Mean F1 = %.7f (Target: %.7f)", len(k4_by_qid), mean_f1, target_f1)

    if abs(mean_f1 - target_f1) > 1e-4:
        raise ValueError(f"CRITICAL INVARIANT VIOLATION: Canonical base k=4 mean F1 ({mean_f1:.5f}) differs from expected baseline ({target_f1:.5f})!")

    logger.info("VERIFICATION PASSED: Canonical base k=4 matches exact 0.38884 baseline!")
    return k4_by_qid


def compute_heuristic_baselines(df_promotions: pd.DataFrame, df_o1: pd.DataFrame, df_sel1: pd.DataFrame) -> pd.DataFrame:
    """Compute comparative table between Base k4, BM25 rank 5, MiniLM rank 5, and Oracles."""
    base_f1 = float(df_o1["base_f1"].mean())
    base_tok = float(df_promotions.groupby("question_id")["base_context_tokens"].first().mean())

    # MiniLM semantic rank 5
    df_sem5 = df_promotions[df_promotions["semantic_rank"] == 5].groupby("question_id").first()
    sem5_f1 = float(df_sem5["promoted_f1"].mean())
    sem5_tok = float(df_sem5["promoted_context_tokens"].mean())
    sem5_delta = sem5_f1 - base_f1

    # BM25 rank 5
    df_bm25_5 = df_promotions[df_promotions["bm25_rank"] == 5].groupby("question_id").first()
    bm25_f1 = float(df_bm25_5["promoted_f1"].mean()) if len(df_bm25_5) > 0 else base_f1
    bm25_tok = float(df_bm25_5["promoted_context_tokens"].mean()) if len(df_bm25_5) > 0 else base_tok
    bm25_delta = bm25_f1 - base_f1

    o1_f1 = float(df_o1["oracle1_f1"].mean())
    o1_tok = float(df_o1["promoted_context_tokens"].mean())
    o1_delta = o1_f1 - base_f1

    sel1_f1 = float(df_sel1["final_f1"].mean())
    sel1_tok = float(df_sel1["final_context_tokens"].mean())
    sel1_delta = sel1_f1 - base_f1
    sel1_rate = float(df_sel1["promoted"].mean()) * 100.0

    rows = [
        {"Policy": "Static Semantic k=4 (Base)", "Mean_F1": base_f1, "Delta_F1": 0.0, "Context_Tokens": base_tok, "Promotion_Rate_Pct": 0.0, "Notes": "Canonical baseline"},
        {"Policy": "Heuristic: BM25 Rank 5", "Mean_F1": bm25_f1, "Delta_F1": bm25_delta, "Context_Tokens": bm25_tok, "Promotion_Rate_Pct": 100.0, "Notes": "Standard lexical retrieval heuristic"},
        {"Policy": "Heuristic: MiniLM Rank 5", "Mean_F1": sem5_f1, "Delta_F1": sem5_delta, "Context_Tokens": sem5_tok, "Promotion_Rate_Pct": 100.0, "Notes": "Standard semantic relevance heuristic"},
        {"Policy": "Oracle-Selective-1 (eps=0.01)", "Mean_F1": sel1_f1, "Delta_F1": sel1_delta, "Context_Tokens": sel1_tok, "Promotion_Rate_Pct": sel1_rate, "Notes": "Selective promotion when utility > 0.01"},
        {"Policy": "Oracle-1 (Unconditional Best)", "Mean_F1": o1_f1, "Delta_F1": o1_delta, "Context_Tokens": o1_tok, "Promotion_Rate_Pct": 100.0, "Notes": "Always promote max-utility candidate"},
    ]
    return pd.DataFrame(rows)


def extract_case_studies(df_enriched: pd.DataFrame) -> Dict[str, Any]:
    """Extract detailed case studies for Q1811 and Q1812."""
    case_studies = {}
    for qid in ["1811", "1812"]:
        sub = df_enriched[df_enriched["question_id"] == qid].sort_values("semantic_rank")
        if sub.empty:
            continue
        first_row = sub.iloc[0]
        promotions_detail = []
        for _, r in sub.iterrows():
            promotions_detail.append({
                "semantic_rank": int(r["semantic_rank"]),
                "bm25_rank": int(r["bm25_rank"]),
                "candidate_passage_index": int(r["candidate_passage_index"]),
                "candidate_text_snippet": str(r["candidate_text"])[:160] + "...",
                "promoted_prediction": str(r["promoted_prediction"]),
                "promoted_f1": float(r["promoted_f1"]),
                "delta_f1": float(r["delta_f1"]),
                "utility_label": str(r["utility_label"]),
                "is_gold_evidence": bool(r["is_gold_evidence"]),
                "evidence_utilization_type": str(r["evidence_utilization_type"]),
            })

        case_studies[f"Q{qid}"] = {
            "question_id": qid,
            "paper_id": str(first_row.get("paper_id", "")),
            "question": str(first_row["question"]),
            "gold_answers": first_row["gold_answers"],
            "base_prediction": str(first_row["base_prediction"]),
            "base_f1": float(first_row["base_f1"]),
            "n_candidates": len(promotions_detail),
            "promotions": promotions_detail,
        }
    return case_studies


def main():
    parser = argparse.ArgumentParser(description="Canonical W8-A Promotion Response Surface Runner")
    parser.add_argument("--smoke", action="store_true", help="Run smoke test on 5 queries")
    parser.add_argument("--full", action="store_true", help="Run full split evaluation (342 queries)")
    parser.add_argument("--split", choices=["fit", "internal_val"], default="internal_val")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--smoke-limit", type=int, default=5)
    args = parser.parse_args()

    cfg = load_config()
    split = args.split
    smoke_limit = args.smoke_limit if args.smoke else None
    tag = "_smoke" if args.smoke else ""

    logger.info("=== Canonical W8-A Promotion Response Surface (split=%s, smoke=%s, device=%s) ===", split, args.smoke, args.device)

    # 1. Load data
    records, sfr_dict, manifest_df = load_week8_split(split, cfg)
    logger.info("Loaded %d queries for split %s", len(records), split)

    # 2. Load and verify canonical base k4
    canonical_w7b_results_path = REPO_ROOT / "results/week7/w7b_semantic_budget/response_surface/train_semantic_fixed_k_results.jsonl"
    canonical_base_k4 = load_and_verify_canonical_base_k4(records, canonical_w7b_results_path)

    # 3. MiniLM Semantic scorer
    scorer = SemanticScorer()
    logger.info("Precomputing MiniLM semantic similarity scores...")
    semantic_scores_by_qid = {}
    items_to_score = records[:smoke_limit] if smoke_limit else records
    for r in items_to_score:
        qid = str(r["question_id"])
        q_text = r["question"]
        passages = [p["text"] for p in r.get("passages", [])[:10]]
        sims = scorer.score_query_passages(q_text, passages).cpu().numpy()
        semantic_scores_by_qid[qid] = sims

    # 4. Load SARA Generator
    sara_ckpt_path = REPO_ROOT / cfg["sara"]["checkpoint_path"]
    logger.info("Loading SARA generator from %s on %s...", sara_ckpt_path, args.device)
    sara_model, sara_tokenizer = load_sara_for_eval(str(sara_ckpt_path), device=args.device)

    # 5. Run response surface
    surfaces_dir = REPO_ROOT / "results/week8/surfaces"
    surfaces_dir.mkdir(parents=True, exist_ok=True)
    cache_path = surfaces_dir / f"promotion_surface_{split}{tag}.jsonl"

    logger.info("Executing promotion surface -> %s", cache_path)
    df_promotions = run_single_promotion_surface(
        sara_model=sara_model,
        sara_tokenizer=sara_tokenizer,
        records=records,
        semantic_scores_by_qid=semantic_scores_by_qid,
        cached_sfr_embeddings=sfr_dict,
        cache_path=cache_path,
        device=args.device,
        smoke_limit=smoke_limit,
        canonical_base_k4=canonical_base_k4,
    )

    logger.info("Promotion surface completed with %d records.", len(df_promotions))

    # 6. Evidence Enrichment
    logger.info("Running evidence enrichment and potential diagnostics...")
    evidence_map = load_qasper_gold_evidence_map()
    enriched_rows = []
    enriched_cache_path = surfaces_dir / f"promotion_surface_{split}_enriched{tag}.jsonl"
    with open(enriched_cache_path, "w", encoding="utf-8") as f_enrich:
        for _, row in df_promotions.iterrows():
            enriched_rec = enrich_promotion_record(row.to_dict(), evidence_map=evidence_map)
            enriched_rows.append(enriched_rec)
            f_enrich.write(json.dumps(enriched_rec) + "\n")

    df_enriched = pd.DataFrame(enriched_rows)
    logger.info("Saved %d enriched promotion records to %s", len(df_enriched), enriched_cache_path)

    # 7. Compute Oracles & Analysis
    out_tables_dir = REPO_ROOT / "results/week8/tables"
    out_figures_dir = REPO_ROOT / "results/week8/figures"
    out_tables_dir.mkdir(parents=True, exist_ok=True)
    out_figures_dir.mkdir(parents=True, exist_ok=True)

    df_o1, o1_summary = compute_oracle_1(df_promotions)
    df_sel1, sel1_summary = compute_oracle_selective_1(df_promotions, epsilon=0.01)
    df_ranks = compute_best_rank_distribution(df_o1)

    logger.info("=== ORACLE RESULTS ===")
    logger.info("Oracle-1: %s", o1_summary)
    logger.info("Oracle-Selective-1: %s", sel1_summary)

    # Heuristic comparisons
    df_heuristics = compute_heuristic_baselines(df_promotions, df_o1, df_sel1)
    logger.info("\nHeuristic Baseline Comparison:\n%s", df_heuristics.to_string(index=False))

    # Evidence utilization taxonomy & cross-tab
    util_summary = compute_evidence_utilization_summary(df_enriched)
    logger.info("\nEvidence Utilization Summary:\n%s", json.dumps(util_summary, indent=2))

    # Cross-tab: is_gold_evidence x utility_label
    crosstab_utility = pd.crosstab(
        df_enriched["is_gold_evidence"],
        df_enriched["utility_label"],
        margins=True,
    )
    # Cross-tab: is_gold_evidence x evidence_utilization_type
    crosstab_taxonomy = pd.crosstab(
        df_enriched["is_gold_evidence"],
        df_enriched["evidence_utilization_type"],
        margins=True,
    )

    logger.info("\nCross-Tab (is_gold_evidence x utility_label):\n%s", crosstab_utility)
    logger.info("\nCross-Tab (is_gold_evidence x taxonomy):\n%s", crosstab_taxonomy)

    # Case studies Q1811 & Q1812
    case_studies = extract_case_studies(df_enriched)

    # Save summary tables
    df_o1.to_csv(out_tables_dir / f"w8a_oracle1_{split}{tag}.csv", index=False)
    df_sel1.to_csv(out_tables_dir / f"w8a_oracle_selective1_{split}{tag}.csv", index=False)
    df_ranks.to_csv(out_tables_dir / f"w8a_rank_distribution_{split}{tag}.csv", index=False)
    df_heuristics.to_csv(out_tables_dir / f"w8a_heuristic_comparison_{split}{tag}.csv", index=False)
    crosstab_taxonomy.to_csv(out_tables_dir / f"w8a_evidence_taxonomy_crosstab_{split}{tag}.csv")
    with open(out_tables_dir / f"w8a_case_studies_q1811_q1812{tag}.json", "w", encoding="utf-8") as f:
        json.dump(case_studies, f, indent=2)
    with open(out_tables_dir / f"w8a_evidence_utilization_summary_{split}{tag}.json", "w", encoding="utf-8") as f:
        json.dump(util_summary, f, indent=2)

    plot_oracle_results(df_o1, df_sel1, out_figures_dir)
    logger.info("Canonical W8-A evaluation complete! All artifacts saved to %s", out_tables_dir)


if __name__ == "__main__":
    main()
