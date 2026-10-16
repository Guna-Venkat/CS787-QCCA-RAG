from __future__ import annotations

import sys
from pathlib import Path
REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
import argparse
import logging
import sys
from pathlib import Path

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
from src.week8.draft_generation import execute_resumable_draft_generation
from src.week8.self_consistency import run_self_consistency_sampling
from src.week8.uncertainty import (
    compute_uncertainty_features_for_row,
    evaluate_sufficiency_models,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


def main():
    parser = argparse.ArgumentParser(description="W8-B Draft Diagnostics Runner")
    parser.add_argument("--smoke", action="store_true", help="Run smoke test on 5 queries")
    parser.add_argument("--full", action="store_true", help="Run full split evaluation")
    parser.add_argument("--split", choices=["fit", "internal_val"], default="internal_val")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--smoke-limit", type=int, default=5)
    args = parser.parse_args()

    cfg = load_config()
    split = args.split
    smoke_limit = args.smoke_limit if args.smoke else None
    tag = "_smoke" if args.smoke else ""

    logger.info("Starting W8-B runner (split=%s, smoke=%s, device=%s)", split, args.smoke, args.device)

    records, sfr_dict, manifest_df = load_week8_split(split, cfg)
    scorer = SemanticScorer()

    logger.info("Precomputing semantic similarity scores...")
    semantic_scores_by_qid = {}
    for r in records[:smoke_limit] if smoke_limit else records:
        qid = str(r["question_id"])
        q_text = r["question"]
        passages = [p["text"] for p in r.get("passages", [])[:10]]
        sims = scorer.score_query_passages(q_text, passages).cpu().numpy()
        semantic_scores_by_qid[qid] = sims

    sara_ckpt_path = REPO_ROOT / cfg["sara"]["checkpoint_path"]
    logger.info("Loading SARA generator from %s...", sara_ckpt_path)
    sara_model, sara_tokenizer = load_sara_for_eval(str(sara_ckpt_path), device=args.device)

    # 1. Draft Generation with diagnostics
    draft_cache_path = REPO_ROOT / "results/week8/drafts" / f"draft_diagnostics_{split}{tag}.jsonl"
    df_drafts = execute_resumable_draft_generation(
        sara_model=sara_model,
        sara_tokenizer=sara_tokenizer,
        records=records,
        semantic_scores_by_qid=semantic_scores_by_qid,
        cached_sfr_embeddings=sfr_dict,
        cache_path=draft_cache_path,
        device=args.device,
        smoke_limit=smoke_limit,
    )
    logger.info("Generated %d drafts with diagnostics.", len(df_drafts))

    # 2. Extract U1..U12 features
    u_rows = []
    for _, row in df_drafts.iterrows():
        u_dict = compute_uncertainty_features_for_row(row.to_dict(), semantic_scorer=scorer)
        u_dict["question_id"] = str(row["question_id"])
        u_rows.append(u_dict)
    df_u = pd.DataFrame(u_rows)

    out_tables = REPO_ROOT / "results/week8/tables"
    out_tables.mkdir(parents=True, exist_ok=True)
    df_u.to_csv(out_tables / f"w8b_uncertainty_features_{split}{tag}.csv", index=False)

    # 3. Self-Consistency Sampling (N=3, T=0.7)
    sc_cache_path = REPO_ROOT / "results/week8/drafts" / f"self_consistency_{split}{tag}.jsonl"
    df_sc = run_self_consistency_sampling(
        sara_model=sara_model,
        sara_tokenizer=sara_tokenizer,
        records=records,
        semantic_scores_by_qid=semantic_scores_by_qid,
        cached_sfr_embeddings=sfr_dict,
        cache_path=sc_cache_path,
        num_samples=3,
        temperature=0.7,
        device=args.device,
        smoke_limit=smoke_limit,
    )
    logger.info("Self-consistency sampling completed with %d records.", len(df_sc))

    logger.info("W8-B runner completed successfully. Artifacts saved to results/week8/.")


if __name__ == "__main__":
    main()
