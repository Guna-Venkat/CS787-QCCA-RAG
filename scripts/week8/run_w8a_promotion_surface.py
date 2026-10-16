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
from src.week8.promotion_oracle import (
    compute_best_rank_distribution,
    compute_oracle_1,
    compute_oracle_selective_1,
    plot_oracle_results,
)
from src.week8.promotion_surface import run_single_promotion_surface

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


def main():
    parser = argparse.ArgumentParser(description="W8-A Promotion Response Surface Runner")
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

    logger.info("Starting W8-A runner (split=%s, smoke=%s, device=%s)", split, args.smoke, args.device)

    # 1. Load data
    records, sfr_dict, manifest_df = load_week8_split(split, cfg)
    logger.info("Loaded %d queries for split %s", len(records), split)

    # 2. Semantic scorer
    scorer = SemanticScorer()
    logger.info("Precomputing semantic similarity scores...")
    semantic_scores_by_qid = {}
    for r in records[:smoke_limit] if smoke_limit else records:
        qid = str(r["question_id"])
        q_text = r["question"]
        passages = [p["text"] for p in r.get("passages", [])[:10]]
        sims = scorer.score_query_passages(q_text, passages).cpu().numpy()
        semantic_scores_by_qid[qid] = sims

    # 3. Model & Tokenizer
    sara_ckpt_path = REPO_ROOT / cfg["sara"]["checkpoint_path"]
    logger.info("Loading SARA generator from %s...", sara_ckpt_path)
    sara_model, sara_tokenizer = load_sara_for_eval(str(sara_ckpt_path), device=args.device)

    # 4. Run response surface
    cache_path = REPO_ROOT / "results/week8/surfaces" / f"promotion_surface_{split}{tag}.jsonl"
    df_promotions = run_single_promotion_surface(
        sara_model=sara_model,
        sara_tokenizer=sara_tokenizer,
        records=records,
        semantic_scores_by_qid=semantic_scores_by_qid,
        cached_sfr_embeddings=sfr_dict,
        cache_path=cache_path,
        device=args.device,
        smoke_limit=smoke_limit,
    )

    logger.info("Promotion surface completed with %d records.", len(df_promotions))

    # 5. Compute Oracles & Analysis
    out_tables_dir = REPO_ROOT / "results/week8/tables"
    out_figures_dir = REPO_ROOT / "results/week8/figures"
    out_tables_dir.mkdir(parents=True, exist_ok=True)
    out_figures_dir.mkdir(parents=True, exist_ok=True)

    df_o1, o1_summary = compute_oracle_1(df_promotions)
    df_sel1, sel1_summary = compute_oracle_selective_1(df_promotions, epsilon=0.01)
    df_ranks = compute_best_rank_distribution(df_o1)

    logger.info("Oracle-1: %s", o1_summary)
    logger.info("Oracle-Selective-1: %s", sel1_summary)

    # Save summary tables
    df_o1.to_csv(out_tables_dir / f"w8a_oracle1_{split}{tag}.csv", index=False)
    df_sel1.to_csv(out_tables_dir / f"w8a_oracle_selective1_{split}{tag}.csv", index=False)
    df_ranks.to_csv(out_tables_dir / f"w8a_rank_distribution_{split}{tag}.csv", index=False)

    plot_oracle_results(df_o1, df_sel1, out_figures_dir)
    logger.info("W8-A completed successfully. Results saved to %s", out_tables_dir)


if __name__ == "__main__":
    main()
