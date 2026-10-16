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
from src.week8.controller import DAFiRController
from src.week8.cost_accounting import summarize_cost_records
from src.week8.data import load_config, load_week8_split
from src.week8.generation import run_dafir_query

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


def main():
    parser = argparse.ArgumentParser(description="W8-H Full DAFiR System Runner")
    parser.add_argument("--smoke", action="store_true", help="Run smoke test on 5 queries")
    parser.add_argument("--full", action="store_true", help="Run full evaluation")
    parser.add_argument("--policy", choices=["dafir1", "dafir2"], default="dafir1")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--smoke-limit", type=int, default=5)
    args = parser.parse_args()

    cfg = load_config()
    tag = "_smoke" if args.smoke else ""
    smoke_limit = args.smoke_limit if args.smoke else None
    max_proms = 1 if args.policy == "dafir1" else 2

    logger.info("Starting W8-H Runner (policy=%s, smoke=%s, device=%s)", args.policy, args.smoke, args.device)

    # 1. Load data
    records, sfr_dict, _ = load_week8_split("internal_val", cfg)
    items = records[:smoke_limit] if smoke_limit else records

    # 2. SARA model
    sara_ckpt_path = REPO_ROOT / cfg["sara"]["checkpoint_path"]
    logger.info("Loading SARA generator from %s...", sara_ckpt_path)
    sara_model, sara_tokenizer = load_sara_for_eval(str(sara_ckpt_path), device=args.device)

    # 3. Setup controller
    controller = DAFiRController(
        tau_help=0.4,
        tau_hurt=0.3,
        lambda_h=1.0,
        lambda_d=0.5,
        max_promotions=max_proms,
    )

    # 4. Scorer mock/function
    def candidate_scorer(query_rec: dict, draft: str, candidate_indices: list) -> list:
        # Mock/proxy utility scoring for execution pipeline
        scored = []
        for idx in candidate_indices:
            scored.append({
                "candidate_idx": idx,
                "prob_help": 0.6 if idx == candidate_indices[0] else 0.2,
                "prob_hurt": 0.05,
                "pred_delta_f1": 0.04 if idx == candidate_indices[0] else 0.005,
            })
        return scored

    # 5. Execute generation
    results = []
    cost_records = []
    for item in items:
        res = run_dafir_query(
            sara_model=sara_model,
            sara_tokenizer=sara_tokenizer,
            query_record=item,
            sfr_embeddings_by_qid=sfr_dict,
            controller=controller,
            candidate_scorer_fn=candidate_scorer,
            max_promotions=max_proms,
            device=args.device,
        )
        cost_records.append(res.pop("cost_record"))
        results.append(res)

    df_res = pd.DataFrame(results)
    summary_costs = summarize_cost_records(cost_records)

    logger.info("Execution complete. Mean F1: %.4f", float(df_res["token_f1"].mean()))
    logger.info("Cost Accounting Summary: %s", summary_costs)

    out_tables = REPO_ROOT / "results/week8/tables"
    out_tables.mkdir(parents=True, exist_ok=True)
    df_res.to_csv(out_tables / f"w8h_{args.policy}_internal_val{tag}.csv", index=False)
    pd.DataFrame([summary_costs]).to_csv(out_tables / f"w8h_{args.policy}_cost_summary{tag}.csv", index=False)
    logger.info("Saved W8-H results to %s", out_tables)


if __name__ == "__main__":
    main()
