from __future__ import annotations

import sys
from pathlib import Path
REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
import argparse
import json
import logging
from pathlib import Path

import numpy as np
import pandas as pd

from src.week8.controller import DAFiRController
from src.week8.data import load_config
from src.week8.evaluation import compute_safety_and_controller_metrics

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)
REPO_ROOT = Path(__file__).resolve().parents[2]


def main():
    parser = argparse.ArgumentParser(description="W8-E Controller Tuning Runner")
    parser.add_argument("--smoke", action="store_true", help="Run smoke sweep over small subset")
    parser.add_argument("--full", action="store_true", help="Run full parameter grid search")
    args = parser.parse_args()

    tag = "_smoke" if args.smoke else ""
    logger.info("Starting W8-E Controller Tuning Runner (smoke=%s)", args.smoke)

    # Grid definition
    if args.smoke:
        lambda_h_grid = [1.0]
        lambda_d_grid = [0.5]
        tau_h_grid = [0.4]
        tau_d_grid = [0.3]
    else:
        lambda_h_grid = [0.5, 1.0, 1.5, 2.0]
        lambda_d_grid = [0.0, 0.5, 1.0]
        tau_h_grid = [0.3, 0.4, 0.5, 0.6]
        tau_d_grid = [0.2, 0.3, 0.4]

    # Load promotion surface & utility predictions
    prom_path = REPO_ROOT / "results/week8/surfaces" / f"promotion_surface_internal_val{tag}.jsonl"
    pred_path = REPO_ROOT / "results/week8/tables" / f"w8c_utility_predictions{tag}.csv"

    if prom_path.is_file() and pred_path.is_file():
        df_prom = pd.read_json(prom_path, lines=True)
        df_preds = pd.read_csv(pred_path)
    else:
        logger.warning("Files missing. Creating mock data for parameter tuning validation.")
        df_prom = pd.DataFrame([
            {"question_id": f"q_{i}", "candidate_passage_index": 4 + (i % 6), "delta_f1": 0.05 if i % 2 == 0 else -0.05, "base_f1": 0.35, "promoted_f1": 0.40 if i % 2 == 0 else 0.30}
            for i in range(12)
        ])
        df_preds = pd.DataFrame([
            {"question_id": f"q_{i}", "u0_prob_help": 0.6 if i % 2 == 0 else 0.2, "u0_prob_hurt": 0.1 if i % 2 == 0 else 0.5, "u0_pred_delta": 0.04 if i % 2 == 0 else -0.03}
            for i in range(12)
        ])

    # Build oracle mapping
    oracle_records = {}
    for qid, grp in df_prom.groupby("question_id"):
        best_r = grp.sort_values("delta_f1", ascending=False).iloc[0]
        c_deltas = {int(r["candidate_passage_index"]): float(r["delta_f1"]) for _, r in grp.iterrows()}
        oracle_records[str(qid)] = {
            "best_candidate_idx": int(best_r["candidate_passage_index"]),
            "max_delta": float(best_r["delta_f1"]),
            "candidate_deltas": c_deltas,
        }

    best_config = {}
    best_objective = -float("inf")
    grid_records = []

    for lh in lambda_h_grid:
        for ld in lambda_d_grid:
            for th in tau_h_grid:
                for td in tau_d_grid:
                    ctrl = DAFiRController(lambda_h=lh, lambda_d=ld, tau_h=th, tau_d=td)
                    decisions = []

                    for qid, grp in df_preds.groupby("question_id"):
                        candidates = []
                        for _, r in grp.iterrows():
                            candidates.append({
                                "candidate_idx": 4,
                                "p_help": float(r.get("u0_prob_help", 0.5)),
                                "p_hurt": float(r.get("u0_prob_hurt", 0.1)),
                                "pred_delta": float(r.get("u0_pred_delta", 0.0)),
                            })
                        dec = ctrl.evaluate_single_promotion({"entropy": 0.8, "margin": 0.5}, candidates)
                        dec["question_id"] = str(qid)
                        decisions.append(dec)

                    safety = compute_safety_and_controller_metrics(decisions, oracle_records)
                    # Objective: maximize HELP precision while penalizing HURT rate
                    obj = safety["help_precision"] - 2.0 * safety["hurt_rate"]

                    rec = {
                        "lambda_h": lh,
                        "lambda_d": ld,
                        "tau_h": th,
                        "tau_d": td,
                        "objective": obj,
                        **safety,
                    }
                    grid_records.append(rec)

                    if obj > best_objective:
                        best_objective = obj
                        best_config = rec

    df_grid = pd.DataFrame(grid_records)
    out_tables = REPO_ROOT / "results/week8/tables"
    out_tables.mkdir(parents=True, exist_ok=True)
    df_grid.to_csv(out_tables / f"w8e_controller_grid_search{tag}.csv", index=False)

    best_cfg_path = REPO_ROOT / "results/week8/surfaces" / f"frozen_controller_config{tag}.json"
    best_cfg_path.parent.mkdir(parents=True, exist_ok=True)
    with open(best_cfg_path, "w", encoding="utf-8") as f:
        json.dump(best_config, f, indent=2)

    logger.info("Optimal Controller Config: %s", best_config)
    logger.info("Saved frozen config to %s", best_cfg_path)


if __name__ == "__main__":
    main()
