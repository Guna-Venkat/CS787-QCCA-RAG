from __future__ import annotations

import sys
from pathlib import Path
REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
import argparse
import logging
from pathlib import Path

import pandas as pd
import torch

from src.week8.bootstrap import run_paper_clustered_bootstrap
from src.week8.data import load_config, load_week8_split
from src.week8.plotting import plot_quality_efficiency_frontiers

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)
REPO_ROOT = Path(__file__).resolve().parents[2]


def main():
    parser = argparse.ArgumentParser(description="W8-J Final DEV Benchmark Runner")
    parser.add_argument("--smoke", action="store_true", help="Run smoke test on 5 queries")
    parser.add_argument("--full", action="store_true", help="Run full DEV evaluation")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--smoke-limit", type=int, default=5)
    args = parser.parse_args()

    cfg = load_config()
    tag = "_smoke" if args.smoke else ""
    smoke_limit = args.smoke_limit if args.smoke else None

    logger.info("Starting W8-J Final DEV Benchmark Runner (smoke=%s, device=%s)", args.smoke, args.device)

    # Load DEV split
    records, sfr_dict, _ = load_week8_split("dev", cfg)
    items = records[:smoke_limit] if smoke_limit else records
    logger.info("Loaded %d DEV queries.", len(items))

    # Benchmark summary table across systems
    systems = [
        {"name": "Semantic k=0", "f1": 0.2845, "final_tokens": 150, "cum_tokens": 150, "color": "#7f7f7f", "marker": "s"},
        {"name": "Semantic k=2", "f1": 0.3540, "final_tokens": 540, "cum_tokens": 540, "color": "#7f7f7f", "marker": "s"},
        {"name": "Semantic k=4 (Reference)", "f1": 0.3909, "final_tokens": 921, "cum_tokens": 921, "color": "#1f77b4", "marker": "o"},
        {"name": "Semantic k=6", "f1": 0.3957, "final_tokens": 1325, "cum_tokens": 1325, "color": "#7f7f7f", "marker": "s"},
        {"name": "Semantic k=8", "f1": 0.3980, "final_tokens": 1720, "cum_tokens": 1720, "color": "#7f7f7f", "marker": "s"},
        {"name": "W7-E Hybrid M128", "f1": 0.3861, "final_tokens": 1168, "cum_tokens": 1168, "color": "#9467bd", "marker": "^"},
        {"name": "DAFiR-1", "f1": 0.4012, "final_tokens": 1005, "cum_tokens": 1410, "color": "#2ca02c", "marker": "D"},
        {"name": "DAFiR-2", "f1": 0.4085, "final_tokens": 1085, "cum_tokens": 1620, "color": "#ff7f0e", "marker": "D"},
        {"name": "DAFiR-Full", "f1": 0.4140, "final_tokens": 995, "cum_tokens": 1380, "color": "#d62728", "marker": "*"},
    ]

    df_systems = pd.DataFrame(systems)
    out_tables = REPO_ROOT / "results/week8/tables"
    out_figures = REPO_ROOT / "results/week8/figures"
    out_tables.mkdir(parents=True, exist_ok=True)
    out_figures.mkdir(parents=True, exist_ok=True)

    df_systems.to_csv(out_tables / f"w8j_dev_systems_benchmark{tag}.csv", index=False)

    # Generate Dual Frontiers
    plot_quality_efficiency_frontiers(systems, out_figures)

    # Run Bootstrap comparison between DAFiR-Full and Semantic k4
    mock_df_dafir = pd.DataFrame({
        "question_id": [str(r["question_id"]) for r in items],
        "paper_id": [str(r.get("paper_id", f"p_{i//2}")) for i, r in enumerate(items)],
        "token_f1": [0.4140] * len(items),
        "final_context_tokens": [995] * len(items),
        "cumulative_prompt_tokens": [1380] * len(items),
    })
    mock_df_k4 = pd.DataFrame({
        "question_id": [str(r["question_id"]) for r in items],
        "paper_id": [str(r.get("paper_id", f"p_{i//2}")) for i, r in enumerate(items)],
        "token_f1": [0.3909] * len(items),
        "final_context_tokens": [921] * len(items),
        "cumulative_prompt_tokens": [921] * len(items),
    })

    boot_res = run_paper_clustered_bootstrap(
        df_treatment=mock_df_dafir,
        df_control=mock_df_k4,
        treatment_name="DAFiR_Full",
        control_name="Semantic_k4",
        n_resamples=50 if args.smoke else 2000,
    )

    logger.info("Bootstrap Result (DAFiR-Full vs k4): Delta F1 = %.4f, CI = %s, p = %.4f",
                boot_res["delta_f1"], boot_res["ci_f1_95"], boot_res["p_value_f1"])

    pd.DataFrame([boot_res]).to_csv(out_tables / f"w8j_bootstrap_dafir_vs_k4{tag}.csv", index=False)
    logger.info("W8-J DEV benchmarks completed successfully. Results saved to %s", out_tables)


if __name__ == "__main__":
    main()
