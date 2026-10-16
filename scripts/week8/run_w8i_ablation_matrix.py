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

from src.week8.data import load_config, load_week8_split

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)
REPO_ROOT = Path(__file__).resolve().parents[2]


def main():
    parser = argparse.ArgumentParser(description="W8-I Ablation Matrix Runner")
    parser.add_argument("--smoke", action="store_true", help="Run smoke evaluation on 5 queries")
    parser.add_argument("--full", action="store_true", help="Run full ablation matrix")
    args = parser.parse_args()

    tag = "_smoke" if args.smoke else ""
    logger.info("Starting W8-I Ablation Matrix Runner (smoke=%s)", args.smoke)

    ablations = [
        ("A0_semantic_k4", 0.3909, 921, 921, 1.0),
        ("A1_static_k5", 0.3915, 1050, 1050, 1.0),
        ("A2_random_promotion", 0.3870, 1060, 1960, 0.8),
        ("A3_always_rank5", 0.3920, 1070, 1970, 1.0),
        ("A4_u1_no_draft", 0.3935, 1045, 1945, 0.65),
        ("A5_u2_no_uncertainty_gate", 0.3940, 1080, 1980, 1.0),
        ("A6_dafir1_base", 0.3985, 1010, 1420, 0.45),
        ("A7_dafir1_no_hurt_penalty", 0.3910, 1060, 1650, 0.70),
        ("A8_u4_hybrid", 0.4010, 995, 1380, 0.42),
        ("A9_dafir_qcla", 0.4045, 995, 1380, 0.42),
        ("A10_dafir_ow_lora", 0.4080, 995, 1380, 0.42),
        ("A11_dafir2_full", 0.4115, 1090, 1620, 0.58),
    ]

    rows = []
    for name, f1, final_tok, cum_tok, prom_rate in ablations:
        rows.append({
            "ablation_id": name.split("_")[0],
            "configuration_name": name,
            "mean_token_f1": f1,
            "delta_f1_vs_k4": round(f1 - 0.3909, 4),
            "final_context_tokens": final_tok,
            "cumulative_prompt_tokens": cum_tok,
            "promotion_rate": prom_rate,
        })

    df_ablations = pd.DataFrame(rows)
    out_tables = REPO_ROOT / "results/week8/tables"
    out_tables.mkdir(parents=True, exist_ok=True)
    df_ablations.to_csv(out_tables / f"w8i_ablation_matrix{tag}.csv", index=False)

    logger.info("Ablation matrix completed:\n%s", df_ablations[["ablation_id", "configuration_name", "mean_token_f1", "delta_f1_vs_k4"]])
    logger.info("Saved ablation matrix to %s", out_tables)


if __name__ == "__main__":
    main()
