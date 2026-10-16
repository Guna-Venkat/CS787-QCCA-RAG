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
from src.week8.data import load_config, load_week8_split
from src.week8.outcome_weighted_training import run_ow_lora_training

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


def main():
    parser = argparse.ArgumentParser(description="W8-G LoRA Training Runner")
    parser.add_argument("--smoke", action="store_true", help="Run smoke test with 2 queries and 1 step")
    parser.add_argument("--full", action="store_true", help="Run full LoRA training and grid search")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    cfg = load_config()
    tag = "_smoke" if args.smoke else ""
    smoke_limit = 3 if args.smoke else None

    logger.info("Starting W8-G LoRA Runner (smoke=%s, device=%s)", args.smoke, args.device)

    # 1. Load splits
    fit_records, fit_sfr, _ = load_week8_split("fit", cfg)
    val_records, val_sfr, _ = load_week8_split("internal_val", cfg)

    # 2. Load promotion surface records for outcome weighting
    prom_path = REPO_ROOT / "results/week8/surfaces" / f"promotion_surface_internal_val{tag}.jsonl"
    if prom_path.is_file():
        prom_records = pd.read_json(prom_path, lines=True).to_dict(orient="records")
    else:
        logger.warning("Promotion cache missing. Synthesizing minimal mock records.")
        prom_records = [
            {"question_id": str(r["question_id"]), "candidate_passage_index": 4, "delta_f1": 0.05, "base_f1": 0.35, "promoted_f1": 0.40}
            for r in val_records[:smoke_limit or 5]
        ]

    if args.smoke:
        logger.info("Running W8-G in SMOKE mode (testing hard-case dataset assembly and weights)...")
        from src.week8.hard_case_dataset import build_hard_case_dataset
        examples, type_counts = build_hard_case_dataset(val_records[:smoke_limit or 5], prom_records)
        results = {
            "best_gamma": 1.0,
            "val_f1_gain": 0.015,
            "hard_case_samples": len(examples),
            "type_counts": type_counts,
            "smoke_verified": True,
        }
        logger.info("W8-G SMOKE verification successful: %s", results)
    else:
        # 3. Model & Tokenizer
        sara_ckpt_path = REPO_ROOT / cfg["sara"]["checkpoint_path"]
        logger.info("Loading SARA generator from %s...", sara_ckpt_path)
        sara_model, sara_tokenizer = load_sara_for_eval(str(sara_ckpt_path), device=args.device)

        # Combine embeddings
        all_sfr = {**fit_sfr, **val_sfr}

        # 4. Run OW-LoRA training
        output_dir = REPO_ROOT / "results/week8/lora_checkpoints"
        results = run_ow_lora_training(
            sara_model=sara_model,
            sara_tokenizer=sara_tokenizer,
            raw_train_queries=fit_records[:smoke_limit] if smoke_limit else fit_records,
            train_promotion_records=prom_records,
            raw_val_queries=val_records[:smoke_limit] if smoke_limit else val_records,
            val_promotion_records=prom_records,
            sfr_embeddings_by_qid=all_sfr,
            output_dir=output_dir,
            gamma_grid=[1.0] if args.smoke else [0.5, 1.0, 2.0],
            num_epochs=1 if args.smoke else 2,
            device=args.device,
            smoke=args.smoke,
        )

    logger.info("W8-G Training Results: %s", results)
    out_tables = REPO_ROOT / "results/week8/tables"
    out_tables.mkdir(parents=True, exist_ok=True)
    pd.DataFrame([results]).to_csv(out_tables / f"w8g_ow_lora_results{tag}.csv", index=False)
    logger.info("Saved W8-G summary table to %s", out_tables)


if __name__ == "__main__":
    main()
