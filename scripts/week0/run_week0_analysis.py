"""Execution Driver for Week 0: Dataset + RAG Evidence Understanding Study.

Orchestrates:
  - Part A: Full characterization of QASPER Train and Dev splits + BM25 retrieval diagnostics.
  - Part B: Controlled sampling and layer-wise representation evolution analysis.
  - Part C: Causal attention mass, intervention importance, concentration metrics, and correlations.
  - Generates all machine-readable CSV tables, publication figures, and detailed markdown reports.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
from pathlib import Path
from typing import Any, Dict, List

import pandas as pd
import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# Ensure SARA-main src is added to src.__path__
SARA_ROOT = PROJECT_ROOT / "Baselines" / "SARA-main"
import src
if hasattr(src, "__path__") and str(SARA_ROOT / "src") not in src.__path__:
    src.__path__.append(str(SARA_ROOT / "src"))

from src.week0.dataset_adapter import QASPERAdapter
from src.week0.dataset_analyzer import analyze_dataset
from src.week0.layer_analyzer import (
    compute_layer_similarity_metrics,
    extract_layer_representations,
    locate_token_spans,
    run_mock_layer_analysis,
    select_controlled_sample,
)
from src.week0.evidence_utilization import (
    compute_concentration_metrics,
    compute_signal_correlations,
    compute_teacher_forced_gold_log_prob,
    extract_causal_passage_attention,
    run_mock_utilization_analysis,
)
from src.week0.plotting import (
    plot_dataset_distributions,
    plot_evidence_characteristics,
    plot_evidence_utilization_figures,
    plot_layer_similarity_evolution,
    plot_retrieval_diagnostics,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


def run_week0_pipeline(
    config_path: str | Path,
    run_gpu_models: bool = False,
    skip_part_a: bool = False,
) -> None:
    """Execute complete Week 0 analysis pipeline."""
    with open(config_path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    res_dir = PROJECT_ROOT / cfg["paths"]["results_dir"]
    tables_dir = PROJECT_ROOT / cfg["paths"]["tables_dir"]
    fig_dir = PROJECT_ROOT / cfg["paths"]["figures_dir"]
    dataset_dir = PROJECT_ROOT / cfg["paths"]["dataset_analysis_dir"]
    layer_dir = PROJECT_ROOT / cfg["paths"]["layer_analysis_dir"]
    evidence_dir = PROJECT_ROOT / cfg["paths"]["evidence_analysis_dir"]

    for d in [tables_dir, fig_dir, dataset_dir, layer_dir, evidence_dir]:
        d.mkdir(parents=True, exist_ok=True)

    adapter = QASPERAdapter(
        train_split_path=PROJECT_ROOT / cfg["dataset"]["train_split"],
        dev_split_path=PROJECT_ROOT / cfg["dataset"]["dev_split"],
        raw_arrow_train_path=PROJECT_ROOT / cfg["dataset"]["raw_arrow_train"],
        manifest_path=PROJECT_ROOT / cfg["dataset"]["manifest"],
        dev_retrieval_path=PROJECT_ROOT / cfg["dataset"]["dev_retrieval_file"],
    )

    # =========================================================================
    # PART A: DATASET CHARACTERIZATION
    # =========================================================================
    if not skip_part_a:
        logger.info("=" * 70)
        logger.info("STARTING PART A: DATASET CHARACTERIZATION (TRAIN & DEV)")
        logger.info("=" * 70)

        dev_results = analyze_dataset(adapter, "dev", cfg)
        train_results = analyze_dataset(adapter, "train", cfg)

        # Combine summary tables
        df_summary = pd.concat([train_results["summary"], dev_results["summary"]], ignore_index=True)
        df_questions = pd.concat([train_results["questions"], dev_results["questions"]], ignore_index=True)
        df_papers = pd.concat([train_results["papers"].assign(split="train"), dev_results["papers"].assign(split="dev")], ignore_index=True)
        df_evidence = pd.concat([train_results["evidence"], dev_results["evidence"]], ignore_index=True)
        df_retrieval = dev_results["retrieval"]

        # Save Part A machine-readable tables
        df_summary.to_csv(tables_dir / "dataset_summary.csv", index=False)
        df_questions.to_csv(tables_dir / "question_level_statistics.csv", index=False)
        df_papers.to_csv(tables_dir / "paper_level_statistics.csv", index=False)
        df_evidence.to_csv(tables_dir / "evidence_statistics.csv", index=False)
        if not df_retrieval.empty:
            df_retrieval.to_csv(tables_dir / "retrieval_statistics.csv", index=False)

        # Duplicate to dataset_analysis directory for convenience
        df_summary.to_csv(dataset_dir / "dataset_summary.csv", index=False)
        df_questions.to_csv(dataset_dir / "question_level_statistics.csv", index=False)
        df_papers.to_csv(dataset_dir / "paper_level_statistics.csv", index=False)
        df_evidence.to_csv(dataset_dir / "evidence_statistics.csv", index=False)
        if not df_retrieval.empty:
            df_retrieval.to_csv(dataset_dir / "retrieval_statistics.csv", index=False)

        # Generate Part A figures
        dev_q = dev_results["questions"]
        dev_p = dev_results["papers"]
        dev_ev = dev_results["evidence"]
        plot_dataset_distributions(dev_q, dev_p, fig_dir / "dataset_distributions.png")
        plot_evidence_characteristics(dev_ev, dev_q, fig_dir / "evidence_characteristics.png")
        if not df_retrieval.empty:
            plot_retrieval_diagnostics(df_retrieval, fig_dir / "retrieval_diagnostics.png")

        logger.info("Part A completed successfully. CSV tables and figures generated.")

    # =========================================================================
    # PART B: MISTRAL LAYER / REPRESENTATION ANALYSIS
    # =========================================================================
    logger.info("=" * 70)
    logger.info("PART B: MISTRAL LAYER ANALYSIS")
    logger.info("=" * 70)

    if not run_gpu_models:
        logger.warning(
            "Production pipeline executed without --run-gpu. "
            "Part B & C empirical execution is marked PENDING_REAL_GPU_EXECUTION. "
            "Mock outputs will NOT be written to empirical results paths."
        )
        status_file = res_dir / "gpu_status.json"
        with open(status_file, "w", encoding="utf-8") as f:
            json.dump({
                "part_b_status": "PENDING_REAL_GPU_EXECUTION",
                "part_c_status": "PENDING_REAL_GPU_EXECUTION",
                "message": "Execution invoked without --run-gpu. No mock data written.",
            }, f, indent=2)
        return

    import torch
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is not available, but --run-gpu was specified. Aborting to avoid mock execution.")

    dev_records = adapter.load_split("dev")
    sample_size = cfg["sampling"].get("layer_analysis_sample_size", 40)
    sample_records = select_controlled_sample(
        records=dev_records,
        adapter=adapter,
        sample_size=sample_size,
        seed=cfg.get("seed", 42),
    )

    selected_ids = [str(r.get("id", r.get("question_id", ""))) for r in sample_records]
    with open(layer_dir / "selected_examples.json", "w", encoding="utf-8") as f:
        json.dump({"sample_size": len(selected_ids), "question_ids": selected_ids}, f, indent=2)

    selected_layers = cfg["models"].get("selected_layers", [0, 4, 8, 16, 24, 31])

    logger.info("Running live GPU extraction with Mistral-7B...")
    from src.model.loader import load_sara_for_eval
    from src.week2.sweep_runner import build_evaluation_prompt
    sara_ckpt = PROJECT_ROOT / cfg["models"]["sara_checkpoint"]
    device = cfg["models"].get("device", "cuda")
    model, tokenizer = load_sara_for_eval(str(sara_ckpt), device=device)

    long_sim_rows: List[Dict[str, Any]] = []
    summary_rows: List[Dict[str, Any]] = []

    for record in sample_records:
        qid = str(record.get("id", record.get("question_id", "")))
        pid = str(record.get("example_id", record.get("paper_id", "")))
        question = adapter.get_query(record)
        passages = adapter.get_retrieved_passages(record)
        gold_ev = adapter.get_gold_evidence(record)
        if not passages:
            continue

        prompt, _ = build_evaluation_prompt(question, passages, k=10, tokenizer=tokenizer)
        spans = locate_token_spans(prompt, question, passages, tokenizer)
        layer_reps = extract_layer_representations(
            model=model,
            tokenizer=tokenizer,
            prompt=prompt,
            spans=spans,
            selected_layers=selected_layers,
            device=device,
        )
        sim_long, sim_summary = compute_layer_similarity_metrics(layer_reps, passages, gold_ev)
        for row in sim_long:
            row["question_id"] = qid
            row["paper_id"] = pid
            long_sim_rows.append(row)
        sim_summary["question_id"] = qid
        sim_summary["paper_id"] = pid
        summary_rows.append(sim_summary)

    df_layer_long = pd.DataFrame(long_sim_rows)
    df_layer_summary = pd.DataFrame(summary_rows)

    df_layer_long.to_csv(tables_dir / "layer_similarity_long.csv", index=False)
    df_layer_summary.to_csv(tables_dir / "layer_representation_summary.csv", index=False)
    df_layer_long.to_csv(layer_dir / "layer_similarity_long.csv", index=False)
    df_layer_summary.to_csv(layer_dir / "layer_representation_summary.csv", index=False)

    plot_layer_similarity_evolution(
        df_long=df_layer_long,
        output_similarity_path=fig_dir / "query_passage_similarity_by_layer.png",
        output_evidence_path=fig_dir / "evidence_vs_nonevidence_similarity.png",
        output_drift_path=fig_dir / "representation_evolution.png",
    )

    logger.info("Part B completed successfully.")

    # =========================================================================
    # PART C: PASSAGE UTILIZATION & ATTENTION DIAGNOSTICS
    # =========================================================================
    logger.info("=" * 70)
    logger.info("STARTING PART C: RETRIEVED PASSAGE UTILIZATION (GPU)")
    logger.info("=" * 70)

    # Clean up model
    del model
    del tokenizer
    import gc
    gc.collect()
    torch.cuda.empty_cache()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="configs/week0/week0.yaml")
    parser.add_argument("--run-gpu", action="store_true", help="Execute live GPU extraction with Mistral")
    args = parser.parse_args()

    run_week0_pipeline(config_path=args.config, run_gpu_models=args.run_gpu)
