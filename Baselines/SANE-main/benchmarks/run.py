#!/usr/bin/env python3
"""Unified benchmark runner for all benchmarks.

Runs all RAG benchmarks through a single entry point with consistent
configuration, then prints two summary tables:

* **Accuracy + Recall** — score, F1, recall@k, recall@wide, selected recall
* **Efficiency** — wall time, avg latency/instance, LM tokens, main context size,
  cost with and without KV-cache

Usage::

    # All benchmarks, SANE pipeline
    python benchmarks/run.py --method sane --workers 10 --judge-workers 16

    # Specific benchmarks, semantic RAG
    python benchmarks/run.py --method rag --benchmarks hotpot,musique --workers 20
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Any

_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_OUTPUT_BASE = os.path.join(_PROJECT_ROOT, "output", "benchmarks", "unified")

# Ensure project is on sys.path
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)


def _write_jsonl(path: str, data: list[dict]):
    """Write a list of dicts as JSONL."""
    import json as _json
    with open(path, "w") as f:
        for obj in data:
            f.write(_json.dumps(obj, ensure_ascii=False) + "\n")


# ──════
# Per-benchmark helpers
# ──════

# ── LongMemEval ────────────────────────────────────────────────────────────────

def _make_lme_handlers(split: str):
    """Return (load, run, evaluate) callables for a specific LME split."""

    def _load() -> list[dict]:
        from benchmarks.longmemeval.data import load_longmemeval_data
        return load_longmemeval_data(split=split)

    def _run(output_path: str, method: str, workers: int,
             config_path: str | None, overrides: str | None,
             model: str | None = None) -> list[dict]:
        from benchmarks.longmemeval.runner import run_longmemeval
        return run_longmemeval(
            split=split, output_path=output_path, limit=None,
            method=method, workers=workers,
            config_path=config_path, overrides=overrides,
            model=model,
        )

    def _evaluate(predictions: list[dict], judge_workers: int,
                  config_path: str | None, judge_model: str | None = None) -> dict[str, Any]:
        from benchmarks.longmemeval.data import load_longmemeval_data
        from benchmarks.longmemeval.run import _JudgeSettings, _judge_prediction
        from sane.llm.openai_client import OpenAIClient
        from sane.core.config import resolve_judge_config, load_config
        import yaml

        qid2data = {e["question_id"]: e for e in load_longmemeval_data(split=split)}

        judge_cfg: dict = {}
        main_cfg = None
        if config_path and os.path.exists(config_path):
            main_cfg = load_config(config_path)
            with open(config_path) as f:
                raw = yaml.safe_load(f) or {}
            judge_cfg = raw.get("judge", {})

        resolved = resolve_judge_config(
            main_config=main_cfg,
            model=judge_model,
            judge_overrides=judge_cfg,
        )
        settings = _JudgeSettings(
            model=resolved["model"],
            base_url=resolved["base_url"],
            api_key=resolved["api_key"],
            max_tokens=resolved["max_tokens"],
            temperature=resolved["temperature"],
        )
        client = OpenAIClient(
            base_url=settings.base_url, api_key=settings.api_key,
            model=settings.model, max_tokens=settings.max_tokens,
            temperature=settings.temperature,
        )

        jobs = [(i, entry) for i, entry in enumerate(predictions)
                if entry.get("question_id") in qid2data]
        results_by_index: dict[int, tuple[dict, str]] = {}
        qtype2acc: dict[str, list[int]] = defaultdict(list)

        if judge_workers <= 1 or len(jobs) <= 1:
            for idx, entry in jobs:
                qdata = qid2data[entry["question_id"]]
                judged_entry, qtype = _judge_prediction(entry, qdata, settings, client)
                results_by_index[idx] = (judged_entry, qtype)
                qtype2acc[qtype].append(1 if judged_entry["autoeval_label"]["label"] else 0)
        else:
            with ThreadPoolExecutor(max_workers=min(judge_workers, len(jobs))) as ex:
                futures = {
                    ex.submit(_judge_prediction, entry, qid2data[entry["question_id"]], settings, client): idx
                    for idx, entry in jobs
                }
                for future in as_completed(futures):
                    idx = futures[future]
                    judged_entry, qtype = future.result()
                    results_by_index[idx] = (judged_entry, qtype)
                    qtype2acc[qtype].append(1 if judged_entry["autoeval_label"]["label"] else 0)

        all_labels = [s for scores in qtype2acc.values() for s in scores]
        accuracy = sum(all_labels) / len(all_labels) if all_labels else 0.0
        return {"score": accuracy, "score_name": "Accuracy", "instances": len(jobs)}

    return _load, _run, _evaluate


_lme_s_load, _lme_s_run, _lme_s_eval = _make_lme_handlers("s_cleaned")
_lme_m_load, _lme_m_run, _lme_m_eval = _make_lme_handlers("m_cleaned")



# ── HotpotQA ───────────────────────────────────────────────────────────────────

def _hotpot_load() -> list[dict]:
    from benchmarks.hotpotqa.data import load_hotpotqa_data
    return load_hotpotqa_data()


def _hotpot_run(data: list[dict], output_path: str, method: str, workers: int,
                config_path: str | None, overrides: str | None,
                model: str | None = None) -> list[dict]:
    from benchmarks.hotpotqa.runner import run_hotpotqa
    return run_hotpotqa(
        data=data, output_path=output_path, method=method,
        workers=workers, config_path=config_path,
        overrides=overrides,
        model=model,
    )


def _hotpot_evaluate(predictions: list[dict], judge_workers: int,
                  config_path: str | None, judge_model: str | None = None) -> dict[str, Any]:
    """Lexical EM/F1 evaluation — no LLM judge needed."""
    from benchmarks.hotpotqa.run import evaluate_predictions
    result = evaluate_predictions(predictions, output_path=None)
    return {
        "score": result["em"],
        "score_name": "EM",
        "f1": result["f1"],
        "instances": result["total"],
    }


# ── MuSiQue ────────────────────────────────────────────────────────────────────

def _musique_load() -> list[dict]:
    from benchmarks.musique.data import load_musique_data
    return load_musique_data(variant="ans")


def _musique_run(data: list[dict], output_path: str, method: str, workers: int,
                 config_path: str | None, overrides: str | None,
                 model: str | None = None) -> list[dict]:
    from benchmarks.musique.runner import run_musique
    return run_musique(
        data=data, output_path=output_path, method=method,
        workers=workers, config_path=config_path,
        overrides=overrides,
        model=model,
    )


def _musique_evaluate(predictions: list[dict], judge_workers: int,
                  config_path: str | None, judge_model: str | None = None) -> dict[str, Any]:
    """Lexical EM/F1 evaluation."""
    from benchmarks.musique.run import evaluate_predictions
    result = evaluate_predictions(predictions, output_path=None)
    return {
        "score": result["em"],
        "score_name": "EM",
        "f1": result["f1"],
        "instances": result["total"],
    }



# ── FinanceBench ───────────────────────────────────────────────────────────────

def _fb_load() -> list[dict]:
    from benchmarks.financebench.data import load_financebench_data
    return load_financebench_data()


def _fb_run(data: list[dict], output_path: str, method: str, workers: int,
            config_path: str | None, overrides: str | None,
            model: str | None = None) -> list[dict]:
    from benchmarks.financebench.runner import run_financebench
    return run_financebench(
        data=data, output_path=output_path, method=method,
        workers=workers, config_path=config_path,
        overrides=overrides,
        model=model,
    )


def _fb_evaluate(predictions: list[dict], judge_workers: int,
                  config_path: str | None, judge_model: str | None = None) -> dict[str, Any]:
    """LLM judge evaluation."""
    from benchmarks.financebench.run import _evaluate_predictions, _JudgeSettings
    from sane.core.config import resolve_judge_config, load_config
    import yaml

    judge_cfg: dict = {}
    main_cfg = None
    if config_path and os.path.exists(config_path):
        main_cfg = load_config(config_path)
        with open(config_path) as f:
            judge_cfg = (yaml.safe_load(f) or {}).get("judge", {})

    resolved = resolve_judge_config(main_config=main_cfg, model=judge_model, judge_overrides=judge_cfg)
    settings = _JudgeSettings(
        model=resolved["model"], base_url=resolved["base_url"], api_key=resolved["api_key"],
        max_tokens=resolved["max_tokens"], temperature=resolved["temperature"],
    )

    judged, qtype2acc, _usage = _evaluate_predictions(
        predictions=predictions, settings=settings, judge_workers=judge_workers,
    )
    all_scores = [s for scores in qtype2acc.values() for s in scores]
    accuracy = sum(all_scores) / len(all_scores) if all_scores else 0.0
    return {"score": accuracy, "score_name": "Accuracy", "instances": len(judged)}


# ──════
# Benchmark registry
# ──════

BENCHMARKS: dict[str, dict[str, Any]] = {
    "lme": {
        "label": "LongMemEval (s)",
        "config": os.path.join(_PROJECT_ROOT, "config", "longmemeval.yaml"),
        "load": _lme_s_load,
        "run": _lme_s_run,
        "evaluate": _lme_s_eval,
        "accepts_data": False,
    },
    "lme_m": {
        "label": "LongMemEval (m)",
        "config": os.path.join(_PROJECT_ROOT, "config", "longmemeval.yaml"),
        "load": _lme_m_load,
        "run": _lme_m_run,
        "evaluate": _lme_m_eval,
        "accepts_data": False,
    },
    "hotpot": {
        "label": "HotpotQA",
        "config": os.path.join(_PROJECT_ROOT, "config", "hotpotqa.yaml"),
        "load": _hotpot_load,
        "run": _hotpot_run,
        "evaluate": _hotpot_evaluate,
        "accepts_data": True,
    },
    "musique": {
        "label": "MuSiQue",
        "config": os.path.join(_PROJECT_ROOT, "config", "musique.yaml"),
        "load": _musique_load,
        "run": _musique_run,
        "evaluate": _musique_evaluate,
        "accepts_data": True,
    },
    "financebench": {
        "label": "FinanceBench",
        "config": os.path.join(_PROJECT_ROOT, "config", "financebench.yaml"),
        "load": _fb_load,
        "run": _fb_run,
        "evaluate": _fb_evaluate,
        "accepts_data": True,
    },
}


# ──════
# Metrics extraction
# ──════

def _extract_metrics(predictions_path: str, eval_result: dict[str, Any]) -> dict[str, Any]:
    """Extract unified metrics from the .metrics.json sidecar + eval result."""
    from benchmarks.metrics import load_metrics, flatten_llm_totals

    m = load_metrics(predictions_path)
    row: dict[str, Any] = {
        "score": eval_result["score"],
        "score_name": eval_result["score_name"],
        "instances": eval_result["instances"],
        "f1": eval_result.get("f1"),
        "wall_time_s": 0.0,
        "avg_time_per_inst_s": 0.0,
        "lm_input_tokens": 0,
        "lm_output_tokens": 0,
        "main_ctx_tokens": 0.0,
        "recall_k": None,       # recall@k (e.g. recall@5)
        "recall_wide": None,    # recall@wide (e.g. wide_recall@20)
        "selected_recall": None, # LM-selected recall
    }

    if not m:
        return row

    run = m.get("run", {})
    row["wall_time_s"] = float(run.get("wall_time_s", 0.0) or 0.0)

    # Avg time per instance (worker-summed, not throughput)
    instance_count = int(run.get("instance_count", 0) or run.get("item_count", 0) or 0)
    sum_inst_wall = float(run.get("sum_instance_wall_time_s", 0.0)
                          or run.get("sum_item_wall_time_s", 0.0) or 0.0)
    if instance_count:
        row["avg_time_per_inst_s"] = sum_inst_wall / instance_count

    # LLM tokens and cost (flattened across all scopes, then averaged per instance)
    llm_scopes = run.get("llm", {}) or {}
    totals = flatten_llm_totals(llm_scopes)
    usage = totals.get("usage", {}) or {}
    row["lm_input_tokens"] = int(usage.get("prompt_tokens", 0) or 0)
    row["lm_output_tokens"] = int(usage.get("completion_tokens", 0) or 0)

    # Normalise token totals to per-instance averages
    if instance_count:
        row["lm_input_tokens"] = round(row["lm_input_tokens"] / instance_count)
        row["lm_output_tokens"] = round(row["lm_output_tokens"] / instance_count)

    # Main agent context (avg prompt tokens for the final answer-generation LLM call)
    main_scope = llm_scopes.get("main", {}) or {}
    main_ctx = main_scope.get("context", {}) or {}
    row["main_ctx_tokens"] = float(main_ctx.get("avg_estimated_prompt_tokens", 0.0) or 0.0)

    # Recall metrics
    recall = run.get("recall", {}) or {}
    if recall:
        # Find the "primary" recall@k — prefer topk_recall@5, then first topk_recall@*
        for key in sorted(recall):
            if key.startswith("topk_recall@") and row["recall_k"] is None:
                row["recall_k"] = recall[key]
        # Fall back to bare topk_recall if no @N variant
        if row["recall_k"] is None and "topk_recall" in recall:
            row["recall_k"] = recall["topk_recall"]

        # Wide recall — prefer wide_recall@20, then first wide_recall@*
        for key in sorted(recall):
            if key.startswith("wide_recall@") and row["recall_wide"] is None:
                row["recall_wide"] = recall[key]
        if row["recall_wide"] is None and "wide_recall" in recall:
            row["recall_wide"] = recall["wide_recall"]

        # Selected recall
        if "selected_recall" in recall:
            row["selected_recall"] = recall["selected_recall"]

    return row


# ──════
# Summary tables
# ──════

def _format_recall(val: float | None) -> str:
    if val is None:
        return "—"
    return f"{val:.4f}"


def _print_table1(results: list[dict[str, Any]]):
    """Accuracy + Recall table."""
    print(f"\n{'='*120}")
    print("TABLE 1 — E2E Accuracy + Recall")
    print(f"{'='*120}")
    header = (
        f"{'Benchmark':<16} {'Mode':<6} {'Inst':>5} "
        f"{'Score Name':<22} {'Score':>10} {'F1':>10} "
        f"{'Recall@k':>10} {'Recall@Wide':>12} {'Sel Recall':>10}"
    )
    print(header)
    print("-" * 120)
    for r in results:
        score_str = f"{r['score']:.4f}" if r["score"] <= 1.0 else f"{r['score']:.1f}"
        f1_str = f"{r['f1']:.4f}" if r.get("f1") is not None else "—"
        print(
            f"{r['benchmark']:<16} {r['instances']:>5} "
            f"{r['score_name']:<22} {score_str:>10} {f1_str:>10} "
            f"{_format_recall(r.get('recall_k')):>10} "
            f"{_format_recall(r.get('recall_wide')):>12} "
            f"{_format_recall(r.get('selected_recall')):>10}"
        )
    print("-" * 120)


def _print_table2(results: list[dict[str, Any]]):
    """Efficiency metrics table."""
    print(f"\n{'='*130}")
    print("TABLE 2 — Efficiency")
    print(f"{'='*130}")
    header = (
        f"{'Benchmark':<16} "
        f"{'Wall Time':>10} {'Avg/Inst':>9} "
        f"{'Avg In Tok':>11} {'Avg Out Tok':>11} {'Main Ctx':>9}"
    )
    print(header)
    print("-" * 130)
    for r in results:
        avg_s = f"{r['avg_time_per_inst_s']:.2f}s" if r["avg_time_per_inst_s"] else "—"
        main_ctx = f"{r['main_ctx_tokens']:.0f}" if r["main_ctx_tokens"] else "—"
        print(
            f"{r['benchmark']:<16} "
            f"{r['wall_time_s']:>9.1f}s {avg_s:>9} "
            f"{r['lm_input_tokens']:>11,} {r['lm_output_tokens']:>11,} {main_ctx:>9}"
        )
    print("-" * 130)

    # Totals: wall time is sum; tokens are weighted averages
    total_time = sum(r["wall_time_s"] for r in results)
    total_instances = sum(r["instances"] for r in results)
    if total_instances:
        avg_in = round(sum(r["lm_input_tokens"] * r["instances"] for r in results) / total_instances)
        avg_out = round(sum(r["lm_output_tokens"] * r["instances"] for r in results) / total_instances)
        avg_ctx = round(sum(r["main_ctx_tokens"] * r["instances"] for r in results) / total_instances)
    else:
        avg_in, avg_out, avg_ctx = 0, 0, 0
    print(
        f"{'TOTAL':<16} "
        f"{total_time:>9.1f}s {'':>9} "
        f"{avg_in:>11,} {avg_out:>11,} {avg_ctx:>9,}"
    )
    print(f"{'='*130}")


# ──════
# CLI
# ──════

def main():
    parser = argparse.ArgumentParser(
        description="Unified benchmark runner for all benchmarks",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""Examples:
  # All benchmarks, SANE pipeline
  python benchmarks/run.py --method sane --workers 10 --judge-workers 16

  # Specific benchmarks, semantic RAG
  python benchmarks/run.py --method rag --benchmarks hotpot,musique --workers 20
""",
    )
    parser.add_argument("--method", default="sane",
                        choices=["rag", "sane"],
                        help="Agent method (default: sane)")
    parser.add_argument("--benchmarks", default="all",
                        help="Comma-separated keys: lme,lme_m,hotpot,musique,financebench (default: all)")
    parser.add_argument("--workers", type=int, default=1,
                        help="Shared worker count for prediction (default: 1)")
    parser.add_argument("--judge-workers", type=int, default=8,
                        help="Shared worker count for LLM judge evaluation (default: 8)")
    parser.add_argument("--config", type=str, default=None,
                        help="Override config YAML path for all benchmarks")
    parser.add_argument("--model", type=str, default=None,
                        help="Override the LLM model for all benchmarks")
    parser.add_argument("--judge-model", type=str, default=None,
                        help="Override the judge model for all benchmarks")
    parser.add_argument("--overrides", type=str, default=None,
                        help="Comma-separated key=value overrides")
    parser.add_argument("--output-dir", type=str, default=None,
                        help="Override output directory")
    args = parser.parse_args()

    # ── Resolve benchmarks ──────────────────────────────────────────────────
    all_keys = list(BENCHMARKS.keys())  # preserve dict insertion order
    if args.benchmarks == "all":
        selected = list(all_keys)
    else:
        selected = [k.strip() for k in args.benchmarks.split(",") if k.strip() in all_keys]
        unknown = [k.strip() for k in args.benchmarks.split(",") if k.strip() not in all_keys]
        if unknown:
            print(f"Warning: unknown benchmark keys ignored: {unknown}")

    if not selected:
        print("No benchmarks selected. Use --benchmarks all or specify keys.")
        print(f"Available: {', '.join(sorted(all_keys))}")
        return

    # ── Output directory ────────────────────────────────────────────────────
    model_tag = (args.model or "default").replace("/", "-")
    output_dir = args.output_dir or os.path.join(
        _OUTPUT_BASE,
        time.strftime(f"unified_{args.method}_{model_tag}_%Y%m%d_%H%M%S"),
    )
    os.makedirs(output_dir, exist_ok=True)

    print(f"Method:    {args.method}")
    print(f"Benchmarks: {', '.join(selected)}")
    print(f"Workers:   {args.workers} gen, {args.judge_workers} judge")
    print(f"Output:    {output_dir}")
    print()

    # ── Run each benchmark ──────────────────────────────────────────────────
    all_rows: list[dict[str, Any]] = []

    for bm_key in selected:
        entry = BENCHMARKS[bm_key]
        label = entry["label"]
        print(f"\n{'─'*70}")
        print(f"  {label} (method={args.method})")
        print(f"{'─'*70}")

        config_path = args.config or entry["config"]
        if config_path and not os.path.exists(config_path):
            print(f"  Warning: config not found at {config_path}, using built-in defaults")
            config_path = None

        # ── Load data ──────────────────────────────────────────────────
        t0 = time.perf_counter()
        try:
            data = entry["load"]()
        except FileNotFoundError as e:
            print(f"  SKIP: {e}")
            continue
        dt_load = time.perf_counter() - t0
        print(f"  Loaded {len(data)} instances ({dt_load:.1f}s)")

        # ── Run ────────────────────────────────────────────────────────
        predictions_path = os.path.join(
            output_dir, f"predictions_{bm_key}_{args.method}_{model_tag}.jsonl",
        )

        t0 = time.perf_counter()
        if entry.get("accepts_data", True):
            predictions = entry["run"](
                data=data, output_path=predictions_path, method=args.method,
                workers=args.workers, config_path=config_path,
                overrides=args.overrides,
                model=args.model,
            )
        else:
            # LME: loads data internally via split
            predictions = entry["run"](
                output_path=predictions_path,
                method=args.method, workers=args.workers,
                config_path=config_path,
                overrides=args.overrides,
                model=args.model,
            )
        dt_run = time.perf_counter() - t0
        print(f"  Generated {len(predictions)} predictions in {dt_run:.1f}s")

        # ── Evaluate ───────────────────────────────────────────────────
        t0 = time.perf_counter()
        eval_result = entry["evaluate"](
            predictions=predictions,
            judge_workers=args.judge_workers,
            config_path=config_path,
            judge_model=args.judge_model,
        )
        dt_eval = time.perf_counter() - t0

        score_str = f"{eval_result['score']:.4f}" if eval_result["score"] <= 1.0 else f"{eval_result['score']:.1f}"
        eval_detail = f"{eval_result['score_name']}={score_str}"
        if eval_result.get("f1") is not None:
            eval_detail += f", F1={eval_result['f1']:.4f}"
        print(f"  {eval_detail} ({dt_eval:.1f}s)")

        # ── Extract metrics ────────────────────────────────────────────
        metrics_row = _extract_metrics(predictions_path, eval_result)
        metrics_row["benchmark"] = label
        all_rows.append(metrics_row)

    # ── Print summary tables ────────────────────────────────────────────────
    if all_rows:
        _print_table1(all_rows)
        _print_table2(all_rows)
    else:
        print("\nNo benchmarks completed.")

    print(f"\nOutput directory: {output_dir}")
    print("Done.")


if __name__ == "__main__":
    main()
