"""CLI entry point for FinanceBench benchmark evaluation.

FinanceBench is a financial QA benchmark over SEC filings.  This runner supports
retrieval-based (rag) and SANE pipeline (sane) methods.

Evaluation uses LLM-as-judge because FinanceBench answers are diverse — numeric,
short text, and paragraph-length explanations.  The judge compares the predicted
answer to the gold answer + human justification.

Usage::

    # Step 1: Pre-extract PDF text (one-time)
    python benchmarks/financebench/setup.py --workers 8

    # Quick test
    python -m benchmarks.financebench.run --method rag --limit 5 --judge-workers 4

    # Semantic RAG
    python -m benchmarks.financebench.run --method rag \\
        --config config/financebench.yaml --workers 10 --judge-workers 16

    # SANE pipeline
    python -m benchmarks.financebench.run --method sane \\
        --config config/financebench.yaml --workers 10 --judge-workers 16
"""

from __future__ import annotations

import argparse
import json
import os
import time
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass

from benchmarks.metrics import load_metrics, print_efficiency_summary, save_eval_summary
from sane.llm.openai_client import OpenAIClient


# ── Judge prompt ────────────────────────────────────────────────────────────

def _build_judge_prompt(question: str, gold_answer: str, justification: str,
                        model_response: str) -> str:
    """Build an LLM-as-judge prompt for FinanceBench evaluation.

    The judge compares the model's response to the gold answer, using the
    human justification as additional context for what constitutes a correct
    answer.  This handles the diversity of FinanceBench answers (numeric,
    short text, paragraph-length explanations).
    """
    return (
        "You are evaluating a financial QA model's response. "
        "I will give you a question, the correct answer, a justification "
        "explaining how the answer was derived, and the model's response.\n\n"
        "Judge whether the model's response is CORRECT. The model's response "
        "is correct if it:\n"
        "- Contains the same factual information as the correct answer\n"
        "- Is numerically equivalent (within rounding tolerance for financial figures)\n"
        "- Expresses the same conclusion even if worded differently\n"
        "- For yes/no questions, correctly answers yes/no with supporting reasoning\n\n"
        "The model's response does NOT need to match verbatim. Semantic equivalence "
        "is sufficient.\n\n"
        f"Question: {question}\n\n"
        f"Correct Answer: {gold_answer}\n\n"
        f"Justification: {justification}\n\n"
        f"Model Response: {model_response}\n\n"
        "Is the model's response correct? Answer YES or NO only."
    )


# ── Judge infrastructure ────────────────────────────────────────────────────

@dataclass(frozen=True)
class _JudgeSettings:
    model: str
    base_url: str
    api_key: str
    max_tokens: int
    temperature: float


def _judge_prediction(
    entry: dict,
    settings: _JudgeSettings,
    client: OpenAIClient,
) -> tuple[dict, str]:
    """Judge a single prediction.  Returns (judged_entry, question_type)."""
    prompt = _build_judge_prompt(
        question=entry.get("question", ""),
        gold_answer=entry.get("answer", ""),
        justification=entry.get("justification", ""),
        model_response=entry.get("hypothesis", ""),
    )
    response = client.chat(
        model=settings.model,
        messages=[{"role": "user", "content": prompt}],
        n=1,
        temperature=settings.temperature,
        max_tokens=settings.max_tokens,
    )
    eval_response = (response.get("content") or "").strip()
    label = eval_response.upper().startswith("YES")
    judged_entry = dict(entry)
    judged_entry["autoeval_label"] = {
        "model": settings.model,
        "label": label,
        "raw": eval_response,
    }
    return judged_entry, entry.get("question_type", "")


def _evaluate_predictions(
    *,
    predictions: list[dict],
    settings: _JudgeSettings,
    judge_workers: int,
) -> tuple[list[dict], dict[str, list[int]], dict]:
    """Run LLM-as-judge evaluation over predictions in parallel.

    Returns (judged_logs, qtype2acc, usage_summary).
    """
    judge_workers = max(1, int(judge_workers or 1))
    client = OpenAIClient(
        base_url=settings.base_url,
        api_key=settings.api_key,
        model=settings.model,
        max_tokens=settings.max_tokens,
        temperature=settings.temperature,
    )
    jobs = list(enumerate(predictions))

    results_by_index = {}
    qtype2acc: dict[str, list[int]] = defaultdict(list)

    if judge_workers == 1 or len(jobs) <= 1:
        for index, entry in jobs:
            judged_entry, qtype = _judge_prediction(entry, settings, client)
            results_by_index[index] = judged_entry
            qtype2acc[qtype].append(1 if judged_entry["autoeval_label"]["label"] else 0)
    else:
        with ThreadPoolExecutor(max_workers=min(judge_workers, len(jobs))) as executor:
            futures = {
                executor.submit(_judge_prediction, entry, settings, client): index
                for index, entry in jobs
            }
            for future in as_completed(futures):
                index = futures[future]
                judged_entry, qtype = future.result()
                results_by_index[index] = judged_entry
                qtype2acc[qtype].append(1 if judged_entry["autoeval_label"]["label"] else 0)

    logs = [results_by_index[i] for i in sorted(results_by_index)]
    return logs, dict(qtype2acc), client.get_usage_summary()


# ── Helpers ──────────────────────────────────────────────────────────────────

def _default_output_dir() -> str:
    project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    return os.path.join(project_root, "output", "benchmarks", "financebench")


# ── CLI ─────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="FinanceBench: financial QA benchmark evaluation over SEC filings",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""Methods:
  rag    Chunk PDF pages → retrieval (top-k) → LLM
  sane   Chunk → wide retrieval → synopsis → LM selection → evidence → LLM

Examples:
  # Quick test: rag, 5 instances
  python -m benchmarks.financebench.run --method rag --limit 5 --judge-workers 4

  # Semantic RAG, 10 workers
  python -m benchmarks.financebench.run --method rag --config config/financebench.yaml \\
      --workers 10 --judge-workers 16

  # SANE pipeline, 10 workers
  python -m benchmarks.financebench.run --method sane --config config/financebench.yaml \\
      --workers 10 --judge-workers 16
""",
    )
    parser.add_argument("--limit", type=int, default=None,
                        help="Limit number of instances for quick testing")
    parser.add_argument("--method", default=None,
                        choices=["rag", "sane"],
                        help="Agent method (default: from config)")
    parser.add_argument("--output-dir", default=None,
                        help="Output directory for predictions")
    parser.add_argument("--workers", type=int, default=1,
                        help="Number of parallel workers (default: 1)")
    parser.add_argument("--judge-workers", type=int, default=8,
                        help="Number of parallel workers for judge evaluation (default: 8)")
    parser.add_argument("--config", type=str, default=None,
                        help="Path to config YAML (default: config/financebench.yaml)")
    parser.add_argument("--model", type=str, default=None,
                        help="Override the LLM model (e.g. qwen3.5-9b, Qwen/Qwen3.5-4B)")
    parser.add_argument("--judge-model", type=str, default=None,
                        help="Override the judge model (default: same as config)")
    parser.add_argument("--overrides", type=str, default=None,
                        help="Comma-separated key=value overrides for method config")
    parser.add_argument("--eval-only", type=str, default=None,
                        help="Skip generation, re-evaluate existing predictions file")
    args = parser.parse_args()

    output_dir = args.output_dir or _default_output_dir()
    os.makedirs(output_dir, exist_ok=True)

    # Resolve default config
    config_path = args.config
    if config_path is None:
        project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        default_config = os.path.join(project_root, "config", "financebench.yaml")
        if os.path.exists(default_config):
            config_path = default_config

    timestamp = time.strftime("%Y%m%d_%H%M%S")
    method_tag = args.method or "default"
    model_tag = (args.model or "default").replace("/", "-")

    # ── Judge settings ─────────────────────────────────────────────────
    from sane.core.config import load_config, resolve_judge_config
    import yaml

    cfg = None
    judge_cfg: dict = {}
    if config_path and os.path.exists(config_path):
        cfg = load_config(config_path)
        with open(config_path) as f:
            judge_cfg = (yaml.safe_load(f) or {}).get("judge", {})
    resolved = resolve_judge_config(
        main_config=cfg,
        model=args.judge_model,
        judge_overrides=judge_cfg,
    )
    judge_settings = _JudgeSettings(
        model=resolved["model"],
        base_url=resolved["base_url"],
        api_key=resolved["api_key"],
        max_tokens=resolved["max_tokens"],
        temperature=resolved["temperature"],
    )
    print(f"Judge: {resolved['model']} @ {resolved['base_url']} ({args.judge_workers} worker(s))")

    # ── Eval-only path ──────────────────────────────────────────────────
    if args.eval_only:
        from benchmarks.financebench.data import load_predictions
        predictions = load_predictions(args.eval_only)
        print(f"Loaded {len(predictions)} predictions from {args.eval_only}")

        judged, qtype2acc, usage = _evaluate_predictions(
            predictions=predictions,
            settings=judge_settings,
            judge_workers=args.judge_workers,
        )

        eval_path = args.eval_only.replace(".jsonl", ".eval-results.jsonl")
        _save_and_print_results(judged, qtype2acc, usage, eval_path)
        return

    # ── Load data ──────────────────────────────────────────────────────
    from benchmarks.financebench.data import load_financebench_data
    data = load_financebench_data()
    print(f"Loaded {len(data)} instances")
    if args.limit:
        data = data[:args.limit]

    # ── Generate predictions ────────────────────────────────────────────
    from benchmarks.financebench.runner import run_financebench
    predictions_path = os.path.join(
        output_dir,
        f"predictions_financebench_{method_tag}_{model_tag}_{timestamp}.jsonl",
    )

    predictions = run_financebench(
        data=data,
        output_path=predictions_path,
        limit=None,  # already limited above
        method=args.method,
        workers=args.workers,
        config_path=config_path,
        overrides=args.overrides,
        model=args.model,
    )

    # ── Judge evaluation ────────────────────────────────────────────────
    print(f"\n{'='*60}")
    print("FinanceBench Evaluation Results (LLM-as-Judge)")
    print(f"{'='*60}")

    judged, qtype2acc, usage = _evaluate_predictions(
        predictions=predictions,
        settings=judge_settings,
        judge_workers=args.judge_workers,
    )

    eval_path = predictions_path.replace(".jsonl", ".eval-results.jsonl")
    _save_and_print_results(judged, qtype2acc, usage, eval_path)

    # ── Efficiency ─────────────────────────────────────────────────────
    efficiency = load_metrics(predictions_path)
    if efficiency:
        print_efficiency_summary(efficiency)

    print("\nDone.")


def _save_and_print_results(
    judged: list[dict],
    qtype2acc: dict[str, list[int]],
    usage: dict,
    eval_path: str,
):
    """Save judged predictions and print summary."""
    # Compute overall accuracy
    all_scores = [s for scores in qtype2acc.values() for s in scores]
    overall_acc = sum(all_scores) / len(all_scores) if all_scores else 0.0

    print(f"\nInstances:  {len(judged)}")
    print(f"Accuracy:   {overall_acc:.4f} ({sum(all_scores)}/{len(all_scores)})")

    # By question type
    if len(qtype2acc) > 1:
        print(f"\n  By question type:")
        for qtype in sorted(qtype2acc):
            scores = qtype2acc[qtype]
            acc = sum(scores) / len(scores) if scores else 0.0
            print(f"    {qtype:<24s}  Acc={acc:.4f}  (n={len(scores)})")

    # Judge usage
    if usage:
        total_input = usage.get("total_input_tokens", 0)
        total_output = usage.get("total_output_tokens", 0)
        print(f"\n  Judge LLM usage: {total_input:,} input + {total_output:,} output tokens")

    # ── Save ───────────────────────────────────────────────────────────
    # Build summary dict
    summary = {
        "total": len(judged),
        "accuracy": overall_acc,
        "correct": sum(all_scores),
        "by_question_type": {
            qtype: {
                "count": len(scores),
                "accuracy": sum(scores) / len(scores) if scores else 0.0,
                "correct": sum(scores),
            }
            for qtype, scores in qtype2acc.items()
        },
        "judge_usage": usage,
    }

    # Save judged predictions
    with open(eval_path, "w") as f:
        for entry in judged:
            f.write(json.dumps(entry, ensure_ascii=False) + "\n")
    print(f"  Saved eval results → {eval_path}")

    # Save summary
    summary_path = eval_path.replace(".eval-results.jsonl", ".summary.json")
    save_eval_summary(summary_path, summary)
    print(f"  Saved summary       → {summary_path}")


if __name__ == "__main__":
    main()
