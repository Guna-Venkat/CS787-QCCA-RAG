"""CLI entry point for HotpotQA benchmark evaluation.

HotpotQA is a multi-hop QA benchmark requiring reasoning across 2+ Wikipedia
documents.  This runner supports retrieval-based (rag) and
SANE pipeline (sane) methods.

Evaluation uses lexical EM + F1 (ported from the official HotpotQA eval script
and the standard lexical evaluation helpers).  No LLM judge — HotpotQA answers are short
text spans where EM/F1 is the standard metric.

Usage::

    # Quick test
    python -m benchmarks.hotpotqa.run --method rag --limit 5

    # Semantic RAG (k=10)
    python -m benchmarks.hotpotqa.run --method rag \\
        --config config/hotpotqa.yaml --workers 50

    # SANE pipeline
    python -m benchmarks.hotpotqa.run --method sane \\
        --config config/hotpotqa.yaml --workers 50
"""

from __future__ import annotations

import argparse
import json
import os
import re
import string
import time
from collections import Counter, defaultdict

from nltk.stem import PorterStemmer

_ps = PorterStemmer()

from benchmarks.metrics import load_metrics, print_efficiency_summary, save_eval_summary


# ── Lexical evaluation (ported from HotpotQA official eval) ──────

def normalize_answer(s: str) -> str:
    """Normalise an answer string for comparison.

    Follows the SQuAD / HotpotQA convention: lowercase, remove articles
    (a/an/the), strip punctuation, collapse whitespace.  ``"and"`` is also
    removed per the standard convention.
    """

    def remove_articles(text):
        return re.sub(r"\b(a|an|the|and)\b", " ", text)

    def white_space_fix(text):
        return " ".join(text.split())

    def remove_punc(text):
        exclude = set(string.punctuation)
        return "".join(ch for ch in text if ch not in exclude)

    def lower(text):
        return text.lower()

    return white_space_fix(remove_articles(remove_punc(lower(s))))


def exact_match_score(prediction: str, ground_truth: str) -> bool:
    """True iff the normalized prediction equals the normalized ground truth."""
    return normalize_answer(prediction) == normalize_answer(ground_truth)


def f1_score(prediction: str, ground_truth: str) -> float:
    """Token-level F1 after stemming (matches the standard convention)."""
    pred_tokens = [_ps.stem(w) for w in normalize_answer(prediction).split()]
    gold_tokens = [_ps.stem(w) for w in normalize_answer(ground_truth).split()]
    common = Counter(pred_tokens) & Counter(gold_tokens)
    num_same = sum(common.values())
    if num_same == 0:
        return 0.0
    precision = num_same / len(pred_tokens) if pred_tokens else 0.0
    recall = num_same / len(gold_tokens) if gold_tokens else 0.0
    if precision + recall == 0:
        return 0.0
    return (2 * precision * recall) / (precision + recall)


def _precision_recall_f1(prediction: str, ground_truth: str) -> tuple[float, float, float]:
    """Return (precision, recall, f1) for a single answer pair."""
    pred_tokens = [_ps.stem(w) for w in normalize_answer(prediction).split()]
    gold_tokens = [_ps.stem(w) for w in normalize_answer(ground_truth).split()]
    common = Counter(pred_tokens) & Counter(gold_tokens)
    num_same = sum(common.values())
    if num_same == 0:
        return (0.0, 0.0, 0.0)
    precision = num_same / len(pred_tokens) if pred_tokens else 0.0
    recall = num_same / len(gold_tokens) if gold_tokens else 0.0
    f1 = (2 * precision * recall) / (precision + recall) if precision + recall > 0 else 0.0
    return (precision, recall, f1)


# ── Evaluation orchestrator ──────────────────────────────────────────────

def evaluate_predictions(
    predictions: list[dict],
    output_path: str | None = None,
) -> dict:
    """Compute EM, F1, and breakdowns for predictions.

    Returns a summary dict suitable for serialisation.
    """
    metrics = defaultdict(float)
    type_stats: dict[str, dict[str, float]] = defaultdict(lambda: defaultdict(float))
    level_stats: dict[str, dict[str, float]] = defaultdict(lambda: defaultdict(float))
    type_level_stats: dict[str, dict[str, dict[str, float]]] = defaultdict(
        lambda: defaultdict(lambda: defaultdict(float))
    )

    total = len(predictions)

    for p in predictions:
        pred = str(p.get("hypothesis", ""))
        gold = str(p.get("answer", ""))
        q_type = p.get("type", "bridge")
        level = p.get("level", "medium")

        em = float(exact_match_score(pred, gold))
        prec, rec, f1 = _precision_recall_f1(pred, gold)

        metrics["em"] += em
        metrics["f1"] += f1
        metrics["prec"] += prec
        metrics["recall"] += rec

        type_stats[q_type]["em"] += em
        type_stats[q_type]["f1"] += f1
        type_stats[q_type]["count"] += 1

        level_stats[level]["em"] += em
        level_stats[level]["f1"] += f1
        level_stats[level]["count"] += 1

        type_level_stats[q_type][level]["em"] += em
        type_level_stats[q_type][level]["f1"] += f1
        type_level_stats[q_type][level]["count"] += 1

    # Normalise
    result = {
        "total": total,
        "em": metrics["em"] / total if total > 0 else 0.0,
        "f1": metrics["f1"] / total if total > 0 else 0.0,
        "prec": metrics["prec"] / total if total > 0 else 0.0,
        "recall": metrics["recall"] / total if total > 0 else 0.0,
    }

    # ── By type ──────────────────────────────────────────────────────
    by_type = {}
    for t in sorted(type_stats):
        count = int(type_stats[t]["count"])
        by_type[t] = {
            "count": count,
            "em": type_stats[t]["em"] / count if count > 0 else 0.0,
            "f1": type_stats[t]["f1"] / count if count > 0 else 0.0,
        }
    result["by_type"] = by_type

    # ── By level ─────────────────────────────────────────────────────
    by_level = {}
    for lv in sorted(level_stats):
        count = int(level_stats[lv]["count"])
        by_level[lv] = {
            "count": count,
            "em": level_stats[lv]["em"] / count if count > 0 else 0.0,
            "f1": level_stats[lv]["f1"] / count if count > 0 else 0.0,
        }
    result["by_level"] = by_level

    # ── By type × level ──────────────────────────────────────────────
    by_type_level = {}
    for t in sorted(type_level_stats):
        by_type_level[t] = {}
        for lv in sorted(type_level_stats[t]):
            count = int(type_level_stats[t][lv]["count"])
            by_type_level[t][lv] = {
                "count": count,
                "em": type_level_stats[t][lv]["em"] / count if count > 0 else 0.0,
                "f1": type_level_stats[t][lv]["f1"] / count if count > 0 else 0.0,
            }
    result["by_type_level"] = by_type_level

    # ── Save eval results ────────────────────────────────────────────
    if output_path:
        scored = []
        for p in predictions:
            pred = str(p.get("hypothesis", ""))
            gold = str(p.get("answer", ""))
            em = exact_match_score(pred, gold)
            f1 = f1_score(pred, gold)
            scored.append({**p, "em": float(em), "f1": f1})
        with open(output_path, "w") as f:
            for s in scored:
                f.write(json.dumps(s, ensure_ascii=False) + "\n")

        summary_path = output_path.replace(".eval-results.jsonl", ".summary.json")
        save_eval_summary(summary_path, result)
        print(f"  Saved eval results → {output_path}")
        print(f"  Saved summary       → {summary_path}")

    return result


def _default_output_dir() -> str:
    project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    return os.path.join(project_root, "output", "benchmarks", "hotpotqa")


# ── CLI ─────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="HotpotQA: multi-hop QA benchmark evaluation",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""Methods:
  rag    Chunk → semantic retrieval (top-k) → LLM
  sane   Chunk → wide retrieval → synopsis → LM selection → evidence → LLM

Examples:
  # Quick test: rag, 5 instances
  python -m benchmarks.hotpotqa.run --method rag --limit 5

  # Semantic RAG, 50 workers
  python -m benchmarks.hotpotqa.run --method rag --config config/hotpotqa.yaml \\
      --workers 50

  # SANE pipeline, 50 workers
  python -m benchmarks.hotpotqa.run --method sane --config config/hotpotqa.yaml \\
      --workers 50
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
    parser.add_argument("--config", type=str, default=None,
                        help="Path to config YAML (default: config/hotpotqa.yaml)")
    parser.add_argument("--overrides", type=str, default=None,
                        help="Comma-separated key=value overrides for method config")
    parser.add_argument("--eval-only", type=str, default=None,
                        help="Skip generation, re-evaluate existing predictions file")
    parser.add_argument("--split", type=str, default="validation",
                        choices=["validation", "train"],
                        help="Dataset split (default: validation)")
    parser.add_argument("--model", type=str, default=None,
                        help="Override agent LLM model (default: from config)")
    parser.add_argument("--judge-model", type=str, default=None,
                        help="Override judge LLM model (not used for HotpotQA lexical eval)")
    args = parser.parse_args()

    output_dir = args.output_dir or _default_output_dir()
    os.makedirs(output_dir, exist_ok=True)

    # Resolve default config
    config_path = args.config
    if config_path is None:
        project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        default_config = os.path.join(project_root, "config", "hotpotqa.yaml")
        if os.path.exists(default_config):
            config_path = default_config

    timestamp = time.strftime("%Y%m%d_%H%M%S")
    method_tag = args.method or "default"

    # ── Eval-only path ────────────────────────────────────────────────
    if args.eval_only:
        from benchmarks.hotpotqa.data import load_predictions
        predictions = load_predictions(args.eval_only)
        print(f"Loaded {len(predictions)} predictions from {args.eval_only}")
        eval_path = args.eval_only.replace(".jsonl", ".eval-results.jsonl")
        result = evaluate_predictions(predictions, output_path=eval_path)
        _print_eval_summary(result)
        return

    # ── Load data ────────────────────────────────────────────────────
    from benchmarks.hotpotqa.data import load_hotpotqa_data
    data = load_hotpotqa_data(split=args.split)
    print(f"Loaded {len(data)} instances (split={args.split})")
    if args.limit:
        data = data[:args.limit]

    # ── Generate predictions ──────────────────────────────────────────
    from benchmarks.hotpotqa.runner import run_hotpotqa
    model_tag = (args.model or "default").replace("/", "-")
    predictions_path = os.path.join(
        output_dir,
        f"predictions_hotpotqa_{method_tag}_{model_tag}_{args.split}_{timestamp}.jsonl",
    )

    predictions = run_hotpotqa(
        data=data,
        output_path=predictions_path,
        limit=None,  # already limited above
        method=args.method,
        workers=args.workers,
        config_path=config_path,
        overrides=args.overrides,
        model=args.model,
    )

    # ── Evaluate ─────────────────────────────────────────────────────
    print(f"\n{'='*60}")
    print("HotpotQA Evaluation Results")
    print(f"{'='*60}")

    eval_path = predictions_path.replace(".jsonl", ".eval-results.jsonl")
    result = evaluate_predictions(predictions, output_path=eval_path)
    _print_eval_summary(result)

    # ── Efficiency ───────────────────────────────────────────────────
    efficiency = load_metrics(predictions_path)
    if efficiency:
        print_efficiency_summary(efficiency)

    print("\nDone.")


def _print_eval_summary(result: dict):
    """Print evaluation results in a readable table format."""
    total = result["total"]
    print(f"\nInstances:  {total}")
    print(f"Answer EM:  {result['em']:.4f}")
    print(f"Answer F1:  {result['f1']:.4f}")
    print(f"Precision:  {result['prec']:.4f}")
    print(f"Recall:     {result['recall']:.4f}")

    # By question type
    by_type = result.get("by_type", {})
    if len(by_type) > 1:
        print(f"\n  By question type:")
        for t, stats in sorted(by_type.items()):
            print(f"    {t:<16s}  EM={stats['em']:.4f}  F1={stats['f1']:.4f}  (n={stats['count']})")

    # By difficulty level
    by_level = result.get("by_level", {})
    if len(by_level) > 1:
        print(f"\n  By difficulty:")
        for lv in ["easy", "medium", "hard"]:
            if lv in by_level:
                stats = by_level[lv]
                print(f"    {lv:<8s}  EM={stats['em']:.4f}  F1={stats['f1']:.4f}  (n={stats['count']})")

    # By type × level
    by_type_level = result.get("by_type_level", {})
    if by_type_level:
        print(f"\n  By type × level:")
        for t in sorted(by_type_level):
            for lv in ["easy", "medium", "hard"]:
                if lv in by_type_level[t]:
                    stats = by_type_level[t][lv]
                    print(f"    {t:<12s} × {lv:<8s}  EM={stats['em']:.4f}  F1={stats['f1']:.4f}  (n={stats['count']})")


if __name__ == "__main__":
    main()
