"""CLI entry point for MuSiQue benchmark evaluation.

MuSiQue is a multi-hop QA benchmark requiring reasoning across 2-4 Wikipedia
documents.  This runner supports retrieval-based (rag) and
SANE pipeline (sane) methods.

Evaluation uses lexical EM + F1 (same as HotpotQA).  Breakdowns are by hop
type (2hop, 3hop1, 3hop2, 4hop1-4) instead of question type/difficulty.
Answer alias matching is also supported.

Usage::

    # Quick test
    python -m benchmarks.musique.run --method rag --limit 5

    # Semantic RAG (k=4)
    python -m benchmarks.musique.run --method rag \\
        --config config/musique.yaml --workers 50

    # SANE pipeline
    python -m benchmarks.musique.run --method sane \\
        --config config/musique.yaml --workers 50

    # MuSiQue-Full (mixed answerable/unanswerable)
    python -m benchmarks.musique.run --method rag --variant full \\
        --config config/musique.yaml --workers 50
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


# ── Lexical evaluation (shared with HotpotQA) ──────────────────

def normalize_answer(s: str) -> str:
    """Normalise an answer string for comparison.

    Follows the SQuAD convention: lowercase, remove articles
    (a/an/the/and), strip punctuation, collapse whitespace.
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


def _match_aliases(prediction: str, ground_truth: str, aliases: list[str]) -> bool:
    """True iff the normalized prediction matches the ground truth or any alias."""
    norm_pred = normalize_answer(prediction)
    if norm_pred == normalize_answer(ground_truth):
        return True
    for alias in aliases:
        if norm_pred == normalize_answer(alias):
            return True
    return False


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

    Breakdowns are by hop_type (2hop, 3hop1, 3hop2, 4hop1-4) and by
    answerable status (for musique_full).

    Returns a summary dict suitable for serialisation.
    """
    metrics = defaultdict(float)
    hop_stats: dict[str, dict[str, float]] = defaultdict(lambda: defaultdict(float))
    answerable_stats: dict[str, dict[str, float]] = defaultdict(lambda: defaultdict(float))

    total = len(predictions)
    alias_matches = 0

    for p in predictions:
        pred = str(p.get("hypothesis", ""))
        gold = str(p.get("answer", ""))
        aliases = p.get("answer_aliases", [])
        if isinstance(aliases, str):
            aliases = [aliases]
        hop_type = p.get("hop_type", "?")
        answerable = p.get("answerable", True)

        em = float(exact_match_score(pred, gold))
        prec, rec, f1 = _precision_recall_f1(pred, gold)

        # Alias match: count instances where prediction matches an alias
        # but not the primary answer (for diagnostic purposes)
        if _match_aliases(pred, gold, aliases) and not em:
            alias_matches += 1

        metrics["em"] += em
        metrics["f1"] += f1
        metrics["prec"] += prec
        metrics["recall"] += rec

        hop_stats[hop_type]["em"] += em
        hop_stats[hop_type]["f1"] += f1
        hop_stats[hop_type]["count"] += 1

        key = "answerable" if answerable else "unanswerable"
        answerable_stats[key]["em"] += em
        answerable_stats[key]["f1"] += f1
        answerable_stats[key]["count"] += 1

    # ── Overall ─────────────────────────────────────────────────────
    result = {
        "total": total,
        "em": metrics["em"] / total if total > 0 else 0.0,
        "f1": metrics["f1"] / total if total > 0 else 0.0,
        "prec": metrics["prec"] / total if total > 0 else 0.0,
        "recall": metrics["recall"] / total if total > 0 else 0.0,
    }
    if alias_matches > 0:
        result["alias_match_count"] = alias_matches

    # ── By hop type ─────────────────────────────────────────────────
    hop_order = ["2hop", "3hop1", "3hop2", "4hop1", "4hop2", "4hop3", "4hop4"]
    by_hop = {}
    for ht in hop_order:
        if ht in hop_stats:
            count = int(hop_stats[ht]["count"])
            by_hop[ht] = {
                "count": count,
                "em": hop_stats[ht]["em"] / count if count > 0 else 0.0,
                "f1": hop_stats[ht]["f1"] / count if count > 0 else 0.0,
            }
    # Include any unexpected hop types not in the canonical order
    for ht in sorted(hop_stats):
        if ht not in by_hop:
            count = int(hop_stats[ht]["count"])
            by_hop[ht] = {
                "count": count,
                "em": hop_stats[ht]["em"] / count if count > 0 else 0.0,
                "f1": hop_stats[ht]["f1"] / count if count > 0 else 0.0,
            }
    result["by_hop_type"] = by_hop

    # ── By answerability (for musique_full) ──────────────────────────
    if len(answerable_stats) > 1:
        by_ans = {}
        for key in ["answerable", "unanswerable"]:
            if key in answerable_stats:
                count = int(answerable_stats[key]["count"])
                by_ans[key] = {
                    "count": count,
                    "em": answerable_stats[key]["em"] / count if count > 0 else 0.0,
                    "f1": answerable_stats[key]["f1"] / count if count > 0 else 0.0,
                }
        result["by_answerable"] = by_ans

    # ── Save eval results ────────────────────────────────────────────
    if output_path:
        scored = []
        for p in predictions:
            pred = str(p.get("hypothesis", ""))
            gold = str(p.get("answer", ""))
            aliases = p.get("answer_aliases", [])
            if isinstance(aliases, str):
                aliases = [aliases]
            em = exact_match_score(pred, gold)
            alias_match = bool(_match_aliases(pred, gold, aliases))
            f1 = f1_score(pred, gold)
            scored.append({**p, "em": float(em), "f1": f1, "alias_match": alias_match})
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
    return os.path.join(project_root, "output", "benchmarks", "musique")


# ── CLI ─────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="MuSiQue: multi-hop QA benchmark evaluation (2-4 hop)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""Methods:
  rag    Chunk → semantic retrieval (top-k) → LLM
  sane   Chunk → wide retrieval → synopsis → LM selection → evidence → LLM

Examples:
  # Quick test: rag, 5 instances
  python -m benchmarks.musique.run --method rag --limit 5

  # Semantic RAG, 50 workers
  python -m benchmarks.musique.run --method rag --config config/musique.yaml \\
      --workers 50

  # SANE pipeline, 50 workers
  python -m benchmarks.musique.run --method sane --config config/musique.yaml \\
      --workers 50

  # MuSiQue-Full (mixed answerable/unanswerable)
  python -m benchmarks.musique.run --method rag --variant full \\
      --config config/musique.yaml --workers 50
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
                        help="Path to config YAML (default: config/musique.yaml)")
    parser.add_argument("--overrides", type=str, default=None,
                        help="Comma-separated key=value overrides for method config")
    parser.add_argument("--eval-only", type=str, default=None,
                        help="Skip generation, re-evaluate existing predictions file")
    parser.add_argument("--variant", type=str, default="ans",
                        choices=["ans", "full"],
                        help="Dataset variant: ans (answerable only) or full (mixed) (default: ans)")
    parser.add_argument("--split", type=str, default="validation",
                        choices=["validation", "train", "test"],
                        help="Dataset split (default: validation)")
    parser.add_argument("--model", type=str, default=None,
                        help="Override agent LLM model (default: from config)")
    parser.add_argument("--judge-model", type=str, default=None,
                        help="Override judge LLM model (not used for MuSiQue lexical eval)")
    args = parser.parse_args()

    output_dir = args.output_dir or _default_output_dir()
    os.makedirs(output_dir, exist_ok=True)

    # Resolve default config
    config_path = args.config
    if config_path is None:
        project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        default_config = os.path.join(project_root, "config", "musique.yaml")
        if os.path.exists(default_config):
            config_path = default_config

    timestamp = time.strftime("%Y%m%d_%H%M%S")
    method_tag = args.method or "default"

    # ── Eval-only path ────────────────────────────────────────────────
    if args.eval_only:
        from benchmarks.musique.data import load_predictions
        predictions = load_predictions(args.eval_only)
        print(f"Loaded {len(predictions)} predictions from {args.eval_only}")
        eval_path = args.eval_only.replace(".jsonl", ".eval-results.jsonl")
        result = evaluate_predictions(predictions, output_path=eval_path)
        _print_eval_summary(result)
        return

    # ── Load data ────────────────────────────────────────────────────
    from benchmarks.musique.data import load_musique_data
    data = load_musique_data(variant=args.variant, split=args.split)
    print(f"Loaded {len(data)} instances (variant={args.variant}, split={args.split})")
    if args.limit:
        data = data[:args.limit]

    # ── Generate predictions ──────────────────────────────────────────
    from benchmarks.musique.runner import run_musique
    model_tag = (args.model or "default").replace("/", "-")
    predictions_path = os.path.join(
        output_dir,
        f"predictions_musique_{method_tag}_{model_tag}_{args.variant}_{args.split}_{timestamp}.jsonl",
    )

    predictions = run_musique(
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
    print("MuSiQue Evaluation Results")
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
    if result.get("alias_match_count"):
        print(f"Alias-only matches: {result['alias_match_count']}")

    # By hop type
    by_hop = result.get("by_hop_type", {})
    if by_hop:
        print(f"\n  By hop type:")
        for ht, stats in by_hop.items():
            print(f"    {ht:<10s}  EM={stats['em']:.4f}  F1={stats['f1']:.4f}  (n={stats['count']})")

    # By answerability (musique_full)
    by_ans = result.get("by_answerable", {})
    if by_ans:
        print(f"\n  By answerability:")
        for key, stats in by_ans.items():
            print(f"    {key:<14s}  EM={stats['em']:.4f}  F1={stats['f1']:.4f}  (n={stats['count']})")


if __name__ == "__main__":
    main()
