"""CLI entry point for LongMemEval benchmark evaluation.

Usage: python -m benchmarks.longmemeval.run [--split oracle] [--limit 10]
"""

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
import json
import os
import re
import time

import numpy as np
import yaml

from benchmarks.metrics import load_metrics, print_efficiency_summary, save_eval_summary
from sane.llm.openai_client import OpenAIClient


def _build_judge_prompt(task: str, question: str, answer: str,
                        response: str, abstention: bool = False) -> str:
    """Task-specific LLM-as-judge prompts from LongMemEval evaluate_qa.py."""
    if not abstention:
        if task in ("single-session-user", "single-session-assistant", "multi-session"):
            return (
                "I will give you a question, a correct answer, and a response from a model. "
                "Please answer yes if the response contains the correct answer. Otherwise, answer no. "
                "If the response is equivalent to the correct answer or contains all the intermediate "
                "steps to get the correct answer, you should also answer yes. If the response only "
                "contains a subset of the information required by the answer, answer no. \n\n"
                f"Question: {question}\n\n"
                f"Correct Answer: {answer}\n\n"
                f"Model Response: {response}\n\n"
                "Is the model response correct? Answer yes or no only."
            )
        elif task == "temporal-reasoning":
            return (
                "I will give you a question, a correct answer, and a response from a model. "
                "Please answer yes if the response contains the correct answer. Otherwise, answer no. "
                "If the response is equivalent to the correct answer or contains all the intermediate "
                "steps to get the correct answer, you should also answer yes. If the response only "
                "contains a subset of the information required by the answer, answer no. In addition, "
                "do not penalize off-by-one errors for the number of days. If the question asks for "
                "the number of days/weeks/months, etc., and the model makes off-by-one errors "
                "(e.g., predicting 19 days when the answer is 18), the model's response is still correct. \n\n"
                f"Question: {question}\n\n"
                f"Correct Answer: {answer}\n\n"
                f"Model Response: {response}\n\n"
                "Is the model response correct? Answer yes or no only."
            )
        elif task == "knowledge-update":
            return (
                "I will give you a question, a correct answer, and a response from a model. "
                "Please answer yes if the response contains the correct answer. Otherwise, answer no. "
                "If the response contains some previous information along with an updated answer, "
                "the response should be considered as correct as long as the updated answer is "
                "the required answer.\n\n"
                f"Question: {question}\n\n"
                f"Correct Answer: {answer}\n\n"
                f"Model Response: {response}\n\n"
                "Is the model response correct? Answer yes or no only."
            )
        elif task == "single-session-preference":
            return (
                "I will give you a question, a rubric for desired personalized response, "
                "and a response from a model. Please answer yes if the response satisfies "
                "the desired response. Otherwise, answer no. The model does not need to reflect "
                "all the points in the rubric. The response is correct as long as it recalls "
                "and utilizes the user's personal information correctly.\n\n"
                f"Question: {question}\n\n"
                f"Rubric: {answer}\n\n"
                f"Model Response: {response}\n\n"
                "Is the model response correct? Answer yes or no only."
            )
        else:
            raise NotImplementedError(f"Unknown task type: {task}")
    else:
        return (
            "I will give you an unanswerable question, an explanation, and a response from a model. "
            "Please answer yes if the model correctly identifies the question as unanswerable. "
            "The model could say that the information is incomplete, or some other information "
            "is given but the asked information is not.\n\n"
            f"Question: {question}\n\n"
            f"Explanation: {answer}\n\n"
            f"Model Response: {response}\n\n"
            "Does the model correctly identify the question as unanswerable? "
            "Answer yes or no only."
        )



def _default_output_dir(project_root: str) -> str:
    return os.path.join(project_root, "output", "benchmarks", "longmemeval")


def _build_judge_chat_params(
    *,
    model: str,
    prompt: str,
    temperature: float,
    max_tokens: int,
) -> dict:
    return {
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "n": 1,
        "temperature": temperature,
        "max_tokens": max_tokens,
    }


@dataclass(frozen=True)
class _JudgeSettings:
    model: str
    base_url: str
    api_key: str
    max_tokens: int
    temperature: float


def _judge_prediction(
    entry: dict,
    qdata: dict,
    settings: _JudgeSettings,
    client: OpenAIClient,
) -> tuple[dict, str]:
    qid = entry["question_id"]
    prompt = _build_judge_prompt(
        qdata["question_type"],
        qdata["question"],
        qdata["answer"],
        entry["hypothesis"],
        abstention="_abs" in qid,
    )
    response = client.chat(
        **_build_judge_chat_params(
            model=settings.model,
            prompt=prompt,
            temperature=settings.temperature,
            max_tokens=settings.max_tokens,
        )
    )
    eval_response = (response.get("content") or "").strip()
    label = "yes" in eval_response.lower()
    judged_entry = dict(entry)
    judged_entry["autoeval_label"] = {"model": settings.model, "label": label}
    return judged_entry, qdata["question_type"]


def _evaluate_predictions(
    *,
    predictions: list[dict],
    qid2data: dict[str, dict],
    settings: _JudgeSettings,
    judge_workers: int,
) -> tuple[list[dict], dict[str, list[int]], dict]:
    judge_workers = max(1, int(judge_workers or 1))
    client = OpenAIClient(
        base_url=settings.base_url,
        api_key=settings.api_key,
        model=settings.model,
        max_tokens=settings.max_tokens,
        temperature=settings.temperature,
    )
    jobs = []
    for index, entry in enumerate(predictions):
        qid = entry["question_id"]
        if qid not in qid2data:
            print(f"Warning: skipping {qid} — not in reference data")
            continue
        jobs.append((index, entry, qid2data[qid]))

    results_by_index = {}
    qtype2acc = {}

    if judge_workers == 1 or len(jobs) <= 1:
        for index, entry, qdata in jobs:
            judged_entry, qtype = _judge_prediction(entry, qdata, settings, client)
            results_by_index[index] = judged_entry
            qtype2acc.setdefault(qtype, []).append(
                1 if judged_entry["autoeval_label"]["label"] else 0
            )
    else:
        with ThreadPoolExecutor(max_workers=min(judge_workers, len(jobs))) as executor:
            futures = {
                executor.submit(_judge_prediction, entry, qdata, settings, client): index
                for index, entry, qdata in jobs
            }
            for future in as_completed(futures):
                index = futures[future]
                judged_entry, qtype = future.result()
                results_by_index[index] = judged_entry
                qtype2acc.setdefault(qtype, []).append(
                    1 if judged_entry["autoeval_label"]["label"] else 0
                )

    logs = [results_by_index[index] for index, _, _ in jobs]
    return logs, qtype2acc, client.get_usage_summary()


def _apply_overrides(config, overrides_str: str | None):
    """Apply comma-separated key=value overrides to config.method.

    Supports dot notation for nested dataclass fields, e.g.
    ``sane.extraction_enabled=true`` sets ``config.method.sane.extraction_enabled``.
    """
    if not overrides_str:
        return
    mc = config.method
    for pair in overrides_str.split(","):
        key, val = pair.strip().split("=", 1)
        key = key.strip()
        val = val.strip()

        # Support dot notation: "sane.extraction_enabled" → mc.sane.extraction_enabled
        parts = key.split(".")
        obj = mc
        for part in parts[:-1]:
            if not hasattr(obj, part):
                print(f"Warning: unknown config key '{key}', skipping")
                break
            obj = getattr(obj, part)
        else:
            leaf = parts[-1]
            if not hasattr(obj, leaf):
                print(f"Warning: unknown config key '{key}', skipping")
                continue
            current = getattr(obj, leaf)
            if isinstance(current, bool):
                typed_val = val.lower() in ("true", "1", "yes")
            elif isinstance(current, int):
                typed_val = int(val)
            elif isinstance(current, float):
                typed_val = float(val)
            else:
                typed_val = val
            setattr(obj, leaf, typed_val)


def main():
    parser = argparse.ArgumentParser(
        description="Run LongMemEval benchmark with RAG methods",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""split descriptions:
  oracle      Evidence-only sessions (~3-80 sessions per instance). Default.
  s_cleaned   ~40 sessions, ~115K tokens. Fits in a 128K context window.
  m_cleaned   ~500 sessions. Requires retrieval; too long for any context window.""",
    )
    parser.add_argument("--split", default="oracle",
                        choices=["oracle", "s_cleaned", "m_cleaned"],
                        help="Data split (default: oracle)")
    parser.add_argument("--limit", type=int, default=None,
                        help="Limit instances for quick testing")
    parser.add_argument("--method", default=None,
                        choices=["rag", "sane"],
                        help="Agent method (default: from config)")
    parser.add_argument("--output-dir", default=None,
                        help="Output directory for predictions")
    parser.add_argument("--workers", type=int, default=1,
                        help="Number of parallel workers for prediction (default: 1)")
    parser.add_argument("--judge-workers", type=int, default=None,
                        help="Number of parallel workers for judge evaluation (default: config or 1)")
    parser.add_argument("--eval-only", type=str, default=None,
                        help="Skip generation, evaluate an existing predictions file")
    parser.add_argument("--config", type=str, default=None,
                        help="Path to config YAML (overrides default)")
    parser.add_argument("--model", type=str, default=None,
                        help="Override the LLM model (e.g. qwen3.5-9b, Qwen/Qwen3.5-4B)")
    parser.add_argument("--judge-model", type=str, default=None,
                        help="Override the judge model (default: same as config or benchmark.yaml)")
    parser.add_argument("--overrides", type=str, default=None,
                        help="Comma-separated key=value overrides for method config, e.g. "
                             "k=5,w=20,sane.extraction_enabled=true")
    args = parser.parse_args()

    project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    default_output_dir = _default_output_dir(project_root)
    output_dir = args.output_dir or default_output_dir
    os.makedirs(output_dir, exist_ok=True)

    timestamp = time.strftime("%Y%m%d_%H%M%S")
    model_tag = (args.model or "default").replace("/", "-")
    predictions_path = os.path.join(output_dir, f"predictions_{args.split}_{model_tag}_{timestamp}.jsonl")

    if args.eval_only:
        predictions_path = args.eval_only
    else:
        from benchmarks.longmemeval.runner import run_longmemeval

        print(f"Running LongMemEval (split={args.split}, method={args.method or 'default'}, workers={args.workers})")
        if args.limit:
            print(f"Limited to {args.limit} instances")
        print(f"Predictions: {predictions_path}\n")

        predictions = run_longmemeval(
            split=args.split,
            output_path=predictions_path,
            limit=args.limit,
            method=args.method,
            workers=args.workers,
            config_path=args.config,
            overrides=args.overrides,
            model=args.model,
        )
        print(f"\nDone. {len(predictions)} predictions saved.\n")

    print("Evaluating predictions...")

    from benchmarks.longmemeval.data import load_predictions, load_longmemeval_data
    predictions = load_predictions(predictions_path)

    ref_data = load_longmemeval_data(split=args.split)

    qid2data = {entry["question_id"]: entry for entry in ref_data}

    # Resolve judge: CLI flag > config YAML judge: section > registry > env > main LLM
    judge_overrides: dict = {}
    main_cfg = None
    if args.config:
        from sane.core.config import load_config
        main_cfg = load_config(args.config)
        with open(args.config) as f:
            raw = yaml.safe_load(f) or {}
        overrides = raw.get("judge", {})
        if overrides:
            judge_overrides = overrides

    from sane.core.config import resolve_judge_config
    judge_cfg_resolved = resolve_judge_config(
        main_config=main_cfg,
        model=args.judge_model,
        judge_overrides=judge_overrides,
    )
    judge_model = judge_cfg_resolved["model"]
    judge_base_url = judge_cfg_resolved["base_url"]
    judge_api_key = judge_cfg_resolved["api_key"]
    judge_max_tokens = judge_cfg_resolved["max_tokens"]
    judge_temperature = judge_cfg_resolved["temperature"]
    judge_workers = args.judge_workers
    if judge_workers is None:
        judge_workers = judge_overrides.get("workers", 1)
    judge_workers = max(1, int(judge_workers or 1))
    print(f"Judge: {judge_model} @ {judge_base_url} ({judge_workers} worker(s))")

    judge_model_safe = re.sub(r"[^a-zA-Z0-9._-]", "_", judge_model)

    logs, qtype2acc, judge_usage = _evaluate_predictions(
        predictions=predictions,
        qid2data=qid2data,
        settings=_JudgeSettings(
            model=judge_model,
            base_url=judge_base_url,
            api_key=judge_api_key,
            max_tokens=judge_max_tokens,
            temperature=judge_temperature,
        ),
        judge_workers=judge_workers,
    )

    eval_results_path = predictions_path + f".eval-results-{judge_model_safe}"
    with open(eval_results_path, "w") as f:
        for entry in logs:
            f.write(json.dumps(entry) + "\n")

    overall = np.mean([1 if x["autoeval_label"]["label"] else 0 for x in logs]).item()
    print(f"\nOverall Accuracy: {overall:.4f}")
    for qtype in sorted(qtype2acc.keys()):
        vals = qtype2acc[qtype]
        print(f"  {qtype}: {np.mean(vals).item():.4f} ({len(vals)})")

    efficiency_metrics = load_metrics(predictions_path)
    if efficiency_metrics and judge_usage:
        run_metrics = efficiency_metrics.setdefault("run", {})
        run_metrics.setdefault("llm", {})["judge"] = judge_usage
    print_efficiency_summary(efficiency_metrics)

    summary = {
        "benchmark": "longmemeval",
        "split": args.split,
        "judge_model": judge_model,
        "judge_workers": judge_workers,
        "llm": {
            "judge": judge_usage,
        },
        "num_predictions": len(logs),
        "overall_accuracy": overall,
        "accuracy_by_question_type": {
            qtype: {
                "accuracy": np.mean(vals).item(),
                "count": len(vals),
            }
            for qtype, vals in sorted(qtype2acc.items())
        },
        "efficiency": efficiency_metrics.get("run") if efficiency_metrics else None,
    }
    summary_path = save_eval_summary(eval_results_path, summary)

    print(f"\nSaved to: {eval_results_path}")
    print(f"Saved summary to: {summary_path}")


if __name__ == "__main__":
    main()
