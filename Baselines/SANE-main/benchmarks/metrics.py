"""Shared efficiency metrics helpers for benchmark runners and evaluators."""

from __future__ import annotations

import json
import os
from collections import defaultdict
from copy import deepcopy
from typing import Any


def metrics_path_for(predictions_path: str) -> str:
    return predictions_path + ".metrics.json"


def summary_path_for(eval_results_path: str) -> str:
    return eval_results_path + ".summary.json"


def save_metrics(predictions_path: str, metrics: dict[str, Any]) -> str:
    path = metrics_path_for(predictions_path)
    with open(path, "w") as f:
        json.dump(metrics, f, indent=2, sort_keys=True, default=str)
        f.write("\n")
    return path


def load_metrics(predictions_path: str) -> dict[str, Any] | None:
    path = metrics_path_for(predictions_path)
    if not os.path.exists(path):
        return None
    with open(path) as f:
        return json.load(f)


def save_eval_summary(eval_results_path: str, summary: dict[str, Any]) -> str:
    path = summary_path_for(eval_results_path)
    with open(path, "w") as f:
        json.dump(summary, f, indent=2, sort_keys=True, default=str)
        f.write("\n")
    return path


def agent_metrics(agent: Any, *, wall_time_s: float, sends: int | None = None) -> dict[str, Any]:
    summary = {}
    if hasattr(agent, "get_timing_summary"):
        summary = agent.get_timing_summary() or {}

    send_count = int(sends if sends is not None else summary.get("send_count", 0) or 0)
    avg_send_latency = wall_time_s / send_count if send_count else 0.0

    return {
        "wall_time_s": wall_time_s,
        "send_count": send_count,
        "avg_wall_time_s_per_send": avg_send_latency,
        "agent_total_s": float(summary.get("total_s", 0.0) or 0.0),
        "stages_s": dict(summary.get("stages", {}) or {}),
        "last_send_s": dict(summary.get("last_send", {}) or {}),
        "memory": deepcopy(summary.get("memory", {}) or {}),
        "ie": deepcopy(summary.get("ie", {}) or {}),
        "llm": deepcopy(summary.get("llm", {}) or {}),
    }


def aggregate_records(records: list[dict[str, Any]]) -> dict[str, Any]:
    total_wall = sum(float(r.get("wall_time_s", 0.0) or 0.0) for r in records)
    total_sends = sum(int(r.get("send_count", 0) or 0) for r in records)
    total_agent = sum(float(r.get("agent_total_s", 0.0) or 0.0) for r in records)

    stages = defaultdict(float)
    memory = defaultdict(int)
    ie = defaultdict(int)
    for record in records:
        for stage, elapsed in (record.get("stages_s") or {}).items():
            stages[stage] += float(elapsed or 0.0)
        for key, value in (record.get("memory") or {}).items():
            memory[key] += int(value or 0)
        for key, value in (record.get("ie") or {}).items():
            if isinstance(value, int):
                ie[key] += value
            elif isinstance(value, float):
                ie[key] += value

    return {
        "wall_time_s": total_wall,
        "send_count": total_sends,
        "avg_wall_time_s_per_send": total_wall / total_sends if total_sends else 0.0,
        "agent_total_s": total_agent,
        "stages_s": dict(stages),
        "memory": dict(memory),
        "ie": dict(ie),
        "llm": aggregate_llm([r.get("llm", {}) or {} for r in records]),
    }


def build_run_summary(
    *,
    unit: str,
    unit_count: int,
    run_wall_time_s: float,
    aggregated: dict[str, Any],
    send_count: int,
    extra: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Build timing metrics that separate latency from parallel throughput.

    `run_wall_time_s` is the end-to-end benchmark duration. Per-unit latency
    should come from worker-summed unit timings, otherwise parallel runs appear
    artificially fast.
    """
    summed_unit_wall = float(aggregated.get("wall_time_s", 0.0) or 0.0)
    run: dict[str, Any] = {
        "wall_time_s": run_wall_time_s,
        f"sum_{unit}_wall_time_s": summed_unit_wall,
        f"{unit}_count": unit_count,
        "send_count": send_count,
        f"avg_wall_time_s_per_{unit}": (
            summed_unit_wall / unit_count if unit_count else 0.0
        ),
        f"throughput_wall_time_s_per_{unit}": (
            run_wall_time_s / unit_count if unit_count else 0.0
        ),
        "avg_wall_time_s_per_send": (
            summed_unit_wall / send_count if send_count else 0.0
        ),
        "throughput_wall_time_s_per_send": (
            run_wall_time_s / send_count if send_count else 0.0
        ),
        "agent_total_s": aggregated.get("agent_total_s", 0.0),
        "stages_s": dict(aggregated.get("stages_s", {}) or {}),
        "memory": dict(aggregated.get("memory", {}) or {}),
        "ie": dict(aggregated.get("ie", {}) or {}),
        "llm": aggregated.get("llm", {}),
    }
    # Compute average recall@k from summed ie values.
    # recall values are per-instance ratios (0-1); the sum across instances
    # must be divided by unit_count to get the average.
    ie = run.get("ie", {})
    if unit_count and ie:
        recall: dict[str, float] = {}
        for key, value in ie.items():
            if isinstance(value, (int, float)) and ("recall" in key or "hit" in key):
                recall[key] = value / unit_count
        if recall:
            run["recall"] = recall
    if extra:
        run.update(extra)
    return run


def aggregate_llm(llm_scopes_list: list[dict[str, Any]]) -> dict[str, Any]:
    scopes: dict[str, dict[str, Any]] = {}
    for llm_scopes in llm_scopes_list:
        for scope, summary in llm_scopes.items():
            if not summary:
                continue
            scopes.setdefault(scope, _empty_llm_summary())
            _merge_llm_summary(scopes[scope], summary)

    for summary in scopes.values():
        calls = summary["call_count"]
        if calls:
            summary["avg_latency_s"] = summary["latency_s"] / calls
            summary["context"]["avg_estimated_prompt_tokens"] = (
                summary["context"]["estimated_prompt_tokens"] / calls
            )
            summary["context"]["avg_messages"] = summary["context"]["messages_total"] / calls
    return scopes


def flatten_llm_totals(llm_scopes: dict[str, Any]) -> dict[str, Any]:
    totals = _empty_llm_summary()
    for summary in (llm_scopes or {}).values():
        _merge_llm_summary(totals, summary)

    calls = totals["call_count"]
    if calls:
        totals["avg_latency_s"] = totals["latency_s"] / calls
        totals["context"]["avg_estimated_prompt_tokens"] = (
            totals["context"]["estimated_prompt_tokens"] / calls
        )
        totals["context"]["avg_messages"] = totals["context"]["messages_total"] / calls
    return totals


def print_efficiency_summary(metrics: dict[str, Any] | None):
    if not metrics:
        print("\nEfficiency metrics: not available (missing .metrics.json sidecar)")
        return

    run = metrics.get("run", {})
    llm_totals = flatten_llm_totals(run.get("llm", {}) or {})
    unit = metrics.get("unit", "item")
    item_count = int(run.get(f"{unit}_count", 0) or run.get("item_count", 0) or 0)
    send_count = int(run.get("send_count", 0) or 0)
    wall_time_s = float(run.get("wall_time_s", 0.0) or 0.0)
    sum_unit_wall = float(run.get(f"sum_{unit}_wall_time_s", 0.0) or 0.0)
    avg_unit_wall = float(run.get(f"avg_wall_time_s_per_{unit}", 0.0) or 0.0)
    throughput_unit_wall = float(
        run.get(f"throughput_wall_time_s_per_{unit}", 0.0) or 0.0
    )
    avg_send_wall = float(run.get("avg_wall_time_s_per_send", 0.0) or 0.0)
    throughput_send_wall = float(run.get("throughput_wall_time_s_per_send", 0.0) or 0.0)

    print("\nEfficiency Metrics")
    if wall_time_s:
        print(f"  Wall time:                 {wall_time_s:.1f}s")
    if sum_unit_wall:
        print(f"  Worker-summed {unit} time: {sum_unit_wall:.1f}s")
    if item_count:
        if not avg_unit_wall:
            avg_unit_wall = wall_time_s / item_count
        print(f"  Avg wall time/{unit}:       {avg_unit_wall:.2f}s")
        if throughput_unit_wall and throughput_unit_wall != avg_unit_wall:
            print(f"  Throughput wall/{unit}:     {throughput_unit_wall:.2f}s")
    if send_count:
        if not avg_send_wall:
            avg_send_wall = wall_time_s / send_count
        send_text = f"{send_count} ({avg_send_wall:.2f}s/send)"
        if throughput_send_wall and throughput_send_wall != avg_send_wall:
            send_text += f", throughput {throughput_send_wall:.2f}s/send"
        print(f"  Sends:                     {send_text}")
    if llm_totals["call_count"]:
        usage = llm_totals["usage"]
        context = llm_totals["context"]
        print(f"  LLM calls:                 {llm_totals['call_count']}")
        print(f"  LLM latency:               {llm_totals['latency_s']:.1f}s ({llm_totals['avg_latency_s']:.2f}s/call)")
        print(f"  Prompt tokens:             {usage['prompt_tokens']}")
        if usage.get("cached_prompt_tokens"):
            print(
                f"  Cached prompt tokens:      "
                f"{usage['cached_prompt_tokens']}"
            )
        print(f"  Completion tokens:         {usage['completion_tokens']}")
        print(f"  Total tokens:              {usage['total_tokens']}")
        scopes = run.get("llm", {}) or {}
        if len(scopes) > 1:
            print("  LLM scopes:")
            for scope, summary in sorted(scopes.items()):
                calls = int(summary.get("call_count", 0) or 0)
                if not calls:
                    continue
                usage = summary.get("usage") or {}
                print(
                    f"    {scope}: {calls} calls, "
                    f"{usage.get('prompt_tokens', 0)} prompt, "
                    f"{usage.get('completion_tokens', 0)} completion"
                )
        print(f"  Est. context tokens/call:  avg {context['avg_estimated_prompt_tokens']:.0f}, max {context['max_estimated_prompt_tokens']}")
        # Retrieval recall@k (averaged across instances)
        recall = run.get("recall", {}) or {}
        if recall:
            print("  Retrieval Recall (avg across instances):")
            for k_name, k_val in sorted(recall.items()):
                print(f"    {k_name}: {k_val:.4f}")


def _empty_llm_summary() -> dict[str, Any]:
    return {
        "call_count": 0,
        "error_count": 0,
        "latency_s": 0.0,
        "avg_latency_s": 0.0,
        "usage": {
            "prompt_tokens": 0,
            "completion_tokens": 0,
            "total_tokens": 0,
            "cached_prompt_tokens": 0,
        },
        "models": {},
        "context": {
            "estimated_prompt_tokens": 0,
            "avg_estimated_prompt_tokens": 0.0,
            "max_estimated_prompt_tokens": 0,
            "last_estimated_prompt_tokens": 0,
            "messages_total": 0,
            "avg_messages": 0.0,
            "max_messages": 0,
            "last_messages": 0,
        },
    }


def _merge_llm_summary(dst: dict[str, Any], src: dict[str, Any]):
    dst["call_count"] += int(src.get("call_count", 0) or 0)
    dst["error_count"] += int(src.get("error_count", 0) or 0)
    dst["latency_s"] += float(src.get("latency_s", 0.0) or 0.0)

    dst_usage = dst["usage"]
    for key in (
        "prompt_tokens",
        "completion_tokens",
        "total_tokens",
        "cached_prompt_tokens",
    ):
        dst_usage[key] += int((src.get("usage") or {}).get(key, 0) or 0)

    model = src.get("model")
    if model:
        dst["models"][model] = dst["models"].get(model, 0) + int(src.get("call_count", 0) or 0)

    dst_context = dst["context"]
    src_context = src.get("context") or {}
    dst_context["estimated_prompt_tokens"] += int(src_context.get("estimated_prompt_tokens", 0) or 0)
    dst_context["messages_total"] += int(src_context.get("messages_total", 0) or 0)
    dst_context["max_estimated_prompt_tokens"] = max(
        dst_context["max_estimated_prompt_tokens"],
        int(src_context.get("max_estimated_prompt_tokens", 0) or 0),
    )
    dst_context["max_messages"] = max(
        dst_context["max_messages"],
        int(src_context.get("max_messages", 0) or 0),
    )
    if src.get("call_count", 0):
        dst_context["last_estimated_prompt_tokens"] = int(
            src_context.get("last_estimated_prompt_tokens", 0) or 0
        )
        dst_context["last_messages"] = int(src_context.get("last_messages", 0) or 0)
