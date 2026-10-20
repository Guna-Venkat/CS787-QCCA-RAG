import time
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from copy import deepcopy

from tqdm import tqdm

from sane import create_agent
from sane.core.config import load_config
from sane.embedding import create_embedding_model

from benchmarks.longmemeval.data import load_longmemeval_data, save_predictions
from benchmarks.metrics import agent_metrics, aggregate_records, build_run_summary, save_metrics


def _write_progress(output_path: str, done: int, total: int):
    """Write a lightweight progress file so external watchers can track completion."""
    import os as _os
    progress_path = output_path + ".progress"
    try:
        with open(progress_path, "w") as f:
            f.write(f"{done}/{total}\n")
    except Exception:
        pass


def _incremental_save(results_by_index: dict[int, dict], output_path: str):
    """Save all completed predictions incrementally (survives kills)."""
    import json as _json
    import os as _os
    try:
        _os.makedirs(_os.path.dirname(output_path), exist_ok=True)
        with open(output_path, "w") as f:
            for i in sorted(results_by_index):
                entry = results_by_index[i]
                clean = {k: v for k, v in entry.items() if not k.startswith("_")}
                f.write(_json.dumps(clean, ensure_ascii=False) + "\n")
    except Exception:
        pass


def _incremental_save_metrics(output_path: str, instance_metrics: list[dict],
                              total_sends: int, t_start: float,
                              method: str, workers: int):
    """Save partial efficiency metrics incrementally (survives kills)."""
    try:
        t_elapsed = time.perf_counter() - t_start
        aggregated = aggregate_records(instance_metrics)
        run_metrics = build_run_summary(
            unit="instance",
            unit_count=len(instance_metrics),
            run_wall_time_s=t_elapsed,
            aggregated=aggregated,
            send_count=total_sends,
        )
        efficiency_metrics = {
            "benchmark": "longmemeval",
            "unit": "instance",
            "method": method,
            "workers": workers,
            "run": run_metrics,
            "instances": instance_metrics,
        }
        save_metrics(output_path, efficiency_metrics)
    except Exception:
        pass


def _warm_up_embedding_model(embedding_model):
    try:
        embedding_model.embed(["warmup"])
    except Exception:
        pass


def _process_instance(index: int, instance: dict, config,
                      embedding_model) -> tuple[int, dict]:
    """Process a single instance (session by session) and return prediction."""
    t_instance_start = time.perf_counter()
    cfg = deepcopy(config)
    agent = None

    try:
        agent = create_agent(cfg, embedding_model=embedding_model)
        agent.reset()
        ingest = getattr(agent, "ingest", None)

        sends = 0
        haystack_sessions = instance.get("haystack_sessions", [])
        haystack_dates = instance.get("haystack_dates", [])
        haystack_session_ids = instance.get("haystack_session_ids", [])

        if cfg.method.method in ("rag", "sane") and hasattr(agent, "ingest_doc"):
            session_ids = instance.get("haystack_session_ids", [])
            for s_idx, session in enumerate(haystack_sessions):
                session_date = haystack_dates[s_idx] if s_idx < len(haystack_dates) else None
                session_id = session_ids[s_idx] if s_idx < len(session_ids) else str(s_idx)
                agent.ingest_doc(session, doc_id=session_id, date=session_date)
        else:
            raise ValueError(f"Unknown method: {cfg.method.method}")

        question = instance.get("question", "")
        question_date = instance.get("question_date", "")
        if question_date:
            question = f"Current Date: {question_date}\n\n{question}"
        # Pass ground-truth answer sessions to agent for recall computation
        if hasattr(agent, "set_answer_doc_ids"):
            agent.set_answer_doc_ids(
                instance.get("answer_session_ids", []))
        answer = agent.send(question)
        sends += 1

        stages = {}
        metrics = {}
        if hasattr(agent, "get_timing_summary"):
            summary = agent.get_timing_summary()
            stages = dict(summary.get("stages", {}))
            metrics = agent_metrics(
                agent,
                wall_time_s=time.perf_counter() - t_instance_start,
                sends=sends,
            )

        return index, {
            "question_id": instance.get("question_id", str(index)),
            "hypothesis": answer,
            "_sends": sends,
            "_stages": stages,
            "_metrics": metrics,
        }
    finally:
        if agent is not None and hasattr(agent, "reset"):
            agent.reset()


def run_longmemeval(
    split: str = "oracle",
    output_path: str | None = None,
    limit: int | None = None,
    method: str | None = None,
    workers: int = 1,
    config_path: str | None = None,
    overrides: str | None = None,
    model: str | None = None,
) -> list[dict]:
    data = load_longmemeval_data(split)
    if limit:
        data = data[:limit]

    config = load_config(config_path=config_path)
    if method is not None:
        config.method.method = method
    if overrides is not None:
        from benchmarks.longmemeval.run import _apply_overrides
        _apply_overrides(config, overrides)
    if model is not None:
        from sane.core.model_registry import apply_model_override
        apply_model_override(config.llm, model)

    # Load embedding model once — shared across all workers
    embedding_model = create_embedding_model(config.embedding)

    total_sends = 0
    cumulative_stages: dict[str, float] = defaultdict(float)
    instance_metrics: list[dict] = []
    t_start = time.perf_counter()

    # Process instances (parallel or sequential)
    results_by_index: dict[int, dict] = {}

    if workers > 1:
        Executor = ThreadPoolExecutor
        _emb = embedding_model
        _warm_up_embedding_model(embedding_model)

        pbar = tqdm(total=len(data), desc="Instances", unit="inst", dynamic_ncols=True)
        with Executor(max_workers=workers) as executor:
            futures = {
                executor.submit(_process_instance, i, inst, config,
                               _emb): i
                for i, inst in enumerate(data)
            }
            for future in as_completed(futures):
                idx, result = future.result()
                results_by_index[idx] = result
                total_sends += result.get("_sends", 0)
                for stage, elapsed in result.get("_stages", {}).items():
                    cumulative_stages[stage] += elapsed
                metrics = result.get("_metrics", {})
                if metrics:
                    instance_metrics.append({
                        "question_id": result.get("question_id"),
                        **metrics,
                    })
                pbar.update(1)
                pbar.set_postfix({
                    "done": len(results_by_index),
                    "active": min(workers, len(data) - len(results_by_index)),
                })
                # Incremental save: write progress + predictions + metrics
                if output_path:
                    _write_progress(output_path, len(results_by_index), len(data))
                    _incremental_save(results_by_index, output_path)
                    _incremental_save_metrics(
                        output_path, instance_metrics, total_sends,
                        t_start, method or config.method.method, workers)
        pbar.close()
    else:
        pbar = tqdm(enumerate(data), total=len(data), desc="Instances", unit="inst",
                    dynamic_ncols=True)
        for i, instance in pbar:
            t_inst = time.perf_counter()
            idx, result = _process_instance(i, instance, config, embedding_model)
            results_by_index[idx] = result
            sends = result.get("_sends", 0)
            total_sends += sends
            for stage, elapsed in result.get("_stages", {}).items():
                cumulative_stages[stage] += elapsed
            metrics = result.get("_metrics", {})
            if metrics:
                instance_metrics.append({
                    "question_id": result.get("question_id"),
                    **metrics,
                })

            t_elapsed = time.perf_counter() - t_inst
            qid = str(result.get("question_id", ""))
            stage_info = ""
            stages = result.get("_stages", {})
            if stages:
                stage_info = ""
            pbar.set_postfix({
                "qid": qid[:24],
                "time": f"{t_elapsed:.1f}s",
                "done": f"{i+1}/{len(data)}",
            })
            pbar.write(f"  ✓ [{i+1}/{len(data)}] {qid} — {t_elapsed:.1f}s{stage_info}")
            # Incremental save: write progress + predictions + metrics
            if output_path:
                _write_progress(output_path, i + 1, len(data))
                _incremental_save(results_by_index, output_path)
                _incremental_save_metrics(
                    output_path, instance_metrics, total_sends,
                    t_start, method or config.method.method, workers)
        pbar.close()

    # Sort results by original index
    predictions = [results_by_index[i] for i in sorted(results_by_index)]
    instance_metrics.sort(key=lambda m: str(m.get("question_id", "")))

    t_total = time.perf_counter() - t_start
    aggregated = aggregate_records(instance_metrics)
    run_metrics = build_run_summary(
        unit="instance",
        unit_count=len(data),
        run_wall_time_s=t_total,
        aggregated=aggregated,
        send_count=total_sends,
    )
    efficiency_metrics = {
        "benchmark": "longmemeval",
        "unit": "instance",
        "method": method or config.method.method,
        "workers": workers,
        "run": run_metrics,
        "instances": instance_metrics,
    }

    # Print timing summary
    print(f"\n{'='*60}")
    print(f"Timing Summary ({method or config.method.method}, {workers} worker(s))")
    print(f"{'='*60}")
    print(f"Total time:        {t_total:.1f}s")
    print(f"Instances:         {len(data)}")
    print(f"Total session-sends: {total_sends} (~{total_sends / max(len(data), 1):.0f}/inst)")
    print(f"Avg per instance:  {run_metrics['avg_wall_time_s_per_instance']:.1f}s")
    if workers > 1:
        print(f"Throughput/inst:   {run_metrics['throughput_wall_time_s_per_instance']:.1f}s")

    if cumulative_stages:
        stage_total = run_metrics.get("sum_instance_wall_time_s") or t_total
        print(f"\nCumulative stage timing:")
        for stage, elapsed in sorted(cumulative_stages.items()):
            if not stage.startswith("_"):
                pct = elapsed / max(stage_total, 1e-10) * 100
                print(f"  {stage:<30s} {elapsed:>8.1f}s ({pct:>5.1f}%)")

    # Strip internal fields before saving
    clean_predictions = [{k: v for k, v in p.items() if not k.startswith("_")}
                         for p in predictions]

    if output_path:
        save_predictions(clean_predictions, output_path)
        print(f"\nSaved {len(clean_predictions)} predictions to {output_path}")
        metrics_path = save_metrics(output_path, efficiency_metrics)
        print(f"Saved efficiency metrics to {metrics_path}")

    return clean_predictions
