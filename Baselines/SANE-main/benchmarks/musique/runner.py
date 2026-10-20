"""MuSiQue runner: per-instance paragraph ingestion → retrieval → answer generation.

Follows the HotpotQA document-style pattern: each instance gets its own agent,
Wikipedia paragraphs are chunked and ingested, then the question is sent
for retrieval + answer generation.

Supports: rag, sane.

Includes retrieval recall tracking against gold supporting paragraphs, which
HotpotQA does not do — the agent's built-in ``_compute_recall()`` infrastructure
produces recall@k and (for SANE) wide_recall@k + selected_recall.
"""

from __future__ import annotations

import time
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from copy import deepcopy

from tqdm import tqdm

from sane import create_agent
from sane.core.config import load_config
from sane.embedding import create_embedding_model

from benchmarks.musique.data import format_paragraph, save_predictions
from sane.utils.chunking import _chunk_text_by_tokens
from benchmarks.metrics import agent_metrics, aggregate_records, build_run_summary, save_metrics


def _process_instance(
    index: int,
    item: dict,
    config,
    embedding_model,
) -> tuple[int, dict]:
    """Process a single MuSiQue instance: ingest context paragraphs → answer question.

    Returns (index, prediction_dict) where prediction_dict includes
    internal keys prefixed with ``_`` for metrics collection.
    """
    t_start = time.perf_counter()
    cfg = deepcopy(config)
    agent = None

    try:
        agent = create_agent(cfg, embedding_model=embedding_model)
        agent.reset()

        method = cfg.method.method
        sends = 0

        if cfg.method.method in ("rag", "sane") and hasattr(agent, "ingest_text"):
            # Chunk each paragraph → ingest → retrieve → answer
            chunk_size = getattr(cfg.method, "rag_chunk_size", 512)
            max_tokens = getattr(embedding_model, "max_input_tokens", None)
            if callable(max_tokens):
                max_tokens = max_tokens()
            effective_max = min(max(256, int(chunk_size)), int((max_tokens or 8192) * 0.9))

            # Track chunk session IDs for supporting paragraphs → recall computation
            supporting_titles = set(item.get("supporting_titles", []))
            answer_session_ids: list[str] = []

            for title, text in item.get("contexts", []):
                para_text = format_paragraph(title, text)
                chunks = _chunk_text_by_tokens(
                    para_text,
                    max_tokens=effective_max,
                    overlap_tokens=0,
                    encoding_name="cl100k_base",
                )
                total = len(chunks)
                for c_idx, chunk_text in enumerate(chunks):
                    if not chunk_text.strip():
                        continue
                    sid = f"{title}_chunk_{c_idx}"
                    agent.ingest_text(
                        text=chunk_text,
                        doc_id=sid,
                        date="",
                        label=f"{title} p.{c_idx + 1}/{total}" if total > 1 else title,
                    )
                    # Collect chunk IDs for gold supporting paragraphs
                    if title in supporting_titles:
                        answer_session_ids.append(sid)

            # Set gold session IDs for retrieval recall computation
            if answer_session_ids and hasattr(agent, "set_answer_doc_ids"):
                agent.set_answer_doc_ids(answer_session_ids)

            # Format query: question + conciseness instruction.
            # MuSiQue expects short extractive answers (names, numbers, yes/no).
            query = (
                f"{item['question']}\n\n"
                f"Give a short, concise answer — just the name, number, date, "
                f"or phrase that answers the question. Do not explain your reasoning."
            )
            answer = agent.send(query)
            sends += 1

        else:
            raise ValueError(f"Unknown method: {method}")

        stages = {}
        metrics = {}
        if hasattr(agent, "get_timing_summary"):
            summary = agent.get_timing_summary()
            stages = dict(summary.get("stages", {}))
            metrics = agent_metrics(
                agent,
                wall_time_s=time.perf_counter() - t_start,
                sends=sends,
            )

        return index, {
            "question_id": item.get("question_id", str(index)),
            "hypothesis": answer,
            "answer": item.get("answer", ""),
            "answer_aliases": item.get("answer_aliases", []),
            "answerable": item.get("answerable", True),
            "hop_type": item.get("hop_type", "?"),
            "num_hops": item.get("num_hops", 0),
            "supporting_titles": item.get("supporting_titles", []),
            "_sends": sends,
            "_stages": stages,
            "_metrics": metrics,
        }

    finally:
        if agent is not None and hasattr(agent, "reset"):
            agent.reset()


def run_musique(
    data: list[dict],
    output_path: str | None = None,
    limit: int | None = None,
    method: str | None = None,
    workers: int = 1,
    config_path: str | None = None,
    overrides: str | None = None,
    model: str | None = None,
) -> list[dict]:
    """Run MuSiQue evaluation.

    Parameters
    ----------
    data:
        List of instance dicts from :func:`load_musique_data`.
    output_path:
        Path for JSONL predictions output.
    limit:
        Truncate to first N instances (for quick testing).
    method:
        Override ``config.method.method``.
    workers:
        Number of ThreadPoolExecutor workers for parallel instance processing.
    config_path:
        Path to YAML config file.
    overrides:
        Comma-separated ``key=value`` overrides for method config fields.
    """
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

    embedding_model = create_embedding_model(config.embedding)

    total_sends = 0
    cumulative_stages: dict[str, float] = defaultdict(float)
    instance_metrics: list[dict] = []
    t_start = time.perf_counter()

    results_by_index: dict[int, dict] = {}

    if workers > 1:
        Executor = ThreadPoolExecutor
        _emb = embedding_model
        try:
            embedding_model.embed(["warmup"])
        except Exception:
            pass

        with Executor(max_workers=workers) as executor:
            futures = {
                executor.submit(_process_instance, i, inst, config, _emb): i
                for i, inst in enumerate(data)
            }
            for future in tqdm(
                as_completed(futures),
                total=len(data),
                desc="Instances",
                unit="inst",
                dynamic_ncols=True,
            ):
                idx, result = future.result()
                results_by_index[idx] = result
                total_sends += result.get("_sends", 0)
                for stage, elapsed in result.get("_stages", {}).items():
                    cumulative_stages[stage] += elapsed
                m = result.get("_metrics", {})
                if m:
                    instance_metrics.append({
                        "question_id": result.get("question_id"),
                        **m,
                    })
    else:
        for i, instance in enumerate(
            tqdm(data, desc="Instances", unit="inst", dynamic_ncols=True)
        ):
            idx, result = _process_instance(i, instance, config, embedding_model)
            results_by_index[idx] = result
            total_sends += result.get("_sends", 0)
            for stage, elapsed in result.get("_stages", {}).items():
                cumulative_stages[stage] += elapsed
            m = result.get("_metrics", {})
            if m:
                instance_metrics.append({
                    "question_id": result.get("question_id"),
                    **m,
                })

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
        "benchmark": "musique",
        "unit": "instance",
        "method": method or config.method.method,
        "workers": workers,
        "run": run_metrics,
        "instances": instance_metrics,
    }

    print(f"\n{'='*60}")
    print(f"Timing Summary ({method or config.method.method}, {workers} worker(s))")
    print(f"{'='*60}")
    print(f"Total time:        {t_total:.1f}s")
    print(f"Instances:         {len(data)}")
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

    clean_predictions = [
        {k: v for k, v in p.items() if not k.startswith("_")}
        for p in predictions
    ]

    if output_path:
        save_predictions(clean_predictions, output_path)
        print(f"\nSaved {len(clean_predictions)} predictions to {output_path}")
        metrics_path = save_metrics(output_path, efficiency_metrics)
        print(f"Saved efficiency metrics to {metrics_path}")

    return clean_predictions
