"""Precompute and audit top-10 BM25 retrieval for all 2,088 QASPER TRAIN questions.

Uses exact Week-2 paper-level BM25 indexing architecture (chunk_size=256, top_k=10).
Produces deterministic output with duplicate/missing checks.
"""

from __future__ import annotations

import json
import logging
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Set

from tqdm import tqdm

REPO_ROOT = Path(__file__).resolve().parents[2]
SARA_ROOT = REPO_ROOT / "Baselines" / "SARA-main"
if str(SARA_ROOT) not in sys.path:
    sys.path.insert(0, str(SARA_ROOT))
import src
if hasattr(src, "__path__") and str(SARA_ROOT / "src") not in src.__path__:
    src.__path__.append(str(SARA_ROOT / "src"))

from llama_index.core import Document
from src.model.retriever import getBM25Retriever

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


def build_train_retrieval_cache():
    train_file = REPO_ROOT / "Baselines/SARA-main/data/reformatted/QASPER_train_split.jsonl"
    out_dir = REPO_ROOT / "results/week5/train_response_surface"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_file = out_dir / "train_retrieval.jsonl"

    logger.info(f"Loading raw training questions from {train_file}...")
    raw_items = []
    with open(train_file, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                raw_items.append(json.loads(line))

    n_total_questions = len(raw_items)
    logger.info(f"Loaded {n_total_questions} training questions.")
    assert n_total_questions == 2088, f"Expected 2,088 train questions, found {n_total_questions}"

    # Check existing records in out_file
    existing_by_qid: Dict[str, Dict[str, Any]] = {}
    if out_file.exists():
        with open(out_file, "r", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    rec = json.loads(line)
                    existing_by_qid[str(rec["question_id"])] = rec
        logger.info(f"Found {len(existing_by_qid)} existing retrieval records.")

    if len(existing_by_qid) == n_total_questions:
        logger.info("Retrieval cache is already 100% complete!")
        records = [existing_by_qid[str(item["id"])] for item in raw_items]
    else:
        # Build paper-level BM25 index and retrieve
        paper_retrievers: Dict[str, Any] = {}
        records: List[Dict[str, Any]] = []

        logger.info("Executing paper-level BM25 retrieval across all train questions...")
        t0 = time.time()
        for item in tqdm(raw_items, desc="Train BM25 Retrieval"):
            qid = str(item["id"])
            pid = str(item.get("example_id", ""))

            if qid in existing_by_qid:
                records.append(existing_by_qid[qid])
                continue

            question = item.get("question", "")
            gold_answer = item.get("answer")
            gold_reformatted = item.get("answer_reformatted", gold_answer)

            if pid not in paper_retrievers:
                raw_context = item.get("context", [])
                if isinstance(raw_context, list):
                    docs = [Document(text=str(p)) for p in raw_context if str(p).strip()]
                else:
                    docs = [Document(text=str(raw_context))] if str(raw_context).strip() else []

                if not docs:
                    docs = [Document(text="Empty document.")]

                retriever, _ = getBM25Retriever(docs, similarity_top_k=10, chunk_size=256)
                paper_retrievers[pid] = retriever

            retriever = paper_retrievers[pid]
            nodes = retriever.retrieve(question)

            passages = []
            for rank_idx, node in enumerate(nodes[:10]):
                p_text = str(node.node.get_text()) if hasattr(node, "node") and hasattr(node.node, "get_text") else str(getattr(node, "text", ""))
                passages.append({
                    "passage_id": f"{pid}_chunk_{rank_idx}",
                    "rank": rank_idx + 1,
                    "score": float(node.score) if getattr(node, "score", None) is not None else 0.0,
                    "text": p_text,
                })

            while len(passages) < 10:
                pad_rank = len(passages) + 1
                passages.append({
                    "passage_id": f"{pid}_pad_{pad_rank}",
                    "rank": pad_rank,
                    "score": 0.0,
                    "text": "",
                })

            rec = {
                "question_id": qid,
                "paper_id": pid,
                "question": question,
                "gold_answers": gold_answer if isinstance(gold_answer, list) else [gold_answer],
                "gold_answers_reformatted": gold_reformatted if isinstance(gold_reformatted, list) else [gold_reformatted],
                "passages": passages,
            }
            records.append(rec)
            existing_by_qid[qid] = rec

            with open(out_file, "a", encoding="utf-8") as f:
                f.write(json.dumps(rec) + "\n")

        elapsed = time.time() - t0
        logger.info(f"Completed retrieval for {len(records)} questions in {elapsed:.2f}s.")

    # Audit integrity
    qids_seen: Set[str] = set()
    duplicates: List[str] = []
    invalid_passage_counts: List[str] = []

    for r in records:
        qid = str(r["question_id"])
        if qid in qids_seen:
            duplicates.append(qid)
        qids_seen.add(qid)
        if len(r["passages"]) != 10:
            invalid_passage_counts.append(qid)

    missing_qids = [str(item["id"]) for item in raw_items if str(item["id"]) not in qids_seen]

    audit_report = {
        "total_expected_questions": n_total_questions,
        "total_retrieved_questions": len(records),
        "unique_questions": len(qids_seen),
        "missing_records_count": len(missing_qids),
        "duplicate_records_count": len(duplicates),
        "invalid_passage_count_records": len(invalid_passage_counts),
        "retrieval_coverage_percentage": round(100.0 * len(qids_seen) / n_total_questions, 2),
        "output_file": str(out_file),
    }

    report_path = out_dir / "retrieval_cache_audit.json"
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(audit_report, f, indent=2)

    logger.info(f"Retrieval cache audit: {json.dumps(audit_report, indent=2)}")
    return audit_report


if __name__ == "__main__":
    build_train_retrieval_cache()
