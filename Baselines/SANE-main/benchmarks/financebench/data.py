"""FinanceBench data loading: local JSONL + PDF text cache → canonical dicts.

FinanceBench is a financial QA benchmark over SEC filings (10-K, 10-Q, 8-K,
earnings transcripts).  The open-source subset has 150 instances across 3 question
types (metrics-generated, domain-relevant, novel-generated) referencing 84 unique
PDFs.

Data source: local files in ``third_party/financebench/``.

PDF text extraction is a separate pre-computation step (``benchmarks/financebench/setup.py``)
that must be run once before any benchmark run.  The extracted page texts are
cached to ``data/benchmarks/financebench/pdf_texts.json`` and loaded instantly
by :func:`load_financebench_data`.

Usage::

    from benchmarks.financebench.data import load_financebench_data, save_predictions, load_predictions

    data = load_financebench_data()        # requires PDF cache to exist
    save_predictions(predictions, path)    # JSONL output
    predictions = load_predictions(path)   # JSONL input (for re-evaluation)
"""

from __future__ import annotations

import json
import os

_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
_CACHE_DIR = os.path.join(_PROJECT_ROOT, "data", "benchmarks", "financebench")
_PDF_CACHE_FILE = os.path.join(_CACHE_DIR, "pdf_texts.json")
_FB_DATA_DIR = os.path.join(_PROJECT_ROOT, "third_party", "financebench", "data")
_QUESTIONS_FILE = os.path.join(_FB_DATA_DIR, "financebench_open_source.jsonl")


def load_financebench_data() -> list[dict]:
    """Load FinanceBench open-source data, merging with pre-extracted PDF page texts.

    The PDF text cache at ``data/benchmarks/financebench/pdf_texts.json`` must
    already exist — run ``benchmarks/financebench/setup.py`` first.  Loading
    from the cache is instant (no PDF extraction at runtime).

    Returns
    -------
    list[dict]
        Each dict has keys: ``financebench_id``, ``question``, ``answer``,
        ``justification``, ``question_type``, ``question_reasoning``,
        ``company``, ``doc_name``, ``evidence`` (list of dicts with
        ``evidence_text``, ``doc_name``, ``evidence_page_num``,
        ``evidence_text_full_page``), ``pdf_path``, ``pdf_pages``.
    """
    # ── Load PDF text cache ───────────────────────────────────────────
    if not os.path.exists(_PDF_CACHE_FILE):
        raise FileNotFoundError(
            f"PDF text cache not found: {_PDF_CACHE_FILE}\n"
            f"Run 'python benchmarks/financebench/setup.py --workers 8' first."
        )
    with open(_PDF_CACHE_FILE) as f:
        pdf_cache = json.load(f)
    print(f"Loaded PDF text cache: {len(pdf_cache)} documents")

    # ── Load question instances ───────────────────────────────────────
    if not os.path.exists(_QUESTIONS_FILE):
        raise FileNotFoundError(
            f"FinanceBench data not found: {_QUESTIONS_FILE}\n"
            f"Expected third_party/financebench/data/financebench_open_source.jsonl"
        )
    raw_instances = _read_jsonl(_QUESTIONS_FILE)
    print(f"Loaded {len(raw_instances)} raw instances from financebench_open_source.jsonl")

    # ── Merge into canonical dicts ────────────────────────────────────
    instances: list[dict] = []
    skipped_pdf = 0

    for row in raw_instances:
        doc_name = row.get("doc_name", "")
        if not doc_name:
            continue

        # Resolve PDF path
        pdf_path = os.path.join(
            _PROJECT_ROOT, "third_party", "financebench", "pdfs", f"{doc_name}.pdf"
        )

        # Look up page texts in cache
        pdf_pages = pdf_cache.get(doc_name)
        if pdf_pages is None:
            skipped_pdf += 1
            print(f"  Warning: PDF not in cache — {doc_name}")
            pdf_pages = []

        # Parse evidence list
        evidence = []
        for ev in row.get("evidence", []):
            evidence.append({
                "evidence_text": ev.get("evidence_text", ""),
                "doc_name": ev.get("doc_name", doc_name),
                "evidence_page_num": int(ev.get("evidence_page_num", 0)),
                "evidence_text_full_page": ev.get("evidence_text_full_page", ""),
            })

        instances.append({
            "financebench_id": str(row.get("financebench_id", "")),
            "question": row.get("question", ""),
            "answer": row.get("answer", ""),
            "justification": row.get("justification", ""),
            "question_type": row.get("question_type", ""),
            "question_reasoning": row.get("question_reasoning"),
            "company": row.get("company", ""),
            "doc_name": doc_name,
            "evidence": evidence,
            "pdf_path": pdf_path,
            "pdf_pages": pdf_pages,
        })

    if skipped_pdf:
        print(f"  Warning: {skipped_pdf} instance(s) have missing PDFs")

    print(f"  Prepared {len(instances)} canonical instances")
    return instances


# ── JSONL I/O ──────────────────────────────────────────────────────────────

def _read_jsonl(path: str) -> list[dict]:
    with open(path) as f:
        return [json.loads(line) for line in f if line.strip()]


def _write_jsonl(data: list[dict], path: str):
    with open(path, "w") as f:
        for obj in data:
            f.write(json.dumps(obj, ensure_ascii=False) + "\n")


def save_predictions(predictions: list[dict], path: str):
    """Write predictions to a JSONL file (one JSON object per line)."""
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    _write_jsonl(predictions, path)


def load_predictions(path: str) -> list[dict]:
    """Read predictions from a JSONL file."""
    return _read_jsonl(path)

