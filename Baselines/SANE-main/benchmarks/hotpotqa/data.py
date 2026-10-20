"""HotpotQA data loading: HF download, local JSONL cache, context formatting.

HotpotQA is a multi-hop QA benchmark where each question requires reasoning across
2+ Wikipedia documents. The distractor setting provides 2 gold + 8 distractor paragraphs
per question — testing both retrieval and multi-hop synthesis.

Data source: HuggingFace ``hotpotqa/hotpot_qa`` (distractor config, validation split).

Usage::

    from benchmarks.hotpotqa.data import load_hotpotqa_data, save_predictions, load_predictions

    data = load_hotpotqa_data()        # downloads + caches on first call
    save_predictions(predictions, path) # JSONL output
    predictions = load_predictions(path) # JSONL input (for re-evaluation)
"""

from __future__ import annotations

import json
import os

_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
_CACHE_DIR = os.path.join(_PROJECT_ROOT, "data", "benchmarks", "hotpotqa")
_CACHE_FILE = os.path.join(_CACHE_DIR, "dev_distractor.jsonl")


def load_hotpotqa_data(
    split: str = "validation",
    config: str = "distractor",
    force_redownload: bool = False,
) -> list[dict]:
    """Load HotpotQA data from HuggingFace, caching locally as JSONL.

    Parameters
    ----------
    split:
        HuggingFace split name.  ``"validation"`` (~7,405 instances) is the
        standard dev/eval set.  ``"train"`` (~90,000 instances) is available
        for large-scale experiments.
    config:
        Dataset configuration.  ``"distractor"`` includes 2 gold + 8 distractor
        paragraphs (standard retrieval challenge).  ``"fullwiki"`` provides
        top-10 IR-retrieved paragraphs from Wikipedia.
    force_redownload:
        If True, re-download from HuggingFace even if a local cache exists.

    Returns
    -------
    list[dict]
        Each dict has keys: ``question_id``, ``question``, ``answer``,
        ``type`` (bridge/comparison), ``level`` (easy/medium/hard),
        ``contexts`` (list of ``[title, text]`` pairs),
        ``supporting_facts`` (list of ``[title, sent_id]`` pairs).
    """
    cache_path = _cache_path_for(split, config)

    # Return cached copy when available
    if os.path.exists(cache_path) and not force_redownload:
        return _read_jsonl(cache_path)

    from datasets import load_dataset as hf_load

    print(f"Downloading hotpotqa/hotpot_qa ({config}, {split}) from HuggingFace...")
    ds = hf_load("hotpotqa/hotpot_qa", config, split=split)
    print(f"  Loaded {len(ds)} instances")

    # Convert to our flat dict format
    instances: list[dict] = []
    for row in ds:
        inst = _normalise_row(row)
        if inst is not None:
            instances.append(inst)

    # Write local cache
    os.makedirs(os.path.dirname(cache_path), exist_ok=True)
    _write_jsonl(instances, cache_path)
    print(f"  Cached {len(instances)} instances → {cache_path}")

    return instances


def _cache_path_for(split: str, config: str) -> str:
    """Local JSONL cache path for a given (config, split) combination."""
    return os.path.join(_CACHE_DIR, f"{config}_{split}.jsonl")


def _normalise_row(row: dict) -> dict | None:
    """Convert one HuggingFace row into our canonical flat dict.

    Returns None for malformed instances (e.g., fewer than 2 supporting
    paragraphs — these aren't testable multi-hop questions).
    """
    question = row.get("question", "")
    answer = row.get("answer", "")
    if not question or not answer:
        return None

    # ── Context paragraphs ──────────────────────────────────────────
    ctx = row.get("context", {})
    titles: list[str] = ctx.get("title", [])
    sentences_list: list[list[str]] = ctx.get("sentences", [])
    # Some HF versions nest sentences inside a list-of-dicts;
    # normalise to list[list[str]]
    if sentences_list and isinstance(sentences_list[0], dict):
        sentences_list = [s.get("sentences", s) for s in sentences_list]
        sentences_list = [
            s if isinstance(s, list) else [str(s)]
            for s in sentences_list
        ]

    contexts: list[list] = []
    for title, sents in zip(titles, sentences_list):
        if isinstance(sents, list):
            text = " ".join(str(s) for s in sents)
        else:
            text = str(sents)
        if text.strip():
            contexts.append([str(title), text])

    if not contexts:
        return None

    # ── Supporting facts ─────────────────────────────────────────────
    sf = row.get("supporting_facts", {})
    sf_titles: list[str] = sf.get("title", [])
    sf_sent_ids: list[int] = sf.get("sent_id", [])
    supporting_facts: list[list] = []
    for t, sid in zip(sf_titles, sf_sent_ids):
        supporting_facts.append([str(t), int(sid)])

    if len(supporting_facts) < 2:
        # Not a multi-hop question — skip
        return None

    q_type = row.get("type", "bridge")
    level = row.get("level", "medium")

    return {
        "question_id": str(row.get("id", "")),
        "question": question,
        "answer": answer,
        "type": q_type,
        "level": level,
        "contexts": contexts,
        "supporting_facts": supporting_facts,
    }


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


# ── Formatting helpers ───────────────────────────────────────────────────────

def format_paragraph(title: str, text: str) -> str:
    """Join a Wikipedia paragraph title and its sentences into a single text block."""
    return f"{title}:\n{text}"

