"""MuSiQue data loading: HF download, local JSONL cache, context formatting.

MuSiQue is a multi-hop QA benchmark where each 2-4 hop question is composed
from single-hop questions across 5 seed datasets (SQuAD, T-REx, Natural Questions,
MLQA, Zero Shot RE).  Each instance provides 10-20 Wikipedia paragraphs (2-4 gold
supporting + distractors), testing retrieval precision and multi-hop synthesis.

Two variants are available via HuggingFace ``bdsaglam/musique``:

* ``answerable`` config (= ``musique_ans``): all questions answerable (~2,417 dev).
* ``default`` config (= ``musique_full``): mix of answerable and unanswerable (~4,834 dev).

Usage::

    from benchmarks.musique.data import load_musique_data, save_predictions, load_predictions

    data = load_musique_data()             # answerable dev set
    data = load_musique_data(variant="full")   # full dev set (incl. unanswerable)
    save_predictions(predictions, path)    # JSONL output
    predictions = load_predictions(path)   # JSONL input (for re-evaluation)
"""

from __future__ import annotations

import json
import os
from collections import Counter

_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
_CACHE_DIR = os.path.join(_PROJECT_ROOT, "data", "benchmarks", "musique")

# HF config name per variant
_VARIANT_CONFIG: dict[str, str] = {
    "ans": "answerable",
    "full": "default",
}


def load_musique_data(
    variant: str = "ans",
    split: str = "validation",
    force_redownload: bool = False,
) -> list[dict]:
    """Load MuSiQue data from HuggingFace, caching locally as JSONL.

    Parameters
    ----------
    variant:
        ``"ans"`` for MuSiQue-Ans (answerable only, ~2,417 dev instances).
        ``"full"`` for MuSiQue-Full (mixed answerable/unanswerable, ~4,834 dev).
    split:
        HuggingFace split name.  ``"validation"`` is the standard dev/eval set.
        ``"train"`` (~19,938 ans / ~39,938 full) and ``"test"`` are also available.
    force_redownload:
        If True, re-download from HuggingFace even if a local cache exists.

    Returns
    -------
    list[dict]
        Each dict has keys: ``question_id``, ``question``, ``answer``,
        ``answer_aliases``, ``answerable``, ``hop_type`` (2hop/3hop1/3hop2/
        4hop1-4), ``num_hops``, ``contexts`` (list of ``[title, text]`` pairs),
        ``supporting_titles`` (list of gold paragraph titles),
        ``question_decomposition`` (list of sub-question dicts).
    """
    hf_config = _VARIANT_CONFIG.get(variant, variant)
    cache_path = _cache_path_for(variant, split)

    # Return cached copy when available
    if os.path.exists(cache_path) and not force_redownload:
        return _read_jsonl(cache_path)

    from datasets import load_dataset as hf_load

    print(f"Downloading bdsaglam/musique ({hf_config}, {split}) from HuggingFace...")
    ds = hf_load("bdsaglam/musique", hf_config, split=split)
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

    # Print hop-type distribution for diagnostics
    hop_counts = Counter(inst.get("hop_type", "?") for inst in instances)
    print(f"  Hop distribution: {dict(sorted(hop_counts.items()))}")

    return instances


def _cache_path_for(variant: str, split: str) -> str:
    """Local JSONL cache path for a given (variant, split) combination."""
    return os.path.join(_CACHE_DIR, f"{variant}_{split}.jsonl")


def _normalise_row(row: dict) -> dict | None:
    """Convert one HuggingFace MuSiQue row into our canonical flat dict.

    Returns None for malformed instances (no question or answer).
    """
    question = (row.get("question") or "").strip()
    answer = (row.get("answer") or "").strip()
    if not question:
        return None
    # Note: unanswerable instances in musique_full have empty answer —
    # we keep them but they score 0 on EM/F1.

    answerable = row.get("answerable", True)
    if isinstance(answerable, str):
        answerable = answerable.lower() == "true"

    question_id = str(row.get("id", ""))

    # ── Hop type from ID ──────────────────────────────────────────
    # MuSiQue IDs look like "2hop__1234_5678", "3hop1__...", "4hop2__..."
    hop_type = "?"
    num_hops = 0
    if "__" in question_id:
        prefix = question_id.split("__")[0]
        hop_type = prefix
        # Extract hop count from prefix: "2hop"→2, "3hop1"→3, "4hop2"→4
        import re
        m = re.match(r"(\d+)", prefix)
        if m:
            num_hops = int(m.group(1))

    # ── Answer aliases ────────────────────────────────────────────
    answer_aliases: list[str] = []
    aliases_raw = row.get("answer_aliases") or []
    if isinstance(aliases_raw, list):
        answer_aliases = [str(a).strip() for a in aliases_raw if str(a).strip()]

    # ── Context paragraphs ─────────────────────────────────────────
    paragraphs = row.get("paragraphs") or []
    contexts: list[list] = []
    supporting_titles: list[str] = []

    for p in paragraphs:
        title = str(p.get("title", "")).strip()
        text = str(p.get("paragraph_text", "")).strip()
        is_supporting = p.get("is_supporting", False)
        if isinstance(is_supporting, str):
            is_supporting = is_supporting.lower() == "true"

        if not title or not text:
            continue

        contexts.append([title, text])
        if is_supporting:
            supporting_titles.append(title)

    if not contexts:
        return None

    # ── Question decomposition ─────────────────────────────────────
    decomposition_raw = row.get("question_decomposition") or []
    decomposition: list[dict] = []
    for d in decomposition_raw:
        decomposition.append({
            "id": d.get("id", ""),
            "question": d.get("question", ""),
            "answer": d.get("answer", ""),
            "paragraph_support_idx": d.get("paragraph_support_idx"),
        })

    return {
        "question_id": question_id,
        "question": question,
        "answer": answer,
        "answer_aliases": answer_aliases,
        "answerable": answerable,
        "hop_type": hop_type,
        "num_hops": num_hops,
        "contexts": contexts,
        "supporting_titles": supporting_titles,
        "question_decomposition": decomposition,
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
    """Join a Wikipedia paragraph title and its text into a single text block."""
    return f"{title}:\n{text}"

