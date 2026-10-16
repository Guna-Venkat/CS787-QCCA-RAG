"""Evidence Utilization and Potential Diagnostics for Week 8 DAFiR.

Formalizes the three-layer conceptual decomposition of Compressed RAG:
Selection -> Fidelity -> Utilization

Decomposes promotion outcomes into a four-way Evidence Potential & Utilization taxonomy:
1. UTILIZED-HELPFUL: Candidate contains gold evidence / high evidence overlap AND Delta_F1 > 0.01.
2. UNDERUTILIZED:    Candidate contains gold evidence / high evidence overlap BUT Delta_F1 <= 0.01.
                    (Exemplified by Q1812: missing information exposed, but generator failed to exploit it).
3. TRUE-NEUTRAL:     Candidate does not contain missing answer info, prediction unchanged (|Delta_F1| <= 0.01).
                    (Exemplified by Q1811: base answer already complete F1=1.0, controller must learn safe-stop).
4. DISTRACTOR:       Promotion decreases answer quality (Delta_F1 < -0.01).

Also supports:
- Dense counterfactual utility signal: Delta_LL = L_promoted - L_base (teacher-forced log-likelihood of gold answer).
- Output stability tracking: AnswerChanged = I[y_promoted != y_base].
- Promotion Utilization Failure Rate (UFR):
  UFR = P(y_promoted == y_base | base_f1 < 1.0, is_gold_evidence == True).
"""

from __future__ import annotations

import json
import logging
import re
import string
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Set, Tuple, Union

import numpy as np
import pandas as pd
import torch

logger = logging.getLogger(__name__)

REPO_ROOT = Path(__file__).resolve().parents[2]


def normalize_text_for_comparison(text: str) -> str:
    """Normalize text by lowercasing and removing punctuation/whitespace."""
    if not text:
        return ""
    text = text.lower()
    text = "".join(ch for ch in text if ch not in string.punctuation)
    return " ".join(text.split())


def compute_token_containment_and_jaccard(
    candidate_text: str,
    target_text: str,
) -> Tuple[float, float]:
    """Compute token containment (recall) and Jaccard similarity between candidate and target text."""
    cand_tokens = set(normalize_text_for_comparison(candidate_text).split())
    target_tokens = set(normalize_text_for_comparison(target_text).split())

    if not cand_tokens or not target_tokens:
        return 0.0, 0.0

    intersection = cand_tokens.intersection(target_tokens)
    containment = len(intersection) / len(target_tokens)
    jaccard = len(intersection) / len(cand_tokens.union(target_tokens))

    return float(containment), float(jaccard)


_CACHED_QASPER_EVIDENCE_MAP: Optional[Dict[str, List[str]]] = None


def load_qasper_gold_evidence_map() -> Dict[str, List[str]]:
    """Load mapping from normalized question string -> list of ground truth annotated evidence paragraphs.

    Caches mapping in memory across calls.
    """
    global _CACHED_QASPER_EVIDENCE_MAP
    if _CACHED_QASPER_EVIDENCE_MAP is not None:
        return _CACHED_QASPER_EVIDENCE_MAP

    evidence_map: Dict[str, List[str]] = {}

    # Attempt to load from allenai/qasper in datasets
    try:
        import datasets
        for split_name in ["train", "validation"]:
            try:
                ds = datasets.load_dataset("allenai/qasper", split=split_name)
                for row in ds:
                    for i, q in enumerate(row["qas"]["question"]):
                        q_norm = normalize_text_for_comparison(q)
                        ans_list = row["qas"]["answers"][i].get("answer", [])
                        ev_list: List[str] = []
                        for ans in ans_list:
                            for ev in ans.get("evidence", []):
                                if ev and isinstance(ev, str):
                                    ev_list.append(ev)
                        if ev_list:
                            existing = evidence_map.get(q_norm, [])
                            evidence_map[q_norm] = list(set(existing + ev_list))
            except Exception as e:
                logger.debug("Failed loading QASPER split %s: %s", split_name, e)
    except Exception as e:
        logger.warning("Could not load QASPER datasets for evidence mapping: %s", e)

    _CACHED_QASPER_EVIDENCE_MAP = evidence_map
    logger.info("Loaded QASPER gold evidence map for %d unique questions.", len(evidence_map))
    return _CACHED_QASPER_EVIDENCE_MAP


def evaluate_evidence_overlap(
    candidate_text: str,
    question: str,
    gold_answers: Sequence[str],
    evidence_map: Optional[Dict[str, List[str]]] = None,
    containment_threshold: float = 0.35,
) -> Tuple[bool, float]:
    """Check if candidate passage contains gold evidence or matches annotated paragraphs.

    Args:
        candidate_text: Text of candidate passage being promoted.
        question: Question text.
        gold_answers: Ground truth answers.
        evidence_map: Preloaded question -> evidence paragraphs mapping.
        containment_threshold: Minimum token containment to qualify as gold evidence.

    Returns:
        (is_gold_evidence, max_overlap_score)
    """
    if evidence_map is None:
        evidence_map = load_qasper_gold_evidence_map()

    q_norm = normalize_text_for_comparison(question)
    annotated_evidence = evidence_map.get(q_norm, [])

    max_containment = 0.0

    # 1. Compare against annotated evidence paragraphs
    for ev in annotated_evidence:
        cont, jacc = compute_token_containment_and_jaccard(candidate_text, ev)
        # Also check reverse containment (if annotated paragraph is a snippet of the passage)
        rev_cont, _ = compute_token_containment_and_jaccard(ev, candidate_text)
        score = max(cont, rev_cont, jacc)
        if score > max_containment:
            max_containment = score

    # 2. Compare against gold answer text directly
    for ans in gold_answers:
        if not ans:
            continue
        ans_norm = normalize_text_for_comparison(ans)
        cand_norm = normalize_text_for_comparison(candidate_text)
        if ans_norm and ans_norm in cand_norm:
            max_containment = max(max_containment, 1.0)
        else:
            cont, _ = compute_token_containment_and_jaccard(candidate_text, ans)
            if cont > max_containment:
                max_containment = cont

    is_gold = max_containment >= containment_threshold
    return is_gold, float(max_containment)


def classify_evidence_utilization_type(
    is_gold_evidence: bool,
    delta_f1: float,
    epsilon_help: float = 0.01,
    epsilon_hurt: float = -0.01,
) -> str:
    """Classify promotion into 4-way Evidence Potential & Utilization taxonomy.

    Categories:
    - DISTRACTOR: delta_f1 < -0.01 (harmful promotion).
    - UTILIZED_HELPFUL: is_gold_evidence == True AND delta_f1 > 0.01 (generator exploited evidence).
    - UNDERUTILIZED: is_gold_evidence == True AND delta_f1 <= 0.01 (generator failed to exploit evidence).
    - TRUE_NEUTRAL: is_gold_evidence == False AND abs(delta_f1) <= 0.01 (neutral passage).
    """
    if delta_f1 < epsilon_hurt:
        return "DISTRACTOR"
    elif is_gold_evidence and delta_f1 > epsilon_help:
        return "UTILIZED_HELPFUL"
    elif is_gold_evidence and delta_f1 <= epsilon_help:
        return "UNDERUTILIZED"
    else:
        return "TRUE_NEUTRAL"


def compute_teacher_forced_gold_logprob(
    sara_model: Any,
    sara_tokenizer: Any,
    prompt_text: str,
    gold_answer: str,
    comp_embeds: Optional[torch.Tensor] = None,
    device: str = "cuda",
) -> float:
    """Compute normalized sequence log-likelihood of gold answer tokens:

    L = (1 / |y*|) sum_t log P(y_t* | q, C, y*_{<t})

    Returns L (which is -loss).
    """
    full_prompt_text = prompt_text + "\nAnswer: " + gold_answer
    prompt_inputs = sara_tokenizer(prompt_text + "\nAnswer: ", return_tensors="pt")
    full_inputs = sara_tokenizer(full_prompt_text, return_tensors="pt").to(device)

    prompt_len = prompt_inputs["input_ids"].shape[1]
    input_ids = full_inputs["input_ids"]
    labels = input_ids.clone()
    labels[:, :prompt_len] = -100

    if (labels != -100).sum() == 0:
        return 0.0

    model_kwargs = {"input_ids": input_ids, "labels": labels}
    if comp_embeds is not None:
        model_kwargs["retrieval_embeds"] = comp_embeds.to(device=device, dtype=sara_model.dtype)

    with torch.no_grad():
        outputs = sara_model(**model_kwargs)
        nll = float(outputs.loss.item())

    # Return log-probability: -NLL
    return -nll


def enrich_promotion_record(
    record: Dict[str, Any],
    evidence_map: Optional[Dict[str, List[str]]] = None,
    epsilon_help: float = 0.01,
    epsilon_hurt: float = -0.01,
) -> Dict[str, Any]:
    """Enrich a promotion record with evidence potential and utilization diagnostics."""
    enriched = dict(record)

    # 1. Output change detection
    base_pred = str(enriched.get("base_prediction", "")).strip()
    prom_pred = str(enriched.get("promoted_prediction", "")).strip()
    norm_base = normalize_text_for_comparison(base_pred)
    norm_prom = normalize_text_for_comparison(prom_pred)
    enriched["answer_changed"] = bool(norm_base != norm_prom)

    # 2. Base answer completeness (F1 >= 0.99)
    base_f1 = float(enriched.get("base_f1", 0.0))
    enriched["base_answer_complete"] = bool(base_f1 >= 0.99)

    # 3. Gold evidence alignment & overlap
    cand_text = str(enriched.get("candidate_text", ""))
    question = str(enriched.get("question", ""))
    gold_answers = enriched.get("gold_answers", [])

    is_gold, overlap = evaluate_evidence_overlap(
        candidate_text=cand_text,
        question=question,
        gold_answers=gold_answers,
        evidence_map=evidence_map,
    )
    enriched["is_gold_evidence"] = bool(is_gold)
    enriched["gold_evidence_overlap"] = float(overlap)

    # 4. 4-way evidence utilization taxonomy
    delta_f1 = float(enriched.get("delta_f1", 0.0))
    enriched["evidence_utilization_type"] = classify_evidence_utilization_type(
        is_gold_evidence=is_gold,
        delta_f1=delta_f1,
        epsilon_help=epsilon_help,
        epsilon_hurt=epsilon_hurt,
    )

    return enriched


def enrich_promotion_jsonl_file(
    input_path: Union[str, Path],
    output_path: Union[str, Path],
    evidence_map: Optional[Dict[str, List[str]]] = None,
) -> pd.DataFrame:
    """Read a promotion surface JSONL, enrich all records, and save to output_path."""
    in_p = Path(input_path)
    out_p = Path(output_path)
    out_p.parent.mkdir(parents=True, exist_ok=True)

    if evidence_map is None:
        evidence_map = load_qasper_gold_evidence_map()

    enriched_records = []
    with open(in_p, "r", encoding="utf-8") as fin:
        for line in fin:
            if not line.strip():
                continue
            rec = json.loads(line)
            en_rec = enrich_promotion_record(rec, evidence_map=evidence_map)
            enriched_records.append(en_rec)

    with open(out_p, "w", encoding="utf-8") as fout:
        for r in enriched_records:
            fout.write(json.dumps(r) + "\n")

    logger.info("Enriched %d promotion records. Saved to %s", len(enriched_records), out_p)
    return pd.DataFrame(enriched_records)
