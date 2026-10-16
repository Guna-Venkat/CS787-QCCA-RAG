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

    # Check local precomputed JSON cache first for instant offline loading
    local_cache_path = Path("/home/gunavenkat/Downloads/CS787-RAG-Project/results/week8/shared_cache/qasper_gold_evidence_map.json")
    if local_cache_path.exists():
        try:
            import json
            with open(local_cache_path, "r", encoding="utf-8") as f:
                evidence_map = json.load(f)
            _CACHED_QASPER_EVIDENCE_MAP = evidence_map
            logger.info("Loaded QASPER evidence map instantly from local cache: %s (%d questions)", local_cache_path, len(evidence_map))
            return _CACHED_QASPER_EVIDENCE_MAP
        except Exception as e:
            logger.debug("Failed reading local cache: %s", e)

    # Fallback to load from allenai/qasper in datasets
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
    containment_threshold: float = 0.65,
    jaccard_threshold: float = 0.28,
) -> Tuple[bool, float]:
    """Check if candidate passage contains gold evidence or matches annotated paragraphs.

    Requires genuine alignment to annotated paragraphs (containment >= 0.65 or jaccard >= 0.28)
    or exact substring match of substantive gold answer tokens.
    """
    if evidence_map is None:
        evidence_map = load_qasper_gold_evidence_map()

    q_norm = normalize_text_for_comparison(question)
    annotated_evidence = evidence_map.get(q_norm, [])

    max_score = 0.0

    # 1. Compare against annotated evidence paragraphs
    for ev in annotated_evidence:
        cont, jacc = compute_token_containment_and_jaccard(candidate_text, ev)
        rev_cont, _ = compute_token_containment_and_jaccard(ev, candidate_text)
        effective_cont = max(cont, rev_cont)
        if effective_cont >= containment_threshold or jacc >= jaccard_threshold:
            max_score = max(max_score, max(effective_cont, jacc))

    # 2. Compare against gold answer text directly (exact phrase substring or high-precision token match)
    cand_norm = normalize_text_for_comparison(candidate_text)
    cand_tokens = set(cand_norm.split())
    stopwords = {"the", "a", "an", "in", "on", "of", "and", "or", "is", "are", "to", "for", "with", "by", "at", "from"}

    for ans in gold_answers:
        if not ans:
            continue
        ans_norm = normalize_text_for_comparison(ans)
        if not ans_norm:
            continue
        # Substring match for short or exact answers
        if len(ans_norm) >= 4 and ans_norm in cand_norm:
            max_score = max(max_score, 1.0)
        else:
            ans_content_tokens = set(ans_norm.split()) - stopwords
            if len(ans_content_tokens) >= 3:
                intersection = cand_tokens.intersection(ans_content_tokens)
                ans_cont = len(intersection) / len(ans_content_tokens)
                if ans_cont >= 0.80:
                    max_score = max(max_score, ans_cont)

    is_gold = max_score >= 0.28
    return is_gold, float(max_score)


def classify_evidence_utilization_type(
    is_gold_evidence: bool,
    delta_f1: float,
    base_f1: float = 0.0,
    epsilon_help: float = 0.01,
    epsilon_hurt: float = -0.01,
) -> str:
    """Classify promotion into 6-way Evidence Potential & Utilization taxonomy:

    1. UTILIZED_HELPFUL: is_gold_evidence == True AND delta_f1 > +0.01.
       (Annotated gold evidence exploited by generator to improve answer).
    2. UNANNOTATED_HELPFUL: is_gold_evidence == False AND delta_f1 > +0.01.
       (Serendipitous or unannotated evidence gain).
    3. HARMFUL_EVIDENCE: is_gold_evidence == True AND delta_f1 < -0.01.
       (Annotated gold passage confused generator, hurting F1).
    4. DISTRACTOR: is_gold_evidence == False AND delta_f1 < -0.01.
       (Non-evidence passage actively degraded generation).
    5. UNDERUTILIZED: is_gold_evidence == True AND base_f1 < 0.99 AND delta_f1 <= 0.01.
       (Generator had incomplete base answer, gold evidence was available in context, but failed to exploit it).
    6. REDUNDANT_EVIDENCE: is_gold_evidence == True AND base_f1 >= 0.99 AND delta_f1 <= 0.01.
       (Base answer was already complete F1>=0.99; evidence promotion is redundant).
    7. TRUE_NEUTRAL: is_gold_evidence == False AND |delta_f1| <= 0.01.
       (Non-evidence passage caused no change in answer quality).
    """
    if delta_f1 < epsilon_hurt:
        return "HARMFUL_EVIDENCE" if is_gold_evidence else "DISTRACTOR"
    elif delta_f1 > epsilon_help:
        return "UTILIZED_HELPFUL" if is_gold_evidence else "UNANNOTATED_HELPFUL"
    elif is_gold_evidence:
        if base_f1 < 0.99:
            return "UNDERUTILIZED"
        else:
            return "REDUNDANT_EVIDENCE"
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

    # 4. Evidence utilization taxonomy
    delta_f1 = float(enriched.get("delta_f1", 0.0))
    enriched["evidence_utilization_type"] = classify_evidence_utilization_type(
        is_gold_evidence=is_gold,
        delta_f1=delta_f1,
        base_f1=base_f1,
        epsilon_help=epsilon_help,
        epsilon_hurt=epsilon_hurt,
    )

    return enriched


def compute_evidence_utilization_summary(df: pd.DataFrame) -> Dict[str, Any]:
    """Compute comprehensive Evidence Potential & Utilization metrics.

    Enforces mathematically rigorous UFR denominator:
    UFR = N(gold_evidence AND base_incomplete AND delta_f1 <= 0.01) /
          N(gold_evidence AND base_incomplete)
    """
    total = len(df)
    if total == 0:
        return {}

    tax_counts = df["evidence_utilization_type"].value_counts().to_dict()
    n_utilized_gold = tax_counts.get("UTILIZED_HELPFUL", 0)
    n_unannotated_help = tax_counts.get("UNANNOTATED_HELPFUL", 0)
    n_under = tax_counts.get("UNDERUTILIZED", 0)
    n_redundant = tax_counts.get("REDUNDANT_EVIDENCE", 0)
    n_harmful_ev = tax_counts.get("HARMFUL_EVIDENCE", 0)
    n_distractor = tax_counts.get("DISTRACTOR", 0)
    n_neutral = tax_counts.get("TRUE_NEUTRAL", 0)

    # All gold evidence promotions
    is_gold = df["is_gold_evidence"] == True
    total_gold_promotions = int(is_gold.sum())

    # Substantive gold opportunity: base answer incomplete (base_f1 < 0.99) AND candidate is gold evidence
    base_incomplete = df["base_f1"] < 0.99
    gold_incomplete_mask = is_gold & base_incomplete
    n_gold_incomplete = int(gold_incomplete_mask.sum())

    # Within gold_incomplete:
    # Failure: delta_f1 <= 0.01 (includes UNDERUTILIZED and HARMFUL_EVIDENCE)
    gold_incomplete_df = df[gold_incomplete_mask]
    n_ufr_failures = int((gold_incomplete_df["delta_f1"] <= 0.01).sum())
    n_her_successes = int((gold_incomplete_df["delta_f1"] > 0.01).sum())

    ufr = (n_ufr_failures / n_gold_incomplete) if n_gold_incomplete > 0 else 0.0
    her = (n_her_successes / n_gold_incomplete) if n_gold_incomplete > 0 else 0.0
    rer = (n_redundant / total_gold_promotions) if total_gold_promotions > 0 else 0.0

    # Answer changed rates
    ans_chg_overall = float(df["answer_changed"].mean()) if "answer_changed" in df.columns else 0.0
    ans_chg_helpful = float(df[df["delta_f1"] > 0.01]["answer_changed"].mean()) if len(df[df["delta_f1"] > 0.01]) > 0 else 0.0
    ans_chg_under = float(df[df["evidence_utilization_type"] == "UNDERUTILIZED"]["answer_changed"].mean()) if n_under > 0 else 0.0

    summary = {
        "total_promotions": total,
        "counts": {
            "UTILIZED_HELPFUL": n_utilized_gold,
            "UNANNOTATED_HELPFUL": n_unannotated_help,
            "UNDERUTILIZED": n_under,
            "REDUNDANT_EVIDENCE": n_redundant,
            "HARMFUL_EVIDENCE": n_harmful_ev,
            "DISTRACTOR": n_distractor,
            "TRUE_NEUTRAL": n_neutral,
        },
        "percentages": {
            "UTILIZED_HELPFUL": (n_utilized_gold / total) * 100.0,
            "UNANNOTATED_HELPFUL": (n_unannotated_help / total) * 100.0,
            "UNDERUTILIZED": (n_under / total) * 100.0,
            "REDUNDANT_EVIDENCE": (n_redundant / total) * 100.0,
            "HARMFUL_EVIDENCE": (n_harmful_ev / total) * 100.0,
            "DISTRACTOR": (n_distractor / total) * 100.0,
            "TRUE_NEUTRAL": (n_neutral / total) * 100.0,
        },
        "total_gold_promotions": total_gold_promotions,
        "substantive_gold_opportunity": n_gold_incomplete,
        "ufr_failures_count": n_ufr_failures,
        "her_successes_count": n_her_successes,
        "real_utilization_failure_rate_ufr": ufr * 100.0,
        "helpful_evidence_rate_her": her * 100.0,
        "redundant_evidence_rate_rer": rer * 100.0,
        "p_answer_changed_overall": ans_chg_overall * 100.0,
        "p_answer_changed_helpful": ans_chg_helpful * 100.0,
        "p_answer_changed_underutilized": ans_chg_under * 100.0,
    }
    return summary


def build_g1_revision_sara_prompt(
    question: str,
    draft: str,
    passages: Sequence[Dict[str, Any]],
    full_indices: Sequence[int],
    comp_indices: Sequence[int],
    tokenizer: Any,
) -> Tuple[str, int]:
    """Build G1 draft-aware revision prompt for SARA with mixed-fidelity context:

    Instruction format:
    Initial answer:
    {draft}

    Additional evidence has been made available.

    Re-evaluate the initial answer using the new evidence.
    Revise it only when the evidence adds necessary information
    or corrects the initial response.

    Return a complete but concise answer supported by the context.
    """
    comp_token = "<COMPRESS>"
    context_lines = []

    for idx in range(min(10, len(passages))):
        p = passages[idx]
        p_text = p.get("text", "") if isinstance(p, dict) else str(p)
        if idx in full_indices:
            context_lines.append(f"Passage {idx + 1}: {p_text}")
        else:
            context_lines.append(f"Passage {idx + 1}: {comp_token}")

    context_str = "\n\n".join(context_lines)

    revision_instruction = (
        f"Initial answer:\n{draft}\n\n"
        "Additional evidence has been made available.\n\n"
        "Re-evaluate the initial answer using the new evidence.\n"
        "Revise it only when the evidence adds necessary information\n"
        "or corrects the initial response.\n\n"
        "Return a complete but concise answer supported by the context."
    )

    prompt = (
        f"You are a helpful assistant. Use the following context and initial answer to answer the question.\n\n"
        f"Context:\n{context_str}\n\n"
        f"Question: {question}\n\n"
        f"{revision_instruction}\n\n"
        f"Answer:"
    )

    tok_len = len(tokenizer.encode(prompt))
    return prompt, tok_len


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
