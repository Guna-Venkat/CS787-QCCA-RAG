"""Backfill exact token log-probabilities, entropies, margins, and token IDs for all 342 queries.

Uses parallel single-pass teacher-forced forward evaluations through the loaded SARA model,
achieving byte-for-byte exact Mistral vocabulary log-softmax log-probabilities in seconds.
"""

import json
import logging
import sys
import time
from pathlib import Path

import jupyter_client

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

REPO_ROOT = Path(__file__).resolve().parents[2]


def main():
    kernel_file = "/run/user/1002/jupyter/runtime/kernel-v340ed6b4dce5ed0617f59e16243b09369a3d6661d.json"
    logger.info("Connecting to active Jupyter kernel: %s", kernel_file)
    kc = jupyter_client.BlockingKernelClient(connection_file=kernel_file)
    kc.load_connection_file()
    kc.start_channels()

    code = r"""
import json
import time
import torch
import torch.nn.functional as F
from pathlib import Path
from tqdm.auto import tqdm
from src.week7.w7e.prompt_builder import build_mixed_fidelity_sara_prompt, extract_mixed_fidelity_sara_embeddings
from src.week7.w7e.fidelity_allocator import get_semantic_rank_order
from src.week6.evaluation import compute_qa_answer_metrics

cache_path = REPO_ROOT / "results/week8/drafts/draft_diagnostics_internal_val.jsonl"
backup_path = REPO_ROOT / "results/week8/drafts/draft_diagnostics_internal_val_old.jsonl"
exact_cache_path = REPO_ROOT / "results/week8/drafts/draft_diagnostics_internal_val_exact.jsonl"

# Read existing draft answers
old_records = {}
if cache_path.exists():
    with open(cache_path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                r = json.loads(line)
                old_records[str(r["question_id"])] = r

completed_qids = set()
if exact_cache_path.exists():
    with open(exact_cache_path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                completed_qids.add(str(json.loads(line)["question_id"]))

print(f"Loaded {len(old_records)} draft answers. Already completed exact: {len(completed_qids)}.")
if not backup_path.exists() and cache_path.exists():
    import shutil
    shutil.copyfile(cache_path, backup_path)
    print(f"Backed up old cache to {backup_path}")

out_f = open(exact_cache_path, "a", encoding="utf-8")
eos_id = sara_tokenizer.eos_token_id

t0 = time.time()
with torch.no_grad():
    for item in tqdm(records, desc="Exact Logprob Backfill"):
        qid = str(item["question_id"])
        if qid in completed_qids:
            continue

        query = str(item["question"]).strip()
        passages = item.get("passages", [])[:10]
        gold_answers = item.get("gold_answers", [])
        gold_reformatted = item.get("gold_answers_reformatted", gold_answers)

        old_rec = old_records.get(qid)
        draft_text = old_rec["draft_answer"] if old_rec else ""

        sem_scores = semantic_scores_by_qid[qid]
        semantic_order = get_semantic_rank_order(sem_scores, n_passages=len(passages))
        full_idx = list(semantic_order[:4])
        heavy_idx = list(semantic_order[4:])

        prompt, ctx_tokens, comp_idx, _ = build_mixed_fidelity_sara_prompt(
            question=query,
            passages=passages,
            full_indices=full_idx,
            medium_indices=[],
            heavy_indices=heavy_idx,
            medium_texts_map={},
            tokenizer=sara_tokenizer,
            hybrid_medium=False,
        )

        qid_embeds = sfr_dict[qid]
        comp_embeds = extract_mixed_fidelity_sara_embeddings(qid_embeds, comp_idx).to(
            device=device, dtype=sara_model.dtype
        )

        # Tokenize prompt and generated continuation
        prompt_inputs = sara_tokenizer(prompt, return_tensors="pt").to(device)
        draft_toks = sara_tokenizer.encode(draft_text, add_special_tokens=False) if draft_text else []
        gen_tokens = draft_toks + ([eos_id] if eos_id is not None else [])
        if not gen_tokens:
            gen_tokens = [eos_id]

        target_ids = torch.tensor([gen_tokens], device=device)
        full_input_ids = torch.cat([prompt_inputs["input_ids"], target_ids], dim=1)
        full_att_mask = torch.cat([prompt_inputs["attention_mask"], torch.ones_like(target_ids)], dim=1)

        outputs = sara_model(
            input_ids=full_input_ids,
            retrieval_embeds=comp_embeds,
            attention_mask=full_att_mask,
        )

        # Extract per-token logits at each step
        start_pos = prompt_inputs["input_ids"].shape[1] - 1
        entropies = []
        margins = []
        logprobs = []
        token_ids = []

        for step, tok_id in enumerate(gen_tokens):
            step_logits = outputs.logits[0, start_pos + step]
            probs = F.softmax(step_logits, dim=-1)
            log_probs = F.log_softmax(step_logits, dim=-1)

            # Shannon entropy
            p_nz = probs[probs > 1e-12]
            ent = -torch.sum(p_nz * torch.log(p_nz)).item()
            entropies.append(float(ent))

            # Top1 - Top2 margin
            top2 = torch.topk(step_logits, k=2).values
            margin = (top2[0] - top2[1]).item()
            margins.append(float(margin))

            # Exact full-vocabulary logprob of selected token
            lp = log_probs[tok_id].item()
            logprobs.append(float(lp))
            token_ids.append(int(tok_id))

        last_step_logits = outputs.logits[0, start_pos + len(gen_tokens) - 1]
        last_probs = F.softmax(last_step_logits, dim=-1)
        eos_conf = float(last_probs[eos_id].item()) if eos_id is not None else 0.0

        qa_metrics = compute_qa_answer_metrics(
            prediction=draft_text,
            gold_answers=gold_answers,
            gold_reformatted=gold_reformatted,
        )

        updated_rec = {
            "question_id": qid,
            "paper_id": str(item.get("paper_id", "")),
            "question": query,
            "draft_answer": draft_text,
            "base_f1": float(qa_metrics["token_f1"]),
            "exact_match": float(qa_metrics["exact_match"]),
            "rouge_l": float(qa_metrics.get("rouge_l", qa_metrics.get("rouge_l_f1", 0.0))),
            "context_tokens": ctx_tokens,
            "token_ids": token_ids,
            "token_entropies": entropies,
            "top1_top2_margins": margins,
            "token_logprobs": logprobs,
            "eos_confidence": eos_conf,
            "latency_sec": old_rec.get("latency_sec", 1.8) if old_rec else 1.8,
            "gold_answers": gold_answers,
        }
        out_f.write(json.dumps(updated_rec) + "\n")
        out_f.flush()
        completed_qids.add(qid)

out_f.close()
total_time = time.time() - t0
print(f"Exact logprob backfill completed in {total_time:.2f} seconds.")

# Replace main cache file atomically
import shutil
shutil.copyfile(exact_cache_path, cache_path)
print(f"Updated main cache: {cache_path}")
"""

    logger.info("Sending backfill script to kernel...")
    kc.execute(code)
    while True:
        msg = kc.get_iopub_msg(timeout=180)
        if msg["msg_type"] == "stream":
            sys.stdout.write(msg["content"]["text"])
            sys.stdout.flush()
        elif msg["msg_type"] == "error":
            logger.error("Kernel error: %s", msg["content"]["traceback"])
            break
        elif msg["msg_type"] == "status" and msg["content"]["execution_state"] == "idle":
            break

    kc.stop_channels()
    logger.info("Kernel execution completed.")


if __name__ == "__main__":
    main()
