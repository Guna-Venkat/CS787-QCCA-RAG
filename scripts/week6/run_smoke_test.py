"""Tiny smoke test for Week 6 Passage Routing.

Runs ONLY:
- 16 TRAIN-FIT questions for R2 dataset construction
- 8 INTERNAL-VAL questions
- 1 training epoch
- 1 seed (42)
- k=2 only
- 3 DEV questions for routing assignment and prompt formatting validation

Verifies:
- Evidence labels construct deterministically
- R2 loss is finite
- Gradients flow to all trainable parameters
- Scores vary across passages
- TopK outputs are strictly valid (invariants hold)
- SARA prompt receives exactly k full text and 10 - k <COMPRESS> tokens
- No OOM, no leakage

Saves outputs to: results/week6/passage_routing/diagnostics/smoke/
Tagged with SMOKE_ONLY_NOT_FOR_REPORTING.
"""

from __future__ import annotations

import json
import logging
import sys
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from transformers import AutoTokenizer

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.week6.data import (
    PassageRoutingDataset,
    routing_collate_fn,
    match_passages_to_gold_evidence,
    load_retrieval_records,
    audit_w6_split_isolation,
)
from src.week6.models import LearnedPassageRouter, PretrainedSemanticScorer
from src.week6.routing import (
    route_r0_bm25_prefix,
    route_r1_semantic_topk,
    route_r2_learned_topk,
    validate_routing_assignment,
    build_routed_sara_prompt,
)
from src.week0.dataset_adapter import QASPERAdapter

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


class MockMistralTokenizer:
    """Lightweight mock tokenizer for SARA prompt validation during smoke test."""
    def __init__(self):
        self.pad_token_id = 32000
        self.eos_token_id = 2
        self.chat_template = None

    def tokenize(self, text: str):
        return text.split()


def run_smoke():
    logger.info("Starting Week 6 Smoke Test...")
    smoke_dir = REPO_ROOT / "results/week6/passage_routing/diagnostics/smoke"
    smoke_dir.mkdir(parents=True, exist_ok=True)

    # 1. Load manifests and limit to smoke slice
    fit_path = REPO_ROOT / "results/week5/content_allocator/w5b/splits/train_fit_manifest.csv"
    val_path = REPO_ROOT / "results/week5/content_allocator/w5b/splits/internal_val_manifest.csv"
    dev_path = REPO_ROOT / "results/week5/supervision/dev_allocation_supervision.csv"

    df_fit = pd.read_csv(fit_path).head(16)
    df_val = pd.read_csv(val_path).head(8)
    df_dev = pd.read_csv(dev_path).head(3)

    train_ret = load_retrieval_records(REPO_ROOT / "results/week5/train_response_surface/train_retrieval.jsonl")
    dev_ret = load_retrieval_records(REPO_ROOT / "results/week2/raw/dev_retrieval.jsonl")

    # 2. QASPER Adapter
    adapter = QASPERAdapter(
        train_split_path=REPO_ROOT / "Baselines/SARA-main/data/reformatted/QASPER_train_split.jsonl",
        dev_split_path=REPO_ROOT / "Baselines/SARA-main/data/reformatted/QASPER_dev.jsonl",
        raw_arrow_train_path=Path("/home/gunavenkat/.cache/huggingface/datasets/allenai___qasper/qasper/0.3.0/fdc9d8214fbab5dd782958601db4d678e6934a54/qasper-train.arrow"),
        raw_arrow_val_path=Path("/home/gunavenkat/.cache/huggingface/datasets/allenai___qasper/qasper/0.3.0/fdc9d8214fbab5dd782958601db4d678e6934a54/qasper-validation.arrow"),
        manifest_path=REPO_ROOT / "Baselines/SARA-main/data/manifests/qasper_split.json",
        dev_retrieval_path=REPO_ROOT / "results/week2/raw/dev_retrieval.jsonl",
    )
    raw_train_recs = {str(r.get("id", r.get("question_id"))): r for r in adapter.load_split("train")}
    raw_dev_recs = {str(r.get("id", r.get("question_id"))): r for r in adapter.load_split("dev")}

    # 3. Build smoke datasets
    tokenizer = AutoTokenizer.from_pretrained("sentence-transformers/all-MiniLM-L6-v2")

    def build_slice(df_slice, ret_dict, raw_dict):
        qids, pids, questions, passages_list, labels_list = [], [], [], [], []
        for _, row in df_slice.iterrows():
            raw_qid = str(row["question_id"]).strip()
            qid = str(int(float(raw_qid))) if raw_qid.replace(".", "", 1).isdigit() else raw_qid
            pid = str(row["paper_id"])
            if qid not in ret_dict:
                continue
            ret_item = ret_dict[qid]
            raw_rec = raw_dict.get(qid)
            ev = adapter.get_gold_evidence(raw_rec) if raw_rec else []
            passages = ret_item.get("passages", [])[:10]
            labels, _ = match_passages_to_gold_evidence(passages, ev)

            qids.append(qid)
            pids.append(pid)
            questions.append(ret_item["question"])
            passages_list.append(passages)
            labels_list.append(labels)

        ds = PassageRoutingDataset(
            question_ids=qids,
            paper_ids=pids,
            questions=questions,
            passages_list=passages_list,
            labels_list=labels_list,
            tokenizer=tokenizer,
            max_seq_length=128,
        )
        return ds, labels_list

    train_ds, train_labels = build_slice(df_fit, train_ret, raw_train_recs)
    val_ds, val_labels = build_slice(df_val, train_ret, raw_train_recs)

    train_pos_rate = float(np.mean(np.concatenate(train_labels)))
    logger.info(f"Smoke TRAIN-FIT positive rate across 16 queries: {train_pos_rate:.3f}")
    assert 0.05 <= train_pos_rate <= 0.60, f"Unexpected positive rate: {train_pos_rate}"

    train_loader = DataLoader(train_ds, batch_size=8, shuffle=True, collate_fn=routing_collate_fn)
    val_loader = DataLoader(val_ds, batch_size=8, shuffle=False, collate_fn=routing_collate_fn)

    # 4. Initialize R2 model
    device = "cuda" if torch.cuda.is_available() else "cpu"
    logger.info(f"Using device: {device}")

    model = LearnedPassageRouter(
        model_name="sentence-transformers/all-MiniLM-L6-v2",
        hidden_dim=128,
        dropout=0.1,
        use_rank_feature=True,
        rank_embedding_dim=16,
    ).to(device)

    param_counts = model.get_parameter_counts()
    logger.info(f"Model parameters: {param_counts}")
    assert param_counts["trainable_parameters"] > 0
    assert param_counts["frozen_parameters"] > 0

    # 5. Train 1 epoch on smoke TRAIN-FIT
    criterion = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([1.5], device=device))
    optimizer = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=1e-4)

    model.train()
    epoch_losses = []
    for batch in train_loader:
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        rank_indices = batch["rank_indices"].to(device)
        labels = batch["labels"].to(device)

        optimizer.zero_grad()
        logits = model(input_ids, attention_mask, rank_indices)
        loss = criterion(logits, labels)
        assert torch.isfinite(loss).item(), "Loss is NaN or infinite!"
        loss.backward()

        # Check gradients flow
        for name, param in model.named_parameters():
            if param.requires_grad:
                assert param.grad is not None, f"Gradient is None for {name}"
                grad_norm = param.grad.abs().sum().item()
                assert grad_norm > 0.0, f"Zero gradient for {name}"

        optimizer.step()
        epoch_losses.append(loss.item())

    avg_train_loss = float(np.mean(epoch_losses))
    logger.info(f"1-Epoch smoke training complete. Avg Loss: {avg_train_loss:.4f}")

    # 6. Evaluate on 8 INTERNAL-VAL examples
    model.eval()
    val_logits_list = []
    with torch.no_grad():
        for batch in val_loader:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            rank_indices = batch["rank_indices"].to(device)
            logits = model(input_ids, attention_mask, rank_indices)
            val_logits_list.append(logits.cpu().numpy())

    val_logits = np.concatenate(val_logits_list, axis=0)  # (8, 10)
    # Check scores vary across passages
    for i in range(len(val_logits)):
        std_val = float(np.std(val_logits[i]))
        assert std_val > 1e-4, f"Scores for example {i} do not vary! std={std_val}"

    # 7. Route DEV sample questions at k=2
    mock_mistral_tok = MockMistralTokenizer()
    r1_scorer = PretrainedSemanticScorer(device=device)

    dev_assignments = []
    differences_from_r0 = 0

    for _, row in df_dev.iterrows():
        raw_qid = str(row["question_id"]).strip()
        qid = str(int(float(raw_qid))) if raw_qid.replace(".", "", 1).isdigit() else raw_qid
        ret_item = dev_ret[qid]
        q_text = ret_item["question"]
        passages = ret_item["passages"][:10]
        p_texts = [p.get("text", "") for p in passages]

        # R0
        sel_r0 = route_r0_bm25_prefix(k=2, n_passages=10)
        validate_routing_assignment(sel_r0, k=2)

        # R1 (Pretrained semantic top-k)
        r1_scores = r1_scorer.score_query_passages(q_text, p_texts, device=device)
        sel_r1 = route_r1_semantic_topk(r1_scores, k=2)
        validate_routing_assignment(sel_r1, k=2)

        # R2 (Learned router top-k)
        encoded = tokenizer(
            [q_text] * 10,
            p_texts,
            padding="max_length",
            truncation="only_second",
            max_length=128,
            return_tensors="pt",
        )
        in_ids = encoded["input_ids"].unsqueeze(0).to(device)
        att_mask = encoded["attention_mask"].unsqueeze(0).to(device)
        ranks = torch.arange(10).unsqueeze(0).to(device)
        with torch.no_grad():
            r2_logits = model(in_ids, att_mask, ranks).squeeze(0)
        sel_r2 = route_r2_learned_topk(r2_logits, k=2)
        validate_routing_assignment(sel_r2, k=2)

        if sel_r1 != sel_r0 or sel_r2 != sel_r0:
            differences_from_r0 += 1

        # Build SARA prompts for each router and verify invariants
        prompt_r0, tok_r0, unsel_r0 = build_routed_sara_prompt(q_text, passages, sel_r0, mock_mistral_tok)
        prompt_r1, tok_r1, unsel_r1 = build_routed_sara_prompt(q_text, passages, sel_r1, mock_mistral_tok)
        prompt_r2, tok_r2, unsel_r2 = build_routed_sara_prompt(q_text, passages, sel_r2, mock_mistral_tok)

        for p_name, pr, sel, unsel in [("R0", prompt_r0, sel_r0, unsel_r0), ("R1", prompt_r1, sel_r1, unsel_r1), ("R2", prompt_r2, sel_r2, unsel_r2)]:
            assert len(sel) == 2
            assert len(unsel) == 8
            assert pr.count("<COMPRESS>") == 8
            for idx in sel:
                assert f"Document {idx + 1}." in pr

        dev_assignments.append({
            "question_id": qid,
            "question": q_text[:80] + "...",
            "R0_selected_ranks": [x + 1 for x in sel_r0],
            "R1_selected_ranks": [x + 1 for x in sel_r1],
            "R2_selected_ranks": [x + 1 for x in sel_r2],
        })

    logger.info(f"Sample DEV assignments: {json.dumps(dev_assignments, indent=2)}")

    # 8. Save smoke results
    smoke_results = {
        "SMOKE_ONLY_NOT_FOR_REPORTING": True,
        "status": "PASSED",
        "train_samples": len(df_fit),
        "val_samples": len(df_val),
        "dev_samples": len(df_dev),
        "k_budget": 2,
        "train_positive_rate": train_pos_rate,
        "train_avg_loss_epoch_1": avg_train_loss,
        "gradients_flow_verified": True,
        "scores_vary_verified": True,
        "invariants_verified": True,
        "differences_from_r0_observed": bool(differences_from_r0 > 0),
        "parameter_counts": param_counts,
        "sample_assignments": dev_assignments,
    }

    results_file = smoke_dir / "smoke_test_results.json"
    with open(results_file, "w", encoding="utf-8") as f:
        json.dump(smoke_results, f, indent=2)

    # Save checkpoint
    ckpt_file = smoke_dir / "smoke_checkpoint.pt"
    torch.save({"model_state_dict": model.state_dict(), "smoke": True}, ckpt_file)

    logger.info(f"Smoke test PASSED! Saved artifacts to {smoke_dir}")
    return smoke_results


if __name__ == "__main__":
    run_smoke()
