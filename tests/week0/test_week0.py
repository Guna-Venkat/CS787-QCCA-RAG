"""Unit tests for Week 0 Dataset and Evidence Diagnostics.

Verifies:
  - QASPERAdapter interfaces and schema conformance.
  - Query classification and observable characteristics.
  - Evidence metrics (redundancy, depth, overlap, matching).
  - Concentration metrics (entropy, Gini, N_eff, top-k shares).
  - Deterministic sampling reproducibility.
  - Generated output table schemas.
"""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
import pandas as pd
import pytest

from src.week0.dataset_adapter import DatasetAdapter, QASPERAdapter
from src.week0.dataset_analyzer import (
    calculate_evidence_depth,
    check_passage_matches_evidence,
    classify_wh_type,
    compute_evidence_redundancy,
    compute_lexical_overlap,
    extract_query_characteristics,
)
from src.week0.evidence_utilization import compute_concentration_metrics
from src.week0.layer_analyzer import select_controlled_sample

PROJECT_ROOT = Path(__file__).resolve().parents[2]


class DummyAdapter(DatasetAdapter):
    """Minimal in-memory adapter for unit testing."""

    def __init__(self, records: List[Dict[str, Any]]):
        self.records = records

    def load_split(self, split_name: str) -> List[Dict[str, Any]]:
        return self.records

    def get_query(self, record: Dict[str, Any]) -> str:
        return record.get("question", "")

    def get_document(self, record: Dict[str, Any]) -> Dict[str, Any]:
        return {
            "paper_id": record.get("paper_id", "p0"),
            "title": "Test Title",
            "section_names": ["Intro", "Methods", "Results"],
            "paragraphs_by_section": [["Intro para 1"], ["Methods para 1"], ["Results para 1"]],
            "all_paragraphs": ["Intro para 1", "Methods para 1", "Results para 1"],
        }

    def get_gold_answer(self, record: Dict[str, Any]) -> List[str]:
        return record.get("answer", ["test answer"])

    def get_gold_evidence(self, record: Dict[str, Any]) -> List[Dict[str, Any]]:
        return [{
            "evidence_paragraphs": ["Methods para 1"],
            "highlighted_spans": ["Methods para 1"],
            "extractive_spans": ["test answer"],
            "unanswerable": False,
            "yes_no": None,
        }]

    def get_metadata(self, record: Dict[str, Any]) -> Dict[str, Any]:
        return {
            "question_id": record.get("id", "q0"),
            "paper_id": record.get("paper_id", "p0"),
            "question_type": record.get("question_type", "extractive"),
        }


def test_classify_wh_type():
    assert classify_wh_type("What datasets are used?") == "what"
    assert classify_wh_type("How does the model perform?") == "how"
    assert classify_wh_type("How many parameters does it have?") == "how_many"
    assert classify_wh_type("Why is accuracy lower?") == "why"
    assert classify_wh_type("Which baseline was compared?") == "which"
    assert classify_wh_type("Is the method effective?") == "boolean"
    assert classify_wh_type("Do they report F1?") == "boolean"
    assert classify_wh_type("Random statement without prefix") == "other"


def test_extract_query_characteristics():
    q = "Does Model A compare favorably to Model B across 10 trials and 5 folds?"
    feats = extract_query_characteristics(q)
    assert feats["query_word_count"] > 5
    assert feats["is_numerical"] is True
    assert feats["is_comparison"] is True
    assert feats["is_multipart"] is True
    assert 0.0 < feats["query_ttr"] <= 1.0


def test_compute_lexical_overlap():
    q = "What datasets are used to evaluate the model?"
    ev = "The model was evaluated on WN18 and FB15k benchmark datasets."
    jaccard, coverage = compute_lexical_overlap(q, ev)
    assert 0.0 < jaccard < 1.0
    assert 0.0 < coverage <= 1.0
    # Empty case
    j_zero, c_zero = compute_lexical_overlap("", "some text")
    assert j_zero == 0.0 and c_zero == 0.0


def test_compute_evidence_redundancy():
    p1 = "Knowledge base completion aims to predict missing links."
    p2 = "Predicting missing links in knowledge bases is known as knowledge completion."
    p3 = "Quantum computing uses qubits and superpositions."

    high_red = compute_evidence_redundancy([p1, p2])
    low_red = compute_evidence_redundancy([p1, p3])
    assert high_red > low_red
    assert compute_evidence_redundancy([p1]) == 0.0
    assert compute_evidence_redundancy([]) == 0.0


def test_calculate_evidence_depth():
    all_paras = ["Title & Intro", "Related Work", "Methodology", "Experiments", "Discussion", "Conclusion"]
    ev_early = ["Title & Intro"]
    ev_mid = ["Methodology"]
    ev_late = ["Conclusion"]

    depth_early = calculate_evidence_depth(ev_early, all_paras)
    depth_mid = calculate_evidence_depth(ev_mid, all_paras)
    depth_late = calculate_evidence_depth(ev_late, all_paras)

    assert depth_early[0] < depth_mid[0] < depth_late[0]
    assert 0.0 <= depth_early[0] <= 0.2
    assert 0.8 <= depth_late[0] <= 1.0


def test_check_passage_matches_evidence():
    ev_paras = ["Table 1 shows the link prediction performance on WN18 and FB15k."]
    hl_spans = ["link prediction performance on WN18"]

    matching_p = "In Table 1, we report link prediction performance on WN18 and FB15k."
    distractor_p = "The training optimizer was Adam with learning rate 0.001."

    assert check_passage_matches_evidence(matching_p, ev_paras, hl_spans) is True
    assert check_passage_matches_evidence(distractor_p, ev_paras, hl_spans) is False


def test_compute_concentration_metrics():
    # 1. Perfectly uniform distribution (10 passages, 0.1 each)
    uniform = [0.1] * 10
    m_uni = compute_concentration_metrics(uniform)
    assert math.isclose(m_uni["entropy_norm"], 1.0, abs_tol=1e-3)
    assert math.isclose(m_uni["effective_n_passages"], 10.0, abs_tol=1e-2)
    assert math.isclose(m_uni["gini_coefficient"], 0.0, abs_tol=1e-3)
    assert math.isclose(m_uni["top1_share"], 0.1, abs_tol=1e-3)
    assert math.isclose(m_uni["top3_share"], 0.3, abs_tol=1e-3)

    # 2. Completely concentrated distribution (one passage has all mass)
    concentrated = [1.0] + [0.0] * 9
    m_conc = compute_concentration_metrics(concentrated)
    assert math.isclose(m_conc["entropy_norm"], 0.0, abs_tol=1e-3)
    assert math.isclose(m_conc["effective_n_passages"], 1.0, abs_tol=1e-2)
    assert m_conc["gini_coefficient"] > 0.8
    assert math.isclose(m_conc["top1_share"], 1.0, abs_tol=1e-3)
    assert math.isclose(m_conc["top3_share"], 1.0, abs_tol=1e-3)


def test_deterministic_sampling():
    records = [
        {"id": f"q_{i}", "paper_id": f"p_{i % 5}", "question_type": "extractive" if i % 2 == 0 else "yes_no"}
        for i in range(50)
    ]
    adapter = DummyAdapter(records)
    sample1 = select_controlled_sample(records, adapter, sample_size=10, seed=42)
    sample2 = select_controlled_sample(records, adapter, sample_size=10, seed=42)
    sample3 = select_controlled_sample(records, adapter, sample_size=10, seed=123)

    ids1 = [r["id"] for r in sample1]
    ids2 = [r["id"] for r in sample2]
    ids3 = [r["id"] for r in sample3]

    assert ids1 == ids2
    assert ids1 != ids3
    assert len(sample1) == 10


def test_qasper_adapter_real_load():
    train_path = PROJECT_ROOT / "Baselines/SARA-main/data/reformatted/QASPER_train_split.jsonl"
    dev_path = PROJECT_ROOT / "Baselines/SARA-main/data/reformatted/QASPER_dev.jsonl"
    arrow_path = Path.home() / ".cache/huggingface/datasets/allenai___qasper/qasper/0.3.0/fdc9d8214fbab5dd782958601db4d678e6934a54/qasper-train.arrow"
    manifest_path = PROJECT_ROOT / "Baselines/SARA-main/data/manifests/qasper_split.json"
    dev_ret_path = PROJECT_ROOT / "results/week2/raw/dev_retrieval.jsonl"

    if not (train_path.exists() and dev_path.exists() and dev_ret_path.exists()):
        pytest.skip("QASPER files not found in standard paths.")

    adapter = QASPERAdapter(
        train_split_path=train_path,
        dev_split_path=dev_path,
        raw_arrow_train_path=arrow_path if arrow_path.exists() else None,
        manifest_path=manifest_path if manifest_path.exists() else None,
        dev_retrieval_path=dev_ret_path,
    )

    dev_records = adapter.load_split("dev")
    assert len(dev_records) == 231

    r0 = dev_records[0]
    assert len(adapter.get_query(r0)) > 0
    assert len(adapter.get_gold_answer(r0)) > 0
    ret_passages = adapter.get_retrieved_passages(r0)
    assert len(ret_passages) == 10


def test_generated_output_schemas():
    summary_path = PROJECT_ROOT / "results/week0/tables/dataset_summary.csv"
    retrieval_path = PROJECT_ROOT / "results/week0/tables/retrieval_statistics.csv"
    corr_path = PROJECT_ROOT / "results/week0/tables/signal_correlations.csv"

    if summary_path.exists():
        df_sum = pd.read_csv(summary_path)
        expected_cols = {"split", "num_papers", "num_questions", "doc_length_words_mean", "evidence_paras_per_question_mean"}
        assert expected_cols.issubset(set(df_sum.columns))

    if retrieval_path.exists():
        df_ret = pd.read_csv(retrieval_path)
        expected_cols = {"question_id", "paper_id", "recall_any_at_2", "recall_any_at_4", "recall_any_at_6", "recall_any_at_10", "delta_12"}
        assert expected_cols.issubset(set(df_ret.columns))

    if corr_path.exists():
        df_corr = pd.read_csv(corr_path)
        expected_cols = {"signal_1", "signal_2", "pearson_r", "spearman_rho"}
        assert expected_cols.issubset(set(df_corr.columns))


def test_synthetic_attention_passage_extraction():
    """Unit test for attention-based passage utilization proxy with synthetic attention tensor."""
    import torch
    from src.week0.evidence_utilization import extract_causal_passage_attention

    # Sequence layout:
    # 0..19: instruction prefix
    # 20..39: passage 0 (len=20)
    # 40..59: passage 1 (len=20)
    # 60..79: passage 2 (len=20)
    # 80..99: query (len=20)
    seq_len = 100
    num_heads = 4
    # Create lower-triangular causal attention matrix for each head
    raw_weights = torch.ones(num_heads, seq_len, seq_len)
    causal_mask = torch.tril(torch.ones(seq_len, seq_len))
    # Apply causal mask and normalize rows to sum to 1
    masked = raw_weights * causal_mask
    row_sums = masked.sum(dim=-1, keepdim=True)
    attn = masked / row_sums

    attention_tensors = (attn.unsqueeze(0),)  # batch size 1
    spans = {
        "passage_0": (20, 40),
        "passage_1": (40, 60),
        "passage_2": (60, 80),
        "query": (80, 100),
    }

    layer_mass = extract_causal_passage_attention(
        attention_tensors=attention_tensors,
        spans=spans,
        num_passages=3,
        selected_layers=[0],
    )

    assert 0 in layer_mass
    p_masses = layer_mass[0]
    assert len(p_masses) == 3
    # Check that each passage receives a non-negative normalized mass
    for p_idx in range(3):
        assert p_masses[p_idx] > 0.0
    # Query tokens attend back to passages, which respects the causal mask


def test_concentration_signed_ablation_handling():
    """Verify that compute_concentration_metrics handles signed/negative ablation drops."""
    # Suppose removing passage 0 and 2 helps (negative drop), while passage 1 hurts (positive drop)
    signed_deltas = [-0.5, 1.2, -0.3, 0.8, 0.0]
    
    # 1. Positive contribution mode (useful evidence allocation)
    res_pos = compute_concentration_metrics(signed_deltas, mode="positive")
    assert res_pos["effective_n_passages"] > 0.0
    assert 0.0 <= res_pos["entropy_norm"] <= 1.0
    assert 0.0 <= res_pos["gini_coefficient"] <= 1.0
    # Top-1 should be 1.2 / (1.2 + 0.8) = 1.2 / 2.0 = 0.60
    assert math.isclose(res_pos["top1_share"], 0.60, abs_tol=1e-3)
    # Top-3 should be 1.0 (since only two items are positive)
    assert math.isclose(res_pos["top3_share"], 1.00, abs_tol=1e-3)

    # 2. Absolute sensitivity mode (perturbation sensitivity)
    res_abs = compute_concentration_metrics(signed_deltas, mode="absolute")
    assert res_abs["effective_n_passages"] > 0.0
    # Sum of absolute values = 0.5 + 1.2 + 0.3 + 0.8 + 0.0 = 2.8
    # Top-1 share = 1.2 / 2.8 = 0.4286
    assert math.isclose(res_abs["top1_share"], 1.2 / 2.8, abs_tol=1e-3)


def test_concentration_all_negative_edge_case():
    """Verify degenerate edge-case when all ablation drops are non-positive."""
    all_neg = [-0.4, -0.1, -0.8, 0.0]
    res = compute_concentration_metrics(all_neg, mode="positive")
    # No positive evidence was contributed; metrics must indicate zero concentration rather than crashing
    assert res["effective_n_passages"] == 0.0
    assert res["gini_coefficient"] == 0.0
    assert res["entropy_norm"] == 0.0
    assert res["top1_share"] == 0.0
    assert res["top3_share"] == 0.0


def test_production_pipeline_disallows_mock_empirical_writes(tmp_path):
    """Verify that the production pipeline NEVER silently writes mock Part B/C data."""
    import json
    from scripts.week0.run_week0_analysis import run_week0_pipeline

    cfg_path = PROJECT_ROOT / "configs/week0/week0.yaml"
    # Execute pipeline with run_gpu_models=False, skip_part_a=True for fast unit testing
    run_week0_pipeline(config_path=cfg_path, run_gpu_models=False, skip_part_a=True)

    status_file = PROJECT_ROOT / "results/week0/gpu_status.json"
    assert status_file.exists()
    with open(status_file, "r") as f:
        status_data = json.load(f)
    assert status_data.get("part_b_status") == "PENDING_REAL_GPU_EXECUTION"
    assert status_data.get("part_c_status") == "PENDING_REAL_GPU_EXECUTION"


def test_gpu_pilot_artifacts_schema_and_validity():
    """Verify that all Week 0 GPU pilot artifacts exist, adhere to strict schemas, and contain real values."""
    pilot_dir = PROJECT_ROOT / "results/week0/gpu_pilot"
    assert pilot_dir.exists()

    # 1. Pilot Questions CSV
    pq_path = pilot_dir / "pilot_questions.csv"
    assert pq_path.exists()
    df_pq = pd.read_csv(pq_path)
    assert len(df_pq) == 10
    assert set(df_pq.columns) == {
        "question_id", "paper_id", "question", "evidence_count",
        "retrieval_any@10", "retrieval_all@10", "selection_reason"
    }

    # 2. Execution Metadata
    meta_path = pilot_dir / "execution_metadata.json"
    assert meta_path.exists()
    with open(meta_path, "r") as f:
        meta = json.load(f)
    assert meta["num_queries"] == 10
    assert meta["num_forward_passes"] == 110
    assert meta["truncation"] is False
    assert meta["model_parameter_count"] == 7282126848
    assert meta["dtype"] == "torch.bfloat16"

    # 3. Layer Results CSV
    lr_path = pilot_dir / "layer_results.csv"
    assert lr_path.exists()
    df_lr = pd.read_csv(lr_path)
    assert len(df_lr) == 10 * 10 * 6  # 10 queries * 10 passages * 6 layers = 600 rows
    assert not df_lr["cosine_sim_with_query"].isna().any()
    assert not df_lr["query_drift"].isna().any()

    # 4. Attention Results CSV
    ar_path = pilot_dir / "attention_results.csv"
    assert ar_path.exists()
    df_ar = pd.read_csv(ar_path)
    assert len(df_ar) == 10 * 10 * 6 * 2  # 10 queries * 10 passages * 6 layers * 2 attention types = 1200 rows
    assert set(df_ar["attention_type"].unique()) == {"query_to_passage", "answer_to_passage"}
    assert not df_ar["length_normalized_attention"].isna().any()

    # 5. Ablation Results CSV
    abl_path = pilot_dir / "ablation_results.csv"
    assert abl_path.exists()
    df_abl = pd.read_csv(abl_path)
    assert len(df_abl) == 100  # 10 queries * 10 passages
    assert not df_abl["delta_mean_logprob"].isna().any()
    # Signed Delta must preserve negative values (distractor interference)
    assert (df_abl["delta_mean_logprob"] < 0).any()

    # 6. Pilot Report Markdown
    rep_path = pilot_dir / "pilot_report.md"
    assert rep_path.exists()
    with open(rep_path, "r") as f:
        rep_text = f.read()
    assert "## 1. Execution Verification" in rep_text
    assert "## 9. Decision on Scaling" in rep_text
    assert "Recommendation: **C. SCALE TO 40**" in rep_text


def test_intervention_validation_artifacts():
    """Verify that all length/position-controlled intervention validation artifacts exist and conform to schemas."""
    val_dir = PROJECT_ROOT / "results/week0/gpu_pilot/intervention_validation"
    assert val_dir.exists()

    # 1. Evidence Match Audit
    audit_path = val_dir / "evidence_match_audit.csv"
    assert audit_path.exists()
    df_audit = pd.read_csv(audit_path)
    assert len(df_audit) == 100
    assert "is_labeled_match" in df_audit.columns
    assert "is_duplicate_chunk_of_same_gold_para" in df_audit.columns

    # 2. Replacement Results
    repl_path = val_dir / "replacement_results.csv"
    assert repl_path.exists()
    df_repl = pd.read_csv(repl_path)
    assert len(df_repl) == 100
    assert "delta_replace_mean_logprob" in df_repl.columns
    assert "abs_token_diff" in df_repl.columns
    assert df_repl["abs_token_diff"].max() <= 5.0

    # 3. Removal vs Replacement Comparison
    comp_path = val_dir / "removal_vs_replacement.csv"
    assert comp_path.exists()
    df_comp = pd.read_csv(comp_path)
    assert len(df_comp) == 100
    assert "delta_remove_mean_logprob" in df_comp.columns
    assert "delta_replace_mean_logprob" in df_comp.columns
    assert "signs_agree" in df_comp.columns

    # 4. Figures
    fig_dir = val_dir / "figures"
    assert fig_dir.exists()
    assert (fig_dir / "length_matching_distribution.png").exists()
    assert (fig_dir / "removal_vs_replacement_scatter.png").exists()
    assert (fig_dir / "delta_replace_by_group.png").exists()
    assert (fig_dir / "per_query_removal_vs_replacement.png").exists()

    # 5. Validation Report
    val_rep_path = val_dir / "validation_report.md"
    assert val_rep_path.exists()
    with open(val_rep_path, "r") as f:
        rep_text = f.read()
    assert "## 2. Token-Length Matching Quality Audit" in rep_text
    assert "## 3. Annotated-Evidence Matching Duplication Audit" in rep_text
    assert "## 7. Final Classification & Recommendation" in rep_text
    assert "Classification: **B. Both semantic and length/position effects appear important**" in rep_text


def test_final40_artifacts_schema_and_validity():
    """Verify that all Final Week 0 study artifacts exist, adhere to strict schemas, and contain real values."""
    final40_dir = PROJECT_ROOT / "results/week0/final40"
    assert final40_dir.exists()

    # 1. Sample Manifest
    sm_path = final40_dir / "sample_manifest.csv"
    assert sm_path.exists()
    df_sm = pd.read_csv(sm_path)
    assert len(df_sm) == 40
    assert df_sm["paper_id"].nunique() == 40
    assert "selection_reason" in df_sm.columns
    assert set(df_sm["stratum"].unique()) == {
        "single_evidence_1para", "dual_evidence_2para", "multi_evidence_3plus",
        "incomplete_bm25_retrieval", "failed_bm25_retrieval"
    }

    # 2. Execution Metadata
    meta_path = final40_dir / "execution_metadata.json"
    assert meta_path.exists()
    with open(meta_path, "r") as f:
        meta = json.load(f)
    assert meta["total_queries"] == 40
    assert meta["distinct_papers"] == 40
    assert meta["forward_pass_count"] == 840
    assert meta["no_mocks_verified"] is True
    assert meta["no_synthetic_values_verified"] is True
    assert meta["gpu"] == "NVIDIA TITAN RTX"

    # 3. Layer Results CSV
    lr_path = final40_dir / "layer_results.csv"
    assert lr_path.exists()
    df_lr = pd.read_csv(lr_path)
    assert len(df_lr) == 40 * 10 * 6  # 40 queries * 10 passages * 6 layers = 2400 rows
    assert not df_lr["cosine_sim_with_query"].isna().any()

    # 4. Attention Results CSV
    ar_path = final40_dir / "attention_results.csv"
    assert ar_path.exists()
    df_ar = pd.read_csv(ar_path)
    assert len(df_ar) == 40 * 10 * 6
    assert not df_ar["query_to_passage_norm"].isna().any()
    assert not df_ar["answer_to_passage_norm"].isna().any()

    # 5. Removal Results CSV
    rem_path = final40_dir / "removal_results.csv"
    assert rem_path.exists()
    df_rem = pd.read_csv(rem_path)
    assert len(df_rem) == 400
    assert not df_rem["delta_remove"].isna().any()

    # 6. Replacement Results CSV
    repl_path = final40_dir / "replacement_results.csv"
    assert repl_path.exists()
    df_repl = pd.read_csv(repl_path)
    assert len(df_repl) == 400
    assert not df_repl["delta_replace"].isna().any()
    assert df_repl["token_length_diff"].abs().mean() <= 2.0
    assert df_repl["token_length_diff"].abs().max() <= 10

    # 7. Evidence Match Results CSV
    em_path = final40_dir / "evidence_match_results.csv"
    assert em_path.exists()
    df_em = pd.read_csv(em_path)
    assert len(df_em) == 400
    assert set(df_em["match_category"].unique()).issubset({"STRONG", "PARTIAL", "WEAK", "NONE"})

    # 8. Question-Level Features CSV
    qf_path = final40_dir / "question_level_features.csv"
    assert qf_path.exists()
    df_qf = pd.read_csv(qf_path)
    assert len(df_qf) == 40
    assert "positive_delta_replace_mass" in df_qf.columns
    assert "bm25_entropy" in df_qf.columns

    # 9. Week 3 Bridge CSV
    wb_path = final40_dir / "week3_bridge.csv"
    assert wb_path.exists()
    df_wb = pd.read_csv(wb_path)
    assert len(df_wb) == 40
    assert "selected_k_epsilon" in df_wb.columns
    assert not df_wb["selected_k_epsilon"].isna().all()

    # 10. Publication Figures
    fig_dir = final40_dir / "figures"
    assert fig_dir.exists()
    required_figs = [
        "evidence_count_distribution.png",
        "bm25_recall_depth.png",
        "delta_replace_by_evidence_category.png",
        "answer_attention_by_evidence_category.png",
        "per_query_intervention_heterogeneity.png",
        "week0_vs_week3_oracle.png",
    ]
    for fig_name in required_figs:
        assert (fig_dir / fig_name).exists(), f"Missing figure: {fig_name}"

    # 11. Final Report Markdown
    rep_path = final40_dir / "final40_report.md"
    assert rep_path.exists()
    with open(rep_path, "r") as f:
        rep_text = f.read()
    assert "WEEK 0 STATUS: FROZEN" in rep_text
    assert "PRIMARY VERIFIED" in rep_text
    assert "EXPLORATORY" in rep_text
    assert "NOT SUPPORTED" in rep_text





