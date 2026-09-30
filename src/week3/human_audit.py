"""Blinded 50-Question Human Audit Module for Week 3.

Implements the official experimental audit protocol:
1. Deterministic stratified sampling of 50 development questions across oracle strata.
2. Complete blinding mechanism: anonymizes k in {2, 5, 10} into System A, B, C.
3. Secret blinding key stored in results/week3/human_audit/blinding_map.json.
4. Generates ready-to-score evaluator artifact: results/week3/human_audit/human_scoring_template.csv
   with columns:
     audit_id, question_id, anonymous_system_id, question_text, generated_answer, score_0_1_2, optional_notes
5. STRICT GUARDRAIL: Does NOT generate or report synthetic human scores.
   If human_scores.csv is absent or contains unfilled scores, the audit is formally classified as
   PENDING_MANUAL_EVALUATION without computing premature or invalid correlations.
"""

import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from scipy import stats

from src.week3.oracle_analysis import load_raw_dev_matrix

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

AUDIT_K_VALUES = [2, 5, 10]
SYSTEM_LABELS = ["System A", "System B", "System C"]


def load_question_texts(retrieval_path: str = "results/week2/raw/dev_retrieval.jsonl") -> Dict[str, str]:
    """Load question_id -> question text mapping."""
    q_map = {}
    with open(retrieval_path, "r", encoding="utf-8") as f:
        for line in f:
            rec = json.loads(line)
            q_map[str(rec["question_id"])] = rec.get("question", "")
    return q_map


def sample_stratified_audit_questions(
    oracle_df: pd.DataFrame,
    n_sample: int = 50,
    seed: int = 42
) -> pd.DataFrame:
    """Sample 50 questions stratified by best_k_qual to ensure balanced representation."""
    rng = np.random.RandomState(seed)
    
    strata = sorted(oracle_df["best_k_qual"].unique())
    n_strata = len(strata)
    base_per_stratum = n_sample // n_strata
    remainder = n_sample % n_strata
    
    sampled_dfs = []
    for i, s in enumerate(strata):
        sub = oracle_df[oracle_df["best_k_qual"] == s]
        k_count = base_per_stratum + (1 if i < remainder else 0)
        replace = len(sub) < k_count
        sampled = sub.sample(n=k_count, replace=replace, random_state=rng)
        sampled_dfs.append(sampled)
        
    df_sample = pd.concat(sampled_dfs, ignore_index=True)
    df_sample = df_sample.sample(frac=1.0, random_state=rng).reset_index(drop=True)
    df_sample["question_id"] = df_sample["question_id"].astype(str)
    df_sample["sampling_stratum"] = "best_k_" + df_sample["best_k_qual"].astype(str)
    return df_sample[["question_id", "paper_id", "sampling_stratum", "best_k_qual"]]


def build_blinded_audit_dataset(
    df_sample: pd.DataFrame,
    df_raw: pd.DataFrame,
    q_texts: Dict[str, str],
    seed: int = 42
) -> Tuple[pd.DataFrame, Dict[str, Dict[str, int]], pd.DataFrame]:
    """Extract predictions for k in {2, 5, 10} and randomize system order per question.
    
    Returns:
        df_eval_template: Evaluator-facing ready-to-score table (strictly hides k)
        blinding_map: Dict mapping question_id -> {System A: k, System B: k, System C: k}
        df_ground_truth: Internal audit ground truth table
    """
    rng = np.random.RandomState(seed)
    sample_q_ids = set(df_sample["question_id"].astype(str))
    
    df_raw = df_raw.copy()
    df_raw["question_id"] = df_raw["question_id"].astype(str)
    
    df_sub = df_raw[
        (df_raw["question_id"].isin(sample_q_ids)) &
        (df_raw["k"].isin(AUDIT_K_VALUES))
    ].copy()
    
    blinding_map = {}
    eval_template_rows = []
    ground_truth_rows = []
    
    audit_id_counter = 1
    grouped = df_sub.groupby("question_id")
    for q_id, group in grouped:
        q_id_str = str(q_id)
        present_k = sorted(group["k"].tolist())
        if present_k != AUDIT_K_VALUES:
            logger.warning("Question %s has incomplete audit k values: %s", q_id_str, present_k)
            continue
            
        paper_id = str(group["paper_id"].iloc[0])
        q_text = q_texts.get(q_id_str, "")
        
        # Random permutation of [2, 5, 10]
        k_perm = rng.permutation(AUDIT_K_VALUES).tolist()
        q_blinding = {label: int(k_val) for label, k_val in zip(SYSTEM_LABELS, k_perm)}
        blinding_map[q_id_str] = q_blinding
        
        preds_by_k = group.set_index("k")["prediction"].to_dict()
        f1_by_k = group.set_index("k")["token_f1"].to_dict()
        
        for label, k_val in q_blinding.items():
            pred_text = str(preds_by_k.get(k_val, ""))
            
            # Evaluator-facing row (no k exposed!)
            eval_template_rows.append({
                "audit_id": audit_id_counter,
                "question_id": q_id_str,
                "anonymous_system_id": label,
                "question_text": q_text,
                "generated_answer": pred_text,
                "score_0_1_2": "",  # Unfilled for human evaluator
                "optional_notes": ""
            })
            
            # Internal secret key row
            ground_truth_rows.append({
                "audit_id": audit_id_counter,
                "question_id": q_id_str,
                "paper_id": paper_id,
                "anonymous_system_id": label,
                "k": k_val,
                "token_f1": float(f1_by_k.get(k_val, 0.0)),
                "prediction": pred_text
            })
            audit_id_counter += 1
            
    df_eval_template = pd.DataFrame(eval_template_rows)
    df_gt = pd.DataFrame(ground_truth_rows)
    return df_eval_template, blinding_map, df_gt


def evaluate_human_scores(
    scores_csv_path: str,
    blinding_map: Dict[str, Dict[str, int]],
    df_gt: pd.DataFrame
) -> Tuple[str, Optional[pd.DataFrame], Optional[Dict[str, Any]]]:
    """Verify if actual human scores are present and calculate metrics if genuine.
    
    If scores_csv_path does not exist or has empty/non-numeric scores,
    returns status = 'PENDING_MANUAL_EVALUATION'.
    """
    path = Path(scores_csv_path)
    if not path.exists():
        logger.info("Human scores file %s does not exist; marking as PENDING_MANUAL_EVALUATION.", scores_csv_path)
        return "PENDING_MANUAL_EVALUATION", None, None
        
    df_scores = pd.read_csv(path)
    if "score_0_1_2" not in df_scores.columns or df_scores["score_0_1_2"].isna().all() or (df_scores["score_0_1_2"] == "").all():
        logger.info("Human scores file %s is unpopulated; marking as PENDING_MANUAL_EVALUATION.", scores_csv_path)
        return "PENDING_MANUAL_EVALUATION", None, None
        
    # Check if scores are valid integers in {0, 1, 2}
    try:
        valid_scores = df_scores["score_0_1_2"].dropna().astype(int)
        if len(valid_scores) < len(df_gt):
            logger.info("Only partial scores (%d/%d); marking as PENDING_MANUAL_EVALUATION.", len(valid_scores), len(df_gt))
            return "PENDING_MANUAL_EVALUATION", None, None
    except Exception:
        return "PENDING_MANUAL_EVALUATION", None, None

    # If genuine human scores exist:
    df_merged = df_scores.merge(df_gt[["audit_id", "k", "token_f1"]], on="audit_id")
    
    k_stats = []
    for k_val in AUDIT_K_VALUES:
        sub = df_merged[df_merged["k"] == k_val]
        scores = sub["score_0_1_2"].astype(int).to_numpy()
        f1_vals = sub["token_f1"].to_numpy()
        k_stats.append({
            "k": k_val,
            "n_judgments": len(scores),
            "mean_human_score": round(float(np.mean(scores)), 3),
            "median_human_score": int(np.median(scores)),
            "pct_incorrect_0": round(float(np.mean(scores == 0) * 100.0), 1),
            "pct_partial_1": round(float(np.mean(scores == 1) * 100.0), 1),
            "pct_correct_2": round(float(np.mean(scores == 2) * 100.0), 1),
            "mean_token_f1": round(float(np.mean(f1_vals)), 4)
        })
    df_k_summary = pd.DataFrame(k_stats)
    
    sp_r, sp_p = stats.spearmanr(df_merged["score_0_1_2"], df_merged["token_f1"])
    summary = {
        "status": "COMPLETED",
        "n_questions": int(df_merged["question_id"].nunique()),
        "total_judgments": len(df_merged),
        "spearman_rho_human_vs_f1": round(float(sp_r), 4),
        "spearman_p_value": round(float(sp_p), 4),
        "per_k_summary": df_k_summary.to_dict(orient="records")
    }
    return "COMPLETED", df_k_summary, summary


def generate_human_audit_report_md(
    status: str,
    df_k_summary: Optional[pd.DataFrame] = None,
    summary: Optional[Dict[str, Any]] = None
) -> str:
    """Generate Markdown report for the blinded human audit."""
    if status == "PENDING_MANUAL_EVALUATION":
        return """# Blinded Human Audit Report (N=50 Development Questions)

**Benchmark:** QASPER Development Split  
**Status:** **PENDING_MANUAL_EVALUATION**  
**Automatic Oracle Analysis:** **COMPLETE**  
**Human Evaluation Artifact:** `results/week3/human_audit/human_scoring_template.csv`  

---

## 1. Audit Protocol & Design

To validate whether automatic Token F1 reliably correlates with qualitative answer correctness across evidence budgets, a blinded triple-condition human audit has been prepared:
- **Questions:** 50 development questions selected via deterministic stratified sampling across optimal budget strata ($k^*_{\\text{qual}}$).
- **Conditions per Question:** $k \\in \\{2, 5, 10\\}$ (150 evaluations total).
- **Blinding Standard:** The presentation order of $k=2$, $k=5$, and $k=10$ is randomized per question under anonymous identifiers (**System A**, **System B**, **System C**). The true budget mapping is stored strictly in `results/week3/human_audit/blinding_map.json` and hidden from the evaluation sheet.
- **Evaluator Rubric:**
  - `0 = Incorrect`: The answer is factually wrong, ungrounded, or fails to address the question.
  - `1 = Partially correct`: The answer contains relevant factual information but is incomplete, imprecise, or contains minor inaccuracies.
  - `2 = Fully correct`: The answer fully, accurately, and concisely answers the scientific question.

---

## 2. Evaluation Status & Scientific Guardrail

- **Current Status:** The evaluation template has been generated with all 150 anonymous answers and question texts ready for manual human evaluation.
- **Scientific Guardrail:** In accordance with project research integrity guidelines, **no synthetic, heuristic, or LLM-judged scores are substituted**. The correlation with Token F1 will be computed and reported only when actual human scores are manually entered into `human_scoring_template.csv`.
- **Downstream Impact:** Automatic oracle and epsilon analyses in Week 3 are fully valid and complete on the frozen Week-2 matrix. The human evaluation remains a secondary qualitative verification.
"""
    else:
        table_md = df_k_summary.to_markdown(index=False)
        return f"""# Blinded Human Audit Report (N=50 Development Questions)

**Benchmark:** QASPER Development Split  
**Status:** **COMPLETED**  
**Evaluated Conditions:** $k \\in \\{{2, 5, 10\\}}$  
**Sample Size:** 50 questions $\\times$ 3 conditions = 150 judgments  

---

## 1. Score Summary by Condition

{table_md}

---

## 2. Correlation with Token F1

- **Spearman Rank Correlation ($\\rho$):** **{summary['spearman_rho_human_vs_f1']:.4f}** ($p = {summary['spearman_p_value']:.4e}$)
"""


def run_human_audit(
    raw_matrix_path: str,
    oracle_csv_path: str,
    retrieval_path: str,
    output_dir: str,
    n_sample: int = 50,
    seed: int = 42
) -> Tuple[pd.DataFrame, str, Optional[pd.DataFrame], Dict[str, Any]]:
    """Execute complete human audit sampling, blinding, template creation, and evaluation check."""
    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)
    
    df_raw = load_raw_dev_matrix(raw_matrix_path)
    oracle_df = pd.read_csv(oracle_csv_path)
    q_texts = load_question_texts(retrieval_path)
    
    # 1. Stratified sampling
    df_sample = sample_stratified_audit_questions(oracle_df, n_sample=n_sample, seed=seed)
    sample_csv = out_path / "audit_sample.csv"
    df_sample.to_csv(sample_csv, index=False)
    logger.info("Saved audit sample manifest to %s", sample_csv)
    
    # 2. Build blinded evaluation dataset and secret blinding map
    df_eval_template, blinding_map, df_gt = build_blinded_audit_dataset(df_sample, df_raw, q_texts, seed=seed)
    
    blinding_json = out_path / "blinding_map.json"
    with open(blinding_json, "w", encoding="utf-8") as f:
        json.dump(blinding_map, f, indent=2)
    logger.info("Saved secret blinding map to %s", blinding_json)
    
    # 3. Save clean evaluator-facing ready-to-score template
    template_csv = out_path / "human_scoring_template.csv"
    df_eval_template.to_csv(template_csv, index=False)
    logger.info("Saved ready-to-score template to %s (150 items awaiting evaluation)", template_csv)
    
    # 4. Remove any synthetic human_scores.csv if it was auto-generated
    scores_csv = out_path / "human_scores.csv"
    if scores_csv.exists():
        # Check if it was synthetic
        content = scores_csv.read_text()
        if "automated_rubric_baseline_judgment" in content:
            scores_csv.unlink()
            logger.info("Removed synthetic human_scores.csv artifact.")
            
    # 5. Evaluate if genuine human scores exist
    status, df_k_summary, summary_dict = evaluate_human_scores(str(scores_csv), blinding_map, df_gt)
    
    if summary_dict is None:
        summary_dict = {
            "status": "PENDING_MANUAL_EVALUATION",
            "n_questions": n_sample,
            "total_judgments": len(df_eval_template),
            "tested_k_values": AUDIT_K_VALUES,
            "spearman_rho_human_vs_f1": None,
            "spearman_p_value": None,
            "message": "Blinded evaluation sheet generated at human_scoring_template.csv. Human evaluation pending manual scoring."
        }
        
    report_md = generate_human_audit_report_md(status, df_k_summary, summary_dict)
    report_file = out_path / "human_audit_report.md"
    with open(report_file, "w", encoding="utf-8") as f:
        f.write(report_md)
    logger.info("Saved human audit markdown report to %s (status: %s)", report_file, status)
    
    return df_sample, status, df_k_summary, summary_dict


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Run Human Audit Pipeline")
    parser.add_argument("--raw_matrix", default="results/week2/raw/dev_fixed_k_matrix.jsonl")
    parser.add_argument("--oracle", default="results/week3/processed/dev_quality_oracle.csv")
    parser.add_argument("--retrieval", default="results/week2/raw/dev_retrieval.jsonl")
    parser.add_argument("--output_dir", default="results/week3/human_audit")
    args = parser.parse_args()
    
    _, status, df_k_sum, summary = run_human_audit(args.raw_matrix, args.oracle, args.retrieval, args.output_dir)
    print(f"\nHuman Audit Status: {status}")
    print(f"Summary: {summary['message']}")
