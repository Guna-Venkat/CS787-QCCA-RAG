import os
import json
import pandas as pd
import numpy as np

files = {
    "validation_checkpoint": "results/week2/processed/week2_validation_checkpoint.json",
    "dev_fixed_k_summary": "results/week2/processed/dev_fixed_k_summary.csv",
    "dev_per_query_k_matrix": "results/week2/processed/dev_per_query_k_matrix.csv",
    "best_static_baseline": "results/week2/processed/best_static_baseline.json",
    "dev_retrieval_features": "results/week2/processed/dev_retrieval_features.jsonl",
    "dev_fixed_k_matrix": "results/week2/raw/dev_fixed_k_matrix.jsonl",
    "dev_retrieval": "results/week2/raw/dev_retrieval.jsonl",
    "dev_sfr_embeddings": "results/week2/raw/dev_sfr_embeddings.pt"
}

print("=== 1. FILE EXISTENCE & SIZES ===")
for k, v in files.items():
    exists = os.path.exists(v)
    size = os.path.getsize(v) if exists else -1
    print(f"  {k}: exists={exists}, size={size} bytes ({v})")

print("\n=== 2. RAW FIXED K MATRIX ===")
records = []
with open("results/week2/raw/dev_fixed_k_matrix.jsonl") as f:
    for line in f:
        records.append(json.loads(line))
df_raw = pd.DataFrame(records)
print(f"Total raw records: {len(df_raw)}")
unique_q = df_raw["question_id"].nunique()
unique_p = df_raw["paper_id"].nunique()
k_vals = sorted(df_raw["k"].unique().tolist())
print(f"Unique questions: {unique_q}")
print(f"Unique papers: {unique_p}")
print(f"Present k values: {k_vals}")

q_k_counts = df_raw.groupby("question_id")["k"].apply(list).reset_index()
expected_k = [0, 2, 4, 5, 6, 8, 10]
all_match = all(sorted(k_list) == expected_k for k_list in q_k_counts["k"])
print(f"Do all questions have exact expected k {expected_k}? {all_match}")
dup_count = df_raw.duplicated(subset=["question_id", "k"]).sum()
print(f"Duplicate (question_id, k) pairs: {dup_count}")

f1_nulls = df_raw["token_f1"].isna().sum()
min_f1 = df_raw["token_f1"].min()
max_f1 = df_raw["token_f1"].max()
mean_f1 = df_raw["token_f1"].mean()
print(f"Token F1 nulls/NaNs: {f1_nulls}")
print(f"Token F1 min: {min_f1:.4f}, max: {max_f1:.4f}, mean: {mean_f1:.4f}")

print("\n=== 3. PROCESSED PER QUERY MATRIX ===")
df_mat = pd.read_csv("results/week2/processed/dev_per_query_k_matrix.csv")
print(f"Shape: {df_mat.shape}")
print(f"Columns: {list(df_mat.columns)}")
print(f"Questions in matrix: {len(df_mat)}, unique papers: {df_mat['paper_id'].nunique()}")

print("\n=== 4. RETRIEVAL FEATURES ===")
feat_records = []
with open("results/week2/processed/dev_retrieval_features.jsonl") as f:
    for line in f:
        feat_records.append(json.loads(line))
df_feats = pd.DataFrame(feat_records)
print(f"Total feature records: {len(df_feats)}")
print(f"Feature columns: {list(df_feats.columns)}")
feat_q_set = set(df_feats["question_id"])
raw_q_set = set(df_raw["question_id"])
print(f"Exact match between feature question_ids and raw question_ids: {feat_q_set == raw_q_set}")
dup_feat_q = df_feats["question_id"].duplicated().sum()
print(f"Any duplicate question_id in features? {dup_feat_q}")

# Check features schema and nulls
feat_cols = ["rho_1", "delta_12", "entropy", "entropy_normalized", "query_length", "n_high", "raw_top1_score"]
for c in feat_cols:
    if c in df_feats.columns:
        print(f"  Feature '{c}': min={df_feats[c].min():.4f}, max={df_feats[c].max():.4f}, nulls={df_feats[c].isna().sum()}")
    else:
        print(f"  Feature '{c}': NOT FOUND!")

print("\n=== 5. BEST STATIC BASELINE ===")
with open("results/week2/processed/best_static_baseline.json") as f:
    best_static = json.load(f)
print(json.dumps(best_static, indent=2))

print("\n=== 6. CHECKPOINT SUMMARY ===")
with open("results/week2/processed/week2_validation_checkpoint.json") as f:
    ckpt = json.load(f)
print(f"Checkpoint keys: {list(ckpt.keys())}")
print(f"SARA candidate set: {ckpt.get('sara_candidate_set')}")
print(f"Mean F1 by k: {ckpt.get('mean_f1_by_k')}")

print("\n=== 7. ORACLE AND EPSILON DEFINITIONS AUDIT ===")
# Quality oracle: k*_{qual}(q) = argmax_{k in K_alloc} F1(q, k), ties broken to smaller k
# K_alloc = {2, 4, 5, 6, 8}
# K_sweep = {0, 2, 4, 5, 6, 8, 10}
K_alloc = [2, 4, 5, 6, 8]
K_sweep = [0, 2, 4, 5, 6, 8, 10]

# Pivot matrix for F1
df_pivot = df_raw.pivot(index="question_id", columns="k", values="token_f1")
df_pivot_alloc = df_pivot[K_alloc]
df_pivot_sweep = df_pivot[K_sweep]

# Quality Oracle on K_alloc
# In case of tie, argmax in pandas/numpy might take the first column in order [2,4,5,6,8] which is smallest k!
best_k_qual = df_pivot_alloc.idxmax(axis=1) # idxmax returns first occurrence of max
best_f1_qual = df_pivot_alloc.max(axis=1)

print(f"Quality Oracle Mean F1 (over K_alloc): {best_f1_qual.mean():.4f}")
print("Quality Oracle best_k distribution on K_alloc:")
print(best_k_qual.value_counts().sort_index())

# Best static F1 on K_alloc
mean_f1_alloc = df_pivot_alloc.mean(axis=0)
best_static_k = mean_f1_alloc.idxmax()
best_static_f1 = mean_f1_alloc.max()
print(f"Best Static k on K_alloc: {best_static_k} with Mean F1: {best_static_f1:.4f}")
print(f"Quality Headroom (Oracle - Static): {best_f1_qual.mean() - best_static_f1:.4f} (relative: {(best_f1_qual.mean() - best_static_f1)/best_static_f1*100:.2f}%)")

# Reference quality in Epsilon Oracle: max_{j in K_sweep} F1(q, j)
ref_quality_sweep = df_pivot_sweep.max(axis=1)
ref_quality_alloc = df_pivot_alloc.max(axis=1)
print(f"Mean Reference Quality over K_sweep: {ref_quality_sweep.mean():.4f}")
print(f"Mean Reference Quality over K_alloc: {ref_quality_alloc.mean():.4f}")
diff_ref = (ref_quality_sweep > ref_quality_alloc).sum()
print(f"Questions where max(K_sweep) > max(K_alloc): {diff_ref} / {len(df_pivot)} ({diff_ref/len(df_pivot)*100:.2f}%)")

# Where does the sweep reference come from when sweep > alloc?
k0_wins = (df_pivot[0] > df_pivot_alloc.max(axis=1)).sum()
k10_wins = (df_pivot[10] > df_pivot_alloc.max(axis=1)).sum()
both_win = ((df_pivot[0] > df_pivot_alloc.max(axis=1)) & (df_pivot[10] > df_pivot_alloc.max(axis=1))).sum()
print(f"  Of those, k=0 strictly exceeds max(K_alloc): {k0_wins}")
print(f"  k=10 strictly exceeds max(K_alloc): {k10_wins}")

# Check epsilon feasibility for eps in [0.00, 0.01, 0.02, 0.05]
for eps in [0.00, 0.01, 0.02, 0.05]:
    feasible_count = 0
    infeasible_count = 0
    for q_id, row in df_pivot_alloc.iterrows():
        ref = ref_quality_sweep.loc[q_id]
        # find min k in K_alloc where row[k] >= ref - eps
        feasible = [k for k in K_alloc if row[k] >= ref - eps - 1e-9]
        if feasible:
            feasible_count += 1
        else:
            infeasible_count += 1
    print(f"Epsilon = {eps:.2f}: Feasible = {feasible_count}, Infeasible = {infeasible_count} ({infeasible_count/len(df_pivot)*100:.2f}%)")
