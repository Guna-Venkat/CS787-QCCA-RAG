import json
import pandas as pd
from pathlib import Path

REPO_ROOT = Path("/home/gunavenkat/Downloads/CS787-RAG-Project")

manifest = pd.read_csv(REPO_ROOT / "results/week5/content_allocator/w5b/splits/internal_val_manifest.csv")
val_qids = set(manifest["question_id"].astype(str))

# Load Week 7 w7b k=4 results
w7b_path = REPO_ROOT / "results/week7/w7b_semantic_budget/response_surface/train_semantic_fixed_k_results.jsonl"
w7b_records = {}
with open(w7b_path) as f:
    for line in f:
        r = json.loads(line)
        qid = str(r["question_id"])
        if qid in val_qids and r.get("k") == 4:
            w7b_records[qid] = r

print(f"Loaded {len(w7b_records)} Week 7 k=4 records for internal_val.")

# Load Week 8-A promotion surface
w8a_path = REPO_ROOT / "results/week8/surfaces/promotion_surface_internal_val.jsonl"
w8a_records = {}
with open(w8a_path) as f:
    for line in f:
        r = json.loads(line)
        qid = str(r["question_id"])
        # Take the base info from the first record for each qid
        if qid not in w8a_records:
            w8a_records[qid] = r

print(f"Loaded {len(w8a_records)} Week 8-A base records for internal_val.")

# Compare query by query
discrepancies = []
f1_diffs = []
pred_diffs = 0
full_idx_diffs = 0
comp_idx_diffs = 0

common_qids = sorted(list(set(w7b_records.keys()) & set(w8a_records.keys())))
print(f"Common queries evaluated: {len(common_qids)}")

for qid in common_qids:
    r7 = w7b_records[qid]
    r8 = w8a_records[qid]
    
    f1_7 = r7.get("token_f1", 0.0)
    f1_8 = r8.get("base_f1", 0.0)
    diff = f1_8 - f1_7
    f1_diffs.append(diff)
    
    pred_7 = str(r7.get("prediction", "")).strip()
    pred_8 = str(r8.get("base_prediction", "")).strip()
    
    # Check passage indices
    full_7 = r7.get("selected_full_indices")
    comp_7 = r7.get("selected_compressed_indices")
    
    full_8 = r8.get("base_full_indices")
    comp_8 = r8.get("base_compressed_indices")
    
    is_pred_diff = pred_7 != pred_8
    is_idx_diff = (full_7 != full_8) or (comp_7 != comp_8)
    
    if is_pred_diff:
        pred_diffs += 1
    if full_7 != full_8:
        full_idx_diffs += 1
    if comp_7 != comp_8:
        comp_idx_diffs += 1
        
    if abs(diff) > 1e-4 or is_pred_diff or is_idx_diff:
        discrepancies.append({
            "question_id": qid,
            "f1_w7": f1_7,
            "f1_w8": f1_8,
            "f1_diff": diff,
            "pred_w7": pred_7,
            "pred_w8": pred_8,
            "full_w7": full_7,
            "full_w8": full_8,
            "comp_w7": comp_7,
            "comp_w8": comp_8,
            "prompt_w7": r7.get("prompt", "")[:120],
            "prompt_w8": r8.get("base_prompt", "")[:120],
        })

print("\n=== Audit Summary ===")
print(f"Mean F1 Week 7 (k=4): {sum(w7b_records[q]['token_f1'] for q in common_qids)/len(common_qids):.5f}")
print(f"Mean F1 Week 8 (base): {sum(w8a_records[q]['base_f1'] for q in common_qids)/len(common_qids):.5f}")
print(f"Difference: {sum(f1_diffs)/len(f1_diffs):.5f}")
print(f"Queries with prediction difference: {pred_diffs} / {len(common_qids)}")
print(f"Queries with FULL indices difference: {full_idx_diffs} / {len(common_qids)}")
print(f"Queries with Compressed indices difference: {comp_idx_diffs} / {len(common_qids)}")

if discrepancies:
    print("\nFirst 5 discrepancies:")
    for d in discrepancies[:5]:
        print(f"\nQID: {d['question_id']}")
        print(f"  F1 w7: {d['f1_w7']:.4f} | F1 w8: {d['f1_w8']:.4f} (diff: {d['f1_diff']:.4f})")
        print(f"  FULL w7: {d['full_w7']} vs w8: {d['full_w8']}")
        print(f"  COMP w7: {d['comp_w7']} vs w8: {d['comp_w8']}")
        print(f"  Pred w7: '{d['pred_w7']}'")
        print(f"  Pred w8: '{d['pred_w8']}'")
        print(f"  Prompt w7 start: {d['prompt_w7']}")
        print(f"  Prompt w8 start: {d['prompt_w8']}")
