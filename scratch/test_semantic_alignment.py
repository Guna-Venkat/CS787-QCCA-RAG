import json
import pandas as pd
import numpy as np
import torch
from pathlib import Path
from sentence_transformers import SentenceTransformer

REPO_ROOT = Path("/home/gunavenkat/Downloads/CS787-RAG-Project")
manifest = pd.read_csv(REPO_ROOT / "results/week5/content_allocator/w5b/splits/internal_val_manifest.csv")
val_qids = set(manifest["question_id"].astype(str))

# Load Week 7 k=4 records
w7b_path = REPO_ROOT / "results/week7/w7b_semantic_budget/response_surface/train_semantic_fixed_k_results.jsonl"
w7b_records = {}
with open(w7b_path) as f:
    for line in f:
        r = json.loads(line)
        qid = str(r["question_id"])
        if qid in val_qids and r.get("k") == 4:
            w7b_records[qid] = r

# Load retrieval data
train_retrieval_path = REPO_ROOT / "results/week5/train_response_surface/train_retrieval.jsonl"
retrieval_data = {}
with open(train_retrieval_path) as f:
    for line in f:
        r = json.loads(line)
        qid = str(r["question_id"])
        if qid in val_qids:
            retrieval_data[qid] = r

# Load MiniLM
device = "cuda" if torch.cuda.is_available() else "cpu"
encoder = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2", device=device)
encoder.eval()

matches = 0
total = len(val_qids)

for qid in sorted(list(val_qids)):
    item = retrieval_data[qid]
    q_text = item["question"]
    passages = [p["text"] for p in item.get("passages", [])[:10]]
    
    q_emb = encoder.encode([q_text], convert_to_tensor=True, device=device)
    p_embs = encoder.encode(passages, convert_to_tensor=True, device=device)
    
    q_norm = q_emb / (q_emb.norm(dim=-1, keepdim=True) + 1e-12)
    p_norm = p_embs / (p_embs.norm(dim=-1, keepdim=True) + 1e-12)
    sims = torch.mm(q_norm, p_norm.t()).squeeze(0).cpu().numpy()
    
    # Week 6/7 R1 selection:
    # 1. Top-k by descending score
    topk_desc = np.argsort(-sims)[:4]
    # 2. Presentation order preserves original BM25 rank order
    selected = sorted(int(idx) for idx in topk_desc)
    
    w7_selected = w7b_records[qid]["selected_full_indices"]
    if selected == w7_selected:
        matches += 1
    else:
        print(f"Mismatch for QID {qid}: computed {selected} vs w7 {w7_selected}")
        print("  sims:", sims)

print(f"\nFinal Alignment: {matches} / {total} ({matches/total*100:.1f}%) matches!")
