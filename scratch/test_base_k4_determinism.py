import sys
from pathlib import Path
REPO_ROOT = Path('.').resolve()
SARA_ROOT = REPO_ROOT / 'Baselines/SARA-main'
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
import src
if hasattr(src, '__path__') and str(SARA_ROOT / 'src') not in src.__path__:
    src.__path__.append(str(SARA_ROOT / 'src'))

import json
import torch
from src.model.loader import load_sara_for_eval
from src.week6.routing import build_routed_sara_prompt, extract_routed_sara_embeddings
from src.week6.evaluation import compute_qa_answer_metrics, extract_generated_text
from src.week8.data import load_config, load_week8_split
from src.week8.semantic import SemanticScorer
from src.week7.w7e.fidelity_allocator import get_semantic_rank_order

cfg = load_config()
records, sfr_dict, manifest_df = load_week8_split('internal_val', cfg)

# Load canonical k4 records from w7b
k4_by_qid = {}
with open('results/week7/w7b_semantic_budget/response_surface/train_semantic_fixed_k_results.jsonl') as f:
    for line in f:
        r = json.loads(line)
        qid = str(r['question_id'])
        if r['k'] == 4:
            k4_by_qid[qid] = r

sara_model, sara_tokenizer = load_sara_for_eval('Baselines/SARA-main/checkpoints/finetune/sara_qasper_proj_lr5e4_seed42', device='cuda')
scorer = SemanticScorer()

matches = 0
for r in records[:5]:
    qid = str(r['question_id'])
    q_text = r['question']
    passages = [p['text'] for p in r.get('passages', [])[:10]]
    sims = scorer.score_query_passages(q_text, passages).cpu().numpy()
    semantic_order = get_semantic_rank_order(sims, n_passages=len(passages))
    top4 = sorted(list(semantic_order[:4]))
    
    prompt, _, unsel = build_routed_sara_prompt(q_text, r.get('passages', [])[:10], top4, sara_tokenizer)
    inputs = sara_tokenizer(prompt, return_tensors='pt').to('cuda')
    comp_embeds = extract_routed_sara_embeddings(sfr_dict[qid], top4).to(device='cuda', dtype=sara_model.dtype)
    
    with torch.no_grad():
        out = sara_model.generate(**inputs, retrieval_embeds=comp_embeds, max_new_tokens=128, do_sample=False, pad_token_id=sara_tokenizer.eos_token_id)
    pred = extract_generated_text(sara_tokenizer, out)[0].strip()
    
    k4_pred = k4_by_qid[qid]['prediction'].strip()
    print(f'QID {qid}: Fresh=\"{pred}\" vs Canonical=\"{k4_pred}\" | Match={pred == k4_pred}')
    if pred == k4_pred:
        matches += 1

print(f'Total matches: {matches}/5', flush=True)
