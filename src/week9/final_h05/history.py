"""Exact recovered sources, isolated imports, and saved non-TEST replay proofs."""
import contextlib
import hashlib
import importlib
import importlib.abc
import importlib.util
import json
import subprocess
import sys
from pathlib import Path
import numpy as np
import pandas as pd
from ..final_h import protocol as p

ARCHIVE = Path(__file__).parent/'historical'


def recovery():
    value = p.read(ARCHIVE/'recovery.json')
    p.require(value['classification'] == 'RECOVERED_EXACT', 'Unverified W8 substitute forbidden')
    for item in value['files'].values():
        p.require(p.file_hash(p.ROOT/item['archive_path']) == item['sha256'], 'Recovered W8 source differs')
    return value


@contextlib.contextmanager
def historical():
    """Temporarily resolve only provenance-listed W8 modules to byte-exact files."""
    rec = recovery()
    names = {name[:-3].replace('/', '.'): p.ROOT/v['archive_path']
             for name, v in rec['files'].items() if name.startswith('src/week8/')}
    class Finder(importlib.abc.MetaPathFinder):
        def find_spec(self, fullname, path=None, target=None):
            if fullname in names:
                return importlib.util.spec_from_file_location(fullname, names[fullname])
    finder = Finder()
    import src.week8
    saved = {n: sys.modules[n] for n in names if n in sys.modules}
    attrs = {n: getattr(src.week8, n.rsplit('.', 1)[1], None) for n in names}
    for n in names:
        sys.modules.pop(n, None)
    sys.meta_path.insert(0, finder)
    try:
        yield importlib.import_module('src.week8.generation'), importlib.import_module('src.week8.full_system'), importlib.import_module('src.week8.final_dev')
    finally:
        sys.meta_path.remove(finder)
        for n in names:
            sys.modules.pop(n, None)
            short = n.rsplit('.', 1)[1]
            if attrs[n] is None:
                if hasattr(src.week8, short): delattr(src.week8, short)
            else: setattr(src.week8, short, attrs[n])
        sys.modules.update(saved)


def imported(relative):
    entry = p.read(p.ROOT/'results/week9/baselines/w8_import_manifest.json')['files'][relative]
    path = p.ROOT/entry['imported_path']
    p.require(p.file_hash(path) == entry['sha256'], 'Historical output checksum differs')
    return path


def replay():
    """Replay saved DEV actions/scoring/prompt accounting; no new LM generation."""
    import joblib
    rec = recovery(); rel = 'results/week8/w8j_final_dev/full/'
    prov = p.read(imported(rel+'provenance.json'))
    p.require(hashlib.sha256(json.dumps({k:v for k,v in prov.items() if k!='fingerprint'}, sort_keys=True).encode()).hexdigest() == prov['fingerprint'], 'Historical fingerprint invalid')
    summary = p.read(imported(rel+'summary.json'))
    prediction_path = imported(rel+'original_SARA_predictions.jsonl')
    p.require(p.file_hash(prediction_path) == summary['artifact_sha256'][prediction_path.name], 'Saved W8 summary does not seal predictions')
    data = subprocess.check_output(['git', 'show', rec['commit']+':'+rel+'live_drafts_and_decisions.jsonl'], cwd=p.ROOT)
    p.require(hashlib.sha256(data).hexdigest() == summary['artifact_sha256']['live_drafts_and_decisions.jsonl'], 'Saved live decisions differ from summary')
    bases = {str(r['question_id']):r for r in map(json.loads, data.decode().splitlines())}
    predictions = [json.loads(line) for line in prediction_path.read_text().splitlines()]
    g0 = {str(r['question_id']):r for r in predictions if r['revision_mode']=='G0'}
    records = {str(r['question_id']):r for r in map(json.loads, (p.ROOT/'results/week2/raw/dev_retrieval.jsonl').read_text().splitlines())}
    p.require(len(g0)==len(bases)==len(records)==231 and set(g0)==set(bases)==set(records), 'Historical DEV membership differs')
    canonical = {str(r['question_id']):r for r in map(json.loads, (p.ROOT/'results/week7/w7b_semantic_budget/response_surface/dev_semantic_fixed_k_results.jsonl').read_text().splitlines()) if r['k']==4}
    frozen = p.read(p.ROOT/'results/week8/w8e_controller/frozen_controller.json')
    gate = joblib.load(p.ROOT/'results/week8/w8e_controller/b0_gw_gate.joblib')
    from src.week8.gate_diagnostics import U_COLUMNS
    from src.week8.selective_controller import make_decisions
    checks = []; ids = []; max_gate_error = 0.
    with historical() as (generation, full, dev):
        for q, row in sorted(g0.items()):
            base=bases[q]; inf=base['inference']
            prob=float(gate.predict_proba(np.asarray([[inf['gate_features'][k] for k in U_COLUMNS]]))[:,1][0])
            max_gate_error=max(max_gate_error,abs(prob-inf['gate_probability']))
            act=make_decisions(pd.DataFrame(inf['candidate_predictions']),{q:prob},frozen['selected_config']).iloc[0]
            same=act.selected_action==base['decision']['selected_action']
            if act.selected_action=='PROMOTE': same= same and int(act.candidate_passage_index)==base['decision']['candidate_passage_index']
            p.require(same, 'Recovered gate/action replay differs')
            p.require(base['draft']['prediction']==canonical[q]['prediction'], 'Historical first-pass D0 differs')
            promotion=row['selected_action']=='PROMOTE'
            revision=None if not promotion else dict(prediction=row['prediction'],prompt_tokens=row['final_prompt_tokens'],
                generation_seconds=row['revision_generation_seconds'],prompt_build_seconds=0.,
                generated_tokens=row['generated_tokens']-base['draft']['generated_tokens'],
                full_indices=row['full_indices'],compressed_indices=row['compressed_indices'])
            replayed=full.evaluate_row(records[q],base,revision,'original_SARA','G0',row['fingerprint'])
            for field in ('prediction','selected_action','candidate_passage_index','token_f1','exact_match','final_prompt_tokens','cumulative_prompt_tokens','generated_tokens','num_generation_passes'):
                p.require(replayed[field]==row[field], 'Historical serialization/scoring replay differs: '+field)
            ids.append(dict(question_id=q,paper_id=str(row['paper_id'])))
            checks.append(p.digest({k:row[k] for k in ('question_id','paper_id','prediction','selected_action','token_f1')}))
    return dict(classification='RECOVERED_EXACT', commit=rec['commit'], parent_commit=rec['parent_commit'],
        historical_fingerprint=prov['fingerprint'], historical_provenance_sha256=p.file_hash(imported(rel+'provenance.json')),
        source_hashes={k:v['sha256'] for k,v in rec['files'].items()}, historical_prediction_sha256=p.file_hash(prediction_path),
        live_decisions_sha256=hashlib.sha256(data).hexdigest(), queries=231, membership=ids,
        action_matches=231, canonical_draft_matches=231, answer_serialization_matches=231,
        metric_matches=231, token_accounting_matches=231, maximum_gate_probability_error=max_gate_error,
        output_row_projection_hash=p.digest(checks),
        limitations=['Saved-output/action replay, not fresh GPU numerical regeneration',
            'Historical latency components are not measured deployment end-to-end latency'],
        test_accessed=False, actual_generation_calls=0)
