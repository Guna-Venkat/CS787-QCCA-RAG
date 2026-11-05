"""Single, outcome-free, preregistered final resolution sweep. CPU only.

No reader, evaluator, attribution audit, fitting, or answer-analysis imports.
Original calibration and predictors are read-only. All new records are separate.
"""
import os
os.environ['CUDA_VISIBLE_DEVICES'] = ''
os.environ.setdefault('OPENBLAS_NUM_THREADS', '1')
os.environ.setdefault('OMP_NUM_THREADS', '1')
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import pandas as pd
from src.post_w12.bea_v2.protocol import (
    ROOT, load, root, read, write, text, sha, digest, implementation,
    dependencies, Store, exclusive,
)
from src.post_w12.bea_v2.models import verify_models
from src.post_w12.bea_v2.stages import allocation, calibration_summary, tokenizer

GRID = [0.00006, 0.00007, 0.00008, 0.00009]
AMENDMENT = (
    'The initial preregistered grid identified the qualifying boundary between '
    '5e-5 and 1e-4, but no grid point satisfied all mechanical constraints. '
    'A single prespecified resolution sweep of 6e-5,7e-5,8e-5,9e-5 will be '
    'performed. The original eligibility rules remain unchanged. Select the '
    'smallest threshold satisfying every original criterion. If none qualifies, '
    'terminate BEA_V2 calibration.'
)

@exclusive
def run(c=None):
    c = c or load()
    verify_models(c)
    original = root(c)/'calibration'
    target = original/'resolution'
    old_cal = read(original/'CALIBRATION.json')
    if old_cal['status'] != 'NEEDS_REVIEW' or old_cal['threshold'] is not None:
        raise ValueError('Resolution applies only to the unchanged failed initial grid')
    parent, v1, _, cohort = dependencies(c)
    small = read(original/'COHORT.json')
    expected = {d: sorted(members, key=lambda r:r['query_id'])[:3]
                for d, members in cohort['selected'].items()}
    if small != expected or sum(map(len, small.values())) != 9:
        raise ValueError('The original nine-query cohort must remain identical')
    # Snapshot only calibration/predictor/protocol inputs, never answer outcomes.
    inputs = [p for p in original.rglob('*') if p.is_file()
              and not p.is_relative_to(target)]
    inputs += [p for p in (root(c)/'utility_model').rglob('*') if p.is_file()]
    inputs += [root(c)/'CROSSFIT_FOLDS.json', root(c)/'FEATURE_SCHEMA.json']
    hashes = {str(p.relative_to(ROOT)):sha(p) for p in sorted(inputs)}
    identity = dict(amendment=AMENDMENT, thresholds=GRID, cohort=small,
                    original_rules=c['calibration'], implementation=implementation(),
                    source_hash=sha(__file__), input_hashes=hashes,
                    answer_generations=0, answer_outcomes_accessed=False,
                    heldout_LOO_labels_accessed=False, predictor_refitting=False)
    # Persist and fsync the amendment BEFORE any new allocation computation.
    write(target/'AMENDMENT.json', identity)
    text(target/'AMENDMENT.md', '# Final allocation-only resolution amendment\n\n'
         +AMENDMENT+'\n\nSame nine queries, predictors, coefficients, aspects and '
         'eligibility rules. No Qwen, F1, gold or held-out LOO inspection. '
         'Only a qualifying threshold can unlock the separately authorized pilot.\n')
    old_frame = pd.read_csv(original/'allocations.csv', dtype={'query_id':str})
    base = old_frame[old_frame.threshold == 0].to_dict('records')
    if len(base) != 18:
        raise ValueError('Exactly 18 saved zero-threshold reference allocations required')
    store = Store(target/'allocation_traces', identity)
    tok = tokenizer(parent)
    rows = []
    for d, members in small.items():
        for member in members:
            for b in c['budgets']:
                for tau in GRID:
                    key = [d,member['query_id'],b,tau]
                    item = store.get(key)
                    if item is None:
                        item = store.put(key, allocation(c,v1,member,d,b,tok,tau))
                    sel, bh = item['selection'], item['behavior']
                    rows.append(dict(dataset=d,query_id=member['query_id'],budget=b,
                        threshold=tau,stop_reason=sel['stop_reason'],
                        utilization=sel['evidence_tokens']/b,units=len(sel['units']),
                        represented_parents=bh['represented_parents'],
                        coverage=bh['aspect_coverage'],full_sources=sel['full_sources']))
            print('Resolution allocation complete:',d,member['query_id'],flush=True)
    summary, cells = calibration_summary(pd.DataFrame(base+rows),c)
    summary = [r for r in summary if r['threshold'] in GRID]
    cells = [r for r in cells if r['threshold'] in GRID]
    eligible = [r['threshold'] for r in summary if r['eligible']]
    chosen = min(eligible) if eligible else None
    for name, values in [('allocations',rows),('threshold_summary',summary),('cell_checks',cells)]:
        text(target/(name+'.csv'),pd.DataFrame(values).to_csv(index=False))
    for path, expected_hash in hashes.items():
        if sha(ROOT/path) != expected_hash:
            raise ValueError('Original calibration/predictor changed: '+path)
    result = dict(status='READY' if eligible else 'CALIBRATION_FAILED_TERMINATED',
                  threshold=chosen, amendment_sha=sha(target/'AMENDMENT.json'),
                  identity_hash=digest(identity), queries=9, new_allocations=len(rows),
                  answer_generations=0, answer_outcomes_accessed=False,
                  heldout_LOO_labels_accessed=False, original_inputs_unchanged=True,
                  summary=summary, cell_checks=cells,
                  further_calibration_permitted=False, reader_pilot_permitted=bool(eligible))
    write(target/'RESOLUTION_RESULT.json',result)
    if not eligible:
        write(root(c)/'CALIBRATION_TERMINATED.json',result)
    text(target/'RESOLUTION_REPORT.md','# Final BEA_V2 allocation-only resolution\n\n'
         +AMENDMENT+'\n\n'+pd.DataFrame(summary).to_markdown(index=False)
         +'\n\n## Dataset/budget checks\n\n'+pd.DataFrame(cells).to_markdown(index=False)
         +'\n\nStatus: **'+result['status']+'**; selected threshold: '+str(chosen)
         +'. Original inputs are unchanged. No reader, answer scores, gold labels, '
         'held-out LOO labels or refitting were used.\n')
    print(pd.DataFrame(summary).to_string(index=False),flush=True)
    print('FINAL RESOLUTION:',result['status'],'threshold:',chosen,flush=True)
    return result

if __name__ == '__main__':
    run()
