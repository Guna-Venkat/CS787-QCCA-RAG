"""Independent read-only oracle arithmetic, source, feature and fold audit."""
import csv
import json
import sys
from pathlib import Path
import math

ROOT=Path(__file__).resolve().parents[4]
sys.path.insert(0,str(ROOT))
from src.week9.provenance import branch_guard,digest,file_hash
from src.week9.caching import read_jsonl,atomic_json


def main():
    branch_guard()
    root=ROOT/'results/week9/w9g_action_verification/v1';pre=root/'preflight'
    ident=json.loads((pre/'protocol.json').read_text());manifest=json.loads((pre/'preflight_manifest.json').read_text())
    assert manifest['complete'] and manifest['lm_calls']==0 and not manifest['controller_training_started']
    assert manifest['protocol_hash']==digest(ident)
    for p,h in {**ident['sources'],**ident['input_hashes'],**manifest['artifact_hashes']}.items():
        path=Path(p) if Path(p).is_absolute() else ROOT/p
        assert file_hash(path)==h,p
    froot=ROOT/'results/week9/w9f_utilization/v1'
    rows=read_jsonl(froot/'fit/results.jsonl')
    groups={p:{r['question_id']:r for r in rows if r['policy']==p} for p in ('D0','F0','F2-A')}
    expected=set(groups['D0']);assert len(expected)==1746 and all(set(g)==expected for g in groups.values())
    observed=list(csv.DictReader((pre/'oracle_summary.csv').open()));distribution=list(csv.DictReader((pre/'oracle_distribution.csv').open()))
    for name,actions in [('PRE-A',('D0','F2-A')),('PRE-B',('D0','F0')),('PRE-C',('D0','F0','F2-A'))]:
        for variant in ('exact','epsilon'):
            selected=[]
            for q in sorted(expected):
                maximum=max(groups[p][q]['f1'] for p in actions)
                preference=[p for p in ('D0','F2-A','F0') if p in actions]
                chosen=next(p for p in preference if (groups[p][q]['f1']==maximum if variant=='exact' else groups[p][q]['f1']>=maximum-.01))
                selected.append((chosen,groups[chosen][q]['f1']))
            record=next(r for r in observed if r['action_space']==name and r['oracle']==variant)
            assert math.isclose(sum(v for _,v in selected)/1746,float(record['f1']),abs_tol=1e-12)
            for p in actions:
                r=next(r for r in distribution if r['action_space']==name and r['oracle']==variant and r['action']==p)
                assert int(r['count'])==sum(a==p for a,_ in selected)
    schema=json.loads((root/'scaffold/feature_schema.json').read_text())
    names=schema['feature_columns'];assert len(names)==32
    forbidden=('gold','f1','oracle','target','help','hurt','paper_id','question_id')
    assert all(not any(s in name.lower() for s in forbidden) for name in names)
    with (root/'scaffold/inference_features.csv').open() as f:
        feature_rows=list(csv.DictReader(f))
    assert len(feature_rows)==1746 and set(feature_rows[0])==set(names+['question_id','paper_id'])
    with (root/'scaffold/fold_assignments.csv').open() as f:
        folds=list(csv.DictReader(f))
    papers={}
    for r in folds:papers.setdefault(r['paper_id'],set()).add(r['outer_fold'])
    assert len(papers)==656 and all(len(v)==1 for v in papers.values())
    assert {r['outer_fold'] for r in folds}=={'0','1','2','3','4'}
    assert not (root/'controller_fit_oof').exists()
    assert not list(root.rglob('*.joblib'))
    decision=json.loads((pre/'go_no_go.json').read_text());head=json.loads((pre/'headroom_analysis.json').read_text())
    result=dict(branch='gagan',preflight_completed=True,controller_training_started=False,lm_calls=0,
        old_artifacts_unchanged=True,queries=1746,papers=656,
        independent_oracle_arithmetic_passed=True,action_distributions_match=True,
        feature_count=32,no_outcome_columns_in_model_features=True,outer_paper_folds=5,
        no_paper_overlap=True,recommended_route=decision['recommended_route'],
        G1=decision['G1']['decision'],G2=decision['G2']['decision'],G3=decision['G3']['decision'],
        f2_unique_exact_best_count=head['f2_unique_contribution']['f2_unique_exact_best_count'],
        source_protocol_hash=digest(ident))
    atomic_json(Path(__file__).with_name('verification.json'),result)
    print(json.dumps(result,indent=2))


if __name__=='__main__':main()
