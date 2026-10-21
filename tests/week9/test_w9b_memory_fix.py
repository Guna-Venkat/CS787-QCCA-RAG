"""The historical A exception must never permit stale downstream results."""
import json
import pytest
from src.week9 import provenance as p


def test_exact_a_source_compatibility_is_phase_and_bundle_specific(tmp_path, monkeypatch):
    old={'src/week9/backend.py':'old','src/week9/retrieval.py':'same'}
    current={**old,'src/week9/backend.py':'new'}
    monkeypatch.setattr(p,'ROOT',tmp_path)
    monkeypatch.setattr(p,'source_hashes',lambda:current)
    path=tmp_path/'results/week9/patch_verification/w9b_cuda_memory_fix/a_source_compatibility.json'
    path.parent.mkdir(parents=True)
    with pytest.raises(ValueError,match='seal missing'):
        p.verify_phase_sources(old,'a')
    proof=dict(migration_id='w9b-titan-fp16-sdpa-v1',phase='a',a_algorithm_ast_verified=True,original_sources=old,approved_sources=dict(current))
    path.write_text(json.dumps(proof))
    p.verify_phase_sources(old,'a')
    for phase in 'bcdefgh':
        with pytest.raises(ValueError,match='Stale'):
            p.verify_phase_sources(old,phase)
    current['src/week9/backend.py']='future change'
    with pytest.raises(ValueError,match='exact audited'):
        p.verify_phase_sources(old,'a')
    current['src/week9/backend.py']='new'
    current['src/week9/retrieval.py']='changed algorithm'
    proof['approved_sources']=current
    path.write_text(json.dumps(proof))
    with pytest.raises(ValueError,match='exact audited'):
        p.verify_phase_sources(old,'a')
