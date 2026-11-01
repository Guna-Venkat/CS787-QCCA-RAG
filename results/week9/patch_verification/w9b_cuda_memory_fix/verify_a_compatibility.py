import ast,json
from pathlib import Path
from src.week9.provenance import source_hashes,file_hash
root=Path('results/week9/patch_verification/w9b_cuda_memory_fix')
before=json.loads((root/'before_sources.json').read_text())
checks={}
exceptions={'backend':{'SaraBackend'},'runner':{'load_phase'},'config':{'load_frozen'},'provenance':set()}
for name,excluded in exceptions.items():
    old=ast.parse((root/'before'/f'{name}.py').read_text())
    new=ast.parse(Path(f'src/week9/{name}.py').read_text())
    defs=lambda tree:{n.name:ast.dump(n,include_attributes=False) for n in tree.body if isinstance(n,(ast.FunctionDef,ast.ClassDef))}
    a,b=defs(old),defs(new)
    checked=[k for k in a if k not in excluded]
    assert all(a[k]==b[k] for k in checked),name
    checks[name]=checked
current=source_hashes()
changed=[p for p in current if current[p]!=before.get(p)]
assert set(changed)=={f'src/week9/{n}.py' for n in exceptions}
frozen=Path('results/week9/frozen/w9a_frozen_config.json')
assert json.loads(frozen.read_text())['source_hashes']==before
proof=dict(migration_id='w9b-titan-fp16-sdpa-v1',phase='a',a_algorithm_ast_verified=True,original_sources=before,approved_sources=current,unchanged_ast_definitions=checks,changed_files=changed,frozen_a_sha256=file_hash(frozen),scope='Read-only historical A compatibility. No A rows, identities, frozen settings, retrieval/chunking algorithms or scientific artifacts changed.')
(root/'a_source_compatibility.json').write_text(json.dumps(proof,indent=2)+'\n')
print('Exact source compatibility verified; frozen A SHA:',proof['frozen_a_sha256'])
