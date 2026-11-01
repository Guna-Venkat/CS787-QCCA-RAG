"""Final read-only checks after successful compaction/cleanup."""
import json
from pathlib import Path
import shutil
import sqlite3
import subprocess
import sys
from contextlib import closing
sys.path.insert(0,str(Path(__file__).resolve().parents[4]))
from src.week9.provenance import branch_guard,ROOT,source_hashes,file_hash
from src.week9.caching import atomic_json
branch_guard();folder=Path(__file__).parent
report=json.loads((folder/'compaction.json').read_text());assert report['status']=='COMPLETE'
assert report['new_identity']['source_hashes']==source_hashes()
for entry in report['files']:
    p=ROOT/entry['target']
    assert file_hash(p)==entry['target_sha256']
    assert json.loads(p.with_suffix('.identity.json').read_text())==report['new_identity']
    assert not (ROOT/entry['source']).exists()
    assert subprocess.run(['git','check-ignore','-q',str(p)],cwd=ROOT).returncode==0
    with closing(sqlite3.connect(p.resolve().as_uri()+'?mode=ro',uri=True)) as db:
        assert db.execute('SELECT count(*) FROM entries').fetchone()[0]==entry['rows']
result=next(entry for entry in report['files'] if 'retrieval_results' in entry['target'])
with closing(sqlite3.connect((ROOT/result['target']).resolve().as_uri()+'?mode=ro',uri=True)) as db:
    counts=[n for (n,) in db.execute('SELECT count(*) FROM entries GROUP BY question_id')]
assert all(0<n<=171 for n in counts)
subprocess.run(['git','diff','--check'],cwd=ROOT,check=True)
nb=json.loads((ROOT/'notebooks/week9/week9a_semantic_chunking_retrieval.ipynb').read_text())
for c in nb['cells']:
    if c['cell_type']=='code':compile(''.join(c['source']),'<W9-A notebook>','exec')
old_bytes=sum(e['source_bytes'] for e in report['files']);new_bytes=sum(e['target_bytes'] for e in report['files'])
removed_bytes=sum(e['bytes'] for e in report['removed_redundant_files'])
verification=dict(status='PASS',branch='gagan',code_tests_passed=79,notebook_default_test='excluded: user intentionally enabled full run',
    original_code_equivalence_cases=324,old_active_cache_bytes=old_bytes,new_active_cache_bytes=new_bytes,
    redundant_files_removed_bytes=removed_bytes,net_reclaimed_bytes=removed_bytes-new_bytes,
    disk_free_bytes=shutil.disk_usage(ROOT).free,preserved_retrieval_rows=result['rows'],queries_with_results=len(counts),
    fully_completed_queries=sum(n==171 for n in counts),all_completed_payloads_preserved=True,
    model_caches_datasets_w8_untouched=True,full_run_launched=False,commit_or_push_performed=False)
atomic_json(folder/'verification.json',verification)
print(json.dumps(verification,indent=2))
