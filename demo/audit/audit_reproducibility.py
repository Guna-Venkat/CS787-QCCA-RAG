"""Read-only inventory of the final selective Git/archive preservation policy."""
import hashlib,json,subprocess
from pathlib import Path
from importlib.metadata import distributions
ROOT=Path(__file__).resolve().parents[2];OUT=Path(__file__).parent
tracked=set(subprocess.check_output(['git','ls-files','--','demo'],cwd=ROOT,text=True).splitlines())
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
# Derived inventory outputs are excluded from their own recursive hash census.
excluded={'demo/audit/demo_preservation_manifest.json','demo/audit/demo_reproducibility_status.md'}
rows=[dict(path=n,bytes=(ROOT/n).stat().st_size,sha256=sha(ROOT/n),tracked=True) for n in sorted(tracked-excluded) if (ROOT/n).is_file()]
for n in ['demo/replay_data/v1/bundle.json']:
 p=ROOT/n
 if p.exists():rows.append(dict(path=n,bytes=p.stat().st_size,sha256=sha(p),tracked=False,preservation='Exact member of versioned immutable archive'))
archive=json.loads((ROOT/'demo/releases/curated-demo-v1.archive.json').read_text())
(OUT/'demo_preservation_manifest.json').write_text(json.dumps(dict(branch=subprocess.check_output(['git','branch','--show-current'],cwd=ROOT,text=True).strip(),demo_ignore_rule='selective runtime/private/intermediate ignores',archive=archive,files=rows),indent=2)+'\n')
(OUT/'requirements_qualified_freeze.txt').write_text('\n'.join(sorted({d.metadata['Name']+'=='+d.version for d in distributions() if d.metadata['Name']}))+'\n')
lines=['# Demo reproducibility status — final freeze','',f'Git-indexed demo files: **{len(tracked)}**. Source, tests, UI/CSS, tokenizer-only assets, documentation, pinned environment snapshot, publication metadata and the versioned curated archive are preserved on branch `gagan`. The user-authorized final freeze replaces the prior blanket `/demo/` ignore with selective ignores.','',
'## Clean checkout and immutable data restore','',
'`demo/releases/curated-demo-v1.tar.gz` is the versioned immutable archive mechanism, stored alongside source. It is 13.93 MiB compressed and contains the original bundle, publication receipt, curated manifest and SHA256SUMS. The raw approximately 98 MiB JSON is ignored; it is not an ordinary mutable Git object. Git LFS or external hosting is not required.','',
'```bash','python3.11 -m venv demo/.venv','demo/.venv/bin/python -m pip install -r demo/requirements.txt','demo/.venv/bin/python demo/restore_curated_bundle.py','demo/.venv/bin/python -m streamlit run demo/app.py','```','',
'The restore checks archive/member checksums before writing, refuses incompatible existing files and is idempotent. The demo can read the Git-tracked compressed split-metadata JSON without recreating an uncompressed research artifact. Optional live model weights/checkpoints remain external; curated playback needs no model download or inference.','',
'## Intentionally excluded','',
'Environments, model weights, checkpoints, credentials, Streamlit/runtime caches, arbitrary user inputs, debug logs, temporary generations and large intermediate build shards remain ignored. The explicitly approved static World Cup diagnostic is a versioned preset fixture; it is not a collection of arbitrary user session input. Its frozen report is documentation, while temporary condition shards/cache remain excluded.','',
'## File integrity inventory','', '| Path | Git indexed? | Bytes | SHA256 |','|---|---|---:|---|']
for r in rows:lines.append(f"| `{r['path']}` | {'Yes' if r['tracked'] else 'Archive member'} | {r['bytes']} | `{r['sha256']}` |")
lines+=['','## Submission procedure','',
'Keep the versioned archive and its .archive.json checksum file unchanged; future data require a new version. Submit or push the source commit containing these files. Verify an isolated checkout by restoring the archive and launching Curated Demo. Do not rebuild the curated model-generation grid for playback. Keep a separate institutional backup of the Git release if required by submission policy. A normal clean checkout plus documented restore is sufficient for the UI; optional live inference additionally needs the locally accessible pinned models and qualified runtime.','',
'`demo_preservation_manifest.json` is the machine-readable inventory. Derived inventory files omit their own recursive hashes. `requirements_qualified_freeze.txt` records qualification versions; `demo/requirements.txt` is the smaller curated-UI dependency entry point.']
(OUT/'demo_reproducibility_status.md').write_text('\n'.join(lines)+'\n');print('Git-indexed demo files:',len(tracked))
