# Immutable curated archive

`curated-demo-v1.tar.gz` is a deterministic, versioned archive of the original
published `bundle.json`, `PUBLISHED.json`, `manifest.json`, and `SHA256SUMS`.
It is about **13.93 MiB compressed**; the raw approximately 98 MiB JSON bundle is
ignored and is not added as an ordinary mutable Git object.

The archive and its independent checksum metadata are versioned with the demo.
Git LFS and an external release download are not required. The committed archive
is the submission's immutable archive mechanism; never replace version `v1` with
new data. A future dataset would require a new named archive and metadata file.

From a clean checkout, after installing the small UI dependencies:

```bash
python3.11 -m venv demo/.venv
demo/.venv/bin/python -m pip install -r demo/requirements.txt
demo/.venv/bin/python demo/restore_curated_bundle.py
demo/.venv/bin/python -m streamlit run demo/app.py
```

Restore verifies the archive SHA256, allowed member names, every member checksum
and the original publication receipt before writing. Existing identical files are
left untouched; incompatible files are never overwritten. No model, inference,
network request or raw dataset is required for curated playback. A Git-tracked
compressed split manifest is read directly by the demo when its uncompressed copy
is absent; research artifacts are not rewritten.

Archive SHA256:

`51361cdded3087f7706e9f3ae3804c19bec93f2d4a78a408595426a0359fee38`

Original bundle SHA256:

`14b4ba43d4cf3ab7359b54f2fc8bfe0ee59166c12f40c09125658c1d63f03036`

Optional live inference still needs the locally accessible pinned model checkpoints
and the qualified research dependencies. These model weights and credentials are
not part of this archive.
