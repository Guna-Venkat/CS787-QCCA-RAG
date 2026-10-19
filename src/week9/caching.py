"""Append-only, fsynced, locked JSONL cache; incomplete final writes are repaired."""
from __future__ import annotations
import fcntl
import json
import os
from pathlib import Path
from .provenance import digest


def read_jsonl(path):
    with Path(path).open(encoding='utf-8') as f:
        return [json.loads(line) for line in f if line.strip()]


def atomic_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + '.tmp')
    with temp.open('w', encoding='utf-8') as f:
        json.dump(value, f, indent=2, ensure_ascii=False, allow_nan=False)
        f.write('\n')
        f.flush()
        os.fsync(f.fileno())
    os.replace(temp, path)


class RowCache:
    def __init__(self, path, identity, resume=False):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.lock = self.path.with_suffix('.lock').open('a')
        try:
            fcntl.flock(self.lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            self.lock.close()
            raise RuntimeError(f'Concurrent writer: {path}')
        self.stamp = digest(identity)
        self.rows = {}
        try:
            meta = self.path.with_suffix('.identity.json')
            if meta.exists():
                if json.loads(meta.read_text()) != identity:
                    raise ValueError(f'Incompatible source/config/model/prompt cache: {path}')
                if self.path.exists() and self.path.stat().st_size and not resume:
                    raise FileExistsError(f'Use --resume or a new output directory: {path}')
            else:
                if self.path.exists() and self.path.stat().st_size:
                    raise ValueError('Unverified cache has no identity sidecar')
                atomic_json(meta, identity)
            self.path.touch(exist_ok=True)
            if self.path.exists():
                with self.path.open('rb+') as f:
                    good_end = 0
                    while line := f.readline():
                        if not line.endswith(b'\n'):
                            f.truncate(good_end)
                            break
                        row = json.loads(line)
                        key = row['cache_key']
                        content = {k: v for k, v in row.items() if k != 'row_content_hash'}
                        if (row.get('row_content_hash') != digest(content) or
                                row['cache_fingerprint'] != self.stamp or key in self.rows):
                            raise ValueError('Cache identity mismatch or duplicate row')
                        self.rows[key] = row
                        good_end = f.tell()
        except BaseException:
            self.close()
            raise

    def key(self, question_id, paper_id, candidate_ids, operation):
        return digest([str(question_id), str(paper_id), list(candidate_ids), operation, self.stamp])

    def put(self, key, row):
        if key in self.rows:
            return self.rows[key]
        row = {k: v for k, v in row.items() if k != 'row_content_hash'}
        row = {**row, 'cache_key': key, 'cache_fingerprint': self.stamp}
        row['row_content_hash'] = digest(row)
        with self.path.open('a', encoding='utf-8') as f:
            f.write(json.dumps(row, ensure_ascii=False, allow_nan=False) + '\n')
            f.flush()
            os.fsync(f.fileno())
        self.rows[key] = row
        return row

    def close(self):
        if not self.lock.closed:
            fcntl.flock(self.lock, fcntl.LOCK_UN)
            self.lock.close()

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.close()
