"""Lossless compressed, indexed disk caches for the large W9-A grid only."""
from collections.abc import Mapping
from contextlib import closing
from functools import lru_cache
import fcntl
import json
from pathlib import Path
import sqlite3
import zlib
from .caching import atomic_json, RowCache
from .provenance import digest

FILTERS = {'question_id', 'paper_id', 'representation', 'retriever', 'retrieval_width'}


def decode(blob, stamp=None):
    row = json.loads(zlib.decompress(blob))
    if row.get('row_content_hash') != digest({k:v for k,v in row.items() if k!='row_content_hash'}):
        raise ValueError('Compressed cache row checksum mismatch')
    if stamp is not None and row.get('cache_fingerprint') != stamp:
        raise ValueError('Compressed cache identity mismatch')
    return row


def where(filters):
    if set(filters)-FILTERS:
        raise ValueError('Unsupported retrieval cache filter')
    return (' WHERE '+' AND '.join(f'{k} = ?' for k in filters) if filters else ''), list(filters.values())


def read_rows(path, filters=None, metrics_only=False):
    """Read a verified subset; never load the entire retrieval grid for B-H."""
    path = Path(path)
    identity = json.loads(path.with_suffix('.identity.json').read_text())
    with closing(sqlite3.connect(path.resolve().as_uri()+'?mode=ro', uri=True)) as db:
        clause, args = where(filters or {})
        for (blob,) in db.execute('SELECT payload FROM entries'+clause+' ORDER BY rowid', args):
            row = decode(blob, digest(identity))
            if metrics_only:
                row = {k:v for k,v in row.items() if k not in ('candidates','chunks','vector')}
            yield row


class DiskRows(Mapping):
    def __init__(self, owner):
        self.owner = owner
        self._get = lru_cache(maxsize=2048)(self._uncached)

    def _uncached(self, key):
        row = self.owner.db.execute('SELECT payload FROM entries WHERE cache_key=?', (key,)).fetchone()
        if row is None:
            raise KeyError(key)
        return decode(row[0], self.owner.stamp)

    def __getitem__(self, key):
        return self._get(key)

    def __contains__(self, key):
        return self.owner.db.execute('SELECT 1 FROM entries WHERE cache_key=?', (key,)).fetchone() is not None

    def __len__(self):
        return self.owner.db.execute('SELECT count(*) FROM entries').fetchone()[0]

    def __iter__(self):
        return (r[0] for r in self.owner.db.execute('SELECT cache_key FROM entries ORDER BY rowid'))

    def values(self):
        return self.query()

    def query(self, **filters):
        clause, args = where(filters)
        for (blob,) in self.owner.db.execute('SELECT payload FROM entries'+clause+' ORDER BY rowid', args):
            yield decode(blob, self.owner.stamp)

    def count(self, **filters):
        clause, args = where(filters)
        return self.owner.db.execute('SELECT count(*) FROM entries'+clause, args).fetchone()[0]


class RetrievalCache:
    """Same key/put/context API as RowCache, with bounded RAM and zlib payloads.

    Commit every 128 rows and on close. An abrupt kill can discard the uncommitted
    CPU-only batch; SQLite rolls it back and resume recomputes it. No partial rows.
    """
    key = RowCache.key

    def __init__(self, path, identity, resume=False):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.lock = self.path.with_suffix('.lock').open('a')
        self.db = None
        self.pending = 0
        ready = False
        try:
            fcntl.flock(self.lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
            sidecar = self.path.with_suffix('.identity.json')
            if sidecar.exists():
                if json.loads(sidecar.read_text()) != identity:
                    raise ValueError('Incompatible compressed retrieval cache identity')
            elif self.path.exists():
                raise ValueError('Unverified compressed cache has no identity')
            else:
                atomic_json(sidecar, identity)
            self.stamp = digest(identity)
            self.db = sqlite3.connect(self.path)
            self.db.execute('PRAGMA journal_mode=WAL')
            self.db.execute('PRAGMA synchronous=FULL')
            self.db.execute('CREATE TABLE IF NOT EXISTS entries (cache_key TEXT PRIMARY KEY, question_id TEXT, paper_id TEXT, representation TEXT, retriever TEXT, retrieval_width INTEGER, payload BLOB NOT NULL)')
            self.db.execute('CREATE INDEX IF NOT EXISTS query_policy ON entries(question_id,representation,retriever,retrieval_width)')
            self.db.execute('CREATE INDEX IF NOT EXISTS policy ON entries(representation,retriever,retrieval_width)')
            self.db.commit()
            self.rows = DiskRows(self)
            if len(self.rows) and not resume:
                raise FileExistsError('Use --resume or a new compressed cache directory')
            ready = True
        finally:
            if not ready:
                self.close()

    def put(self, key, row):
        if key in self.rows:
            return self.rows[key]
        row = {k:v for k,v in row.items() if k!='row_content_hash'}
        row = dict(row, cache_key=key, cache_fingerprint=self.stamp)
        row['row_content_hash'] = digest(row)
        payload = zlib.compress(json.dumps(row,ensure_ascii=False,allow_nan=False,separators=(',',':')).encode(), 6)
        self.db.execute('INSERT INTO entries VALUES (?,?,?,?,?,?,?)', (key, row.get('question_id'), row.get('paper_id'), row.get('representation'), row.get('retriever'), row.get('retrieval_width'), payload))
        self.pending += 1
        if self.pending >= 128:
            self.db.commit()
            self.pending = 0
        return row

    def metric_rows(self):
        for row in self.rows.values():
            yield {k:v for k,v in row.items() if k not in ('candidates','chunks','vector')}

    def close(self):
        if self.db is not None:
            self.db.commit()
            self.db.execute('PRAGMA wal_checkpoint(TRUNCATE)')
            self.db.close()
            self.db = None
        if not self.lock.closed:
            self.lock.close()

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.close()
