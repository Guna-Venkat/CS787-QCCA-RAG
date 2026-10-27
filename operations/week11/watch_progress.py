"""Passive, durable 15-minute progress recorder. Never launches model work."""
import argparse
from collections import Counter
from datetime import datetime, timezone
import fcntl
import json
import os
from pathlib import Path
import shutil
import signal
import sqlite3
import subprocess
import sys
import threading
import time

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from src.week11.cache import atomic, file_hash, digest
from src.week11.protocol import load, DIRECTORIES


def writer_active(path):
    if not path.exists():
        return False
    with path.open('r') as stream:
        try:
            fcntl.flock(stream, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            return True
        fcntl.flock(stream, fcntl.LOCK_UN)
    return False


def snapshot(cfg):
    result = dict(timestamp_utc=datetime.now(timezone.utc).isoformat(),
                  branch='gagan', disk_free_GiB=round(shutil.disk_usage(ROOT).free / 2**30, 2),
                  phases={}, embedding_vectors={}, warnings=[])
    result['selections_cached'] = sum(len(p.stem) == 64 for p in (ROOT / cfg['cache_root'] / 'selections').rglob('*.json'))
    result['notebook_states'] = {}
    for path in (ROOT / 'results/week11/monitoring').glob('cell_state_*.json'):
        state = json.loads(path.read_text())
        result['notebook_states'][state['phase']] = {k: state.get(k) for k in ['status', 'current_cell', 'pid', 'error_type']}
        if state['status'] == 'ERROR':
            result['warnings'].append(dict(phase=state['phase'], notebook_error=state.get('error_type')))
    for phase, directory in DIRECTORIES.items():
        root = ROOT / cfg['output_root'] / directory
        status = json.loads((root / 'status.json').read_text()) if (root / 'status.json').exists() else {}
        counts = Counter()
        malformed = 0
        for path in (root / 'rows').glob('*.json'):
            if len(path.stem) != 64:
                continue
            # Keys precede potentially multi-megabyte rankings. Do not load rankings.
            with path.open() as stream:
                head = stream.read(4096)
            try:
                key = json.JSONDecoder().raw_decode(head.split('"key":', 1)[1].lstrip())[0]
                counts[str(key[0])] += 1
            except (ValueError, IndexError):
                malformed += 1
        active = writer_active(root / 'run.lock')
        report_ready = all((root / name).exists() for name in ['REPORT.md', 'figures.json'])
        result['phases'][phase] = dict(status=status, checkpoint_rows=dict(counts),
                                      writer_active=active, report_ready=report_ready)
        if malformed:
            result['warnings'].append(dict(phase=phase, malformed_checkpoint_headers=malformed))
    for name in ['MiniLM', 'SFR']:
        path = ROOT / cfg['cache_root'] / 'embeddings/shared_units' / name / 'embeddings.sqlite3'
        if path.exists():
            try:
                with sqlite3.connect('file:' + str(path) + '?mode=ro', uri=True, timeout=2) as connection:
                    result['embedding_vectors'][name] = connection.execute('SELECT COUNT(*) FROM vectors').fetchone()[0]
            except sqlite3.Error as exc:
                result['warnings'].append(dict(embedding_database=name, error=str(exc)))
    try:
        result['gpu'] = subprocess.check_output(
            ['nvidia-smi', '--query-gpu=utilization.gpu,memory.used,memory.total', '--format=csv,noheader'],
            text=True, timeout=10).strip()
    except (OSError, subprocess.SubprocessError):
        result['gpu'] = 'unavailable'
    if result['disk_free_GiB'] < 10:
        result['warnings'].append('Disk free space below 10 GiB; review before launching next phase')
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--interval', type=int, default=900)
    parser.add_argument('--once', action='store_true')
    args = parser.parse_args()
    if args.interval < 60:
        raise ValueError('Progress polling interval must be at least 60 seconds')
    cfg = load()
    directory = ROOT / 'results/week11/monitoring'
    directory.mkdir(parents=True, exist_ok=True)
    stop = threading.Event()
    signal.signal(signal.SIGTERM, lambda *_: stop.set())
    signal.signal(signal.SIGINT, lambda *_: stop.set())
    with (directory / 'watcher.lock').open('a') as stream:
        fcntl.flock(stream, fcntl.LOCK_EX | fcntl.LOCK_NB)
        while not stop.is_set():
            record = snapshot(load())
            previous = json.loads((directory / 'latest.json').read_text()) if (directory / 'latest.json').exists() else {}
            activity = digest([record['selections_cached'], record['embedding_vectors'],
                               {p: v['checkpoint_rows'] for p, v in record['phases'].items()}, record['notebook_states']])
            changed_at = previous.get('last_activity_change_epoch', time.time()) if previous.get('activity_signature') == activity else time.time()
            record.update(activity_signature=activity, last_activity_change_epoch=changed_at, monitor_pid=os.getpid())
            if any(v['writer_active'] for v in record['phases'].values()) and time.time() - changed_at >= 2700:
                record['warnings'].append('No new selections, embeddings or query checkpoints for 45 minutes; inspect CPU/kernel before assuming a stall')
            record.update(interval_seconds=args.interval, next_check_epoch=time.time() + args.interval,
                          monitor_sha256=file_hash(Path(__file__)), mode='passive; agent reviews and starts phases')
            atomic(directory / 'latest.json', record)
            with (directory / 'history.jsonl').open('a') as output:
                output.write(json.dumps(record, allow_nan=False) + '\n')
            print(json.dumps(record, allow_nan=False), flush=True)
            if args.once or record['phases']['I']['status'].get('executable_complete') or (directory / 'STOP_MONITOR').exists():
                break
            stop.wait(args.interval)


if __name__ == '__main__':
    main()
