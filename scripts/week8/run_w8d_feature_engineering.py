"""Extract measured W8-D masking/MIG features; no placeholder drafts or gold inputs."""
from __future__ import annotations
import argparse
import sys
from pathlib import Path
import torch
REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.week8.generator_features import run_feature_extraction, source_fingerprint
from src.week8.utility_training import load_surface


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument('--check', action='store_true', help='Validate real inputs; do not load SARA')
    mode.add_argument('--smoke', action='store_true', help='Extract a small isolated real-query cache')
    mode.add_argument('--full', action='store_true', help='Extract all queries, resuming completed queries')
    parser.add_argument('--split', choices=['fit', 'internal_val'], default='fit')
    parser.add_argument('--smoke-limit', type=int, default=2)
    parser.add_argument('--device', default='cuda')
    parser.add_argument('--output-dir', type=Path)
    args = parser.parse_args()
    # Avoid excessive CPU thread overhead in MiniLM and single-query SARA calls.
    torch.set_num_threads(4)
    if args.check:
        df = load_surface(args.split)
        fingerprint, _ = source_fingerprint(args.split)
        print(f'{args.split}: {df.question_id.nunique()} queries, {len(df)} candidates; source fingerprint {fingerprint}')
        return
    if args.smoke_limit < 1:
        parser.error('--smoke-limit must be positive')
    out = args.output_dir or REPO_ROOT / 'results/week8/w8d_attribution' / ('smoke' if args.smoke else 'full')
    result = run_feature_extraction(args.split, out, device=args.device,
                                    smoke_limit=args.smoke_limit if args.smoke else None)
    print(f'Saved {len(result)} measured candidate rows to {out}')


if __name__ == '__main__':
    main()
