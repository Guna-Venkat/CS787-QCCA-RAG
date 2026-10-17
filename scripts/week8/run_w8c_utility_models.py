"""W8-C runner: train on FIT, choose by FIT OOF, evaluate winner on INTERNAL-VAL."""
from __future__ import annotations
import argparse
import json
import sys
from dataclasses import asdict
from pathlib import Path

import pandas as pd
import torch

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))
from src.week8.utility_training import (
    TrainingConfig, load_surface, prepare_features, run_oof, fit_final, predict_final,
)
from src.week8.utility_dataset import get_non_leaking_feature_columns
from src.week8.evaluation import compute_query_grouped_selector_metrics


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--check', action='store_true', help='Check canonical data readiness without training')
    parser.add_argument('--split', choices=['fit', 'internal_val'], default='internal_val',
                        help='fit: FIT OOF only; internal_val: additionally evaluate selected full-FIT model')
    parser.add_argument('--models', nargs='+', choices=['U0','U1','U2-F','U2-A','U3','U4-A'],
                        default=['U0','U1','U2-F','U2-A','U3','U4-A'])
    parser.add_argument('--smoke', action='store_true', help='Real FIT subset; writes only to smoke directory, never evaluates held-out data')
    parser.add_argument('--epochs', type=int, default=3)
    parser.add_argument('--folds', type=int, default=5)
    parser.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu')
    args = parser.parse_args()
    if args.check:
        # Validate the held-out surface structurally, without fitting or scoring it.
        for split in ['internal_val','fit']:
            try:
                df = load_surface(split)
                print(f'{split}: canonical surface verified ({df.question_id.nunique()} queries / {len(df)} candidates)')
            except (FileNotFoundError, ValueError, KeyError) as exc:
                print(f'{split}: NOT READY: {exc}')
                if split == 'fit':
                    print('The W8-C notebook preparation cell generates/resumes this real FIT surface.')
                return 2
        return 0
    df = load_surface('fit')  # Hard failure if missing: never fabricate FIT or train on VAL.
    if args.smoke:
        papers = df.paper_id.drop_duplicates().head(3)
        qids = df[df.paper_id.isin(papers)].groupby('paper_id').question_id.first()
        df = df[df.question_id.isin(qids)].reset_index(drop=True)
    df = prepare_features(df)
    features = get_non_leaking_feature_columns(df)
    cfg = TrainingConfig(n_splits=2 if args.smoke else args.folds,
                         epochs=1 if args.smoke else args.epochs, device=args.device)
    metrics, predictions = {}, {}
    for name in args.models:
        predictions[name], metrics[name] = run_oof(name, df, features, cfg)
        print(name, metrics[name], flush=True)
    best = max(metrics, key=lambda name: metrics[name]['selector_headroom_recovery'])
    out = REPO_ROOT / 'results/week8/w8c_utility' / ('smoke' if args.smoke else 'cli')
    out.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(metrics).T.to_csv(out / 'fit_oof_comparison.csv')
    result = {'selected_model':best, 'training_config':asdict(cfg), 'feature_columns':features,
              'fit_oof_metrics':metrics[best], 'smoke':args.smoke}
    if args.split == 'internal_val' and not args.smoke:
        model = fit_final(best, df, features, cfg)
        val = prepare_features(load_surface('internal_val'))
        pred = predict_final(best, model, val, features, cfg)
        result['internal_val_metrics'] = compute_query_grouped_selector_metrics(
            val.assign(score=pred['utility_score']), selector_scores_col='score')
        if best == 'U0':
            import joblib
            joblib.dump(model, out / 'selected_u0.joblib')
        else:
            payload = {'model_name':best, 'training_config':asdict(cfg), 'feature_cols':features,
                       'state_dict':{k:v.detach().cpu() for k,v in model.state_dict().items()}}
            if hasattr(model, 'scaler'): payload['scaler'] = model.scaler
            torch.save(payload, out / 'selected_neural.pt')
        print('Held-out:', result['internal_val_metrics'])
    (out / 'run_summary.json').write_text(json.dumps(result, indent=2))
    print('Saved measured results to', out)
    return 0

if __name__ == '__main__':
    raise SystemExit(main())
