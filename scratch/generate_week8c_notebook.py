"""Generate the W8-C notebook; never substitute synthetic data or untrained scores."""
import json
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
NOTEBOOK_PATH = REPO_ROOT / 'notebooks/week8/week8c_utility_modeling.ipynb'


def build_notebook():
    cells = []
    def cell(kind, text):
        c = dict(cell_type=kind, metadata={}, source=text.strip()+'\n')
        if kind == 'code': c.update(execution_count=None, outputs=[])
        cells.append(c)
    def section(number, title, code, note=''):
        cell('markdown', f'## SECTION {number} — {title}\n\n{note}')
        cell('code', code)
    cell('markdown', '''# W8-C: Real passage utility modeling

Run with the **sara_env** kernel, from top to bottom. The first run generates the
missing real FIT promotion surface: **1,746 queries × 6 SARA generations**. This
stage uses the TITAN RTX and resumes its cache if interrupted. Release the SARA
model from other notebook kernels before starting. Existing INTERNAL-VAL results
are preserved. There are no synthetic-data fallbacks.

U0, U1, U2-F, U2-A, U3 and U4-A are trained and compared using paper-grouped FIT
OOF predictions. Only the selected model is refitted on all FIT and evaluated on
INTERNAL-VAL. This is held out from fitting but has been inspected in earlier
development; it is not a pristine final test set. TEST remains untouched. U3 here is the lightweight joint encoder variant, not Mistral QLoRA.
U4-B awaits measured W8-D attribution features and is explicitly skipped.

Reported downstream F1 uses an **oracle gate** to isolate passage selection; it
is not the complete deployed DAFiR controller result. STOP calibration is W8-E.
W8-B token uncertainty is excluded from both splits until genuine FIT diagnostics
are available; all models see the canonical base prediction as their draft.''')
    section(1, 'Configuration and real FIT preparation', '''import sys, json, subprocess, gc, hashlib
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from IPython.display import display

# Discover the repository from the notebook or repository working directory.
REPO_ROOT = next(p for p in [Path.cwd(), *Path.cwd().parents]
                 if (p / 'configs/week8/w8_dafir.yaml').is_file())
sys.path.insert(0, str(REPO_ROOT))
from src.week8.utility_training import (
    TrainingConfig, load_surface, prepare_features, run_oof, fit_final, predict_final,
)
from src.week8.utility_dataset import get_non_leaking_feature_columns
from src.week8.data import load_config, assert_paper_disjoint
from src.week8.evaluation import compute_query_grouped_selector_metrics, compute_calibration_metrics
from src.week8.utility_models import (
    S0_Rank5Selector, S1_BM25Selector, S2_SemanticSelector,
    S3_DraftSimilaritySelector, S4_MarginalInfoSelector, OracleSelector,
)

PREPARE_MISSING_FIT = True
MODELS = ['U0', 'U1', 'U2-F', 'U2-A', 'U3', 'U4-A']
TRAIN = TrainingConfig(device='cuda' if torch.cuda.is_available() else 'cpu')
print('Python:', sys.executable, '| Device:', TRAIN.device)
print('Training settings:', TRAIN)
cfg = load_config()
fit_manifest = pd.read_csv(REPO_ROOT / cfg['splits']['fit_manifest'], dtype=str)
fit_path = REPO_ROOT / 'results/week8/surfaces/promotion_surface_fit.jsonl'
val_path = REPO_ROOT / 'results/week8/surfaces/promotion_surface_internal_val.jsonl'
assert val_path.is_file(), 'Canonical INTERNAL-VAL surface must already exist.'
# A partial append-safe FIT cache is resumable, never accepted as a full surface.
fit_rows = pd.read_json(fit_path, lines=True) if fit_path.exists() and fit_path.stat().st_size else pd.DataFrame()
complete = (not fit_rows.empty and len(fit_rows) == 6 * len(fit_manifest)
            and set(fit_rows.question_id.astype(str)) == set(fit_manifest.question_id))
if not complete:
    if not PREPARE_MISSING_FIT:
        raise FileNotFoundError('Real FIT surface missing/incomplete. Enable PREPARE_MISSING_FIT to generate it.')
    if not torch.cuda.is_available():
        raise RuntimeError('FIT SARA generation requires the GPU. Select the sara_env kernel with CUDA.')
    free, total = torch.cuda.mem_get_info()
    print(f'FIT generation required. Free GPU memory: {free/2**30:.1f}/{total/2**30:.1f} GiB', flush=True)
    if free < 18 * 2**30:
        raise RuntimeError('Free the SARA model in other notebook kernels before generating FIT (at least 18 GiB free required). Then rerun this cell.')
    command = [sys.executable, '-u', str(REPO_ROOT / 'scripts/week8/run_w8a_promotion_surface.py'),
               '--split', 'fit', '--full', '--surface-only', '--device', TRAIN.device]
    process = subprocess.Popen(command, cwd=REPO_ROOT, stdout=subprocess.PIPE,
                               stderr=subprocess.STDOUT, text=True, bufsize=1)
    try:
        for line in process.stdout:
            print(line, end='', flush=True)
        if process.wait() != 0:
            raise RuntimeError('FIT generation failed. Fix the error above and rerun; completed rows are cached.')
    except BaseException:
        process.terminate()
        process.wait()
        raise
print('Real FIT preparation complete. No model has been evaluated on INTERNAL-VAL.')''')
    section(2, 'Validate canonical FIT and split membership', '''df_fit = load_surface('fit')
val_manifest = pd.read_csv(REPO_ROOT / cfg['splits']['internal_val_manifest'], dtype=str)
assert_paper_disjoint(set(fit_manifest.paper_id), set(val_manifest.paper_id))
assert not set(fit_manifest.question_id) & set(val_manifest.question_id)
print(f'FIT: {len(df_fit)} candidates, {df_fit.question_id.nunique()} queries, {df_fit.paper_id.nunique()} papers')
print('Six unique candidates per query; canonical W7 k4 drafts and delta labels verified.')''')
    section(3, 'Build genuine semantic and draft features', '''df_fit = prepare_features(df_fit)
feature_cols = get_non_leaking_feature_columns(df_fit)
assert feature_cols and np.isfinite(df_fit[feature_cols].to_numpy(float)).all()
print('Approved pre-promotion features:', feature_cols)
print('Excluded: gold/evidence labels, promotion latency, promoted output and outcome metrics.')''')
    section(4, 'FIT target distribution', '''display(df_fit[['target_help_binary', 'target_hurt_binary', 'target_regret', 'target_mask_ll']].describe())
print('LL supervision rows:', int(df_fit.target_mask_ll.sum()), '/', len(df_fit))
print('Unavailable continuous LL targets are masked, never treated as observed zeros.')''')
    section(5, 'Heuristic floors and oracle ceiling on FIT', '''fit_metrics, oof_predictions = {}, {}
selectors = {'S0 Rank5': S0_Rank5Selector(), 'S1 BM25': S1_BM25Selector(),
             'S2 Semantic': S2_SemanticSelector(), 'S3 DraftSemantic': S3_DraftSimilaritySelector(),
             'S4 MIG': S4_MarginalInfoSelector(), 'Oracle': OracleSelector()}
for name, selector in selectors.items():
    fit_metrics[name] = compute_query_grouped_selector_metrics(df_fit, selector_obj=selector)
display(pd.DataFrame(fit_metrics).T)
# S0 and S2 should select the same candidate, because ranks are defined by MiniLM.
assert np.isclose(fit_metrics['S0 Rank5']['final_downstream_f1'], fit_metrics['S2 Semantic']['final_downstream_f1'])''')
    for num, name, title in [(6,'U0','U0 tabular paper-grouped OOF'),(7,'U1','U1 query–passage OOF'),
            (8,'U2-F','U2-F frozen draft encoder OOF'),(9,'U2-A','U2-A top-two-layer adaptation OOF'),
            (10,'U3','U3 joint six-candidate encoder OOF'),(11,'U4-A','U4-A learned encoder and numeric fusion OOF')]:
        section(num, title, f'''if {name!r} in MODELS:
    predictions, metrics = run_oof({name!r}, df_fit, feature_cols, TRAIN)
    oof_predictions[{name!r}] = predictions
    fit_metrics[{name!r}] = metrics
    display(pd.Series(metrics, name={name!r}))''',
            'Each fold starts from a fresh backbone/head and trains only on that fold’s FIT papers.')
    section(12, 'U4-B attribution status', '''print('U4-B skipped: no validated W8-D attribution features are available on both splits.')
print('No fabricated attribution scores or zero-imputed attribution experiment is reported.')''')
    section(13, 'Compare FIT OOF selectors', '''comparison = pd.DataFrame(fit_metrics).T.rename_axis('selector')
display(comparison.sort_values('selector_headroom_recovery', ascending=False))
print('Top-1 accepts every candidate tied for maximum true utility; query metrics weight queries equally.')''')
    section(14, 'Inspect oracle-gated headroom', '''import matplotlib.pyplot as plt
comparison['selector_headroom_recovery'].mul(100).plot.bar(figsize=(10,4), ylabel='Oracle-gated headroom recovery (%)')
plt.axhline(0, color='black', linewidth=0.5)
plt.tight_layout()
plt.show()
print('Actual baseline and oracle F1 are recomputed for this split; no hard-coded reference F1.')''')
    section(15, 'Select using FIT OOF only', '''eligible = [name for name in MODELS if name in oof_predictions]
assert eligible, 'Run at least one model OOF cell before selecting.'
best_name = max(eligible, key=lambda name: fit_metrics[name]['selector_headroom_recovery'])
print('Selected:', best_name, '| FIT OOF SHR:', fit_metrics[best_name]['selector_headroom_recovery'])
print('INTERNAL-VAL has not influenced model selection.')''')
    section(16, 'Refit selected model on full FIT', '''final_model = fit_final(best_name, df_fit, feature_cols, TRAIN)
print('Actual full-FIT training complete:', best_name)''')
    section(17, 'Evaluate selected model on INTERNAL-VAL', '''# Cache in this kernel so rerunning this cell does not repeat the evaluation.
if 'heldout_result' in globals():
    raise RuntimeError('Held-out evaluation already ran in this kernel. Inspect heldout_result; do not retune on it.')
df_val = prepare_features(load_surface('internal_val'))
assert set(feature_cols) <= set(df_val.columns)
val_predictions = predict_final(best_name, final_model, df_val, feature_cols, TRAIN)
heldout_result = compute_query_grouped_selector_metrics(
    df_val.assign(score=val_predictions['utility_score']), selector_scores_col='score')
display(pd.Series(heldout_result, name=f'{best_name}: held-out INTERNAL-VAL'))
calibration = None
if 'prob_help' in val_predictions:
    calibration = compute_calibration_metrics(df_val.target_help_binary.to_numpy(), val_predictions['prob_help'])
    print('HELP Brier:', calibration['brier_score'], '| ECE:', calibration['ece'])
    display(calibration['decile_table'])
else:
    print('U3 uses utility/action heads; no candidate HELP probability calibration is claimed.')''')
    section(18, 'Save measured results and reloadable selected model', '''from dataclasses import asdict
import joblib
out = REPO_ROOT / 'results/week8/w8c_utility'
out.mkdir(parents=True, exist_ok=True)
# Persist the actual selected model, including tokenizer/backbone configuration and scaler.
if best_name == 'U0':
    joblib.dump(final_model, out / 'selected_u0.joblib')
    model_file = 'selected_u0.joblib'
else:
    payload = {'model_name': best_name, 'training_config': asdict(TRAIN),
               'feature_cols': feature_cols, 'state_dict': {k:v.detach().cpu() for k,v in final_model.state_dict().items()}}
    if hasattr(final_model, 'scaler'):
        payload['scaler'] = final_model.scaler
    torch.save(payload, out / 'selected_neural.pt')
    model_file = 'selected_neural.pt'
comparison.to_csv(out / 'fit_oof_comparison.csv')
for name, pred in oof_predictions.items():
    pd.concat([df_fit[['question_id','paper_id','semantic_rank','candidate_passage_index']].reset_index(drop=True),
               pd.DataFrame(pred)], axis=1).to_csv(out / f'{name}_fit_oof_predictions.csv', index=False)
pd.concat([df_val[['question_id','paper_id','semantic_rank','candidate_passage_index']].reset_index(drop=True),
           pd.DataFrame(val_predictions)], axis=1).to_csv(out / 'selected_internal_val_predictions.csv', index=False)
if calibration is not None:
    calibration['decile_table'].to_csv(out / 'selected_calibration.csv', index=False)
frozen = {'selected_model': best_name, 'model_file': model_file, 'models_compared': eligible,
          'training_config': asdict(TRAIN), 'feature_columns': feature_cols,
          'fit_oof_metrics': fit_metrics[best_name], 'internal_val_metrics': heldout_result,
          'feature_policy': 'canonical draft + tabular + measured MiniLM MIG; no W8-B token diagnostics or W8-D attribution',
          'gate': 'oracle epsilon=0.01; deployed controller remains W8-E',
          'surface_sha256': {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in [fit_path, val_path]}}
(out / 'w8c_frozen_selector_config.json').write_text(json.dumps(frozen, indent=2))
print('Saved measured W8-C results and selected model to', out)
# Reload later: from src.week8.utility_training import load_selected_model
# name, model, columns, settings = load_selected_model(out, device=TRAIN.device)''')
    notebook = dict(cells=cells, metadata={'kernelspec': {'display_name':'sara_env', 'language':'python', 'name':'sara_env'},
                    'language_info':{'name':'python', 'version':'3.11'}}, nbformat=4, nbformat_minor=4)
    NOTEBOOK_PATH.write_text(json.dumps(notebook, indent=2))
    print(f'Generated {NOTEBOOK_PATH} ({len(cells)} cells)')

if __name__ == '__main__': build_notebook()
