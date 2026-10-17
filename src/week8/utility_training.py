"""Real W8-C training: FIT-only paper-grouped OOF, then one held-out evaluation.

Every optimizer step contains exactly one question's six candidates. No label or
post-promotion measurement enters the text encoder or numeric feature vector.
"""
from __future__ import annotations

import gc
import json
import random
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.preprocessing import StandardScaler
from tqdm.auto import tqdm

from src.week6.evaluation import compute_qa_answer_metrics
from src.week8.data import PROJECT_ROOT, assert_paper_disjoint, load_config
from src.week8.evaluation import compute_query_grouped_selector_metrics
from src.week8.utility_dataset import (
    build_utility_dataset, get_non_leaking_feature_columns, get_paper_grouped_splits,
)
from src.week8.utility_models import (
    CrossEncoderUtilityModel, FullHybridUtilityModelU4, MultitaskUtilityLoss,
    PairwiseRankingLoss, truncate_and_budget_tokens, train_u0_paper_grouped_oof,
)
from src.week8.listwise_utility_model import ListwiseDraftUtilityModelU3


@dataclass
class TrainingConfig:
    n_splits: int = 5
    epochs: int = 3
    learning_rate: float = 2e-4
    backbone_learning_rate: float = 2e-5
    seed: int = 42
    max_length: int = 512
    model_name: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"
    device: str = "cuda"


def validate_surface(df, manifest, canonical=None):
    """Reject partial caches, incorrect split membership, and invalid labels."""
    required = {"question_id", "paper_id", "semantic_rank", "candidate_passage_index",
                "candidate_text", "question", "base_prediction", "base_f1",
                "promoted_f1", "delta_f1", "semantic_score"}
    if required - set(df):
        raise ValueError(f"Missing surface columns: {sorted(required - set(df))}")
    df = df.copy()
    for col in ("question_id", "paper_id"):
        df[col] = df[col].astype(str)
    manifest = manifest.astype({"question_id": str, "paper_id": str})
    if set(df.question_id) != set(manifest.question_id):
        raise ValueError("Surface is incomplete or contains queries outside its split manifest")
    if df.duplicated(["question_id", "semantic_rank"]).any():
        raise ValueError("Duplicate candidate ranks in surface")
    papers = manifest.set_index("question_id").paper_id
    if not df.paper_id.eq(df.question_id.map(papers)).all():
        raise ValueError("Surface paper IDs disagree with manifest")
    if not np.isfinite(df[["base_f1", "promoted_f1", "delta_f1"]].to_numpy(float)).all():
        raise ValueError("Non-finite utility labels")
    if not np.allclose(df.promoted_f1 - df.base_f1, df.delta_f1, atol=1e-8):
        raise ValueError("Delta labels disagree with promoted minus base F1")
    for qid, g in df.groupby("question_id"):
        if set(g.semantic_rank) != set(range(5, 11)) or len(g) != 6:
            raise ValueError(f"{qid}: expected exactly semantic ranks 5..10")
        if g.candidate_passage_index.nunique() != 6 or not g.candidate_passage_index.between(0, 9).all():
            raise ValueError(f"{qid}: invalid original passage indices")
        if g.base_f1.nunique() != 1 or g.base_prediction.nunique() != 1:
            raise ValueError(f"{qid}: inconsistent base across counterfactuals")
        if canonical is not None:
            ref = canonical[qid]
            if str(g.iloc[0].base_prediction).strip() != str(ref['prediction']).strip() or not np.isclose(g.iloc[0].base_f1, ref['token_f1'], atol=1e-8):
                raise ValueError(f"{qid}: base differs from canonical W7 k4")
    return df.reset_index(drop=True)


def load_surface(split, root=PROJECT_ROOT):
    if split not in ("fit", "internal_val"):
        raise ValueError("W8-C accepts FIT and INTERNAL-VAL only")
    cfg = load_config()
    manifest = pd.read_csv(root / cfg['splits'][f'{split}_manifest'], dtype=str)
    path = root / 'results/week8/surfaces' / f'promotion_surface_{split}.jsonl'
    if not path.is_file():
        raise FileNotFoundError(f"Missing real {split.upper()} surface: {path}. Run the notebook preparation cell.")
    canonical = {}
    with (root / 'results/week7/w7b_semantic_budget/response_surface/train_semantic_fixed_k_results.jsonl').open() as f:
        for line in f:
            r = json.loads(line)
            if int(r.get('k', 0)) == 4:
                # Re-score canonical text with the current scorer. W7 historical metrics
                # include one known scoring discrepancy; keep W7 files untouched.
                r['token_f1'] = compute_qa_answer_metrics(r['prediction'], r['gold_answers'],
                    r.get('gold_answers_reformatted', r['gold_answers']))['token_f1']
                canonical[str(r['question_id'])] = r
    df = validate_surface(pd.read_json(path, lines=True), manifest, canonical)
    for row in df.itertuples():
        observed = compute_qa_answer_metrics(row.promoted_prediction, row.gold_answers, row.gold_answers)['token_f1']
        if not np.isclose(observed, row.promoted_f1, atol=1e-8):
            raise ValueError(f'{row.question_id}: promoted F1 differs from the current scorer')
    # The canonical base prediction is the actual draft used by every model.
    # W8-B token diagnostics are intentionally excluded until available on BOTH splits.
    return build_utility_dataset(df)


def _validate_saved_routing(group, passages, semantic_scores, score_atol=1e-5):
    """Retain generation-time routing while allowing CPU/GPU near-tie drift.

    The six saved candidate indices define the original FULL set by their
    complement. Re-sorting fresh scores can change that set for tied passages.
    """
    qid = str(group.iloc[0].question_id)
    ranked = group.sort_values('semantic_rank')
    indices = ranked.candidate_passage_index.to_numpy(int)
    scores = np.asarray(semantic_scores, dtype=float)
    if (len(passages) != 10 or scores.shape != (10,) or
            not np.isfinite(scores).all() or
            ranked.semantic_rank.tolist() != list(range(5, 11)) or
            len(set(indices)) != 6 or np.any(indices < 0) or np.any(indices >= 10)):
        raise ValueError(f'{qid}: invalid saved candidate routing')
    for row in ranked.itertuples():
        if str(row.candidate_text).strip() != passages[int(row.candidate_passage_index)].strip():
            raise ValueError(f'{qid}: candidate text does not match its saved passage index')
    if not np.allclose(ranked.semantic_score.to_numpy(float), scores[indices], rtol=0, atol=score_atol):
        raise ValueError(f'{qid}: saved MiniLM scores differ from the current scorer')
    full_indices = sorted(set(range(10)) - set(indices))
    candidate_scores = scores[indices]
    if (np.any(np.diff(candidate_scores) > score_atol) or
            scores[full_indices].min() < candidate_scores.max() - score_atol):
        raise ValueError(f'{qid}: saved candidate ranks disagree with MiniLM routing')
    return full_indices


def prepare_features(df, root=PROJECT_ROOT, scorer=None):
    """Compute genuine semantic draft similarity and MIG from pre-promotion text.

    Fixed alpha=beta=0.5; no held-out labels used to tune these features.
    """
    from src.week8.semantic import SemanticScorer
    from src.week8.marginal_information import compute_mig_features_for_query
    scorer = scorer or SemanticScorer(device='cpu')
    cfg = load_config()
    wanted = set(df.question_id.astype(str))
    records = {}
    with (root / cfg['splits']['train_retrieval']).open() as f:
        for line in f:
            r = json.loads(line)
            if str(r['question_id']) in wanted:
                records[str(r['question_id'])] = r
    result = df.copy()
    for qid, g in tqdm(result.groupby('question_id'), desc='Semantic/MIG features'):
        r = records[str(qid)]
        passages = [p['text'] for p in r['passages'][:10]]
        sims = scorer.score_query_passages(r['question'], passages).cpu().numpy()
        full_indices = _validate_saved_routing(g, passages, sims)
        draft = str(g.iloc[0].draft_answer)
        cand_map = {int(row.semantic_rank): str(row.candidate_text) for _, row in g.iterrows()}
        feats = compute_mig_features_for_query(r['question'], draft,
                    [passages[i] for i in full_indices], cand_map, scorer)
        for idx, row in g.iterrows():
            for name, val in feats[int(row.semantic_rank)].items():
                result.loc[idx, name] = val
        words = draft.lower().split()
        result.loc[g.index, 'draft_length_words'] = len(words)
        result.loc[g.index, 'draft_empty'] = int(not words)
        qw, dw = set(r['question'].lower().split()), set(words)
        result.loc[g.index, 'draft_query_overlap'] = len(qw & dw) / max(len(qw | dw), 1)
    return result


def assert_training_split(df, root=PROJECT_ROOT):
    cfg = load_config()
    manifest = pd.read_csv(root / cfg['splits']['fit_manifest'], dtype=str)
    allowed = manifest.set_index('question_id').paper_id.to_dict()
    if not all(allowed.get(str(r.question_id)) == str(r.paper_id) for r in df.itertuples()):
        raise ValueError('Training data must be drawn exclusively from the FIT manifest')


class HybridSelector(torch.nn.Module):
    """Train U2 representation and a real numeric fusion head together."""
    def __init__(self, cfg, feature_cols, scaler):
        super().__init__()
        self.encoder = CrossEncoderUtilityModel(model_name=cfg.model_name, mode='U2', freeze_backbone=True)
        self.fusion = FullHybridUtilityModelU4(
            cross_encoder_dim=self.encoder.backbone.config.hidden_size,
            numeric_dim=len(feature_cols))
        # Encoder utility heads are unused; only fusion heads are optimized.
        for p in self.encoder.heads.parameters():
            p.requires_grad = False
        self.tokenizer = self.encoder.tokenizer
        self.feature_cols, self.scaler = feature_cols, scaler
        self.mode = 'U2'

    def forward(self, numeric_features, **encoded):
        return self.fusion(self.encoder(**encoded)['ce_embedding'], numeric_features)

    def predict_scores(self, df_query):
        return predict_neural(self, df_query, self.training_config)['utility_score']


def listwise_text(tokenizer, group, max_length=512):
    """Allocate a separate token budget to every candidate, retaining rank 10."""
    def cut(text, n):
        return tokenizer.decode(tokenizer.encode(str(text), add_special_tokens=False)[:n], skip_special_tokens=True)
    q = cut(group.iloc[0].question, 64)
    d = cut(group.iloc[0].draft_answer, 64)
    wrappers = '[QUESTION] ' + q + ' [DRAFT] ' + d
    overhead = len(tokenizer.encode(wrappers + ' '.join(f'[CANDIDATE {r}]' for r in range(5,11)), add_special_tokens=True)) + 12
    budget = (max_length - overhead) // 6
    if budget < 8:
        raise ValueError('Listwise context budget cannot accommodate all six candidates')
    text = wrappers
    for _, row in group.sort_values('semantic_rank').iterrows():
        text += f' [CANDIDATE {int(row.semantic_rank)}] ' + cut(row.candidate_text, budget)
    # Re-tokenization can add boundaries: reduce every candidate equally if necessary.
    if len(tokenizer.encode(text, add_special_tokens=True)) > max_length:
        return listwise_text(tokenizer, group, max_length - 12)
    return text


def encode_group(model, group, cfg):
    if isinstance(model, ListwiseDraftUtilityModelU3):
        text = [listwise_text(model.tokenizer, group, cfg.max_length)]
    else:
        mode = model.mode
        text = [truncate_and_budget_tokens(model.tokenizer, row.question,
                    '' if mode == 'U1' else row.draft_answer,
                    row.candidate_text, max_length=cfg.max_length)[0]
                for row in group.itertuples()]
    enc = model.tokenizer(text, padding=True, truncation=True, max_length=cfg.max_length, return_tensors='pt')
    if isinstance(model, ListwiseDraftUtilityModelU3):
        enc.pop('token_type_ids', None)
    if isinstance(model, HybridSelector):
        enc['numeric_features'] = torch.tensor(model.scaler.transform(group[model.feature_cols]), dtype=torch.float32)
    return dict(enc)


def make_model(name, cfg, train_df, feature_cols):
    if name == 'U3':
        model = ListwiseDraftUtilityModelU3(base_model_name=cfg.model_name, backbone_type='encoder')
        for p in model.model.parameters():
            p.requires_grad = False
        for layer in model.model.encoder.layer[-2:]:
            for p in layer.parameters():
                p.requires_grad = True
    elif name == 'U4-A':
        scaler = StandardScaler().fit(train_df[feature_cols])
        model = HybridSelector(cfg, feature_cols, scaler)
    elif name in ('U1', 'U2-F', 'U2-A'):
        model = CrossEncoderUtilityModel(model_name=cfg.model_name,
                 mode='U1' if name == 'U1' else 'U2', freeze_backbone=name != 'U2-A',
                 adapted_top_layers=2 if name == 'U2-A' else 0)
    else:
        raise ValueError(f'Unknown neural model: {name}')
    model.training_config = cfg
    return model.to(cfg.device)


def targets_for(group, device):
    return {c: torch.tensor(group[c].to_numpy(), dtype=torch.long if c == 'target_3way_label' else torch.float32, device=device)
            for c in ('target_help_binary', 'target_hurt_binary', 'target_continuous_delta',
                      'target_regret', 'target_dense_gold_ll', 'target_mask_ll', 'target_3way_label')}


def fit_neural(name, train_df, feature_cols, cfg):
    assert_training_split(train_df)
    random.seed(cfg.seed); np.random.seed(cfg.seed); torch.manual_seed(cfg.seed)
    model = make_model(name, cfg, train_df, feature_cols)
    slow, fast = [], []
    for key, p in model.named_parameters():
        if p.requires_grad:
            (slow if 'backbone.' in key or key.startswith('model.') else fast).append(p)
    optimizer = torch.optim.AdamW([
        {'params': slow, 'lr': cfg.backbone_learning_rate},
        {'params': fast, 'lr': cfg.learning_rate}], weight_decay=0.01)
    loss_fn = MultitaskUtilityLoss()
    groups = [g.sort_values('semantic_rank') for _, g in train_df.groupby('question_id')]
    # Tokenize once per fold; retain CPU tensors, move one query at a time to GPU.
    inputs = [encode_group(model, g, cfg) for g in groups]
    rng = np.random.default_rng(cfg.seed)
    for epoch in range(cfg.epochs):
        model.train()
        # Frozen backbones must not inject random dropout into their representations.
        backbone = model.encoder.backbone if isinstance(model, HybridSelector) else getattr(model, 'backbone', None)
        if backbone is not None and not any(p.requires_grad for p in backbone.parameters()):
            backbone.eval()
        running = []
        for i in tqdm(rng.permutation(len(groups)), desc=f'{name} epoch {epoch+1}/{cfg.epochs}'):
            enc = {k: v.to(cfg.device) for k, v in inputs[i].items()}
            optimizer.zero_grad(set_to_none=True)
            pred = model(**enc)
            t = targets_for(groups[i], cfg.device)
            if name == 'U3':
                delta = t['target_continuous_delta']
                scores = pred['utility_scores'][0]
                action = 0 if delta.max() <= 0.01 else 1 + int(delta.argmax())
                loss = (torch.nn.functional.mse_loss(scores, delta)
                      + PairwiseRankingLoss()(scores, delta)
                      + torch.nn.functional.cross_entropy(pred['action_logits'], torch.tensor([action], device=cfg.device)))
            else:
                loss = loss_fn(pred, t)['loss_total']
            if not torch.isfinite(loss):
                raise ValueError(f'{name}: non-finite training loss')
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            running.append(float(loss.detach()))
        print(f'{name} epoch {epoch+1}: training loss={np.mean(running):.5f}', flush=True)
    model.eval()
    return model


def predict_neural(model, df, cfg):
    result = {'utility_score': np.zeros(len(df))}
    if not isinstance(model, ListwiseDraftUtilityModelU3):
        result.update(prob_help=np.zeros(len(df)), prob_hurt=np.zeros(len(df)))
    model.eval()
    work = df.reset_index(drop=True)
    with torch.no_grad():
        for _, raw in work.groupby('question_id'):
            group = raw.sort_values('semantic_rank')
            enc = {k: v.to(cfg.device) for k,v in encode_group(model, group, cfg).items()}
            pred = model(**enc)
            for key in result:
                vals = pred['utility_scores'][0] if isinstance(model, ListwiseDraftUtilityModelU3) else pred[key]
                result[key][group.index] = vals.detach().cpu().numpy()
    return result


def run_oof(name, df, feature_cols, cfg):
    assert_training_split(df)
    if cfg.epochs < 1 or cfg.n_splits < 2:
        raise ValueError('Need positive epochs and at least two OOF folds')
    if name == 'U0':
        predictions, final = train_u0_paper_grouped_oof(df, feature_cols, cfg.n_splits)
        del final
    else:
        predictions = None
        for fold, (tr, va) in enumerate(get_paper_grouped_splits(df, cfg.n_splits)):
            train, val = df.iloc[tr], df.iloc[va]
            assert_paper_disjoint(set(train.paper_id), set(val.paper_id))
            print(f'{name}: paper fold {fold+1}/{cfg.n_splits}', flush=True)
            model = fit_neural(name, train, feature_cols, cfg)
            pred = predict_neural(model, val, cfg)
            if predictions is None:
                predictions = {k: np.full(len(df), np.nan) for k in pred}
            for k, v in pred.items():
                predictions[k][va] = v
            del model; gc.collect()
            if torch.cuda.is_available(): torch.cuda.empty_cache()
    if not all(np.isfinite(v).all() for v in predictions.values()):
        raise ValueError('Incomplete/non-finite OOF predictions')
    scored = df.assign(score=predictions['utility_score'])
    metrics = compute_query_grouped_selector_metrics(scored, selector_scores_col='score')
    return predictions, metrics


def fit_final(name, df, feature_cols, cfg):
    assert_training_split(df)
    if name == 'U0':
        from src.week8.utility_models import UtilityModelU0
        model = UtilityModelU0()
        model.fit(df[feature_cols].to_numpy(), df.target_help_binary.to_numpy(),
                  df.target_hurt_binary.to_numpy(), df.target_continuous_delta.to_numpy(),
                  df.target_3way_label.to_numpy(), y_regret=df.target_regret.to_numpy(), feature_cols=feature_cols)
        return model
    return fit_neural(name, df, feature_cols, cfg)


def predict_final(name, model, df, feature_cols, cfg):
    if name == 'U0':
        return model.predict(df[feature_cols].to_numpy())
    return predict_neural(model, df, cfg)


def load_selected_model(directory, device='cpu'):
    """Reload a locally produced notebook/CLI selector and its preprocessing."""
    import joblib
    directory = Path(directory)
    config_path = directory / 'w8c_frozen_selector_config.json'
    if not config_path.exists(): config_path = directory / 'run_summary.json'
    saved = json.loads(config_path.read_text())
    name = saved['selected_model']
    cfg = TrainingConfig(**{**saved['training_config'], 'device':device})
    features = saved['feature_columns']
    if name == 'U0':
        return name, joblib.load(directory / 'selected_u0.joblib'), features, cfg
    # These checkpoints are created locally by this workflow, including its fitted scaler.
    payload = torch.load(directory / 'selected_neural.pt', map_location=device, weights_only=False)
    if payload['model_name'] != name or payload['feature_cols'] != features:
        raise ValueError('Saved model and selector metadata disagree')
    if name == 'U4-A':
        model = HybridSelector(cfg, features, payload['scaler']).to(device)
        model.training_config = cfg
    else:
        model = make_model(name, cfg, None, features)
    model.load_state_dict(payload['state_dict'])
    model.eval()
    return name, model, features, cfg
