"""Manual phase orchestration. No phase automatically invokes another phase."""
from __future__ import annotations
import argparse
from collections import defaultdict
import json
from pathlib import Path
import random
import time
import numpy as np
import pandas as pd
from .provenance import ROOT, digest, file_hash, branch_guard, source_hashes
from .config import load_config, load_frozen, check_test, PREREQUISITES
from .caching import RowCache, read_jsonl, atomic_json
from .data import resolve, load_records, source_document, evaluation_evidence
from .manifests import manifest, freeze
from .retrieval import LocalEncoder, rank_candidates, unit_vectors
from .chunking import locate_passages, original_chunks, fixed_chunks
from .semantic_chunking import semantic_chunks
from .evidence_alignment import retrieval_metrics
from .selection import select, validate_selection
from .cost_accounting import QueryCost
from .extraction import extract
from .revision import revise, revision_messages, normalize_mode, read_revision_mode, write_revision_freeze
from .backend import SaraBackend, CachedClient, checkpoint_hashes, load_tokenizer
from .evaluation import answer_f1, annotate_outcome, policy_summary, choose_global, paired_paper_bootstrap, selector_diagnostics
from .ke_surface import surface_analysis
from .adaptive_ke import online_features, headroom_gate, training_actions, train_oof, sequential_stop

FOLDERS = dict(a='retrieval', b='ke_surface', c='selection', d='extraction',
               e='faithfulness', f='revision', g='controller', h='frontier')
FILENAMES = dict(a='w9a_retrieval_results', b='w9b_ke_surface', c='w9c_selection',
                 d='w9d_representation', e='w9e_faithfulness', f='w9f_revision',
                 g='w9g_adaptive', h='w9h_final')


def frozen_path(cfg, phase):
    return resolve(cfg['output_root']) / 'frozen' / f'w9{phase}_frozen_config.json'


def phase_dir(cfg, phase, split, smoke=False):
    p = resolve(cfg['output_root']) / FOLDERS[phase] / split
    return p / 'smoke' if smoke else p


def rows_path(cfg, phase, split, smoke=False):
    return phase_dir(cfg, phase, split, smoke) / (FILENAMES[phase] + '.jsonl')


def load_phase(cfg, phase, split):
    path = rows_path(cfg, phase, split)
    rows = read_jsonl(path)
    meta = json.loads(path.with_suffix('.identity.json').read_text())
    if meta['source_hashes'] != source_hashes():
        raise ValueError(f'Stale W9-{phase.upper()} source fingerprints')
    from .provenance import verify_sources
    verify_sources(meta['source_hashes'])
    for name, value in meta['input_hashes'].items():
        if file_hash(name) != value:
            raise ValueError('Upstream input fingerprint changed')
    if any(r['cache_fingerprint'] != digest(meta) for r in rows):
        raise ValueError('Saved result identity mismatch')
    if any(r.get('row_content_hash') != digest({k:v for k,v in r.items() if k!='row_content_hash'}) for r in rows):
        raise ValueError('Saved result content changed')
    return rows


def dependencies(cfg, phase, split):
    frozen = {}
    needed=set()
    def visit(current):
        for dep in PREREQUISITES[current]:
            visit(dep)
            needed.add(dep)
    visit(phase)
    for dependency in sorted(needed):
        frozen[dependency] = load_frozen(frozen_path(cfg, dependency), dependency)
    if split != 'fit' and not (phase == 'h' and split == 'fit_heldout'):
        frozen[phase] = load_frozen(frozen_path(cfg, phase), phase)
    for artifact in frozen.values():
        if any(cfg.get(k) != value for k,value in artifact['protocol'].items()):
            raise ValueError('Backend/decoding/controlled scoring differs from FIT-frozen components')
    return frozen


def encoder_fingerprints(cfg):
    from huggingface_hub import snapshot_download
    path = Path(snapshot_download(cfg['encoder'], local_files_only=True))
    files = [p for p in path.rglob('*') if p.is_file() and p.suffix in ('.safetensors','.bin','.json','.txt')]
    if not files:
        raise FileNotFoundError('Missing local semantic model files')
    return {str(p): file_hash(p) for p in files}


def build_identity(cfg, phase, split, seed, paths, frozen, artifact_split=None):
    inputs = [p for name, p in paths.items() if name != 'sfr']
    if phase != 'a':
        inputs.append(paths['sfr'])
    inputs += [frozen_path(cfg, name) for name in frozen]
    for name in PREREQUISITES[phase]:
        # Prerequisite phase outcomes on this split are required except G/H's
        # analysis artifacts (FIT training is not re-run on held-out data).
        p = rows_path(cfg, name, artifact_split or split)
        if name not in ('g',) and p.exists():
            inputs.append(p)
    if phase=='h':
        from .baseline_registry import baseline_source_paths
        inputs.extend(baseline_source_paths())
        inputs.extend(resolve(p) for p in cfg.get('generator_robustness_exports',[]))
        if cfg.get('generator_robustness_exports'):
            inputs.append(resolve(cfg.get('robustness_panel','results/week9/preflight/frozen_model_panel.json')))
    if cfg.get('textual_model_roles'):
        inputs.append(resolve(cfg['textual_model_roles']['frozen_panel']))
    models = checkpoint_hashes(cfg)
    models.update(encoder_fingerprints(cfg))
    for artifact in frozen.values():
        if artifact['model_hashes'] != models:
            raise ValueError('Model/tokenizer fingerprints differ from frozen upstream components')
    return dict(schema='week9-v1', phase=phase, split=split, seed=seed,
        config_hash=digest(cfg), source_hashes=source_hashes(), model_hashes=models,
        prompt_version_hash=digest(source_hashes()),
        input_hashes={str(p.resolve()): file_hash(p) for p in inputs})


def cached_embeddings(chunks, encoder, cache):
    vectors = []
    missing = []
    for chunk in chunks:
        key = cache.key('INDEX', chunk['paper_id'], [chunk['chunk_id']], 'chunk_embedding')
        if key not in cache.rows:
            missing.append((key,chunk))
    if missing:
        started = time.perf_counter()
        encoded = encoder.encode([c['text'] for _,c in missing])
        elapsed = time.perf_counter()-started
        for (key,chunk),vector in zip(missing,encoded):
            cache.put(key,dict(paper_id=chunk['paper_id'],chunk_id=chunk['chunk_id'],
                vector=np.asarray(vector).tolist(),index_embedding_seconds=elapsed/len(missing),
                timing_basis='Amortized batch embedding time; index-time only'))
    for chunk in chunks:
        key=cache.key('INDEX',chunk['paper_id'],[chunk['chunk_id']],'chunk_embedding')
        vectors.append(cache.rows[key]['vector'])
    return unit_vectors(vectors)


def run_a(cfg, records, paths, cache, directory, encoder, frozen, index_cache, chunk_cache):
    """Index first, FIT-match fixed token medians, then audit all retrieval widths."""
    documents = {}
    grid = cfg['semantic_grid'] if 'a' not in frozen else frozen['a']['settings']['semantic_grid']
    # Cache chunk manifests by query so query-specific original passage IDs survive.
    for record_index, record in enumerate(records):
        inf = record['inference']
        qid, pid = inf['question_id'], inf['paper_id']
        if record_index % 25 == 0:
            print(f'W9-A indexing {record_index+1}/{len(records)}: question {qid}',flush=True)
        document = source_document(record, paths['raw'])
        lineage = locate_passages(document, record['original_passages'])
        documents[qid] = (document, lineage)
        variants = [('ORIG', None)] + [('SEM-' + str(i), setting) for i, setting in enumerate(grid)]
        for name, setting in variants:
            key = chunk_cache.key(qid, pid, [], name)
            if key not in chunk_cache.rows:
                started=time.perf_counter()
                chunks = original_chunks(document, lineage, encoder.tokenizer) if setting is None else semantic_chunks(
                    document, lineage, encoder.tokenizer, encoder, **setting)
                chunk_cache.put(key, dict(question_id=qid, paper_id=pid, representation=name,
                    index_chunking_seconds=time.perf_counter()-started,
                    chunks=[dict(c, question_id=qid) for c in chunks]))
    lengths = defaultdict(list)
    # Do not let multiple questions from the same paper overweight matching sizes.
    seen = set()
    for row in chunk_cache.rows.values():
        ident = (row['paper_id'], row['representation'])
        if ident not in seen:
            lengths[row['representation']].extend(c['token_length'] for c in row['chunks'])
            seen.add(ident)
    sizes = ({name: max(1, int(np.median(values))) for name, values in lengths.items() if name.startswith('SEM-')}
             if 'a' not in frozen else frozen['a']['settings']['fixed_sizes'])
    for record in records:
        inf = record['inference']
        qid, pid = inf['question_id'], inf['paper_id']
        document, lineage = documents[qid]
        for name, size in sizes.items():
            variant = name.replace('SEM-', 'FIXED-')
            key = chunk_cache.key(qid, pid, [], variant)
            if key not in chunk_cache.rows:
                started=time.perf_counter()
                chunks = fixed_chunks(document, lineage, encoder.tokenizer, size, cfg['fixed_overlap'])
                chunk_cache.put(key, dict(question_id=qid, paper_id=pid, representation=variant,
                    index_chunking_seconds=time.perf_counter()-started,
                    chunks=[dict(c, question_id=qid) for c in chunks]))
        evidence = evaluation_evidence(record, paths['raw'])
        variants = [r for r in chunk_cache.rows.values() if r['question_id'] == qid]
        for row in variants:
            chunks = row['chunks']
            vectors = cached_embeddings(chunks, encoder, index_cache)
            embedding_seconds=sum(index_cache.rows[index_cache.key('INDEX',pid,[c['chunk_id']],'chunk_embedding')]['index_embedding_seconds'] for c in chunks)
            for method in ('BM25', 'SEMANTIC', 'RRF'):
                for width in cfg['retrieval_widths']:
                    operation = dict(representation=row['representation'], retriever=method, retrieval_width=width)
                    key = cache.key(qid, pid, [c['chunk_id'] for c in chunks], operation)
                    if key in cache.rows:
                        continue
                    started = time.perf_counter()
                    ranked = rank_candidates(inf['question'], pid, chunks, encoder, method, width, vectors)
                    elapsed = time.perf_counter()-started
                    metrics = retrieval_metrics(ranked, chunks, evidence)
                    cache.put(key, dict(question_id=qid, paper_id=pid, **operation,
                        retrieval_actual_count=len(ranked), available_chunks=len(chunks), candidates=ranked,
                        index_time_embedding_seconds=embedding_seconds,
                        index_time_chunking_seconds=row['index_chunking_seconds'],
                        retrieval_query_time=elapsed, **metrics))
    return dict(semantic_grid=grid, fixed_sizes=sizes)


def get_pool(cfg, frozen, split, qid):
    settings = frozen['a']['settings']
    rows = load_phase(cfg, 'a', split)
    found = [r for r in rows if r['question_id'] == qid and all(r[k] == settings[k] for k in
            ('representation','retriever','retrieval_width'))]
    if len(found) != 1:
        raise ValueError(f'{qid}: missing/ambiguous frozen W9-A candidate pool; run A on this split')
    return found[0]


def canonical_context(record, embeddings, tokenizer, device, dtype):
    from src.week7.w7e.prompt_builder import build_mixed_fidelity_sara_prompt, extract_mixed_fidelity_sara_embeddings
    import torch
    inf = record['inference']
    matrix = embeddings.get(inf['question_id'])
    if matrix is None or tuple(matrix.shape) != (10,4096) or not torch.isfinite(matrix).all():
        raise ValueError('Missing real finite canonical SFR embeddings')
    prompt, historical_count, compressed, _ = build_mixed_fidelity_sara_prompt(inf['question'],
        inf['passages'], inf['selected_full_indices'], [], inf['compressed_indices'], {}, tokenizer)
    if historical_count != record['canonical']['context_tokens'] or compressed != inf['compressed_indices']:
        raise ValueError('Canonical W8 prompt/token order changed')
    comp = extract_mixed_fidelity_sara_embeddings(matrix, compressed).to(device=device, dtype=dtype)
    tokens = len(tokenizer.encode(prompt))
    return prompt, comp, tokens


def interventions(phase, cfg, frozen):
    if phase == 'b':
        return [dict(policy=f'K_E={count}', selector='DraftSemantic', evidence_k=count, representation='raw', revision=False)
                for count in cfg['evidence_counts']]
    count = frozen['b']['settings']['evidence_k']
    selector = frozen['c']['settings']['selector'] if 'c' in frozen else None
    if phase == 'c':
        return [dict(policy=s, selector=s, evidence_k=count, representation='raw', revision=False) for s in cfg['selectors']]
    if phase == 'd':
        return [dict(policy='D'+str(i), selector=selector, evidence_k=0 if i==0 else count,
            representation=r, revision=False) for i, r in enumerate(('none','raw','query','draft'))]
    if phase == 'e':
        return [dict(policy='E'+str(i), selector=selector, evidence_k=count, representation='draft',
            format=f, revision=False) for i, f in enumerate(('extractive','structured','sane_blueprint'))]
    if phase == 'f':
        return [dict(policy=f'F{i}-K_E={k}', selector=selector, evidence_k=k, representation='draft',
            format=frozen['e']['settings']['format'], revision=f'F{i}') for i in (0,1,2) for k in (0,1,2,3,4)]
    if phase == 'h':
        result = [dict(policy='B5-SANE-adapted', selector='SANE-adapted', evidence_k=count,
                       representation='query', revision=False, sane_reference=True)]
        for i, r, revision in [(6,'raw',False),(7,'query',False),(8,'draft',False),(9,'draft',read_revision_mode(cfg,frozen['f']))]:
            result.append(dict(policy='B'+str(i), selector=selector, evidence_k=count,
                representation=r, revision=revision,
                format=frozen['e']['settings']['format'] if i==9 else 'sane_blueprint'))
        for mode in ('F0','F1','F2'):
            result.append(dict(policy='W9-'+mode,selector=selector,evidence_k=count,representation='draft',format=frozen['e']['settings']['format'],revision=mode))
        return result
    raise ValueError(phase)


def run_intervention(record, pool, spec, encoder, backend, call_cache, embeddings, cfg, extraction_cache,
                     frozen_evidence=None, prepare_only=False):
    inf = record['inference']
    textual=getattr(backend,'text_only',False)
    if textual:
        import torch
        context_tokenizer=backend.canonical_tokenizer
        prompt,comp,tokens=canonical_context(record,embeddings,context_tokenizer,'cpu',torch.float16)
    else:
        context_tokenizer=backend.tokenizer
        prompt,comp,tokens=canonical_context(record,embeddings,backend.tokenizer,backend.device,backend.model.dtype)
    cost = QueryCost(initial_draft_prompt_tokens=tokens,
        initial_draft_generated_tokens=len(record['canonical'].get('saved_generated_token_ids',
            context_tokenizer.encode(inf['initial_draft'], add_special_tokens=False))),
        final_prompt_tokens=tokens, retrieval_query_time=pool['retrieval_query_time'])
    cost.index_time_embedding_cost=pool.get('index_time_embedding_seconds')
    selector_client = CachedClient(backend, call_cache, inf, cost, 'selector',evidence_ids=[c['chunk_id'] for c in pool['candidates']])
    started = time.perf_counter()
    # Project surfaces use one fixed ranked prefix to make sequential states
    # well defined. The separately labelled SANE baseline retains exact k.
    rank_budget = spec['evidence_k'] if spec.get('sane_reference') else 4
    selected = (select(inf, pool['candidates'], rank_budget, spec['selector'], encoder,
        selector_client, cfg['query_draft_alpha'])[:spec['evidence_k']] if spec['evidence_k'] else [])
    selection_seconds = time.perf_counter()-started
    validate_selection(pool['candidates'], selected, spec['evidence_k'])
    ids = [c['chunk_id'] for c in selected]
    result = dict(question_id=inf['question_id'], paper_id=inf['paper_id'], initial_full_k=4,
        retrieval_width=pool['retrieval_width'], retrieval_actual_count=pool['retrieval_actual_count'],
        policy=spec['policy'], specification=spec, evidence_k=spec['evidence_k'], evidence_actual_count=len(selected),
        selected_chunk_ids=ids, initial_draft=inf['initial_draft'], question=inf['question'],
        source_chunks=selected, extraction_rate=bool(selected) and spec['representation'] in ('query','draft'),
        promotion_rate=bool(selected), proposed_features=None)
    if not selected:
        result.update(prediction=inf['initial_draft'], evidence_text='', faithfulness=None)
    else:
        representation = spec['representation']
        if representation == 'raw':
            evidence = '\n\n'.join('['+c['chunk_id']+'] '+c['text'] for c in selected)
            result['faithfulness'] = 1.
        else:
            format_name = spec.get('format','sane_blueprint')
            key = extraction_cache.key(inf['question_id'], inf['paper_id'], ids,
                dict(draft_aware=representation=='draft', format=format_name))
            ext = extraction_cache.rows.get(key)
            ext_cost = QueryCost(num_lm_calls=0)
            client = CachedClient(backend, call_cache, inf, ext_cost, 'extraction',evidence_ids=ids)
            if frozen_evidence is not None:
                ext = frozen_evidence
                if ext['selected_chunk_ids'] != ids:
                    raise ValueError('Frozen extraction selected IDs changed')
            elif ext is None:
                value = extract(inf, selected, client, draft_aware=representation=='draft',
                    format_name=format_name, max_tokens=cfg['extraction_max_tokens'])
                ext = extraction_cache.put(key, dict(value, question_id=inf['question_id'], paper_id=inf['paper_id'],
                    extraction_prompt_tokens=ext_cost.extraction_prompt_tokens,
                    extraction_generated_tokens=ext_cost.extraction_generated_tokens,
                    extraction_call_count=ext_cost.num_lm_calls, extraction_latency=client.latency,model_calls=ext_cost.model_calls))
            cost.model_calls.extend(ext.get('model_calls',[]))
            cost.extraction_prompt_tokens += ext['extraction_prompt_tokens']
            cost.extraction_generated_tokens += ext['extraction_generated_tokens']
            cost.num_lm_calls += ext['extraction_call_count']
            cost.extraction_latency = ext['extraction_latency']
            evidence = ext['text']
            result['extraction'] = ext
            result['faithfulness'] = ext['audit']['exact_span_support_rate']
            result['extraction_compression_ratio'] = len(backend.tokenizer.encode(evidence,add_special_tokens=False))/max(sum(c['token_length'] for c in selected),1)
        # Reference-style SANE answer sees evidence only; project regeneration also
        # retains canonical compressed context to isolate the intervention.
        revision_client = CachedClient(backend, call_cache, inf, cost, 'revision',
            context=None if spec.get('sane_reference') or textual else prompt,
            retrieval_embeds=None if spec.get('sane_reference') or textual else comp,evidence_ids=ids,revision_mode=normalize_mode(spec['revision']))
        proposed_prompt_tokens = (len(backend.tokenizer.encode(revision_client.prepare_prompt(
            revision_messages(inf, evidence, spec['revision'])))) if evidence.strip() else 0)
        result['proposed_features'] = online_features(inf, selected[-1], encoder,
            len(selected)-1, proposed_prompt_tokens+cost.extraction_prompt_tokens)
        if prepare_only:
            cost.selection_latency = selector_client.latency if spec['selector']=='SANE-adapted' else selection_seconds
            result.update(evidence_text=evidence)
            return dict(result=result,cost=cost,prompt=prompt,comp=comp)
        prediction = revise(inf, evidence, revision_client, spec['revision'])
        cost.revision_latency = revision_client.latency if evidence.strip() else None
        result.update(prediction=prediction, evidence_text=evidence)
    cost.selection_latency = selector_client.latency if spec['selector']=='SANE-adapted' else selection_seconds
    result.update(cost.to_dict(), initial_draft_generated_tokens_basis=('saved canonical continuation IDs, including EOS'
        if 'saved_generated_token_ids' in record['canonical'] else 'retokenized frozen draft; continuation IDs unavailable'),
        latency_note='Cached stage timings; no measured end-to-end latency. Initial generation time excluded.')
    return annotate_outcome(result, inf, record['evaluation'])


def run_generation(cfg, phase, records, paths, frozen, cache, call_cache, extraction_cache):
    import torch
    # Load frozen A pools once, not once per query.
    retrieval_rows = load_phase(cfg, 'a', paths['split'])
    pools = retrieval_rows
    choice = frozen['a']['settings']
    pools = {r['question_id']: r for r in pools if all(r[k]==choice[k] for k in
             ('representation','retriever','retrieval_width'))}
    if set(r['inference']['question_id'] for r in records)-set(pools):
        raise ValueError('Missing frozen retrieval candidates for this split')
    backend, encoder = None, None
    embeddings = torch.load(paths['sfr'], map_location='cpu', weights_only=False)
    if 'embeddings' in embeddings:
        embeddings = embeddings['embeddings']
    specs = interventions(phase, cfg, frozen)
    frozen_extractions = {}
    if phase == 'f':
        frozen_extractions = {r['question_id']: r for r in load_phase(cfg,'e',paths['split'])
            if r['policy'] == frozen['e']['settings']['policy']}
    for record in records:
        inf = record['inference']
        for spec in specs:
            key = cache.key(inf['question_id'], inf['paper_id'], [], spec)
            if key in cache.rows:
                continue
            if backend is None:
                if cfg.get('textual_model_roles'):
                    from .text_model_roles import TextRoleBackend
                    backend=TextRoleBackend(cfg)
                else:backend=SaraBackend(cfg)
                encoder = LocalEncoder(cfg['encoder'],cfg['encoder_device'])
            # F0/F1 retrieve the SAME extraction artifact committed by E for the
            # chosen K, and otherwise commit once then reuse across F0/F1 prefixes.
            frozen_ext = None
            if phase == 'f' and spec['evidence_k'] == frozen['b']['settings']['evidence_k'] and spec['evidence_k']:
                source = frozen_extractions.get(inf['question_id'])
                if source is None:
                    raise ValueError('Missing frozen W9-E extraction on this split')
                frozen_ext = source['extraction']
            pool = pools[inf['question_id']]
            if spec.get('sane_reference'):
                dense = [r for r in retrieval_rows if r['question_id']==inf['question_id'] and
                    r['representation']==choice['representation'] and r['retriever']=='SEMANTIC' and
                    r['retrieval_width']==choice['retrieval_width']]
                if len(dense)!=1:
                    raise ValueError('Missing dense candidate pool for adapted SANE baseline')
                pool = dense[0]
            row = run_intervention(record,pool,spec,encoder,backend,call_cache,
                embeddings,cfg,extraction_cache,frozen_ext)
            cache.put(key,row)
    if backend is not None and hasattr(backend,'close'):backend.close()
    return {}


def run_g(cfg, split, frozen, directory, cache, headroom_rows=None, allowed_qids=None):
    """Replay true sequential prefix outcomes; labels train only inside FIT folds."""
    import joblib
    all_rows = load_phase(cfg, 'f', split)
    mode = read_revision_mode(cfg,frozen['f'])
    rows = [r for r in all_rows if normalize_mode(r['specification']['revision']) == mode]
    if allowed_qids is not None:
        rows = [r for r in rows if r['question_id'] in allowed_qids]
    by_key = {(r['question_id'],r['evidence_k']):r for r in rows}
    if split == 'fit':
        # Frozen encoders / fixed inference features; headroom is retrospective
        # analysis only. The learned verifier's decisions are always paper OOF.
        if cfg.get('_paper_fold', {}).get('role') == 'train':
            # Training-only architecture choice inside the OUTER FIT fold.
            # Its effect is subsequently evaluated on the outer held-out papers.
            gate_rows = rows
        else:
            if headroom_rows is None:
                raise FileNotFoundError('G requires verified FIT OOF headroom. Run outer A-F folds, then use --collect-headroom-oof.')
            gate_rows = headroom_rows
        gate = headroom_gate(gate_rows,cfg['adaptive'])
        gate['analysis_role'] = 'outer_fit_oof' if headroom_rows is not None else 'outer_training_only'
        atomic_json(directory/'w9g_gate.json',gate)
        if not gate['proceed']:
            return dict(status=gate['status'], gate=gate,revision=mode)
        actions = training_actions(rows)
        predictions, fold_ids, models, model = train_oof(actions,cfg['adaptive']['folds'])
        actions.assign(predicted_utility=predictions,fold=fold_ids).to_csv(directory/'fit_oof_actions.csv',index=False)
        model_path = directory/'sequential_verifier.joblib'
        joblib.dump(model,model_path)
        model_info = dict(path=str(model_path.resolve()),sha256=file_hash(model_path),features=list(actions.columns))
    else:
        frozen_g = load_frozen(frozen_path(cfg,'g'),'g')['settings']
        if frozen_g['status'].startswith('SKIPPED'):
            return frozen_g
        model_info = frozen_g['model']
        if file_hash(model_info['path'])!=model_info['sha256']:
            raise ValueError('Frozen verifier checkpoint changed')
        model = joblib.load(model_info['path'])
        from .adaptive_ke import FEATURES
        # Labels in training_actions are only retained for retrospective scoring;
        # model.predict receives FEATURES alone.
        actions = training_actions(rows)
        predictions = model.predict(actions[FEATURES].to_numpy(float))
        gate = frozen_g['gate']
    decisions = sequential_stop(actions,predictions,cfg['adaptive']['lambda_cost'])
    for decision in decisions:
        source = by_key[(decision['question_id'],decision['evidence_k'])]
        if source['selected_chunk_ids']!=decision['selected_chunk_ids']:
            raise ValueError('Sequential prefix identity differs from measured outcome')
        key = cache.key(source['question_id'],source['paper_id'],decision['selected_chunk_ids'],dict(operation='sequential_stop',revision_mode=mode))
        attempted = [by_key[(decision['question_id'],j)] for j in range(1,decision['attempted_count']+1)]
        # Deployed verifier proposes/extracts each prefix, then generates once.
        # Count rejected-proposal extraction and all earlier proposal calls too.
        fields = QueryCost.__dataclass_fields__
        cost = QueryCost(**{k:source[k] for k in fields if k in source})
        for name in ('extraction_prompt_tokens','extraction_generated_tokens'):
            setattr(cost,name,sum(r[name] for r in attempted))
        first = attempted[0]
        cost.selector_prompt_tokens=first['selector_prompt_tokens']
        cost.selector_generated_tokens=first['selector_generated_tokens']
        cost.num_lm_calls=1+int(first['selector_prompt_tokens']>0)+sum(r['extraction']['extraction_call_count'] for r in attempted)+int(cost.revision_prompt_tokens>0)
        cost.extraction_latency=sum(r['extraction']['extraction_latency'] for r in attempted)
        cost.model_calls=[c for c in first.get('model_calls',[]) if c.get('stage')=='selector']
        cost.model_calls.extend(c for r in attempted for c in r.get('extraction',{}).get('model_calls',[]))
        cost.model_calls.extend(c for c in source.get('model_calls',[]) if c.get('stage')=='revision')
        cache.put(key,dict(source,**cost.to_dict(),policy='SequentialStop',decision=decision,
            analysis_role='paper_grouped_verifier_oof' if split=='fit' else 'confirmation',
            verifier_latency=None, verifier_latency_note='Cached replay; verifier overhead not measured'))
    return dict(status='PROCEED',gate=gate,model=model_info,revision=mode,
                lambda_cost=cfg['adaptive']['lambda_cost'])


def baseline_rows(cfg, split, records, tokenizer, paths, required_policies=('B0','B1')):
    """Strict adapters for real frozen baselines. Missing OOF QCLA is a blocker."""
    canonical = read_jsonl(paths['canonical'])
    wanted = {r['inference']['question_id']:r for r in records}
    output = []
    from src.week7.w7e.prompt_builder import build_mixed_fidelity_sara_prompt
    for r in canonical:
        qid = str(r['question_id'])
        if qid not in wanted or int(r['k']) not in (4,6):
            continue
        record = wanted[qid]
        inf = record['inference']
        prompt,_,_,_=build_mixed_fidelity_sara_prompt(inf['question'],inf['passages'],
            r['selected_full_indices'],[],r['compressed_indices'],{},tokenizer)
        n = len(tokenizer.encode(prompt))
        generated = len(tokenizer.encode(r['prediction'],add_special_tokens=False))
        cost = QueryCost(initial_draft_prompt_tokens=n, initial_draft_generated_tokens=generated,final_prompt_tokens=n)
        row = dict(question_id=qid,paper_id=inf['paper_id'],policy='B0' if r['k']==4 else 'B1',prediction=r['prediction'],
                   **cost.to_dict(),promotion_rate=0.,extraction_rate=0.,faithfulness=None)
        output.append(annotate_outcome(row,inf,record['evaluation']))
    if not set(required_policies) & {'B2','B3','B4'}:
        return [r for r in output if r['policy'] in required_policies]
    if split in ('fit','internal_val'):
        directory = ROOT/'results/week8/w8e_controller'
        selected = json.loads((directory/'frozen_controller.json').read_text())['selected_policy']
        surface = read_jsonl(paths['surface'])
        lookup = {(str(r['question_id']),int(r['candidate_passage_index'])):r for r in surface}
        for name, policy in [('E1','B2'),(selected,'B3')]:
            if policy not in required_policies:
                continue
            suffix = 'fit_oof' if split=='fit' else 'internal_val'
            frame = pd.read_csv(directory/f'{name}_{suffix}_replay.csv',dtype={'question_id':str,'paper_id':str})
            for r in frame.to_dict('records'):
                qid = r['question_id']
                if qid not in wanted:
                    continue
                record = wanted[qid]
                promote = r['selected_action']=='PROMOTE'
                inf = record['inference']
                sf = lookup[(qid,int(r['candidate_passage_index']))] if promote else None
                prediction = sf['promoted_prediction'] if sf else inf['initial_draft']
                if str(r['paper_id'])!=inf['paper_id']:
                    raise ValueError('W8 baseline paper mismatch')
                row=dict(question_id=qid,paper_id=inf['paper_id'],policy=policy,prediction=prediction,
                    final_prompt_tokens=int(r['final_prompt_tokens']),total_query_prompt_tokens=int(r['cumulative_prompt_tokens']),
                    total_query_generated_tokens=None,num_lm_calls=2 if promote else 1,promotion_rate=float(promote),extraction_rate=0.,
                    retrieval_query_time=None,selection_latency=None,extraction_latency=None,revision_latency=None,
                    measured_end_to_end_latency=None,faithfulness=None,
                    latency_note='Cached W8 replay; routing/selection/gate overhead unmeasured.',
                    generated_tokens_note='Original continuation IDs unavailable; no invented count.')
                output.append(annotate_outcome(row,inf,record['evaluation']))
    else:
        # Old baseline evaluation must be explicitly exported for DEV/TEST.
        for policy in ('B2','B3'):
            if policy not in required_policies:
                continue
            path = cfg['baselines'].get(f'{policy.lower()}_{split}')
            if not path:
                raise FileNotFoundError(f'Configure real frozen {policy} outcomes for {split}')
            output.extend(read_jsonl(resolve(path)))
    if 'B4' not in required_policies:
        return [r for r in output if r['policy'] in required_policies]
    path = cfg['baselines'].get('qcla_fit_oof' if split=='fit' else 'qcla_'+split)
    if not path:
        raise FileNotFoundError('Missing real QCLA-1 FIT OOF outcomes. Set baselines.qcla_fit_oof; full-FIT training scores are forbidden.')
    qcla = read_jsonl(resolve(path))
    if split=='fit':
        metadata = json.loads(resolve(path).with_suffix('.oof.json').read_text())
        if metadata.get('paper_grouped') is not True or metadata.get('training_split')!='fit':
            raise ValueError('QCLA FIT export requires verified paper OOF metadata')
        for r in qcla:
            fold = metadata['folds'][str(r['fold'])]
            if str(r['paper_id']) in set(map(str,fold['training_papers'])) or str(r['paper_id']) not in set(map(str,fold['heldout_papers'])):
                raise ValueError('QCLA OOF paper contamination')
            outer = cfg.get('_paper_fold', {})
            if outer.get('role')=='heldout' and set(map(str,fold['training_papers'])) & set(outer['heldout_papers']):
                raise ValueError('QCLA training includes outer-heldout papers')
    for r in qcla:
        qid = str(r['question_id'])
        if qid not in wanted:
            continue
        record = wanted[qid]
        if str(r['paper_id'])!=record['inference']['paper_id']:
            raise ValueError('QCLA paper mismatch')
        row=dict(question_id=qid,paper_id=str(r['paper_id']),policy='B4',prediction=r['prediction'],
            final_prompt_tokens=r['prompt_tokens'],total_query_prompt_tokens=r['cumulative_prompt_tokens'],
            total_query_generated_tokens=r['generated_tokens'],num_lm_calls=r['generation_passes'],
            retrieval_query_time=None,selection_latency=None,extraction_latency=None,
            revision_latency=r['generation_latency_sec'],measured_end_to_end_latency=None,
            promotion_rate=0.,extraction_rate=0.,faithfulness=None)
        output.append(annotate_outcome(row,record['inference'],record['evaluation']))
    for policy in required_policies:
        rows=[r for r in output if r['policy']==policy]
        if len(rows)!=len(wanted) or {r['question_id'] for r in rows}!=set(wanted):
            raise ValueError(f'Incomplete real {policy} baseline coverage')
    return output


def summarize(cfg, phase, split, rows, directory, extra, records, frozen):
    directory.mkdir(parents=True,exist_ok=True)
    settings = {}
    if phase=='a':
        frame = pd.DataFrame(rows)
        columns=['any_gold_evidence_recall','all_gold_evidence_recall','mrr','ndcg',
            'mean_chunk_tokens','median_chunk_tokens','retrieved_context_tokens','evidence_overlap_density','retrieval_query_time',
            'index_time_embedding_seconds','index_time_chunking_seconds']
        summary=frame.groupby(['representation','retriever','retrieval_width'])[columns].mean().reset_index()
        summary.to_csv(directory/'w9a_summary.csv',index=False)
        if split=='fit':
            options=[]
            for r in summary.to_dict('records'):
                if pd.isna(r['all_gold_evidence_recall']):
                    raise ValueError('No annotated evidence; cannot select retrieval system')
                options.append(dict(r,policy=f'{r["representation"]}/{r["retriever"]}/{r["retrieval_width"]}',
                    quality=r['all_gold_evidence_recall'],
                    measured_latency=r['retrieval_query_time']+r['index_time_embedding_seconds']+r['index_time_chunking_seconds'],
                    complexity=(r['representation']!='ORIG')+(r['retriever']=='RRF')))
            choice=choose_global(options,cfg['retrieval_tolerance'],primary='quality',cost='retrieved_context_tokens')
            settings={**extra,**{k:choice[k] for k in ('representation','retriever','retrieval_width')}}
    elif phase=='g':
        settings=extra
        if rows:
            pd.DataFrame([policy_summary(rows)]).to_csv(directory/'w9g_summary.csv',index=False)
    else:
        by_policy=defaultdict(list)
        for r in rows:
            by_policy[r['policy']].append(r)
        expected={r['inference']['question_id'] for r in records}
        for name, group in by_policy.items():
            if len(group)!=len(expected) or {r['question_id'] for r in group}!=expected:
                raise ValueError(f'Incomplete/duplicate query coverage for {name}')
        summaries=[dict(policy=name,**policy_summary(group),complexity=i) for i,(name,group) in enumerate(by_policy.items())]
        pd.DataFrame(summaries).to_csv(directory/f'w9{phase}_summary.csv',index=False)
        if phase=='b':
            matrix, oracle=surface_analysis(rows)
            matrix.to_csv(directory/'w9b_query_matrix.csv',index=False)
            atomic_json(directory/'w9b_oracle_summary.json',oracle)
        if phase=='c':
            reference=by_policy['DraftSemantic']
            comparisons={name:paired_paper_bootstrap(group,reference,seed=cfg['_seed']) for name,group in by_policy.items() if name!='DraftSemantic'}
            atomic_json(directory/'w9c_paired_bootstrap.json',comparisons)
            atomic_json(directory/'w9c_selector_diagnostics.json',
                {name:selector_diagnostics(group,rows) for name,group in by_policy.items()})
        if phase=='d':
            atomic_json(directory/'w9d_central_comparisons.json',{f'{a}-{b}':paired_paper_bootstrap(by_policy[a],by_policy[b],seed=cfg['_seed'])
                for a,b in [('D2','D1'),('D3','D2'),('D3','D1')]})
        if phase=='e':
            audit=[]
            for r in rows:
                ext=r.get('extraction')
                if ext:
                    audit.append(dict(question_id=r['question_id'],paper_id=r['paper_id'],policy=r['policy'],
                        **{k:v for k,v in ext['audit'].items() if k!='rows'},
                        evidence_tokens=r['extraction_generated_tokens'],compression_ratio=r['extraction_compression_ratio']))
            pd.DataFrame(audit).to_csv(directory/'w9e_span_audit.csv',index=False)
            sample=random.Random(cfg['_seed']).sample(rows,min(20,len(rows)))
            atomic_json(directory/'w9e_qualitative_sample.json',sample)
        if phase=='f':
            atomic_json(directory/'w9f_utilization_comparisons.json',
                {f'{a}-{b}/K={k}':paired_paper_bootstrap(by_policy[f'{a}-K_E={k}'],by_policy[f'{b}-K_E={k}'],seed=cfg['_seed'])
                 for a,b in [('F1','F0'),('F2','F0'),('F2','F1')] for k in (1,2,3,4)})
        if phase=='h':
            from .baseline_registry import final_tables
            final_tables(cfg,split,records,rows,directory)
            atomic_json(directory/'w9h_paired_bootstrap.json',
                {name:paired_paper_bootstrap(group,by_policy['B0'],seed=cfg['_seed']) for name,group in by_policy.items() if name!='B0'})
        if split=='fit':
            options=summaries
            if phase=='h':
                # External trained baselines are reported, not selected as honest outer-OOF W9 architectures.
                options=[r for r in summaries if r['policy'] not in ('B2','B3','B4') and not r['policy'].startswith('W9-F')]
            if phase=='c':
                # More elaborate selectors must clear the global tolerance over DraftSemantic.
                baseline=next(s for s in summaries if s['policy']=='DraftSemantic')
                options=[s for s in summaries if s['policy']=='DraftSemantic' or s['f1']>baseline['f1']+cfg['quality_tolerance']]
            if phase=='f':
                # One controlled revision decision, using the FIT-frozen K from B.
                k=frozen['b']['settings']['evidence_k']
                options=[s for s in summaries if s['policy'] in (f'F0-K_E={k}',f'F1-K_E={k}',f'F2-K_E={k}')]
            winner=choose_global(options,cfg['quality_tolerance'])
            chosen=by_policy[winner['policy']][0]
            settings=dict(policy=winner['policy'])
            if phase=='b':
                settings.update(evidence_k=chosen['evidence_k'])
            if phase=='c':
                settings.update(selector=chosen['specification']['selector'])
            if phase=='e':
                settings.update(format=chosen['specification']['format'])
            if phase=='f':
                settings.update(revision=normalize_mode(chosen['specification']['revision']))
            if phase=='h':
                # Outer folds may select different K/format settings. Deploy the
                # selected architecture with its FULL-FIT frozen components.
                specs={s['policy']:s for s in interventions('h',cfg,frozen)}
                settings.update(specification=specs.get(winner['policy']),runtime_config={k:v for k,v in cfg.items() if not k.startswith('_')})
    report = directory/f'W9{phase.upper()}_REPORT.md'
    report.write_text(f'# W9-{phase.upper()}\n\nSplit: {split}. Real saved rows: {len(rows)}.\n\n'
        'FIT selects; INTERNAL-VAL confirms. DEV has been inspected. TEST remains locked.\n\n'
        'No conclusions are implied beyond the saved metrics. See summary CSV, manifests and paired intervals. '
        'Stage timings are cached component measurements; total end-to-end latency is unmeasured.\n\n'
        f'Proposed FIT decision (freeze requires explicit --freeze):\n\n```json\n{json.dumps(settings,indent=2)}\n```\n')
    return settings


def parser(phase):
    p=argparse.ArgumentParser(description=f'Run ONLY W9-{phase.upper()}; full experiments are manual.')
    p.add_argument('--split',choices=['fit','internal_val','dev'],default='fit')
    p.add_argument('--config',default=f'configs/week9/w9{phase}.yaml')
    p.add_argument('--resume',action='store_true')
    p.add_argument('--smoke-limit',type=int)
    p.add_argument('--seed',type=int,default=42)
    p.add_argument('--start-index',type=int,default=0)
    p.add_argument('--max-records',type=int)
    p.add_argument('--freeze',action='store_true',help='Freeze after a COMPLETE FIT run only')
    p.add_argument('--analyze-only',action='store_true')
    p.add_argument('--paper-fold',type=int,choices=range(5))
    p.add_argument('--fold-role',choices=['train','heldout'])
    p.add_argument('--collect-oof',action='store_true',help='H only: validate and aggregate five held-out fold runs')
    p.add_argument('--collect-headroom-oof',action='store_true',help='G only: gate using verified outer-heldout F surfaces')
    return p


def main(phase, argv=None):
    args=parser(phase).parse_args(argv)
    branch_guard()
    cfg=load_config(args.config)
    if phase=='h' and cfg.get('textual_model_roles'):
        raise ValueError('H method comparison keeps canonical SARA fixed; use separately verified generator_robustness_exports')
    if cfg['phase']!=phase:
        raise ValueError('Wrong phase config')
    cfg['_seed']=args.seed
    if args.start_index<0 or any(n is not None and n<1 for n in (args.smoke_limit,args.max_records)):
        raise ValueError('Record bounds must be positive; start index nonnegative')
    random.seed(args.seed)
    np.random.seed(args.seed)
    records, paths=load_records(cfg,args.split)
    original_records=records
    artifact_split=args.split
    fold_meta=None
    if args.paper_fold is not None:
        if args.split!='fit' or not args.fold_role:
            raise ValueError('Paper folds require --split fit and --fold-role')
        from sklearn.model_selection import GroupKFold
        groups=[r['inference']['paper_id'] for r in records]
        tr,va=list(GroupKFold(5).split(np.zeros(len(records)),groups=groups))[args.paper_fold]
        train_papers=sorted({groups[i] for i in tr})
        heldout_papers=sorted({groups[i] for i in va})
        fold_meta=dict(fold=args.paper_fold,training_papers=train_papers,heldout_papers=heldout_papers,
                       role=args.fold_role,fold_method='GroupKFold(5), sorted canonical retrieval rows')
        records=[records[i] for i in (tr if args.fold_role=='train' else va)]
        cfg['output_root']=str(resolve(cfg['output_root'])/'folds'/f'fold_{args.paper_fold}')
        if args.fold_role=='heldout':
            artifact_split='fit_heldout'
    elif args.fold_role:
        raise ValueError('--fold-role requires --paper-fold')
    smoke=args.smoke_limit is not None
    restricted=smoke or args.start_index>0 or args.max_records is not None
    if args.freeze and (args.split!='fit' or restricted or args.fold_role=='heldout'):
        raise ValueError('Cannot freeze partial/smoke/heldout runs')
    total_expected=len(records)
    records=records[args.start_index:]
    if args.max_records:
        records=records[:args.max_records]
    if args.smoke_limit:
        records=records[:args.smoke_limit]
    if not records:
        raise ValueError('No real records in requested scope')
    frozen=dependencies(cfg,phase,artifact_split)
    # During held-out fold evaluation all inference decisions come from the
    # fold's training-only freezes, never from held-out outcomes.
    if artifact_split=='fit_heldout':
        frozen['a']=load_frozen(frozen_path(cfg,'a'),'a')
    directory=phase_dir(cfg,phase,artifact_split,smoke)
    directory.mkdir(parents=True,exist_ok=True)
    paths_for_hash=dict(paths)
    identity=build_identity(cfg,phase,args.split,args.seed,paths_for_hash,frozen,artifact_split)
    if fold_meta:
        cfg['_paper_fold']=fold_meta
        identity['paper_fold']=fold_meta
    result_path=rows_path(cfg,phase,artifact_split,smoke)
    if args.collect_oof or args.collect_headroom_oof:
        if args.collect_headroom_oof and (phase!='g' or args.paper_fold is not None or args.split!='fit' or restricted):
            raise ValueError('--collect-headroom-oof requires unrestricted global G FIT')
        collected_phase = 'f' if args.collect_headroom_oof else 'h'
        for fold in range(5):
            fold_cfg=dict(cfg,output_root=str(resolve(cfg['output_root'])/'folds'/f'fold_{fold}'))
            for p in [rows_path(fold_cfg,collected_phase,'fit_heldout'), phase_dir(fold_cfg,collected_phase,'fit_heldout')/'paper_fold.json']:
                identity['input_hashes'][str(p.resolve())]=file_hash(p)
    scope='partial' if restricted else 'full'
    manifest_value=manifest(cfg,phase,args.split,args.seed,[Path(p) for p in identity['input_hashes']],
        [result_path],scope)
    manifest_value['model_hashes']=identity['model_hashes']
    if fold_meta:
        manifest_value['paper_fold']=fold_meta
    manifest_path=directory/f'w9{phase}_manifest.json'
    paths['split']=artifact_split
    extra={}
    if args.collect_oof:
        if phase!='h' or args.paper_fold is not None or args.split!='fit' or restricted:
            raise ValueError('--collect-oof requires unrestricted global H FIT')
        rows=collect_oof(cfg,original_records)
        with RowCache(result_path,identity,args.resume) as combined:
            for row in rows:
                combined.put(combined.key(row['question_id'],row['paper_id'],[],row['policy']),row)
            rows=list(combined.rows.values())
        extra={'analysis_role':'fit_oof'}
    elif args.analyze_only:
        rows=read_jsonl(result_path)
        saved=json.loads(result_path.with_suffix('.identity.json').read_text())
        if saved!=identity:
            raise ValueError('Analysis inputs/source/config changed')
        if phase=='a':
            extra=json.loads((directory/'w9a_index_settings.json').read_text())
        if phase=='g':
            extra=json.loads((directory/'w9g_settings.json').read_text())
    else:
        with RowCache(result_path,identity,args.resume) as cache:
            if phase=='a':
                encoder=LocalEncoder(cfg['encoder'],cfg['encoder_device'])
                with RowCache(directory/'w9a_embeddings.jsonl',identity,args.resume) as index_cache, \
                     RowCache(directory/'w9a_chunk_manifest.jsonl',identity,args.resume) as chunk_cache:
                    extra=run_a(cfg,records,paths,cache,directory,encoder,frozen,index_cache,chunk_cache)
                atomic_json(directory/'w9a_index_settings.json',extra)
            elif phase=='g':
                headroom_rows=collect_f_headroom(cfg,original_records) if args.collect_headroom_oof else None
                extra=run_g(cfg,artifact_split,frozen,directory,cache,headroom_rows,
                    {r['inference']['question_id'] for r in records})
                atomic_json(directory/'w9g_settings.json',extra)
            else:
                # Check all H baseline artifacts BEFORE any model generation.
                if phase=='h':
                    baselines=baseline_rows(cfg,args.split,records,load_tokenizer(cfg),paths)
                    for row in baselines:
                        key=cache.key(row['question_id'],row['paper_id'],[],row['policy'])
                        cache.put(key,row)
                with RowCache(directory/'lm_calls.jsonl',identity,args.resume) as call_cache, \
                     RowCache(directory/'extraction_units.jsonl',identity,args.resume) as extraction_cache:
                    run_generation(cfg,phase,records,paths,frozen,cache,call_cache,extraction_cache)
                if phase=='h' and frozen['g']['settings']['status']=='PROCEED':
                    for row in load_phase(cfg,'g',artifact_split):
                        key=cache.key(row['question_id'],row['paper_id'],row['selected_chunk_ids'],'B10')
                        cache.put(key,dict(row,policy='B10'))
            rows=list(cache.rows.values())
    # Results from resume may cover earlier slices: validate against full split
    # before claiming complete. Smoke artifacts always remain nonfreezable.
    available={r['question_id'] for r in rows}
    expected={r['inference']['question_id'] for r in records}
    if args.resume and not restricted:
        expected={r['inference']['question_id'] for r in original_records} if not fold_meta else expected
    complete=(available==expected and len(expected)==total_expected and not restricted)
    if phase=='g' and extra.get('status','').startswith('SKIPPED'):
        complete=not restricted
    if phase=='a' and complete:
        expected_per_query=(1+2*len(extra['semantic_grid']))*3*len(cfg['retrieval_widths'])
        counts=pd.Series([r['question_id'] for r in rows]).value_counts()
        if not (counts==expected_per_query).all():
            raise ValueError('Incomplete W9-A representation/retriever/width grid')
    report_split='fit' if artifact_split=='fit' else artifact_split
    settings=summarize(cfg,phase,report_split,rows,directory,extra,records,frozen)
    atomic_json(manifest_path,dict(manifest_value,complete=complete))
    if fold_meta:
        atomic_json(directory/'paper_fold.json',fold_meta)
    if args.freeze:
        if phase=='h' and not args.collect_oof:
            raise ValueError('H freeze requires --collect-oof; ordinary FIT outcomes are not held out')
        freeze(frozen_path(cfg,phase),phase,settings,manifest_value,complete,
            analysis_role='fit_oof' if phase=='h' else 'fit_fixed')
        if phase=='f':write_revision_freeze(cfg,settings)
    print(f'W9-{phase.upper()}: {len(rows)} saved real rows; complete={complete}; artifacts={directory}')


def collect_oof(cfg, records):
    """Validate true outer-heldout final policies; never relabel in-sample FIT."""
    all_rows=[]
    expected={r['inference']['question_id']:r['inference']['paper_id'] for r in records}
    seen=set()
    for fold in range(5):
        local=dict(cfg,output_root=str(resolve(cfg['output_root'])/'folds'/f'fold_{fold}'))
        directory=phase_dir(local,'h','fit_heldout')
        meta=json.loads((directory/'paper_fold.json').read_text())
        manifest_value=json.loads((directory/'w9h_manifest.json').read_text())
        if not manifest_value['complete'] or meta['role']!='heldout' or meta['fold']!=fold:
            raise ValueError('Incomplete held-out fold run')
        train=set(meta['training_papers']);heldout=set(meta['heldout_papers'])
        if train & heldout:
            raise ValueError('Outer fold paper contamination')
        for phase in 'abcdefg':
            artifact=load_frozen(frozen_path(local,phase),phase)
            training_manifest=json.loads((phase_dir(local,phase,'fit')/f'w9{phase}_manifest.json').read_text())
            if training_manifest['paper_fold']['training_papers']!=meta['training_papers']:
                raise ValueError('Fold component was not frozen on these training papers')
        rows=load_phase(local,'h','fit_heldout')
        for r in rows:
            pair=(r['question_id'],r['policy'])
            if pair in seen or expected.get(r['question_id'])!=r['paper_id'] or r['paper_id'] not in heldout:
                raise ValueError('Duplicate or contaminated OOF outcomes')
            seen.add(pair)
            all_rows.append(dict(r,oof_fold=fold,analysis_role='fit_oof'))
    policies={r['policy'] for r in all_rows}
    if 'B10' in policies and {r['question_id'] for r in all_rows if r['policy']=='B10'}!=set(expected):
        all_rows=[r for r in all_rows if r['policy']!='B10']
        policies.remove('B10')
    for policy in policies:
        if {r['question_id'] for r in all_rows if r['policy']==policy}!=set(expected):
            raise ValueError('Incomplete outer OOF policy coverage')
    required={'B0','B1','B5-SANE-adapted','B6','B7','B8','B9','W9-F0','W9-F1','W9-F2'}
    if not required<=policies:
        raise ValueError('Missing required final frontier policy')
    # Adaptive proceeds in some outer folds only: no full-coverage B10 means it
    # cannot enter the final global ranking. Keep those rows in individual folds.
    return all_rows


def final_test(argv=None):
    """No tuning flags. The exact FIT-OOF frozen policy is the only executable."""
    p=argparse.ArgumentParser(description='Explicit locked final TEST evaluation')
    p.add_argument('--allow-test',action='store_true')
    p.add_argument('--frozen-config',required=True)
    p.add_argument('--resume',action='store_true')
    args=p.parse_args(argv)
    final=check_test('test',args.allow_test,args.frozen_config)
    branch_guard()
    cfg=final['settings']['runtime_config']
    records,paths=load_records(cfg,'test',allow_test=True,frozen_config=args.frozen_config)
    frozen=dependencies(cfg,'h','test')
    # TEST retrieval is only the frozen representation/retriever/width. Users
    # prepare its candidates with the explicit runner below; no generic dropdown.
    output=phase_dir(cfg,'h','test')
    identity=build_identity(cfg,'h','test',42,paths,frozen)
    paths['split']='test'
    policy=final['settings']['policy']
    with RowCache(rows_path(cfg,'h','test'),identity,args.resume) as cache:
        if policy in ('B0','B1','B2','B3','B4'):
            for row in baseline_rows(cfg,'test',records,load_tokenizer(cfg),paths,required_policies=(policy,)):
                if row['policy']==policy:
                    key=cache.key(row['question_id'],row['paper_id'],[],policy)
                    cache.put(key,row)
        else:
            import torch
            embeddings=torch.load(paths['sfr'],map_location='cpu',weights_only=False)
            embeddings=embeddings.get('embeddings',embeddings)
            backend=SaraBackend(cfg)
            encoder=LocalEncoder(cfg['encoder'],cfg['encoder_device'])
            with RowCache(output/'lm_calls.jsonl',identity,args.resume) as calls, \
                 RowCache(output/'extraction_units.jsonl',identity,args.resume) as extraction_cache:
                for record in records:
                    inf=record['inference']
                    pool=locked_test_pool(cfg,frozen,record,paths,encoder,force_dense=policy=='B5-SANE-adapted')
                    spec=final['settings']['specification']
                    key=cache.key(inf['question_id'],inf['paper_id'],[],policy)
                    if key not in cache.rows:
                        if policy=='B10':
                            row=run_online_adaptive(record,pool,encoder,backend,calls,embeddings,cfg,extraction_cache,
                                frozen['g']['settings'],frozen)
                        else:
                            row=run_intervention(record,pool,spec,encoder,backend,calls,embeddings,cfg,extraction_cache)
                        cache.put(key,row)
        rows=list(cache.rows.values())
    if len(rows)!=len(records):
        raise ValueError('Incomplete TEST results; resume required')
    atomic_json(output/'test_summary.json',policy_summary(rows))
    atomic_json(output/'test_manifest.json',manifest(cfg,'h','test',42,
        [Path(n) for n in identity['input_hashes']]+[Path(args.frozen_config)],
        [rows_path(cfg,'h','test')],'full'))


def locked_test_pool(cfg,frozen,record,paths,encoder,force_dense=False):
    """Frozen candidate construction inside explicit TEST authorization only."""
    inf=record['inference'];settings=frozen['a']['settings']
    document=source_document(record,paths['raw'])
    lineage=locate_passages(document,record['original_passages'])
    representation=settings['representation']
    if representation=='ORIG':
        chunks=original_chunks(document,lineage,encoder.tokenizer)
    elif representation.startswith('SEM-'):
        chunks=semantic_chunks(document,lineage,encoder.tokenizer,encoder,
            **settings['semantic_grid'][int(representation.split('-')[1])])
    else:
        name=representation.replace('FIXED-','SEM-')
        chunks=fixed_chunks(document,lineage,encoder.tokenizer,settings['fixed_sizes'][name],cfg['fixed_overlap'])
    start=time.perf_counter()
    ranked=rank_candidates(inf['question'],inf['paper_id'],chunks,encoder,
        'SEMANTIC' if force_dense else settings['retriever'],settings['retrieval_width'])
    return dict(candidates=ranked,retrieval_width=settings['retrieval_width'],
        retrieval_actual_count=len(ranked),retrieval_query_time=time.perf_counter()-start)


def run_online_adaptive(record,pool,encoder,backend,calls,embeddings,cfg,extraction_cache,frozen_g,frozen):
    """Propose/extract a prefix, verify THIS action, stop, then revise once."""
    import joblib
    from .adaptive_ke import FEATURES
    model_info=frozen_g['model']
    if file_hash(model_info['path'])!=model_info['sha256']:
        raise ValueError('Sequential model fingerprint changed')
    model=joblib.load(model_info['path'])
    inf=record['inference']
    prompt,comp,tokens=canonical_context(record,embeddings,backend.tokenizer,backend.device,backend.model.dtype)
    cost=QueryCost(initial_draft_prompt_tokens=tokens,
        initial_draft_generated_tokens=len(record['canonical'].get('saved_generated_token_ids',
            backend.tokenizer.encode(inf['initial_draft'],add_special_tokens=False))),
        final_prompt_tokens=tokens,retrieval_query_time=pool['retrieval_query_time'])
    selected=None
    proposals=[]
    verifier_seconds=0.
    for count in range(1,min(4,len(pool['candidates']))+1):
        spec=dict(policy='B10',selector=frozen['c']['settings']['selector'],evidence_k=count,
            representation='draft',format=frozen['e']['settings']['format'],revision=read_revision_mode(cfg,frozen['f']))
        prepared=run_intervention(record,pool,spec,encoder,backend,calls,embeddings,cfg,extraction_cache,prepare_only=True)
        row,proposal_cost=prepared['result'],prepared['cost']
        # One ranked selection for all prefixes; cache reuse does not multiply its
        # scientific per-query cost. Each new joint extraction is a real call.
        if count==1:
            cost.selector_prompt_tokens=proposal_cost.selector_prompt_tokens
            cost.selector_generated_tokens=proposal_cost.selector_generated_tokens
            cost.num_lm_calls+=int(proposal_cost.selector_prompt_tokens>0)
            cost.selection_latency=proposal_cost.selection_latency
        cost.model_calls.extend(c for c in proposal_cost.model_calls if c.get('stage')=='extraction' or count==1 and c.get('stage')=='selector')
        cost.extraction_prompt_tokens+=proposal_cost.extraction_prompt_tokens
        cost.extraction_generated_tokens+=proposal_cost.extraction_generated_tokens
        cost.num_lm_calls+=row['extraction']['extraction_call_count']
        cost.extraction_latency=(cost.extraction_latency or 0.)+(proposal_cost.extraction_latency or 0.)
        feature=row['proposed_features']
        start=time.perf_counter()
        utility=float(model.predict([[feature[name] for name in FEATURES]])[0])
        verifier_seconds+=time.perf_counter()-start
        accept=utility-frozen_g['lambda_cost']*feature['proposed_prompt_tokens']>0
        proposals.append(dict(chunk_id=row['selected_chunk_ids'][-1],predicted_utility=utility,accepted=accept))
        if not accept:
            break
        selected=row
    if selected is None:
        result=dict(question_id=inf['question_id'],paper_id=inf['paper_id'],policy='B10',prediction=inf['initial_draft'],
            evidence_k=0,selected_chunk_ids=[],evidence_text='',faithfulness=None,promotion_rate=0.,extraction_rate=bool(proposals))
    else:
        client=CachedClient(backend,calls,inf,cost,'revision',context=prompt,retrieval_embeds=comp,evidence_ids=selected['selected_chunk_ids'],revision_mode=read_revision_mode(cfg,frozen['f']))
        result=dict(selected,prediction=revise(inf,selected['evidence_text'],client,read_revision_mode(cfg,frozen['f'])))
        cost.revision_latency=client.latency
    result.update(cost.to_dict(),policy='B10',proposals=proposals,verifier_latency=verifier_seconds,
                  measured_end_to_end_latency=None)
    return annotate_outcome(result,inf,record['evaluation'])


def collect_f_headroom(cfg, records):
    """OOF headroom collected before the global G go/no-go decision."""
    expected={r['inference']['question_id']:r['inference']['paper_id'] for r in records}
    seen=set();output=[]
    for fold in range(5):
        local=dict(cfg,output_root=str(resolve(cfg['output_root'])/'folds'/f'fold_{fold}'))
        directory=phase_dir(local,'f','fit_heldout')
        meta=json.loads((directory/'paper_fold.json').read_text())
        result_manifest=json.loads((directory/'w9f_manifest.json').read_text())
        if not result_manifest['complete'] or meta['role']!='heldout' or meta['fold']!=fold:
            raise ValueError('Incomplete F headroom fold')
        if set(meta['training_papers']) & set(meta['heldout_papers']):
            raise ValueError('F headroom paper leakage')
        for phase in 'abcdef':
            load_frozen(frozen_path(local,phase),phase)
            m=json.loads((phase_dir(local,phase,'fit')/f'w9{phase}_manifest.json').read_text())
            if m['paper_fold']['training_papers']!=meta['training_papers']:
                raise ValueError('Headroom component was selected on held-out papers')
        mode=read_revision_mode(local,load_frozen(frozen_path(local,'f'),'f'))
        rows=[r for r in load_phase(local,'f','fit_heldout') if normalize_mode(r['specification']['revision'])==mode]
        for r in rows:
            key=(r['question_id'],r['evidence_k'])
            if key in seen or expected.get(r['question_id'])!=r['paper_id'] or r['paper_id'] not in meta['heldout_papers']:
                raise ValueError('Invalid OOF headroom identity')
            seen.add(key);output.append(dict(r,oof_fold=fold))
    if seen!={(q,k) for q in expected for k in range(5)}:
        raise ValueError('Incomplete OOF headroom coverage')
    return output
