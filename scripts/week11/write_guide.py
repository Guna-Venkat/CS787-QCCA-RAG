"""Inventory and preregistration documentation, never experiments."""
from pathlib import Path
import json
import sys
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT))
from src.week11.protocol import load,DIRECTORIES
from src.week11.catalog import NOTEBOOKS,TABLES,PLOTS
from src.week11.dashboard import resources
from src.week11.cache import atomic,digest


def main():
    cfg=load();base=ROOT/cfg['output_root'];resource=resources(cfg)
    resource.to_csv(base/'setup/resource_plan.csv',index=False)
    atomic(base/'setup/preregistered_protocol.json',dict(configuration=cfg,verification_dataset='SciFact',
        default_safe=True,full_started=False,primary_reader='Qwen',sampling='50 hashed IDs/dataset, seed42; outcomes not used',
        qasper_test='BLOCKED',historical_artifacts='READ_ONLY',optional_token='PENDING; no qualification/model run performed by agent'))
    lines=['# Week 11 — multi-granular selective-fidelity laboratory','',
        'Implementation framework only. No Week-11 scientific result or winner is claimed from smoke fixtures. Full runs are manual and opt-in.','',
        '## Start here','',
        '1. Open `notebooks/week11/week11_MASTER.ipynb` for status. It never launches generation.',
        '2. Open `week11_0_setup_and_data.ipynb` and Run All with its defaults. This is fabricated smoke only.',
        '3. Development data has now been prepared at the user’s request. `sara_env/bin/python scripts/week11/prepare_datasets.py --download` verifies and reuses locked files. If the original Hotpot host is unavailable, the downloader explicitly uses the pinned `hotpotqa/hotpot_qa` distractor/validation source, restores the native JSON field layout without altering source strings, and records its revision, conversion and hashes in `data/week11/source_lock.json`. No experimental phase is started by this command.',
        '4. Inspect `configs/week11/`: primary reader Qwen; SciFact verification; sample50 IDs per dataset; beta5; gamma1; hybrid0.5; budgets256/512/1024/FULL. Change design choices before the first real setup/freeze, not after outcomes.',
        '5. For a manual real phase, change only `RUN_FULL=True` and `RUN_SMOKE=False` in its first FLAGS cell, then Run All. Derived phase flags require no other changes. Validated existing rows and identical-text reader outputs are reused.',
        '6. After real results finish, set `REVIEW_PHASE`, `REVIEWED_CONFIG` (dictionary), and `REVIEW_REASON` in FLAGS. Rerun the review cell, or Run All to reuse existing rows. The next dependent full phase refuses a missing/changed review.',
        '7. If optional LLMLingua conditions remain pending, review can explicitly acknowledge `REVIEWED_CONFIG={"waive_pending":["LLMLingua-2"]}`. This does not claim those conditions were executed. Otherwise qualify the token compressor before the first real freeze.',
        '8. Before F, copy actual reviewed action IDs and their configuration dictionaries into `week11_runs.yaml`. Empty/nonexistent finalist IDs refuse execution. Later finalist/analysis fields are scoped so they do not invalidate unrelated earlier freezes.',
        '9. H/I consume saved results only. No generation occurs in them. Default generation notebooks prefer saved real results when their status receipt exists, otherwise they run fabricated smoke. Both smoke and full disabled loads saved views; no work starts in generation notebooks.','',
        'For F, the dictionary keyed by a copied action ID uses input fields such as `reader: Qwen`, `granularity: SENTENCE`, `selector: semantic`, `budget: 512`, `compressor: C0`, `pool_width: 20`. Do not paste a whole output summary row with metric fields; unknown settings are rejected. These are manually selected candidate settings, not automatically discovered winners.','',
        '## Files created / modified','',
        'All implementation changes are Week11-scoped. Historical Week0–10 files are read-only. Source/config/notebook entry points:','']
    for directory,pattern in [('src/week11','*.py'),('scripts/week11','*.py'),('configs/week11','*.yaml'),('tests/week11','*.py'),('notebooks/week11','*.ipynb')]:
        lines.extend('- `'+str(p.relative_to(ROOT))+'`' for p in sorted((ROOT/directory).glob(pattern)))
    lines+=['','Additional new artifacts: Week11-scoped cache/raw-data/output ignore files; `setup/historical_sha256.json`, protocol/resource inventories, fabricated smoke tables/figures, and this guide.','',
        '## Dataset protocol','',
        '| Dataset | Split | Candidate corpus | Evaluation |', '|---|---|---|---|',
        '| QASPER | Existing project DEV, drawn from raw QASPER TRAIN | Full corresponding scientific paper | Existing project token F1/EM; answer-type breakdown |',
        '| HotpotQA | Official dev, distractor setting | Supplied ten paragraphs, including titles | Official normalization/yes-no answer F1 and EM; selected-source evidence coverage |',
        '| SciFact | Official claims_dev + corpus | Entire abstract corpus including titles; not oracle cited documents | Strict normalized label accuracy, fixed-three-class macro-F1, per-class metrics, invalid-label rate |','',
        'SciFact SUPPORT/CONTRADICT map to SUPPORT/REFUTE. Empty-evidence claims retain NOT_ENOUGH_INFO. Conflicting labels fail explicitly. Corpus titles are source text and native title units; original sentence IDs are retained for gold mapping. Gold is evaluation-only, and no labels enter inference dictionaries. No QASPER TEST file is read, selected, generated against, or evaluated. These are characterization splits, not blind confirmation.','',
        '**Default study size:** deterministic SHA256 sample of50 queries per dataset (150 total), seed42, independent of labels/outcomes. Full available counts and exact selected IDs are saved in `cohort.json`. Change this predefined size before freezing if desired.','',
        'External dataset byte hashes are locked before outcomes. The official SciFact release URL is mutable; `source_lock.json` freezes the actual first-downloaded bytes. Following implementation, the user authorized development-data downloads; HotpotQA DEV and SciFact DEV/corpus are now present, with no experimental phase started.','',
        'Primary schema references: [HotpotQA official site](https://hotpotqa.github.io/), [official answer evaluator](https://github.com/hotpotqa/hotpot/blob/master/hotpot_evaluate_v1.py), [SciFact schema](https://github.com/allenai/scifact/blob/master/doc/data.md).','',
        '## Evidence/retrieval protocol','',
        '- Canonical hierarchy: Document → Passage → Sentence → PROPOSITION_RULE_V1. Every unit has exact offsets, source hash, parent identity, document order, and canonical Qwen token count.',
        '- Native Hotpot paragraphs and SciFact abstracts are passages. QASPER uses frozen exact-word-boundary passages under256 actual Qwen tokens, zero overlap. Single lexical units exceeding the ceiling fail, not silently truncate.',
        '- Academic sentence splitter is reused read-only from E3. PROPOSITION_RULE_V1 conservatively splits semicolon/dash clauses. These are exact source fragments, not Dense X propositions or generated atomic facts.',
        '- BM25 is the project Okapi k1=1.5/b=.75 implementation; MiniLM and SFR use exact W10 revisions. SFR query instruction and explicit long-input window pooling are reused read-only.',
        '- P0 direct normalized passage vectors; P1 mean normalized sentence vectors then renormalize; P2 maximum query/sentence cosine; P3 softmax(beta5 × cosine) weighted sentence vector then renormalize. Empty sentence sets use direct passage vectors.',
        '- Selectors: lexical, SFR semantic, deterministic min-max hybrid0.5/0.5; constant score columns normalize to zero. Density is raw semantic relevance/token_count^1, used by H5. No outcome-dependent tuning.',
        '- C/D share a query-retrieved SFR top20 passage pool; E uses top10 for matched SARA source exposure. FULL means this frozen available candidate pool, capped once at6144 evidence tokens using the maximum actual count across all three tokenizers. This is not unlimited document input.',
        '- Ranking selects WHAT; presentation restores original document/source order. Unit additions use strict prefix-under-budget; an oversized next unit stops selection. Parent/child overlaps cannot be double-exposed.',
        '- Shared evidence must satisfy ALL pinned reader tokenizers. Per-reader actual evidence/prompt/generated token counts remain separate. This conservative shared-budget convention guarantees identical textual evidence in D; it is not independently optimized token filling for each reader.',
        '- Retrieval gold metrics use full non-whitespace annotated-character union coverage, repeated QASPER occurrences as alternatives, and strict all-annotation coverage. Whitespace gaps created at exact clause/sentence boundaries do not turn otherwise complete content into a recall miss. Unmapped evidence remains missed. No-gold recall is null with an explicit denominator. This is not official SciFact rationale scoring or Hotpot joint answer/support F1.','',
        '## Frozen models','', '| Name | Model | Revision |','|---|---|---|']
    for k,v in cfg['models'].items():lines.append(f'| {k} | {v["id"]} | `{v["revision"]}` |')
    lines+=['','Readers use FP16, no quantization, efficient SDPA, greedy decoding, seed42, max128 new tokens, max8192 context. Qwen thinking is disabled. The same user instruction/evidence is wrapped in each native chat template. Context overflow fails without truncation. Peak VRAM is recorded only when measured.','',
        '## Compressors and hierarchy','',
        '| Condition | State | Meaning |','|---|---|---|',
        '| C0 | Implemented; synthetic smoke | Selected source text, no additional transformation |',
        '| C1 / TOKEN | Adapter + mock interface tested; full pending | LLMLingua-2 token deletion, not cosine-ranked wordpieces |',
        '| C2 | Implemented; synthetic smoke | Query-only SFR-ranked top8 exact sentences, budget-constrained and source-order restored; E3-inspired mechanism on W11 pools, not reuse/reinterpretation of W9 outcomes |',
        '| C3 | Existing runtime integrated; mocked/synthetic restriction tests | MISTRAL_ONLY_LATENT_BASELINE, QASPER local SARA checkpoint, k4 text/n10 total; no new projector training |',
        '| C4 | Future work | Faithful SeleCom not implemented; no approximation is labelled SeleCom |','',
        'LLMLingua is not installed in this environment, and no compressor model was downloaded. Probe: `sara_env/bin/python scripts/week11/check_token_compressor.py`. After user installation, a separate explicit `--qualify --revision <immutable-revision> --download --write-config` performs one fabricated-input qualification and pins the configuration. Do this before real setup/freeze, or retain the explicit pending conditions. No token-compression performance is invented. [Official LLMLingua-2 example](https://github.com/microsoft/LLMLingua/blob/main/examples/LLMLingua2.ipynb).','',
        'H0 all passages; H1 full head plus sentence remainder; H2 full head plus clause remainder; H3 sentence head/middle plus clause tail; H4 full head + sentence middle + LLMLingua token tail (pending); H5 density-ranked mixed units with overlap suppression. Head count, middle region and tail fraction are configured. These are deterministic prototypes, not learned controllers.','',
        'H4 compresses tail passages separately, records their parent IDs/hashes, and restores all full/sentence/compressed components to source order. Token output is not falsely assigned exact contiguous source spans. C3 uses an adapted local checkpoint with its own adapted no-evidence baseline for HELP/HURT, so it is a distinct specialist reader variant: portable reader/compressor comparisons and Pareto frontiers exclude it. A C3-versus-public-Mistral difference does not isolate compression alone.','',
        '## Cache, resume, safety','',
        'All caches are new under `cache/week11/v1/`. Normalized queries reference shared documents rather than copying the SciFact corpus into every query shard. Hierarchy is cached per document. Text vectors share a deduplicated SQLite store; passage/sentence/proposition index manifests reference that store rather than duplicating large SFR vectors. BM25 inverted postings and query rankings persist separately.','',
        'Selection keys include dataset/split/query, source identity, retriever, granularity, selector, budget, chunking config, encoder revision, tokenizer identity and strategy. Actual encoder/compressor weight hashes identify the selection cache namespace. D is reuse-only and refuses missing C evidence instead of recomputing rankings per reader. Generation reuse keys include question, exact evidence, prompt/task, reader weights, decoding and any latent vectors/checkpoint.','',
        'Before resume: config, relevant source/dependency code, dataset bytes, pinned model bytes and evidence-selection hashes are checked. Existing rows are checksummed and immutable. A mismatch requires a new version; overwrite is unsupported. Completed rows skip model generation. Embedding preprocessing and cached-original timing are distinguished; no false zero conceptual cost or unmeasured end-to-end latency is reported. Candidate text in memory is not counted as LM prompt tokens.','',
        'Sparse output shards/cache databases and downloaded raw data have Week11-scoped ignore files; summaries, protocols and plots remain inspectable. No historical .gitignore, source, notebook, result or model file was edited.','',
        '## Notebook inventory and exact order','',
        '| Order | Notebook | Purpose/output | Flags and action |','|---|---|---|']
    for phase,name in NOTEBOOKS.items():
        lines.append(f'| {phase} | `{name}` | `{DIRECTORIES[phase]}/` | '+('Saved-result analysis; no generation' if phase in ('H','I') else 'Default RUN_SMOKE=True/RUN_FULL=False; opt in manually after prerequisite review')+' |')
    lines+=['','`week11_MASTER.ipynb` is an additional read-only dashboard. It displays status, expected/completed counts, configuration, resources, setup/cache tables and current completed figures. It never executes the study.','',
        '## DataFrame and plot inventory','']
    for phase in NOTEBOOKS:
        lines+=[f'### {phase} — {DIRECTORIES[phase]}','', 'Tables: '+', '.join('`'+x+'`' for x in TABLES[phase])+'.','',
            'Plots: '+', '.join(x[0] for x in PLOTS[phase])+'.','']
    lines+=['Every notebook also displays configuration and run status. Each major figure displays its underlying DataFrame, saves source CSV, PNG and PDF, and labels synthetic smoke. Recall curves are faceted by granularity; pooled descriptive means name the averaged factors in the title. Final reader/compressor plots use their matched experimental blocks rather than averaging incompatible phase settings. Unavailable data is left pending, never drawn as fabricated zeros. Figure12 supports reviewed historical W3/W6/W8/W9/W10/W11 records in separate protocol panels; populate `history_trajectory_records` explicitly rather than invent or pool old metrics.','',
        '## Analysis protocol','',
        'QA HELP/HURT uses paired task-score differences >+0.01 / <−0.01 versus the same-reader no-evidence control. Verification uses wrong→correct / correct→wrong. Undefined metrics are null. Verification reports strict label-format errors, accuracy, macro-F1 and class metrics; QA reports token F1 and EM. No combined raw QA-F1/SciFact-accuracy “overall F1” is produced.','',
        'Paired bootstrap:10,000 resamples, seed42; QASPER paper clusters, Hotpot query and SciFact claim clusters. Comparisons are labelled primary/secondary/exploratory. Main factorial effects and interaction cells are descriptive within datasets; dataset summaries use normalized baseline gains/ranks, not pooled raw task scores. Explicit four-cell difference-in-differences contrasts can be supplied in config, with paired bootstrap. No unstable mixed-effects/ANOVA assumptions are forced.','',
        'H requires paired eligible surfaces for exact/epsilon oracle (QA epsilon0.01, verification exact ties). Cost breaks ties deterministically. It saves action distributions, top1/5/10/20% gain concentration, cumulative curves and OCR when its denominator is nontrivial. Global within-dataset and within-reader oracles are separate. Incomplete pairing is reported, never silently complete-case filtered. Oracles are retrospective, not deployable methods. W10 seed diagnostics are read-only hooks with a distinct protocol label.','',
        '## Resources: configurations and answer-row upper bounds','',
        'These are work counts, not runtime predictions. No full W11 rate has been measured. Identical-text response reuse can substantially reduce physical calls.','',resource.to_markdown(index=False),'',
        'A/B use retrieval/embedding only. Setup and H/I do not generate answers. F has no configurations until manually supplied. Pending C1/H4 conditions are separately counted. Default150 queries give C6150 configured answer rows (5550 executable without token), D18450 (16650 executable; Qwen rows reusable from C), E5950 (4150 executable, including the specialist own-baseline control), and G3750 (3150 executable). Baseline/duplicate evidence results can be reused; these are upper bounds, not promised physical call counts.','',
        '## Validation status','']
    validation=base/'smoke/notebook_validation.json'
    if validation.exists():
        data=json.loads(validation.read_text());lines.append('Safe-default notebook checks: '+', '.join(f'{r["phase"]}={r["status"]}' for r in data)+'.')
    else:lines.append('Safe notebook execution validation pending.')
    verification=base/'setup/verification.json'
    if verification.exists():lines+=['',json.dumps(json.loads(verification.read_text()),indent=2)]
    lines+=['','## Remaining limits','',
        '- No real Week11 model generation or full dataset run was executed by the agent.',
        '- LLMLingua-2 model qualification and full TOKEN/H4 support remain pending; faithful learned compression remains future work.',
        '- External development files are prepared and byte-locked before outcomes. The failed CMU Hotpot host was replaced by a recorded, pinned HotpotQA-maintained Hugging Face source; this changes the file serialization, not the selected dataset setting.',
        '- Exact source grounding is not semantic faithfulness or answer correctness.',
        '- Strict-prefix budgets can leave unused capacity, especially when a passage exceeds the budget; this is a declared condition, not a bug or silent closest-fit substitution.',
        '- Default50-query samples characterize these splits; they do not establish fresh confirmation or universal superiority.',
        '- No winner, significance, controller success or research conclusion can be inferred from fabricated smoke data.','']
    text='\n'.join(lines)
    (ROOT/'notebooks/week11/README.md').write_text(text)
    (base/'setup/IMPLEMENTATION_REPORT.md').write_text(text)
    print('Guide and resource inventory written; no experiments executed.')


if __name__=='__main__':main()
