"""Identity guards for a manual operating point, never a calibration continuation."""
import contextlib
import fcntl
from pathlib import Path
from src.post_w12.bea_v2 import protocol as original

ROOT = original.ROOT
LABEL = 'POST_W12_BEA_V2_POSTHOC_EXPLORATORY'
METHODS = ['FIXED_SENTENCE', 'QBGF_FULL', 'BEA_QBGF_V1', 'BEA_V2']
BUDGETS = [256, 1024]
DATASETS = ['HotpotQA', 'QASPER', 'SciFact']
NOTICE = ('This is a post-hoc exploratory evaluation at a manually selected engineering '
          'operating point (tau_stop=8e-5) after the preregistered BEA_V2 calibration '
          'failed. It cannot be interpreted as confirmation or as evidence that the '
          'original calibration protocol succeeded.')
read, sha, digest, write, text = original.read, original.sha, original.digest, original.write, original.text
CONFIG = ROOT/'W12-POST/bea_v2_posthoc_config.json'

def load():
    c = read(CONFIG)
    if (c['label'] != LABEL or c['output_root'] != 'results/post_w12/bea_v2/posthoc_tau8e5'
        or c['threshold'] != 0.00008 or c['methods'] != METHODS or c['budgets'] != BUDGETS
        or c['statistics'] != {'resamples':10000,'seed':42}
        or c['automatic_expansion'] or c['retrain_predictors']
        or not c['resume'] or c['overwrite']):
        raise ValueError('Post-hoc registry is fixed; no threshold search or method changes')
    return c

def output(c):
    return ROOT/c['output_root']

def context(c=None, persist=False):
    c = c or load()
    if c != load():
        raise ValueError('Use the exact on-disk fixed post-hoc configuration')
    # Also applies the original branch, model, cache and package guards.
    old = original.load()
    from src.post_w12.bea_v2.models import verify_models
    models = verify_models(old)
    for name, expected in c['protected_inputs'].items():
        if sha(ROOT/name) != expected:
            raise ValueError('Protected input changed: '+name)
    terminal = read(original.root(old)/'CALIBRATION_TERMINATED.json')
    if terminal['status'] != 'CALIBRATION_FAILED_TERMINATED' or terminal['reader_pilot_permitted']:
        raise ValueError('The official terminated result must remain terminated')
    parent, v1, splits, cohort = original.dependencies(old)
    if any(len(cohort['selected'][d]) != 50 for d in DATASETS):
        raise ValueError('Use exactly the original 50-query-per-dataset development cohort')
    for name, expected in models['training_sources'].items():
        if sha(ROOT/name) != expected:
            raise ValueError('Frozen predictor training provenance changed: '+name)
    paths = list(Path(__file__).parent.glob('*.py')) + [CONFIG, ROOT/'W12-POST/run_bea_v2_posthoc.py']
    identity = dict(label=LABEL, notice=NOTICE, config=c,
        implementation={str(p.relative_to(ROOT)):sha(p) for p in sorted(paths)},
        original_implementation=original.implementation(), original_config=old,
        model_manifest_sha=sha(original.root(old)/'utility_model/MODELS.json'),
        fold_sha=sha(original.root(old)/'CROSSFIT_FOLDS.json'),
        feature_schema_sha=sha(original.root(old)/'FEATURE_SCHEMA.json'),
        cohort_hash=digest(cohort), v1_identity_hash=v1['identity_hash'],
        parent_identity=v1['identity']['parent_identity'],
        reader=v1['identity']['reader'], tokenizer=v1['identity']['tokenizer'],
        prompt_sha=v1['identity']['prompt_sha'], parser_sha=v1['identity']['parser_sha'],
        decoding=v1['identity']['decoding'], serializer=v1['identity']['serializer'],
        threshold=0.00008, calibration_status='CALIBRATION_FAILED_TERMINATED',
        threshold_selected_posthoc=True, fresh_confirmation=False, automatic_expansion=False)
    record = dict(identity=identity, identity_hash=digest(identity), label=LABEL)
    if persist:
        write(output(c)/'POSTHOC_ENGINEERING_IDENTITY.json', record)
        text(output(c)/'PROTOCOL.md', '# BEA_V2 post-hoc engineering protocol\n\n'+NOTICE
             +'\n\nFixed tau=0.00008; no further search. Original 150 development queries, '
             'two budgets, four methods: 1,200 logical rows. Up to 300 missing BEA_V2 '
             'answers; historical controls reused read-only. Any genuinely missing '
             'control is restored canonically into this namespace only and accounted '
             'separately. Eighteen exploratory paired contrasts, 10,000 cluster '
             'resamples, seed 42, Holm adjustment across all 18. No expansion.\n')
    elif (output(c)/'POSTHOC_ENGINEERING_IDENTITY.json').exists():
        if read(output(c)/'POSTHOC_ENGINEERING_IDENTITY.json') != record:
            raise ValueError('Incompatible existing post-hoc identity; do not overwrite')
    return c, old, parent, v1, splits, cohort, record

class Store(original.Store):
    def get(self, key):
        row = super().get(key)
        if row is not None and read(self.file(key)).get('label') != LABEL:
            raise ValueError('Wrong post-hoc envelope label')
        return row

    def put(self, key, row):
        write(self.file(key), dict(key=key,row=row,checksum=digest(row),
              identity_hash=digest(self.identity),label=LABEL))
        return row

@contextlib.contextmanager
def lock(c):
    output(c).mkdir(parents=True,exist_ok=True)
    with (output(c)/'.lock').open('a') as handle:
        try:
            fcntl.flock(handle,fcntl.LOCK_EX|fcntl.LOCK_NB)
        except BlockingIOError:
            raise RuntimeError('Another post-hoc stage is active')
        yield

def keys(cohort):
    return [[d,m['query_id'],method,b] for d in DATASETS
            for m in cohort['selected'][d] for b in BUDGETS for method in METHODS]
