"""Session-only history; inspecting a pair never executes the pipeline."""
from .protocol import validity
from .preset import is_preset,load_preset
from .reference import metrics_text


def remember(history, result):
    """Keep one row per condition, retaining it beyond the generation cache."""
    if validity(result) is not None:
        return
    key = (result['input_key'], result['method'], result.get('device'))
    for entry in history:
        if entry['key'] == key:
            entry['result'] = result
            return
    history.append(dict(number=max((e['number'] for e in history), default=0)+1,
                        key=key, result=result))
    # Bound session memory while keeping stable, monotonically increasing IDs.
    del history[:-30]


def rows(history):
    saved=[dict(Run=e['number'], Retriever=r['retriever'], Method=r['method'],
                 Budget=r['budget'], Evidence=r['selection']['evidence_tokens'],
                 Units=len(r['selection']['units']),
                 Reader='Answer saved' if r['answer'] is not None else 'Context only',
                 Answer=r['answer'] if r['answer'] is not None else 'No answer — reader unavailable or disabled',
                 Metrics=metrics_text(r))
            for e in history if validity(r := e['result']) is None]
    if any(is_preset(e['result']['query'],e['result']['passages'])
           for e in history if validity(e['result']) is None):
        ref=load_preset()['reference']
        saved.insert(0,dict(Run=0,Retriever='—',Method=ref['label'],Budget=None,
                           Evidence=None,Units=None,Reader='Stored reference · no inference',
                           Answer=ref['answer'],Metrics='Reference anchor · not a model run'))
    return saved


def differences(a, b):
    """Describe actual changes, never assume arbitrary runs are controlled."""
    fields = [('query','Query'), ('passages','Source passages'),
              ('retriever','Retriever'), ('method','Method'), ('budget','Budget'),
              ('segmentation','Segmentation'), ('scoring_mode','Scoring mode'),
              ('scorer_identity','Downstream scorer'), ('reader','Reader'),
              ('device','Device')]
    changed = [label for field,label in fields if a.get(field) != b.get(field)]
    for field,label in [('prompt_parser_sha256','Prompt/parser'),
                        ('decoding','Decoding'), ('seed','Seed')]:
        if a.get('invariants',{}).get(field) != b.get('invariants',{}).get(field):
            changed.append(label)
    return changed
