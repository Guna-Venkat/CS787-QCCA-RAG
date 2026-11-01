"""Unchanged canonical context plus controlled regeneration/revision interventions."""
from .provenance import assert_inference_clean

REVISION_INSTRUCTION = ('You produced an initial answer. Additional source-grounded evidence is now available. '
    'Re-evaluate the initial answer using this evidence. If the initial answer is already complete and supported, '
    'preserve it. If the evidence adds missing information or corrects the answer, revise it. '
    'Return only the final concise complete answer.')


def revision_messages(inference, evidence, draft_aware):
    assert_inference_clean(inference)
    mode=normalize_mode(draft_aware)
    if mode=='F2':
        content=F2_INSTRUCTION+'\n\nInitial answer: '+inference['initial_draft']
    elif mode=='F1':
        content = REVISION_INSTRUCTION + '\n\nInitial answer: ' + inference['initial_draft']
    else:
        content = 'Answer the question using the additional source-grounded evidence. Return only a concise complete answer.'
    content += '\n\nAdditional evidence:\n' + evidence + '\n\nQuestion: ' + inference['question']
    return [dict(role='user', content=content)]


def revise(inference, evidence, client, draft_aware=True):
    if not evidence.strip():
        return inference['initial_draft']
    return client.chat(revision_messages(inference, evidence, draft_aware), max_tokens=client.answer_max_tokens)['content'].strip()


F2_INSTRUCTION = ('You are given a question, an initial answer, and newly extracted source-grounded evidence. '
    'Produce the most accurate and complete final answer supported by the evidence. '
    'Use the initial answer as a starting point, not as an authority. '
    'Preserve portions of the initial answer that remain correct. '
    'Add information that the evidence shows is missing. '
    'Correct portions that conflict with the evidence. '
    'Do not introduce unsupported facts. Return only the final concise answer.')


def normalize_mode(value):
    if isinstance(value,bool):return 'F1' if value else 'F0'
    if value not in ('F0','F1','F2'):raise ValueError('Unknown revision mode')
    return value


def prompt_definition(mode):
    from .provenance import digest
    mode=normalize_mode(mode)
    inf=dict(question='{question}',initial_draft='{initial_draft}')
    messages=revision_messages(inf,'{evidence}',mode)
    return dict(mode=mode,messages=messages,prompt_hash=digest(messages),variant='predeclared-v1',no_prompt_search=True)


def write_revision_freeze(cfg,settings):
    from .data import resolve
    from .caching import atomic_json
    from .provenance import branch_guard,digest
    import json
    mode=normalize_mode(settings['revision'])
    directory=resolve(cfg['output_root'])/'revision'
    values={'f2_frozen_prompt.json':dict(prompt_definition('F2'),selection_split='fit',frozen=True),
        'frozen_revision_mode.json':dict(prompt_definition(mode),selection_split='fit',frozen=True,
            phase_settings_hash=digest(settings))}
    branch_guard()
    for name,value in values.items():
        path=directory/name
        if path.exists() and json.loads(path.read_text())!=value:raise FileExistsError('Refusing to replace frozen revision definition')
        atomic_json(path,value)


def read_revision_mode(cfg,frozen_f):
    import json
    from .data import resolve
    from .provenance import digest
    value=json.loads((resolve(cfg['output_root'])/'revision/frozen_revision_mode.json').read_text())
    mode=normalize_mode(frozen_f['settings']['revision'])
    if (not value.get('frozen') or value.get('selection_split')!='fit' or value.get('mode')!=mode or
        value.get('phase_settings_hash')!=digest(frozen_f['settings']) or value['prompt_hash']!=prompt_definition(mode)['prompt_hash']):
        raise ValueError('Revision mode/prompt does not match the verified FIT freeze')
    return mode
