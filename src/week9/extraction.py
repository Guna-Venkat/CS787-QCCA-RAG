"""One joint extraction call, holding backend/decoding fixed across Q vs QD."""
from .blueprints import system_prompt
from .sane_adapter import reference_namespace
from .faithfulness import audit_extraction, supported_evidence
from .provenance import assert_inference_clean


def extract(inference, selected, client, *, draft_aware=False, format_name='sane_blueprint', max_tokens=1024):
    assert_inference_clean(inference)
    assert_inference_clean(selected)
    if not selected:
        return dict(text='', raw_text='', audit=audit_extraction('', []), selected_chunk_ids=[])
    content = '\n\n'.join('## Source ' + c['chunk_id'] + '\n' + c['text'] for c in selected)
    question = inference['question']
    if draft_aware:
        question += ('\n\n## Initial draft\n' + inference['initial_draft'] +
            '\nExtract information that confirms, completes, or corrects this initial answer.')
    raw = reference_namespace()['extract_evidence'](client, question, content,
        prompt_template=system_prompt(format_name), max_tokens=max_tokens)
    audit = audit_extraction(raw, selected)
    # E0 enforces spans only. E1 preserves a valid table for faithful-format analysis;
    # E2 is the unaltered reference table including unsupported rows for measurement.
    evidence = supported_evidence(audit) if format_name == 'extractive' else raw
    return dict(text=evidence, raw_text=raw, audit=audit, selected_chunk_ids=[c['chunk_id'] for c in selected],
                draft_aware=draft_aware, format=format_name,
                source_chunks=[dict(chunk_id=c['chunk_id'], text=c['text']) for c in selected])
