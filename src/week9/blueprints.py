"""Actual local SANE schemas; project formats carry distinct labels."""
from .sane_adapter import reference_namespace


def system_prompt(format_name):
    if format_name == 'sane_blueprint':
        return reference_namespace()['BLUEPRINT_GUIDED_PROMPT']
    if format_name == 'sane_plain':
        return reference_namespace()['EVIDENCE_EXTRACTION_PROMPT']
    if format_name == 'extractive':
        return ('Extract only relevant contiguous verbatim source spans. Output a Markdown table with '
                'columns support_span, source_chunk_id. No paraphrase, elision or answer.')
    if format_name == 'structured':
        return ('Extract only relevant evidence. Output a Markdown table with columns support_span, '
                'structured_fact, source_chunk_id. Each support_span must be an exact contiguous source quote. '
                'Do not add information absent from the source. No preamble or answer.')
    raise ValueError(format_name)
