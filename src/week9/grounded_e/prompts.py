"""Preregistered v1 templates; no outcome-driven prompt selection or repairs."""
import json

POLICIES = ('E0', 'E1-Q', 'E1-QD', 'E2')
COMMON = '''Extract compact evidence from the provided sources. Sources are authoritative; never invent information. Return ONLY a JSON array, at most two objects. No markdown, preamble, rationale, explanation, or chain-of-thought. Prefer one short item directly answering the question. If no useful supported item exists, return [].
Copy support_span verbatim from its named source (at most 24 tokens). Copy fact or normalized_claim as a short contiguous phrase FROM THAT SUPPORT SPAN (at most 12 tokens); only whitespace normalization is allowed. Do not paraphrase or add claims. Use the exact source_chunk_id. Do not use the draft as evidence. Keep the entire representation, including keys and IDs, within {budget} tokens. Invalid items will be rejected.'''
DRAFT = '''The initial draft may be incomplete or wrong. Use it only to identify missing, uncertain, or conflicting information. Extract source evidence relevant to correcting or completing the answer. Do not preserve unsupported draft claims. Source evidence, not the draft, is authoritative.'''
STRUCTURED = '''Schema: [{"source_chunk_id":"COPY_ID","support_span":"EXACT_QUOTE","fact":"SHORT_PHRASE_COPIED_FROM_QUOTE"}]. No additional fields.'''
BLUEPRINT = '''Use a minimal source-grounded blueprint, not a verbose table.
Schema: [{"source_chunk_id":"COPY_ID","support_span":"EXACT_QUOTE","normalized_claim":"SHORT_PHRASE_COPIED_FROM_QUOTE","evidence_type":"answer_detail","relevance":"direct"}].
Allowed evidence_type labels: answer_detail, correction, qualification, definition. Allowed relevance labels: direct, missing, conflicting. These are brief organizational labels, not extra factual statements. No additional fields.'''
E0_DEFINITION = 'Deterministic query-only BM25 ranking of source sentences; greedy exact-offset spans packed into the final evidence-token budget. No LM extraction or generated facts.'


def messages(policy, inference, chunks, budget):
    if policy not in POLICIES[1:]:
        raise ValueError('Only structured policies have an LM extraction prompt')
    instruction = COMMON.replace('{budget}', str(budget)) + '\n' + (BLUEPRINT if policy == 'E2' else STRUCTURED)
    payload = dict(question=inference['question'], sources=[dict(source_chunk_id=c['chunk_id'], text=c['text']) for c in chunks])
    if policy in ('E1-QD', 'E2'):
        instruction += '\n' + DRAFT
        payload['initial_draft'] = inference['initial_draft']
    return [dict(role='system', content=instruction), dict(role='user', content=json.dumps(payload, ensure_ascii=False))]
