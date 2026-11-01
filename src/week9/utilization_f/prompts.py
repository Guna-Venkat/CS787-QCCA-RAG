"""Predeclared answer-only interfaces. Never search prompts after FIT outcomes."""
from ..provenance import assert_inference_clean, digest
from ..revision import revision_messages

F1 = (
    'The initial answer may already be correct. Review the additional source evidence. '
    'Preserve the original answer unless the evidence clearly requires a correction or addition. '
    'Make the smallest necessary change. Do not add unsupported information. '
    'Use only the supplied source evidence for any correction or addition. '
    'Output only the final concise answer, without reasoning or chain-of-thought.'
)
F2_A = (
    'Answer the question using the supplied source evidence. Treat the source evidence as authoritative. '
    'The initial answer is only a starting point and may be incomplete or wrong. '
    'Preserve parts of the initial answer only when they remain supported by the evidence. '
    'Add missing information supported by the evidence, correct conflicting claims, and remove unsupported material when necessary. '
    'Do not invent facts. Output only the final concise answer, without reasoning or chain-of-thought.'
)
F2_B = (
    'Answer the question using the supplied source evidence as the authoritative source. '
    'Use the initial answer as a starting point, not as an authority; it may be incomplete or wrong. '
    'Consider information missing from the draft, claims contradicted by the evidence, and supported draft content worth preserving. '
    'Add supported missing information, correct contradictions, and remove unsupported material when necessary. '
    'Do not invent facts. Output only the final concise answer; do not output these categories, reasoning, or chain-of-thought.'
)
INSTRUCTIONS = {'F1': F1, 'F2-A': F2_A, 'F2-B': F2_B}
POLICIES = ('F0', 'F1', 'F2-A', 'F2-B')
F2_VARIANTS = ('F2-A', 'F2-B')


def messages(inference, evidence, policy):
    assert_inference_clean(inference)
    # Whitelist fields: outcomes/evaluation never enter the generation payload.
    clean = dict(question=inference['question'], initial_draft=inference['initial_draft'])
    if policy == 'F0':
        return revision_messages(clean, evidence, False)
    if policy not in INSTRUCTIONS:
        raise ValueError('Unknown preregistered W9-F policy')
    content = (INSTRUCTIONS[policy] + '\n\nInitial answer: ' + clean['initial_draft']
               + '\n\nAdditional evidence:\n' + evidence + '\n\nQuestion: ' + clean['question'])
    return [dict(role='user', content=content)]


def definitions():
    inf = dict(question='{question}', initial_draft='{initial_draft}')
    return {p: dict(messages=messages(inf, '{evidence}', p),
                    prompt_hash=digest(messages(inf, '{evidence}', p))) for p in POLICIES}
