"""Unchanged canonical context plus controlled regeneration/revision interventions."""
from .provenance import assert_inference_clean

REVISION_INSTRUCTION = ('You produced an initial answer. Additional source-grounded evidence is now available. '
    'Re-evaluate the initial answer using this evidence. If the initial answer is already complete and supported, '
    'preserve it. If the evidence adds missing information or corrects the answer, revise it. '
    'Return only the final concise complete answer.')


def revision_messages(inference, evidence, draft_aware):
    assert_inference_clean(inference)
    if draft_aware:
        content = REVISION_INSTRUCTION + '\n\nInitial answer: ' + inference['initial_draft']
    else:
        content = 'Answer the question using the additional source-grounded evidence. Return only a concise complete answer.'
    content += '\n\nAdditional evidence:\n' + evidence + '\n\nQuestion: ' + inference['question']
    return [dict(role='user', content=content)]


def revise(inference, evidence, client, draft_aware=True):
    if not evidence.strip():
        return inference['initial_draft']
    return client.chat(revision_messages(inference, evidence, draft_aware), max_tokens=client.answer_max_tokens)['content'].strip()
