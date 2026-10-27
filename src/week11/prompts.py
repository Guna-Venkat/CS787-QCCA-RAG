def user_prompt(question,evidence,task_type):
    if task_type=='verification':
        instruction='Classify the claim using the evidence. Output exactly one label: SUPPORT, REFUTE, or NOT_ENOUGH_INFO.'
        field='Claim'
    else:
        instruction='Answer the question in a single phrase or short sentence using the context if provided.'
        field='Question'
    return f'{instruction}\nContext:\n{evidence}\n{field}: {question}'
