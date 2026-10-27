"""Two fabricated datasets, two documents each, three queries each. No TEST data."""
from .datasets import hotpot,scifact


def examples():
    context=[['Alpha',['Alpha supports the red treatment.','The trial reports a benefit; replication is needed.']],
             ['Beta',['Beta studies the blue treatment.','The blue treatment has no measured benefit.']]]
    qa=[dict(_id=f'fixture-hotpot-{i}',question=q,answer=a,context=context,type='bridge',supporting_facts=[['Alpha',0]])
        for i,(q,a) in enumerate([('What treatment does Alpha support?','red'),('What does Beta study?','blue'),('What is needed after the trial?','replication')])]
    corpus=[dict(doc_id=1,title='Alpha',abstract=context[0][1]),dict(doc_id=2,title='Beta',abstract=context[1][1])]
    claims=[dict(id='fixture-claim-1',claim='Alpha supports the red treatment.',evidence={'1':[dict(label='SUPPORT',sentences=[0])]}),
            dict(id='fixture-claim-2',claim='The blue treatment is beneficial.',evidence={'2':[dict(label='CONTRADICT',sentences=[1])]}),
            dict(id='fixture-claim-3',claim='Green treatment is beneficial.',evidence={})]
    return hotpot(qa)+scifact(claims,corpus)
