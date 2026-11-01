"""PROPOSITION_RULE_V1: exact contiguous clauses, not Dense X propositions."""
import re

METHOD = 'PROPOSITION_RULE_V1'


def spans(sentence):
    # Conservative punctuation boundaries; do not synthesize implicit arguments.
    cuts = [0] + [m.end() for m in re.finditer(r';\s*|\s+[—–]\s+', sentence)] + [len(sentence)]
    out=[]
    for a,b in zip(cuts,cuts[1:]):
        part=sentence[a:b]; a+=len(part)-len(part.lstrip());b-=len(part)-len(part.rstrip())
        if b>a and re.search(r'\w',sentence[a:b]):out.append((a,b))
    return out


class RulePropositions:
    method = METHOD
    def extract(self, sentence):
        return [dict(start_char=a,end_char=b,proposition_text=sentence[a:b],
                     exact_source_support=True,fuzzy_source_support=None,method=METHOD) for a,b in spans(sentence)]
