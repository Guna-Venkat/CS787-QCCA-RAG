"""Explicit reporting-only provenance, preserving existing inference/cache identities."""
from src.post_w12.bea_qbgf import protocol as p

PATCH=p.ROOT/'W12-POST/REPORTING_PATCH.json'
ALLOWED={'src/post_w12/bea_qbgf/analysis.py','src/post_w12/bea_qbgf/protocol.py'}

def canonical_implementation(actual):
    """Accept only the exact documented repair; all scientific sources stay strict."""
    if not PATCH.exists():return actual
    patch=p.read(PATCH)
    if set(patch['identity_preserving_sources'])!=ALLOWED:raise ValueError('Invalid reporting-only patch scope')
    for name,h in patch['reporting_sources'].items():
        if p.sha(p.ROOT/name)!=h:raise ValueError('Unreviewed reporting repair: '+name)
    canonical=dict(actual)
    for name,change in patch['identity_preserving_sources'].items():
        if actual.get(name)!=change['repaired_sha']:raise ValueError('Reporting source checksum mismatch: '+name)
        canonical[name]=change['original_sha']
    return canonical

def record(c,f):
    # implementation() already verified the exact patch before returning the freeze.
    p.save(p.output(c)/'reporting/REPAIR.json',dict(p.read(PATCH),identity_hash=f['identity_hash']))
