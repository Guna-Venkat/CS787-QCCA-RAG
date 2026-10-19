"""Dependency-isolated adapter to LOCAL SANE helpers; originals remain unchanged."""
from __future__ import annotations
import ast
from functools import lru_cache
from .provenance import ROOT, file_hash

REFERENCE = ROOT / 'Baselines/SANE-main'


@lru_cache(maxsize=1)
def reference_namespace():
    prompts = REFERENCE / 'sane/agents/sane_prompts.py'
    helpers = REFERENCE / 'sane/agents/sane_helpers.py'
    ns = {'__name__': 'week9_local_sane_reference'}
    exec(compile(prompts.read_text(), str(prompts), 'exec'), ns)
    tree = ast.parse(helpers.read_text())
    # Remove ONLY imports of SANE/OpenAI; pure helper bodies are unmodified.
    tree.body = [node for node in tree.body if not
                 (isinstance(node, ast.ImportFrom) and (node.module or '').startswith('sane.'))]
    exec(compile(tree, str(helpers), 'exec'), ns)
    return ns


def fingerprints():
    return {str(p.relative_to(ROOT)): file_hash(p) for p in
        [REFERENCE / 'sane/agents/sane_helpers.py', REFERENCE / 'sane/agents/sane_prompts.py',
         REFERENCE / 'sane/agents/sane_agent.py', REFERENCE / 'config/default.yaml']}


def select_synopses(question, candidates, evidence_k, client):
    n = min(evidence_k, len(candidates))
    if n == 0:
        return [], ''
    ns = reference_namespace()
    docs = [[dict(role='user', content=c['text'])] for c in candidates]
    synopsis, mapping = ns['generate_doc_synopses'](docs, ['']*len(docs), list(range(len(docs))),
        keyword_n=10, labels=[c['chunk_id'] for c in candidates], query=question, include_excerpt=True)
    positions = ns['select_docs'](client, question, synopsis, mapping, max_select=n)
    for i in range(len(candidates)):
        if len(positions) >= n:
            break
        if i not in positions:
            positions.append(i)
    return [candidates[i] for i in positions], synopsis
