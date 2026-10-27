"""Checksummed immutable stores; namespaces isolate fabricated and real work."""
from src.week10.root_cause.storage import Store, atomic, digest, file_hash, lock


class Cache:
    def __init__(self, root, namespace, identity):
        self.store = Store(root/namespace, identity)

    def get(self, key):
        return self.store.get(key)

    def put(self, key, value):
        return self.store.put(key, value)


def selection_key(example, action, cfg, tokenizer_identity):
    from .protocol import guard
    guard(example['dataset'],example['split'])
    return dict(dataset=example['dataset'], split=example['split'], query_id=example['query_id'],
        source_hash=digest(example['inference']), retriever=action['retriever'],
        granularity=action['granularity'], selector=action['selector'], budget=action['budget'],
        compressor=action['compressor'], embedding_revision=cfg['models']['SFR']['revision'],
        chunking_hash=digest(cfg['hierarchy']), tokenizers=tokenizer_identity,
        action_hash=digest(action))
