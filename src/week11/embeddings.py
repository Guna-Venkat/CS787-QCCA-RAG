"""Pinned W10 encoder in NEW W11 cache directories; explicit long-input pooling."""
import hashlib
import re
import numpy as np
from src.week10.root_cause.encoding import Encoder, unit
from .protocol import ROOT


class Embeddings(Encoder):
    def __init__(self,cfg,model,granularity='passage'):
        directory=ROOT/cfg['cache_root']/'embeddings'/granularity.lower()/model
        super().__init__(cfg['models'][model],directory,'cuda' if model=='SFR' else 'cpu')


class SyntheticEncoder:
    """Hash-bag-of-words fixture, explicitly NEVER a real MiniLM/SFR result."""
    def encode(self,texts,role='document',allow_compute=True):
        values=np.zeros((len(texts),64),dtype=np.float32)
        for i,text in enumerate(texts):
            for word in re.findall(r'\w+',text.lower()):
                values[i,int(hashlib.sha256(word.encode()).hexdigest()[:8],16)%64]+=1
        return values,dict(synthetic=True,embedding_lookup_compute_seconds=0.,embedding_original_compute_seconds=None)
    def close(self):pass
