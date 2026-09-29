"""Unit tests for SFR embedding cache structure and traceability."""

import pytest
import torch
from pathlib import Path


def test_mock_embedding_cache_structure():
    # Simulate an embedding cache dict as constructed by precompute_dev_embeddings
    num_questions = 3
    num_passages = 10
    dim = 4096
    
    cache = {
        "metadata": {
            "model_name": "Salesforce/SFR-Embedding-Mistral",
            "dtype": "bfloat16",
            "dim": dim,
            "num_questions": num_questions,
            "passages_per_question": num_passages,
        },
        "embeddings": {},
        "passage_keys": {}
    }
    
    for q_idx in range(num_questions):
        qid = str(1000 + q_idx)
        pid = f"paper_{q_idx}"
        # (10, 4096) tensor
        embeds = torch.randn(num_passages, dim, dtype=torch.bfloat16)
        cache["embeddings"][qid] = embeds
        cache["passage_keys"][qid] = [
            (pid, f"{pid}_chunk_{r}", r + 1) for r in range(num_passages)
        ]
        
    # Verify metadata
    assert cache["metadata"]["dim"] == 4096
    assert cache["metadata"]["num_questions"] == 3
    assert cache["metadata"]["passages_per_question"] == 10
    
    # Verify per-question shapes and traceability
    for qid, tensor in cache["embeddings"].items():
        assert tensor.shape == (10, 4096)
        assert tensor.dtype == torch.bfloat16
        keys = cache["passage_keys"][qid]
        assert len(keys) == 10
        # Check ranks 1 through 10 in order
        ranks = [k[2] for k in keys]
        assert ranks == list(range(1, 11))


def test_slice_compressed_embeddings():
    # Verify that slice [k:10] matches the (10 - k) compressed passages
    dim = 4096
    full_embeds = torch.arange(10 * dim).reshape(10, dim)
    
    for k in [0, 2, 4, 5, 6, 8, 10]:
        compressed_embeds = full_embeds[k:10]
        expected_count = 10 - k
        assert compressed_embeds.shape[0] == expected_count
        assert compressed_embeds.shape[1] == dim
        
        # Verify rank mapping: ranks k+1..10
        if expected_count > 0:
            assert torch.equal(compressed_embeds[0], full_embeds[k])
            assert torch.equal(compressed_embeds[-1], full_embeds[9])
