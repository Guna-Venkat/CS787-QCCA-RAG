"""Unit tests for fixed-k partition mapping contract."""

import pytest
from typing import List, Tuple, Dict, Any


def partition_passages(passages: List[Dict[str, Any]], k: int) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    """Partition 10 passages into k natural-language and 10-k compressed."""
    assert len(passages) == 10, f"Expected exactly 10 passages, got {len(passages)}"
    assert 0 <= k <= 10, f"Invalid k: {k}"
    text_passages = passages[:k]
    compressed_passages = passages[k:10]
    return text_passages, compressed_passages


def test_k_partition_counts():
    mock_passages = [{"rank": i + 1, "id": f"p_{i}"} for i in range(10)]
    
    expected_contracts = {
        10: (10, 0),
        8: (8, 2),
        6: (6, 4),
        5: (5, 5),
        4: (4, 6),
        2: (2, 8),
        0: (0, 10),
    }
    
    for k, (exp_text, exp_comp) in expected_contracts.items():
        text_p, comp_p = partition_passages(mock_passages, k)
        assert len(text_p) == exp_text, f"Failed text count for k={k}"
        assert len(comp_p) == exp_comp, f"Failed comp count for k={k}"
        assert len(text_p) + len(comp_p) == 10
        
        # Verify rank ordering: text is ranks 1..k
        if exp_text > 0:
            assert text_p[0]["rank"] == 1
            assert text_p[-1]["rank"] == k
        # Compressed is ranks k+1..10
        if exp_comp > 0:
            assert comp_p[0]["rank"] == k + 1
            assert comp_p[-1]["rank"] == 10


def test_order_invariance():
    # Passages must NEVER change relative ordering across k
    mock_passages = [{"rank": i + 1, "text": f"Content {i + 1}"} for i in range(10)]
    
    for k in [0, 2, 4, 5, 6, 8, 10]:
        text_p, comp_p = partition_passages(mock_passages, k)
        recombined = text_p + comp_p
        assert [p["rank"] for p in recombined] == list(range(1, 11))
