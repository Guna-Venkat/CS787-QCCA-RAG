"""Lightweight tests for the run-orchestration helpers.

Runnable on CPU with no GPU/HF dependencies::

    cd <repo-root>
    PYTHONPATH=. python -m pytest tests/test_run_utils.py -q
"""

from __future__ import annotations

from pathlib import Path

import pytest

from src.utils.run_utils import (
    iter_matrix,
    make_output_path,
    parse_csv,
    parse_int_csv,
    parse_kn_pairs,
    should_skip,
)


def test_parse_csv_strips_and_drops_empty():
    assert parse_csv("a, b ,,c") == ["a", "b", "c"]


def test_parse_int_csv():
    assert parse_int_csv("1, 2,3") == [1, 2, 3]


def test_parse_kn_pairs():
    assert parse_kn_pairs(["5,5", "5,10", "0, 0"]) == [(5, 5), (5, 10), (0, 0)]


def test_parse_kn_pairs_rejects_bad_format():
    with pytest.raises(ValueError):
        parse_kn_pairs(["5"])
    with pytest.raises(ValueError):
        parse_kn_pairs(["5,5,5"])


def test_make_output_path_kn_with_evi_select():
    p = make_output_path(
        model="Mistral7B_QA_Compress_Stage0_Stella",
        retriever="bm25",
        dataset="qasper",
        k=5, n=10,
        repetition_penalty=1.5,
        evidence_selection="self-info",
    )
    assert p == Path(
        "outputs/Mistral7B_QA_Compress_Stage0_Stella/"
        "answers_Mistral7B_QA_Compress_Stage0_Stella_bm25_qasper_k5_n10_rep1.5_evi-select-self-info.jsonl"
    )


def test_make_output_path_kn_minimal():
    """Mirrors Eval_Mistral7B_Vary_k.sh, which omitted rep / evi-select."""
    p = make_output_path(
        model="Mistral7B", retriever="bm25", dataset="qasper", k=3, n=3,
    )
    assert p == Path("outputs/Mistral7B/answers_Mistral7B_bm25_qasper_k3_n3.jsonl")


def test_make_output_path_retriever_basename_only():
    """Slashed retriever names (BAAI/bge-reranker-v2-m3) collapse to basename."""
    p = make_output_path(
        model="M", retriever="BAAI/bge-reranker-v2-m3", dataset="qasper", k=5, n=10,
    )
    assert "bge-reranker-v2-m3" in p.name
    assert "BAAI" not in p.name


def test_iter_matrix_product_order():
    combos = list(iter_matrix(model=["A", "B"], dataset=["x", "y"]))
    assert combos == [
        {"model": "A", "dataset": "x"},
        {"model": "A", "dataset": "y"},
        {"model": "B", "dataset": "x"},
        {"model": "B", "dataset": "y"},
    ]


def test_should_skip(tmp_path: Path):
    f = tmp_path / "a.jsonl"
    assert should_skip(f, skip_existing=True) is False
    f.write_text("{}")
    assert should_skip(f, skip_existing=True) is True
    assert should_skip(f, skip_existing=False) is False
