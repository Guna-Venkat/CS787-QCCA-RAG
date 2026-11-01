import importlib.util
from pathlib import Path

import pytest

from src.week10.root_cause.storage import Store, atomic, digest

path = Path(__file__).resolve().parents[2] / 'scripts/week10/report_full_panel.py'
spec = importlib.util.spec_from_file_location('full_panel_report', path)
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


def test_evidence_identity_rejects_changed_text_or_paper():
    row = dict(question_id='q', paper_id='p', action='full', evidence_text='Exact source.',
               evidence_hash=digest('Exact source.'))
    module.evidence_identity([row], {'Llama': [row]})
    for change in [dict(evidence_text='Altered source.'), dict(paper_id='other'),
                   dict(evidence_hash='wrong')]:
        with pytest.raises(ValueError, match='evidence mismatch'):
            module.evidence_identity([row], {'Llama': [dict(row, **change)]})


def test_completion_requires_expected_keys_and_receipt(tmp_path):
    identity = dict(freeze_hash='frozen')
    store = Store(tmp_path / 'runs/G-Llama', identity)
    store.put(['q', 'full'], dict(question_id='q'))
    with pytest.raises(ValueError, match='Incomplete'):
        module.complete_rows(tmp_path, 'G-Llama', [['q', 'full']], 'frozen')
    atomic(store.path / 'complete.json', dict(queries=1))
    assert len(module.complete_rows(tmp_path, 'G-Llama', [['q', 'full']], 'frozen')) == 1
    with pytest.raises(ValueError, match='Incomplete'):
        module.complete_rows(tmp_path, 'G-Llama', [['other', 'full']], 'frozen')
    with pytest.raises(ValueError, match='Mismatched'):
        module.complete_rows(tmp_path, 'G-Llama', [['q', 'full']], 'changed')


def test_reader_utility_uses_own_paired_baseline():
    base = [dict(question_id='q', paper_id='p', f1=.2)]
    rows = [dict(question_id='q', paper_id='p', f1=.6)]
    assert module.utility(rows, base)[0]['f1'] == pytest.approx(.4)
    with pytest.raises(ValueError, match='paper identity'):
        module.utility([dict(rows[0], paper_id='other')], base)
