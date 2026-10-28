import tempfile
import unittest
from operations.week11.label_compliance import LEADING_LABEL,diagnose


class LabelDiagnosticTests(unittest.TestCase):
    def test_only_unambiguous_leading_labels_are_accepted(self):
        for text,label in [('SUPPORT','SUPPORT'),(' REFUTE. explanation','REFUTE'),('not_enough_info: missing evidence','NOT_ENOUGH_INFO')]:
            self.assertEqual(LEADING_LABEL.match(text).group(1).upper(),label)
        for text in ['I think SUPPORT','SUPPORTED','REFUTED','**SUPPORT**','']:
            self.assertIsNone(LEADING_LABEL.match(text))

    def test_diagnostic_does_not_change_strict_results(self):
        row=dict(dataset='SciFact',reader='Mistral',action_id='fixture',query_id='1',evidence_granularity='SENTENCE',selector='semantic',budget=256,compressor='C0',prediction='SUPPORT. explanation',gold='SUPPORT',task_score=0.,generated_tokens=4)
        with tempfile.TemporaryDirectory() as directory:
            summary=diagnose([row],directory)
            self.assertEqual(summary.strict_accuracy.iloc[0],0.)
            self.assertEqual(summary.leading_label_diagnostic_accuracy.iloc[0],1.)
            self.assertEqual(row['task_score'],0.)


if __name__=='__main__':unittest.main()
