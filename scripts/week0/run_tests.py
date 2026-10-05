"""Standalone test runner for Week 0."""

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# Ensure SARA-main src is added
SARA_ROOT = PROJECT_ROOT / "Baselines" / "SARA-main"
import src
if hasattr(src, "__path__") and str(SARA_ROOT / "src") not in src.__path__:
    src.__path__.append(str(SARA_ROOT / "src"))

from tests.week0.test_week0 import (
    test_calculate_evidence_depth,
    test_check_passage_matches_evidence,
    test_classify_wh_type,
    test_compute_concentration_metrics,
    test_compute_evidence_redundancy,
    test_compute_lexical_overlap,
    test_concentration_all_negative_edge_case,
    test_concentration_signed_ablation_handling,
    test_deterministic_sampling,
    test_extract_query_characteristics,
    test_generated_output_schemas,
    test_qasper_adapter_real_load,
    test_synthetic_attention_passage_extraction,
    test_production_pipeline_disallows_mock_empirical_writes,
    test_gpu_pilot_artifacts_schema_and_validity,
    test_intervention_validation_artifacts,
    test_final40_artifacts_schema_and_validity,
)


def run_all_tests():
    print("Running Week 0 Unit Tests...", flush=True)

    test_classify_wh_type()
    print("PASS: test_classify_wh_type", flush=True)

    test_extract_query_characteristics()
    print("PASS: test_extract_query_characteristics", flush=True)

    test_compute_lexical_overlap()
    print("PASS: test_compute_lexical_overlap", flush=True)

    test_compute_evidence_redundancy()
    print("PASS: test_compute_evidence_redundancy", flush=True)

    test_calculate_evidence_depth()
    print("PASS: test_calculate_evidence_depth", flush=True)

    test_check_passage_matches_evidence()
    print("PASS: test_check_passage_matches_evidence", flush=True)

    test_compute_concentration_metrics()
    print("PASS: test_compute_concentration_metrics", flush=True)

    test_concentration_signed_ablation_handling()
    print("PASS: test_concentration_signed_ablation_handling", flush=True)

    test_concentration_all_negative_edge_case()
    print("PASS: test_concentration_all_negative_edge_case", flush=True)

    test_synthetic_attention_passage_extraction()
    print("PASS: test_synthetic_attention_passage_extraction", flush=True)

    test_deterministic_sampling()
    print("PASS: test_deterministic_sampling", flush=True)

    test_generated_output_schemas()
    print("PASS: test_generated_output_schemas", flush=True)

    test_qasper_adapter_real_load()
    print("PASS: test_qasper_adapter_real_load", flush=True)

    test_production_pipeline_disallows_mock_empirical_writes(None)
    print("PASS: test_production_pipeline_disallows_mock_empirical_writes", flush=True)

    test_gpu_pilot_artifacts_schema_and_validity()
    print("PASS: test_gpu_pilot_artifacts_schema_and_validity", flush=True)

    test_intervention_validation_artifacts()
    print("PASS: test_intervention_validation_artifacts", flush=True)

    test_final40_artifacts_schema_and_validity()
    print("PASS: test_final40_artifacts_schema_and_validity", flush=True)

    print("\n==========================================", flush=True)
    print("ALL 17 WEEK 0 UNIT TESTS PASSED!", flush=True)
    print("==========================================", flush=True)


if __name__ == "__main__":
    run_all_tests()
