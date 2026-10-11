# Week 5 Preflight: Repository & Path Audit Report

**Date:** October 2026  
**Auditor:** SARA Research Engineering Agent  
**Status:** COMPLETE — REPOSITORY CLEAN & WEEK 4 CONSOLIDATED  

---

## 1. Executive Summary

Following the consolidation of previous Week 5 nonlinear modeling experiments into Week 4, a full repository audit was conducted across all module paths, test suites, configs, notebooks, and artifact directories. All legacy references have been resolved, and the repository is cleanly partitioned.

---

## 2. Week 4 Consolidation & Current Structure

All artifacts, modeling code, test files, and notebooks from the historical nonlinear benchmark have been moved under `week4`:

| Component | Consolidated Path | Status |
| :--- | :--- | :--- |
| **Model Benchmark Library** | `src/week4/model_benchmark.py` | Active, tested, imports from `src.week4` |
| **Benchmark Driver** | `src/week4/run_benchmark.py` | Active, verified, writes to `results/week4/model_benchmark/` |
| **Notebook Generator** | `src/week4/generate_model_benchmark_notebook.py` | Active, builds unexecuted notebook |
| **Benchmark Notebook** | `notebooks/week4/week4_model_benchmark.ipynb` | Regenerated, clean imports |
| **Benchmark Test Suite** | `tests/week4/test_model_benchmark.py` | 12/12 unit tests passing |
| **Benchmark Results** | `results/week4/model_benchmark/` | 10 CSV tables, 10 figures, synthesis reports |

### Complete Test Verification
The consolidated `tests/week4/` suite executes 65 unit tests covering:
- `test_cv.py` (2 passed)
- `test_evaluation.py` (2 passed)
- `test_feature_discovery.py` (6 passed)
- `test_feature_validation.py` (7 passed)
- `test_features.py` (2 passed)
- `test_model_benchmark.py` (12 passed)
- `test_models.py` (2 passed)
- `test_qcca_v2.py` (13 passed)
- `test_qcca_v2_diagnostic.py` (8 passed)
- `test_qcca_v2_final_validation.py` (9 passed)
- `test_targets.py` (2 passed)

**Result:** **65 passed, 0 failed** (100% pass rate).

---

## 3. Stale Week-5 References & Fixes Made

1. **Import Paths:**
   - In `src/week4/model_benchmark.py` and `src/week4/run_benchmark.py`, all import statements referencing `src.week5` were updated to `src.week4`.
   - In `tests/week4/test_model_benchmark.py`, imports updated to `from src.week4.model_benchmark import ...`.
   - In `notebooks/week4/week4_model_benchmark.ipynb`, code cells updated to import from `src.week4`.

2. **Output Directories:**
   - Updated `OUTPUT_DIR` in `run_benchmark.py` and `generate_model_benchmark_notebook.py` to `results/week4/model_benchmark`.
   - Updated image and CSV references in `results/week4/model_benchmark/week5_modeling_report.md` to `results/week4/model_benchmark/`.
   - Created alias `model_benchmark_report.md` for consistent naming.

3. **Complete Removal of Old Week 5 Dirs:**
   - `rm -rf results/week5 src/week5 tests/week5 notebooks/week5` executed.
   - Verified with filesystem searches: zero stale week5 directories remain.

---

## 4. Frozen Status of Weeks 0–4

- **Week 0:** FROZEN (17/17 tests passing, reports locked in `results/week0/`).
- **Weeks 1–3:** FROZEN (Response surfaces in `results/week2/`, oracle definitions in `results/week3/`).
- **Week 4:** FROZEN (All linear and nonlinear QCCA policies, GroupKFold evaluations, and benchmark results are permanently archived under `results/week4/`).
