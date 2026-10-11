# Week 5 Content Allocator: Artifact Inventory

## Execution State
- **Notebook:** `notebooks/week5/week5_content_aware_allocator.ipynb` (Executed manually via VS Code / Jupyter, 17 code cells completed).
- **Environment:** Linux x86_64, Python 3.11.16, PyTorch 2.14.0+cu130, CUDA Available (NVIDIA TITAN RTX, 24 GB VRAM).
- **Seeds Evaluated:** 42, 123, 2026 across all 8 experimental cells (I1-T1, I1-T2, I2-T1, I2-T2, I3-T1, I3-T2, I4-T1, I4-T2).
- **Primary Seed:** 42.

## Artifact Manifest
| Component | Artifact Path | Format | Status | Description |
| :--- | :--- | :--- | :--- | :--- |
| **Extracted Runs** | `results/week5/content_allocator/extracted_experiment_runs.pt` | PyTorch binary | Preserved | Exact memory snapshot of all 24 runs (8 cells x 3 seeds) with predictions, raw logits/regrets, training histories, state dicts |
| **Main Results Table** | `results/week5/content_allocator/tables/main_results.csv` | CSV | Generated | Full policy metrics for static baselines, W4 best, and all 8 learned cells |
| **Information Ablation** | `results/week5/content_allocator/tables/information_ablation.csv` | CSV | Generated | I1 vs I2 vs I3 vs I4 breakdown for T1 and T2 |
| **Target Comparison** | `results/week5/content_allocator/tables/target_comparison.csv` | CSV | Generated | T1 classification vs T2 regret regression side-by-side |
| **Bootstrap Results** | `results/week5/content_allocator/bootstrap/paper_clustered_bootstrap_results.csv` | CSV | Generated | 2,000 cluster resamples vs static k=8 |
| **Pareto Plot** | `results/week5/content_allocator/figures/pareto_quality_vs_tokens.png` | PNG | Generated | Quality F1 vs Context Tokens tradeoff plot |
| **Budget Dist Plot** | `results/week5/content_allocator/figures/k_distribution.png` | PNG | Generated | Selected budget distribution for best model |
| **TRAIN Supervision** | `results/week5/supervision/train_allocation_supervision.csv` | CSV | Canonical | 2,088 queries with fixed-k F1, oracle_k, and regret columns |
| **DEV Supervision** | `results/week5/supervision/dev_allocation_supervision.csv` | CSV | Canonical | 231 queries with fixed-k F1, oracle_k, and regret columns |
| **Config** | `configs/week5/content_allocator.yaml` | YAML | Canonical | Architecture, training, and evaluation hyperparameters |
