# Manual execution and resume

Open `notebooks/week10/week10_root_cause.ipynb` with sara_env. Defaults validate the audit hashes and run synthetic smoke only. No LM or embedding model loads.

Phase R only: `SMOKE_ONLY=False`, `RUN_RETRIEVER_FULL=True`; keep all other full flags False. Three retrieval jobs run first (BM25, MiniLM, SFR), then six FULL/SARA jobs serially. At DEV231, 3x4x2x231=5,544 generations; planning estimate 5–12 GPU hours plus SFR indexing/loading. This is an estimate, not a measured W10 throughput.

Resume by rerunning the same cell with the SAME config/output root. Query/action checkpoints are atomic and checksummed. Completed answers are skipped before loading a model. Failed calls have no completed checkpoint and retry individually. Model/data/source changes fail loudly and require a new version. Do not clear caches or edit the method mid-run. One writer per arm is enforced.

After R, run the report cell. Review source-span alignment/unmapped-evidence limitations, recall, matched F1/EM and paired paper CIs. `GO_NEXT=False` until an explicit phase-review freeze. C/G/D otherwise reject execution. C uses identical reader weights and the same frozen candidate pool; LEAD/QCS compare textual conditioning under the same ceiling. The faithful SeleCom adapter remains gated by Turing compatibility/memory verification. The textual surrogate must not be called SeleCom.

G additionally requires panel/model feasibility review. Compatible older readers are provisional; Qwen3.5-9B needs a separate runtime integration. D includes paired seed baselines, not independent noisy comparisons. P is disabled in code. Stop controller work if corrected fixed upstream choices remove meaningful residual heterogeneity.

No W0–W9 outputs are writable by this implementation. QASPER TEST is rejected.
