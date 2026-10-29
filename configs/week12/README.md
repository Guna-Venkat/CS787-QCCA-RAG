# Active Week12 protocol — v3

`protocol.yaml` implements TRAIN → VAL → frozen confirmation. v1 is historical development; v2 is a superseded metadata-only split audit. See the active notebook guide and `results/week12/W12_CORRECTED_PROTOCOL_REPORT.md`.

- All eligible frozen TRAIN examples; no final 500-query cap. One deterministic evidence mode/budget per source query; deterministic full coverage without replacement and explicit inverse-dataset-size loss weights.
- Full frozen VAL replaces the W11 50-query pilot for C/D/E/G/H/I and checkpoint selection. W11 remains hypothesis generation only.
- QASPER TEST permanently blocked. No credible fresh QASPER confirmation is available. HotpotQA/SciFact confirmation is reserved before any v3 training or validation outcome and excluded from all Week12 scoring/gradients.
- Dataset versions are content-addressed by SHA-256. Membership and group hashes are immutable; changing a scientific configuration requires a new version.
- Pinned Qwen3-8B FP16/SFR; N=10; tau_p=tau_s=5; primary budgets 256/1024. Secondary 512 is declared only, not silently added to primary surfaces.
- QBGF is exact multiple-choice knapsack over additive Qwen token-ID costs and relevance mass, not downstream oracle F1. Source order and exact spans are preserved. RAW_SKIP and strong fixed sentence controls remain mandatory.
- NO_QUERY removes allocation conditioning within the same query-retrieved pool. NO_MASS uses uniform k/m coverage but retains query ranking. BM25 is a secondary retrieval-plus-relevance sensitivity.
- LoRA checkpoint selection minimizes equal-dataset mean VAL answer NLL, with task metrics/compliance recorded; one epoch, no confirmation access. G uses matched NF4 base/adapted readers and identical evidence.
- Final method/LoRA retention uses paired VAL evidence, token admissibility and simplicity. No full run or final method freeze was produced by the agent. Week13 execution remains blocked until I creates the final immutable freeze.

The frozen YAML/splits are unchanged. The user-authorized scheduler/execution patch is documented in W12_SCHEDULER_CORE_REPORT.md: CORE_FINAL defaults True, secondary arms require explicit opt-in, and an isolated official smaller LLMLingua-2 runtime has its own qualification lock.
