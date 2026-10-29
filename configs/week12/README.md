# Frozen Week12 design

See `protocol.yaml`, `notebooks/week12/README.md`, and the implementation report under `results/week12/`.

- Primary reader: pinned Qwen3-8B FP16; SFR pinned from W11.
- N=10; tau_p=tau_s=5; budgets 256/1024.
- NO_QUERY: uniform passage weights, uniform sentence mass, source-order sentence choices; upstream query-dependent retrieval held fixed.
- NO_MASS: R_i * min(k,m_i)/m_i; query-ranked sentence choices unchanged.
- Greedy: deterministic feasible marginal utility/token upgrades; no fitted controller.
- Source/cost ties: unit source order then stable ID; DP ties at 1e-12 prefer fewer tokens then lexicographic option indices. Equivalent sentence k values are deduplicated.
- Exact DP objective is the additive relevance surrogate. It does not optimize true F1 or model factual support.
- Final selection defaults to fixed semantic sentences. Non-simple candidates must pass the documented conservative paired noninferiority/token admissibility checks over all six dataset/budget cells; smoke cannot freeze a winner. This may remain inconclusive on 50-query cells.
- LoRA is NF4 QLoRA with a matched BASE_NF4 control, not a causal comparison against the FP16 main surface.
- Candidate confirmation reservations have `history_status=REVIEW_REQUIRED`. No Week12 entrypoint evaluates confirmation data. Unknown historical exposure remains a blocker for Week13.
