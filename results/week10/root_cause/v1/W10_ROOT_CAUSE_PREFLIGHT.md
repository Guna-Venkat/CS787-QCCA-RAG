# W10 root-cause preflight

## 1. Executive conclusion
Start with matched BM25/MiniLM/SFR retrieval, not another controller. The implementation and notebook support manual, resumable development experiments. Full experiments were not started. New-model memory qualification and faithful SeleCom integration remain explicit gates.

## 2. Why this audit was necessary
Historical oracle ceilings describe heterogeneous counterfactual outcomes, not deployable quality. TEST is now closed. Repeated HELP/HURT cancellation has several plausible interacting causes.

## 3. Exact SARA reproduction comparison
See `SARA_REPRODUCTION_AUDIT.md` and the source-linked crosswalk. Local backbone architecture matches; trained-checkpoint equivalence and full paper refinement do not. Do not label this an exact paper reproduction.

## 4. Retriever discrepancy
No completed matched SFR retrieval evaluation was located. MiniLM's small frozen scorer became the semantic routing foundation without located evidence that it beat SFR. Three-way full-paper retrieval is mandatory.

## 5. Compression literature
The literature matrix separates textual pruning, latent encoders, query conditioning and reader adaptation. Its results come from different datasets/protocols and do not establish a universal ranking.

## 6. Query-conditioned compression evidence
SeleCom's released pipeline is a candidate for faithful checkpoint evaluation; attention/runtime feasibility is pending. C uses QCS256 as an explicitly named textual surrogate, not as a reproduction. LEAD256 controls for sentence suppression at the same token ceiling. A strong learned textual compressor is still a candidate integration, not silently replaced by LEAD.

## 7. Generator/reader candidates
Mistral remains the control. Qwen3-8B nonthinking and Llama3.1-8B are provisional compatible alternatives, not guaranteed stronger or newest. Qwen3.5-9B is a contemporary candidate with unqualified runtime support. Pin and qualify at most two alternatives before reader outcomes; do not choose by QASPER scores.

## 8. Adaptive-policy literature
Direct-reward approaches differ from F1-delta regression. ACC-RAG learns ordered-prefix stopping; SPARKLE learns retrieval behavior with PPO; MBA uses a bandit; Adaptive-RAG uses outcome-derived complexity classes. None justifies skipping upstream controls.

## 9. Dataset/metric concerns
Token F1 may react to wording and annotation multiplicity. D records answer/evidence distributions and paired stochastic label agreement. Semantic correctness cannot be inferred from lexical mismatch; human adjudication is explicitly unmeasured. No post-hoc judge rewrites history.

## 10. Root-cause hypotheses
Priority1 retrieval foundation, priority2 compression/interface, priority3 reader/label instability. These are hypotheses, not W10 findings. Policy/dataset/metric causes remain untested.

## 11. Matched experimental matrix
R:3 retrievers x4k x2 interfaces using the same trained checkpoint and chunk corpus. C:one frozen retriever, one trained reader, same top10 pool, natural/full/latent and budgeted source-sentence controls. G:identical textual evidence and decoding across public readers. D:paired per-seed evidence/no-evidence repeats. Inspect `experiment_matrix.csv`.

## 12. Computational feasibility
Single reader or SFR encoder at a time, FP16, batch1, efficient SDPA, no automatic quantization. DEV R is5,544 calls and roughly5–12 GPU hours plus indexing; estimates need bounded measured validation. No full GPU load was performed by this pass.

## 13. Leakage/split plan
Use development only. Project DEV maps to raw QASPER TRAIN; paper identity remains explicit. Future training requires grouped paper folds. TEST path and split are rejected before dataset reads.

## 14. Fresh-confirmation plan
Propose frozen TriviaQA single-hop and HotpotQA multi-hop panels; audit exposure and freeze identities before outcome evaluation. Freshness is not yet established. No confirmation claim without it.

## 15. Recommended execution order
Audit/synthetic tests, bounded separate-version model smoke, R retrieval, R matched generation, R report/review/freeze, then C, G, D and root-cause synthesis. Only afterward consider policy research.

## 16. Explicit STOP conditions
Stop on identity mismatch, corrupt cache, missing source/weights, unsupported attention or memory overflow. Never substitute a model or truncate evidence silently. Stop controller development if a stable fixed upstream policy makes residual headroom negligible.

## 17. What Weeks0–9 remain valid for
They characterize their actual recorded configurations and splits. New upstream comparisons do not invalidate old measurements. No historical code/results were modified.

## 18. Assumptions now challenged
MiniLM is not synonymous with dense retrieval; high recall need not imply reader utility; query-aware routing is not query-conditioned latent compression; large retrospective oracle ceilings need not be predictable; a dynamic controller need not be the final system.

## Manual flags and resume
For Phase R only set `SMOKE_ONLY=False`, `RUN_RETRIEVER_FULL=True`; keep C/G/D/P false. Rerun the same cell after interruption. Completed answers are checksummed and skipped; only pending calls execute. Source/config/model changes require a new version.

## Remaining implementation gates
Faithful SeleCom adapter, strong learned textual compressor, contemporary reader runtime/memory smoke, and external confirmation materialization are not completed. C/G scaffolding is useful for controlled diagnostics but cannot be described as a completed reproduction of those methods. No root-cause classification has experimental support yet.

## Information per GPU-hour and handoff

1. R0 BM25/MiniLM/SFR exact-source recall and token-cost comparison: no answer generations, directly checks the untested retrieval foundation.
2. Matched R FULL/SARA retriever surface: isolates whether recall gains translate to reading and whether the latent interface changes that effect. Preregistered k2/4/6/8; any bounded throughput qualification uses a separate output version.
3. C same-pool FULL versus SARA and LEAD256 versus QCS256: separates the natural latent interface from query-conditioned textual suppression at a matched ceiling. Faithful SeleCom remains a separate compatibility gate.

Validation:27 tests passed; all10 safe-default notebook code cells executed with an allowlist forbidding expensive subprocess phases. Two real DEV questions passed source-offset/schema checks (45 chunks), without loading an encoder/generator. The notebook's optional `RUN_TINY_MODEL_SMOKE=True` section is for the user; it was not executed here.
