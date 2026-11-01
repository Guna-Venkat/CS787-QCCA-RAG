# Week 12 J/K/L — attribution implementation readiness

Branch: `gagan`. Post-hoc diagnostic on frozen VAL only. E/F/G full experiments are skipped by user decision; H reads saved C/C-EXT/D.


## Frozen diagnostic membership and zero confirmation overlap

Exactly **150 distinct queries**, 50 per dataset, selected before any perturbation answer was generated. QASPER uses 50 distinct papers; HotpotQA/SciFact have 50 distinct frozen groups each. Both query-ID and group overlap with reserved confirmation are zero. Original W12 splits/config/retriever/reader remain unchanged. No historical TEST payloads or labels were opened.

| dataset   | stratum         |   n |
|:----------|:----------------|----:|
| HotpotQA  | comparison      |  10 |
| HotpotQA  | bridge          |  40 |
| QASPER    | extractive      |  22 |
| QASPER    | mixed           |  13 |
| QASPER    | free_form       |   7 |
| QASPER    | yes_no          |   5 |
| QASPER    | unanswerable    |   3 |
| SciFact   | SUPPORT         |  23 |
| SciFact   | REFUTE          |  11 |
| SciFact   | NOT_ENOUGH_INFO |  16 |

Sampling: proportional dataset-native strata, deterministic sorted-ID shuffle seed42. QASPER prefers one query per paper, without conditioning on outcomes; HotpotQA bridge/comparison types and SciFact labels are represented. Frozen artifact: `attribution/ATTRIBUTION_QUERY_FREEZE.json`.


## Comparators fixed before attribution

| dataset   |   budget | comparator        |   VAL_quality |   VAL_queries |
|:----------|---------:|:------------------|--------------:|--------------:|
| HotpotQA  |      256 | H3                |      0.425311 |          7155 |
| HotpotQA  |     1024 | QBGF_PASSAGE_ONLY |      0.452507 |          7155 |
| QASPER    |      256 | FIXED_SENTENCE    |      0.263430 |          1005 |
| QASPER    |     1024 | H3                |      0.299487 |          1005 |
| SciFact   |      256 | C2                |      0.710248 |           177 |
| SciFact   |     1024 | BM25_PLAIN_RAG    |      0.710716 |           177 |

Rule: maximize completed full-VAL primary task score among FIXED_SENTENCE, C2 (256 only), H3, SFR RAW_SKIP, BM25 RAW_SKIP, and D sentence-only/passage-only allocation. Exact score ties: lower mean prompt tokens, then stable method name. NO_QUERY is excluded because it still uses mixed granularity; SARA specialists and incomplete E/F/G are excluded. Selection is diagnostic, not an unbiased new generalization estimate. Saved internal `BM25_PLAIN_RAG` means budget-matched BM25 RAW_SKIP.


## Prepared evidence, cache reuse and call budget

| causal_ONLY_top3_unknown_until_LOO   |   cells |   estimated_additional_disk_GiB_upper |   known_LOO_and_relevance_ONLY_unique |   known_action_cache_reuse_percent |   known_actions_already_cached |   minimum_free_disk_GiB |   no_evidence_reference_cache_reuse_percent | optional_replacement   | optional_third_method   |   original_reference_answers |   original_reference_cache_reuse_percent |   perturbation_call_upper_bound_before_global_dedup |   queries |   remaining_call_upper_bound |   units |
|:-------------------------------------|--------:|--------------------------------------:|--------------------------------------:|-----------------------------------:|-------------------------------:|------------------------:|--------------------------------------------:|:-----------------------|:------------------------|-----------------------------:|-----------------------------------------:|----------------------------------------------------:|----------:|-----------------------------:|--------:|
| True                                 |     600 |                              0.268601 |                                 10505 |                           0.514041 |                             54 |                       2 |                                         100 | False                  | False                   |                          600 |                                      100 |                                               16534 |       150 |                        16534 |    9607 |

There are **600 cells** (150 queries × two budgets × two methods) and **9,607 evidence units**. Original answers and NONE references are **100% reused** from compatible C/D caches. Of 10,505 known unique LOO/relevance-ONLY evidence sets, 54 already match compatible cached answers (0.514%); 10,451 are currently uncached. Additional causal-top3 and top-m sets depend on LOO outcomes. The conservative pre-dedup upper bound is **16,534 new perturbation calls**, not a prediction that every call will be made. Global evidence-token-ID deduplication, NONE/full reuse and resume reduce physical calls. No original full answer is regenerated.

Expected additional storage allowance is ~0.27 GiB by the conservative per-shard model, plus figure/table overhead. The generator enforces at least 2 GiB free; current free storage is 9.31 GiB. This is an estimate, not an end-to-end latency claim. No optional third system, attention study or replacement experiment is enabled.


## Exact protocol

- Units are the exact selected sentences/rule-proposition spans, or a complete selected passage for FULL/RAW_SKIP. FULL is not silently split into sentences. Standalone token counts and grouped serialization costs are both recorded.
- Every original selection reproduces its saved token IDs and source hash. Perturbations retain source order, use the original serializer, and change only selected evidence. Reader model, prompt, tokenizer, seed, decoding, parser and evaluator are inherited unchanged.
- Unit relevance is the canonical query-to-unit score; parent passage retrieval scores/ranks and QBGF option/utility metadata are stored separately. Embeddings are read from immutable-key SQLite in read-only mode; absent embeddings fail rather than loading/recomputing a model. BM25 uses the matching source pool.
- LOO importance is full score minus score after removing one unit, epsilon0. SciFact uses per-query strict label correctness, never per-example macro-F1. QA correctness flips use EM; F1 remains the attribution score.
- ONLY actions cover the union of relevance-top3 and causal-top3 units. Causal top1/top2/top4 ONLY and REMOVE are run only when m units exist. Causal ties use source order then stable ID, not relevance, to avoid artificial agreement.
- Normalized retention is defined only for strictly positive original gain over NONE; nonpositive denominators stay undefined. Values are not clipped.
- Outputs include exact/normalized answer flips, score/correctness/label changes, actual tokens, physical versus conceptual LM calls, gold overlap with unmapped cases kept unknown, redundancy and QBGF WIN/LOSS/TIE profiles.


## Statistics and interpretation limits

10,000 resamples, seed42 for the real study; paper clusters for QASPER, query clusters for HotpotQA, frozen document/claim groups for SciFact. Spearman is primary, Pearson secondary. Correlation CIs resample whole clusters with all their units; undefined constant correlations stay null. Deterministic top1 agreement, tie-aware top1 and top3 overlap/Jaccard are recorded. Paired harmful-fraction differences are query weighted to avoid comparing raw unit counts between differently sized units. Unit token lengths are explicit.

The causal ranking and sufficiency/removal checks use the same outcomes: these are retrospective diagnostics/ceilings, not deployable selection. A zero removal effect can reflect redundancy, interactions or an unchanged metric; it does not prove non-use. LOO effects are context-dependent and not additive causal credit. Removal also changes prompt length/context; optional length-matched replacement is not performed. Gold annotation overlap is not generator-optimal evidence or causal correctness. No attention-as-cause claim or internal-model explanation is made.

Quantitative table hashes are frozen before selecting qualitative candidates. Up to three available cases per WIN/LOSS/surprise category are provided, with raw answers, evidence, relevance and ablation effects; automated interpretation is withheld for manual review. If a category has fewer than three qualifying cases, none are invented.


## Tests and smoke

**136 unit tests passed**, including deterministic stratified sampling, unique QASPER paper preference, positive-gain retention guards, exact source-order perturbations, gold mapping uncertainty, degenerate correlations, manual real-generation guard and synthetic end-to-end/resume. The synthetic study has 12 cells across three fixture datasets; every physical LM-call count is zero. All **14 notebooks passed safe-copy smoke execution**, preserving live full-run flags. Cached real setup verified all 600 source selections with no real attribution generation. No full attribution status/results exist.


## Manual order

| Step | Notebook | Action |
|---|---|---|
| H | `week12_H_validation_analysis.ipynb` | Run All; saved C/C-EXT/D analysis is the default (`RUN_SMOKE=False`). E/F/G are explicitly not run. |
| J | `week12_J_attribution_setup.ipynb` | Set `RUN_SMOKE=False`, Run All. Reuses the already frozen 150 IDs and prepared evidence; inspect strata/comparators/cost. |
| K | `week12_K_causal_evidence_ablation.ipynb` | Set `RUN_FULL=True`, Run All to manually launch real perturbations. Resume=True, Overwrite=False. |
| L | `week12_L_attribution_analysis.ipynb` | After K completes, set `RUN_SMOKE=False`, Run All for saved-output analysis, seven plot types and candidate cases. |

Keep I/final freeze for a separate reviewed decision. No confirmation run is started. CLI equivalents:

```bash
sara_env/bin/python scripts/week12/run_attribution.py --stage setup
sara_env/bin/python scripts/week12/run_attribution.py --stage generate --full --explicit
sara_env/bin/python scripts/week12/run_attribution.py --stage analyze
```

The second command is manual-only and was NOT executed by the agent. Default notebooks retain RUN_FULL=False, RUN_SMOKE=True, RESUME=True, OVERWRITE=False for J/K/L.


## All frozen query IDs

| dataset   | query_id                                 | paper_id                 | group                                                                     | stratum         |
|:----------|:-----------------------------------------|:-------------------------|:--------------------------------------------------------------------------|:----------------|
| HotpotQA  | 5a7100435542994082a3e4a3                 | 5a7100435542994082a3e4a3 | question:0f48f453a4dc82feb9414202ceeccb6facb1a7773d58bad82dc73aa0d828e0a8 | comparison      |
| HotpotQA  | 5a737d3c55429908901be2c8                 | 5a737d3c55429908901be2c8 | question:2eb1ba17ca9ae61bc9185026fd1ca3f213d6a5ac795567a977319ba65ab75db8 | bridge          |
| HotpotQA  | 5a739d8c55429905862fe073                 | 5a739d8c55429905862fe073 | question:d82209d4d37595d382d9100df8ca776f2ade59da60b55ad5e6d1d697bc6fba52 | bridge          |
| HotpotQA  | 5a7741775542994aec3b724f                 | 5a7741775542994aec3b724f | question:635bc6aac3d28c8ad42afabdb7dfeb91ee8f55ba562b7eda5c86c13215a1e17a | bridge          |
| HotpotQA  | 5a7949d45542994bb9456fdd                 | 5a7949d45542994bb9456fdd | question:ffa64ffa2f3f7eebe996e58d9aa3a6a0a48a8e47bdac83624eeae73ae7294aff | bridge          |
| HotpotQA  | 5a7a1e635542990783324e69                 | 5a7a1e635542990783324e69 | question:0ae6b8cd7565f4be6fa9ccc17171586fb06ef52d7dc1e73fdfdf0ff5cfaf281b | bridge          |
| HotpotQA  | 5a7c76275542990527d554b4                 | 5a7c76275542990527d554b4 | question:897bba2d9bd7216232a74926003d01d5d9d14b8e3f71c2d259104eaf0917a275 | comparison      |
| HotpotQA  | 5a7df25a5542990b8f503b02                 | 5a7df25a5542990b8f503b02 | question:235c74b6db1108e1136692c1b036043aa87618748f1aeecd3614086c5360b651 | bridge          |
| HotpotQA  | 5a7e30b255429934daa2fbeb                 | 5a7e30b255429934daa2fbeb | question:86a7048c9e2735aba19c8444ee2ad4b237cb698f3e22f9095d92ae62e1d9b01a | bridge          |
| HotpotQA  | 5a807e6f5542995d8a8ddf80                 | 5a807e6f5542995d8a8ddf80 | question:75f32d09e626f4b7472520af4f5ba2bd6418ab144827dba915c4609d772d0a64 | bridge          |
| HotpotQA  | 5a8241a855429940e5e1a819                 | 5a8241a855429940e5e1a819 | question:eb4681094142be237da8bad80ea6d02e176ce0a9627c486655fd674667d61ed9 | bridge          |
| HotpotQA  | 5a82a1ae55429954d2e2eb89                 | 5a82a1ae55429954d2e2eb89 | question:905cf41d1def2145ac220766adb0d00d3d5a899b7d65f9e85a05a424c0df3157 | comparison      |
| HotpotQA  | 5a8333115542993344745feb                 | 5a8333115542993344745feb | question:26616edbf4896d33574cd5509fd9e24a2b33c9b39abfb0742a1af96ca44dca2a | bridge          |
| HotpotQA  | 5a8484495542997175ce1ee8                 | 5a8484495542997175ce1ee8 | question:c44758b4ce16a68b86bfecbff4a6824cc41e02e5696b9f1064bd0b6ddcc2dd7a | bridge          |
| HotpotQA  | 5a84b7db5542991dd0999db0                 | 5a84b7db5542991dd0999db0 | question:d0df9df83f3b9a8077f72be9f5d204f06a6a7a038213afcaaf26e0e4a8b1e584 | comparison      |
| HotpotQA  | 5a84bc5b5542991dd0999dc3                 | 5a84bc5b5542991dd0999dc3 | question:ce2bf7cd96ab38b94476832c37914e3c88de319bc6a0bbe75da2b01d048a7c87 | bridge          |
| HotpotQA  | 5a862f985542991e771815af                 | 5a862f985542991e771815af | question:59f98372ea3d7b35192932dacde8edd3fe6f3b7e58b8c04a7ac936c0cb75687f | bridge          |
| HotpotQA  | 5a8901a2554299515336125a                 | 5a8901a2554299515336125a | question:a7c987a3d70448e25f7f81f6fcd26e59c68c5146ce4cd7252ecfd00e5fe76e87 | bridge          |
| HotpotQA  | 5a89ba6b554299669944a584                 | 5a89ba6b554299669944a584 | question:9517e7c4f624fc9e1048f080c8a675c457438c2df2eeef101a8d58030f42d5f2 | bridge          |
| HotpotQA  | 5a8ac0595542992d82986f6c                 | 5a8ac0595542992d82986f6c | question:a847b438d7b2eecb60d25d456966d94f46906316bd22eff71cbba34036486cfd | bridge          |
| HotpotQA  | 5a8dbc8b554299441c6ba077                 | 5a8dbc8b554299441c6ba077 | question:a09ec6cefb1e0b81e2839d8eacf9f429f821cd108c6cd16de2220d4fcb253b96 | bridge          |
| HotpotQA  | 5a8eed625542995085b374b9                 | 5a8eed625542995085b374b9 | question:8072624ed07dbc181f61138e71a5854ac3bf563bc995746180588bf81ef2043b | bridge          |
| HotpotQA  | 5a8f6e295542992414482ac6                 | 5a8f6e295542992414482ac6 | question:d34a49b5884e32e1b18769d393dc8e689abb3f6786a17fc45e51f7b8a224fc00 | bridge          |
| HotpotQA  | 5a8f7c9a55429918e830d239                 | 5a8f7c9a55429918e830d239 | question:aff6ef76424f250cb4f981e9ea3bdc32398cb3c95cedc28c14df1955663e4adb | bridge          |
| HotpotQA  | 5ab1e36d554299449642c7f9                 | 5ab1e36d554299449642c7f9 | question:f712e62191bbceb75fdfa692c2ac8ff217ce5bac400a61b2b7f547c6f38e44b8 | comparison      |
| HotpotQA  | 5ab2b2a7554299295394681d                 | 5ab2b2a7554299295394681d | question:5e595cbbf848d825a046016a1b6d85d2169cfff6d8a5c2116a3cc365ebbe6365 | comparison      |
| HotpotQA  | 5ab2f88755429976abd1bbe0                 | 5ab2f88755429976abd1bbe0 | question:1e2eb4d042545aacdfd49d176ce28ea608e42b9a43f2eebe8e030da48e0feab6 | comparison      |
| HotpotQA  | 5ab7c6995542993667794005                 | 5ab7c6995542993667794005 | question:1eeb997fcb26e7415ee5465eea6d0464b861701228dd4e29f364179c5bec5db4 | bridge          |
| HotpotQA  | 5ab8a7f65542991b5579efca                 | 5ab8a7f65542991b5579efca | question:c980e7a0898236216362015534d29c438904863e3f843e6690f48d0c6478e7aa | bridge          |
| HotpotQA  | 5ab8eabb5542991b5579f03d                 | 5ab8eabb5542991b5579f03d | question:195a4e78b301bf6324728e82c2ea7aa8a7419263039171b2db2a40c99cd0fd76 | bridge          |
| HotpotQA  | 5ab9b7d555429970cfb8eb7a                 | 5ab9b7d555429970cfb8eb7a | question:9010e82bce92d349b53427ee957fb53476ede2e8715c4b5815b1f742fa024505 | bridge          |
| HotpotQA  | 5abb7ff15542996cc5e4a015                 | 5abb7ff15542996cc5e4a015 | question:830874b40475631ab01061ac9b47f10c1e37367bfd3db9c217984c4a7d0f9943 | bridge          |
| HotpotQA  | 5abd8c295542992ac4f382ab                 | 5abd8c295542992ac4f382ab | question:2ab36daa4febf71143a9e139b37cc15f36d2c05de834173a4bc114ab3544ae62 | bridge          |
| HotpotQA  | 5ac046c65542996f0d89cb71                 | 5ac046c65542996f0d89cb71 | question:fdad71ce1d4b961be25276b81939d05b211c9a719228fecf8dbaa9beff47e411 | comparison      |
| HotpotQA  | 5ac11a8d5542992a796dedee                 | 5ac11a8d5542992a796dedee | question:1f9cb5ed452e9b1150991af2f8c7a7ed6f2febabf957647e1ccc3f2bdc6f8cff | comparison      |
| HotpotQA  | 5adc3d7455429944faac248a                 | 5adc3d7455429944faac248a | question:054111295b637bc5ae6a0824877f37122f4095999af4457b30edfd910fb164d3 | bridge          |
| HotpotQA  | 5adc75525542994650320d0e                 | 5adc75525542994650320d0e | question:cc12abb1dbceb663484db8a00615f3ea479bd86bd421c3109bba28b0afe4ba06 | comparison      |
| HotpotQA  | 5adc7f4a5542994650320d28                 | 5adc7f4a5542994650320d28 | question:f19724ef7f8a6c05ebec7b7b59287fd3ece06e0d3a8cadc29737dbe1faeb5916 | bridge          |
| HotpotQA  | 5adcca7f5542994ed6169c06                 | 5adcca7f5542994ed6169c06 | question:e998e570ae83e80048c33422220a46bdfef4ded34c30cbfe0e944c8ee1eb3f5d | bridge          |
| HotpotQA  | 5adce88b5542992c1e3a249a                 | 5adce88b5542992c1e3a249a | question:ef26820ff093fb655bd8f04ae90b6ee893611cc140b963983fd5abca95ab0f72 | bridge          |
| HotpotQA  | 5add1d575542992c1e3a2540                 | 5add1d575542992c1e3a2540 | question:a264b7e9c8ea43633385cf84fae5d87045573b41a30b1a69e2dc160dd62e018d | bridge          |
| HotpotQA  | 5adfa3da554299025d62a317                 | 5adfa3da554299025d62a317 | question:96a26cd07395b1517c33b7b342a37d8e29fe07d5b6b4c63055ddd4a5303df713 | bridge          |
| HotpotQA  | 5ae29d8a5542996483e64a0f                 | 5ae29d8a5542996483e64a0f | question:ab7c32e26d4885659d6bf0d21b253cb287e031ac0a57b0213e5dae8470cd161e | bridge          |
| HotpotQA  | 5ae2eb7f55429928c423955b                 | 5ae2eb7f55429928c423955b | question:406e1ec1ce9f4c90e60f03c1bd49bc2f4bd8a196c1d867a69367195977db107a | bridge          |
| HotpotQA  | 5ae35b6b5542990afbd1e116                 | 5ae35b6b5542990afbd1e116 | question:626d0f609d2d46599cddf1a3f8165cf834f51fd2dd596bafaf22299012f82713 | bridge          |
| HotpotQA  | 5ae531695542990ba0bbb1fb                 | 5ae531695542990ba0bbb1fb | question:df072800f8889404003de68417f2eb3ac41650e0640bdf6430ca792eac1f115d | bridge          |
| HotpotQA  | 5ae72b96554299572ea5474e                 | 5ae72b96554299572ea5474e | question:bc719e53a9cd96d67a786188570a843e610e68660a9ed3909994a7895b65da63 | bridge          |
| HotpotQA  | 5ae73d505542991e8301cc2c                 | 5ae73d505542991e8301cc2c | question:4c3e3c66748459b9c2e0b96cd54abe69f04fda3d7035f71b61132b43465f3bc4 | bridge          |
| HotpotQA  | 5ae7936d5542997ec27276a7                 | 5ae7936d5542997ec27276a7 | question:cabdbfd70af13fb07c637527c59b8ca17038f48cd08e07927f7023d40af9da99 | bridge          |
| HotpotQA  | 5ae824295542997ec272772f                 | 5ae824295542997ec272772f | question:bcefd15d33551fb07b60f4b24f102cdf74db131d9cc72e43a33020f43b05d626 | bridge          |
| QASPER    | 01a41c0a4a7365cd37d28690735114f2ff5229f2 | 1612.08205               | 1612.08205                                                                | extractive      |
| QASPER    | 05887a8466e0a2f0df4d6a5ffc5815acd7d9066a | 1706.06894               | 1706.06894                                                                | extractive      |
| QASPER    | 113d791df6fcfc9cecfb7b1bebaf32cc2e4402ab | 1708.01464               | 1708.01464                                                                | mixed           |
| QASPER    | 15e481e668114e4afe0c78eefb716ffe1646b494 | 1908.09246               | 1908.09246                                                                | extractive      |
| QASPER    | 1bb7eb5c3d029d95d1abf9f2892c1ec7b6eef306 | 1703.07090               | 1703.07090                                                                | free_form       |
| QASPER    | 21548433abd21346659505296fb0576e78287a74 | 1909.13104               | 1909.13104                                                                | free_form       |
| QASPER    | 2ccc26e11df4eb26fcccdd1f446dc749aff5d572 | 1912.03804               | 1912.03804                                                                | yes_no          |
| QASPER    | 37a79be0148e1751ffb2daabe4c8ec6680036106 | 1611.03599               | 1611.03599                                                                | extractive      |
| QASPER    | 3de0487276bb5961586acc6e9f82934ef8cb668c | 2003.03106               | 2003.03106                                                                | extractive      |
| QASPER    | 443d2448136364235389039cbead07e80922ec5c | 1712.00991               | 1712.00991                                                                | free_form       |
| QASPER    | 50be4a737dc0951b35d139f51075011095d77f2a | 1503.00841               | 1503.00841                                                                | mixed           |
| QASPER    | 545e92833b0ad4ba32eac5997edecf97a366a244 | 1908.06267               | 1908.06267                                                                | mixed           |
| QASPER    | 567dc9bad8428ea9a2658c88203a0ed0f8da0dc3 | 1908.05828               | 1908.05828                                                                | extractive      |
| QASPER    | 57f23dfc264feb62f45d9a9e24c60bd73d7fe563 | 1811.12254               | 1811.12254                                                                | mixed           |
| QASPER    | 5a8cc8f80509ea77d8213ed28c5ead501c68c725 | 1902.09666               | 1902.09666                                                                | extractive      |
| QASPER    | 5cb610d3d5d7d447b4cd5736d6a7d8262140af58 | 1809.01541               | 1809.01541                                                                | extractive      |
| QASPER    | 5cc2daca2a84ddccba9cdd9449e51bb3f64b3dde | 1912.01772               | 1912.01772                                                                | mixed           |
| QASPER    | 612c3675b6c55b60ae6d24265ed8e20f62cb117e | 2003.08385               | 2003.08385                                                                | yes_no          |
| QASPER    | 63a1cbe66fd58ff0ead895a8bac1198c38c008aa | 1912.08960               | 1912.08960                                                                | extractive      |
| QASPER    | 67ec8ef85844e01746c13627090dc2706bb2a4f3 | 1905.11268               | 1905.11268                                                                | unanswerable    |
| QASPER    | 6bfba3ddca5101ed15256fca75fcdc95a53cece7 | 1910.09982               | 1910.09982                                                                | extractive      |
| QASPER    | 70a1b0f9f26f1b82c14783f1b76dfb5400444aa4 | 1906.07662               | 1906.07662                                                                | free_form       |
| QASPER    | 75df70ce7aa714ec4c6456d0c51f82a16227f2cb | 2002.01664               | 2002.01664                                                                | free_form       |
| QASPER    | 78292bc57ee68fdb93ed45430d80acca25a9e916 | 1911.03343               | 1911.03343                                                                | mixed           |
| QASPER    | 78536da059b884d6ad04680baeb894895458055c | 2001.02885               | 2001.02885                                                                | extractive      |
| QASPER    | 7b4992e2d26577246a16ac0d1efc995ab4695d24 | 1707.05236               | 1707.05236                                                                | extractive      |
| QASPER    | 7b4fb6da74e6bd1baea556788a02969134cf0800 | 1909.03405               | 1909.03405                                                                | yes_no          |
| QASPER    | 869feb7f47606105005efdb6bea1c549824baea0 | 1907.06292               | 1907.06292                                                                | extractive      |
| QASPER    | 88e62ea7a4d1d2921624b8480b5c6b50cfa5ad42 | 1609.00559               | 1609.00559                                                                | mixed           |
| QASPER    | 8b0abc1907c2bf3e0256f8cf85e0ba66a839bd92 | 1909.05359               | 1909.05359                                                                | yes_no          |
| QASPER    | 8b9c12df9f89040f1485b3847a29f11b5c9262e0 | 2003.02249               | 2003.02249                                                                | mixed           |
| QASPER    | 907b3af3cfaf68fe188de9467ed1260e52ec6cf1 | 1712.05999               | 1712.05999                                                                | free_form       |
| QASPER    | 90d946ccc3abf494890e147dd85bd489b8f3f0e8 | 1905.12801               | 1905.12801                                                                | mixed           |
| QASPER    | 975e60535724f4149c7488699a199ba2920a062c | 1907.01468               | 1907.01468                                                                | unanswerable    |
| QASPER    | 97d1ac71eed13d4f51f29aac0e1a554007907df8 | 1908.08345               | 1908.08345                                                                | extractive      |
| QASPER    | a0543b4afda15ea47c1e623c7f00d4aaca045be0 | 1802.07862               | 1802.07862                                                                | mixed           |
| QASPER    | a103636c8d1dbfa53341133aeb751ffec269415c | 1806.04511               | 1806.04511                                                                | extractive      |
| QASPER    | a15bc19674d48cd9919ad1cf152bf49c88f4417d | 1805.06966               | 1805.06966                                                                | extractive      |
| QASPER    | a5e5cda1f6195ab1336855f1e39a609d61326d62 | 1807.07279               | 1807.07279                                                                | extractive      |
| QASPER    | b3d01ac226ee979e188a4141877a6d2a5482de98 | 1910.14599               | 1910.14599                                                                | extractive      |
| QASPER    | b9d07757e2d2c4be41823dd1ea3b9c7f115b5f72 | 1808.03738               | 1808.03738                                                                | mixed           |
| QASPER    | bc16ce6e9c61ae13d46970ebe6c4728a47f8f425 | 1603.09631               | 1603.09631                                                                | free_form       |
| QASPER    | c262d3d1c5a8b6fef6b594d5eee86bc2b09e3baf | 1908.07195               | 1908.07195                                                                | unanswerable    |
| QASPER    | c81f215d457bdb913a5bade2b4283f19c4ee826c | 1910.12574               | 1910.12574                                                                | extractive      |
| QASPER    | cfbec1ef032ac968560a7c76dec70faf1269b27c | 1704.06194               | 1704.06194                                                                | extractive      |
| QASPER    | d5a8fd8bb48dd1f75927e874bdea582b4732a0cd | 1901.01010               | 1901.01010                                                                | mixed           |
| QASPER    | d6401cece55a14d2a35ba797a0878dfe2deabedc | 1601.02403               | 1601.02403                                                                | extractive      |
| QASPER    | ddb23a71113cbc092cbc158066d891cae261e2c6 | 1707.06806               | 1707.06806                                                                | extractive      |
| QASPER    | deb89bca0925657e0f91ab5daca78b9e548de2bd | 1904.04358               | 1904.04358                                                                | mixed           |
| QASPER    | f20a389ace2267aa61eddcc235535452ccdae0e6 | 1909.09534               | 1909.09534                                                                | yes_no          |
| SciFact   | 1012                                     | 1012                     | cited-paper-component:1011                                                | SUPPORT         |
| SciFact   | 1019                                     | 1019                     | cited-paper-component:1018                                                | SUPPORT         |
| SciFact   | 1029                                     | 1029                     | cited-paper-component:1028                                                | REFUTE          |
| SciFact   | 1049                                     | 1049                     | cited-paper-component:1031                                                | REFUTE          |
| SciFact   | 1132                                     | 1132                     | cited-paper-component:550                                                 | SUPPORT         |
| SciFact   | 1202                                     | 1202                     | cited-paper-component:559                                                 | SUPPORT         |
| SciFact   | 1216                                     | 1216                     | cited-paper-component:1133                                                | SUPPORT         |
| SciFact   | 1266                                     | 1266                     | cited-paper-component:1265                                                | SUPPORT         |
| SciFact   | 1290                                     | 1290                     | cited-paper-component:1288                                                | SUPPORT         |
| SciFact   | 1292                                     | 1292                     | cited-paper-component:44                                                  | NOT_ENOUGH_INFO |
| SciFact   | 1298                                     | 1298                     | cited-paper-component:1298                                                | SUPPORT         |
| SciFact   | 132                                      | 132                      | cited-paper-component:132                                                 | NOT_ENOUGH_INFO |
| SciFact   | 1335                                     | 1335                     | cited-paper-component:1316                                                | SUPPORT         |
| SciFact   | 1359                                     | 1359                     | cited-paper-component:272                                                 | REFUTE          |
| SciFact   | 1379                                     | 1379                     | cited-paper-component:178                                                 | SUPPORT         |
| SciFact   | 1389                                     | 1389                     | cited-paper-component:1389                                                | SUPPORT         |
| SciFact   | 148                                      | 148                      | cited-paper-component:91                                                  | SUPPORT         |
| SciFact   | 212                                      | 212                      | cited-paper-component:212                                                 | REFUTE          |
| SciFact   | 213                                      | 213                      | cited-paper-component:214                                                 | NOT_ENOUGH_INFO |
| SciFact   | 279                                      | 279                      | cited-paper-component:277                                                 | SUPPORT         |
| SciFact   | 298                                      | 298                      | cited-paper-component:299                                                 | SUPPORT         |
| SciFact   | 314                                      | 314                      | cited-paper-component:1209                                                | SUPPORT         |
| SciFact   | 324                                      | 324                      | cited-paper-component:223                                                 | NOT_ENOUGH_INFO |
| SciFact   | 343                                      | 343                      | cited-paper-component:342                                                 | SUPPORT         |
| SciFact   | 354                                      | 354                      | cited-paper-component:354                                                 | NOT_ENOUGH_INFO |
| SciFact   | 385                                      | 385                      | cited-paper-component:385                                                 | SUPPORT         |
| SciFact   | 411                                      | 411                      | cited-paper-component:410                                                 | NOT_ENOUGH_INFO |
| SciFact   | 452                                      | 452                      | cited-paper-component:452                                                 | REFUTE          |
| SciFact   | 475                                      | 475                      | cited-paper-component:900                                                 | SUPPORT         |
| SciFact   | 521                                      | 521                      | cited-paper-component:494                                                 | SUPPORT         |
| SciFact   | 525                                      | 525                      | cited-paper-component:679                                                 | SUPPORT         |
| SciFact   | 536                                      | 536                      | cited-paper-component:537                                                 | SUPPORT         |
| SciFact   | 539                                      | 539                      | cited-paper-component:539                                                 | SUPPORT         |
| SciFact   | 54                                       | 54                       | cited-paper-component:55                                                  | REFUTE          |
| SciFact   | 552                                      | 552                      | cited-paper-component:553                                                 | NOT_ENOUGH_INFO |
| SciFact   | 554                                      | 554                      | cited-paper-component:555                                                 | NOT_ENOUGH_INFO |
| SciFact   | 578                                      | 578                      | cited-paper-component:578                                                 | REFUTE          |
| SciFact   | 587                                      | 587                      | cited-paper-component:588                                                 | NOT_ENOUGH_INFO |
| SciFact   | 619                                      | 619                      | cited-paper-component:620                                                 | REFUTE          |
| SciFact   | 628                                      | 628                      | cited-paper-component:126                                                 | NOT_ENOUGH_INFO |
| SciFact   | 644                                      | 644                      | cited-paper-component:642                                                 | NOT_ENOUGH_INFO |
| SciFact   | 700                                      | 700                      | cited-paper-component:703                                                 | NOT_ENOUGH_INFO |
| SciFact   | 72                                       | 72                       | cited-paper-component:73                                                  | NOT_ENOUGH_INFO |
| SciFact   | 756                                      | 756                      | cited-paper-component:756                                                 | SUPPORT         |
| SciFact   | 847                                      | 847                      | cited-paper-component:847                                                 | REFUTE          |
| SciFact   | 870                                      | 870                      | cited-paper-component:871                                                 | NOT_ENOUGH_INFO |
| SciFact   | 879                                      | 879                      | cited-paper-component:879                                                 | REFUTE          |
| SciFact   | 887                                      | 887                      | cited-paper-component:887                                                 | NOT_ENOUGH_INFO |
| SciFact   | 975                                      | 975                      | cited-paper-component:976                                                 | NOT_ENOUGH_INFO |
| SciFact   | 993                                      | 993                      | cited-paper-component:990                                                 | REFUTE          |


ATTRIBUTION STUDY IMPLEMENTED

VALIDATION ONLY

CONFIRMATION UNTOUCHED

CAUSAL INPUT ABLATION, NOT INTERNAL-MODEL EXPLANATION

NO FULL ATTRIBUTION RUN EXECUTED
