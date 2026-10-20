# CS787 RAG Project: Week 0–8 Results and Research Story

Prepared 20 October 2026; updated with commit `12901d0588f057d9ddf06ff92ec175fe979b5cfd` (W8G executed), matching the locally recorded `origin/main`. Changes under `Baselines/` were excluded from this update. Scope: the experiment artifacts currently available in this repository, through Week 8. This is an evidence-based synthesis of saved results, not a new experimental run. Week numbers identify research phases; some experiments and audits were completed later than their phase number suggests.

## 1. The story so far

The project began with a practical question: **can a retrieval-augmented generator decide how much retrieved evidence to represent as full text for each question, preserving answer quality while reducing context cost?** SARA provides the mechanism: retain some retrieved passages as text and represent the others through compressed embeddings. QCCA attempts to choose that allocation from the question and retrieved context.

The first important finding was that questions differ considerably. Evidence is unevenly distributed, model utilization of passages is uneven, and the best observed textual budget varies across queries. The fixed-budget experiments exposed substantial oracle headroom. However, an oracle can inspect the answer quality at every budget; a deployed allocator cannot. Much of Weeks 4–7 therefore investigated the gap between observing this opportunity and predicting the correct action before generation.

A small, balanced classifier for the specific decision “stay at four passages or expand to six?” produced a useful development tradeoff. Larger semantic allocation models did not consistently improve on it. A major implementation defect in the first semantic allocator was diagnosed and corrected, but healthy models still did not close the oracle gap.

The next advance came from changing **which passages receive full text**, rather than only changing their number. Semantic passage routing made four full-text passages much more effective than a BM25 prefix. That stronger static baseline proved difficult to beat: retrained budget policies, utility regression, cross-encoders, a listwise 7B router, and intermediate-fidelity extracts all failed to demonstrate a better overall operating point in their tested configurations.

Week 8 reframed the problem around feedback from a first answer. Start with semantic k=4, generate a draft, estimate whether more detail would help, select a compressed passage to promote to full text, and regenerate if needed. This yielded real promotion surfaces and a complete measured one-promotion controller. Yet the frozen controller did not establish a quality or cost advantage over static k=6. Attribution added substantial extraction work without confirmed benefit. Query-conditioned representation adapters produced modest validation gains, with additional costs for the draft-conditioned variant. The completed W8-G generator-adaptation study subsequently achieved 0.4150 F1 with direct k4 HC-LoRA, while uniform SFT was strongest within the frozen controller at 0.4123. Outcome weighting did not outperform these simpler variants.

**The defensible result is a sequence of increasingly precise findings about selection, fidelity, and utilization. The project has not yet established that its final adaptive system reliably outperforms strong static baselines on an untouched final test.**

## 2. Experimental setting and how to read the numbers

The principal benchmark is QASPER scientific-paper question answering. The documented training population contains 2,088 questions from 772 papers; later training experiments use 1,746 FIT questions from 656 papers and 342 INTERNAL-VAL questions from 116 papers. DEV contains 231 questions from 86 papers. Later records explicitly describe INTERNAL-VAL as development confirmation after previous inspection. DEV also has a long history of exploratory use. Neither should be described as an untouched final test.

The generator is based on Mistral-7B-Instruct-v0.2. The SARA projection checkpoint is identified as `sara_qasper_proj_lr5e4_seed42`; the uncompressed RAG reference uses `rag_qasper_lora_r16_seed42`. In the main early experiments, the retrieved pool has ten passages. The deployed SARA budget space is {2,4,5,6,8}; k=0 is a diagnostic compression-only condition, while k=10 uses a separate uncompressed RAG checkpoint.

Token F1 is the primary answer metric. EM and ROUGE-L supplement the early fixed-budget sweep. An absolute F1 change of 0.01 means one percentage point on a 0–100 scale. Oracle performance is an offline upper bound that uses gold-reference outcomes; it is not a deployable policy.

Cost definitions change across phases. Early reports use context-token counts; some later tables differ by approximately eight tokens because of prompt-accounting conventions. Week 8 controller results use exact chat-template prompt tokens and distinguish the final prompt from cumulative prompts across both generations. Those columns must not be interchanged. Prompt tokens alone do not measure full FLOPs, output-token cost, or deployment latency.

Paper-grouped folds and paper-clustered bootstrap intervals address dependence between questions from the same paper. They do not erase the effects of repeated development-set exploration. An interval containing zero does not prove equivalence or noninferiority.

## 3. Week 0 — Establishing the motivation and checking model utilization

### Question and work

Week 0 characterized document lengths, answer types, annotated evidence, retrieval coverage, prompt order, attention, and passage interventions. It also audited whether the diagnostics came from real model inference.

The descriptive dataset table reports long papers: approximately 3,694 training words and 4,228 development words per paper on average. Questions are short, averaging 8.39 words; development answers average 14.45 words. This makes selective evidence representation a reasonable research direction.

### Main empirical result

The final real-model study used 40 questions from 40 development papers and 840 forward passes: 40 full-context passes, 400 passage removals, and 400 length-matched replacements. The report records a TITAN RTX run, approximately 18.40 GB peak VRAM, and 4,615.58 seconds.

Strong gold-evidence matches had a mean replacement effect of +0.6187 in gold-answer log-probability. The gap between STRONG and NONE/WEAK matches was +0.6029, with a paper-clustered 95% interval [+0.3775,+0.8570]. Strong matches received about 2.6 times the answer attention mass of NONE matches. Replacement effects ranged from −0.6094 to +3.1289, showing heterogeneous functional contributions.

The exploratory bridge to Week 3 found a Spearman correlation of +0.394 between positive replacement mass and oracle budget, with p=0.0118. This is a small-sample retrospective association, not a ready-made predictor.

### What changed in our understanding

Retrieved passages do not contribute uniformly. The motivation for adaptive fidelity is empirical rather than purely intuitive. However, attention remains a utilization proxy; a negative replacement effect compares the original passage with a particular unrelated replacement and does not independently prove harmful distraction.

Prompt order also matters: with context preceding the query in a causal decoder, passage states cannot attend forward to the question. Query and answer tokens can attend backward to context. Claims about question-conditioned passage representations must respect that asymmetry.

### Audit and unresolved discrepancies

The initial layer/utilization diagnostics contained deterministic simulations and were marked invalid for empirical citation. The final 40-paper live study is the appropriate source for real-model findings.

The initial Week-0 report and final40 report disagree on evidence percentages and recall. For example, the initial table reports Recall-any@10=87.01% and Recall-all@10=80.09%, while final40 prose reports 84.8% and 64.9%. Their scopes and aggregation are not reconciled in the available reports. This synthesis therefore does not merge them into one canonical retrieval statistic. The initial report also pairs 142/220 with 67.10%; that arithmetic is inconsistent, since 142/220 is 64.55%.

Sources: [initial report](../results/week0/week0_report.md), [scientific audit](../results/week0/scientific_audit.md), [final live study](../results/week0/final40/final40_report.md).

## 4. Week 1 — Baseline preparation: an explicit documentation gap

No standalone Week-1 report, result directory, or Week-1 notebook was located in the inspected repository. It would be inappropriate to invent Week-1 scores, training losses, runtimes, or a completion date.

The downstream artifacts establish that a SARA projection checkpoint and an uncompressed RAG checkpoint were available for the Week-2 comparison. Baseline implementation, retrieval/cache preparation, and evaluation setup are prerequisites for that comparison, but assigning their precise execution history to Week 1 is a reconstruction, not a documented experiment result.

For a final thesis or course submission, add the original baseline training/configuration logs if they are available elsewhere. Until then, label this phase **baseline preparation inferred from downstream artifacts; independent Week-1 measurements unavailable**.

## 5. Week 2 — Measuring the fixed-budget response surface

Week 2 generated a frozen DEV response surface: 231 questions across seven conditions, or 1,617 evaluation points.

| Condition | F1 | EM | ROUGE-L | Mean context tokens |
|---|---:|---:|---:|---:|
| SARA k=0 | 0.1388 | 0.0866 | 0.1400 | 138.8 |
| SARA k=2 | 0.3038 | 0.1429 | 0.2680 | 534.5 |
| SARA k=4 | 0.3596 | 0.1688 | 0.3134 | 946.1 |
| SARA k=5 | 0.3629 | 0.1861 | 0.3169 | 1157.0 |
| SARA k=6 | 0.3857 | 0.1861 | 0.3340 | 1359.1 |
| SARA k=8 | 0.3950 | 0.1861 | 0.3377 | 1758.5 |
| Uncompressed RAG k=10 | 0.4090 | 0.2035 | 0.3523 | 2137.9 |

Average quality increases with larger budgets, but this average hides individual response variation. Compression alone is substantially weaker. The SARA parent default k=5 is not the best static quality setting; k=8 is best within the main SARA action space.

The Week-2 oracle reaches 0.5008 because it includes k=0 in {0,2,4,5,6,8}. That is a different action space from the later deployed allocator. Also, uncompressed k=10 is a reference using another checkpoint, not a causal isolation of “two more passages.”

The phrase “reference ceiling” in older reports should be read as an observed baseline reference, not a mathematical upper bound: adaptive oracle scores can exceed it.

Source: [Week-2 validation](../results/week2/week2_scientific_validation_report.md), with the full metric table reproduced in the [Week-3 final report](../results/week3/week3_final_scientific_report.md).

## 6. Week 3 — Quantifying oracle headroom and defining the learning target

Week 3 restricted the oracle to deployable SARA actions {2,4,5,6,8}. For each query, it selected the smallest k achieving maximum observed F1. It also defined an epsilon target: choose the smallest budget within 0.01 F1 of that query’s best result.

The deployable quality oracle achieved **0.4895 F1**, compared with static k=8 at 0.3950: +0.0945 absolute F1, or 23.93% relative improvement. The paper-clustered interval for the gain was [+0.0693,+0.1226].

| Tie-broken optimal k | Questions | Share |
|---|---:|---:|
| 2 | 130 | 56.28% |
| 4 | 40 | 17.32% |
| 5 | 17 | 7.36% |
| 6 | 21 | 9.09% |
| 8 | 23 | 9.96% |

These are tie-broken assignments, not proof that every assigned query strictly needs that budget. Static k=8 was strictly worse than a smaller budget on 68 questions (29.44%), tied with smaller budgets on 140 (60.61%), and uniquely best on 23 (9.96%).

At epsilon=0.01, the oracle selected mean k=3.49, used 841.3 context tokens, and achieved 0.4894 F1. That implies 52.16% fewer context tokens than static k=8, assuming perfect outcome-aware selection.

The cheap observable signals were weak: top-1/top-2 score gap correlated +0.1320 with the target and competitive-passage count −0.1421. Query length and score entropy had negligible univariate associations.

The resulting learning problem was therefore difficult: there was ample retrospective headroom, but little simple evidence that inexpensive features could predict it.

A blinded human audit was prepared for 50 questions × three conditions =150 judgments. The final Week-3 report explicitly marks human evaluation pending; no human correctness claim is supported.

Source: [Week-3 final report](../results/week3/week3_final_scientific_report.md).

## 7. Week 4 — From global budget classification to selective expansion

### Initial formulation

Predicting the smallest epsilon-optimal budget directly encouraged low-budget collapse. Initial multinomial logistic regression achieved only 0.3034 F1 at 539.8 tokens, nearly identical to static k=2 and slightly worse in both point estimates. The heuristic rule achieved 0.3286; a small tree achieved 0.3282. High target accuracy was misleading because the smallest-budget label was common and ties were frequent.

Some initial report prose calls these results competitive or superior to chance despite contradictory tables. The numerical outcomes do not support that broad wording.

### Revised policy and final Week-4 result

The work shifted to a narrower decision: **start at k=4, selectively expand to k=6, and stop there**. Feature discovery explored 64 candidates; the final balanced logistic classifier retained five: semantic similarity over the top six, mean BM25 score, lexical coverage in the top two, query cluster span, and mean lexical overlap.

The final OOF classifier had ROC-AUC=0.6503, PR-AUC=0.3199, balanced accuracy=0.6448, precision=0.3053, and recall=0.6444. It caught 29 of 45 beneficial expansions. The confusion counts were 120 true stops, 29 true expansions, 66 false expansions, and 16 missed beneficial expansions.

Policy C1 achieved **0.3842 F1**, mean k=4.82, and 1115.9 context tokens: 36.54% fewer tokens than static k=8. Compared with static k=6, it saved 243.2 tokens for a point-estimate F1 difference of −0.0015. The paired F1 interval was [−0.0241,+0.0195]; the token interval was [−276.6,−211.0]. Against k=8, the F1 difference was −0.0108 with interval [−0.0451,+0.0225].

This is a useful development efficiency result with statistically supported token savings. The F1 intervals do not establish formal noninferiority. The gain over static k=4 was +0.0246, but its interval [−0.0017,+0.0516] included zero.

Expansion from k=6 to k=8 was much less predictable, with reported AUC=0.5341. Restricting the decision reduced opportunities for costly mistakes.

Sources: [initial report](../results/week4/processed/week4_scientific_report.md), [final C1 validation](../results/week4/qcca_v2_final_validation/final_validation_report.md). The latter mistakenly labels a 231-query/86-paper population as TEST in one paragraph; the evaluation is DEV, as stated throughout the same report.

## 8. Week 5 — Testing passage semantics and encoder adaptation

### W5-A: implementation pathology

The first content-aware allocator mixed semantic vectors with oversized rank embeddings. Rank norm was approximately 19.5 versus semantic norm approximately 2.12. Passage-aware models collapsed toward static k=8. Such behavior could not fairly establish the scientific value of passage semantics.

### W5-B: corrected representations and admissibility targets

Small rank initialization and pre-attention LayerNorm fixed the scale defect. A multi-label target marked every budget within 0.01 regret as admissible, rather than forcing a single smallest-k label.

The corrected passage-aware I3-T3 model achieved **0.3810 ±0.0090 F1**, mean k=5.57, and **1258.2 ±174.9 tokens** across three seeds. Query-only I1 achieved 0.3723; the passage-aware point estimate was higher by 0.0087. Passage-plus-tabular I4 achieved only 0.3489 at 924.5 tokens.

Against the Week-4 reference, the across-seed F1 difference was −0.0031, with interval [−0.0336,+0.0259], while token cost increased by 149.5, interval [+93.6,+207.4]. The result supports passage-related signal under this formulation, but not a superior quality–context tradeoff.

### W5-C: capacity ladder

| Variant | DEV F1, mean ± std | Context tokens | Interpretation |
|---|---:|---:|---|
| C0 frozen MiniLM | 0.3810 ±0.0090 | 1258.2 | Strongest tested semantic allocation reference |
| C1 last block trainable | 0.3620 ±0.0119 | 1158.9 | Quality degradation |
| C2 last two blocks trainable | 0.3743 ±0.0127 | 1337.6 | More cost without improvement over C0 |

C1−C0 was −0.0191 F1 with interval [−0.0377,−0.0031]. C2−C0 was −0.0068 with interval [−0.0312,+0.0175], plus 80.1 tokens. Gradient and parameter-change checks showed the intended layers actually trained; failure was not simply absent updates.

The safe conclusion is that **these frozen and shallow-adaptation MiniLM variants failed to beat the Week-4 baseline**. The stronger historical wording that small encoders are a “fundamental barrier” exceeds what a limited architecture and training search can establish. Representation drift, overfitting, and task mismatch remain plausible explanations, not fully isolated causes.

Sources: [W5 overview](../results/week5/README.md), [W5-B](../results/week5/content_allocator/w5b/week5b_report.md), [W5-C](../results/week5/content_allocator/w5c_encoder_adaptation/week5c_report.md).

## 9. Week 6 — Choosing which passages receive full fidelity

Week 6 changed the action from selecting a prefix length to selecting passage identities for full-text representation. The key comparison was BM25-prefix routing versus semantic top-k versus a learned router.

| Router | k=2 F1 / tokens | k=4 F1 / tokens | k=6 F1 / tokens |
|---|---:|---:|---:|
| BM25 prefix | 0.3038 / 526.5 | 0.3596 / 938.1 | 0.3857 / 1351.1 |
| Semantic top-k | 0.3127 / 509.6 | **0.3913 / 921.1** | **0.3953 / 1325.1** |
| Learned router | 0.2983 / 565.4 | 0.3572 / 994.7 | 0.3694 / 1422.2 |
| Fixed fusion | 0.3015 / 514.2 | 0.3782 / 935.9 | 0.3775 / 1344.0 |
| Reciprocal-rank fusion | **0.3258 / 513.6** | 0.3752 / 923.8 | 0.3791 / 1338.4 |

At k=4, semantic selection improved F1 by 0.0317 over the BM25 prefix with slightly fewer tokens. It approached BM25 k=8 quality using approximately half the context tokens. This was a major practical shift: improving the selection of full-fidelity passages mattered more than several attempts to learn the total budget.

Hybrid fusion did not consistently beat semantic ranking. RRF was strongest at k=2 but weaker at k=4 and k=6. Learned routing underperformed the corresponding semantic settings.

These are table point estimates; this synthesis does not infer significance without paired evidence. Also, BM25 latency is recorded as 0.0 in these tables, indicating incomplete comparable timing rather than free inference.

The Week-6 README says no experiments were executed, but the detailed result tables contain completed measurements. The README is stale.

Sources: [routing table](../results/week6/passage_routing/tables/w6_main_results.csv), [hybrid table](../results/week6/hybrid_routing/tables/w6b_main_results.csv).

## 10. Week 7 — Stress-testing adaptive allocation against semantic baselines

Week 7 tested whether learned budgets and representations could improve on semantic passage routing. The mature semantic surface reports k=4 at 0.3909 F1/921.1 tokens, k=6 at 0.3957/1325.1, and k=8 at 0.4010/1737.4. Small changes from Week 6 should not be silently treated as identical runs.

| Experiment | DEV F1 | Tokens | Finding |
|---|---:|---:|---|
| W7-A old W4 allocator + semantic routing | 0.3782 | 1095.0 | Naive composition did not preserve W4 quality |
| W7-B old features retrained on semantic targets | 0.3522 | 815.4 | Lower cost, substantial quality loss |
| W7-C retrieval-aware sufficiency classifier | 0.3783 | 1503.8 | Over-allocated relative to semantic k=4 |
| W7-C2 direct utility/regret regression winner | 0.3922 | 1622.1 | Near k=4 quality at much higher cost |
| W7-C3 anchor-and-deviate | 0.3801 | 903.1 | Small token saving, quality loss |
| W7-D frozen cross-encoder + GBDT | 0.3661 | 792.0 | Cheaper, weaker answers |
| W7-D hierarchical frozen | 0.3909 | 921.1 | Returned static k=4 behavior |
| W7-D hierarchical unfrozen | 0.3909 | 921.1 | Returned static k=4 behavior |
| W7-D listwise Mistral-7B LoRA | 0.3877 | 901.6 | Did not improve on static k=4 |

W7-A shows that budget decisions and passage identity choices interact. A classifier fitted around BM25-prefix outcomes does not automatically remain calibrated after changing the routing policy. W7-B indicates that simple retraining of the old features did not solve that mismatch.

W7-C2 improved on some weaker allocators but selected mean k=7.39. A near-static high-budget policy can score reasonably without demonstrating successful selective allocation. W7-C3’s mean k=3.90 was close to the four-passage anchor, yet the deviations reduced aggregate quality.

The semantic epsilon-oracle remained approximately 0.487 F1 at mean k=3.3 and 775.2 tokens. Thus the retrospective opportunity remained large even though the learned policies failed to recover it reliably.

### W7-E: intermediate fidelity

| Representation | DEV F1 | Tokens |
|---|---:|---:|
| Semantic k=4: 4 full +6 compressed | 0.3909 | 921.1 |
| Initial medium extracts, nominal M96 | 0.3672 | 1107.0 |
| Corrected extract-only M64 | 0.3755 | 1158.2 |
| Repaired hybrid M64 plus retained latent embeddings | 0.3861 | 1168.2 |
| Semantic k=6: 6 full +4 compressed | 0.3957 | 1325.1 |

Keeping latent embeddings improved the repaired hybrid by +0.0106 over corrected extract-only, but it still had lower F1 and more tokens than k=4. Medium-fidelity extraction did not establish a useful intermediate operating point in these variants.

Sources: the `tables/*main*results.csv` files under each [Week-7 experiment directory](../results/week7), and [W7-E representation ablation](../results/week7/w7e_multifidelity/tables/w7e_m0_vs_m1_ablation.csv).

## 11. Week 8 — Draft feedback, promotion utility, attribution, and adaptation

### W8-A: measuring the value of promoting one passage

A semantic k=4 draft is generated first. Each of the remaining six candidates is promoted individually to full text, yielding its actual answer-quality change. The available surface contains **10,476 FIT candidate rows** and **2,052 INTERNAL-VAL candidate rows**, covering all 1,746 and 342 queries respectively.

Direct recomputation from the saved surfaces gives:

| Split | Base k=4 F1 | Best forced single promotion | Exact best of STOP or promotion | Queries with best gain >0.01 |
|---|---:|---:|---:|---:|
| FIT | 0.385631 | 0.481899 | 0.485209 | 528 /1746 |
| INTERNAL-VAL | 0.389061 | 0.480024 | 0.480760 | 98 /342 |

These maxima are outcome-aware diagnostics. The W8-C epsilon-selective oracle is slightly different: 0.485174 FIT and 0.480670 validation. Exact-max and epsilon-selective numbers must retain their labels.

The surface establishes large single-promotion potential. It does not establish that a model can find the right candidate or decide safely when to stop.

### W8-B: uncertainty and sufficiency gates

Measured token diagnostics U1–U12, logistic gates, gain weighting, MLPs, and stochastic self-consistency were explored. The older gate-baseline table reports values such as 0.4742 for B0-GW, but it uses an oracle selector. Such scores do not represent an end-to-end learned controller. W8-E explicitly rebuilt five paper-grouped FIT OOF gate predictions rather than reusing validation-only/random-fold estimates.

### W8-C: learning passage utility

The selector ladder compared tabular utility, query/passage cross-encoders, draft-conditioned variants, a listwise model, and a fused U4-A model. U4-A was frozen using FIT OOF selection diagnostics.

| Selector | FIT always-promote F1 | FIT oracle-gated diagnostic F1 |
|---|---:|---:|
| DraftSemantic heuristic | **0.405454** | 0.424430 |
| U0 tabular | 0.396208 | 0.413699 |
| U1 | 0.398401 | 0.419636 |
| U2-F | 0.397832 | 0.419232 |
| U2-A | 0.402600 | 0.422009 |
| U3 | 0.397721 | 0.413827 |
| U4-A | 0.402770 | **0.424463** |

The tiny oracle-gated advantage of U4-A over DraftSemantic did not correspond to superior always-promote F1. On validation, U4-A always-promote F1 was 0.408329, with 13.45% HELP and 7.02% HURT. Its reported 0.415323 “final downstream F1” used an oracle gate and must remain a diagnostic.

### W8-D: attribution closeout

Masking attribution and marginal-information-gain scores tested whether compressed passages’ support for the existing draft predicts the value of exposing their full text. A whitespace mismatch affected 23 FIT and five validation questions; those masking measurements were repeated, and superseded results were archived.

On validation, DraftSemantic achieved 0.412572 always-promote F1. Attribution-only S5 achieved 0.389016; MIG-only S6 0.396327; S7 and S8 both 0.412572. U4-B, with attribution features, reached 0.416716, but its difference from DraftSemantic had interval [−0.017637,+0.025976]. It was not a confirmed improvement and was not the FIT-selected confirmation candidate.

The frozen decision was **no demonstrated attribution benefit in the tested variants; online attribution not retained**. Original extraction required about 8.8 hours, plus targeted repair. Online masking would require seven SARA forward passes for a nonempty draft, so any future benefit needs a complete compute accounting.

The conceptual distinction matters: supporting a current compressed-context answer is different from improving an answer when a passage is exposed at full fidelity. Neither attention nor masking support is automatically a promotion-utility estimate.

### W8-E: the frozen measured controller

The controller was selected on FIT OOF realized downstream F1. A global 0.003 tolerance preferred lower cumulative prompt cost near the best FIT quality. The selected policy was **E5-DraftSemantic**, not whichever row later won validation.

| Policy | Validation F1 | Promotion rate | HURT / all queries | Cumulative prompt tokens |
|---|---:|---:|---:|---:|
| E0: return k4 draft | 0.389061 | 0% | 0% | 935.16 |
| E1: always promote DraftSemantic | **0.412572** | 100% | 9.94% | 2068.39 |
| E2: gate + DraftSemantic | 0.406122 | 62.57% | 8.77% | 1651.94 |
| E3: gate + U4-A | 0.401963 | 62.57% | 7.02% | 1675.19 |
| **E5-DraftSemantic, selected** | **0.400080** | **42.11%** | **4.68%** | **1416.11** |
| E5-U4-A | 0.399381 | 48.54% | 3.51% | 1509.90 |
| Static k6 reference | 0.402005 | — | — | 1352.27 |

The selected policy’s final prompt averaged only 1017.59 tokens, but its cumulative prompt cost was 1416.11 because promoted queries use two passes. It saved 31.54% cumulative prompt tokens versus always promoting, while sacrificing 0.012493 F1; the paired interval [−0.025562,−0.000327] was entirely negative.

Against k4, its gain was +0.011018 with interval [−0.004866,+0.027197]. Against k6, its difference was −0.001926 with interval [−0.022801,+0.019613], while cumulative prompt cost increased 4.72%. **The selected controller did not demonstrate a better quality/cost operating point than k6.**

The policy replay uses real saved deterministic generations and preserves harmful outcomes. It is not a fresh deployed latency benchmark. E4 was omitted by the closeout decision, and no second-promotion experiment is reported.

### W8-F: query-conditioned latent adapters

The full QCLA artifact records two training epochs, seed42, and validation on 342 queries.

| Variant | Validation F1 | Trainable parameters | Final prompt tokens | Cumulative prompt tokens |
|---|---:|---:|---:|---:|
| QCLA-0 baseline | 0.389061 | 0 | 935.16 | 935.16 |
| QCLA-1 | 0.393799 | 52,439,552 | 935.16 | 935.16 |
| QCLA-2, draft-conditioned | 0.398994 | 62,925,312 | 935.16 | 1870.32 |

QCLA-1 improves the point estimate by +0.004738 at the same prompt length; QCLA-2 by +0.009932 but requires the draft pass. Conditioning encode time was approximately 0.1024 seconds for QCLA-1 and 0.2048 for QCLA-2. Estimated cumulative generation time was 2.5231 and 5.2472 seconds respectively. These measurements are not complete deployment latency.

No paired confidence interval or final DEV evaluation is established by the inspected QCLA summary. The initial tracker’s 20.9M/25.1M parameter claims are superseded by the full artifact’s actual parameter counts above.

### W8-G: completed generator adaptation — results from the latest commit

Commit `12901d0` (“W8G executed”, 20 October 2026) adds full comparison tables, protocol/provenance records, training examples and four prediction files. This supersedes the previous missing-results assessment. The older readiness document still ends by saying full training has not run; the full-run artifacts and executed notebook take precedence.

**Design.** Original SARA is compared with uniform supervised fine-tuning (uniform SFT), hard-case weighted LoRA (HC-LoRA), and outcome-weighted LoRA (OW-LoRA). All three train on the same 1,746 FIT questions, using identical outcome-derived training prompts and first-reference gold targets. Therefore, this is a comparison of training weights under matched prompts; uniform SFT is not a separate random-context baseline.

The training case composition is 528 helpful, 230 distractor-safe/hurt-only, and 988 neutral questions. Helpful cases expose the best measured FIT promotion; remaining cases retain semantic k4. Using outcomes to construct supervised FIT examples is part of training, not an inference-time oracle. HC raw weights are 2/2/1 for helpful/hurt-only/neutral. OW uses 1+2×best positive delta for helpful cases, 1+2×the magnitude of the most negative delta for hurt-only cases, and 1 otherwise. Weights are normalized by each variant’s complete FIT mean.

Each variant starts afresh from original SARA and adds 6,815,744 trainable parameters on q/k/v/o projections. The original model, existing adapter and projector remain frozen. The fixed protocol uses seed42, two epochs, learning rate 1e-4, rank8, alpha16, dropout0.05, and gradient accumulation16. No validation-selected gamma, epoch, or checkpoint is reported. Training logs verify new-adapter-only gradients and unchanged original parameter versions. Inputs were checked for preserved compression slots and target boundaries.

**Two distinct evaluation settings.** All variants are evaluated on 342 previously inspected INTERNAL-VAL questions from 116 papers:

- **Direct k4:** the adapted generator answers every question directly from semantic k4.
- **Frozen controller:** the original E5-DraftSemantic actions, thresholds, selected passage identities, and original drafts remain fixed. STOP returns the original unadapted draft; only PROMOTE uses the adapted revision generator. Consequently this evaluates revision adaptation under the old policy, not a controller recalibrated around an adapted first pass.

| Generator | Direct k4 F1 | Gain over original k4 | Frozen-controller F1 | Gain over original controller |
|---|---:|---:|---:|---:|
| Original SARA | 0.389061 | — | 0.400080 | — |
| Uniform SFT | 0.406722 | +0.017660 | **0.412340** | **+0.012260** |
| HC-LoRA | **0.415027** | **+0.025966** | 0.409215 | +0.009135 |
| OW-LoRA | 0.403986 | +0.014925 | 0.403553 | +0.003473 |

All three adaptation variants improve both original-generator point estimates. HC-LoRA is strongest for direct k4; uniform SFT is strongest with the frozen controller. OW-LoRA is below uniform SFT in both settings. Thus the completed results support generator adaptation as a promising direction, but do not establish an advantage for continuous outcome weighting.

**Costs and tradeoffs.** Every direct k4 variant uses 935.16 final/cumulative prompt tokens. Every frozen-controller variant retains the same 42.11% promotion rate, 1017.59 final prompt tokens, and 1416.11 cumulative prompt tokens. Direct k4 therefore uses 33.96% fewer cumulative prompt tokens than the frozen-controller setting.

Estimated cumulative generation seconds for original/uniform/HC/OW are respectively 2.692/2.887/3.047/2.966 for direct k4 and 3.931/4.046/4.151/4.098 with the controller. These combine newly timed generations with cached original baseline/draft timings; they are not contemporaneous end-to-end latency benchmarks. Training epoch totals are approximately 6.28 hours for uniform, 6.59 for HC, and 6.68 for OW, excluding validation and other overhead. Equal prompt lengths do not imply zero adaptation compute cost.

HC-LoRA direct k4 has higher point-estimate F1 than HC-LoRA with the controller by 0.005813, while using fewer cumulative prompt tokens. OW direct k4 is also slightly higher than its controller result. Uniform gains 0.005618 from the controller, at an additional 480.95 cumulative prompt tokens. The value of promotion therefore depends on the generator and cannot be assumed to transfer unchanged from the original SARA policy.

HC-LoRA direct k4 also exceeds the original static k6 validation point estimate (0.402005) by 0.013022 while using about 30.85% fewer prompt tokens. This is a useful cross-artifact reference, **not a matched adaptation comparison**: adapted k6 and always-promote adaptation controls were not evaluated here. These validation observations do not supersede the original W8-E conclusion about its unadapted controller.

**HELP/HURT.** Relative to original SARA under the same setting, direct-k4 HELP/HURT percentages are 19.88/16.08 for uniform, 21.35/15.20 for HC, and 19.59/14.04 for OW. With the frozen controller they are 8.77/5.56, 8.77/6.14, and 7.31/5.85 respectively. These are per-query changes relative to the corresponding original generator, not promotion HELP/HURT relative to each adapted model’s own draft. Higher mean F1 does not mean every question improves.

### Additional paired analysis of the saved W8-G predictions

For this report update, prediction hashes were checked against the full summary; each variant has 684 unique query/context rows, exactly 342 per setting. Recomputed means agree with the published comparison. Passage identities, actions, compressed/full indices, prompt counts, and original STOP predictions match across variants.

The following **new retrospective analysis** uses 10,000 paired paper-bootstrap resamples, seed42, resampling all 116 papers and retaining all questions within each sampled paper. Estimates are query-weighted. These intervals condition on the saved predictions; they do not include retraining variability, correct for multiple comparisons, or convert previously inspected validation into an untouched test.

| Comparison | F1 difference | Paired 95% interval |
|---|---:|---:|
| Uniform k4 − original k4 | +0.017660 | [−0.007460,+0.042038] |
| HC k4 − original k4 | **+0.025966** | **[+0.000704,+0.050589]** |
| OW k4 − original k4 | +0.014925 | [−0.010466,+0.040497] |
| HC k4 − uniform k4 | +0.008306 | [−0.005427,+0.022684] |
| OW k4 − uniform k4 | −0.002735 | [−0.020941,+0.016052] |
| Uniform controller − original controller | +0.012260 | [−0.009237,+0.032225] |
| HC controller − original controller | +0.009135 | [−0.011342,+0.028646] |
| OW controller − original controller | +0.003473 | [−0.016636,+0.022509] |
| HC controller − uniform controller | −0.003125 | [−0.014620,+0.007279] |
| OW controller − uniform controller | −0.008787 | [−0.018893,+0.001103] |

Only HC direct k4 versus original k4 has a strictly positive interval among these comparisons, and its lower bound is close to zero. The evidence supports a development improvement for that fixed-prediction comparison, with the stated exploratory and single-seed limits. HC’s advantage over matched-context uniform SFT remains uncertain, as do all adapted-controller gains over the original controller.

**Updated scientific conclusion.** W8-G provides measured evidence that changing the generator can improve answers without exposing additional text. It shifts the research story toward utilization-side adaptation. However, the gains are not specific proof of outcome-weighting superiority, and the frozen promotion controller does not consistently add value once the generator is adapted. The next discriminating comparison should include matched adapted k4/k6 and adapted promotion policies before concluding that adaptivity itself delivers the improvement.

Sources: [W8-G full comparison](../results/week8/w8g_lora/full/comparison.csv), [full summary](../results/week8/w8g_lora/full/summary.json), [frozen protocol](../results/week8/w8g_lora/full/protocol_frozen_before_validation.json), [provenance](../results/week8/w8g_lora/full/provenance.json), [FIT verification](../results/week8/w8g_lora/full_FIT_input_verification.json). Additional analysis: [bootstrap results](w8g_paired_paper_bootstrap.csv), [reproducible analysis script](analyze_w8g_saved_results.py).

### W8-H through W8-J: remaining integrated closeout

W8-G is complete. No reportable files were found in the inspected W8-H integrated-system, W8-I ablation, or W8-J final-DEV directories. Smoke tables and archived provisional artifacts are not evidence of a completed final benchmark. The initial Week-8 tracker is stale for A–G and cannot serve as the current execution ledger.

Sources: [promotion surfaces](../results/week8/surfaces), [W8-C frozen configuration](../results/week8/w8c_utility/w8c_frozen_selector_config.json), [W8-D closeout](../results/week8/w8d_closeout/CLOSEOUT.md), [W8-E results](../results/week8/w8e_controller/RESULTS.md), [W8-E verification](../results/week8/w8e_controller/final_verification.json), [QCLA full results](../results/week8/w8f_qcla/full/summary.json).

## 12. What the complete sequence establishes

**First, there is real outcome heterogeneity.** Fixed budgets leave substantial oracle headroom. This finding recurs across the original response surface, the semantic surface, and the single-promotion surface.

**Second, predicting useful context is harder than recognizing topical relevance.** Cheap features, sentence embeddings, cross-encoders, and larger listwise models did not consistently translate oracle potential into better deployed policy outcomes. Classification accuracy and oracle-gated selector scores can conceal that difficulty.

**Third, semantic selection is an important practical baseline.** Four well-chosen full-text passages are much more useful than four BM25-prefix passages in the measured experiments. Every later adaptive policy must be compared with this stronger baseline rather than only the original SARA default.

**Fourth, exposing evidence does not guarantee using it.** The Week-8 examples motivate separating selection, representation fidelity, and generator utilization. This is a useful framework, though repository-wide utilization-failure rates should not be inferred from two example questions.

**Fifth, adaptation and feedback add real costs.** A smaller final prompt can coexist with higher cumulative prompt cost. The measured one-promotion controller illustrates this directly. Query-conditioned adapters also add parameters and conditioning work even when prompt length is unchanged. W8-G adds an important qualification: generator adaptation can improve direct-k4 answers at unchanged prompt length, although training and generation costs remain real.

**Sixth, negative results narrowed the problem.** The project ruled out easy versions of “just predict k,” “just add semantic embeddings,” “just fine-tune a few encoder blocks,” and “just use attribution” within the tested settings. These are bounded empirical findings, not universal impossibility results.

## 13. Claims suitable for a report or presentation

| Claim | Evidence status |
|---|---|
| Per-query textual allocation has substantial retrospective headroom | Supported on development response surfaces |
| Week-4 C1 reduces context cost relative to k8 | Supported; quality noninferiority not formally established |
| Semantic top-k improves the k4 point estimate over BM25-prefix routing | Supported by measured DEV tables |
| Week-5 semantic/adapted allocators beat the Week-4 tradeoff | Not supported |
| Week-7 learned policies beat strong semantic static baselines | Not demonstrated |
| Intermediate-fidelity extracts improve the frontier | Not demonstrated in tested variants |
| W8 attribution improves the deployed policy | Not demonstrated; excluded from online policy |
| Frozen W8 controller beats k6 in quality and cumulative prompt cost | Not supported |
| QCLA has modest validation point-estimate improvements | Supported; uncertainty and generalization unresolved |
| W8-G generator adaptation improves validation point estimates | Supported in both settings; HC direct-k4 vs original has a positive unadjusted paired interval |
| W8-G outcome weighting outperforms uniform SFT | Not supported; OW point estimates are lower in both settings |
| Frozen promotion adds value to every adapted generator | Not supported; HC and OW direct-k4 point estimates are higher than their controller results |
| Human evaluation confirms answer correctness | Unsupported; recorded as pending |
| Final integrated DAFiR results are complete | Not established by inspected artifacts |
| Untouched TEST performance establishes generalization | Not available in the inspected results |

## 14. Remaining work for a complete end-to-end research conclusion

1. Recover the missing Week-1 provenance. W8-G numerical comparisons, training protocol, prediction hashes and cost definitions are now incorporated; add further seeds and matched adapted-k6/promotion controls to distinguish weighting gains from general fine-tuning gains.
2. Reconcile Week-0 retrieval/evidence discrepancies and normalize token definitions before preparing a single cross-week comparison figure.
3. Freeze the final candidate and selection rule before further confirmation; do not select a validation winner retrospectively.
4. Complete the integrated-system evaluation and meaningful ablations, comparing semantic k4, semantic k6, the frozen controller, relevant adapters, and generator adaptation under the same split and prompting protocol.
5. Measure cumulative prompt/output tokens, routing overhead, conditioning, and actual end-to-end latency. Bootstrap paired paper-level quality differences and state any noninferiority margin explicitly.
6. Complete the genuine blinded human audit, keeping model-generated heuristics separate from human scores.
7. Conduct any authorized final test evaluation only after the policy and analysis are frozen. Existing repeated development results cannot substitute for it.

## 15. A ready-to-use narrative

> We began by asking whether every scientific question should receive the same amount of full-text evidence. Our dataset and real-model diagnostics showed that evidence structure and passage utilization vary, while fixed-budget sweeps revealed substantial per-query oracle headroom. Turning that headroom into a reliable decision rule proved difficult. A lightweight selective-expansion classifier provided a useful context-saving baseline, but richer semantic budget models and shallow encoder adaptation did not consistently improve it. We then found that selecting the right passages for full-text representation was more effective than learning the passage count alone: semantic routing made a four-passage baseline particularly strong. Subsequent budget learners, larger routers, and intermediate-fidelity representations struggled to beat it. Week 8 introduced draft-conditioned evidence promotion and measured the value of individual promotions. Although these surfaces again showed substantial potential, the frozen learned controller did not outperform static k6 on quality and cumulative prompt cost, and attribution did not provide a confirmed benefit. The work therefore identifies a persistent gap between relevance, fidelity, and actual generator utilization. The completed generator-adaptation study strengthens the utilization-side direction: direct-k4 HC-LoRA reaches 0.4150 validation F1 at unchanged prompt length, while uniform SFT leads under the frozen promotion controller at 0.4123. Outcome weighting does not beat the simpler variants, and adding promotion does not consistently improve adapted generation. A final integrated evaluation, complete runtime accounting, and human validation remain necessary before claiming end-to-end superiority.

## Appendix: Reporting corrections applied in this synthesis

- Week-2 oracle 0.5008 includes k0; Week-3 oracle 0.4895 excludes it. They are not competing estimates of the same bound.
- Original and later prompt-token conventions remain separate. No cross-phase token normalization was invented.
- Initial simulated Week-0 diagnostics and smoke-only Week-8 results are excluded as empirical system results.
- Initial W4 collapse is retained as a historical result; the later C1 policy is the appropriate W4 comparison baseline.
- W5 prose saying “fundamental barrier” is narrowed to the tested models and protocols.
- Oracle-gated W8-C/B results are explicitly separated from always-promote and learned-STOP performance.
- W8-D whitespace repairs and the frozen candidate-selection rule take precedence over superseded attribution outputs.
- W8-E cumulative prompt cost takes precedence over final-prompt length for comparing its two-pass operation with static k6.
- The completed W8-G results from commit `12901d0` supersede the earlier missing-files assessment and stale readiness prose. H–J remain without a reportable final closeout in the inspected directories. New W8-G paired intervals are explicitly labeled retrospective report analysis.
- This report did not rerun GPU experiments, train models, or access the held-out test dataset. Numerical checks used saved artifacts only.
