# Project glossary: methods, metrics and formulas

**Companion to the [final comprehensive project report](FINAL_PROJECT_REPORT.md)**  
**Scope:** the CS787 RAG project, including historical Weeks 0–11 and the final Week12 study.  
**Date:** 2 November 2026.

This document explains what the names mean, what each method actually does, how the main measurements are calculated, and how to interpret them. Examples are illustrative unless explicitly identified as measured results. Historical names are always interpreted within their week and experiment; the same short label can mean different things elsewhere.

## 1. The whole pipeline in one picture

```text
                         Candidate pool
                              ↓
Question → retriever → passages / sentences → evidence selector → reader → answer
                                                     ↑
                                              token budget
```

- The **retriever** finds potentially useful source text.
- The **selector** decides which retrieved information to expose.
- The **reader**, also called the **generator**, uses that evidence and the question to produce an answer or classification.
- The **evaluator** compares the prediction against accepted answers or labels. Gold information belongs here, not in inference-time selection.

A retrieval failure means the needed evidence was absent from the available candidates. A selection failure means it was available but omitted. A utilization failure means supplying information did not produce the intended improvement. A final wrong answer alone cannot identify which failure occurred.

## 2. Basic vocabulary

| Term | Meaning in plain language | Example or caution |
| --- | --- | --- |
| RAG | Retrieval-augmented generation: answer using retrieved source material | Retrieval and answer generation are separate stages |
| Query / question, `q` | The information request | For SciFact, the input is a claim to classify |
| Reader / generator | The model that produces the prediction | Final matched study: Qwen/Qwen3-8B |
| Corpus | The searchable collection | A QASPER paper; HotpotQA provided contexts; SciFact’s abstract corpus |
| Document | A source item in that collection | A scientific paper or abstract |
| Passage / chunk | A contiguous block of a document | Chunk boundaries affect what fits and what information stays together |
| Sentence | A smaller source unit produced by deterministic segmentation | Abbreviations and scientific notation make splitting nontrivial |
| Proposition | A small rule-derived unit in the project’s hierarchy | This is an operational unit name, not a guarantee of logical atomicity or truth |
| Granularity | The size/type of evidence unit | Passage, sentence and proposition are different granularities |
| Context / evidence block | Material supplied with the question | It need not be the whole retrieved pool |
| Token | A tokenizer’s text unit | May be a word, word fragment or punctuation; not identical to a word |
| Tokenizer | The model-specific rule converting text to token IDs | Different readers may count the same text differently |
| Evidence budget, `B` | Maximum tokens allocated to evidence | Final budgets: 256 and 1024; question/instructions are additional |
| Prompt | The entire input supplied to the reader | Includes instructions, evidence, question and chat framing |
| Prediction / answer, `ŷ` | The reader’s output | A short answer or one allowed SciFact label |
| Gold / reference | Accepted answer, class or annotated supporting evidence | Evaluation-only information |
| Draft, `y₀` | An initial answer made before an extra intervention | A draft is not an authoritative source |
| Revision | A later answer after inspecting extra evidence | May improve, worsen or leave the draft unchanged |
| Embedding | A numeric representation of text | Used to compare semantic similarity |
| Latent representation | Learned vectors supplied through a model interface | Not the same as readable source text |
| Adapter / projector | Trainable components that adapt a model or map vectors into its input space | Their presence can confound whole-system comparisons |
| Checkpoint | Saved model/training state | Matching model names alone does not establish matching checkpoints |
| Source-exact / extractive | Evidence copied from validated source spans | Exact copying does not guarantee relevance or answer correctness |
| Generative extraction | An LM produces an intermediate evidence representation | Adds cost and can fail formatting/support checks |
| Deterministic | Fixed inputs/settings and specified tie rules lead to repeatable decisions | GPU numerical behavior still requires appropriate controls |

### 2.1 The different meanings of “k”

| Symbol / label | What it counts | What it does not mean |
| --- | --- | --- |
| Retrieval `top-k`, `W`, or retrieval width | Number of candidate units retrieved | Number finally shown to the reader |
| Early SARA `k` | Candidates exposed as explicit text | A sentence count or token budget |
| SARA `n` | Total candidates in the specified text/latent setup | Always the number of whole passages included in plain text |
| W9 `K_E` | Coarse evidence units used in the post-draft intervention | E3’s number of selected sentences |
| E3 `sentence_k` | Requested count of selected sentences | Always the actual count available |
| `actual_k` | Number of available units actually selected | Permission to duplicate units to reach the request |
| Final `B=256/1024` | Evidence-token limit | A count of sentences or passages |
| `k*` / oracle k | Retrospective best tested action count | A prediction available at inference time |

**Important:** `FIXED_SENTENCE` means a **fixed sentence-selection policy**. It does not mean “always use three sentences.” The number varies with sentence lengths and the token budget. E3-k3, by contrast, specifically requests three sentences.

## 3. Retrievers, routing and similarity

### 3.1 Retrieval and routing names

| Name | What it does | Interpretation limit |
| --- | --- | --- |
| BM25 | Scores lexical matches between a question and source text | Strong lexical matching can remain competitive with dense retrieval |
| MiniLM semantic retrieval / routing | Compares embeddings from the established compact encoder, commonly all-MiniLM-L6-v2 | Earlier MiniLM results are not SFR results |
| SFR | Uses Salesforce/SFR-Embedding-Mistral for dense semantic comparison | Final frozen retriever; not universally best in every dataset/budget cell |
| Semantic routing | Reorders/selects which members of an existing pool remain explicit | Does not necessarily change the original searchable corpus |
| BM25 prefix | Keeps the leading BM25-ranked candidates | A simpler ordering control |
| DraftSemantic | Uses similarity to the initial draft to guide candidate selection | A wrong draft may bias the selection |
| QuerySemantic | Uses question–candidate semantic similarity | Relevance is not proof that the candidate helps the answer |
| QueryDraft | A specified question-and-draft selection variant | Exact combination belongs to its phase configuration |
| SANE-adapted | The project’s adapted LM-assisted selection route, using an extra LM stage | Not automatically a faithful reproduction of every original SANE setting |
| RRF | Reciprocal rank fusion: combine rankings using reciprocal ranks | Rank fusion is different from averaging raw similarity scores |
| Learned router / selector | A trained model scores candidates or actions | Must be tested on downstream answer quality, not label accuracy alone |
| FIXED-5 / SEM-4 and related chunk names | Identifiers for registered chunking configurations | The suffix is not automatically the final token budget or sentence count; consult the frozen configuration |

### 3.2 Cosine similarity

For nonzero embedding vectors `u` and `v`:

\[
\operatorname{cos}(u,v)=\frac{u^\top v}{\|u\|_2\|v\|_2}.
\]

After unit normalization, cosine equals the dot product. Larger values mean greater alignment in that embedding space. Negative values are possible. Cosine is not a calibrated probability, entailment score or factual-support certificate.

### 3.3 Reciprocal rank fusion

A standard RRF form is:

\[
\operatorname{RRF}(d)=\sum_{r\in\mathcal R}\frac{1}{c+\operatorname{rank}_r(d)}.
\]

`d` is a candidate, `r` is one ranking, and `c` is a stabilizing constant. Use the experiment’s configured constant and missing-candidate convention. Do not assume every historical fusion experiment used the same settings.

## 4. Main final methods: exactly what each does

The definitions below describe the final Week12 implementation. All matched final methods share the frozen reader, retrieval pool, prompt/parser, and relevant budget. Some historical Week11 allocation rules differ.

### 4.1 `NONE`

**Procedure:** provide the question/claim and instructions without retrieved evidence.

**Purpose:** determine whether the complete evidence-assisted pipeline helps over the no-evidence reader.

**Caution:** no evidence does not mean no prompt or no LM call. The identical NONE prediction can be reused across both budget labels because those labels add no evidence.

### 4.2 `RAW_STRICT`

**Procedure:** visit whole passages in rank order. Add a passage if the cumulative context fits. Stop at the first passage that would exceed the budget.

**Purpose:** preserve the historical strict-prefix passage control.

**Consequence:** if the first passage is oversized, no passage may be exposed even when later, shorter passages would fit. This creates a packing limitation that can favor smaller units in a comparison.

**Status:** implemented secondary control; not a final Part-M system.

### 4.3 `RAW_SKIP`

**Procedure:** visit whole passages in retrieval order. If the next passage fits, include it. If not, skip it and try later passages. Do not silently truncate oversized passages.

**Purpose:** a fairer whole-passage baseline than stopping at the first failure.

**Consequence:** preserves whole-unit context but can leave unused budget. It can still produce empty evidence when no whole passage fits.

**Status:** final validation and confirmation control.

### 4.4 `FIXED_SENTENCE`

**Procedure:**

1. Split the frozen retrieved pool into source sentences.
2. Rank sentences by question–sentence semantic similarity, with deterministic source-order ties.
3. Consider the ranked sentences and include each if it fits; skip non-fitting units and continue.
4. Restore selected units to source order for presentation.
5. Generate the answer from the same frozen reader.

**Purpose:** a simple query-aware way to suppress irrelevant surrounding text under a token limit.

**Consequence:** sentence count varies. It does not create a generated summary, use a fixed top-three limit, or learn a global k predictor.

**Status:** final frozen primary policy, evaluated at 256 and 1024 evidence tokens.

### 4.5 `C2`

**Procedure:** rank exact source sentences by the question and restrict consideration to the top eight. The final port preserves its specified rank-prefix allocation behavior: stop when the next considered unit fails to fit.

**Purpose:** a historical simple extractive compression control.

**Difference from FIXED_SENTENCE:** C2 caps the candidate list at eight and has prefix behavior. FIXED_SENTENCE considers the larger sentence pool with skip-to-fit behavior. At larger budgets, C2 may saturate before the budget is used.

**Status:** evaluated in final validation at 256, not in the uniform two-budget final confirmation registry.

### 4.6 `H3`

**Procedure:** prioritize query-ranked sentences from the top three retrieved passages, then query-ranked rule-derived propositions from the remaining passages. The fixed allocation uses its specified prefix rule and restores chosen units to source order.

**Purpose:** a prevalidated hierarchical evidence control.

**Caution:** H3 is one specific fixed policy. It does not mean every hierarchical/mixed-granularity method, nor that its hierarchy was learned dynamically for each query.

**Status:** final validation and confirmation contextual control.

### 4.7 `QBGF_FULL`

**Full name:** query-conditioned budgeted granularity fusion.

**Procedure:** for each retrieved passage, construct mutually exclusive alternatives:

- DROP: expose nothing;
- SENT_1 / SENT_2 / SENT_4 / SENT_8: expose that many highest-scoring sentences when available;
- FULL: expose the whole passage.

Score alternatives using question-dependent passage weight and selected sentence relevance mass. Use a deterministic budget-constrained optimizer to select at most one option from each passage, then restore selected source text to document order.

**Purpose:** let different passages receive different fidelity under the same total budget.

**Caution:** the optimizer maximizes a relevance surrogate. It does not know the true answer F1. “Exact optimization” therefore does not mean “guaranteed best answer.”

**Status:** proposed method; not universally promoted after validation; retained as a frozen comparison system in confirmation.

### 4.8 QBGF mechanism ablations

| Name | What changes relative to FULL | What the comparison tests |
| --- | --- | --- |
| `QBGF_NO_QUERY` | Use uniform passage weights and question-independent sentence scores/order inside the same retrieved pool | Value of query-conditioned allocation/fidelity selection; retrieval is still question-dependent |
| `QBGF_SENTENCE_ONLY` | Remove FULL passage options; keep sentence options and DROP | Whether allowing whole passages adds benefit beyond the QBGF sentence allocator |
| `QBGF_PASSAGE_ONLY` | Remove sentence options; keep FULL and DROP | Whether sentence options add value beyond whole-passage optimization |
| `QBGF_NO_MASS` | Replace relevance-weighted sentence mass with the fraction of sentences selected | Value of the mass surrogate |
| `QBGF_GREEDY` | Replace exact dynamic programming with a specified incremental utility/cost heuristic | Value of the solver under the same design |
| `QBGF_BM25` | The configured lexical-retrieval variant | Effect of that retrieval substitution, not query ablation |

NO_MASS/GREEDY and other secondary variants being implemented does not establish that their full runs were executed. Final confirmation deliberately excludes these mechanism ablations.

### 4.9 The QBGF objective, with formulas

Let `i` identify one retrieved passage, `j` one sentence within it, and `a` an available representation option.

Question–passage similarities `rᵢ` become weights:

\[
R_i=\frac{e^{\tau_p r_i}}{\sum_{h=1}^{W} e^{\tau_p r_h}}.
\]

Within passage `i`, sentence similarities `sᵢⱼ` become sentence mass:

\[
p_{ij}=\frac{e^{\tau_s s_{ij}}}{\sum_{\ell} e^{\tau_s s_{i\ell}}}.
\]

For an option exposing selected sentences `Sᵢₐ`:

\[
m_{ia}=\sum_{j\in S_{ia}}p_{ij},\qquad u_{ia}=R_i m_{ia}.
\]

FULL has mass 1; DROP has mass 0. The registered temperatures use `τₚ=τₛ=5`. A larger multiplier sharpens the distribution; it is not the inverse-temperature notation used in every other implementation.

With token cost `cᵢₐ` and binary choice `xᵢₐ`, solve:

\[
\max_x\sum_i\sum_a x_{ia}u_{ia}
\quad\text{subject to}\quad
\sum_i\sum_a x_{ia}c_{ia}\le B,
\qquad \sum_a x_{ia}=1.
\]

The equality is feasible because DROP is an option. This is a **multiple-choice knapsack problem**. The final implementation uses additive token-ID block costs and deterministic tie handling, including preference for lower cost when objective values tie. It is exact for this serialized-cost/utility problem, not for unknown downstream answer quality.

Source: [Week12 algorithm](../src/week12/algorithm.py).

### 4.10 A packing example

Suppose the budget is 100 tokens and the ranked whole-passage costs are 120, 60 and 35. Assume these are the full serialized costs for this example.

| Rule | What happens |
| --- | --- |
| RAW_STRICT | Stops at 120; exposes nothing |
| RAW_SKIP | Skips 120; includes 60 and 35; spends 95 |
| FIXED_SENTENCE | Can select fitting sentences from those passages; the actual result depends on sentence scores/costs |
| QBGF_FULL | Can choose a mix of sentence options and whole passages using its utility objective |

This shows why a gain over RAW_STRICT is partly about packing. It is not sufficient proof that one granularity is intrinsically better.

## 5. Specialist systems and adaptation labels

| Name | Meaning | Comparison caution |
| --- | --- | --- |
| SARA | Explicit passages combined with learned vectors for additional evidence | Adapter, projector, prompt and reader checkpoint all matter |
| `SARA_CANONICAL_BM25_K5_N10` | Canonical BM25/SARA specialist: k=5 explicit, n=10 total candidate units in its specified interface | Adapted Mistral specialist; not Qwen 256/1024 budget matched |
| `SARA_SFR_K5_N10` | Qualified SFR retrieval route with the SARA k5/n10 specialist interface | Changes retrieval within a specialist system; retains specialist comparison scope |
| Historical SFR-SARA k6 | Earlier Week10 specialist reference | Different historical cohort/settings; not a fresh final matched condition |
| `BM25_PLAIN_RAG` | Internal alias for budget-matched BM25 + RAW_SKIP | Report as BM25_RAG_RAW_SKIP_256/1024, not unrestricted RAG |
| `SFR_PLAIN_RAG` | Internal alias for budget-matched SFR + RAW_SKIP | Matches the final RAW_SKIP route when inputs/identities agree |
| `BASE_FP16` | Base reader in half precision without the final LoRA | Final matched validation/confirmation reader |
| `BASE_NF4` | Four-bit quantized base reader | Different precision arm; not interchangeable with FP16 |
| `LORA_NF4` | Quantized base with a qualified LoRA adapter | Isolate adaptation against its matched quantized base |
| LoRA | Low-rank adaptation of selected model weights | Training existence does not prove improved task quality |
| QLoRA | Low-rank adaptation with a quantized base | Memory technique, not an evidence-selection method |
| QCLA | Week8 conditioning/adaptation variants | Specific historical experiments, not the final Week12 reader |
| DAFiR | Historical draft-aware integrated-system label | Several surviving integrated outputs were smoke-only; a name is not full-run evidence |
| LLMLingua-2 | External token-level compression method | Final mBERT checkpoint was qualified, but full Week12 quality evaluation was skipped |
| RECOMP | External extractive/generative compression reference checkout | Bundled outputs do not establish a locally executed matched benchmark |

**Final LLMLingua qualification:** `microsoft/llmlingua-2-bert-base-multilingual-cased-meetingbank`. Do not call this a reproduction of the XLM-R-large checkpoint. Full final LoRA training/evaluation and LLMLingua quality comparisons were not completed.

## 6. Historical draft/intervention labels

### 6.1 Week9 A–G: a map of the labels

| Label | Meaning in Week9 |
| --- | --- |
| W9-A | Build/audit the frozen post-draft retrieval pool; final setup FIXED-5 / SEMANTIC / width 20 |
| `K_E=0` / `D0` | Keep the initial draft with no extra evidence intervention; this draft already came from an evidence-based system |
| `D1` | Raw forced intervention with the same two selected coarse chunks |
| `D2` | Query-only generative extraction from those chunks |
| `D3` | Draft-aware generative extraction from those chunks |
| `E0` | Deterministic query-only BM25 sentence/span packing; no extraction LM |
| `E1-Q` | LM-produced structured exact-support extraction, using the question |
| `E1-QD` | Corresponding structured extraction using question and draft as guidance |
| `E2` | Draft-aware minimal source-grounded blueprint with exact support spans and organizing labels |
| `E3-SentTopK` | Non-generative MiniLM query–sentence ranking over the same two chunks |
| `E3-k1/k2/k3/k4/k6/k8` | Fixed requested sentence-count surface; selected sentences restored to source order |
| `E3B-DynamicSent` | Possible later adaptive stopping direction; oracle headroom alone does not establish a trained working controller |
| `F0` | Existing E3-k3 answer interface, only reused when evidence/prompt/model identity matches |
| `F1` | Conservative minimal revision: preserve the draft unless the new evidence clearly requires change |
| `F2-A`, `F2-B` | Predefined stronger evidence-authoritative revision instructions |
| W9-G verifier | Learned choice between completed candidate actions and the baseline using inference-available features |

**Three important distinctions:**

- W9 D0/K=0 is **not** Week12 NONE: it keeps an evidence-based draft rather than starting with an evidence-free reader.
- Week8 E0 means the no-promotion controller reference. Week9 E0 means a specific deterministic extraction control. Always include the week.
- `QBGF_FULL` means all representation options are enabled. It does not mean unrestricted full-document context. `RUN_FULL` is an execution flag, not this method.

### 6.2 Selection order versus presentation order

**Selection order** is the priority used to choose units, usually highest relevance first. **Presentation order** is how chosen units appear to the reader. E3 and final methods restore selected evidence to original source order. This avoids treating score order as if it were the original narrative and separates what is selected from how it is shown.

### 6.3 Promotion, gating and fail-closed

- **Promotion:** expose additional or higher-fidelity evidence after the draft.
- **Always promote:** apply the selected intervention for every eligible query.
- **Gate / STOP:** decide whether to intervene at all.
- **Oracle gate:** use measured outcome quality to retain only helpful interventions. This is a retrospective diagnostic, not an implementable inference gate.
- **Fail-closed:** when an intermediate extraction is invalid or empty, retain the specified baseline rather than inject unsupported evidence.
- **Source validation failure:** altered/mismatched source identity is an error, not permission to silently substitute another passage.

Sources: [Week9 grounded policy definitions](../src/week9/grounded_e/prompts.py), [revision definitions](../src/week9/utilization_f/prompts.py), [audited historical reports](weekly_reports/ExperimentTimeline.csv).

## 7. Notation used in the formulas

| Symbol | Definition |
| --- | --- |
| `N` | Number of evaluated queries/claims in the stated cohort |
| `q` | One query or claim |
| `m`, `a` | A method or action |
| `s_q(m)` | Per-query score: answer F1 for QA; correctness for per-claim diagnostics |
| `Q(m)` | Dataset-level quality; mean answer F1 or aggregate class macro-F1 as appropriate |
| `b_q` | Score of the specified matched baseline for query q |
| `Δ_q` | Per-query score change, `s_q(m) − b_q` |
| `ε` | A registered tolerance; its meaning depends on the metric/experiment |
| `I[condition]` | Indicator: 1 if true, otherwise 0 |
| `T_e`, `T_p`, `T_g` | Evidence, full prompt and generated token counts |
| `A_q` | Available measured actions for a query |
| `G_q` | Mapped gold evidence annotations, under the stated matching rule |

Macro-F1 is nonlinear: a difference of dataset macro-F1 values is not the average of per-claim macro-F1 differences. Per-claim correctness can still define HELP/HURT and attribution diagnostics.

## 8. Answer-quality metrics

### 8.1 Token precision, recall and answer F1

Normalize prediction and reference using the task’s implemented scorer, then count tokens with multiplicity. Let `L_p` and `L_g` be predicted/reference token counts, and let `O` be their multiset overlap.

\[
P=\frac{O}{L_p},\qquad R=\frac{O}{L_g},\qquad
F1=\frac{2PR}{P+R}=\frac{2O}{L_p+L_g}.
\]

Use the implemented empty-answer and special-answer rules when denominators vanish. The score is zero for a nonempty wrong answer with no token overlap.

**Example:** reference “red blue green”; prediction “red blue.” Then `O=2`, `P=1`, `R=2/3`, and `F1=0.8`. It is partially correct, not an 80% probability of being right.

For multiple accepted references:

\[
F1_q=\max_{g\in\mathcal G_q}F1(\hat y_q,g),
\qquad \overline{F1}=\frac1N\sum_q F1_q.
\]

Task normalization matters. HotpotQA removes case, punctuation and articles and has special yes/no/noanswer rules. The QASPER scorer uses its own established path, including available stop-word handling and reformatted references. Do not rescore historical results using a new normalization and present it as the original metric.

### 8.2 Exact match, EM

\[
EM_q=\max_{g\in\mathcal G_q}I[\operatorname{norm}(\hat y_q)=\operatorname{norm}(g)],
\qquad EM=\frac1N\sum_q EM_q.
\]

EM rewards an exactly accepted normalized answer. The F1 example above has EM=0. A high F1 can coexist with modest EM.

### 8.3 Accuracy

\[
\operatorname{accuracy}=\frac{\text{correct predictions}}N.
\]

For SciFact, the prediction must equal the gold label. Invalid outputs count as incorrect, not as excluded examples.

### 8.4 Class precision, recall and F1

For one class `c`:

- `TP_c`: true c, predicted c;
- `FP_c`: true another class, predicted c;
- `FN_c`: true c, predicted another class or invalid output.

\[
P_c=\frac{TP_c}{TP_c+FP_c},\quad
R_c=\frac{TP_c}{TP_c+FN_c},\quad
F1_c=\frac{2TP_c}{2TP_c+FP_c+FN_c}.
\]

The final implementation assigns zero when the relevant F1 denominator is zero.

### 8.5 Three-class macro-F1

\[
\operatorname{MacroF1}=\frac{F1_{SUPPORT}+F1_{REFUTE}+F1_{NOT\_ENOUGH\_INFO}}3.
\]

If class F1 values are 0.8, 0.6 and 0.4, macro-F1 is 0.6. Each class receives equal weight regardless of its frequency. Report class support counts to show sample sizes. Do not average SciFact macro-F1 with QA answer F1 into an invented overall project score.

### 8.6 ROUGE-L

Conventionally, ROUGE-L uses longest common subsequence length `L` to calculate sequence-overlap precision/recall and their specified F measure. Unlike a substring, a subsequence preserves order but need not be contiguous.

**Project caution:** historical interfaces expose a `rouge_l` field, but some fallback paths return zero rather than computing meaningful ROUGE-L. It was not the final primary quality metric. A stored zero in such a path should not be interpreted as a measured absence of sequence overlap.

Sources: [QA scorer](../src/week6/evaluation.py), [task evaluation](../src/week11/evaluation.py), [final class aggregation](../src/week12/statistics.py).

## 9. Retrieval and evidence-coverage metrics

### 9.1 Any-gold and all-gold recall

For each query with evaluable gold annotations, define `C_qj(k)=1` if annotation j is fully covered by the retrieved top-k units under the experiment’s matching rule.

\[
\operatorname{AnyRecall@k}=\frac1{N_G}\sum_{q:G_q\ne\varnothing}
 I[\exists j:C_{qj}(k)=1],
\]

\[
\operatorname{AllRecall@k}=\frac1{N_G}\sum_{q:G_q\ne\varnothing}
 I[\forall j:C_{qj}(k)=1].
\]

`N_G` is the number of eligible queries with gold evidence. Later source-span metrics treat no-gold cases as unavailable/null. Do not silently call them retrieval failures or successes.

If two of three annotations are recovered, any-gold success is 1 and all-gold success is 0. These metrics answer whether evidence was found, not whether the reader answered correctly.

### 9.2 Partial evidence coverage

For a mapped annotated span, later evaluation computes covered non-whitespace source characters divided by all non-whitespace characters in that span. Selected overlaps are unioned so repeated coverage is not double counted. Repeated possible occurrences can be treated as alternatives according to the mapping.

\[
\operatorname{coverage}_q=\frac1{|G_q|}\sum_j
 \frac{\text{covered annotated characters}_{qj}}{\text{annotated characters}_{qj}}.
\]

Any/all success requires full annotation coverage, not merely a tiny overlap. Early lexical paragraph-matching metrics use different rules. Unmapped evidence annotations and alternative-location conventions must remain visible.

### 9.3 Evidence precision / density

A common unit-level definition is:

\[
\operatorname{precision}=\frac{\text{selected units judged relevant}}{\text{selected units}},
\qquad
\operatorname{recall}=\frac{\text{gold units recovered}}{\text{gold units}}.
\]

A token/character **density** instead measures the fraction of selected text overlapping mapped evidence. These are not interchangeable. The project used different unit/mapping definitions at different stages; a column called density or precision must be read with its artifact’s denominator. No single density formula is silently imposed on all weeks.

### 9.4 Reciprocal rank and MRR

If the first fully relevant unit is at rank `r_q`:

\[
RR_q=1/r_q,\qquad MRR=\frac1{N_G}\sum_q RR_q.
\]

Use zero when no relevant unit is found in the evaluated ranked list, and the specified null convention when gold is unavailable. Check whether the recorded first relevant rank was evaluated over the full stored list or a cutoff; do not assume every repeated MRR field means cutoff-specific MRR@k.

### 9.5 Duplicate rate and redundancy

For `n` selected units with `u` unique exact text/hash identities:

\[
\operatorname{duplicate\ rate}=1-u/n.
\]

In the Week11 retrieval diagnostic, redundancy averages pairwise word-set Jaccard similarity:

\[
J(A,B)=\frac{|A\cap B|}{|A\cup B|},\qquad
\operatorname{redundancy}=\operatorname{mean}_{i<j}J(A_i,A_j).
\]

Candidate diversity there is `1 − redundancy`. Week8 uses cosine-based features instead, described below. Exact duplicates, near duplicates and semantically redundant text are different concepts; near-duplicate rates need a specified threshold.

### 9.6 Source-support validation

\[
\operatorname{source\ support\ rate}=\frac{\text{selected units passing exact source-span validation}}{\text{selected units tested}}.
\]

Validation checks IDs, character offsets and source text/hashes. A rate of 1 means the evidence is source-exact under those checks. It does not mean the selected text entails the answer, is factually true, or is causally responsible for correctness. Empty evidence needs an explicit denominator convention rather than a fabricated perfect rate.

Source: [coverage and retrieval implementation](../src/week11/evaluation.py).

## 10. HELP, HURT and answer behavior

### 10.1 Score changes and rates

For a fixed matched baseline:

\[
\Delta_q=s_q(m)-b_q.
\]

For QA, the final behavioral convention uses `ε=0.01`; for per-claim classification it uses `ε=0`:

\[
HELP_q=I[\Delta_q>\epsilon],\quad
HURT_q=I[\Delta_q<-\epsilon],\quad
NEUTRAL_q=I[|\Delta_q|\le\epsilon].
\]

Each rate is the average of the corresponding indicator over the stated cohort. A QA gain exactly equal to 0.01 is neutral under this strict inequality. Classification HELP means incorrect→correct; HURT means correct→incorrect.

**Example:** of 100 interventions, 30 help, 20 hurt and 50 are neutral. HELP=0.30, HURT=0.20. This does not by itself guarantee a positive mean F1 change: losses can be larger than gains.

### 10.2 Conditional versus unconditional rates

Historical **help precision** commonly uses only promoted queries:

\[
\operatorname{help\ precision}=\frac{\#\text{helpful promoted queries}}{\#\text{promoted queries}}.
\]

A full-cohort HELP rate uses all queries in its denominator. **Promotion rate** is promoted queries divided by all queries. A model can have high conditional precision while helping very few examples overall.

For a descriptive intervention balance:

\[
\frac{HELP}{HELP+HURT}.
\]

This excludes neutral cases and is undefined when HELP+HURT=0. It is not a calibrated success probability or a substitute for mean quality.

### 10.3 Answer changed

\[
\operatorname{answer\ changed}=\frac1N\sum_q I[\hat y_q^{(m)}\ne\hat y_q^{(base)}].
\]

Use the implementation’s string/parser normalization convention. Changing wording can leave F1 unchanged; a textually unchanged answer can still be wrong. “Answer changed” is not equivalent to HELP.

### 10.4 Beneficial/harmful revision and utilization proxies

A beneficial revision requires an answer change and a positive score change beyond tolerance. A harmful revision requires an answer change and a negative score change beyond tolerance.

Historical lexical utilization proxies check whether newly introduced answer words occur in the evidence. They measure lexical behavior, not factual entailment or internal attention. An unchanged answer under gold evidence does not prove that evidence was ignored.

Sources: [final behavioral definitions](../src/week12/statistics.py), [Week9 evaluation](../src/week9/evaluation.py), [Week8 promotion metrics](../src/week8/evaluation.py).

## 11. Oracles, regret and headroom

### 11.1 Best fixed policy versus per-query oracle

A **fixed policy** uses one registered rule/action setting for the whole cohort. A **per-query oracle** chooses the best of the already evaluated actions separately for each query:

\[
s_q^*=\max_{a\in A_q}s_q(a),\qquad
Q_{oracle}=\frac1N\sum_q s_q^*.
\]

This equation is for an additive per-query score such as answer F1. For aggregate class macro-F1, apply the declared oracle objective and recompute the aggregate from selected predictions; do not silently replace it with mean correctness.

The oracle only covers measured actions. It is not a ceiling over every possible source combination or future model.

### 11.2 Exact and epsilon sentence-count oracles

With tested sentence counts, exact ties choose the smallest k:

\[
k_q^*=\min\{k:s_q(k)=\max_j s_q(j)\}.
\]

The epsilon oracle allows a small quality loss:

\[
k_{q,\epsilon}^*=\min\{k:s_q(k)\ge\max_j s_q(j)-\epsilon\}.
\]

For E3, `ε=0.01`. The **intervention-only oracle** chooses among tested E3 counts. The **full-action oracle** additionally includes k=0, using D0. These answer different questions: how to intervene versus whether to intervene at all.

An oracle can be correct retrospectively without being predictable. Varying oracle k values alone does not prove a dynamic controller is worth training.

### 11.3 Regret and near-oracle rate

For a chosen action `â_q`:

\[
\operatorname{regret}_q=s_q^*-s_q(\hat a_q),\qquad
\overline{\operatorname{regret}}=\frac1N\sum_q\operatorname{regret}_q.
\]

Some historical promotion routines explicitly clip negative comparisons to zero or define the oracle over promotion candidates rather than including STOP. Always check the action set.

\[
\operatorname{near\ oracle\ rate}=\frac1N\sum_q
 I[s_q^*-s_q(\hat a_q)\le\epsilon].
\]

**Tie-aware top-1 oracle accuracy** counts any candidate whose true utility ties the maximum under the numerical tolerance. It is not ordinary QA accuracy. Many tied neutral candidates can produce a high oracle-selection rate without a large answer-quality gain.

### 11.4 Selector headroom recovery, SHR

The generic form is:

\[
SHR=\frac{Q_{policy}-Q_{base}}{Q_{oracle}-Q_{base}}.
\]

If base=0.40, oracle=0.50 and policy=0.43, SHR=0.30: the policy recovers 30% of the measured oracle headroom. This is not 30% accuracy. A zero denominator requires an explicit undefined/fallback convention.

**Critical Week8 distinction:** some tables’ `final_downstream_f1` uses an oracle gate:

\[
Q_{oracle\ gated}=\frac1N\sum_q
 \left(b_q+\Delta_q I[\Delta_q>0.01]\right).
\]

It discards harmful/non-helpful interventions using observed outcomes. `unconditional_downstream_f1` instead includes the actual chosen interventions, including harm. SHR calculated from the gated value is an oracle-assisted diagnostic, not a deployable system result. Later closeout tables label these separately.

Source: [Week8 selector metrics](../src/week8/evaluation.py), [Week9 selector diagnostics](../src/week9/evaluation.py).

## 12. Evidence and computation costs

### 12.1 Token fields

| Field | Definition |
| --- | --- |
| Evidence tokens, `T_e` | Tokens of exposed source evidence under the registered serialization |
| Final prompt tokens | Entire input to the last answer call, including instructions/question/framing |
| Cumulative query-time prompt tokens | Sum of prompt-token counts over every conceptual LM stage for that query |
| Generated tokens, `T_g` | Tokens output by a call, using the backend’s recorded token count |
| Cumulative generated tokens | Sum across draft, selection/extraction and final answer calls where used |
| Conceptual LM calls | Calls required by the policy, even if compatible answers were cached |
| Physical LM calls | Actual new model invocations made during this execution |
| Preprocessing / indexing cost | Work preparing corpus units/embeddings before answering queries |
| Embedding lookup/computation latency | Actual recorded time retrieving or computing vector features |

For calls `j=1,…,L`:

\[
T_{prompt,cumulative}=\sum_j T_{prompt,j},\qquad
T_{generated,cumulative}=\sum_j T_{generated,j}.
\]

Candidate text stored in memory is not automatically an LM prompt token cost. Cached embeddings have original preparation/computation cost; a fast lookup should not be described as zero conceptual cost.

### 12.2 Budget utilization and unused budget

\[
\operatorname{utilization}_q=T_{e,q}/B,\qquad
\operatorname{unused}_q=B-T_{e,q}.
\]

Valid evidence must not exceed B. A 256-token budget using 192 evidence tokens has utilization 0.75 and 64 unused tokens. High utilization is not necessarily high quality: filling the budget with distractions can hurt.

The final allocator measures additive serialized token-ID blocks. Retokenizing a whole concatenated display string can differ because of token boundaries; use the registered cost definition.

### 12.3 Compression ratio: a name that needs care

Week12’s stored `compression_ratio` means **fraction removed** relative to the complete frozen top-10 pool:

\[
C_{removed}=1-\frac{T_{selected}}{T_{reference}}.
\]

Larger means more reduction. If 250 tokens are selected from a 1,000-token reference, the stored value is 0.75.

Other commonly used conventions are:

\[
C_{retained}=T_{selected}/T_{reference}=0.25,
\qquad C_{factor}=T_{reference}/T_{selected}=4.
\]

All three can describe the same selection. Always state the formula and reference. Historical E3 cost comparisons use their raw two-chunk reference; do not combine them with Week12 top-10 compression values as if denominators matched. A nonpositive reference makes the Week12 ratio unavailable/null.

### 12.4 Token reduction relative to a baseline

\[
\operatorname{token\ reduction\ percent}=100\left(1-\frac{\overline T_{policy}}{\overline T_{baseline}}\right).
\]

A ratio of cohort means is not generally the mean of per-query ratios. Identify which one the table reports. Historical inconsistent token constants are retained as audit issues, not silently averaged.

### 12.5 Mean, median and p90

- **Mean:** sum divided by number of observations; sensitive to long tails.
- **Median:** middle value; half the observations lie on each side under the standard convention.
- **p90:** 90th percentile; about 90% of values are at or below it.

For evidence length, these distinguish typical cost from expensive examples. A small mean alone can hide a long tail.

### 12.6 Latency and Pareto efficiency

**Answer-generation latency** measures one model stage. **End-to-end latency** measures the complete registered query path, including relevant retrieval, feature lookup, selection and generation stages. Generation-only timing cannot establish total serving latency or throughput.

A method is **Pareto-dominated** if another comparable method is at least as good in quality and no more costly, with a strict improvement in at least one dimension. Non-dominated points form the frontier. This geometric criterion ignores statistical uncertainty unless explicitly extended; a small noisy score advantage does not establish a reliable practical improvement.

Source: [cost/compression definitions](../src/week12/statistics.py), [recorded final row fields](../src/week12/runner.py).

## 13. Feature and calibration terms from earlier experiments

### 13.1 Query relevance, draft relevance, redundancy and MIG

For candidate passage `D_i`, query embedding `e_q`, draft embedding `e_y`, and currently explicit passage set `F`:

\[
R_q(i)=\cos(e_q,e_i),\qquad R_y(i)=\cos(e_y,e_i),
\]

\[
Red(i)=\max_{j\in F}\cos(e_i,e_j),
\]

\[
Diversity(i)=1-\operatorname{mean}_{j\ne i}\cos(e_i,e_j),
\]

\[
MIG_i=R_q(i)+\alpha R_y(i)-\beta Red(i).
\]

`α` and `β` use the phase’s registered/tuned values. MIG here is a handcrafted relevance/draft/redundancy score. Despite its name, it is **not a measured information-theoretic mutual-information gain**. Negative cosine can make `1 − cosine` exceed 1; these are not probabilities.

`a6_draft_sim` in the measured fallback duplicates draft semantic similarity. It is not an independently measured attention attribution. Missing A1–A5 attention measurements must not be inferred from its presence.

### 13.2 Teacher forcing, log likelihood and NLL

Teacher forcing evaluates a specified target sequence `y₁…y_T`, conditioning each token on the correct preceding target tokens:

\[
\log P(y\mid x)=\sum_{t=1}^{T}\log P(y_t\mid x,y_{<t}),
\]

\[
NLL_{token}=-\frac1T\sum_t\log P(y_t\mid x,y_{<t}).
\]

Lower NLL means the model assigns more probability to that target. It does not necessarily generate a better free-running answer. A mean of per-query mean NLLs differs from a token-weighted mean over the whole cohort; use the reported aggregation.

### 13.3 Masking attribution

For the existing draft target, the Week8 masking feature compares its likelihood with a latent slot present versus zeroed:

\[
\Delta\log P_i=\log P(y_0\mid x)-\log P(y_0\mid x\text{ with slot }i\text{ masked}).
\]

The per-token feature divides by the number of scored target tokens. Positive values mean that slot supports the model’s likelihood of the **existing draft** under this intervention. A confident wrong draft can also receive support. It is not promotion utility, correctness or generated-answer F1.

`masking_available` records whether the measurement exists. `masking_target_tokens` is the scored target length, which can be capped. An unavailable value is not an observed zero. Similarly, `target_mask_ll` controls which likelihood-supervision targets participate in training; a zero mask excludes the term.

### 13.4 Brier score

For predicted binary HELP probability `p_i` and observed label `y_i∈{0,1}`:

\[
Brier=\frac1N\sum_i(p_i-y_i)^2.
\]

Lower is better for this probabilistic scoring rule. Class imbalance can make a constant-probability baseline surprisingly strong, so report one when assessing predictive value. Brier is not answer F1.

### 13.5 Expected calibration error, ECE

Partition probabilities into bins `B_b`. Let `conf_b` be mean probability and `rate_b` the observed positive rate in a bin:

\[
ECE=\sum_b\frac{|B_b|}{N}|conf_b-rate_b|.
\]

Lower means closer calibration under that binning. If a bin predicts 0.10 HELP but HELP occurs 0.20 of the time, its gap is 0.10. “Decile” tables may use fixed 0.1-wide probability bins, not equal-count groups. Calibration is different from discrimination and does not prove a useful downstream gate.

Sources: [MIG features](../src/week8/marginal_information.py), [masking attribution](../src/week8/attribution.py), [calibration implementation](../src/week8/evaluation.py).

## 14. Evidence-attribution metrics from J/K/L

### 14.1 Leave-one-unit-out importance, LOO

Let `E_q` be the complete selected evidence and `u` one selected unit:

\[
I_{qu}=s_q(E_q)-s_q(E_q\setminus\{u\}).
\]

- Positive: deleting u lowered the score; helpful under this intervention.
- Negative: deleting u improved the score; harmful under this intervention.
- Zero: no observed score change.

The attribution diagnostic uses zero tolerance. For SciFact, `s_q` here is per-claim correctness, not per-claim macro-F1. These signs therefore answer a particular input-intervention question.

### 14.2 Sufficiency, ONLY and REMOVE

- **ONLY u:** supply only u; test whether that unit is sufficient for a good answer under this reader.
- **REMOVE u:** delete u from the selected set; test its marginal contribution with the rest still available.
- **Oracle/“causal” top-k:** rank using observed deletion importance, then keep/remove a retrospective top-k set.

A unit may be sufficient alone but unnecessary in a redundant collection. Two units can jointly matter even when deleting either alone has no effect. Importance is not generally additive, and source-exact ablation does not reveal the model’s internal reasoning.

### 14.3 Relevance–utility Spearman correlation

Spearman rho is the Pearson correlation of tied ranks:

\[
\rho_s=\operatorname{Corr}(\operatorname{rank}(relevance),
\operatorname{rank}(importance)).
\]

It ranges from −1 to +1 when defined. Constant arrays make it undefined; many zero-importance ties can make interpretation weak. A positive point estimate with a wide interval crossing zero is not strong evidence of reliable alignment.

**Strict top-1 agreement** asks whether the top relevance unit matches the designated most-important unit. **Tie-aware agreement** allows any maximum-importance unit. Many neutral ties can inflate the latter without identifying a uniquely helpful unit.

### 14.4 Retention ratios

A relative gain-retention diagnostic can compare a reduced set R with full evidence E and a no-evidence baseline:

\[
retention(R)=\frac{s_q(R)-s_q(NONE)}{s_q(E)-s_q(NONE)}.
\]

The denominator can be zero or negative; do not replace undefined values with invented zeros. Interpret the artifact’s exact eligibility convention. The project’s finite-only correction reports small eligible sample counts, so those curves cannot support an all-query retention claim. Ratios may exceed 1 or be negative and are not probabilities.

Source: [completed attribution report and caveats](week12/W12_K_L_ATTRIBUTION_REPORT.md).

## 15. Statistics and scientific interpretation

### 15.1 Paired difference

\[
\Delta_{A-B}=Q(A)-Q(B).
\]

Positive favors the left method. For mean QA F1 on matched examples this equals the mean per-query difference. For SciFact macro-F1, compute each cohort-level macro-F1 first and subtract.

Always read the sign convention. “Sentence minus passage = −0.02” favors passages by point estimate. It is not a 2% relative loss unless that relative denominator is explicitly calculated.

### 15.2 Paired cluster bootstrap confidence interval

1. Identify frozen groups: paper, query group or connected claim/document group.
2. Sample groups with replacement, preserving each selected group’s examples and both systems’ paired outcomes.
3. Recompute both quality scores and their difference for each resample.
4. Use the 2.5th and 97.5th percentiles for the nominal 95% interval.

Final confirmation uses 10,000 resamples, seed 42. For macro-F1 the class counts are recomputed within resamples; one must not bootstrap invented independent per-query macro-F1 values.

An interval including zero means the direction is unresolved at that nominal level. It does not prove zero effect, equivalence or noninferiority. A frequentist 95% interval is not literally a 95% posterior probability assigned to this one fixed effect.

### 15.3 Paired cluster label-swap p-value

Under the registered exchangeability null, randomly swap system labels within whole clusters and calculate a null difference. With R draws:

\[
p=\frac{1+\#\{|\Delta_{null}|\ge|\Delta_{observed}|\}}{R+1}.
\]

The +1 avoids a zero p-value from finite simulation. The final registered test uses R=10,000. A p-value is not the probability the hypothesis is true or the effect is practically important.

### 15.4 Multiple comparisons: Holm and FDR

Testing many hypotheses creates more opportunities for a chance finding. For sorted p-values `p_(1)≤…≤p_(m)`, Holm adjusted values are:

\[
p^{adj}_{(i)}=\min\left(1,\max_{j\le i}(m-j+1)p_{(j)}\right).
\]

Final confirmation has a predefined family of eight primary and a separate family of four secondary contrasts. Do not invent a new family after seeing the scores. Bootstrap CIs are nominal, while Holm p-values control their registered families; they need not yield identical decision thresholds.

Earlier **FDR** procedures control expected false-discovery proportion among rejected tests under their assumptions. FDR q=0.10 does not mean every retained feature has a 90% chance of being real. Those exploratory tests are separate from final confirmation’s Holm families.

### 15.5 Noninferiority and equivalence

For candidate A and reference B, noninferiority margin δ permits an acceptable loss:

\[
\Delta_{A-B}>-\delta.
\]

The project gate uses the specified interval criterion: the lower bound must clear `−δ`, plus other required conditions. QA and macro-F1 margins differ. Failure to establish noninferiority is not proof of inferiority. Equivalence would require a separate two-sided acceptable interval; “not significantly different” does not establish it.

### 15.6 Interaction and heterogeneity

An interaction asks whether one effect changes across another factor. A budget interaction can take the form:

\[
[Q(A,1024)-Q(B,1024)]-[Q(A,256)-Q(B,256)].
\]

A nonzero point estimate is suggestive; assess its own paired uncertainty. Different rankings or one significant cell and one nonsignificant cell do not automatically establish a significant interaction. **Heterogeneity** simply means effects/actions vary across examples or settings; it does not prove that the variation is predictable at inference time.

Sources: [registered statistics implementation](../src/week12/statistics.py), [final confirmation contrasts](week12/final_confirmation/v1/contrasts.csv).

## 16. Split, execution and provenance vocabulary

| Term | Meaning and boundary |
| --- | --- |
| TRAIN | Eligible examples used to fit parameters; eligible membership is not proof each was actually seen |
| FIT | A development fitting cohort used in particular historical stages |
| INTERNAL-VAL | Held-out internal development cohort; separate from final untouched confirmation |
| DEV / VAL | Development/validation data used for comparisons and selection; repeated use limits independence |
| TEST | A held-out evaluation designation; historical QASPER TEST was consumed and remains closed |
| Confirmation | Reserved cohort evaluated after the final decision; Week12 M, not a new tuning phase |
| Query/paper/group disjoint | No membership overlap under the specified grouping rule; does not automatically mean all background documents are distinct |
| OOF | Out-of-fold predictions: each prediction is made by a model not fitted on that example’s fold |
| Nested OOF | Model selection/threshold selection is kept inside training folds; avoids using the evaluated fold to choose its own controller |
| Leakage | Evaluation labels or future/outcome information contaminates fitting, selection or inference |
| Preregistered / predefined | A decision/hypothesis family fixed before the relevant outcomes are inspected |
| Post-hoc | Analysis chosen after earlier results exist; explanatory/exploratory, not fresh confirmation |
| Primary / secondary | Prespecified scientific importance; not labels assigned to favorable scores afterward |
| Full run | All intended examples/conditions in the specified grid, not necessarily every optional project experiment |
| Logical row | One query × method × budget condition in the intended result grid |
| Physical generation | One newly executed model call; compatible logical rows can share it |
| Resume | Validate identities and reuse completed compatible work; not blindly accept any file with the right name |
| Overwrite protection | Refuse replacement of an existing incompatible result |
| Hash / SHA256 | Fingerprint of bytes or a registered identity payload | 
| Freeze artifact | Saved model/method/split/prompt/registry decision used to protect a later evaluation |
| Immutable | Preserve existing compatible scientific artifacts; changed experiments need a new identity/version |
| Source lineage | Trace from a result back to query, document, selected chunks/spans and frozen configuration |
| Smoke / synthetic fixture | Small plumbing test, often artificial; not real full experiment evidence |
| Qualified model | Passed specified loading/access/tiny execution checks | 
| Missing / pending / not run | No established result; must not be assigned a zero quality score |
| CORE_FINAL | Execution restriction to the registered primary grid; not a scientific method |
| `RUN_FULL` | Permission flag for an expensive/full execution; semantics depend on that notebook |

A matching SHA verifies the specified bytes/identity, not scientific validity by itself. A freeze can faithfully preserve a flawed design; auditing and statistical interpretation remain necessary.

The final confirmation has 2,500 logical rows but 2,213 distinct cached generations. Those counts describe different things; neither implies missing rows. Both early QASPER TEST and the final reserved confirmation have now been observed and cannot be called untouched again.

## 17. Common misreadings to avoid

| Misreading | Correct interpretation |
| --- | --- |
| FIXED_SENTENCE always uses the same number of sentences | Its rule is fixed; the number depends on lengths and budget |
| RAW_SKIP means unrestricted full-context RAG | It is whole-passage RAG constrained by a token limit |
| QBGF_FULL reads the whole corpus | It enables the full option set under budget |
| NO_QUERY removes the question from the entire pipeline | The SFR candidate pool remains question-dependent |
| High cosine proves support | It measures embedding alignment |
| Source-exact means factually faithful and answer-correct | It verifies source matching, not truth or utility |
| High recall means the answer must be correct | Retrieval and reader utilization are distinct |
| High oracle accuracy means high QA accuracy | Oracle-choice agreement and final answer scores differ |
| Oracle-gated F1 is deployable F1 | It uses unavailable outcome knowledge to filter interventions |
| Many neutral deletions mean evidence was ignored | Redundancy, interactions and metric insensitivity can hide effects |
| Smaller token cost proves faster serving | End-to-end timings must actually be measured |
| Cache reuse makes a method zero-cost | It saves repeated execution; conceptual prompts/calls remain |
| A nonsignificant difference proves equivalence | Uncertainty may be too large to resolve a difference |
| QBGF wins one cell, so it becomes the new final method | Frozen selection is not changed after confirmation |
| A qualified compressor has proven quality | Qualification and full benchmarking are separate |
| Week13 is another completed confirmation | The historical wrapper label refers to the Week12 M closeout; count it once |

## 18. A practical checklist for reading any result table

1. Identify **dataset, split and sample size**. Is this a pilot, VAL, historical TEST or reserved confirmation?
2. Identify **reader, checkpoint and precision**. Are the systems actually matched?
3. Identify **retrieval pool, evidence rule and token budget**. Does “plain RAG” really mean RAW_SKIP?
4. Identify **metric and direction**. Is it answer F1, macro-F1, accuracy, likelihood or a cost?
5. Identify **reference and denominator**. All queries, promoted queries, mapped-gold cases or selected units?
6. Separate **measured policy quality from oracle-assisted ceilings**.
7. Read **paired uncertainty, correction family and practical effect size**, not just the highest printed score.
8. Check **cost aggregation and cache accounting**.
9. Check **completion status and caveats** before treating a stored field as a full result.
10. Keep the claim within the tested scope; never infer universal superiority from one table.

## 19. Where to go next

- [Comprehensive project report](FINAL_PROJECT_REPORT.md): the research journey, full final tables and conclusions.
- [Audited week-by-week timeline](weekly_reports/ExperimentTimeline.csv): experiment/status map.
- [Consistency audit](weekly_reports/ProjectWideConsistencyAudit.csv): known conflicts and interpretation limits.
- [Final method freeze](week12/W13_FINAL_METHOD_FREEZE.json): authoritative final identities and decision.
- [Final confirmation report](week12/final_confirmation/v1/week12_M_final_confirmation_report.md): strongest final held-out results.

**Document boundary:** this glossary explains existing definitions. It does not change a method, scorer, split, result, freeze or experimental decision. No model training or inference was run to create it.
