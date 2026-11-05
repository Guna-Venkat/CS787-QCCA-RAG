# Evidence Lab — complete application and method guide

**CS787 · Evidence Allocation Study**

**Documented implementation: 3 November 2026**

This guide explains the application as implemented, especially what each **method** does and what the **budget** controls. It is intended for readers who have not worked on RAG before. It documents the software and how to interpret it; it does not introduce new experimental results.

## Contents

1. [What the application demonstrates](#1-what-the-application-demonstrates)
2. [The four workspaces](#2-the-four-workspaces)
3. [What the budget actually means](#3-what-the-budget-actually-means)
4. [Method availability](#4-method-availability)
5. [NONE — no external evidence](#5-none--no-external-evidence)
6. [RAW_SKIP — include whole passages that fit](#6-raw_skip--include-whole-passages-that-fit)
7. [FIXED_SENTENCE — select fitting query-relevant sentences](#7-fixed_sentence--select-fitting-query-relevant-sentences)
8. [C2 — top-eight sentence control](#8-c2--top-eight-sentence-control)
9. [H3 — fixed hierarchical policy](#9-h3--fixed-hierarchical-policy)
10. [QBGF_FULL — jointly allocate representation fidelity](#10-qbgf_full--jointly-allocate-representation-fidelity)
11. [Fine-grained source spans — punctuation-based selection](#11-fine-grained-source-spans--punctuation-based-selection)
12. [MULTI_SENTENCE_2 and MULTI_SENTENCE_4](#12-multi_sentence_2-and-multi_sentence_4)
13. [How the methods differ](#13-how-the-methods-differ)
14. [Curated Demo: every panel explained](#14-curated-demo-every-panel-explained)
15. [LiveDemo: inputs, stages, results and comparisons](#15-livedemo-inputs-stages-results-and-comparisons)
16. [CPU/GPU execution and model lifecycle](#16-cpugpu-execution-and-model-lifecycle)
17. [Metrics, relevance and deletion utility](#17-metrics-relevance-and-deletion-utility)
18. [Research Findings](#18-research-findings)
19. [Research Journey and the frozen decision](#19-research-journey-and-the-frozen-decision)
20. [Presentation, provenance, storage and safety](#20-presentation-provenance-storage-and-safety)
21. [Launch instructions and walkthroughs](#21-launch-instructions-and-walkthroughs)
22. [Frequently misunderstood details](#22-frequently-misunderstood-details)
23. [Implementation map](#23-implementation-map)

> **LiveDemo segmentation audit update (3 November 2026):** Custom-input LiveDemo now exposes two explicit segmentation policies. The default citation/Markdown-aware mode delegates to the frozen academic splitter through a source-offset-preserving boundary adapter. Exact raw frozen behavior remains selectable. The adapter fixes cited-sentence merges, but is a labelled live-only preprocessing variant, not a replacement for frozen research segmentation. Headings and fragments are displayed as source segments. See [the complete World Cup input audit](audit/live_sentence_segmentation_audit.md) for every raw passage, before/after unit, token count and source identity.

> **Correctness audit update:** Frozen research results retain their original preprocessing exactly. LiveDemo custom text can use a citation/Markdown-aware adapter because arbitrary pasted web text differs from the datasets' original preprocessed inputs. Therefore LiveDemo demonstrates the algorithms and evidence-flow concepts, but custom runs are not exact reproductions of benchmark preprocessing unless Frozen research splitter is explicitly selected. Even then, custom source corpora differ from the original datasets.
>
> **Retriever protocol:** Pipeline-native mode is available only in Researcher Mode and can change both candidate retrieval and downstream relevance scoring. Controlled retriever-only mode, the presentation default, changes candidate retrieval and uses the same pinned SFR query–passage/query–unit scorer for both pools. New BM25-native QBGF execution is blocked because raw lexical scores have an incompatible softmax scale. Saved curated lexical QBGF variants are unchanged and labelled accordingly. Read [FINAL_DEMO_CORRECTNESS_AUDIT.md](audit/FINAL_DEMO_CORRECTNESS_AUDIT.md).

## 1. What the application demonstrates

A language model can receive a question together with external evidence. This is the basic idea of **retrieval-augmented generation**, or RAG.

The application separates the operations that are often hidden inside one “ask a question” button:

```text
Question
  → retrieve candidate passages
  → construct possible evidence units
  → choose what fits the evidence budget
  → assemble the reader's input
  → generate an answer
  → inspect observed effects where reference data exist
```

The **reader** is the language model that generates the answer. The **method** is the policy that decides which evidence reaches that reader. Changing the method usually changes the input context, not the reader's weights.

For example, given the same question and candidate passages:

- RAW_SKIP may expose two complete passages.
- FIXED_SENTENCE may expose six sentences from several passages.
- QBGF_FULL may expose one complete passage, two sentences from another, and nothing from the rest.
- NONE exposes no external evidence.

These are alternative evidence policies. They are not four separately trained answer models.

The application's central question is: **Which pieces of available evidence should be exposed, and at what level of detail, when input space is limited?**

## 2. The four workspaces

| Workspace | Purpose | What happens when you interact? | Gold-reference evaluation? |
|---|---|---|---|
| **Curated Demo** | Inspect complete saved examples | Loads saved sources, selections, answers and deletion experiments | Yes, for the saved examples |
| **LiveDemo** | Run the evidence pipeline on your own passages and query | Performs actual operations; optionally generates an answer | No; custom questions have no supplied gold-reference evaluator |
| **Research Findings** | Inspect the study's audited aggregate results | Reads existing research tables and plots | Uses the original task-specific research metrics |
| **Research Journey** | Understand how the project reached its final decision | Presents the research progression and saved reference results | Explains existing results; does not evaluate a new run |

**Curated Demo and Research Findings have different roles.** The former is an illustrative collection of saved examples; the latter displays the actual study results. The curated collection does not replace the validation or confirmation tables.

Researcher Mode also exposes a **Historical exports** replay collection. This older view depends on whatever historical traces survived. Missing historical traces are not filled with invented selections or answers. Its limitations do not describe the newer complete curated collection.

## 3. What the budget actually means

### 3.1 Direct answer

**The 256 / 1024 selector is an input-evidence token budget. It is not an output-answer token budget.**

It caps the number of tokenizer IDs used by the selected, serialized evidence blocks supplied to Qwen.

Let:

- \(B\): selected evidence budget, either 256 or 1024;
- \(E\): actual selected evidence tokens;
- \(P\): complete reader prompt tokens;
- \(G\): newly generated answer tokens.

The allocation constraint is:

\[
E\leq B.
\]

The full prompt also contains the question, instructions and chat framing. Consequently:

\[
P = E + \text{question/instruction/chat-framing token contribution}.
\]

The input is assembled from exact token-ID blocks, so the application measures the actual prompt IDs. A word count or re-tokenized display string is not a substitute for that measurement.

**A 256-token evidence budget can therefore produce a prompt longer than 256 tokens.** That is expected.

### 3.2 What counts toward this budget?

| Item | Counts toward the selected evidence budget? |
|---|---|
| Included source passage/sentence/span text | Yes |
| Evidence-block separators and serialized spacing | Yes, as tokenized in the evidence blocks |
| Retrieved candidates that are not selected | No |
| The question | No; it still counts toward the full prompt |
| Reader instructions and chat template | No; they still count toward the full prompt |
| The generated answer | No; it is measured separately |
| Embedding inputs processed by SFR | No; they are a separate computation |
| A gold answer shown for evaluation | No; it is not evidence supplied for inference |

The evidence tokenizer is the pinned **Qwen tokenizer**, even when another model supplied retrieval embeddings.

### 3.3 Why the meter may not reach 100%

The budget is a ceiling, not a requirement to fill every token. Whole units can fail to fit. C2 and H3 stop at a non-fitting unit. QBGF's best feasible utility can be achieved below the limit. NONE always uses zero evidence tokens.

The meter shows:

\[
\text{utilization}=100\frac{E}{B}\%.
\]

Increasing the budget allows more evidence; it does not guarantee a better answer. Extra context can help, distract or have no measured effect.

### 3.4 Separate output and context limits

The pinned reader configuration has separate limits:

| Setting | Current value | Meaning |
|---|---:|---|
| Evidence budget | 256 or 1024 | Selected external evidence only |
| QA answer cap | 128 new tokens | Maximum new tokens for question answering |
| Verification answer cap | 16 new tokens | Maximum new tokens for a classification label |
| Reader context limit | 8192 tokens | Checks complete prompt plus allowed answer cap |

Answers may terminate before the cap. The reader checks that prompt length plus the configured generation cap fits the context limit; it does not silently truncate an overflowing input.

Curated SciFact examples use the verification task. **Custom LiveDemo queries currently use free-form QA**, even if the user enters a scientific claim, so they use the QA instruction and cap rather than automatically becoming a SciFact classifier.

### 3.5 Serialization detail

Selected units are restored to original source order and grouped by their parent passage. Each passage block is tokenized with its separators; the reader receives the concatenated evidence token IDs.

This makes the budget additive across passage options, which is necessary for QBGF's optimizer. The visible evidence text presents the selected source strings. Its rendered whitespace is not the authoritative token count; the stored token IDs are.

## 4. Method availability

There are **nine saved evidence methods in the complete Curated Demo** and **four in LiveDemo**. Presentation Mode hides the saved BM25-native QBGF combination; Researcher Mode retains it as a historical native lexical variant.

| Visible curated name | Internal ID | Curated | Live | Main decision |
|---|---|:---:|:---:|---|
| No evidence | `NONE` | Yes | Yes | Supply no external evidence |
| Whole passages | `RAW_SKIP` | Yes | Yes | Keep complete passages that fit |
| Query-selected sentences | `FIXED_SENTENCE` | Yes | Yes | Globally rank sentences and keep fitting ones |
| Top-eight sentences | `C2` | Yes | No | Consider only the first eight ranked sentences, using prefix packing |
| Fixed hierarchy | `H3` | Yes | No | Prioritize sentences from the first three passages, then fine spans from remaining passages |
| Adaptive QBGF | `QBGF_FULL` | Yes | Yes | Optimize one representation choice per passage jointly |
| Fine-grained source spans | `PROPOSITION` | Yes | No | Globally rank deterministic source spans and keep fitting ones |
| Two sentences per passage | `MULTI_SENTENCE_2` | Yes | No | Try each passage's best two sentences as a group |
| Four sentences per passage | `MULTI_SENTENCE_4` | Yes | No | Try each passage's best four sentences as a group |

LiveDemo uses more explanatory labels for the same four IDs: **No Evidence**, **Whole passages · RAW_SKIP**, **Query-selected sentences**, and **Adaptive fidelity · QBGF**.

The three additional curated representation policies—`PROPOSITION`, `MULTI_SENTENCE_2`, and `MULTI_SENTENCE_4`—were added for the demo. Their saved illustrative results must not be described as additional frozen confirmation arms.

## 5. NONE — no external evidence

**Meaning:** let the reader answer from the question and its pretrained knowledge.

**Procedure:**

1. Construct the question/instruction prompt.
2. Use an empty evidence block.
3. Generate the answer with the same reader.

There is no evidence retrieval or allocation in a live NONE run. The operational view marks retrieval and representation as skipped and shows **0 evidence tokens**.

NONE still has prompt tokens, generated tokens and one conceptual reader call. Zero evidence does not mean zero cost or zero model computation.

**Why it exists:** it tests whether supplying evidence changes performance relative to answering without it. A plausible NONE answer is not proof that the model retrieved or verified anything.

## 6. RAW_SKIP — include whole passages that fit

**Meaning:** preserve complete passages, processing them in retrieval order.

**Procedure:**

1. Start with an empty context.
2. Consider the highest-ranked remaining passage.
3. Calculate the exact serialized cost of including it with the current selection.
4. Include it if the total remains within the budget.
5. Otherwise skip it and continue to later passages.
6. Present the chosen passages in source order.

For an illustrative 256-token allocation, suppose candidate block costs are 180, 310 and 62 tokens. RAW_SKIP can include the first, skip the second, and include the third. Actual decisions use the serializer's measured combined costs rather than these illustrative numbers.

**What it preserves:** complete passage wording and local context.

**What it can lose:** a highly relevant long passage can be completely excluded because it does not fit.

**Important:** RAW_SKIP is **budget-matched passage RAG**, not unrestricted full-context RAG. It does not crop an oversized passage to the remaining token allowance.

The live packing trace reports included/skipped decisions and the remaining budget after each decision.

## 7. FIXED_SENTENCE — select fitting query-relevant sentences

**Meaning:** distribute the available evidence space across individual source sentences rather than committing to complete passages.

**Procedure:**

1. Take sentences from the retrieved candidate pool.
2. Score each sentence against the query.
3. Rank all candidate sentences globally by descending relevance.
4. Break equal-score ties deterministically by source order and stable identity.
5. Add a sentence if the exact combined evidence cost fits.
6. Skip a non-fitting sentence and continue looking at later sentences.
7. Restore the chosen sentences to original source order before presentation.

**“Fixed” means the selection policy is fixed. It does not mean a fixed number of sentences.** A run might select three long sentences or ten short ones.

Ranking decides **what** to expose; source order decides **how** the chosen text is presented. There is no paraphrasing or extraction LLM.

In a pipeline-native BM25 demonstration, lexical BM25 relevance is used. Native SFR uses cosine. In controlled retriever-only mode, both candidate pools use the same pinned SFR cosine scorer. The packing rule remains the same; native pipeline switching is not a retrieval-only experiment.

**Strength:** can spread a small budget over useful material from several sources.

**Limitation:** isolated sentences can lose context, and high similarity does not ensure that a sentence contains the answer.

This is the **frozen final evidence policy** in the main research pipeline, used with SFR top-10 retrieval and the base FP16 Qwen reader, without LoRA.

## 8. C2 — top-eight sentence control

**Meaning:** use a capped shortlist of query-ranked sentences with the historical prefix allocator.

**Procedure:**

1. Globally rank sentences by query relevance.
2. Keep only the first eight as eligible candidates.
3. Process this list in ranking order.
4. Include sentences while the combined context fits.
5. **Stop at the first sentence that does not fit.**

There are two differences from FIXED_SENTENCE:

| Property | C2 | FIXED_SENTENCE |
|---|---|---|
| Eligible ranked sentences | At most eight | All sentences in the candidate pool |
| Encounter a non-fitting sentence | Stop | Skip it and continue |

C2 does not guarantee eight selected sentences. It can select fewer or none. Increasing the budget cannot overcome the eight-candidate ceiling.

**Why it exists:** it is a simple, restricted sentence-selection control. Its behavior should not be attributed solely to sentence representation: its shortlist cap and prefix packing also matter.

## 9. H3 — fixed hierarchical policy

**Meaning:** apply a fixed rule that gives different representation treatment to early and later retrieved passages.

**Procedure:**

1. Mark the first **three retrieved passages** as the head group.
2. Collect their sentences and rank those sentences by query relevance.
3. From the remaining passages, collect the deterministic fine-grained source spans and rank those by query relevance.
4. Concatenate the sentence list first, then the fine-span list.
5. Process this combined list with the historical **prefix allocator**: stop at the first non-fitting unit.
6. Present the selected units in source order.

The number three refers to the first three passages receiving sentence treatment. It does not mean three layers of learned reasoning or three guaranteed selected units.

H3 is not QBGF. It does not jointly choose full versus partial passage representations. Its sequence is fixed, although the rankings within its two groups are query-dependent.

**Important consequence:** high-scoring fine spans from later passages do not jump ahead of the head-group sentence list. If prefix packing stops early, they may never be considered.

## 10. QBGF_FULL — jointly allocate representation fidelity

**Meaning:** **query-conditioned budgeted granularity fusion**. For each retrieved passage, choose how much of it to expose, under one shared token constraint.

### 10.1 Available choices per passage

| Option | Meaning |
|---|---|
| `DROP` | Expose none of this passage |
| `SENT_1` | Expose its best one query-ranked sentence |
| `SENT_2` | Expose its best two sentences |
| `SENT_4` | Expose its best four sentences |
| `SENT_8` | Expose its best eight sentences |
| `FULL` | Expose the complete passage |

If fewer sentences exist, an option uses the available sentences. Equivalent sentence subsets are not duplicated in the option list. Chosen sentences are restored to source order.

`QBGF_FULL` means the **full version of the policy**, including sentence and complete-passage choices. It does not mean “always choose full passages.”

### 10.2 How relevance becomes an optimization objective

Let \(r_i\) be passage \(i\)'s relevance score. The code transforms passage scores into normalized weights:

\[
R_i=\frac{e^{5r_i}}{\sum_j e^{5r_j}}.
\]

For sentence \(j\) inside passage \(i\), relevance \(s_{ij}\) becomes:

\[
w_{ij}=\frac{e^{5s_{ij}}}{\sum_\ell e^{5s_{i\ell}}}.
\]

For a selected sentence subset \(A_i\), its retained sentence mass is:

\[
m_i(A_i)=\sum_{j\in A_i}w_{ij}.
\]

The utility assigned to a sentence option is \(R_i m_i(A_i)\). FULL receives utility \(R_i\); DROP receives zero. These are relevance-based surrogate values, **not measured answer improvements or probabilities of correctness**.

### 10.3 Global selection

Each option has an exact serialized token cost. The dynamic program solves:

\[
\max_{o_1,\ldots,o_n}\sum_i U_i(o_i)
\quad\text{subject to}\quad
\sum_i C_i(o_i)\leq B,
\]

with exactly one option chosen for every passage, including DROP.

This is a multiple-choice knapsack allocation. It may decide:

```text
P1 → SENT_2
P2 → DROP
P3 → FULL
P4 → SENT_1
```

The choices are made jointly; this is more than passage reranking. A long passage can receive a compact representation so another passage can retain more detail.

The implementation uses deterministic tie rules. Within a fixed-cost state, equal utility favors the lexicographically smaller option path. Final selection uses the code's rounded-utility ordering, then lower cost and option path. The solver is exact for its defined cost/utility states; it does not optimize downstream answer F1.

**No gold answer, answer score or deletion outcome enters this allocation. No controller is trained during a demo run.**

**Limit:** optimizing relevance mass need not optimize evidence usefulness. The research did not establish a universal QBGF advantage over simpler policies.

## 11. Fine-grained source spans — punctuation-based selection

**Visible name:** Fine-grained source spans.

**Internal compatibility ID:** `PROPOSITION`.

Description: **Deterministic punctuation-based subdivisions of exact source sentences.** The visible name avoids claims of semantic proposition extraction or grammatical clause parsing. The internal compatibility ID does not imply independently complete facts.

**How spans are constructed:**

1. Begin with a source sentence.
2. Apply the deterministic `PROPOSITION_RULE_V1` punctuation rule.
3. Split at semicolons and spaced em/en dashes.
4. Trim boundary whitespace while retaining exact source offsets.
5. Keep spans containing word characters.

The rule does not synthesize missing subjects, resolve pronouns or rewrite clauses. If no supported boundary exists, the fine unit may be the whole sentence. These are punctuation-based subdivisions; they are not grammatical clause parsing.

**How the method selects:** globally rank these spans by query relevance; try them one at a time; include fitting spans and skip non-fitting ones; restore source order.

**Strength:** smaller contiguous units can distribute the budget more finely.

**Limit:** removing surrounding text can remove qualifications or context. Exact source matching establishes textual origin, not truth, semantic completeness or answer correctness.

This is a **demo-only representation policy**, not an additional confirmed research method. The internal ID is retained so saved results remain compatible.

## 12. MULTI_SENTENCE_2 and MULTI_SENTENCE_4

**Meaning:** represent each candidate passage with up to two or four of its most query-relevant sentences.

**Procedure:**

1. Visit passages in retrieval order.
2. Rank the sentences within the current passage.
3. Choose its best two or best four, depending on the method.
4. Treat that selected set as one candidate group.
5. Include the **whole group** if it fits with the evidence already selected.
6. Otherwise skip the group and continue to the next passage.
7. Present retained text in source order.

If a passage has fewer than the requested number, use its available sentences.

Unlike FIXED_SENTENCE, selection is not one global sentence list. Unlike QBGF, the number of requested sentences per passage is preset and there is no global choice between fidelity options.

**Important:** if a four-sentence group does not fit, `MULTI_SENTENCE_4` does not automatically fall back to two or one sentence from that passage. It skips the group.

These are demo-only representation policies. They help show how fixed within-passage compression differs from global sentence selection and adaptive allocation.

## 13. How the methods differ

| Method | Ranking/decision unit | If the next candidate does not fit | Complete passages possible? | Learned policy or extraction LLM? |
|---|---|---|---|---|
| NONE | None | Not applicable | No | No |
| RAW_SKIP | Passage, retrieval order | Skip and continue | Yes | No |
| FIXED_SENTENCE | Globally ranked sentence | Skip and continue | Not as a FULL option | No |
| C2 | First eight ranked sentences | Stop | No FULL option | No |
| H3 | Head sentences, then tail fine spans | Stop | No FULL option | No |
| QBGF_FULL | One fidelity option per passage | Joint optimization finds a feasible combination | Yes | No |
| PROPOSITION | Globally ranked exact fine span | Skip and continue | No FULL option | No |
| MULTI_SENTENCE_2 | Passage's top-two sentence group | Skip whole group | No FULL option | No |
| MULTI_SENTENCE_4 | Passage's top-four sentence group | Skip whole group | No FULL option | No |

Selecting every sentence can incidentally expose most or all of a short passage. That does not make the policy a full-passage allocator.

All these selection rules operate before answer generation. The answer reader is a separate model call.

## 14. Curated Demo: every panel explained

### 14.1 What is saved

The published collection contains:

- 10 frozen VAL examples from each of QASPER, HotpotQA and SciFact: 30 questions total;
- three retrieval settings: SFR, BM25 and MiniLM;
- two budgets: 256 and 1024;
- nine evidence methods;
- **1,620 saved method/retriever/budget conditions**;
- saved selected-unit deletion outcomes for attribution.

These are deliberately chosen illustrations, not a representative benchmark sample. This provenance belongs in documentation and the Researcher-mode expander; the main presentation remains clean.

Choosing a condition **does not run a model again**. It loads the corresponding saved answer and evidence. Curated Demo needs CPU resources for display, not SFR/Qwen inference.

### 14.2 Dataset, query, retriever and budget controls

The controls select a saved condition. Saved BM25-native QBGF is researcher-only and labelled **HISTORICAL NATIVE LEXICAL VARIANT**: raw lexical relevance has a different score scale from cosine, so it is not a controlled BM25-versus-SFR QBGF comparison. The dataset determines the task and scoring interpretation. The retriever determines the saved candidate pool. The budget determines the saved allocation. Left/right method selectors let you compare two policies for that same query, retrieval setting and budget.

The gold-reference expander is for retrospective evaluation. It is not evidence handed to the selector or answer reader.

### 14.3 Retrieved candidate pools

The expandable comparison displays the stored ranked sources for all three retrievers, including rank, score, source token count and full passage text.

A candidate is something that **could** be selected. Its presence in the retrieved pool does not mean the reader saw it. Relevance scores from different retrievers are not automatically on a comparable scale.

### 14.4 Answer panels

Each side shows the method, evidence meter, answer, task-specific example score, prompt tokens and generated tokens. The exact-evidence expander shows the saved selected context and its units.

For SciFact, the per-example score is correctness, 0 or 1. It is not an individual “macro-F1.” Cohort macro-F1 belongs in aggregate findings.

Where available, provenance includes prompt/evidence/source/model hashes, measured generation seconds and cache-reuse information.

### 14.5 Same source, every granularity

Select a parent passage to inspect its full text, sentence units and deterministic fine source spans. These are actual retained units, not generated summaries or artificial propositions.

In this shared-source panel, **SELECTED refers to either of the two displayed contexts**. Inspect the individual answer panels to determine membership in each method separately.

### 14.6 Real packing constraints

The app compares a passage's standalone serialized cost against the current budget. It explicitly shows when the full passage cannot fit even into an empty context.

A passage fitting alone does not guarantee selection: other sources may consume the available budget or the method may prefer another representation.

### 14.7 Representations and budgets table

The table shows saved scores and token counts for every method at both budgets under the currently selected query and retriever. It is a per-example comparison, not an aggregate ranking of methods.

### 14.8 What actually changed the answer?

Choose a displayed method, then select a point in the relevance-versus-deletion-utility scatter or choose a source unit directly. The panel shows:

- the selected unit's exact text;
- its relevance and source length;
- the original answer;
- the answer after removing that unit;
- the measured score change and category.

NONE has no units to delete and therefore no selected-unit deletion experiment.

## 15. LiveDemo: inputs, stages, results and comparisons

### 15.1 Inputs

The left column contains editable passage cards. Add or remove passages, use the authored sample, or import blocks separated by a line containing `---`. Optional `Passage 1:` headers are removed during bulk parsing.

The right column contains the query, evidence budget, method, retriever and **Run Evidence Pipeline** button. The **Budget-sensitive multi-aspect question** button loads the explicitly approved static World Cup diagnostic preset without starting inference. It is a supplied scenario, not verified factual ground truth and not a tuning set.

Current validation limits:

| Input | Limit |
|---|---:|
| Number of passages | 1–20 |
| Characters per passage | 8,000 |
| Total passage characters | 30,000 |
| Query characters | 2,000 |
| Query length under the reader tokenizer | 512 tokens |
| Total source passage tokens | 16,000 |

Empty queries and empty passage cards produce errors. The approximate word-based input counter is a convenience; exact token limits and evidence costs use the tokenizer.

Each supplied passage remains an intact native candidate passage. The live app does not silently split an oversized passage into smaller passage candidates just to make RAW_SKIP fit.

Live retrieval ranks **only the user-provided passages**. It does not search the web or an external benchmark corpus. At most the top 10 passages enter the selection pool.

### 15.2 Actual execution stages

| Stage | What the software does | What the user can inspect |
|---|---|---|
| Query | Validate/tokenize; encode with SFR if used | Query, token count and embedding identity |
| Passages | Build exact source-linked passage/sentence units | Passage count, source tokens, sentence count |
| Retriever | Rank supplied passages | Candidate ranking, scores, lengths |
| Granularity | Prepare the units needed by the chosen method | Whole-passage or sentence units; QBGF option family |
| Allocation | Apply the selected policy under the token cap | Included/skipped units or QBGF choices, exact cost |
| Reader | Construct the prompt and optionally generate | Reader identity, prompt/evidence tokens, status |
| Answer | Assemble the result view | Answer or a clear reader-unavailable message |

Presentation Mode uses **Controlled retriever-only comparison** by default. Pipeline-native comparison and the exact historical splitter are advanced Researcher Mode choices. Controlled scoring needs the pinned local SFR model even when BM25 supplies the candidate pool. No lexical fallback is silently substituted.

The progress indicator uses pending, processing, completed, failed and skipped states. It reflects real operations and has no artificial waiting period. It does not display hidden model reasoning or chain-of-thought.

NONE still validates the inputs and prepares a query-only prompt, while retrieval and representation are marked skipped.

### 15.3 Results

Live results use the same visual components as curated examples:

- query header and selected configuration;
- ranked candidate table with expandable passage text;
- model answer, when generation succeeded;
- evidence-budget meter;
- source tokens, selected-unit count, prompt tokens and generated tokens;
- exact evidence sent to the reader, or **prepared** evidence if generation was unavailable;
- actual allocation decisions;
- a same-source full-passage/sentence/selected-representation view;
- a deterministic “What happened in this run?” summary;
- a method-specific explanation.

The allocation trace for RAW_SKIP and FIXED_SENTENCE is checked against the actual selected context. QBGF displays its chosen option and cost for each passage. These are operational explanations, not reconstructed model reasoning.

No answer F1, EM, gold answer or correctness badge is calculated for arbitrary custom queries. The exact frozen World Cup preset has a separately labelled, user-supplied **Preset Reference Answer** for demonstration-only overlap metrics (see §15.10); it is not official gold. Under generated answers, a neutral caption states that custom answers are not automatically verified for factual correctness or complete support; readers should inspect exact selected evidence. No automatic correctness classifier is added.

### 15.4 Method comparison

**Compare with another method** preserves the original run's question, passages, budget and retriever. It runs the other method if a completed answer is not already cached for that configuration in the session.

**Run History / Comparison Tray** keeps the last 30 distinct LiveDemo runs in browser-session memory, including alternate-method and end-to-end retriever runs. Its compact table shows run number, retriever, method, budget, actual evidence tokens, evidence-unit count and whether an answer is available. Reusing a cached condition updates its existing row instead of adding a duplicate. Ranking-only retriever previews are not saved as answer runs. Context-only runs remain inspectable and are labelled explicitly.

Select any two rows using **Select two saved runs**, then click **Compare Selected Runs**. This opens two saved answer/evidence cards, candidate pools and allocation traces without executing retrieval, embeddings or generation again. Runs remain available after changing the current inputs or budget, even if removed from the smaller generation cache. Changed input/configuration fields are listed; pairs with different inputs, protocols or multiple settings are descriptive comparisons, not isolated causal effects. No answer is declared better without a gold reference. The tray is session-only, clears when the session ends, and never writes experimental results. It summarizes many runs while showing only two detailed outputs at once.

The comparison shows answers, evidence tokens, source count, evidence-unit count and prompt tokens. It does not declare an answer better without a reference.

Editing the main inputs does not silently rewrite the old comparison. The app identifies the last completed run and requires another run to apply edits.

### 15.5 Retriever comparison

When SFR is locally available and enabled, **Compare Retrievers** displays rankings without requesting an answer-reader call. **Compare Retrievers End-to-End** holds question, source text, segmentation, common SFR downstream scoring, method, budget, reader, prompt template and decoding fixed, changing candidate retrieval only. It shows both pools, selections, exact contexts and answers; it does not score custom answers. The invariants panel records what stayed fixed. A different evidence context naturally gives different exact prompt IDs even though the prompt template is held fixed.

### 15.6 Reader unavailable or disabled

With live models disabled, BM25-based context exploration remains usable. If Qwen loading or generation fails, the app preserves the prepared evidence and reports generation as unavailable.

A semantic SFR run still requires its actual encoder. It is not silently replaced with a lexical method if the model is missing. No fake answer is supplied.

### 15.7 What the controlled retriever effect means

- **RAW_SKIP:** retrieval ranking itself matters because whole passages are considered in that order. Candidate membership and order can therefore change what fits.
- **FIXED_SENTENCE:** common SFR unit scores determine the global relevance ordering after candidate retrieval. Raw BM25 ranks are not reused as sentence relevance. The retriever primarily changes which passages enter the pool.
- **QBGF:** common SFR passage weights and unit scores drive the unchanged global allocator. The retriever primarily changes candidate availability; identities and deterministic tie order can naturally affect the feasible allocation. Retriever scores are not substituted for downstream semantic utility.

### 15.8 The frozen diagnostic preset

The World Cup preset exposes all ordinary live views plus a focused P7/P9 panel. P7 is the workload passage, P9 the abandoned-format passage. Separate columns show whether each was retrieved, its rank, whether it was dropped, selected in full, represented by all source units or represented partially, the actual selected unit count and the QBGF choice where relevant. The original and selected excerpts are shown. These are actual algorithmic states, not correctness or reader-utilization scores. The example may demonstrate failure and must not be used to retune QBGF. Retrieval, selection, allocation and reader utilization remain separate concepts.

### 15.9 Run-history presentation workflow

Use the tray to summarize several configurations, then inspect only two in detail. It does not create a three-, four- or six-column comparison.

1. Keep the query, passages, SFR retriever and scoring policy fixed. Run RAW_SKIP at 256.
2. Use **Compare with another method** to run FIXED_SENTENCE, then QBGF_FULL. Each completed condition appears in the history table.
3. Change the main budget to 1024 and press **Run Evidence Pipeline**. Repeat the method comparisons at that budget. Changing the selector alone does not run a new condition.
4. Scroll to **Run History / Comparison Tray**. Under **Select two saved runs**, choose exactly two entries, then click **Compare Selected Runs**.
5. Inspect the two saved answers, exact evidence, budget meters, expandable candidate pools and budget-allocation decisions. Selecting rows alone does not replace the displayed pair; click the comparison button to apply the selection.
6. Choose a different pair and click the same button again. This inspection makes no model calls, even when the pair uses different budgets.

The compact history columns mean:

| Column | Meaning |
|---|---|
| Run | Stable session run number; older entries leave the tray after 30 distinct conditions |
| Retriever | Candidate retriever recorded for that run; NONE still skips retrieval |
| Method | Actual evidence-policy ID |
| Budget | Configured input-evidence token limit, not output length |
| Evidence | Actual selected evidence-token count |
| Units | Number of selected evidence units, whose granularity depends on the method |
| Reader | **Answer saved** or **Context only**; the latter has no generated answer to compare |
| Saved answer | The actual saved answer text, displayed directly in the history table for quick comparison. Context-only runs show **No answer — reader unavailable or disabled**. Select a pair to inspect long answers in full alongside their evidence. No new generation is requested. |
| Reference metrics | For the exact World Cup preset only: token F1, normalized EM, token precision/recall and required-fact cue matches. Arbitrary custom runs show **No gold-reference evaluation available**. |

For an isolated method comparison, choose runs with the same inputs, retriever, budget and protocol. For a budget comparison, keep the method and other settings fixed. For the reviewed retriever-only experiment, use **Compare Retrievers End-to-End**; its completed runs also enter history. The tray lists changed settings and warns when a selected pair differs in multiple settings or input/protocol details. Arbitrary saved pairs are useful for inspection but do not establish which change caused an answer difference.

History is temporary server-side session memory associated with the browser session. It is not an export or durable experiment record; ending the session or restarting the app can clear it. It does not write custom inputs or answers to research artifacts.

### 15.10 World Cup Preset Reference Answer and checklist

The preset metadata (`demo/assets/presets/world_cup.json`) includes the supplied **Preset Reference Answer**, nine required facts and a versioned metric protocol. It is cached with the preset metadata. It is not an official gold answer, verified factual account, benchmark annotation or new confirmation outcome. The original query and source passages are unchanged.

When history includes a run whose query and all source passages match this frozen preset, the table adds **row 0 — Preset Reference Answer**. Row 0 is a stored reference anchor, not a generated run; it has no evidence budget, retrieval, token cost or measured performance. The two-run selector continues to select actual saved runs. Inspect the reference text in the table or reference expander.

The column immediately after **Saved answer** shows:

| Diagnostic | Definition |
|---|---|
| P | Multiset token overlap divided by generated-answer token count |
| R | Multiset token overlap divided by reference-answer token count |
| F1 | `2 × P × R / (P + R)`; zero when no tokens overlap |
| EM | 1 when normalized answer and reference strings are identical, otherwise 0 |
| Fact cues | Matched required-fact cues out of 9, also shown as a percentage |

Overlap and EM reuse the project's HotpotQA normalization: lowercase, remove ASCII punctuation and English articles, and collapse whitespace. This normalization is distinct from the checklist's cue preprocessing. An empty answer scores zero; an unavailable answer is **not evaluated**, rather than scored as a failure. Overlap is lexical, not a probability of correctness; a good paraphrase can have low F1 or EM.

The answer-card expander **Preset Reference Answer and required-facts checklist** shows the reference and per-answer matched/not-matched cues for:

1. 48 teams.
2. 12 groups of 4.
3. Top 2 + 8 best third-place teams advance.
4. Original 16 groups of 3 proposal.
5. 64 → 104 matches.
6. 7 → 8 matches for semifinalists.
7. 39-day tournament.
8. Collusion concern.
9. Penalty shootouts would not fully solve collusion risk.

Checklist matching uses documented regular-expression cues, common number-word normalization and same-sentence/semicolon-clause co-occurrence. It is a surface heuristic: it can miss paraphrases or match a statement containing a contradiction. It does not establish factual support, completeness or truth. No LLM judge is called.

Scoring runs only after generation, while displaying saved answers. The reference and checklist never enter the query, retrieval, sentence scoring, allocation or reader prompt. Editing the preset query or any passage disables these metrics; unrelated LiveDemo inputs keep **No gold-reference evaluation available**. Mixed histories score only the matching preset rows. No metrics or references are appended to benchmark, validation or confirmation results, and viewing them makes no new model calls.

## 16. CPU/GPU execution and model lifecycle

### 16.1 Default CPU FP16 mode

The device selector defaults to **CPU · FP16 · sequential models**. Enable Live Models to permit actual inference.

For an SFR run:

1. Load the pinned local SFR model on CPU in FP16.
2. Encode the question and needed source units.
3. Retain the small embedding arrays needed for selection.
4. Unload SFR before loading Qwen.
5. Load the pinned Qwen3-8B reader on CPU in FP16.
6. Generate the answer.
7. Unload Qwen.

Context-only CPU runs also release the encoder at completion. Error paths attempt the same cleanup. The tokenizer and bounded embedding cache can remain available without keeping the large models resident.

Before loading a large CPU model, the app requires at least **24,000 MiB available RAM**. It uses up to eight CPU threads. Freed tensor arenas are returned to the operating system where supported. Reclaimable file-cache memory may remain visible after unloading; this does not mean model objects remain active.

A small real authored two-passage check on this server completed in **25.8 seconds**, peaking at **18.2 GiB process resident memory**, and returned “Sensor B lasts longer.” Both models were unloaded. This is a functional check, not a prediction for every input size.

### 16.2 Optional GPU mode

GPU FP16 mode remains selectable. It checks available GPU capacity and makes SFR and Qwen take turns rather than keeping both large models on the GPU. GPU mode may keep its last reader resident for reuse until a handoff or explicit unload.

The **Unload live models** button releases loaded live resources. Neither mode loads large models merely because the application starts.

### 16.3 Identities and decoding

| Component | Pinned identity |
|---|---|
| SFR | `Salesforce/SFR-Embedding-Mistral` @ `33a60fe7b5e00b9414981280c571e8a4f9f42f1b` |
| Reader | `Qwen/Qwen3-8B` @ `b968826d9c46dd6066d109eabc6255188de91218` |
| Curated MiniLM | `sentence-transformers/all-MiniLM-L6-v2` @ `1110a243fdf4706b3f48f1d95db1a4f5529b4d41` |

The reader is base FP16, with no LoRA adapter. Decoding is greedy (`do_sample=False`), with seed 42 and thinking disabled in the chat template. CPU uses a CPU-supported attention backend rather than the CUDA-only efficient-attention backend.

CPU and GPU numerical differences can affect live outputs. A custom live answer is not a replacement for a saved curated answer or research score.

Models are loaded from local pinned checkpoints only. The live UI does not automatically download checkpoints. A small integrity-checked tokenizer-only bundle supports context construction without loading Qwen's weights.

## 17. Metrics, relevance and deletion utility

### 17.1 Source tokens, evidence tokens, prompt tokens, generated tokens

- **Source tokens:** tokenizer length of source content before selection. A card's source length does not necessarily include serialized block overhead.
- **Evidence tokens:** exact IDs allocated to selected evidence, including its block separators.
- **Prompt tokens:** all IDs sent into the reader, including query, instructions and evidence.
- **Generated tokens:** newly produced answer IDs, excluding the input prompt.

These counts answer different questions and should not be substituted for each other.

### 17.2 Answer token F1

For QA, compare normalized answer tokens with the reference using token multiset overlap. Let \(O\) be overlap, \(N_p\) predicted-token count and \(N_g\) reference-token count:

\[
P=\frac{O}{N_p},\quad R=\frac{O}{N_g},\quad
F1=\frac{2PR}{P+R}.
\]

The dataset evaluator handles empty strings, normalization, answer types and multiple references. The application reuses these evaluators; it does not introduce a new common scorer.

Token F1 is an overlap score, not a probability of correctness or a complete semantic judgment. A single saved example's F1 should not be confused with the dataset average.

### 17.3 Exact match

\[
EM=\mathbf{1}\{\text{normalized prediction} = \text{normalized reference}\}.
\]

It is stricter than partial token overlap. Accepted references and normalization follow the relevant dataset evaluator.

### 17.4 SciFact correctness, accuracy and macro-F1

SciFact uses SUPPORT, REFUTE and NOT_ENOUGH_INFO.

Per-example correctness is 1 for the correct parsed label and 0 otherwise. Accuracy is the mean of those correctness values.

For each class \(c\):

\[
F1_c=\frac{2TP_c}{2TP_c+FP_c+FN_c}.
\]

The cohort's three-class macro-F1 is:

\[
\text{macro-F1}=\frac{F1_{SUPPORT}+F1_{REFUTE}+F1_{NOT\_ENOUGH\_INFO}}{3}.
\]

The strict reader parser accepts an allowed label after trimming whitespace; other outputs are invalid. Invalid outputs count as incorrect. Format adherence matters, which is why reader comparisons must consider invalid-output rates.

### 17.5 HELP and HURT relative to NONE

Where these stored diagnostics are available, let \(\Delta\) be the example's method score minus its NONE score.

- For QA, HELP means \(\Delta>0.01\), HURT means \(\Delta<-0.01\).
- For verification, the threshold is zero, so correctness flips determine HELP/HURT.
- Intermediate changes are neutral under the registered threshold.

This is an answer-outcome comparison. It is different from a source unit's deletion category.

### 17.6 Relevance

Relevance is the ranking signal used before answer generation. Dense relevance is cosine similarity:

\[
\cos(q,x)=\frac{q^\top x}{\|q\|\|x\|}.
\]

BM25 uses lexical scores rather than cosine. Neither score is ground-truth evidence support or a probability that the answer is correct.

### 17.7 Leave-one-out deletion utility

For selected context \(C\) and one selected unit \(u\), keep the remaining selection fixed, remove \(u\), serialize again and regenerate. Do not rerun the selector to refill its space.

\[
I(u)=S(\text{answer with }C)-S(\text{answer with }C\setminus\{u\}).
\]

The curated categories are:

- **HELPFUL:** \(I(u)>0.01\) for QA, or \(I(u)>0\) for verification;
- **HARMFUL:** \(I(u)<-0.01\) for QA, or \(I(u)<0\) for verification;
- **NEUTRAL:** otherwise.

A helpful unit is one whose deletion reduced the measured score. A harmful unit is one whose deletion increased it. “Neutral” means no sufficiently large observed score change, not “ignored,” “irrelevant” or “useless.” Redundancy, interactions and scoring insensitivity can all yield neutral effects.

The scatter plot has **relevance on the horizontal axis** and **deletion utility on the vertical axis**. High relevance with low utility illustrates the gap between matching the question and improving the final answer.

Deletion tests change an input under a particular reader/prompt/context. They do not expose hidden model reasoning or prove a general causal property of the source statement.

### 17.8 Gold coverage and retention diagnostics

Research panels also show retrospective annotation overlap and eligible-subset retention diagnostics. Gold overlap is computed after selection for evaluation; it does not guide selection. Unknown mappings are not silently treated as known non-gold evidence.

Retrospective subsets chosen using outcomes are not deployable selection policies. Relative retention can be undefined when full evidence does not improve over NONE, and can exceed one. Read the eligibility counts with any reported ratio.

### 17.9 Uncertainty and correlations

Research Findings loads saved paired differences, cluster-bootstrap confidence intervals, adjusted p-values where registered, and relevance/utility correlations. The app does not recompute a new significance test when you select a tab.

A paired difference compares methods on matched examples. A confidence interval spanning zero does not establish a positive effect. Nominal intervals and multiplicity-adjusted tests have different roles. Directional consistency across validation and confirmation is descriptive; the two cohorts are not paired samples.

### 17.10 Timing and cache costs

Saved generation seconds measure the recorded generation stage, not necessarily retrieval, loading or complete end-to-end execution. Live measured run duration includes the operations and interface updates performed in that run; it is not a serving-throughput benchmark.

An exact-prompt cache hit avoids another physical generation. The corresponding method still conceptually requires a reader call when describing its algorithm. Cached study rows are not zero-cost answer systems.

## 18. Research Findings

The workspace contains eight topic tabs. It reads audited primary CSV/JSON artifacts rather than extracting numbers from prose reports.

| Tab | What it explains | Main interpretation boundary |
|---|---|---|
| Evidence helps | FIXED_SENTENCE versus NONE | Tests the evidence-assisted pipeline, not retrieval in isolation |
| Granularity × budget | RAW_SKIP, FIXED_SENTENCE, H3 and QBGF | Task/budget dependence; no universal fine-granularity winner |
| Query conditioning | QBGF_FULL versus QBGF_NO_QUERY | Allocation ablation on VAL; upstream retrieval remains query-dependent |
| Fusion ablation | QBGF_FULL versus QBGF_SENTENCE_ONLY | Whether FULL options add robust benefit over sentence-only allocation |
| Relevance ≠ utility | Deletion categories, correlations, overlap and retention | Post-hoc diagnostics, not an inference-time oracle |
| Retriever effects | Coverage, matched downstream controls and specialist context | Different pools/scopes and score scales matter |
| Reader effects | Saved reader panels and format compliance | Format failure can confound comparisons of reasoning quality |
| Decision & complexity | Quality/cost tables and the freeze protocol | More options did not imply monotonically better quality |

The research-only ablations shown here are **not extra live method choices**:

- `QBGF_NO_QUERY`: retain the query-dependent retrieved pool but remove query conditioning within allocation. Passage weights are uniform; sentence ordering/mass no longer uses query scores.
- `QBGF_SENTENCE_ONLY`: retain the query-conditioned allocation but remove FULL-passage options.
- Specialist SARA references: historical/specialist systems with their own checkpoints and evidence interfaces. They are not interchangeable with the current Qwen reader or selectable live methods.

The dataset task cards explain why QA F1 and SciFact macro-F1 should not be averaged into one project score. Simple/Technical views control the detail shown.

**Scope of confirmation:** HotpotQA has 200 reserved queries; SciFact has 50 claims in 43 frozen groups. There is no fresh QASPER confirmation. These reserved outcomes have now been observed, so the cohort is no longer untouched for future model selection.

The application preserves the distinction between validation mechanisms, post-hoc attribution and final confirmation.

## 19. Research Journey and the frozen decision

Research Journey connects the stages:

```text
W10 diagnosis
  → W11 multi-dataset study
  → W12 C validation comparisons
  → W12 D controlled ablations
  → W12 H/I final decision and freeze
  → W12 K/L post-hoc attribution
  → W12 M reserved confirmation
```

The final retained policy is:

```text
SFR top-10 retrieval
  → FIXED_SENTENCE allocation
  → Qwen3-8B BASE_FP16
  → no LoRA
  → 256 / 1024 evidence-token budgets
```

QBGF remains the experimental adaptive allocator. It was not promoted as a universal winner. A favorable individual example or confirmation cell does not retroactively change the frozen method-selection decision.

The main supported story is that evidence helps, allocation conditioning can matter, and the best evidence representation depends on task and budget. Universal sentence superiority, universal QBGF superiority, and “high relevance means useful evidence” were not established.

## 20. Presentation, provenance, storage and safety

### 20.1 Display modes

- **Presentation Mode:** emphasizes readable cards, progress and explanations; some live stage details are collapsed.
- **Researcher Mode:** exposes provenance, exact costs, timing, collection metadata and the historical replay option.

These switches do not change research scores or model weights. Researcher Mode also makes the exact historical segmentation option available. Selecting that option explicitly changes custom-input segmentation and uses distinct cache identities.

### 20.2 Saved-source integrity

At startup, the application verifies registered primary-source fingerprints and the frozen final-method artifacts. A mismatch is reported rather than replaced with guessed values.

The curated loader verifies its published bundle checksum, manifest identity, condition registry and complete-publication receipt. The UI-only rename of PROPOSITION does not rewrite its saved records or method ID.

Missing or unavailable measurements remain missing; they are not invented zeros.

### 20.3 Curated data to preserve

These are durable published demo assets:

```text
demo/replay_data/v1/bundle.json
demo/replay_data/v1/PUBLISHED.json
demo/replay_data/v1/manifest.json
```

The bundle contains the sources and selected evidence, answers, task scores, allocation metadata and deletion outcomes needed by the complete curated UI. It is not a disposable embedding cache.

The receipt records bundle SHA256:

```text
14b4ba43d4cf3ab7359b54f2fc8bfe0ee59166c12f40c09125658c1d63f03036
```

Intermediate build shards and model caches are different from the final published bundle. Do not assume that any directory named “cache” can be removed without checking which workflows depend on it.

### 20.4 Live state and privacy

Custom query/passages, prepared rankings and completed results are held in Streamlit session memory. They are not appended to benchmark, validation or confirmation files.

Large model resources are process-cached and accessed under a lock. The process-level embedding cache can retain a bounded set of input-derived keys/vectors. Prepared/comparison results are session-cached. This is a local research demo, not a production multi-user privacy service.

Only bounded authored-input qualification outputs are stored under `demo/runtime/qa/`. They are software checks, not research observations. The normal live pipeline does not persist custom answers there automatically.

The blanket demo ignore rule has been replaced by selective ignores for the final freeze. Application source, tests, documentation, checksums, tokenizer-only assets and a versioned compressed curated archive are versioned. Model weights, environments, arbitrary live inputs, runtime caches, debug logs and intermediate build shards remain excluded. The raw `bundle.json` stays ignored; restore it from the immutable archive as described below.

### 20.5 What is deliberately absent

- No hidden chain-of-thought display.
- No LLM-generated explanation of the pipeline's own decisions.
- No custom-input gold scoring or claimed correctness.
- No automatic internet search or checkpoint download.
- No automatic rewriting of frozen results.
- No live LoRA training or confirmation evaluation.
- No automatic switch from a failed requested model to an unlabelled substitute.

## 21. Launch instructions and walkthroughs

For a clean checkout, first restore the immutable published data using the local versioned archive; no regeneration is required:

```bash
python3.11 -m venv demo/.venv
demo/.venv/bin/python -m pip install -r demo/requirements.txt
demo/.venv/bin/python demo/restore_curated_bundle.py
demo/.venv/bin/python -m streamlit run demo/app.py
```

If the demo environment already exists, run just the restore and Streamlit commands. The restore is idempotent and refuses incompatible existing files.

If updating an already running app, stop that Streamlit process with Ctrl+C and restart it. Refreshing the browser alone may retain an old imported Python module or cached resource instance. For a remote server, forward the printed local port through SSH/your editor.

### Walkthrough A — understand the saved evidence policies

1. Open Curated Demo.
2. Pick a dataset and a question.
3. Hold the retriever and budget fixed.
4. Compare RAW_SKIP with FIXED_SENTENCE.
5. Open both exact-evidence panels.
6. Inspect whether a long passage fails to fit.
7. Change the budget and inspect how the saved selections change.
8. Select QBGF and inspect its allocation choices.
9. Inspect a unit's original versus deletion answer.

Changing one control at a time makes the differences easier to attribute.

### Walkthrough B — run actual CPU inference

1. Open LiveDemo.
2. Keep **CPU · FP16 · sequential models** selected.
3. Enable Live Models.
4. Use the authored sample or enter your own non-empty passages and query.
5. Select SFR, FIXED_SENTENCE and budget 256.
6. Press Run Evidence Pipeline.
7. Watch the real stage transitions, then inspect the context and answer.
8. Run the RAW_SKIP comparison on the same inputs.
9. Compare input construction without declaring a winner from plausibility alone.
10. Scroll to **Run History / Comparison Tray**, select the two completed runs, and click **Compare Selected Runs** to revisit their saved evidence without new inference.

For an evidence-only demonstration, leave live models disabled and use BM25. For NONE, expect no retrieval and an empty evidence context.

### Walkthrough C — explain the research conclusion

1. Open Research Findings and choose a dataset.
2. Compare the evidence-assisted policy with NONE.
3. Inspect granularity at both budgets.
4. Separate the VAL query-conditioning ablation from confirmation.
5. Inspect fusion results and relevance/utility diagnostics.
6. Open Research Journey to explain why the simple sentence policy was frozen.

## 22. Frequently misunderstood details

| Question | Answer |
|---|---|
| Does budget 256 mean a 256-token answer? | No. It limits selected input evidence. |
| Does it mean the whole input is at most 256 tokens? | No. The query, instructions and chat framing are additional. |
| Does FIXED_SENTENCE select a fixed number? | No. Its policy is fixed; the selected count varies. |
| Does C2 always select eight? | No. Eight is the eligible shortlist cap. Prefix packing can stop earlier. |
| Does H3 mean three generated summaries? | No. It treats the first three retrieved passages with sentence units. |
| Does QBGF_FULL mean every passage is full? | No. It enables the complete option family, including DROP and sentence subsets. |
| Is QBGF trained when I press Run? | No. The relevance-based optimizer runs; no controller training happens. |
| Are fine-grained source spans generated facts? | No. They are exact source substrings from punctuation rules. |
| Does MULTI_SENTENCE_4 fall back to two if four do not fit? | No. It skips that passage's selected group. |
| Does SELECTED mean helpful? | No. Selection and observed deletion utility are separate labels. |
| Does neutral deletion mean the reader ignored the unit? | No. Redundancy and interactions can conceal its contribution. |
| Does exact source support mean the answer is correct? | No. It verifies origin of the text, not correctness or sufficiency. |
| Are the curated answers generated again when I change a selector? | No. The saved condition is loaded. |
| Does LiveDemo evaluate my answer against a hidden gold reference? | No. Arbitrary inputs remain unscored. Only the exact frozen World Cup preset has a visible, user-supplied Preset Reference Answer for demo-only overlap and fact-cue diagnostics. |
| Does LiveDemo search the whole dataset? | No. It ranks the passages you supply. |
| Can live CPU results differ from saved GPU results? | Yes. Inputs, task mode and numerical execution can differ. |
| Are all research controls available live? | No. LiveDemo intentionally supports four reviewed policies. |
| Is QBGF the frozen final method? | No. FIXED_SENTENCE is the retained final policy. |
| Can the app tune on confirmation now? | No. Confirmation is consumed and remains separate from development. |

## 23. Implementation map

Paths are relative to the repository root.

| File/module | Responsibility |
|---|---|
| `demo/app.py` | Top-level navigation, modes, research panels and journey |
| `demo/curated/ui.py` | Complete curated collection, paired methods and deletion inspection |
| `demo/curated/protocol.py` | Curated registry, identities and immutable storage |
| `demo/curated/preparation.py` | Canonical selections plus three demo-only policies |
| `demo/curated/generation.py` | Saved answers, exact-prompt reuse and selected-unit deletions |
| `demo/curated/publication.py` | Complete bundle publication and checksum-verified loading |
| `demo/live/ui.py` | Live inputs, stage updates, output panels and comparisons |
| `demo/live/history.py` | Session-only run deduplication, compact history rows and comparison-setting differences |
| `demo/live/reference.py` | Post-generation, preset-only token-overlap and required-fact cue diagnostics |
| `demo/live/engine.py` | Observable execution, validation and packing traces |
| `demo/live/resources.py` | Local model loading, CPU FP16 backend and sequential unloading |
| `demo/components.py` | Shared source cards and budget meter |
| `demo/plots.py` | Quality bars, paired differences and deletion scatter |
| `demo/research_extensions.py` | Reader/retriever panels, protocol context and diagnostics |
| `demo/data_loader.py` | Read-only research loading and fingerprint checks |
| `demo/assets/live_tokenizer/` | Pinned tokenizer-only assets; no model weights |
| `src/week12/algorithm.py` | Canonical allocation rules and additive serialization |
| `src/week12/reader.py` | Canonical reader prompt, decoding and parser |
| `src/week11/hierarchy.py` | Exact source-linked units and hierarchy |
| `src/week11/propositions.py` | Conservative deterministic fine-span splitter |
| `src/week11/evaluation.py` | Task evaluation and evidence diagnostics |
| `demo/tests/` | Automated application and pipeline checks |
| `demo/runtime/qa/` | Authored-input qualification receipts, logs and screenshots |

This guide describes the currently implemented application. The primary artifacts and frozen protocols remain the authority for research numbers; the current source code remains the authority for execution details.

## Historical fallback-splitter limitation audit

A read-only census of frozen source text found no direct terminal-citation merge cases in the 40-paper Week-11 QASPER pilot, and three among 50,274 fallback units in the 281-paper Week-12 VAL source population (0.00597%). A broader citation/Markdown adapter would subdivide 52 units, but most of those differences are equation-layout or leading-citation effects, not confirmed sentence errors. None of the three direct merge cases occurs in the retained FIXED_SENTENCE/QBGF attribution subset; removed full prepared contexts prevent a complete selected-context sensitivity audit. Frozen results are unchanged, and no answer-score effect is inferred. See [the detailed limitation audit](audit/qasper_frozen_splitter_impact.md).

## Final release freeze

The final application version is **evidence-lab-demo-1.0.0**. See [FINAL_FREEZE_STATUS.md](FINAL_FREEZE_STATUS.md). The versioned archive is `demo/releases/curated-demo-v1.tar.gz` (13.93 MiB), accompanied by `curated-demo-v1.archive.json`; its members retain the original published bundle, receipt and manifest byte-for-byte. See [archive restore instructions](releases/README.md). Source and small assets are ordinary version-controlled files; arbitrary user session inputs, caches and model checkpoints are not. The explicitly approved static World Cup fixture is the presentation preset, not a mechanism for collecting users' custom inputs.

The historical QASPER conclusion is closed: W11 direct terminal-citation merges observed **0 / 7,066**; W12 **3 / 50,274 (0.00597%)**. None of those three appears in the retained selected-context subset. Complete historical selections are unavailable, so zero downstream impact cannot be proven; this audit establishes no material score damage. The W11 counterfactual total of 7,067 versus 7,066 frozen units is not inconsistent with zero subdivided frozen units: whole-passage segmentation and isolated-unit resplitting are different procedures, and citation/boundary ownership can change a whole-passage total without splitting an old unit in isolation. The discrepancy was not classified as a grammatical merge and was not used to infer score effects. No QASPER inference or source census was rerun in the freeze pass.
