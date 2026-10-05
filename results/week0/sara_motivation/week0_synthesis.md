# Week 0 Synthesis: Connecting Evidence Structure to SARA

**Document Purpose:** Ground the theoretical and empirical motivation for adaptive evidence allocation using local SARA codebases, repository documentation, and Week 0 findings.  
**Location:** `results/week0/sara_motivation/week0_synthesis.md`  

---

## 1. The Core Problem SARA Addresses

In long-document question answering (such as QASPER, where documents average over 4,000 words), standard language models face a fundamental trade-off between **context capacity, retrieval coverage, and inference cost**:

1. **The Cost of Pure Full-Text RAG ($k = n$):**
   - In standard RAG, all $n$ retrieved passages are formatted as raw text inside the LM prompt.
   - For $n=10$ passages of chunk size 256, full text consumes **$\approx 1,750\text{--}2,200$ context tokens** per query (`results/week2/raw/dev_fixed_k_matrix.jsonl`).
   - Because transformer self-attention scales quadratically ($\mathcal{O}(L^2)$) with prompt length $L$, feeding full text for large $n$ increases time-to-first-token latency, memory consumption, and API costs.
   - Furthermore, as demonstrated in our Week 0 evidence analysis (`results/week0/dataset_analysis/dataset_analysis_report.md`), approximately 67% of development questions contain a single annotated evidence paragraph, while BM25 recall continues to increase with retrieval depth, illustrating heterogeneity in retrieval/evidence structure.
2. **The Loss of Pure Soft Compression ($k = 0$):**
   - Pure context compression approaches (e.g. compressing entire passages into a single vector) dramatically reduce prompt length.
   - However, soft-compressing an entire 256-word technical scientific passage into a single 4096-dimensional vector inevitably loses fine-grained lexical details—such as exact mathematical formulas, numerical metrics, table cells, and entity names.
   - On QASPER, pure soft compression ($k=0$) achieves an answer $F_1$ of only **0.1856** (falling far short of full-text RAG at $0.3950$).

---

## 2. SARA's Hybrid Architectural Solution

To resolve this dilemma, SARA (*Selective and Adaptive Retrieval-augmented Generation with Context Compression*, Jin et al., ACL 2026; local documentation: [`Baselines/SARA-main/README.md`](file:///home/gunavenkat/Downloads/CS787-RAG-Project/Baselines/SARA-main/README.md#L41-L53)) introduces a hybrid prompt representation:

$$\text{Evidence Pool } (n \text{ candidates}) \longrightarrow k \text{ Full-Text Passages} \;+\; (n - k) \text{ Soft-Compressed Passages}$$

### Exact Local Architectural Implementation:
- **Token Extension:** The tokenizer is extended with `<pad>` (id 32000) and `<COMPRESS>` (id 32001) (`Baselines/SARA-main/src/model/loader.py`, lines 44–55).
- **Prompt Construction:** The prompt builder (`src/week2/sweep_runner.py`, lines 57–120) formats the top-$k$ retrieved passages as natural language text in `## Context`, while ranks $k+1 \dots n$ are represented as $n - k$ individual `<COMPRESS>` tokens in `## Additional Context`.
- **Soft Injection:** In `XMistralForCausalLM` (`Baselines/SARA-main/src/model/xMistral/`), the input embeddings at each `<COMPRESS>` token position are intercepted and replaced with the output of a trained projection layer mapping the 4096-dim retriever embedding (from `Salesforce/SFR-Embedding-Mistral`) into Mistral's hidden dimension.
- **Fixed-Budget Operation in Original SARA:**
  In the original SARA implementation (`Baselines/SARA-main/src/run_sara.py`, lines 158–164), **$k$ is a static, globally fixed hyperparameter** specified at runtime via `--k <int>`. Every single query in the evaluation split is forced to use the exact same budget $k$ (e.g., $k=5$ for all queries).

---

## 3. What Parameter $k$ Controls Conceptually

Conceptually, $k$ is not merely a hyperparameter; **$k$ is a representation fidelity allocation knob**:

| Component | Representation Type | Token Cost per Passage | Information Fidelity | Best Suited For |
|---|---|---|---|---|
| **Ranks $1 \dots k$** | Natural Language Text | $\approx 150\text{--}200$ tokens | High (Verbatim tokens) | Precise entity answers, tables, numbers |
| **Ranks $k+1 \dots n$** | Soft Embedding Vector | **Exactly 1 token** (`<COMPRESS>`) | Coarse (Semantic gist) | Broad background context, coarse grounding |

Varying $k$ shifts the evidence boundary:
- Increasing $k$ increases textual fidelity and recall, but increases context tokens and non-annotated context.
- Decreasing $k$ cuts context tokens and inference cost, but risks losing critical verbatim details if gold evidence is relegated to compressed status.

---

## 4. The Research Motivation for Adaptive Allocation (QCCA)

Week 0 establishes three empirical and architectural realities:

```text
1. Dataset Evidence Requirements are Structurally Heterogeneous
   - 67.1% of queries need only 1 paragraph
   - 22.1% need 2 paragraphs
   - 10.8% need 3+ paragraphs across multiple sections
   (Source: results/week0/dataset_analysis/dataset_analysis_report.md)
                        +
2. Retrieval Depth Exhibits Differential Gains
   - BM25 Recall_any reaches 80.95% at k=6, but complete Recall_all reaches only 70.56%
   - Deeper retrieval captures additional complete evidence at the cost of larger prompts
   (Source: results/week0/dataset_analysis/dataset_analysis_report.md)
                        +
3. SARA Exposes an Evidence Allocation Parameter k
   - Original SARA fixes k globally (e.g. k=5 for every query)
   - This uniformly allocates tokens regardless of query evidence requirements
   - Single-fact queries receive the same textual budget as complex multi-section queries
                        ↓
             THE SCIENTIFIC QUESTION:
   Should the high-fidelity evidence budget k be dynamically conditioned
   on observable query and retrieval characteristics per query?
```

### The Progression to Weeks 1–3:
Week 0 does **not** claim to prove that adaptive $k$ works. Instead, it provides the quantitative and mechanistic justification for the experimental campaign:
- **Week 1 / 2:** Evaluate SARA across fixed budgets $k \in \{0, 2, 4, 5, 6, 8, 10\}$ to map the static Pareto frontier.
- **Week 3:** Quantify the theoretical headroom between static baselines and an optimal per-query oracle allocator ($k^*$).
- **Weeks 4–5:** Train and validate lightweight pre-generation predictors to dynamically decide evidence allocation.
