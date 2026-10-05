# Week 0 — Understanding Evidence Utilization in RAG

**Project:** Query-Conditioned Context Allocation (QCCA) for SARA  
**Framework:** SARA (*Selective and Adaptive Retrieval-Augmented Generation with Context Compression*)  
**Base Model:** `mistralai/Mistral-7B-Instruct-v0.2`  
**Retriever:** BM25 (chunk size 256, overlap 5)  
**Dataset:** QASPER (*Question Answering on Scientific Papers*)  
**Date:** October 2026  
**Status:** AUDITED & VERIFIED (See [`scientific_audit.md`](file:///home/gunavenkat/Downloads/CS787-RAG-Project/results/week0/scientific_audit.md))  
**Artifact Directory:** `results/week0/`  

---

## 1. Research Motivation

Standard Retrieval-Augmented Generation (RAG) typically operates under a **fixed-budget paradigm**: a uniform number of retrieved passages ($n$) is converted to text and appended to the language model's prompt for every query, regardless of the question's intrinsic difficulty, evidence dispersion, or retrieval clarity.

In scientific literature question answering (QASPER), documents are long ($\approx 4,000$ words) and multi-faceted. Providing a small, fixed textual budget risks missing essential evidence (under-retrieval), while providing a large, fixed textual budget introduces non-annotated context, increases quadratic self-attention latency, and inflates context-token costs.

The SARA architecture introduces a hybrid mechanism: representing retrieved candidates as $k$ textual passages and $n - k$ soft-compressed retrieval embeddings injected at `<COMPRESS>` tokens. However, the original SARA implementation evaluates globally fixed $k$ values.

**Week 0 investigates the empirical and structural foundation for adaptive allocation:**
1. Does the evidence requirement vary across queries in scientific QA?
2. What are the retrieval characteristics of BM25 across retrieval depths $k$?
3. How is the prompt formatted under causal decoder-only constraints, and how does attention flow between query and retrieved context?
4. How do these observations motivate dynamic evidence allocation ($k$) in later weeks?

---

## 2. Dataset Characterization (Verified on Real QASPER Data)

The QASPER dataset was analyzed using the modular `QASPERAdapter` across the official document-level split (772 train papers / 2,088 queries; 86 dev papers / 231 queries; test split strictly frozen).

### 2.1 Structural Profile
- **Document Length Distribution:**
  - Train papers average **$3,694.3$ words** (median $3,526.5$ words); Dev papers average **$4,228.1$ words** (median $3,617.5$ words, corresponding to $\approx 4,700\text{--}5,500$ tokens).
  - Whole-document ingestion without selective retrieval or compression is computationally prohibitive for batch inference.
- **Question & Answer Length:**
  - Queries average **$8.39$ words** (compact, specific inquiries).
  - Target answers average **$14.45$ words** on dev ($13.65$ on train; concise phrases, entities, or brief explanatory sentences).
- **Answer Categories (Dev Split):**
  - Extractive: **$57.58\%$** (133 / 231)
  - Free-form / Abstractive: **$25.97\%$** (60 / 231)
  - Yes / No: **$16.45\%$** (38 / 231)

### 2.2 Evidence Distribution
- **Paragraph Concentration:**
  - An average query requires **$1.64$ gold evidence paragraphs** in Dev ($1.78$ in Train).
  - **$67.10\%$** of answerable queries are satisfied by a **single** paragraph (142 / 220).
  - **$22.27\%$** require **two** paragraphs (49 / 220).
  - Only **$13.18\%$** require **three or more** paragraphs (29 / 220).
- **Document Depth & Redundancy:**
  - Evidence paragraphs are centered at relative depth **$0.42 \pm 0.23$** within papers.
  - Multiple evidence paragraphs for the same query have very low lexical redundancy (pairwise Jaccard $= 0.0306$), indicating that when multi-paragraph evidence is required, the chunks provide complementary information across sections.

### 2.3 Retrieval Structure (BM25 Diagnostics on Dev Split)
To evaluate retrieval depth rigorously, two distinct recall definitions are evaluated across all 231 dev queries:
- **`Recall_any@k`:** At least one annotated evidence paragraph retrieved in top $k$.
- **`Recall_all@k`:** All annotated evidence paragraphs retrieved in top $k$.
- **`Mean_Coverage@k`:** Mean fraction of annotated evidence paragraphs retrieved.

| Retrieval Depth ($k$) | `Recall_any@k` (%) | `Recall_all@k` (%) | `Mean_Coverage@k` (%) |
|---|---|---|---|
| **$k=1$** | 33.33% | 22.94% | 28.99% |
| **$k=2$** | 51.52% | 37.23% | 45.83% |
| **$k=4$** | 70.13% | 55.84% | 66.52% |
| **$k=6$** | 80.95% | 70.56% | 79.85% |
| **$k=8$** | 84.85% | 77.06% | 85.53% |
| **$k=10$** | 87.01% | 80.09% | 88.27% |

**Observational Finding:**
Approximately 67% of development questions contain a single annotated evidence paragraph, while BM25 recall continues to increase with retrieval depth, illustrating heterogeneity in retrieval/evidence structure. At $k=2$, nearly half of all queries miss gold evidence, whereas at $k=6$, over $80\%$ retrieve at least one evidence paragraph.

![Dataset Distributions](file:///home/gunavenkat/Downloads/CS787-RAG-Project/results/week0/figures/dataset_distributions.png)
![Retrieval Diagnostics](file:///home/gunavenkat/Downloads/CS787-RAG-Project/results/week0/figures/retrieval_diagnostics.png)

---

## 3. Model Input Format & Causal Attention Architecture

### 3.1 Prompt Formatting & Token Order
In SARA, evaluation prompts are formatted with the context preceding the query:
1. Instruction prefix (`<s> [INST] Answer my questions based on the given context...`)
2. Retrieved passages (`Document 1. ... Document 10. ...`)
3. Task specification (`## Your Task ...`)
4. Query (`Question: <query>`)
5. Answer prefix / closure (`Your Answer: [/INST]`)

A concrete token-span map was verified and saved in [`results/week0/layer_analysis/token_span_map_example.json`](file:///home/gunavenkat/Downloads/CS787-RAG-Project/results/week0/layer_analysis/token_span_map_example.json).

### 3.2 Directional Validity Under Causal Mask
Because Mistral-7B enforces a lower-triangular causal attention mask ($j \le i$):
- **Passages cannot attend to the query:** Passage tokens appear at positions $19 \dots 2301$, whereas query tokens appear at positions $2331 \dots 2347$. Passage hidden states at all layers are strictly question-agnostic encodings of the context chunks.
- **The query attends back to passages:** Query tokens at positions $2331 \dots 2347$ attend over all preceding context tokens.
- **Terminology:** All attention metrics represent an **attention-based passage utilization proxy** measured in the **query-to-passage** direction.

---

## 4. Audit Status of Part B & Part C Diagnostics

> [!CAUTION]
> **Audit Status: Offline Verification Smoke-Tests (Unverified on GPU)**  
> As documented in [`results/week0/scientific_audit.md`](file:///home/gunavenkat/Downloads/CS787-RAG-Project/results/week0/scientific_audit.md), the numerical values previously generated for Part B (layer cosine similarity trajectories) and Part C (passage concentration $N_{\text{eff}}$, Gini, and correlation coefficients) were executed using offline deterministic simulations (`run_mock_layer_analysis` and `run_mock_utilization_analysis`) rather than live Mistral-7B inference.
> 
> In accordance with scientific integrity rules, those simulated values are classified as **RED / INVALID** and must not be cited as empirical findings of Mistral-7B.

### Verified Diagnostic Framework (Ready for Live Execution):
1. **Attention-Based Passage Utilization Proxy:**
   - Head-averaged query-to-passage attention mass, length-normalized by $|I(Q)| \cdot |I(D_i)|$.
2. **Leave-One-Passage-Out Importance ($\Delta_i$):**
   - Teacher-forced gold answer log-probability drop:
     $$\Delta_i = \log P(y_{\text{gold}} \mid q, D_1 \dots D_{10}) - \log P(y_{\text{gold}} \mid q, D_{\setminus \{i\}})$$
   - Preserves signed differences: $\Delta_i > 0$ indicates beneficial evidence, $\Delta_i < 0$ indicates negative interference.
3. **Concentration Metrics ($N_{\text{eff}}$, Gini):**
   - Formulated over positive contributions: $c_i = \max(\Delta_i, 0)$ with $p_i = c_i / \sum c_j$, correctly handling negative interventions without mathematical breakdown.

---

## 5. Connection to SARA Architecture

SARA represents retrieved candidates as a hybrid prompt: $k$ full-text passages and $n - k$ soft-compressed retrieval embeddings injected at `<COMPRESS>` tokens via a trained projection layer into Mistral's embedding space.

```text
       k Text Passages (High Fidelity, ~180-250 tokens/chunk)
                             +
  (n - k) Compressed Embeddings (Low Overhead, 1 token/chunk)
                             ↓
              SARA Hybrid Context Budget
```

### The Role of $k$:
- Parameter $k$ acts as a **textual fidelity allocation knob**.
- High $k$ maximizes textual detail for complex questions requiring multi-paragraph synthesis, but increases quadratic self-attention latency and prompt length.
- Low $k$ minimizes latency and token consumption, but may omit essential evidence for multi-part questions.
- In the original SARA implementation, **$k$ is a static, globally fixed value** applied to every query uniformly.

---

## 6. What Week 0 Establishes

1. **Evidence Demand is Heterogeneous:** In scientific QA, evidence requirements vary by query (from 1 paragraph for $67.1\%$ of queries to 2+ paragraphs across multiple sections for $32.9\%$).
2. **Retrieval Depth Displays Asymmetric Trade-offs:** BM25 `Recall_any@k` reaches $80.95\%$ at $k=6$, but complete evidence (`Recall_all@k`) reaches only $70.56\%$ at $k=6$ and $80.09\%$ at $k=10$.
3. **Prompt Formatting Imposes Strict Causal Asymmetry:** Under causal masking, passage representations are question-agnostic; query tokens contextualize against passages in the query-to-passage direction.
4. **Motivation for Query-Conditioned Context Allocation:** The structural divergence between single-paragraph queries and multi-paragraph queries motivates testing whether adaptive context allocation can outperform fixed textual budgets.

---

## 7. What Week 0 Does NOT Establish

1. **Week 0 does NOT prove that adaptive $k$ outperforms fixed $k$:** Week 0 provides descriptive dataset characterization; it does not measure downstream answer generation quality under an adaptive policy.
2. **Attention proxy is NOT causal proof of sufficiency:** Internal query-to-passage attention indicates geometric information flow, not necessary or sufficient evidence.
3. **Does not prove optimal $k$ for individual queries:** Determining the optimal pre-generation budget requires empirical generation sweeps across $k \in \{0, 2, 4, 5, 6, 8, 10\}$.

---

## 8. Verified Reproducibility & Code Artifacts

- **Scientific Audit Report:** [`results/week0/scientific_audit.md`](file:///home/gunavenkat/Downloads/CS787-RAG-Project/results/week0/scientific_audit.md)
- **Configuration:** [`configs/week0/week0.yaml`](file:///home/gunavenkat/Downloads/CS787-RAG-Project/configs/week0/week0.yaml)
- **Source Code:** [`src/week0/`](file:///home/gunavenkat/Downloads/CS787-RAG-Project/src/week0/)
- **Unit Tests:** [`tests/week0/test_week0.py`](file:///home/gunavenkat/Downloads/CS787-RAG-Project/tests/week0/test_week0.py) (13/13 passed)
- **Token Span Map Artifact:** [`results/week0/layer_analysis/token_span_map_example.json`](file:///home/gunavenkat/Downloads/CS787-RAG-Project/results/week0/layer_analysis/token_span_map_example.json)
- **Part A Summary Tables:** [`results/week0/tables/dataset_summary.csv`](file:///home/gunavenkat/Downloads/CS787-RAG-Project/results/week0/tables/dataset_summary.csv)
- **Part A Retrieval Statistics:** [`results/week0/tables/retrieval_statistics.csv`](file:///home/gunavenkat/Downloads/CS787-RAG-Project/results/week0/tables/retrieval_statistics.csv)
