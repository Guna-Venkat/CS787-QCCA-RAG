# Week 0: Dataset Characterization & Evidence Analysis Report

**Dataset:** QASPER (*Question Answering on Scientific Papers*)  
**Splits Analyzed:** Official Train Split ($n=2,088$ queries across 772 papers) and Development Split ($n=231$ queries across 86 papers)  
**Governance:** Paper-disjoint partition (`Baselines/SARA-main/data/manifests/qasper_split.json`). Official test split ($n=1,309$) remains strictly frozen and unaccessed.  
**Artifact Directory:** `results/week0/dataset_analysis/`  

---

## 1. Executive Summary

This report establishes the baseline dataset characterization of QASPER within the SARA experimental pipeline. The analysis is conducted using the modular `QASPERAdapter` and `analyze_dataset` engine, reporting empirical distributions across document structure, observable query characteristics, gold evidence annotations, and BM25 retrieval dynamics.

### Key Empirical Findings:
1. **Document-Level Scale:**
   - Papers average **3,694 words in Train** and **4,228 words in Dev** (median $\approx 3,617$ words, corresponding to $\approx 4,700\text{--}5,500$ tokens).
   - This far exceeds the typical dense context window of Standard RAG without compression, confirming that scientific QA inherently demands selective or compressed context handling.
2. **Evidence Sparsity & Local Concentration:**
   - On average, queries require only **1.64 gold evidence paragraphs in Dev** and **1.78 in Train**.
   - Evidence is not uniformly distributed throughout the paper: average relative depth is **0.42** (clustered primarily in the Methodology and Experimental sections, with few references in the Introduction or Discussion).
3. **Retrieval Plateau Dynamic (BM25 Recall@k):**
   - At $k=2$, BM25 captures only **51.52%** of queries with gold evidence.
   - At $k=4$, recall jumps to **70.13%** ($+18.61\%$).
   - At $k=6$, recall reaches **80.95%** ($+10.82\%$).
   - At $k=10$, recall plateaus at **87.01%** (expanding from $k=6 \rightarrow 10$ yields only $+6.06\%$ additional recall for a $66\%$ increase in evidence budget).
   - This non-linear diminishing return curve provides strong empirical justification for studying intermediate evidence budgets ($k \in [4, 6]$).

---

## 2. Dataset Structure: Train vs. Dev

All metrics are computed separately for the 772 training papers and 86 development papers to verify that the document-level split preserves structural balance without paper leakage.

| Metric | Train Split ($n=2,088$) | Dev Split ($n=231$) | Stability Assessment |
|---|---|---|---|
| **Papers Count** | 772 | 86 | Exact 90/10 document partition |
| **Questions Count** | 2,088 | 231 | Exact 90/10 question partition |
| **Questions per Paper (Mean $\pm$ Std)** | 2.70 $\pm$ 1.84 | 2.69 $\pm$ 1.76 | Identical density |
| **Questions per Paper (Median [IQR])** | 2.0 [1, 4] | 2.0 [1, 4] | Identical median |
| **Document Length (Words, Mean)** | 3,694.3 | 4,228.1 | Dev slightly longer on average |
| **Document Length (Words, Median)** | 3,526.5 | 3,617.5 | Well matched |
| **Estimated Document Tokens (Mean)** | 4,802.6 | 5,496.5 | Requires context selection |
| **Question Length (Words, Mean)** | 8.39 | 8.39 | Identical |
| **Primary Answer Length (Words, Mean)** | 13.65 | 14.45 | Concise phrase/sentence answers |
| **Unanswerable Questions (%)** | 0.29% | 0.00% | Negligible in filtered splits |
| **Extractive Answer Questions (%)** | 56.47% | 57.58% | Balanced across splits |
| **Yes / No Questions (%)** | 17.19% | 16.45% | Balanced across splits |
| **Free-form / Abstractive (%)** | 26.05% | 25.97% | Balanced across splits |

![Dataset Distributions](file:///home/gunavenkat/Downloads/CS787-RAG-Project/results/week0/figures/dataset_distributions.png)

---

## 3. Observable Query Characteristics

To understand query diversity without making unfalsifiable claims about "inherent complexity," we extract observable surface and grammatical properties:

| Query Feature | Train Frequency | Dev Frequency | Heuristic Definition |
|---|---|---|---|
| **What Questions** | 52.4% | 53.2% | Prefix "What" / "What's" |
| **How Questions** | 18.2% | 19.5% | Prefix "How" (process/mechanism) |
| **How Many / How Much** | 6.8% | 6.1% | Prefix "How many" (quantitative) |
| **Which Questions** | 8.4% | 8.2% | Prefix "Which" (selective set) |
| **Why Questions** | 4.1% | 3.9% | Prefix "Why" (causal explanation) |
| **Boolean (Is / Does / Can / Did)** | 16.5% | 16.0% | Leading auxiliary verb |
| **Contains Numerical Term** | 21.4% | 22.1% | Digit or number word |
| **Comparison Query** | 9.8% | 10.4% | Terms: *compare, versus, differ, baseline, outperform* |
| **Multi-part / Conjunction** | 24.3% | 25.1% | Conjunctions: *and, or, whereas, while, both* |
| **Lexical Diversity (TTR)** | 0.94 $\pm$ 0.08 | 0.95 $\pm$ 0.07 | Type-Token Ratio |

**Interpretation:**
- Over half of all queries are targeted entity/fact inquiries ("What").
- Multi-part queries (25%) and numerical inquiries (22%) represent substantial minorities that likely demand evidence aggregation from multiple document regions.

---

## 4. Gold Evidence Properties

QASPER provides paragraph-level evidence annotations verified by multiple independent NLP annotators.

| Evidence Property | Train Value | Dev Value | Scientific Interpretation |
|---|---|---|---|
| **Evidence Paragraphs / Query (Mean)** | 1.78 | 1.64 | Highly concentrated |
| **Single-Paragraph Evidence (%)** | 64.2% | 67.1% | 2/3 of queries need exactly 1 chunk |
| **Two-Paragraph Evidence (%)** | 23.5% | 22.1% | Multi-hop / joint verification |
| **Three+ Paragraph Evidence (%)** | 12.3% | 10.8% | Complex multi-evidence questions |
| **Evidence Relative Depth (Mean)** | 0.43 $\pm$ 0.24 | 0.42 $\pm$ 0.23 | Centered in mid-paper (Methods/Results) |
| **Evidence Redundancy (Pairwise Jaccard)** | 0.0363 | 0.0306 | Multiple evidence chunks are non-redundant |
| **Query-Evidence Lexical Jaccard** | 0.0682 | 0.0664 | Low raw surface overlap |
| **Query-Evidence Content Coverage** | 0.3773 | 0.3725 | Moderate content-word overlap (~37%) |

![Evidence Characteristics](file:///home/gunavenkat/Downloads/CS787-RAG-Project/results/week0/figures/evidence_characteristics.png)

---

## 5. BM25 Retrieval Diagnostics (Development Split)

Using the precomputed BM25 retrieval matrix (`results/week2/raw/dev_retrieval.jsonl`), we evaluate retrieval effectiveness against annotator gold evidence under two distinct definitions:
- **`Recall_any@k`:** At least one annotated gold evidence paragraph is retrieved in the top $k$ chunks.
- **`Recall_all@k`:** All annotated gold evidence paragraphs are retrieved in the top $k$ chunks.
- **`Mean_Coverage@k`:** Mean fraction of annotated gold evidence paragraphs retrieved in the top $k$ chunks.

| Retrieval Depth ($k$) | `Recall_any@k` (%) | `Recall_all@k` (%) | `Mean_Coverage@k` (%) | Implications for Evidence Budget $k$ |
|---|---|---|---|---|
| **$k=1$** | 33.33% (77/231) | 22.94% (53/231) | 28.99% | Captures any evidence for only 1/3 of queries |
| **$k=2$** | 51.52% (119/231) | 37.23% (86/231) | 45.83% | Fixed budget $k=2$ misses complete evidence for 62.8% of queries |
| **$k=4$** | 70.13% (162/231) | 55.84% (129/231) | 66.52% | Strong jump (+18.6% any, +18.6% all): captures complete evidence for over half |
| **$k=6$** | 80.95% (187/231) | 70.56% (163/231) | 79.85% | Captures $> 80\%$ any evidence and $> 70\%$ complete evidence |
| **$k=8$** | 84.85% (196/231) | 77.06% (178/231) | 85.53% | Marginal gain (+3.9% any, +6.5% all) |
| **$k=10$** | 87.01% (201/231) | 80.09% (185/231) | 88.27% | Ceiling for top-10 retrieval candidate pool |

- **Mean Rank of First Gold Passage:** 2.58 (when present in top-10, first gold evidence ranks near rank 2–3).
- **Top-1 / Top-2 Margin ($\Delta_{12}$):** 0.2159 (moderate retrieval score margin).
- **Normalized Score Entropy ($H_{\text{norm}}$):** 0.9250 (high score dispersion across the top-10 candidates).

![Retrieval Diagnostics](file:///home/gunavenkat/Downloads/CS787-RAG-Project/results/week0/figures/retrieval_diagnostics.png)

---

## 6. Synthesis: Distinguishing Observation from Hypothesis

1. **Observed Fact:**
   - QASPER documents are long ($\approx 4,000$ words), while gold evidence is sparse ($1.64$ paragraphs per query).
   - Approximately 67% of development questions contain a single annotated evidence paragraph, while BM25 recall continues to increase with retrieval depth, illustrating heterogeneity in retrieval/evidence structure.
2. **Heuristic Finding:**
   - Queries with multi-part conjunctions and numerical indicators exhibit higher evidence dispersion across the document.
3. **Hypothesis for Downstream Modeling:**
   - A globally fixed, small evidence budget (e.g. $k=2$) misses complete evidence in $62.8\%$ of queries.
   - Conversely, a globally maximal evidence budget (e.g. $k=10$) introduces extra context and token overhead for the $67\%$ of queries whose evidence is fully satisfied by 1 paragraph.
   - This motivates investigating adaptive evidence allocation in subsequent experiments.
