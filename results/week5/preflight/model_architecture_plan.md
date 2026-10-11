# Week 5 Preflight: Semantic Encoder Model Architecture Plan

**Date:** October 2026  
**Auditor:** SARA Research Engineering Agent  
**Artifact:** `results/week5/preflight/model_architecture_plan.md`  

---

## 1. Core Design Philosophy: Information, Not Parameter Scale

The central scientific question of Week 5 is:
> **"Does directly reading retrieved passage natural-language text provide allocation information that handcrafted query/retrieval features miss?"**

This question tests **information availability**, not extreme model capacity. Therefore:
- **No Mistral-7B / 7B-parameter generation models** for allocation classification.
- **No LoRA / QLoRA fine-tuning** on generation checkpoints (scheduled for Week 6).
- **Architecture is deliberately modest, fast, memory-safe, and fair** across all four information ablations.

---

## 2. Architectural Candidates Evaluated

### Candidate 1: Per-Passage Cross-Encoder + Shallow Transformer Aggregator (RECOMMENDED DEFAULT)
- **Passage Representation:**
  For each passage $i \in \{1 \dots 10\}$, format pair:
  $$\mathbf{u}_i = \text{Encoder}([\text{CLS}] \circ q \circ [\text{SEP}] \circ D_i \circ [\text{SEP}])$$
  Backbone: `sentence-transformers/all-MiniLM-L6-v2` or `BAAI/bge-small-en-v1.5` ($d=384$, ~22M–33M parameters).
- **Passage-Level Aggregation:**
  Add learnable passage-rank embeddings $\mathbf{r}_i \in \mathbb{R}^{d}$ for rank $i \in \{1 \dots 10\}$:
  $$\mathbf{z}_i = \mathbf{u}_i + \mathbf{r}_i$$
  Pass the sequence $[\mathbf{z}_1, \dots, \mathbf{z}_{10}]$ through a 2-layer Transformer Encoder (hidden size $d=384$, 4 heads, feedforward 1024, ~1.5M params).
  Pool across the 10 passage positions via attention-pooling or `[CLS]` token to obtain candidate context state $\mathbf{c} \in \mathbb{R}^{d}$.
- **Feature Fusion:**
  If W4 features are present (Conditions B and D):
  $$\mathbf{h}_{\text{final}} = [\mathbf{c} \,\|\, \text{LayerNorm}(\mathbf{W}_f \mathbf{x}_{\text{W4}})]$$
  If W4 features absent (Conditions A and C):
  $$\mathbf{h}_{\text{final}} = \mathbf{c}$$
- **Prediction Heads:**
  - Direct $k$ Classifier (T1): Linear layer to 5 classes ($k \in \{2, 4, 5, 6, 8\}$).
  - Regret Predictor (T2): Multi-output MLP predicting $[\widehat{\text{regret}}_2, \dots, \widehat{\text{regret}}_8]$.
- **Pros:**
  - Strict pair-level cross-attention models exact query-passage relevance.
  - Scales gracefully: 10 parallel encoder forward passes per query (batch size 10 on GPU takes <15 ms).
  - Memory footprint < 1.2 GB VRAM.
  - Exactly 23M–35M parameters.

---

### Candidate 2: Frozen Bi-Encoder + Deep Interaction MLP
- **Passage Representation:**
  Use precomputed query embedding $\mathbf{e}_q$ and passage embeddings $\mathbf{e}_1 \dots \mathbf{e}_{10}$ (e.g. from SFR-Embedding-Mistral or BGE).
- **Passage-Level Aggregation:**
  Pass concatenated embeddings $[\mathbf{e}_q, \mathbf{e}_1, \dots, \mathbf{e}_{10}]$ through an MLP or 1D convolution.
- **Pros:** Ultra-fast, zero GPU training needed.
- **Cons:** Cannot model lexical cross-terms or exact question-chunk interactions directly; largely duplicates Week 4 Family C features without providing a true "content reading" test.

---

### Candidate 3: Concatenated Full-Prompt ModernBERT-Base
- **Passage Representation:**
  Pack all 10 passages into a single sequence: `[CLS] q [SEP] Doc 1: D1 ... Doc 10: D10 [SEP]`.
  Feed into an 8,192-token context encoder (`ModernBERT-base`, ~149M parameters).
- **Pros:** Full cross-attention across all passages simultaneously.
- **Cons:** High memory requirements, quadratic self-attention cost over 2,500+ tokens, risks overfitting on small sample sizes ($N=2,088$ train queries).

---

## 3. Selected Primary Architecture

**Candidate 1 (MiniLM / BGE Cross-Encoder + Shallow Transformer Aggregator)** is selected as the primary Week 5 architecture.

### Fairness Invariant Across Information Conditions:
- **I1 (Query Only):** Passage inputs are masked / dummy; encoder processes $q$ only $\rightarrow$ single token $\mathbf{c} = \text{Encoder}(q) \rightarrow \text{Head}$.
- **I2 (Query + Features):** $\mathbf{c} = \text{Encoder}(q)$, fused with $\mathbf{x}_{\text{W4}} \rightarrow \text{Head}$.
- **I3 (Query + Passages):** Full 10-passage Cross-Encoder + Transformer Aggregator $\rightarrow \mathbf{c} \rightarrow \text{Head}$.
- **I4 (Query + Passages + Features):** Full 10-passage Cross-Encoder + Transformer Aggregator $\rightarrow \mathbf{c} \oplus \mathbf{x}_{\text{W4}} \rightarrow \text{Head}$.

By keeping the encoder backbone and head dimension identical, any performance difference between I1, I2, I3, and I4 strictly reflects the **information content** of the input representation.
