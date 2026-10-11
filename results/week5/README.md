# Week 5: Content-Aware Adaptive Representation Allocation

This directory contains the complete three-phase Week 5 research program investigating whether direct semantic access to retrieved passage content improves evidence allocation over query-only and handcrafted retrieval signals.

---

## Program Structure

### 1. W5-A — Initial Semantic Allocator Formulations & Collapse Diagnostics
* **Path:** `results/week5/content_allocator/`
* **Focus:** Initial $I_1 \dots I_4 \times T_1 \dots T_2$ evaluation.
* **Findings:** Identified severe policy collapse toward static $k=8$ ($I_3$-T1, $I_3$-T2), traced to an unnormalized rank-embedding initialization defect (rank norm $\approx 19.5$ completely swamping MiniLM semantic norm $\approx 2.12$).

### 2. W5-B — Corrected Multi-Label Admissibility ($T_3$) with Frozen MiniLM
* **Path:** `results/week5/content_allocator/w5b/`
* **Report:** [`week5b_report.md`](file:///home/gunavenkat/Downloads/CS787-RAG-Project/results/week5/content_allocator/w5b/week5b_report.md)
* **Focus:** Target $T_3$ ($z_k(q) = 1[\text{regret}(q, k) \le 0.01]$), rank embedding initialization $\mathcal{N}(0, 0.02^2)$, pre-attention LayerNorm.
* **Findings:** Completely eliminated scale defects; proved direct passage content contains allocation signal ($I_3$ beats $I_1$ by $+0.0087$ $F_1$); matched $W_4$ quality ($F_1 = 0.3810$ vs $0.3842$) but required $+142.3$–$149.5$ more context tokens under a frozen sentence encoder.

### 3. W5-C — Semantic Capacity & Encoder Adaptation Ladder
* **Path:** `results/week5/content_allocator/w5c_encoder_adaptation/`
* **Report:** [`week5c_report.md`](file:///home/gunavenkat/Downloads/CS787-RAG-Project/results/week5/content_allocator/w5c_encoder_adaptation/week5c_report.md)
* **Notebook:** [`notebooks/week5/week5c_encoder_adaptation.ipynb`](file:///home/gunavenkat/Downloads/CS787-RAG-Project/notebooks/week5/week5c_encoder_adaptation.ipynb)
* **Config:** [`configs/week5/w5c_encoder_adaptation.yaml`](file:///home/gunavenkat/Downloads/CS787-RAG-Project/configs/week5/w5c_encoder_adaptation.yaml)
* **Focus:** Evaluated whether task-specific adaptation of the base semantic encoder unlocks superior evidence allocation:
  - **C0:** Frozen MiniLM ($F_1 = 0.3810$, tokens $= 1258.2$)
  - **C1:** Last-1 Transformer block trainable ($F_1 = 0.3620$, tokens $= 1158.9$)
  - **C2:** Last-2 Transformer blocks trainable ($F_1 = 0.3743$, tokens $= 1337.6$)
* **Findings:** Shallow adaptation of 1–2 layers suffered negative transfer/drift on the QA training set; frozen $C_0$ remained strongest among semantic variants, but neither beat $W_4$ Balanced Logistic Regression ($F_1 = 0.3842$, tokens $= 1107.9$).

---

## Final Week 5 Scientific Conclusion

> **Direct retrieved-passage content contains genuine evidence allocation signal, but neither frozen lightweight semantic encoding nor shallow task-specific MiniLM adaptation surpasses the strong handcrafted Week-4 allocator on the quality–context frontier. Small encoder capacity constitutes a fundamental barrier to evidence sufficiency prediction.**
