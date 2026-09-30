# Blinded Human Audit Report (N=50 Development Questions)

**Benchmark:** QASPER Development Split  
**Status:** **PENDING_MANUAL_EVALUATION**  
**Automatic Oracle Analysis:** **COMPLETE**  
**Human Evaluation Artifact:** `results/week3/human_audit/human_scoring_template.csv`  

---

## 1. Audit Protocol & Design

To validate whether automatic Token F1 reliably correlates with qualitative answer correctness across evidence budgets, a blinded triple-condition human audit has been prepared:
- **Questions:** 50 development questions selected via deterministic stratified sampling across optimal budget strata ($k^*_{\text{qual}}$).
- **Conditions per Question:** $k \in \{2, 5, 10\}$ (150 evaluations total).
- **Blinding Standard:** The presentation order of $k=2$, $k=5$, and $k=10$ is randomized per question under anonymous identifiers (**System A**, **System B**, **System C**). The true budget mapping is stored strictly in `results/week3/human_audit/blinding_map.json` and hidden from the evaluation sheet.
- **Evaluator Rubric:**
  - `0 = Incorrect`: The answer is factually wrong, ungrounded, or fails to address the question.
  - `1 = Partially correct`: The answer contains relevant factual information but is incomplete, imprecise, or contains minor inaccuracies.
  - `2 = Fully correct`: The answer fully, accurately, and concisely answers the scientific question.

---

## 2. Evaluation Status & Scientific Guardrail

- **Current Status:** The evaluation template has been generated with all 150 anonymous answers and question texts ready for manual human evaluation.
- **Scientific Guardrail:** In accordance with project research integrity guidelines, **no synthetic, heuristic, or LLM-judged scores are substituted**. The correlation with Token F1 will be computed and reported only when actual human scores are manually entered into `human_scoring_template.csv`.
- **Downstream Impact:** Automatic oracle and epsilon analyses in Week 3 are fully valid and complete on the frozen Week-2 matrix. The human evaluation remains a secondary qualitative verification.
