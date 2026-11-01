# Week 10 — Summary Analysis

## Objective

Diagnose the retrieval, reader and evidence-interface contributions behind historical RAG/SARA failures without another TEST-driven tuning cycle.

## Experiments performed

| Part            | Question                                                                                  | Status             | Main_result                                                                                                                                                                                                                  |
|:----------------|:------------------------------------------------------------------------------------------|:-------------------|:-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| AUDIT-PREFLIGHT | Can historical source/interface identities be made explicit before interpreting failures? | SMOKE-TESTED       | The verification artifact records27 unit tests and a synthetic/schema smoke; it explicitly says full experiments had not started at that earlier verification point.                                                         |
| R               | Was the semantic retrieval foundation limiting downstream quality?                        | COMPLETED          | At retrieval k10, SFR any/all recall is0.727273/0.522727 versus MiniLM0.636364/0.445455 and BM250.618182/0.431818. SFR/full k6 F10.419880 and SFR/SARA k6 0.421290 are strong points.                                        |
| C               | Does shortening evidence improve a fine-tuned reader relative to full context?            | COMPLETED          | Fine-tuned C-Mistral full F1 is0.407095, SARA_k40.381233, query-selected2560.344075 and lead2560.321490. Full versus SARA interval crosses zero; full exceeds compressed-text modes in these paired C comparisons.           |
| G               | Do evidence-mode results depend on the reader under the public-text interface?            | COMPLETED          | All three G readers are complete on231 queries per action. Full-context F1 is Mistral0.235322, Qwen0.273760, Llama0.289647; qcs256 scores are0.189431/0.211520/0.197178.                                                     |
| D               | Are observed effects stable under the preregistered perturbation/decoding diagnostic?     | PARTIALLY EXECUTED | Full-study metadata records144 D diagnostic generation rows, distinct from the2079-row complete G panel. D_partial_panel_provenance records scope restrictions; do not present it as a full repeated three-reader benchmark. |
| ORACLE-REPORT   | How much headroom remains when choosing observed actions retrospectively?                 | COMPLETED          | C-Mistral exact retrospective oracle F1 is0.525774 versus best fixed full0.407095. This headroom does not establish a learnable policy or a sole bottleneck.                                                                 |

## Key results

### sara configuration crosswalk

| component               | official paper                                                         | official repo                                              | local implementation                                                  | W0-W9 usage                  | exact match?                                   | potentially material?   | evidence/source                                                               | notes                                                                                        |
|:------------------------|:-----------------------------------------------------------------------|:-----------------------------------------------------------|:----------------------------------------------------------------------|:-----------------------------|:-----------------------------------------------|:------------------------|:------------------------------------------------------------------------------|:---------------------------------------------------------------------------------------------|
| Default retriever       | BM25 plus dense experiments                                            | BM25 default                                               | paper-local BM25                                                      | W2/W5 top10                  | family only                                    | yes                     | https://github.com/Ahren09/SARA; src/week2/precompute_dev_embeddings.py       | Tokenization/stemming implementations not proven equivalent                                  |
| QASPER retrievers       | Table12 SFR/BGE/BM25; section3.4 also names multi-qa-mpnet-base-cos-v1 | BM25 pipeline helper                                       | MiniLM ranking/routing                                                | W6-W9                        | no                                             | yes                     | https://arxiv.org/pdf/2507.05633 Table12                                      | Section3.1 names bge-reranker-v2-m3: cross-encoder ambiguity; exact BGE corpus not specified |
| QASPER SFR              | F1=55.44; ROUGE-L=52.93                                                | SFR component                                              | No completed equivalent full-paper SFR retrieval experiment located   | SFR compression/features     | not established                                | yes                     | Paper Table12; src/week2/precompute_dev_embeddings.py                         | Paper scores use different protocol; not expected project scores                             |
| QASPER BGE              | F1=44.47; ROUGE-L=45.24                                                | Dense config inventory                                     | not canonical                                                         | MiniLM is different          | no                                             | yes                     | Paper Table12                                                                 | Do not substitute an arbitrary BGE bi-encoder                                                |
| QASPER BM25             | F1=36.15; ROUGE-L=39.54                                                | BM25                                                       | BM25                                                                  | historical foundation        | family only                                    | yes                     | Paper Table12                                                                 | Not directly comparable with local TEST F1                                                   |
| Compressor encoders     | SFR; Linq-Embed-Mistral; gte-Qwen2-7B-instruct; stella_en_1.5B_v5      | SFR default                                                | SFR cached4096-dimensional vectors                                    | W2 onward                    | family yes                                     | yes                     | Paper Table6/AppendixB.6                                                      | Encoder ablations are separate from retriever ablations                                      |
| Base model              | Mistral plus Llama/Gemma families                                      | Mistral-7B-Instruct-v0.2                                   | same public base, locally trained adapter/projector                   | W0-W9                        | architecture yes; exact paper weights unproven | yes                     | local checkpoint args.json; official README                                   | Hashing local weights establishes identity, not equivalence to paper checkpoint              |
| Query conditioning      | document encoder query-agnostic; query used by refinement              | cached document embedding projector                        | query-agnostic SFR document vector                                    | W2-W9                        | mechanism yes                                  | yes                     | Paper method; src/week2/precompute_dev_embeddings.py                          | Query-aware routing does not make document latent query-conditioned                          |
| Refinement/presentation | iterative relevance/novelty/CSI                                        | current simplified two-method CLI and BM25 helper          | MiniLM subset restored original BM25 rank                             | W6 routing/W9 interventions  | not full paper algorithm                       | yes                     | src/week6/routing.py; paper Algorithm1                                        | Paper and current public CLI expose different workflows                                      |
| Input source handling   | evaluation setup                                                       | rank_documents list branch joins then wraps Document again | list strings mapped separately to Document                            | local canonical pipeline     | no                                             | potentially             | audit/sources/sara_data.py:27; Baselines/SARA-main/src/utils/data_utils.py:27 | Upstream current list branch appears inconsistent; not executed                              |
| Precision               | paper hardware setting                                                 | BF16 default SDPA                                          | historical BF16; later FP16 efficientSDPA                             | W9 revisions vs first passes | not bitwise                                    | potentially             | src/week9/backend.py; src/week9/final_h05/historical_backend.py               | W10 fixes FP16; preserve historical artifacts                                                |
| MiniLM origin           | not a Table12 retriever                                                | not default                                                | W5 cheap pooled semantic allocator encoder; W6 zero-shot route scorer | 97657ca then2133292          | project extension                              | yes                     | git log -S all-MiniLM-L6-v2; src/week5/models.py; src/week6/models.py         | No located prerequisite matched SFR retrieval comparison                                     |

Source: [results/week10/root_cause/v1/sara_configuration_crosswalk.csv](../../week10/root_cause/v1/sara_configuration_crosswalk.csv).

### All matched retrieval-depth conditions

|   all_recall |   any_recall |   candidate_diversity |   coverage |   duplicate_rate |   k |      mrr |   n_queries |   retrieved_tokens | retriever   |   unmapped_gold_spans |
|-------------:|-------------:|----------------------:|-----------:|-----------------:|----:|---------:|------------:|-------------------:|:------------|----------------------:|
|     0.095455 |     0.127273 |            nan        |   0.127608 |         0.000000 |   1 | 0.269185 |         231 |         200.783550 | BM25        |              0.155844 |
|     0.150000 |     0.209091 |              0.850845 |   0.215886 |         0.000000 |   2 | 0.269185 |         231 |         413.852814 | BM25        |              0.155844 |
|     0.254545 |     0.386364 |              0.862657 |   0.378124 |         0.000000 |   4 | 0.269185 |         231 |         832.277056 | BM25        |              0.155844 |
|     0.345455 |     0.504545 |              0.868114 |   0.509237 |         0.000000 |   6 | 0.269185 |         231 |        1242.727273 | BM25        |              0.155844 |
|     0.395455 |     0.568182 |              0.871672 |   0.586772 |         0.000000 |   8 | 0.269185 |         231 |        1651.398268 | BM25        |              0.155844 |
|     0.431818 |     0.618182 |              0.873878 |   0.650002 |         0.000000 |  10 | 0.269185 |         231 |        2056.406926 | BM25        |              0.155844 |
|     0.072727 |     0.145455 |            nan        |   0.124316 |         0.000000 |   1 | 0.293698 |         231 |         172.943723 | MiniLM      |              0.155844 |
|     0.172727 |     0.272727 |              0.871296 |   0.259924 |         0.000000 |   2 | 0.293698 |         231 |         366.424242 | MiniLM      |              0.155844 |
|     0.263636 |     0.418182 |              0.876615 |   0.406129 |         0.000000 |   4 | 0.293698 |         231 |         755.346320 | MiniLM      |              0.155844 |
|     0.322727 |     0.500000 |              0.878391 |   0.495579 |         0.000000 |   6 | 0.293698 |         231 |        1136.623377 | MiniLM      |              0.155844 |
|     0.409091 |     0.581818 |              0.880485 |   0.593982 |         0.000000 |   8 | 0.293698 |         231 |        1523.688312 | MiniLM      |              0.155844 |
|     0.445455 |     0.636364 |              0.881101 |   0.667897 |         0.000000 |  10 | 0.293698 |         231 |        1909.523810 | MiniLM      |              0.155844 |

Source: [results/week10/root_cause/v1/retrieval_summary.csv](../../week10/root_cause/v1/retrieval_summary.csv). First 12 of 18 rows shown; detailed report and source preserve the full comparison.

### Matched retrieval/interface answer scores

| run           | action         |   n_queries |       f1 |   exact_match |   mean_prompt_tokens |   mean_generated_tokens |
|:--------------|:---------------|------------:|---------:|--------------:|---------------------:|------------------------:|
| R-BM25-full   | BM25/full/k2   |         231 | 0.299486 |      0.164502 |           495.571429 |               14.419913 |
| R-BM25-full   | BM25/full/k6   |         231 | 0.383935 |      0.186147 |          1348.008658 |               15.748918 |
| R-BM25-full   | BM25/full/k8   |         231 | 0.377042 |      0.186147 |          1768.398268 |               17.489177 |
| R-BM25-full   | BM25/full/k4   |         231 | 0.363272 |      0.194805 |           925.770563 |               15.567100 |
| R-BM25-sara   | BM25/sara/k2   |         231 | 0.297712 |      0.177489 |           545.051948 |               20.341991 |
| R-BM25-sara   | BM25/sara/k6   |         231 | 0.375524 |      0.190476 |          1377.575758 |               17.411255 |
| R-BM25-sara   | BM25/sara/k8   |         231 | 0.382535 |      0.177489 |          1787.943723 |               16.173160 |
| R-BM25-sara   | BM25/sara/k4   |         231 | 0.340516 |      0.186147 |           965.294372 |               15.718615 |
| R-MiniLM-full | MiniLM/full/k2 |         231 | 0.347328 |      0.190476 |           448.134199 |               15.757576 |
| R-MiniLM-full | MiniLM/full/k6 |         231 | 0.389471 |      0.194805 |          1241.956710 |               17.787879 |
| R-MiniLM-full | MiniLM/full/k8 |         231 | 0.397290 |      0.203463 |          1640.761905 |               17.956710 |
| R-MiniLM-full | MiniLM/full/k4 |         231 | 0.382600 |      0.199134 |           848.891775 |               15.523810 |

Source: [results/week10/root_cause/v1/summary.csv](../../week10/root_cause/v1/summary.csv). First 12 of 24 rows shown; detailed report and source preserve the full comparison.

### Fine-tuned Mistral evidence modes

| run       | action      |   n_queries |       f1 |   exact_match |   mean_prompt_tokens |   mean_generated_tokens |
|:----------|:------------|------------:|---------:|--------------:|---------------------:|------------------------:|
| C-Mistral | sara_k4     |         231 | 0.381233 |      0.199134 |           863.645022 |               18.086580 |
| C-Mistral | lead256     |         231 | 0.321490 |      0.168831 |           338.987013 |               16.974026 |
| C-Mistral | no_evidence |         231 | 0.154569 |      0.103896 |            70.805195 |               27.086580 |
| C-Mistral | qcs256      |         231 | 0.344075 |      0.199134 |           339.155844 |               12.173160 |
| C-Mistral | full        |         231 | 0.407095 |      0.194805 |          1957.636364 |               17.541126 |

Source: [results/week10/root_cause/v1/summary.csv](../../week10/root_cause/v1/summary.csv).

### Matched C contrasts

| a                     | b                     |   ci_high |    ci_low |     point |
|:----------------------|:----------------------|----------:|----------:|----------:|
| C-Mistral|sara_k4     | C-Mistral|lead256     |  0.100071 |  0.023821 |  0.059743 |
| C-Mistral|sara_k4     | C-Mistral|no_evidence |  0.289511 |  0.165429 |  0.226664 |
| C-Mistral|sara_k4     | C-Mistral|qcs256      |  0.086709 | -0.011186 |  0.037158 |
| C-Mistral|sara_k4     | C-Mistral|full        |  0.012674 | -0.065663 | -0.025862 |
| C-Mistral|lead256     | C-Mistral|full        | -0.042237 | -0.129921 | -0.085605 |
| C-Mistral|no_evidence | C-Mistral|lead256     | -0.111531 | -0.223883 | -0.166921 |
| C-Mistral|no_evidence | C-Mistral|full        | -0.195202 | -0.310302 | -0.252526 |
| C-Mistral|qcs256      | C-Mistral|lead256     |  0.063554 | -0.018719 |  0.022585 |
| C-Mistral|qcs256      | C-Mistral|no_evidence |  0.253808 |  0.127506 |  0.189506 |
| C-Mistral|qcs256      | C-Mistral|full        | -0.019431 | -0.109377 | -0.063020 |

Source: [results/week10/root_cause/v1/bootstrap.csv](../../week10/root_cause/v1/bootstrap.csv).

### full reader panel summary

| reader   | action      | status   |   n_queries |       f1 |   exact_match |   net_evidence_utility |     HELP |     HURT |   mean_prompt_tokens |   mean_generated_tokens |   generation_cap_hit_rate |   mean_generation_seconds |   lm_calls |   measured_end_to_end_seconds |
|:---------|:------------|:---------|------------:|---------:|--------------:|-----------------------:|---------:|---------:|---------------------:|------------------------:|--------------------------:|--------------------------:|-----------:|------------------------------:|
| Mistral  | no_evidence | COMPLETE |         231 | 0.066586 |      0.000000 |               0.000000 | 0.000000 | 0.000000 |            41.805195 |               21.991342 |                  0.000000 |                  0.665612 |          1 |                           nan |
| Mistral  | full        | COMPLETE |         231 | 0.235322 |      0.004329 |               0.168735 | 0.619048 | 0.129870 |          1928.636364 |               53.207792 |                  0.025974 |                  2.222297 |          1 |                           nan |
| Mistral  | qcs256      | COMPLETE |         231 | 0.189431 |      0.004329 |               0.122845 | 0.493506 | 0.121212 |           310.155844 |               29.900433 |                  0.000000 |                  0.999506 |          1 |                           nan |
| Qwen     | no_evidence | COMPLETE |         231 | 0.073060 |      0.000000 |               0.000000 | 0.000000 | 0.000000 |            42.142857 |               17.744589 |                  0.000000 |                  0.611891 |          1 |                           nan |
| Qwen     | full        | COMPLETE |         231 | 0.273760 |      0.000000 |               0.200700 | 0.662338 | 0.047619 |          1754.246753 |               21.800866 |                  0.000000 |                  1.243445 |          1 |                           nan |
| Qwen     | qcs256      | COMPLETE |         231 | 0.211520 |      0.008658 |               0.138459 | 0.484848 | 0.112554 |           272.818182 |               19.030303 |                  0.000000 |                  0.730902 |          1 |                           nan |
| Llama    | no_evidence | COMPLETE |         231 | 0.068372 |      0.000000 |               0.000000 | 0.000000 | 0.000000 |            65.125541 |               18.623377 |                  0.000000 |                  0.594164 |          1 |                           nan |
| Llama    | full        | COMPLETE |         231 | 0.289647 |      0.034632 |               0.221275 | 0.662338 | 0.086580 |          1750.519481 |               29.645022 |                  0.000000 |                  1.416510 |          1 |                           nan |
| Llama    | qcs256      | COMPLETE |         231 | 0.197178 |      0.025974 |               0.128806 | 0.476190 | 0.121212 |           292.536797 |               18.489177 |                  0.000000 |                  0.672849 |          1 |                           nan |

Source: [results/week10/root_cause/v1/full_reader_panel_summary.csv](../../week10/root_cause/v1/full_reader_panel_summary.csv).

## Main scientific conclusions

- The verification artifact records27 unit tests and a synthetic/schema smoke; it explicitly says full experiments had not started at that earlier verification point. Source: [results/week10/root_cause/v1/execution_freeze.json](../../week10/root_cause/v1/execution_freeze.json), [results/week10/root_cause/v1/verification.json](../../week10/root_cause/v1/verification.json)
- At retrieval k10, SFR any/all recall is0.727273/0.522727 versus MiniLM0.636364/0.445455 and BM250.618182/0.431818. SFR/full k6 F10.419880 and SFR/SARA k6 0.421290 are strong points. Source: [results/week10/root_cause/v1/retrieval_summary.csv](../../week10/root_cause/v1/retrieval_summary.csv), [results/week10/root_cause/v1/summary.csv](../../week10/root_cause/v1/summary.csv)
- Fine-tuned C-Mistral full F1 is0.407095, SARA_k40.381233, query-selected2560.344075 and lead2560.321490. Full versus SARA interval crosses zero; full exceeds compressed-text modes in these paired C comparisons. Source: [results/week10/root_cause/v1/summary.csv](../../week10/root_cause/v1/summary.csv), [results/week10/root_cause/v1/freeze_C.json](../../week10/root_cause/v1/freeze_C.json)
- All three G readers are complete on231 queries per action. Full-context F1 is Mistral0.235322, Qwen0.273760, Llama0.289647; qcs256 scores are0.189431/0.211520/0.197178. Source: [results/week10/root_cause/v1/full_reader_panel_summary.csv](../../week10/root_cause/v1/full_reader_panel_summary.csv), [results/week10/root_cause/v1/full_reader_bootstrap.csv](../../week10/root_cause/v1/full_reader_bootstrap.csv)
- Full-study metadata records144 D diagnostic generation rows, distinct from the2079-row complete G panel. D_partial_panel_provenance records scope restrictions; do not present it as a full repeated three-reader benchmark. Source: [results/week10/root_cause/v1/full_study_summary.json](../../week10/root_cause/v1/full_study_summary.json), [results/week10/root_cause/v1/D_partial_panel_provenance.json](../../week10/root_cause/v1/D_partial_panel_provenance.json)
- C-Mistral exact retrospective oracle F1 is0.525774 versus best fixed full0.407095. This headroom does not establish a learnable policy or a sole bottleneck. Source: [results/week10/root_cause/v1/oracle_diagnostics.json](../../week10/root_cause/v1/oracle_diagnostics.json), [results/week10/root_cause/v1/retrieval_interface_interactions.csv](../../week10/root_cause/v1/retrieval_interface_interactions.csv)

## What did NOT work?

Compact text modes do not exceed full context here; no isolated latent-vector superiority; no sole-bottleneck inference.

## Limitations

231 historical DEV queries; repeated exploration; gold mapping gaps; public-versus-adapted interface confounds.

## Decision after this week

SFR is retained as the reviewed retrieval foundation; reader/interface isolation motivates broader multi-dataset study.

## One-paragraph week takeaway

The verification artifact records27 unit tests and a synthetic/schema smoke; it explicitly says full experiments had not started at that earlier verification point. At retrieval k10, SFR any/all recall is0.727273/0.522727 versus MiniLM0.636364/0.445455 and BM250.618182/0.431818. SFR/full k6 F10.419880 and SFR/SARA k6 0.421290 are strong points. Fine-tuned C-Mistral full F1 is0.407095, SARA_k40.381233, query-selected2560.344075 and lead2560.321490. Full versus SARA interval crosses zero; full exceeds compressed-text modes in these paired C comparisons. All three G readers are complete on231 queries per action. Full-context F1 is Mistral0.235322, Qwen0.273760, Llama0.289647; qcs256 scores are0.189431/0.211520/0.197178. Full-study metadata records144 D diagnostic generation rows, distinct from the2079-row complete G panel. D_partial_panel_provenance records scope restrictions; do not present it as a full repeated three-reader benchmark. C-Mistral exact retrospective oracle F1 is0.525774 versus best fixed full0.407095. This headroom does not establish a learnable policy or a sole bottleneck.

# Primary Sources Used

- [configs/week10/root_cause_v1.json](../../../configs/week10/root_cause_v1.json)
- [notebooks/week10/week10_root_cause.ipynb](../../../notebooks/week10/week10_root_cause.ipynb)
- [results/week10/root_cause/v1/D_partial_panel_provenance.json](../../week10/root_cause/v1/D_partial_panel_provenance.json)
- [results/week10/root_cause/v1/bootstrap.csv](../../week10/root_cause/v1/bootstrap.csv)
- [results/week10/root_cause/v1/execution_freeze.json](../../week10/root_cause/v1/execution_freeze.json)
- [results/week10/root_cause/v1/freeze_C.json](../../week10/root_cause/v1/freeze_C.json)
- [results/week10/root_cause/v1/freeze_G.json](../../week10/root_cause/v1/freeze_G.json)
- [results/week10/root_cause/v1/freeze_R.json](../../week10/root_cause/v1/freeze_R.json)
- [results/week10/root_cause/v1/full_reader_bootstrap.csv](../../week10/root_cause/v1/full_reader_bootstrap.csv)
- [results/week10/root_cause/v1/full_reader_panel_summary.csv](../../week10/root_cause/v1/full_reader_panel_summary.csv)
- [results/week10/root_cause/v1/full_study_summary.json](../../week10/root_cause/v1/full_study_summary.json)
- [results/week10/root_cause/v1/oracle_diagnostics.json](../../week10/root_cause/v1/oracle_diagnostics.json)
- [results/week10/root_cause/v1/retrieval_interface_interactions.csv](../../week10/root_cause/v1/retrieval_interface_interactions.csv)
- [results/week10/root_cause/v1/retrieval_summary.csv](../../week10/root_cause/v1/retrieval_summary.csv)
- [results/week10/root_cause/v1/sara_configuration_crosswalk.csv](../../week10/root_cause/v1/sara_configuration_crosswalk.csv)
- [results/week10/root_cause/v1/summary.csv](../../week10/root_cause/v1/summary.csv)
- [results/week10/root_cause/v1/verification.json](../../week10/root_cause/v1/verification.json)

# Existing Markdown Used as Evidence

NONE
