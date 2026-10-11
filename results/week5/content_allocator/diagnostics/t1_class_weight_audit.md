# Week 5: T1 Class Weighting Audit

## Formula Verified
The training pipeline used standard balanced inverse class frequency weighting:
`w_c = N_total / (C * N_c)`
normalized such that sum(w_c) / C = 1.0.

## Empirical Values
| Budget (k) | Class Index | TRAIN Count | TRAIN % | Raw Weight | Normalized Weight ($w_c$) | Weight Ratio vs k=2 |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **k=2** | 0 | 1234 | 59.10% | 0.3384 | **0.1829** | 1.00x (Suppressed) |
| **k=4** | 1 | 354 | 16.95% | 1.1797 | **0.6374** | 3.48x |
| **k=5** | 2 | 163 | 7.81% | 2.5620 | **1.3843** | 7.57x |
| **k=6** | 3 | 134 | 6.42% | 3.1164 | **1.6839** | 9.21x |
| **k=8** | 4 | 203 | 9.72% | 2.0571 | **1.1115** | **6.08x** |

## Audit Finding: Did Weighting Drive I3-T1 Collapse?
1. **Severe Suppression of k=2:** k=2 comprises 59.1% of TRAIN, so its loss contribution is heavily downweighted ($w_0 = 0.1829$). Conversely, k=8 is amplified by **6.08x** relative to k=2, and k=6 is amplified by **9.21x**.
2. **Asymmetric Risk in Optimization:** A false negative on k=8 penalizes the cross-entropy loss **6.08 times more heavily** than a false negative on k=2. When the model representation is noisy or uninformative, predicting k=8 or k=6 incurs far lower penalized loss than predicting k=2.
3. **Difference between I1/I2/I4 vs I3:** In I1 and I2, the query and W4 features provide sufficient low-entropy bias toward k=4/k=5/k=6. In I3, however, the aggregator combines 10 passage vectors with rank embeddings; under high entropy and frozen base encoder representations, the weighted cross-entropy heavily incentivized predicting minority classes, and coupled with DEV policy selection (which scores higher on k=8 than k=2), drove the model to 100% k=8 collapse.
