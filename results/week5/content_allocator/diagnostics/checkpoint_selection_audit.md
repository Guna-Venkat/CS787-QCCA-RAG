# Week 5: Checkpoint-Selection Rule Audit

## Hierarchy Specified
The intended selection rule in `src/week5/training.py` was:
1. Higher DEV Policy F1 (by at least `min_delta = 1e-4`)
2. If F1 nearly tied ($|\Delta F_1| \le 1e-4$), lower context tokens ($> 1$ token reduction)
3. If tokens also tied, lower oracle regret

## Observed Epoch Selection Dynamics
| Model | Seed | Selected Epoch | DEV Policy F1 | DEV Context Tokens | Mean k | DEV Regret |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **I1-T1** | 42 | 5 | 0.3775 | 1097.4 | 4.79 | 0.1130 |
| **I1-T1** | 123 | 3 | 0.3656 | 1019.2 | 4.40 | 0.1245 |
| **I1-T1** | 2026 | 1 | 0.3695 | 1111.6 | 4.87 | 0.1183 |
| **I2-T1** | 42 | 3 | 0.3731 | 1195.5 | 5.21 | 0.1171 |
| **I2-T1** | 123 | 1 | 0.3737 | 1250.9 | 5.52 | 0.1158 |
| **I2-T1** | 2026 | 1 | 0.3610 | 1072.6 | 4.67 | 0.1278 |
| **I3-T1** | 42 | 1 | 0.3950 | 1750.5 | 8.00 | 0.0945 |
| **I3-T1** | 123 | 2 | 0.3950 | 1750.5 | 8.00 | 0.0945 |
| **I3-T1** | 2026 | 6 | 0.3950 | 1750.5 | 8.00 | 0.0945 |
| **I4-T1** | 42 | 5 | 0.3625 | 1117.2 | 4.87 | 0.1264 |
| **I4-T1** | 123 | 3 | 0.3574 | 1035.1 | 4.49 | 0.1326 |
| **I4-T1** | 2026 | 1 | 0.3691 | 1310.0 | 5.78 | 0.1205 |
| **All T2** | All | 1 | 0.3950 | 1750.5 | 8.00 | 0.0945 |

## Audit Finding: The "Attractor" Effect of Static k=8
1. **F1 Dominates Context Tokens:** In the policy response surface, static $k=8$ achieves $F_1 = 0.3950$, whereas non-trivial learned allocations achieve $F_1 \approx 0.3630 - 0.3775$.
2. **Early Stopping Prematurely Locked k=8:** Because the primary criterion was Policy F1, if an epoch collapsed to $k=8$ (yielding $F_1 = 0.3950$), no subsequent epoch with balanced allocations ($F_1 \approx 0.3750$, tokens $\approx 1100$) could ever beat it, because $0.3750 < 0.3950 - 1e-4$.
3. **The Tie-Breaker Never Fired:** The secondary tie-breaker (lower context tokens) only activates when $|F_1 - best\_F_1| \le 1e-4$. Since static $k=8$ sits at $0.3950$, it forms a strict barrier that blocks models from selecting balanced cost-efficient checkpoints!
