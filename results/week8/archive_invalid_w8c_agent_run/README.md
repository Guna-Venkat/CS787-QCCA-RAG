# Invalidated W8-C agent-run comparison

The comparison CSV is retained only as provenance. The previous runner trained U0
using INTERNAL-VAL instead of FIT, included post-promotion latency as a feature,
confused original passage indices with semantic ranks in S0, reported S4 without
MIG features, and used hard-coded baseline/oracle F1. Do not cite these results.

The repaired notebook writes measured FIT OOF and selected-model validation
results under `results/week8/w8c_utility/` after the real FIT surface exists.
