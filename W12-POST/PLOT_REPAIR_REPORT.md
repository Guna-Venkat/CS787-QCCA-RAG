# BEA plotting repair

The plot lookup incorrectly compared `(dataset, budget, method)` with `(dataset, budget)`, so every search failed with `StopIteration`. It now indexes each exact three-field cell and reports missing/duplicate cells explicitly.

- **18 tests passed**, including method-specific bars, missing cells, duplicate cells, undefined redundancy values and all BEA unit tests.
- Completed analysis successfully from **900 existing logical rows**: four figures, pilot/final Markdown reports and `analysis_complete.json`.
- Verified **1,802 existing result/freeze/decision files byte-identical**. No answer regeneration or scientific setting changes.
- `REPORTING_PATCH.json` records exact before/after source hashes. The compatibility guard permits only this exact reporting repair; changes to allocator, data, models, config, scoring or other scientific implementation still invalidate the freeze.
- The analysis notebook cell now reloads the repaired module, so rerunning that cell uses the fix without restarting answer generation.

Outputs: `results/post_w12/bea_qbgf/v3/pilot/` and `results/post_w12/bea_qbgf/v3/FINAL_POST_W12_REPORT.md`.

If the notebook is already open, reload it from disk and rerun only the analysis cell. CLI alternative:

```bash
sara_env/bin/python W12-POST/run.py analyze
```

Branch: gagan. Historical Week12 artifacts unchanged.
