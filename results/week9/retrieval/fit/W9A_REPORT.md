# W9-A

Split: fit. Real saved rows: 298566.

FIT selects; INTERNAL-VAL confirms. DEV has been inspected. TEST remains locked.

No conclusions are implied beyond the saved metrics. See summary CSV, manifests and paired intervals. Stage timings are cached component measurements; total end-to-end latency is unmeasured.

Proposed FIT decision (freeze requires explicit --freeze):

```json
{
  "semantic_grid": [
    {
      "buffer": 0,
      "threshold": 0.25,
      "max_tokens": 1024,
      "overlap": 128
    },
    {
      "buffer": 0,
      "threshold": 0.35,
      "max_tokens": 1024,
      "overlap": 128
    },
    {
      "buffer": 0,
      "threshold": 0.5,
      "max_tokens": 1024,
      "overlap": 128
    },
    {
      "buffer": 1,
      "threshold": 0.25,
      "max_tokens": 1024,
      "overlap": 128
    },
    {
      "buffer": 1,
      "threshold": 0.35,
      "max_tokens": 1024,
      "overlap": 128
    },
    {
      "buffer": 1,
      "threshold": 0.5,
      "max_tokens": 1024,
      "overlap": 128
    },
    {
      "buffer": 2,
      "threshold": 0.25,
      "max_tokens": 1024,
      "overlap": 128
    },
    {
      "buffer": 2,
      "threshold": 0.35,
      "max_tokens": 1024,
      "overlap": 128
    },
    {
      "buffer": 2,
      "threshold": 0.5,
      "max_tokens": 1024,
      "overlap": 128
    }
  ],
  "fixed_sizes": {
    "SEM-0": 27,
    "SEM-1": 27,
    "SEM-2": 30,
    "SEM-3": 81,
    "SEM-4": 205,
    "SEM-5": 1022,
    "SEM-6": 258,
    "SEM-7": 1024,
    "SEM-8": 1024
  },
  "representation": "FIXED-5",
  "retriever": "SEMANTIC",
  "retrieval_width": 20
}
```

## Post-run freeze audit

The proposed decision above was audited and frozen on 2026-10-21 using the unchanged preregistered selection rule. See `freeze_audit/W9A_FREEZE_AUDIT.md` and `results/week9/frozen/w9a_frozen_config.json`. W9-B readiness checks pass; no B model generation was launched by the audit.
