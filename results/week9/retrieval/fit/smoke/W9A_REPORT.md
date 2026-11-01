# W9-A

Split: fit. Real saved rows: 342.

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
    "SEM-0": 34,
    "SEM-1": 35,
    "SEM-2": 44,
    "SEM-3": 204,
    "SEM-4": 504,
    "SEM-5": 1024,
    "SEM-6": 425,
    "SEM-7": 1024,
    "SEM-8": 1024
  },
  "representation": "SEM-1",
  "retriever": "SEMANTIC",
  "retrieval_width": 10
}
```
