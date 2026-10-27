# Fresh confirmation plan

QASPER TEST is closed. FIT, project DEV and INTERNAL-VAL are development. Project DEV maps to raw QASPER TRAIN papers (verified), not official validation. No unused paper subset is claimed without an exposure ledger.

Proposed external panel: TriviaQA (single-hop/open QA) and HotpotQA distractor (multi-hop). Freeze revisions, 200 IDs each, seed42, native corpus mode, exact/near duplicate checks against all development questions, prompts, metrics and method choices before generating answers. Reserve disjoint integration and confirmation IDs. No external dataset was downloaded in this pass. External does not guarantee absence from backbone pretraining; disclose that uncertainty.

Keep corpus scope consistent across methods. Supplied-evidence HotpotQA is not global web retrieval. If freshness cannot be established, all results remain diagnostic; do not claim confirmed improvement.
