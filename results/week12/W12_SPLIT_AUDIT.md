# Week 12 split and leakage audit

Development reuses the explicit W11 pilot and is NOT confirmation. Candidate confirmation identities are reserved before W12 outcomes, but **not certified fresh** until prior-history review. Week 12 never evaluates them. QASPER TEST remains permanently closed. Historical IDs are scoped by dataset; paper/document grouping and question hashes must be checked, not bare numeric IDs alone.

## QASPER

{
  "dataset": "QASPER",
  "official_source_split": "TRAIN(project DEV); official validation candidates",
  "requested_confirmation": 100,
  "available_proposed_confirmation": 100,
  "training_source": {
    "path": "/home/gunavenkat/Downloads/CS787-RAG-Project/Baselines/SARA-main/data/reformatted/QASPER_train_split.jsonl",
    "available": true,
    "filter": "exclude development/confirmation question hashes and document groups; historical TRAIN reuse is allowed; never reuse historical validation groups"
  },
  "overlap_queries": 0,
  "history_coverage": "Known project QASPER reformatted sources + complete W11 cohort; W0-W10 additional exposure not yet exhaustively certified",
  "confirmation_ready": false,
  "training_ids_status": "NOT_PREPARED; full TRAIN/FIT manifest generated and audited before manual training",
  "indexed_corpus_overlap_is_not_label_leakage": true,
  "document_group_policy": "QASPER paper; SciFact connected cited-document components; Hotpot normalized question; corpus documents may be shared",
  "historical_used_id_count": 2319,
  "historical_document_count": 858,
  "development_queries": 50
}

## HotpotQA

{
  "dataset": "HotpotQA",
  "official_source_split": "DEV",
  "requested_confirmation": 200,
  "available_proposed_confirmation": 200,
  "training_source": {
    "path": "/home/gunavenkat/Downloads/CS787-RAG-Project/data/week12/hotpot_train_v1.1.json",
    "available": false,
    "filter": "exclude development/confirmation question hashes and document groups; historical TRAIN reuse is allowed; never reuse historical validation groups"
  },
  "overlap_queries": 0,
  "history_coverage": "Known project QASPER reformatted sources + complete W11 cohort; W0-W10 additional exposure not yet exhaustively certified",
  "confirmation_ready": false,
  "training_ids_status": "NOT_PREPARED; full TRAIN/FIT manifest generated and audited before manual training",
  "indexed_corpus_overlap_is_not_label_leakage": true,
  "document_group_policy": "QASPER paper; SciFact connected cited-document components; Hotpot normalized question; corpus documents may be shared",
  "historical_used_id_count": 50,
  "historical_document_count": 499,
  "development_queries": 50
}

## SciFact

{
  "dataset": "SciFact",
  "official_source_split": "DEV",
  "requested_confirmation": 100,
  "available_proposed_confirmation": 100,
  "training_source": {
    "path": "/home/gunavenkat/Downloads/CS787-RAG-Project/data/week12/scifact/claims_train.jsonl",
    "available": false,
    "filter": "exclude development/confirmation question hashes and document groups; historical TRAIN reuse is allowed; never reuse historical validation groups"
  },
  "overlap_queries": 0,
  "history_coverage": "Known project QASPER reformatted sources + complete W11 cohort; W0-W10 additional exposure not yet exhaustively certified",
  "confirmation_ready": false,
  "training_ids_status": "NOT_PREPARED; full TRAIN/FIT manifest generated and audited before manual training",
  "indexed_corpus_overlap_is_not_label_leakage": true,
  "document_group_policy": "QASPER paper; SciFact connected cited-document components; Hotpot normalized question; corpus documents may be shared",
  "historical_used_id_count": 50,
  "historical_document_count": 5183,
  "development_queries": 50
}

## Readiness

HotpotQA/SciFact training files may need explicit preparation. QASPER official-validation candidates must pass historical exposure certification and native-adapter qualification before Week 13. Unknown W0-W10 exposure is a blocker, not evidence of freshness. Full identity lists are in protocol/cohorts.json and protocol/split_audit.json. No fresh QASPER confirmation is claimed.
