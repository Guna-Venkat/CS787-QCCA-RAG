"""Flexible Dataset Adapter Interface for QA/RAG Datasets.

Defines an abstract base class `DatasetAdapter` and a concrete `QASPERAdapter`
linking SARA reformatted records with raw document hierarchies and evidence spans.
"""

from __future__ import annotations

import abc
import json
import logging
import os
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)


class DatasetAdapter(abc.ABC):
    """Abstract interface for QA/RAG datasets."""

    @abc.abstractmethod
    def load_split(self, split_name: str) -> List[Dict[str, Any]]:
        """Load all query records for the specified split."""
        pass

    @abc.abstractmethod
    def get_query(self, record: Dict[str, Any]) -> str:
        """Extract query string from record."""
        pass

    @abc.abstractmethod
    def get_document(self, record: Dict[str, Any]) -> Dict[str, Any]:
        """Extract full document structure (title, sections, paragraphs)."""
        pass

    @abc.abstractmethod
    def get_gold_answer(self, record: Dict[str, Any]) -> List[str]:
        """Extract list of valid reference answers."""
        pass

    @abc.abstractmethod
    def get_gold_evidence(self, record: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Extract annotated evidence spans / paragraphs."""
        pass

    @abc.abstractmethod
    def get_metadata(self, record: Dict[str, Any]) -> Dict[str, Any]:
        """Extract metadata (question type, answerability, paper ID, question ID)."""
        pass


class QASPERAdapter(DatasetAdapter):
    """Adapter for the QASPER dataset in the SARA repository setup.

    Merges SARA reformatted JSONL files with raw offline Arrow datasets
    to provide full access to paper hierarchies, evidence spans, and retrieval candidates.
    """

    def __init__(
        self,
        train_split_path: str | Path,
        dev_split_path: str | Path,
        raw_arrow_train_path: Optional[str | Path] = None,
        raw_arrow_val_path: Optional[str | Path] = None,
        manifest_path: Optional[str | Path] = None,
        dev_retrieval_path: Optional[str | Path] = None,
    ):
        self.train_split_path = Path(train_split_path)
        self.dev_split_path = Path(dev_split_path)
        self.raw_arrow_train_path = Path(raw_arrow_train_path) if raw_arrow_train_path else None
        self.raw_arrow_val_path = Path(raw_arrow_val_path) if raw_arrow_val_path else None
        self.manifest_path = Path(manifest_path) if manifest_path else None
        self.dev_retrieval_path = Path(dev_retrieval_path) if dev_retrieval_path else None

        self._raw_train_ds: Optional[Any] = None
        self._raw_val_ds: Optional[Any] = None
        self._retrieval_by_qid: Dict[str, Dict[str, Any]] = {}
        self._loaded_splits: Dict[str, List[Dict[str, Any]]] = {}

        self._load_retrieval_if_available()

    def _get_raw_dataset(self, is_dev: bool = False) -> Any:
        """Lazily load the offline Arrow dataset without network calls."""
        from datasets import Dataset

        if is_dev and self.raw_arrow_val_path and self.raw_arrow_val_path.exists():
            if self._raw_val_ds is None:
                self._raw_val_ds = Dataset.from_file(str(self.raw_arrow_val_path))
            return self._raw_val_ds
        
        # In SARA setup, dev was carved out of the official train split
        if self._raw_train_ds is None and self.raw_arrow_train_path and self.raw_arrow_train_path.exists():
            self._raw_train_ds = Dataset.from_file(str(self.raw_arrow_train_path))
        return self._raw_train_ds

    def _load_retrieval_if_available(self) -> None:
        """Pre-index development retrieval records by question ID."""
        if self.dev_retrieval_path and self.dev_retrieval_path.exists():
            with open(self.dev_retrieval_path, "r", encoding="utf-8") as f:
                for line in f:
                    if line.strip():
                        rec = json.loads(line)
                        qid = str(rec.get("question_id", "")).strip()
                        if qid:
                            self._retrieval_by_qid[qid] = rec

    def load_split(self, split_name: str) -> List[Dict[str, Any]]:
        """Load JSONL records for 'train' or 'dev' split."""
        split_lower = split_name.lower().strip()
        if split_lower in self._loaded_splits:
            return self._loaded_splits[split_lower]

        if split_lower == "train":
            target_path = self.train_split_path
        elif split_lower in ("dev", "val", "validation"):
            target_path = self.dev_split_path
        else:
            raise ValueError(f"Unsupported split: {split_name}. Must be 'train' or 'dev'.")

        if not target_path.exists():
            raise FileNotFoundError(f"Split file not found: {target_path}")

        records: List[Dict[str, Any]] = []
        with open(target_path, "r", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    records.append(json.loads(line))

        self._loaded_splits[split_lower] = records
        return records

    def get_query(self, record: Dict[str, Any]) -> str:
        return str(record.get("question", "")).strip()

    def get_document(self, record: Dict[str, Any]) -> Dict[str, Any]:
        """Retrieve full paper hierarchy from raw dataset or reformatted record."""
        example_id_str = str(record.get("example_id", "")).strip()
        raw_ds = self._get_raw_dataset(is_dev=False)

        if raw_ds is not None and example_id_str.isdigit():
            idx = int(example_id_str)
            if idx < len(raw_ds):
                paper = raw_ds[idx]
                sections = paper.get("full_text", {}).get("section_name", [])
                paragraphs = paper.get("full_text", {}).get("paragraphs", [])
                return {
                    "paper_id": str(paper.get("id", example_id_str)),
                    "title": str(paper.get("title", "")),
                    "abstract": str(paper.get("abstract", "")),
                    "section_names": list(sections),
                    "paragraphs_by_section": list(paragraphs),
                    "all_paragraphs": [p for sec in paragraphs for p in sec if isinstance(sec, list)],
                }

        # Fallback to context chunks in reformatted record
        context = record.get("context", [])
        return {
            "paper_id": example_id_str,
            "title": "",
            "abstract": "",
            "section_names": [],
            "paragraphs_by_section": [],
            "all_paragraphs": list(context) if isinstance(context, list) else [str(context)],
        }

    def get_gold_answer(self, record: Dict[str, Any]) -> List[str]:
        """Extract all valid answer variations."""
        answers = []
        raw_ans = record.get("answer", [])
        if isinstance(raw_ans, list):
            answers.extend([str(a) for a in raw_ans if a])
        elif isinstance(raw_ans, str) and raw_ans:
            answers.append(raw_ans)

        ref_ans = record.get("answer_reformatted", [])
        if isinstance(ref_ans, list):
            for a in ref_ans:
                if a and str(a) not in answers:
                    answers.append(str(a))
        elif isinstance(ref_ans, str) and ref_ans and ref_ans not in answers:
            answers.append(ref_ans)

        return answers if answers else ["Unanswerable"]

    def get_gold_evidence(self, record: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Retrieve structured gold evidence annotations (evidence text and highlighted spans)."""
        example_id_str = str(record.get("example_id", "")).strip()
        q_text = self.get_query(record)
        raw_ds = self._get_raw_dataset(is_dev=False)

        if raw_ds is not None and example_id_str.isdigit():
            idx = int(example_id_str)
            if idx < len(raw_ds):
                paper = raw_ds[idx]
                qas = paper.get("qas", {})
                questions = qas.get("question", [])
                answers_list = qas.get("answers", [])

                # Find matching question
                for q_idx, q_candidate in enumerate(questions):
                    if q_candidate.strip().lower() == q_text.lower():
                        if q_idx < len(answers_list):
                            ev_items: List[Dict[str, Any]] = []
                            ans_entry = answers_list[q_idx]
                            for annotator in ans_entry.get("answer", []):
                                ev_paragraphs = annotator.get("evidence", [])
                                highlighted = annotator.get("highlighted_evidence", [])
                                extractive = annotator.get("extractive_spans", [])
                                unans = annotator.get("unanswerable", False)
                                yes_no = annotator.get("yes_no", None)

                                ev_items.append({
                                    "evidence_paragraphs": list(ev_paragraphs),
                                    "highlighted_spans": list(highlighted),
                                    "extractive_spans": list(extractive),
                                    "unanswerable": bool(unans),
                                    "yes_no": yes_no,
                                })
                            if ev_items:
                                return ev_items

        # Fallback if raw dataset not loaded
        return [{
            "evidence_paragraphs": [],
            "highlighted_spans": [],
            "extractive_spans": [],
            "unanswerable": record.get("question_type") == "unanswerable",
            "yes_no": None,
        }]

    def get_retrieved_passages(self, record: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Get precomputed top-10 BM25 retrieved passages if available."""
        qid = str(record.get("id", record.get("question_id", ""))).strip()
        if qid in self._retrieval_by_qid:
            return self._retrieval_by_qid[qid].get("passages", [])
        return []

    def get_metadata(self, record: Dict[str, Any]) -> Dict[str, Any]:
        qid = str(record.get("id", record.get("question_id", ""))).strip()
        pid = str(record.get("example_id", record.get("paper_id", ""))).strip()
        q_type = str(record.get("question_type", "unknown")).strip()
        return {
            "question_id": qid,
            "paper_id": pid,
            "question_type": q_type,
            "choices": record.get("choices", None),
        }
