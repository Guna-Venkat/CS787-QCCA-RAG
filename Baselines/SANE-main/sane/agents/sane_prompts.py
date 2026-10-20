"""Prompt templates for the SANE (Select-And-Extract) method.

Contains only string constants. All logic lives in sane_helpers.py.
"""

from __future__ import annotations

_ANSWER_TEMPLATE_EVIDENCE = (
    "I will give you extracted evidence from history records. "
    "Please answer the question based on this evidence.\n\n"
    "Extracted Evidence:\n"
    "{evidence}\n\n"
    "Current Date: {question_date}\n"
    "Question: {question}\n"
    "Answer:"
)

DOC_SELECTION_PROMPT = """You are a document selector for a retrieval system. Given a question and synopses of candidate documents, pick the ones most likely to contain information relevant to answering the question.

Each line shows a quoted excerpt and keyword tags.

Output ONLY a JSON list of up to {max_select} line numbers (1-based).
Example: [2, 5, 7]
Example (no relevant docs): []
"""

DOC_SELECTION_PROMPT_FULL_TEXT = """You are a document selector for a retrieval system. Given a question and full-text records, pick the ones most likely to contain information relevant to answering the question.

Each record is shown as a ## Record N section with its full content.

Output ONLY a JSON list of up to {max_select} record numbers (1-based).
Example: [2, 5, 7]
Example (no relevant docs): []
"""


# ──
# Default: Blueprint-Guided Extraction (TRIAD V2 — soft guidance)
# ──

BLUEPRINT_GUIDED_PROMPT = """You are a precise evidence extractor. Given a question and retrieved content, extract ONLY the information needed to answer the question. The schemas below are design inspiration — useful starting points, not rigid templates.

## Schema Lookup Table

| Schema | Information Structure | Use When | Columns |
|---|---|---|---|
| ATOMIC | Independent facts — no row relates to any other | What/Who/Where/Define/Find/Yes-no questions | fact, evidence_quote, source, source_date |
| COMPARATIVE | Aligned dimensions — entities compared on shared attributes | Compare/Difference/Versus/Which is larger-better-faster/Rank | entity, dimension, value, unit, evidence_quote, source |
| TEMPORAL | Sequenced events — rows ordered by occurrence time | When/First/Last/Most recent/Timeline/Before-After/Sequence | event, position, resolved_date, date_precision, evidence_quote, source |

## Column Definitions

### ATOMIC
| Column | Definition | Rules |
|--------|-----------|-------|
| `fact` | A single atomic proposition expressed in one sentence | Split compound statements with "and"/"but"/"or" into separate rows. Use the entity or topic as the subject. |
| `evidence_quote` | Verbatim text from source | Exact quote. Use `...` for elision. Never paraphrase. |
| `source` | Which document/session | Consistent label across rows. |
| `source_date` | When the source was created | ISO 8601 when resolvable; raw expression otherwise. NULL if unknown. |

### COMPARATIVE
| Column | Definition | Rules |
|--------|-----------|-------|
| `entity` | Which thing is being described | Same canonical name in every row for the same entity. |
| `dimension` | What attribute is being compared | Same dimension label for all entities on the same comparison axis. Rows sharing a dimension form a comparison group. |
| `value` | The attribute value | Numeric when the source provides a number. Source text otherwise. NULL when this entity has no information for this dimension — asymmetry is information. |
| `unit` | Unit of measurement | Standardized (`million`, `km`, `percent`, `USD`). Empty for categorical or unquantified values. |
| `evidence_quote` | Verbatim text | Exact quote. |
| `source` | Which document/session | — |

### TEMPORAL
| Column | Definition | Rules |
|--------|-----------|-------|
| `event` | What happened | Brief description (5-15 words). Include the key entity or action. |
| `position` | Ordinal position in the sequence | 1-based integer. Ties (same position number) are allowed and expected when order within a group is ambiguous. |
| `resolved_date` | Best-guess absolute date | ISO 8601 (YYYY-MM-DD). NULL when unresolvable — this is common and acceptable. |
| `date_precision` | How precisely the date is known | `day` (exact), `month` (month known, day unknown), `year` (year only), `relative` (only order known, not absolute date). |
| `evidence_quote` | Verbatim text | Exact quote. |
| `source` | Which document/session | — |

## How to Use the Schemas

- Schemas are design aids, not constraints. The goal is a clear evidence table — not schema compliance.
- Use a schema directly when it fits the question well.
- Adapt columns freely — rename domain-specific ones, drop irrelevant ones, borrow from multiple schemas.
- When the evidence is sparse or the question doesn't map cleanly to any schema, a simple fact table is perfectly fine.
- When multiple perspectives apply (e.g., comparison + temporal order), you may blend columns or extract multiple complementary tables.

## Instructions

1. **Read the question carefully.** Identify what information structure best fits: independent facts, aligned comparison, or sequenced events.

2. **Consult the schemas above for inspiration.** Use them as flexible starting points. Let the retrieved content shape the schema as much as the blueprints do.

3. **Scan the retrieved content.** Find every piece of relevant information. IGNORE everything irrelevant.

4. **Design a schema that fits the evidence you found.** Be exhaustive — the answerer will filter and reason.

5. **Every row MUST include an exact evidence_quote.** Never paraphrase.

6. **Output ONLY the markdown table.** No preamble, no commentary, no answer.
"""


EVIDENCE_EXTRACTION_PROMPT = """You are a precise evidence extractor. Given a question and retrieved content, extract ONLY the information needed to answer the question.

Output ONLY the extracted information.
"""

# ──
# Suite prompt resolver
# ──

def get_suite_prompt(suite: str) -> str | None:
    """Return the extraction prompt for a suite, or None to use the no-BP fallback.

    Args:
        suite: Suite name. ``"blueprint"`` returns the blueprint-guided prompt.

    Returns:
        Prompt string, or None for unknown/empty suite names.
    """
    if suite == "blueprint":
        return BLUEPRINT_GUIDED_PROMPT
    return None
