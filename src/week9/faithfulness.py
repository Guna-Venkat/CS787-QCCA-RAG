"""Deterministic support audit; quote validity is not fact entailment."""
import re


def norm(text):
    return ' '.join(text.split())


def table_rows(text):
    lines = [line.strip() for line in text.splitlines() if line.strip().startswith('|')]
    if len(lines) < 2:
        return []
    # Support escaped Markdown pipes. Non-table prose is never converted to evidence.
    def cells(line):
        return [v.strip().strip('`') for v in re.split(r'(?<!\\)\|', line.strip('|'))]
    headers = [h.lower().replace(' ', '_') for h in cells(lines[0])]
    result = []
    for line in lines[1:]:
        values = cells(line)
        if all(re.fullmatch(r'[:\s-]+', v) for v in values):
            continue
        if len(values) != len(headers):
            continue
        result.append(dict(zip(headers, values)))
    return result


def audit_extraction(text, selected):
    by_id = {c['chunk_id']: c for c in selected}
    audited = []
    for row in table_rows(text):
        span = row.get('support_span', row.get('evidence_quote', '')).strip('"')
        source = row.get('source_chunk_id', row.get('source', ''))
        # Strict identity: no positional or invented source aliases accepted.
        chunk = by_id.get(source)
        supported = bool(chunk and span and norm(span) in norm(chunk['text']))
        fact = row.get('structured_fact', row.get('fact'))
        fact_copied = None if fact is None or chunk is None else norm(fact) in norm(chunk['text'])
        audited.append(dict(source_chunk_id=source, support_span=span, source_identity_valid=chunk is not None,
            exact_span_supported=supported, structured_fact=fact,
            fact_directly_copied=fact_copied,
            fact_entailment='unmeasured' if fact and not fact_copied else 'verbatim' if fact_copied else None))
    return dict(rows=audited, exact_span_support_rate=(sum(r['exact_span_supported'] for r in audited)/len(audited) if audited else None),
        unsupported_structured_fact_rate=None, unsupported_fact_note='Entailment not measured; exact quotes alone do not establish fact support.',
        empty_extraction=not audited, invalid_format=bool(text.strip()) and not audited)


def supported_evidence(audit):
    """Strict project format exposes copied spans only; reference tables stay intact."""
    return '\n'.join(f'[{r["source_chunk_id"]}] {r["support_span"]}' for r in audit['rows'] if r['exact_span_supported'])
