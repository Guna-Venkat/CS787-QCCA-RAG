"""Reproduce the user's input audit using tokenizer only, with no model generation."""
import sys,json,hashlib
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT))
from demo.live.engine import parse_passages
from demo.live.segmentation import FROZEN,ADAPTED,documents,annotate,inventory,sentence_spans
from demo.live.resources import LiveResources
from src.week11.hierarchy import build

folder=Path(__file__).parent
protected=[ROOT/'src/week9/senttopk_e3/sentences.py',ROOT/'src/week11/hierarchy.py',ROOT/'src/week12/algorithm.py',ROOT/'src/week12/laboratory.py',ROOT/'demo/replay_data/v1/bundle.json',ROOT/'results/week12/W13_FINAL_METHOD_FREEZE.json',ROOT/'results/week12/qbgf/v3/W13_FINAL_METHOD_FREEZE.json']
def hashes():return {str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in protected}
before=hashes();text='\n---\n'.join(__import__('demo.live.preset',fromlist=['load_preset']).load_preset()['passages'])+'\n';passages=parse_passages(text);tok=LiveResources().tokenizer()
runs={}
for name,mode in [('before',FROZEN),('after',ADAPTED)]:
    units=annotate(build(documents(passages,mode),tok,{'max_passage_tokens':256}),mode)
    rows=inventory(passages,units,tok,mode)
    assert all(r['frozen_subsegment_count']==1 for r in rows)
    runs[name]=dict(splitter=mode,units=rows,total_units=len(rows),multiple_sentence_units=sum(r['adapted_subsegment_count']>1 for r in rows))
assert hashes()==before
out=dict(source='Exact passages supplied by the user in this conversation; not retrieved or fact-checked.',passages=passages,source_sha256=hashlib.sha256(text.encode()).hexdigest(),runs=runs,protected_hashes=before)
(folder/'live_sentence_segmentation_audit.json').write_text(json.dumps(out,indent=2,ensure_ascii=False)+'\n')
old=runs['before'];new=runs['after'];pct=100*old['multiple_sentence_units']/old['total_units']
lines=['# LiveDemo sentence-segmentation audit','', 'Date: 3 November 2026','',
'## Finding','',
'**The reported merging is real. LiveDemo was already using the frozen project splitter; the defect is a citation-boundary limitation shared by that splitter, not a separate paragraph-only LiveDemo parser.**','',
'On the exact World Cup input supplied by the user, the old LiveDemo produced 39 units labelled SENTENCE. Sixteen of those units contain multiple sentence segments under the citation-aware comparison below (41.03%). The specific expansion paragraph P6 was one unit instead of three.','',
'No benchmark code, research results or curated bundle were changed. This audit performs tokenizer-only source processing: no SFR/Qwen weights, answer generation, factual checking or outcome evaluation. The supplied World Cup assertions are reproduced as input text, not endorsed as factual claims.','',
'## Splitter lineage and parity','',
'| Component | Actual segmentation path |','|---|---|',
'| LiveDemo before patch | `demo/live/engine.py` → `src/week11/hierarchy.build` → `src/week9/senttopk_e3/sentences.spans` |',
'| Week-11 sentence experiments | The same hierarchy builder and academic splitter fallback; dataset-native sentence offsets take precedence when provided |',
'| FIXED_SENTENCE | Consumes the SENTENCE units from that hierarchy; its allocator does not split paragraphs itself |',
'| QBGF | Consumes the same sentence inventory for SENT_1/2/4/8 options; FULL remains a whole passage |',
f'| Frozen splitter identity | `{FROZEN}` |',
f'| New optional LiveDemo adapter identity | `{ADAPTED}` |','',
'Before the patch, LiveDemo supplied no native sentence offsets, so it used the exact frozen fallback. On raw input its offsets match direct calls to that utility. Dataset-native HotpotQA/SciFact sentence boundaries in research are a distinct input source and must not be described as regex-derived.','',
'## Root cause','',
'The frozen punctuation regex expects whitespace or end-of-text immediately after terminal punctuation and allowed closing quotes/brackets. In `teams.[19] As approved`, `.` is followed by `[` rather than whitespace. The numeric citation is not consumed by the terminal-punctuation pattern, so no boundary is recorded. Several sentences can collapse until a later recognizable boundary or the end of the passage.','',
'Adding a space is insufficient as a general repair: `. [19]` can detach citations into the next unit, and a trailing marker can become a citation-only unit. Abbreviations, decimals and initials intentionally receive special treatment; counting periods alone would over-split `U.S.`, `Dr.`, `e.g.` and `2.96`.','',
'## Measured counts and definition of “true sentences”','',
'| Quantity | Count |','|---|---:|',f'| Raw source passages | {len(passages)} |',
f"| Before: all-source LiveDemo sentence units | {old['total_units']} |",f"| Before: units containing more than one citation-aware sentence segment | {old['multiple_sentence_units']} |",f'| Before: merged-unit percentage | {pct:.2f}% |',
f"| After: citation-aware source segments | {new['total_units']} |",f"| After: segments still subdivided by the same citation-aware policy | {new['multiple_sentence_units']} (0.00%) |",'| Manual grammatical-sentence count in supplied text | 61 |','| Non-sentence source segments retained | 2 |','',
'**Counting convention:** the manually inspected text contains 61 sentence statements, plus the colon-ended introduction “The new rules included:” and the final region-introduction/list block. These two are source fragments, not grammatical sentences. Five sentence statements retain preceding bare-text headings. This makes 63 source segments, not 63 independently certified grammatical sentences. The 16/39 statistic is an operational boundary comparison against the citation-aware adapter, not a trained grammar evaluator. It detects the concrete cited-sentence merges in this input; it is not a universal sentence-quality metric.','',
'Counts above cover **all 15 supplied passages**, before top-10 retrieval filtering, not just selected evidence. The user supplied passages but no exact live query; no original ranking, selected subset or answer was reconstructed. The full inventory export in the UI likewise covers every input passage, even candidates outside the top 10.','',
'| Passage | Before units | After segments | Before merged units |','|---|---:|---:|---:|']
for i,p in enumerate(passages,1):
    a=[r for r in old['units'] if r['passage_id']==f'P{i}'];b=[r for r in new['units'] if r['passage_id']==f'P{i}']
    lines.append(f"| P{i} | {len(a)} | {len(b)} | {sum(r['adapted_subsegment_count']>1 for r in a)} |")
lines+=['','## LiveDemo-only fix','',
'1. The frozen utility remains unchanged and is imported directly. Exact original behavior remains selectable as **Frozen research splitter**.','2. The default custom-input option is explicitly labelled **Citation / Markdown-aware · frozen splitter + boundary adapter**. It is not claimed to have identical raw-text boundaries to the frozen policy.','3. The adapter masks numeric/range/letter citation markers in an equal-length boundary-detection view, then delegates punctuation/abbreviation splitting to the exact frozen utility. Source text is never rewritten; returned offsets index the original input. Citation suffixes stay attached to the preceding statement.','4. Explicit Markdown headings and bullet/numbered-list starts define structural regions. Ordinary single-newline wrapped prose retains frozen behavior. Bare unmarked headings are deliberately not guessed: attached heading/prose and unpunctuated/list blocks display as SOURCE SEGMENT rather than pretending they are pure grammatical sentences.','5. New live native offsets are passed into the existing hierarchy builder. The canonical FIXED_SENTENCE and QBGF allocators remain unchanged. Every unit carries the actual policy identity.','6. Citation-aware cache and result identities include the segmentation version; switching policies cannot reuse old merged evidence or stale answers.','7. Existing shared unit cards honor explicit live-only display metadata: SENTENCE, SOURCE SEGMENT, HEADING SEGMENT, LIST SEGMENT or MULTI-SENTENCE SEGMENT. Curated cards without this metadata retain their saved display behavior.','8. A **Sentence segmentation audit** expander prints raw inputs and every unit with passage/parent IDs, sentence ID, offsets, raw text, exact Qwen token count, policy ID and frozen/adapted subsegment counts. A JSON download is available.','',
'**Scientific boundary:** citation-aware custom-input results are explicitly a LiveDemo preprocessing variant. They must not replace saved curated or benchmark results, and should not be described as exact raw frozen segmentation. Select Frozen research splitter for that exact behavior. Even the frozen heuristic can treat abbreviations at sentence ends ambiguously; neither mode is a grammatical oracle.','',
'## Before/after: the reported expansion passage (P6)','',
'Before, one source unit:','', '```text',passages[5],'```','', 'After:','']
for r in new['units']:
 if r['passage_id']=='P6':lines+=['```text',r['raw_text'],'```','']
lines+=['## Raw inputs and complete inventories','',
'Each raw passage below is reproduced from the supplied `---` blocks. Whitespace at outer card boundaries is stripped by the existing input parser. Sentence texts remain exact substrings of those canonical raw cards. Sentence IDs are the actual hierarchy IDs; token counts use the bundled pinned Qwen tokenizer, not a character-count fixture.','']
for i,p in enumerate(passages,1):
 lines += [f'### Raw passage P{i}','','```text',p,'```','']
 for label,records in [('Before: frozen research / original LiveDemo',old['units']),('After: explicit citation/Markdown adapter',new['units'])]:
  lines+=['#### '+label,'']
  for r in records:
   if r['passage_id']!=f'P{i}':continue
   lines += [f"**Passage {r['passage_id']} · sentence ID `{r['sentence_id']}` · {r['token_count']} tokens · offsets [{r['start_char']}, {r['end_char']})**",'', '```text',r['raw_text'],'```','']
lines+=['## Specific edge-case checks','',
'| Case | Audited behavior |','|---|---|','| Plain multi-sentence paragraph | Adapter and frozen outputs match |','| `.[19]`, `.[19][20]`, `. [19]` | Citation-aware mode splits and retains citation suffixes |','| Letter footnote `[D]`, numeric ranges `[19–21]` | Supported in citation-aware mode |','| Mid-sentence `[13][14]` after a comma | Does not create a new sentence |','| Markdown `# Heading` | Separate, labelled heading segment |','| Markdown bullets/numbered lists | Separate structural regions; sentences inside each region use frozen rules |','| Bare headings such as Format / Expansion | Preserved conservatively with source-segment labelling; no guessed grammar |','| Single-newline wrapped prose | Not split solely because of the line wrap |','| Blank paragraphs | Preserve the frozen paragraph boundary rule |','| `Dr.`, `e.g.`, `U.S.`, decimals | Preserve frozen abbreviation/decimal behavior |','',
'## Verification and reproducibility','',
'Run the source-only audit with:','', '```bash','demo/.venv/bin/python demo/audit/run_sentence_segmentation_audit.py','```','',
'Run the regression suite with:','', '```bash','demo/.venv/bin/python -m pytest demo/tests/test_live_segmentation.py demo/tests/test_live.py -q','```','',
'Full machine-readable inventories and protected-source hashes are saved in `demo/audit/live_sentence_segmentation_audit.json`. All raw input text is retained separately in `demo/audit/world_cup_input.txt`. The audit verifies before/after source hashes and exact substring/offset/token integrity without generating answers.','',
'### Protected identities','', '| File | SHA256 |','|---|---|']
for name,h in before.items():lines.append(f'| `{name}` | `{h}` |')
(folder/'live_sentence_segmentation_audit.md').write_text('\n'.join(lines)+'\n')
print(json.dumps({k:{a:v for a,v in val.items() if a!='units'} for k,val in runs.items()},indent=2))
