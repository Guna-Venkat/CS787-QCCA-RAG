# W8-D readiness — updated 18 October 2026

The notebook is ready for measured MIG and generator masking attribution.
It does not claim to measure attention A1–A5 or train a U4-B model.

## Completed extraction and correction

Full extraction now covers 1,746 FIT queries (10,476 candidates) and 342
INTERNAL-VAL queries (2,052 candidates). An additional check found that canonical
W7/W8-A prompts strip question whitespace while the initial masking extractor did
not. The 23 affected FIT queries and 5 affected validation queries were remeasured
using the canonical prompt. The remaining cached measurements were retained;
all original W8-C base feature values were verified unchanged. Superseded files
and the repair ledger are under
`results/week8/archive_w8d_question_normalization/`.

The cached selector closeout is complete. Its fixed FIT OOF protocol and paired
paper-bootstrap confirmation found no demonstrated attribution benefit in the
tested variants. See `results/week8/w8d_closeout/CLOSEOUT.md` and
`week8d_cached_closeout.ipynb`. This is a conclusion about the tested variants,
not proof that attribution cannot help. W8-E excludes online attribution.

## Corrections

- Removed gold-answer drafts from the notebook and placeholder drafts from the runner.
- Use complete canonical W8-C surfaces: 1,746 FIT and 342 INTERNAL-VAL queries.
- Use saved semantic candidate identities and their complement as the original four
  FULL passages, including near-tied MiniLM scores. Never assume BM25 positions 0–3.
- Reconstruct the canonical k4 SARA prompt and map original passage indices to
  positions in the six compressed SFR vectors.
- Score the actual draft with explicit draft-only teacher-forcing labels. Preserve
  generation prompt token IDs and append draft tokens without another BOS.
- Zero one requested compressed vector at a time before the SARA projector. Save
  both sum and per-token change in draft log probability. No gold answer is scored.
- Removed fabricated attention features derived from masking scores. True attention
  features remain unmeasured. A6 is measured MiniLM draft/passage cosine similarity.
- Empty drafts have null attribution values and an explicit availability mask.
- Full extraction is resumable with one complete query per cache line; source
  fingerprints reject stale data, code, or checkpoint inputs. Smoke caches are separate.
- Output features contain identifiers and measured pre-promotion inputs, not gold,
  promoted predictions, outcome metrics, or HELP/HURT targets.

## Run

Open `week8d_generator_interaction.ipynb` using the `sara_env` kernel and run from
the top. The default `RUN_FULL = True` runs readiness checks, a two-query FIT smoke,
full FIT extraction, full INTERNAL-VAL extraction, and coverage checks. Set it to
False to run only readiness and smoke checks.

Full outputs are under `results/week8/w8d_attribution/full/`. Prior placeholder
tables under `results/week8/tables/` are not read or overwritten. The frozen W8-C
model and its results are unchanged.

At least 18 GiB free GPU memory is checked before loading SARA. Each nonempty
draft requires seven forward passes. The two real FIT readiness queries took
roughly 15 seconds each after loading; full runtime can vary considerably with
prompt length. This is substantial GPU work despite not generating new answers.
Extraction time is recorded and must be included in future controller cost analysis.

## Verification

- 75 Week 8 tests passed, including draft-only labels, masking sign and slot order,
  canonical draft use, semantic/BM25 mapping, unavailable targets, and stale caches.
- Notebook schema and every code cell validated.
- Two actual FIT queries completed on the real SARA GPU checkpoint: 12 candidate
  rows with finite measured masking scores and correct slot mappings.
- Repeating that extraction reused validated cache rows without reloading SARA.
- FIT and INTERNAL-VAL canonical surface and checkpoint preflight checks passed.

The masking features measure support for an existing draft, not correctness or
promotion gain. They do not replace missing gold log-likelihood supervision from
W8-C. The extraction's descriptive MIG coefficients remain fixed; the separate
S7/S8 closeout tunes a small stated coefficient grid on FIT training papers only,
then freezes it before development confirmation. TEST remains untouched.

## Draft text versus generation token IDs

The final greedy-token audit verified the decoded canonical text for all 2,088
queries. Four nonempty FIT drafts and two validation drafts have different token
IDs after text retokenization, due to whitespace or alternative token splits.
W8-D measures likelihood of the retokenized frozen draft text; W8-E token
uncertainty uses the actual generated IDs. W8-D does not claim identical original
generation token streams. The two empty FIT drafts remain explicitly unavailable
for masking. Query IDs are recorded in
`results/week8/w8e_controller/final_verification.json`.
