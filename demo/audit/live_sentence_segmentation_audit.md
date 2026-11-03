# LiveDemo sentence-segmentation audit

Date: 3 November 2026

## Finding

**The reported merging is real. LiveDemo was already using the frozen project splitter; the defect is a citation-boundary limitation shared by that splitter, not a separate paragraph-only LiveDemo parser.**

On the exact World Cup input supplied by the user, the old LiveDemo produced 39 units labelled SENTENCE. Sixteen of those units contain multiple sentence segments under the citation-aware comparison below (41.03%). The specific expansion paragraph P6 was one unit instead of three.

No benchmark code, research results or curated bundle were changed. This audit performs tokenizer-only source processing: no SFR/Qwen weights, answer generation, factual checking or outcome evaluation. The supplied World Cup assertions are reproduced as input text, not endorsed as factual claims.

## Splitter lineage and parity

| Component | Actual segmentation path |
|---|---|
| LiveDemo before patch | `demo/live/engine.py` → `src/week11/hierarchy.build` → `src/week9/senttopk_e3/sentences.spans` |
| Week-11 sentence experiments | The same hierarchy builder and academic splitter fallback; dataset-native sentence offsets take precedence when provided |
| FIXED_SENTENCE | Consumes the SENTENCE units from that hierarchy; its allocator does not split paragraphs itself |
| QBGF | Consumes the same sentence inventory for SENT_1/2/4/8 options; FULL remains a whole passage |
| Frozen splitter identity | `academic-offset-rules-v1` |
| New optional LiveDemo adapter identity | `academic-offset-rules-v1+live-citation-markdown-v1` |

Before the patch, LiveDemo supplied no native sentence offsets, so it used the exact frozen fallback. On raw input its offsets match direct calls to that utility. Dataset-native HotpotQA/SciFact sentence boundaries in research are a distinct input source and must not be described as regex-derived.

## Root cause

The frozen punctuation regex expects whitespace or end-of-text immediately after terminal punctuation and allowed closing quotes/brackets. In `teams.[19] As approved`, `.` is followed by `[` rather than whitespace. The numeric citation is not consumed by the terminal-punctuation pattern, so no boundary is recorded. Several sentences can collapse until a later recognizable boundary or the end of the passage.

Adding a space is insufficient as a general repair: `. [19]` can detach citations into the next unit, and a trailing marker can become a citation-only unit. Abbreviations, decimals and initials intentionally receive special treatment; counting periods alone would over-split `U.S.`, `Dr.`, `e.g.` and `2.96`.

## Measured counts and definition of “true sentences”

| Quantity | Count |
|---|---:|
| Raw source passages | 15 |
| Before: all-source LiveDemo sentence units | 39 |
| Before: units containing more than one citation-aware sentence segment | 16 |
| Before: merged-unit percentage | 41.03% |
| After: citation-aware source segments | 63 |
| After: segments still subdivided by the same citation-aware policy | 0 (0.00%) |
| Manual grammatical-sentence count in supplied text | 61 |
| Non-sentence source segments retained | 2 |

**Counting convention:** the manually inspected text contains 61 sentence statements, plus the colon-ended introduction “The new rules included:” and the final region-introduction/list block. These two are source fragments, not grammatical sentences. Five sentence statements retain preceding bare-text headings. This makes 63 source segments, not 63 independently certified grammatical sentences. The 16/39 statistic is an operational boundary comparison against the citation-aware adapter, not a trained grammar evaluator. It detects the concrete cited-sentence merges in this input; it is not a universal sentence-quality metric.

Counts above cover **all 15 supplied passages**, before top-10 retrieval filtering, not just selected evidence. The user supplied passages but no exact live query; no original ranking, selected subset or answer was reconstructed. The full inventory export in the UI likewise covers every input passage, even candidates outside the top 10.

| Passage | Before units | After segments | Before merged units |
|---|---:|---:|---:|
| P1 | 3 | 4 | 1 |
| P2 | 2 | 5 | 1 |
| P3 | 4 | 6 | 2 |
| P4 | 3 | 4 | 1 |
| P5 | 1 | 2 | 1 |
| P6 | 1 | 3 | 1 |
| P7 | 3 | 4 | 1 |
| P8 | 2 | 4 | 2 |
| P9 | 1 | 3 | 1 |
| P10 | 1 | 2 | 1 |
| P11 | 8 | 9 | 1 |
| P12 | 3 | 4 | 1 |
| P13 | 2 | 6 | 1 |
| P14 | 4 | 6 | 1 |
| P15 | 1 | 1 | 0 |

## LiveDemo-only fix

1. The frozen utility remains unchanged and is imported directly. Exact original behavior remains selectable as **Frozen research splitter**.
2. The default custom-input option is explicitly labelled **Citation / Markdown-aware · frozen splitter + boundary adapter**. It is not claimed to have identical raw-text boundaries to the frozen policy.
3. The adapter masks numeric/range/letter citation markers in an equal-length boundary-detection view, then delegates punctuation/abbreviation splitting to the exact frozen utility. Source text is never rewritten; returned offsets index the original input. Citation suffixes stay attached to the preceding statement.
4. Explicit Markdown headings and bullet/numbered-list starts define structural regions. Ordinary single-newline wrapped prose retains frozen behavior. Bare unmarked headings are deliberately not guessed: attached heading/prose and unpunctuated/list blocks display as SOURCE SEGMENT rather than pretending they are pure grammatical sentences.
5. New live native offsets are passed into the existing hierarchy builder. The canonical FIXED_SENTENCE and QBGF allocators remain unchanged. Every unit carries the actual policy identity.
6. Citation-aware cache and result identities include the segmentation version; switching policies cannot reuse old merged evidence or stale answers.
7. Existing shared unit cards honor explicit live-only display metadata: SENTENCE, SOURCE SEGMENT, HEADING SEGMENT, LIST SEGMENT or MULTI-SENTENCE SEGMENT. Curated cards without this metadata retain their saved display behavior.
8. A **Sentence segmentation audit** expander prints raw inputs and every unit with passage/parent IDs, sentence ID, offsets, raw text, exact Qwen token count, policy ID and frozen/adapted subsegment counts. A JSON download is available.

**Scientific boundary:** citation-aware custom-input results are explicitly a LiveDemo preprocessing variant. They must not replace saved curated or benchmark results, and should not be described as exact raw frozen segmentation. Select Frozen research splitter for that exact behavior. Even the frozen heuristic can treat abbreviations at sentence ends ambiguously; neither mode is a grammatical oracle.

## Before/after: the reported expansion passage (P6)

Before, one source unit:

```text
Starting with the 2026 edition, the World Cup expanded to 48 teams, an increase of 16 teams compared to the previous seven tournaments.[19] As approved by the FIFA Council on March 14, 2023, the teams were split into 12 groups of 4 teams, with the top 2 teams in each group and the 8 best third-place teams progressing to a new round of 32.[20] It was the first expansion and format change since 1998.[21]
```

After:

```text
Starting with the 2026 edition, the World Cup expanded to 48 teams, an increase of 16 teams compared to the previous seven tournaments.[19]
```

```text
As approved by the FIFA Council on March 14, 2023, the teams were split into 12 groups of 4 teams, with the top 2 teams in each group and the 8 best third-place teams progressing to a new round of 32.[20]
```

```text
It was the first expansion and format change since 1998.[21]
```

## Raw inputs and complete inventories

Each raw passage below is reproduced from the supplied `---` blocks. Whitespace at outer card boundaries is stripped by the existing input parser. Sentence texts remain exact substrings of those canonical raw cards. Sentence IDs are the actual hierarchy IDs; token counts use the bundled pinned Qwen tokenizer, not a character-count fixture.

### Raw passage P1

```text
The 2026 FIFA World Cup[A] was the 23rd FIFA World Cup, the quadrennial international men's soccer championship contested by the national teams of the member associations of FIFA. The tournament began on June 11, 2026, and concluded on July 19 with Spain winning the championship for the second time.[3] It was jointly hosted by 16 cities—11 in the United States, 3 in Mexico, and 2 in Canada. The tournament was the first FIFA World Cup to be hosted by three countries and the first to include 48 teams, an expansion from the previous 32-team format.[4]
```

#### Before: frozen research / original LiveDemo

**Passage P1 · sentence ID `2c665e4498c777bb2894ee0dfb16405e` · 43 tokens · offsets [0, 179)**

```text
The 2026 FIFA World Cup[A] was the 23rd FIFA World Cup, the quadrennial international men's soccer championship contested by the national teams of the member associations of FIFA.
```

**Passage P1 · sentence ID `0cc2dc8b6678fcb6dd42caf7cc05391d` · 62 tokens · offsets [180, 393)**

```text
The tournament began on June 11, 2026, and concluded on July 19 with Spain winning the championship for the second time.[3] It was jointly hosted by 16 cities—11 in the United States, 3 in Mexico, and 2 in Canada.
```

**Passage P1 · sentence ID `c6913b54530f2ddd15f674403f3304c1` · 37 tokens · offsets [394, 554)**

```text
The tournament was the first FIFA World Cup to be hosted by three countries and the first to include 48 teams, an expansion from the previous 32-team format.[4]
```

#### After: explicit citation/Markdown adapter

**Passage P1 · sentence ID `2c665e4498c777bb2894ee0dfb16405e` · 43 tokens · offsets [0, 179)**

```text
The 2026 FIFA World Cup[A] was the 23rd FIFA World Cup, the quadrennial international men's soccer championship contested by the national teams of the member associations of FIFA.
```

**Passage P1 · sentence ID `3973d6a210f4ed46aa33dc9c0eb8df5b` · 34 tokens · offsets [180, 303)**

```text
The tournament began on June 11, 2026, and concluded on July 19 with Spain winning the championship for the second time.[3]
```

**Passage P1 · sentence ID `a8488dc3ce3514addbff3d74f040bf3a` · 28 tokens · offsets [304, 393)**

```text
It was jointly hosted by 16 cities—11 in the United States, 3 in Mexico, and 2 in Canada.
```

**Passage P1 · sentence ID `c6913b54530f2ddd15f674403f3304c1` · 37 tokens · offsets [394, 554)**

```text
The tournament was the first FIFA World Cup to be hosted by three countries and the first to include 48 teams, an expansion from the previous 32-team format.[4]
```

### Raw passage P2

```text
The 2026 tournament was the second World Cup to be hosted by multiple nations, after the 2002 edition. Mexico became the first country to host the World Cup three times, having hosted the 1970 and 1986 tournaments.[5] The United States previously hosted the World Cup in 1994, while it was Canada's first time hosting the tournament.[5] The event returned to its traditional June–July schedule after the 2022 World Cup in Qatar, which was uniquely held in November and December to avoid Qatar's extreme summer heat.[6] Canada, Mexico, and the United States qualified automatically as hosts, while Cape Verde, Curaçao, Jordan, and Uzbekistan made their World Cup debuts.[5]
```

#### Before: frozen research / original LiveDemo

**Passage P2 · sentence ID `95b8d7028781a6598852fd1cee898af1` · 28 tokens · offsets [0, 102)**

```text
The 2026 tournament was the second World Cup to be hosted by multiple nations, after the 2002 edition.
```

**Passage P2 · sentence ID `33561fe9a220477bc1613e6a140c9aef` · 135 tokens · offsets [103, 672)**

```text
Mexico became the first country to host the World Cup three times, having hosted the 1970 and 1986 tournaments.[5] The United States previously hosted the World Cup in 1994, while it was Canada's first time hosting the tournament.[5] The event returned to its traditional June–July schedule after the 2022 World Cup in Qatar, which was uniquely held in November and December to avoid Qatar's extreme summer heat.[6] Canada, Mexico, and the United States qualified automatically as hosts, while Cape Verde, Curaçao, Jordan, and Uzbekistan made their World Cup debuts.[5]
```

#### After: explicit citation/Markdown adapter

**Passage P2 · sentence ID `95b8d7028781a6598852fd1cee898af1` · 28 tokens · offsets [0, 102)**

```text
The 2026 tournament was the second World Cup to be hosted by multiple nations, after the 2002 edition.
```

**Passage P2 · sentence ID `0fe4575aa8450176163f80cbdb52361b` · 31 tokens · offsets [103, 217)**

```text
Mexico became the first country to host the World Cup three times, having hosted the 1970 and 1986 tournaments.[5]
```

**Passage P2 · sentence ID `e9d6325a39122f05a3a26aed5ff2cf03` · 28 tokens · offsets [218, 336)**

```text
The United States previously hosted the World Cup in 1994, while it was Canada's first time hosting the tournament.[5]
```

**Passage P2 · sentence ID `6c3cb4b2863f137fe3f7a03c04aa5484` · 40 tokens · offsets [337, 518)**

```text
The event returned to its traditional June–July schedule after the 2022 World Cup in Qatar, which was uniquely held in November and December to avoid Qatar's extreme summer heat.[6]
```

**Passage P2 · sentence ID `d6d9ee8f5e7a5ba95951c5ff0e25e092` · 36 tokens · offsets [519, 672)**

```text
Canada, Mexico, and the United States qualified automatically as hosts, while Cape Verde, Curaçao, Jordan, and Uzbekistan made their World Cup debuts.[5]
```

### Raw passage P3

```text
For the first time in the tournament's history, the top four teams from the FIFA Men's World Ranking at the time of the contest (Argentina, Spain, France, and England) advanced to the semifinals.[7] Spain won the final against defending champion Argentina 1–0 after extra time, winning its second title and first since 2010. Spain became the first nation to become the reigning champion for both the men's and women's World Cups simultaneously,[8] and the first team to win the World Cup while conceding only one goal during the entire tournament. France captain Kylian Mbappé won his second consecutive Golden Boot for scoring the most goals (10) during the tournament. He became the all-time leading goalscorer at the World Cup and the first player to win multiple World Cup Golden Boots.[9] The average number of goals per match in the tournament was 2.96, the highest since the 1970 edition.[10]
```

#### Before: frozen research / original LiveDemo

**Passage P3 · sentence ID `2f9acb2e1334e4fe9b87724530b3fe5b` · 74 tokens · offsets [0, 324)**

```text
For the first time in the tournament's history, the top four teams from the FIFA Men's World Ranking at the time of the contest (Argentina, Spain, France, and England) advanced to the semifinals.[7] Spain won the final against defending champion Argentina 1–0 after extra time, winning its second title and first since 2010.
```

**Passage P3 · sentence ID `c28234548f77ea61a3b022844d3a6bac` · 44 tokens · offsets [325, 547)**

```text
Spain became the first nation to become the reigning champion for both the men's and women's World Cups simultaneously,[8] and the first team to win the World Cup while conceding only one goal during the entire tournament.
```

**Passage P3 · sentence ID `73b34aee872262e3e8b42d3d4a0071f6` · 26 tokens · offsets [548, 670)**

```text
France captain Kylian Mbappé won his second consecutive Golden Boot for scoring the most goals (10) during the tournament.
```

**Passage P3 · sentence ID `bf641d087f76a4874c43ddcdfeb49517` · 58 tokens · offsets [671, 899)**

```text
He became the all-time leading goalscorer at the World Cup and the first player to win multiple World Cup Golden Boots.[9] The average number of goals per match in the tournament was 2.96, the highest since the 1970 edition.[10]
```

#### After: explicit citation/Markdown adapter

**Passage P3 · sentence ID `d9df4466ce8ffc27f8f46a5e7f144075` · 45 tokens · offsets [0, 198)**

```text
For the first time in the tournament's history, the top four teams from the FIFA Men's World Ranking at the time of the contest (Argentina, Spain, France, and England) advanced to the semifinals.[7]
```

**Passage P3 · sentence ID `41e2692930edfef2b55bfc4683080244` · 29 tokens · offsets [199, 324)**

```text
Spain won the final against defending champion Argentina 1–0 after extra time, winning its second title and first since 2010.
```

**Passage P3 · sentence ID `c28234548f77ea61a3b022844d3a6bac` · 44 tokens · offsets [325, 547)**

```text
Spain became the first nation to become the reigning champion for both the men's and women's World Cups simultaneously,[8] and the first team to win the World Cup while conceding only one goal during the entire tournament.
```

**Passage P3 · sentence ID `73b34aee872262e3e8b42d3d4a0071f6` · 26 tokens · offsets [548, 670)**

```text
France captain Kylian Mbappé won his second consecutive Golden Boot for scoring the most goals (10) during the tournament.
```

**Passage P3 · sentence ID `a694151157007e6b5ea979d0e75608b9` · 27 tokens · offsets [671, 793)**

```text
He became the all-time leading goalscorer at the World Cup and the first player to win multiple World Cup Golden Boots.[9]
```

**Passage P3 · sentence ID `db1ae5a177474b80104274336e325953` · 31 tokens · offsets [794, 899)**

```text
The average number of goals per match in the tournament was 2.96, the highest since the 1970 edition.[10]
```

### Raw passage P4

```text
Spain's captain Rodri won the Golden Ball as the best player of the tournament. Unai Simón won the Golden Glove as the tournament's best goalkeeper, while Pau Cubarsí won the Young Player Award as the tournament's best young player. Nearly 7 million people attended matches during the tournament, surpassing the 1994 edition to set a new record for the highest attendance in World Cup history.[11] The 2026 World Cup reportedly generated a record $15 billion in revenue.[12]
```

#### Before: frozen research / original LiveDemo

**Passage P4 · sentence ID `0e0e1797b16f8524ff4ad2b2e01f8e78` · 17 tokens · offsets [0, 79)**

```text
Spain's captain Rodri won the Golden Ball as the best player of the tournament.
```

**Passage P4 · sentence ID `eb3823a77b3e2d336201efaea6410fab` · 35 tokens · offsets [80, 232)**

```text
Unai Simón won the Golden Glove as the tournament's best goalkeeper, while Pau Cubarsí won the Young Player Award as the tournament's best young player.
```

**Passage P4 · sentence ID `7766fae2440e96fed6f3627cd837d8bd` · 59 tokens · offsets [233, 474)**

```text
Nearly 7 million people attended matches during the tournament, surpassing the 1994 edition to set a new record for the highest attendance in World Cup history.[11] The 2026 World Cup reportedly generated a record $15 billion in revenue.[12]
```

#### After: explicit citation/Markdown adapter

**Passage P4 · sentence ID `0e0e1797b16f8524ff4ad2b2e01f8e78` · 17 tokens · offsets [0, 79)**

```text
Spain's captain Rodri won the Golden Ball as the best player of the tournament.
```

**Passage P4 · sentence ID `eb3823a77b3e2d336201efaea6410fab` · 35 tokens · offsets [80, 232)**

```text
Unai Simón won the Golden Glove as the tournament's best goalkeeper, while Pau Cubarsí won the Young Player Award as the tournament's best young player.
```

**Passage P4 · sentence ID `fbc9c6c457b70cfeee045cfacee238e2` · 37 tokens · offsets [233, 397)**

```text
Nearly 7 million people attended matches during the tournament, surpassing the 1994 edition to set a new record for the highest attendance in World Cup history.[11]
```

**Passage P4 · sentence ID `e16b7183e98260884b9b183131a23f70` · 22 tokens · offsets [398, 474)**

```text
The 2026 World Cup reportedly generated a record $15 billion in revenue.[12]
```

### Raw passage P5

```text
Format
Expansion
The idea of expanding the tournament had been suggested as early as 2013 by then-UEFA president Michel Platini,[13][14] as well as in 2016 by current FIFA president Gianni Infantino.[15] Opponents of the proposal argued that the number of matches played was already at an unacceptable level, that the expansion would dilute the quality of the matches,[16][17] and that the decision was driven by political rather than sporting concerns, specifically that Infantino was trying to win his election by promising to bring more countries to the World Cup.[18]
```

#### Before: frozen research / original LiveDemo

**Passage P5 · sentence ID `2c7706fbba629ece4547a9db80717092` · 132 tokens · offsets [0, 571)**

```text
Format
Expansion
The idea of expanding the tournament had been suggested as early as 2013 by then-UEFA president Michel Platini,[13][14] as well as in 2016 by current FIFA president Gianni Infantino.[15] Opponents of the proposal argued that the number of matches played was already at an unacceptable level, that the expansion would dilute the quality of the matches,[16][17] and that the decision was driven by political rather than sporting concerns, specifically that Infantino was trying to win his election by promising to bring more countries to the World Cup.[18]
```

#### After: explicit citation/Markdown adapter

**Passage P5 · sentence ID `4112d024344476f34280c182de914de9` · 58 tokens · offsets [0, 203)**

```text
Format
Expansion
The idea of expanding the tournament had been suggested as early as 2013 by then-UEFA president Michel Platini,[13][14] as well as in 2016 by current FIFA president Gianni Infantino.[15]
```

**Passage P5 · sentence ID `81e7b2d917366a61dd4b58e099352b74` · 74 tokens · offsets [204, 571)**

```text
Opponents of the proposal argued that the number of matches played was already at an unacceptable level, that the expansion would dilute the quality of the matches,[16][17] and that the decision was driven by political rather than sporting concerns, specifically that Infantino was trying to win his election by promising to bring more countries to the World Cup.[18]
```

### Raw passage P6

```text
Starting with the 2026 edition, the World Cup expanded to 48 teams, an increase of 16 teams compared to the previous seven tournaments.[19] As approved by the FIFA Council on March 14, 2023, the teams were split into 12 groups of 4 teams, with the top 2 teams in each group and the 8 best third-place teams progressing to a new round of 32.[20] It was the first expansion and format change since 1998.[21]
```

#### Before: frozen research / original LiveDemo

**Passage P6 · sentence ID `fec45a386db2a13a6323f7952104b9d6` · 117 tokens · offsets [0, 405)**

```text
Starting with the 2026 edition, the World Cup expanded to 48 teams, an increase of 16 teams compared to the previous seven tournaments.[19] As approved by the FIFA Council on March 14, 2023, the teams were split into 12 groups of 4 teams, with the top 2 teams in each group and the 8 best third-place teams progressing to a new round of 32.[20] It was the first expansion and format change since 1998.[21]
```

#### After: explicit citation/Markdown adapter

**Passage P6 · sentence ID `11bdde878be83c3350f7fd9dea539b7a` · 37 tokens · offsets [0, 139)**

```text
Starting with the 2026 edition, the World Cup expanded to 48 teams, an increase of 16 teams compared to the previous seven tournaments.[19]
```

**Passage P6 · sentence ID `3014b5368d61d594f24bcdd0e3faf8cf` · 62 tokens · offsets [140, 344)**

```text
As approved by the FIFA Council on March 14, 2023, the teams were split into 12 groups of 4 teams, with the top 2 teams in each group and the 8 best third-place teams progressing to a new round of 32.[20]
```

**Passage P6 · sentence ID `420c36d94b483b8f0c01a51e7b2490dc` · 18 tokens · offsets [345, 405)**

```text
It was the first expansion and format change since 1998.[21]
```

### Raw passage P7

```text
The total number of matches played increased from 64 to 104, and the number of matches played by teams reaching the semifinals increased from 7 to 8. The tournament lasted 39 days, an increase from the 32 days of the 2014 and 2018 tournaments, and from the 29 days of the 2022 tournament,[22][23] although each team still played three group matches.[24][25] The final matchday at club level for players named in the final squads was May 24, 2026; clubs had to release their players by May 25, with exceptions granted to players participating in continental club competition finals up until May 30. The combined rest, release, and tournament periods remained 56 days, the same as in the 2010, 2014, and 2018 World Cups.[20]
```

#### Before: frozen research / original LiveDemo

**Passage P7 · sentence ID `168fefcea10d1d787bec4d90dbd4641d` · 37 tokens · offsets [0, 149)**

```text
The total number of matches played increased from 64 to 104, and the number of matches played by teams reaching the semifinals increased from 7 to 8.
```

**Passage P7 · sentence ID `534da352ea8eb00c0676b5ed1de59850` · 124 tokens · offsets [150, 597)**

```text
The tournament lasted 39 days, an increase from the 32 days of the 2014 and 2018 tournaments, and from the 29 days of the 2022 tournament,[22][23] although each team still played three group matches.[24][25] The final matchday at club level for players named in the final squads was May 24, 2026; clubs had to release their players by May 25, with exceptions granted to players participating in continental club competition finals up until May 30.
```

**Passage P7 · sentence ID `e7e684c6c37549ffb3505084b0e04898` · 44 tokens · offsets [598, 722)**

```text
The combined rest, release, and tournament periods remained 56 days, the same as in the 2010, 2014, and 2018 World Cups.[20]
```

#### After: explicit citation/Markdown adapter

**Passage P7 · sentence ID `168fefcea10d1d787bec4d90dbd4641d` · 37 tokens · offsets [0, 149)**

```text
The total number of matches played increased from 64 to 104, and the number of matches played by teams reaching the semifinals increased from 7 to 8.
```

**Passage P7 · sentence ID `0ae5336e04e67c701407003c7437ef60` · 68 tokens · offsets [150, 357)**

```text
The tournament lasted 39 days, an increase from the 32 days of the 2014 and 2018 tournaments, and from the 29 days of the 2022 tournament,[22][23] although each team still played three group matches.[24][25]
```

**Passage P7 · sentence ID `ad14c1b9c1e1696c4a1308a3a37023f7` · 56 tokens · offsets [358, 597)**

```text
The final matchday at club level for players named in the final squads was May 24, 2026; clubs had to release their players by May 25, with exceptions granted to players participating in continental club competition finals up until May 30.
```

**Passage P7 · sentence ID `e7e684c6c37549ffb3505084b0e04898` · 44 tokens · offsets [598, 722)**

```text
The combined rest, release, and tournament periods remained 56 days, the same as in the 2010, 2014, and 2018 World Cups.[20]
```

### Raw passage P8

```text
Other expansion formats explored
After the expansion to 48 teams was approved on January 10, 2017, it was initially decided that the tournament would include 16 groups of 3 teams, with the top two teams of each group progressing to a round of 32.[19][26] This format would have included 80 matches in total, with the maximum number of matches per team remaining at seven, although each team would have played one fewer group match than in the 2022 tournament. The tournament would still have been completed within 32 days.[27] This format was chosen over three alternatives involving 40 to 48 teams, 76 to 88 matches, and a minimum of 1 to 4 matches per team.[B]
```

#### Before: frozen research / original LiveDemo

**Passage P8 · sentence ID `817c1c050d17b1d79260b219c8b5d099` · 112 tokens · offsets [0, 459)**

```text
Other expansion formats explored
After the expansion to 48 teams was approved on January 10, 2017, it was initially decided that the tournament would include 16 groups of 3 teams, with the top two teams of each group progressing to a round of 32.[19][26] This format would have included 80 matches in total, with the maximum number of matches per team remaining at seven, although each team would have played one fewer group match than in the 2022 tournament.
```

**Passage P8 · sentence ID `74d2265eb117394e70cb6e143d1cd56b` · 57 tokens · offsets [460, 662)**

```text
The tournament would still have been completed within 32 days.[27] This format was chosen over three alternatives involving 40 to 48 teams, 76 to 88 matches, and a minimum of 1 to 4 matches per team.[B]
```

#### After: explicit citation/Markdown adapter

**Passage P8 · sentence ID `96c78cdd99bbdac948b82eafdfe8415c` · 68 tokens · offsets [0, 254)**

```text
Other expansion formats explored
After the expansion to 48 teams was approved on January 10, 2017, it was initially decided that the tournament would include 16 groups of 3 teams, with the top two teams of each group progressing to a round of 32.[19][26]
```

**Passage P8 · sentence ID `15660f3a56a7952277340dc518ef9398` · 44 tokens · offsets [255, 459)**

```text
This format would have included 80 matches in total, with the maximum number of matches per team remaining at seven, although each team would have played one fewer group match than in the 2022 tournament.
```

**Passage P8 · sentence ID `380e752ac315dd0b351ee6f8172e2608` · 16 tokens · offsets [460, 526)**

```text
The tournament would still have been completed within 32 days.[27]
```

**Passage P8 · sentence ID `1e83b4cfab274c28383f9eb411a83641` · 41 tokens · offsets [527, 662)**

```text
This format was chosen over three alternatives involving 40 to 48 teams, 76 to 88 matches, and a minimum of 1 to 4 matches per team.[B]
```

### Raw passage P9

```text
Critics of this format argued that the use of three-team groups with two teams progressing significantly increased the risk of collusion between teams.[31] This prompted FIFA to suggest that penalty shootouts could be used to prevent draws in the group stage,[32] although even then some risk of collusion would remain, as it was possible for the two teams playing the last group match to engineer a circle of death result that would allow both of them to qualify.[31] The tournament format was finalized in 2023, with an announcement that there would be 12 groups of 4 teams.[33][34]
```

#### Before: frozen research / original LiveDemo

**Passage P9 · sentence ID `3d222832ed6f6db6a3764f4be8a3e8d1` · 128 tokens · offsets [0, 584)**

```text
Critics of this format argued that the use of three-team groups with two teams progressing significantly increased the risk of collusion between teams.[31] This prompted FIFA to suggest that penalty shootouts could be used to prevent draws in the group stage,[32] although even then some risk of collusion would remain, as it was possible for the two teams playing the last group match to engineer a circle of death result that would allow both of them to qualify.[31] The tournament format was finalized in 2023, with an announcement that there would be 12 groups of 4 teams.[33][34]
```

#### After: explicit citation/Markdown adapter

**Passage P9 · sentence ID `b96226b873c246e276e896b9ae850ce8` · 29 tokens · offsets [0, 155)**

```text
Critics of this format argued that the use of three-team groups with two teams progressing significantly increased the risk of collusion between teams.[31]
```

**Passage P9 · sentence ID `5c05efed179e4d2781f7fbf1f34fe156` · 65 tokens · offsets [156, 468)**

```text
This prompted FIFA to suggest that penalty shootouts could be used to prevent draws in the group stage,[32] although even then some risk of collusion would remain, as it was possible for the two teams playing the last group match to engineer a circle of death result that would allow both of them to qualify.[31]
```

**Passage P9 · sentence ID `8a1d7dd2ae8742cbb6aa39549c6d397a` · 34 tokens · offsets [469, 584)**

```text
The tournament format was finalized in 2023, with an announcement that there would be 12 groups of 4 teams.[33][34]
```

### Raw passage P10

```text
New rules
The 2026 FIFA World Cup introduced several rule changes, primarily designed to reduce time-wasting.[35] The new rules included:
```

#### Before: frozen research / original LiveDemo

**Passage P10 · sentence ID `1c169987c2f398eea9c217f11cfc98b2` · 33 tokens · offsets [0, 137)**

```text
New rules
The 2026 FIFA World Cup introduced several rule changes, primarily designed to reduce time-wasting.[35] The new rules included:
```

#### After: explicit citation/Markdown adapter

**Passage P10 · sentence ID `b90685a8667971e86d0baa69689eeee0` · 28 tokens · offsets [0, 113)**

```text
New rules
The 2026 FIFA World Cup introduced several rule changes, primarily designed to reduce time-wasting.[35]
```

**Passage P10 · sentence ID `7a218a3431ff13b637bdf965bcbdb238` · 5 tokens · offsets [114, 137)**

```text
The new rules included:
```

### Raw passage P11

```text
10-second substitutions: Players being substituted had ten seconds to exit the field; otherwise, the substitute had to wait for one minute before entering the match.
5-second restarts: A visual 5-second countdown was shown by the referee for throw-ins and goal kicks in situations of time-wasting. If the ball was not put into play in time, a restart was awarded to the opposing team; a corner kick was awarded if the infraction occurred with a goal kick.
Medical treatment: Any outfield player who received medical attention on the field had to leave the field and wait for 1 minute before returning to play.
Expanded video assistant referee (VAR): The VAR could review and overturn wrongly awarded corner kicks, certain attacking fouls, and clear mistakes on red cards given from a second yellow card.
Mouth-covering red cards: To stop confrontational or insulting behavior which was concealed to avoid lip-reading, any player who covered his mouth with his hand, arm, or shirt while confronting an opponent was sent off.
Player leaving the field in protest: Players or officials who left the field in protest were sent off.[35]
Another change was the tiebreaker formula used when two or more teams finish the group stage with an equal number of points. The first tiebreaker criteria was the number of points earned in head-to-head matches between the tied teams, not the goal difference among all matches.[C]
```

#### Before: frozen research / original LiveDemo

**Passage P11 · sentence ID `173ea12e6b723612e0a4583ff8b1ebb9` · 31 tokens · offsets [0, 165)**

```text
10-second substitutions: Players being substituted had ten seconds to exit the field; otherwise, the substitute had to wait for one minute before entering the match.
```

**Passage P11 · sentence ID `05d15144f6a32b6920587e9187066aa9` · 29 tokens · offsets [166, 297)**

```text
5-second restarts: A visual 5-second countdown was shown by the referee for throw-ins and goal kicks in situations of time-wasting.
```

**Passage P11 · sentence ID `257d18c6f94806c07ae3e9a97d26dab1` · 35 tokens · offsets [298, 455)**

```text
If the ball was not put into play in time, a restart was awarded to the opposing team; a corner kick was awarded if the infraction occurred with a goal kick.
```

**Passage P11 · sentence ID `5e5cc09717d3416b86956c92256a3eb4` · 29 tokens · offsets [456, 609)**

```text
Medical treatment: Any outfield player who received medical attention on the field had to leave the field and wait for 1 minute before returning to play.
```

**Passage P11 · sentence ID `1d7795c1b1c26aae328ace0070d5daa6` · 36 tokens · offsets [610, 803)**

```text
Expanded video assistant referee (VAR): The VAR could review and overturn wrongly awarded corner kicks, certain attacking fouls, and clear mistakes on red cards given from a second yellow card.
```

**Passage P11 · sentence ID `0a5b2fe160b1913fee772d706a8da845` · 45 tokens · offsets [804, 1023)**

```text
Mouth-covering red cards: To stop confrontational or insulting behavior which was concealed to avoid lip-reading, any player who covered his mouth with his hand, arm, or shirt while confronting an opponent was sent off.
```

**Passage P11 · sentence ID `9f5e3a671a70a919815af11f86daea0e` · 47 tokens · offsets [1024, 1255)**

```text
Player leaving the field in protest: Players or officials who left the field in protest were sent off.[35]
Another change was the tiebreaker formula used when two or more teams finish the group stage with an equal number of points.
```

**Passage P11 · sentence ID `a562a374d9609316bf9583254b56afe8` · 31 tokens · offsets [1256, 1411)**

```text
The first tiebreaker criteria was the number of points earned in head-to-head matches between the tied teams, not the goal difference among all matches.[C]
```

#### After: explicit citation/Markdown adapter

**Passage P11 · sentence ID `173ea12e6b723612e0a4583ff8b1ebb9` · 31 tokens · offsets [0, 165)**

```text
10-second substitutions: Players being substituted had ten seconds to exit the field; otherwise, the substitute had to wait for one minute before entering the match.
```

**Passage P11 · sentence ID `05d15144f6a32b6920587e9187066aa9` · 29 tokens · offsets [166, 297)**

```text
5-second restarts: A visual 5-second countdown was shown by the referee for throw-ins and goal kicks in situations of time-wasting.
```

**Passage P11 · sentence ID `257d18c6f94806c07ae3e9a97d26dab1` · 35 tokens · offsets [298, 455)**

```text
If the ball was not put into play in time, a restart was awarded to the opposing team; a corner kick was awarded if the infraction occurred with a goal kick.
```

**Passage P11 · sentence ID `5e5cc09717d3416b86956c92256a3eb4` · 29 tokens · offsets [456, 609)**

```text
Medical treatment: Any outfield player who received medical attention on the field had to leave the field and wait for 1 minute before returning to play.
```

**Passage P11 · sentence ID `1d7795c1b1c26aae328ace0070d5daa6` · 36 tokens · offsets [610, 803)**

```text
Expanded video assistant referee (VAR): The VAR could review and overturn wrongly awarded corner kicks, certain attacking fouls, and clear mistakes on red cards given from a second yellow card.
```

**Passage P11 · sentence ID `0a5b2fe160b1913fee772d706a8da845` · 45 tokens · offsets [804, 1023)**

```text
Mouth-covering red cards: To stop confrontational or insulting behavior which was concealed to avoid lip-reading, any player who covered his mouth with his hand, arm, or shirt while confronting an opponent was sent off.
```

**Passage P11 · sentence ID `6b21428dd99c7027dc7db3b0352f11cc` · 23 tokens · offsets [1024, 1130)**

```text
Player leaving the field in protest: Players or officials who left the field in protest were sent off.[35]
```

**Passage P11 · sentence ID `a41994d011bf8d427b037dbda0b88b48` · 24 tokens · offsets [1131, 1255)**

```text
Another change was the tiebreaker formula used when two or more teams finish the group stage with an equal number of points.
```

**Passage P11 · sentence ID `a562a374d9609316bf9583254b56afe8` · 31 tokens · offsets [1256, 1411)**

```text
The first tiebreaker criteria was the number of points earned in head-to-head matches between the tied teams, not the goal difference among all matches.[C]
```

### Raw passage P12

```text
Mandatory breaks
Cooling or hydration breaks were introduced at the 2014 World Cup in Brazil. For the 2026 World Cup, FIFA introduced mandatory three-minute hydration breaks in every half for all matches. Broadcasters were permitted to run commercials during these pauses.[39] Since the clock did not stop during the breaks, the three minutes were automatically included in the stoppage time for each half.[40]
```

#### Before: frozen research / original LiveDemo

**Passage P12 · sentence ID `b1bc7931fe870029a44d5de8f91ca288` · 23 tokens · offsets [0, 93)**

```text
Mandatory breaks
Cooling or hydration breaks were introduced at the 2014 World Cup in Brazil.
```

**Passage P12 · sentence ID `f55c0fe8a115adb87bfda2c292988073` · 24 tokens · offsets [94, 204)**

```text
For the 2026 World Cup, FIFA introduced mandatory three-minute hydration breaks in every half for all matches.
```

**Passage P12 · sentence ID `39cb249ddc9e4489d48fb762ab3c46a9` · 42 tokens · offsets [205, 410)**

```text
Broadcasters were permitted to run commercials during these pauses.[39] Since the clock did not stop during the breaks, the three minutes were automatically included in the stoppage time for each half.[40]
```

#### After: explicit citation/Markdown adapter

**Passage P12 · sentence ID `b1bc7931fe870029a44d5de8f91ca288` · 23 tokens · offsets [0, 93)**

```text
Mandatory breaks
Cooling or hydration breaks were introduced at the 2014 World Cup in Brazil.
```

**Passage P12 · sentence ID `f55c0fe8a115adb87bfda2c292988073` · 24 tokens · offsets [94, 204)**

```text
For the 2026 World Cup, FIFA introduced mandatory three-minute hydration breaks in every half for all matches.
```

**Passage P12 · sentence ID `b0e479e930b6b92385784531d3471708` · 14 tokens · offsets [205, 276)**

```text
Broadcasters were permitted to run commercials during these pauses.[39]
```

**Passage P12 · sentence ID `4da777424c534700b152dcaaaf9b75d1` · 28 tokens · offsets [277, 410)**

```text
Since the clock did not stop during the breaks, the three minutes were automatically included in the stoppage time for each half.[40]
```

### Raw passage P13

```text
Match schedule
The match schedule, without group assignments, was announced on February 4, 2024.[D] On June 13, FIFA released an updated schedule, with specific pairings assigned to venues for the knockout stage.[43] The full schedule was unveiled in a live broadcast on December 6, 2025, the day after the draw.[44] The opening match of the tournament included Mexico, and took place on June 11, 2026, at the Estadio Azteca in Mexico City.[45] Canada's first match occurred on June 12 at BMO Field in Toronto, while the United States' opening match took place on the same day at SoFi Stadium in Inglewood. The national team of each host nation played its three group stage matches in its own country.[41]
```

#### Before: frozen research / original LiveDemo

**Passage P13 · sentence ID `f3c21f0a7ea93c11e0f9145cde0fbb22` · 159 tokens · offsets [0, 606)**

```text
Match schedule
The match schedule, without group assignments, was announced on February 4, 2024.[D] On June 13, FIFA released an updated schedule, with specific pairings assigned to venues for the knockout stage.[43] The full schedule was unveiled in a live broadcast on December 6, 2025, the day after the draw.[44] The opening match of the tournament included Mexico, and took place on June 11, 2026, at the Estadio Azteca in Mexico City.[45] Canada's first match occurred on June 12 at BMO Field in Toronto, while the United States' opening match took place on the same day at SoFi Stadium in Inglewood.
```

**Passage P13 · sentence ID `f3f6606d6297add0a5bf6088b954ca3c` · 21 tokens · offsets [607, 705)**

```text
The national team of each host nation played its three group stage matches in its own country.[41]
```

#### After: explicit citation/Markdown adapter

**Passage P13 · sentence ID `712bdff336d773bfb8b07f5036c39d4a` · 26 tokens · offsets [0, 99)**

```text
Match schedule
The match schedule, without group assignments, was announced on February 4, 2024.[D]
```

**Passage P13 · sentence ID `2744311eccb5784e722048703b8591fe` · 27 tokens · offsets [100, 216)**

```text
On June 13, FIFA released an updated schedule, with specific pairings assigned to venues for the knockout stage.[43]
```

**Passage P13 · sentence ID `bb6975c119fed5cdf3bbb484b651b431` · 29 tokens · offsets [217, 316)**

```text
The full schedule was unveiled in a live broadcast on December 6, 2025, the day after the draw.[44]
```

**Passage P13 · sentence ID `1a1b3f7e00953722b130a070e801dd7e` · 38 tokens · offsets [317, 444)**

```text
The opening match of the tournament included Mexico, and took place on June 11, 2026, at the Estadio Azteca in Mexico City.[45]
```

**Passage P13 · sentence ID `9b21808a4a6158bb233aa047465cc41a` · 39 tokens · offsets [445, 606)**

```text
Canada's first match occurred on June 12 at BMO Field in Toronto, while the United States' opening match took place on the same day at SoFi Stadium in Inglewood.
```

**Passage P13 · sentence ID `f3f6606d6297add0a5bf6088b954ca3c` · 21 tokens · offsets [607, 705)**

```text
The national team of each host nation played its three group stage matches in its own country.[41]
```

### Raw passage P14

```text
AT&T Stadium in Arlington hosted the most matches of any venue, with nine. The U.S. hosted 78 matches, including all quarterfinals, semifinals, and the final, which was played at MetLife Stadium in East Rutherford on July 19. Canada and Mexico each hosted 13 matches. Each tournament venue, except for the Estadio Akron, hosted at least one knockout stage match.[46] The match schedule overlapped with the 2026 Canadian Football League season, resulting in scheduling conflicts and loss of home games for the Toronto Argonauts and BC Lions.[47][48] The match schedule also affected the 2026 Major League Baseball season schedules of the Kansas City Royals, Philadelphia Phillies, Seattle Mariners, and Texas Rangers, with home stadiums near World Cup venues.[49]
```

#### Before: frozen research / original LiveDemo

**Passage P14 · sentence ID `67153991a729b612b261dc7588ddb59d` · 16 tokens · offsets [0, 74)**

```text
AT&T Stadium in Arlington hosted the most matches of any venue, with nine.
```

**Passage P14 · sentence ID `8b8aaae8c1580b18791b119162d5b243` · 40 tokens · offsets [75, 225)**

```text
The U.S. hosted 78 matches, including all quarterfinals, semifinals, and the final, which was played at MetLife Stadium in East Rutherford on July 19.
```

**Passage P14 · sentence ID `021012dea440fd2be21c0fe3518e94c7` · 10 tokens · offsets [226, 267)**

```text
Canada and Mexico each hosted 13 matches.
```

**Passage P14 · sentence ID `b3b30528df9f9c057443ef5d003956e8` · 107 tokens · offsets [268, 762)**

```text
Each tournament venue, except for the Estadio Akron, hosted at least one knockout stage match.[46] The match schedule overlapped with the 2026 Canadian Football League season, resulting in scheduling conflicts and loss of home games for the Toronto Argonauts and BC Lions.[47][48] The match schedule also affected the 2026 Major League Baseball season schedules of the Kansas City Royals, Philadelphia Phillies, Seattle Mariners, and Texas Rangers, with home stadiums near World Cup venues.[49]
```

#### After: explicit citation/Markdown adapter

**Passage P14 · sentence ID `67153991a729b612b261dc7588ddb59d` · 16 tokens · offsets [0, 74)**

```text
AT&T Stadium in Arlington hosted the most matches of any venue, with nine.
```

**Passage P14 · sentence ID `8b8aaae8c1580b18791b119162d5b243` · 40 tokens · offsets [75, 225)**

```text
The U.S. hosted 78 matches, including all quarterfinals, semifinals, and the final, which was played at MetLife Stadium in East Rutherford on July 19.
```

**Passage P14 · sentence ID `021012dea440fd2be21c0fe3518e94c7` · 10 tokens · offsets [226, 267)**

```text
Canada and Mexico each hosted 13 matches.
```

**Passage P14 · sentence ID `af78557d2faa23412f2ccc8cb6a88b51` · 22 tokens · offsets [268, 366)**

```text
Each tournament venue, except for the Estadio Akron, hosted at least one knockout stage match.[46]
```

**Passage P14 · sentence ID `c945e7664f8c205bdf560ac84d52d65f` · 42 tokens · offsets [367, 548)**

```text
The match schedule overlapped with the 2026 Canadian Football League season, resulting in scheduling conflicts and loss of home games for the Toronto Argonauts and BC Lions.[47][48]
```

**Passage P14 · sentence ID `95cc6c04d1cbaf9b8ad6ea24d94e3125` · 43 tokens · offsets [549, 762)**

```text
The match schedule also affected the 2026 Major League Baseball season schedules of the Kansas City Royals, Philadelphia Phillies, Seattle Mariners, and Texas Rangers, with home stadiums near World Cup venues.[49]
```

### Raw passage P15

```text
Host cities were geographically grouped into three regions:[3]
Western Region (Vancouver, Seattle, San Francisco Bay Area, Los Angeles)
Central Region (Guadalajara, Mexico City, Monterrey, Houston, Dallas, Kansas City)
Eastern Region (Atlanta, Miami, Toronto, Boston, Philadelphia, New York/New Jersey)
```

#### Before: frozen research / original LiveDemo

**Passage P15 · sentence ID `3eb7f4850abc0e2d035c2299358f37a2` · 67 tokens · offsets [0, 302)**

```text
Host cities were geographically grouped into three regions:[3]
Western Region (Vancouver, Seattle, San Francisco Bay Area, Los Angeles)
Central Region (Guadalajara, Mexico City, Monterrey, Houston, Dallas, Kansas City)
Eastern Region (Atlanta, Miami, Toronto, Boston, Philadelphia, New York/New Jersey)
```

#### After: explicit citation/Markdown adapter

**Passage P15 · sentence ID `3eb7f4850abc0e2d035c2299358f37a2` · 67 tokens · offsets [0, 302)**

```text
Host cities were geographically grouped into three regions:[3]
Western Region (Vancouver, Seattle, San Francisco Bay Area, Los Angeles)
Central Region (Guadalajara, Mexico City, Monterrey, Houston, Dallas, Kansas City)
Eastern Region (Atlanta, Miami, Toronto, Boston, Philadelphia, New York/New Jersey)
```

## Specific edge-case checks

| Case | Audited behavior |
|---|---|
| Plain multi-sentence paragraph | Adapter and frozen outputs match |
| `.[19]`, `.[19][20]`, `. [19]` | Citation-aware mode splits and retains citation suffixes |
| Letter footnote `[D]`, numeric ranges `[19–21]` | Supported in citation-aware mode |
| Mid-sentence `[13][14]` after a comma | Does not create a new sentence |
| Markdown `# Heading` | Separate, labelled heading segment |
| Markdown bullets/numbered lists | Separate structural regions; sentences inside each region use frozen rules |
| Bare headings such as Format / Expansion | Preserved conservatively with source-segment labelling; no guessed grammar |
| Single-newline wrapped prose | Not split solely because of the line wrap |
| Blank paragraphs | Preserve the frozen paragraph boundary rule |
| `Dr.`, `e.g.`, `U.S.`, decimals | Preserve frozen abbreviation/decimal behavior |

## Verification and reproducibility

Run the source-only audit with:

```bash
demo/.venv/bin/python demo/audit/run_sentence_segmentation_audit.py
```

Run the regression suite with:

```bash
demo/.venv/bin/python -m pytest demo/tests/test_live_segmentation.py demo/tests/test_live.py -q
```

Full machine-readable inventories and protected-source hashes are saved in `demo/audit/live_sentence_segmentation_audit.json`. All raw input text is retained separately in `demo/audit/world_cup_input.txt`. The audit verifies before/after source hashes and exact substring/offset/token integrity without generating answers.

### Protected identities

| File | SHA256 |
|---|---|
| `src/week9/senttopk_e3/sentences.py` | `9ff672d6ee70a51d9bcd274dd5a550590fcb1e61ca5750c866b1fb9b917636e1` |
| `src/week11/hierarchy.py` | `29332314a9e085f590860b4b3a55be38888f0623974f9540c7f1c545ed73d019` |
| `src/week12/algorithm.py` | `433222886e997116cea673df24d32e2a3f81210663cfabcb5b689720f5bb5132` |
| `src/week12/laboratory.py` | `ed4d4c9d696e7a045891a80641aed1922e7d8eaa53374f0d0e59c615b989c604` |
| `demo/replay_data/v1/bundle.json` | `14b4ba43d4cf3ab7359b54f2fc8bfe0ee59166c12f40c09125658c1d63f03036` |
| `results/week12/W13_FINAL_METHOD_FREEZE.json` | `d0188a3582d6aead0a3a31f9f15bad9b51a88a6b475b05ac79ae38e58f05366b` |
| `results/week12/qbgf/v3/W13_FINAL_METHOD_FREEZE.json` | `d0188a3582d6aead0a3a31f9f15bad9b51a88a6b475b05ac79ae38e58f05366b` |

## Completed test results

- Targeted segmentation/live suite: **56 passed**.
- Complete demo regression suite: **123 passed in 35.51 seconds**.
- Registered research-source manifest and frozen-method hashes: verified unchanged.
- No answer generation, GPU model loading or benchmark run was performed.

Restart Streamlit before using the new mode; browser refresh alone can retain imported code/resources. Re-run the custom pipeline after selecting the desired segmentation policy.
