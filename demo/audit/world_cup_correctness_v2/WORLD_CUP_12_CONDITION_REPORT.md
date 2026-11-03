# World Cup LiveDemo corrective regression — 12 conditions

These are custom-input diagnostics, not benchmark results. User-supplied source assertions are not fact-checked. No gold scores or answer-quality winner are inferred.

Query: Explain how the expansion of the 2026 World Cup changed both the group-stage structure and the total tournament workload, and explain why FIFA abandoned the originally planned three-team group format.

Device: cuda; common pinned SFR downstream scoring; segmentation: academic-offset-rules-v1+live-citation-markdown-v1. Both candidate retrievers use identical method, budget, reader, prompt template, parser and decoding. Exact prompt contents differ when evidence differs.

| Candidate retriever | Method | Budget | Evidence tokens | Prompt tokens | Generated tokens |
|---|---|---:|---:|---:|---:|
| SFR | RAW_SKIP | 256 | 245 | 316 | 56 |
| SFR | RAW_SKIP | 1024 | 993 | 1064 | 82 |
| SFR | FIXED_SENTENCE | 256 | 256 | 327 | 65 |
| SFR | FIXED_SENTENCE | 1024 | 1020 | 1091 | 70 |
| SFR | QBGF_FULL | 256 | 251 | 322 | 63 |
| SFR | QBGF_FULL | 1024 | 1022 | 1093 | 82 |
| BM25 | RAW_SKIP | 256 | 169 | 240 | 61 |
| BM25 | RAW_SKIP | 1024 | 941 | 1012 | 60 |
| BM25 | FIXED_SENTENCE | 256 | 256 | 327 | 65 |
| BM25 | FIXED_SENTENCE | 1024 | 1024 | 1095 | 79 |
| BM25 | QBGF_FULL | 256 | 246 | 317 | 77 |
| BM25 | QBGF_FULL | 1024 | 1011 | 1082 | 64 |

## SFR · RAW_SKIP · 256

Top-10 IDs: P6, P8, P9, P5, P1, P7, P10, P2, P15, P11

**Answer**

The 2026 World Cup expanded to 48 teams, using 12 groups of 4 teams instead of the originally planned three-team groups, increasing the tournament workload and leading FIFA to abandon the three-team format due to concerns about collusion and draw risks.

**Exact selected evidence**

```text
Starting with the 2026 edition, the World Cup expanded to 48 teams, an increase of 16 teams compared to the previous seven tournaments.[19] As approved by the FIFA Council on March 14, 2023, the teams were split into 12 groups of 4 teams, with the top 2 teams in each group and the 8 best third-place teams progressing to a new round of 32.[20] It was the first expansion and format change since 1998.[21]

Critics of this format argued that the use of three-team groups with two teams progressing significantly increased the risk of collusion between teams.[31] This prompted FIFA to suggest that penalty shootouts could be used to prevent draws in the group stage,[32] although even then some risk of collusion would remain, as it was possible for the two teams playing the last group match to engineer a circle of death result that would allow both of them to qualify.[31] The tournament format was finalized in 2023, with an announcement that there would be 12 groups of 4 teams.[33][34]
```

Unit types: PASSAGE, PASSAGE

**Descriptive phrase inspection (not a metric)**

| Source concept | Candidate-pool passages containing phrase | Selected units containing phrase |
|---|---|---|
| 48-team expansion | P6, P8, P1 | 32ddf2d0a50ce8012deaec895be7037c |
| 12 groups of four | P6, P9 | 32ddf2d0a50ce8012deaec895be7037c, e2698dde51d4f3b2cb647d7b399baeb3 |
| original 16 groups of three | P8 | No literal match |
| 64 to 104 matches | P7 | No literal match |
| 7 to 8 semifinalist matches | P7 | No literal match |
| 39-day duration | P7 | No literal match |
| collusion concern | P9 | e2698dde51d4f3b2cb647d7b399baeb3 |
| penalty shootouts | P9 | e2698dde51d4f3b2cb647d7b399baeb3 |

## SFR · RAW_SKIP · 1024

Top-10 IDs: P6, P8, P9, P5, P1, P7, P10, P2, P15, P11

**Answer**

The 2026 World Cup expanded to 48 teams, changing the group stage from 16 groups of 3 teams to 12 groups of 4 teams, and increased the total tournament workload by adding 40 matches, extending the tournament duration to 39 days. FIFA abandoned the three-team format due to concerns over collusion risks and the potential for drawn group matches.

**Exact selected evidence**

```text
The 2026 FIFA World Cup[A] was the 23rd FIFA World Cup, the quadrennial international men's soccer championship contested by the national teams of the member associations of FIFA. The tournament began on June 11, 2026, and concluded on July 19 with Spain winning the championship for the second time.[3] It was jointly hosted by 16 cities—11 in the United States, 3 in Mexico, and 2 in Canada. The tournament was the first FIFA World Cup to be hosted by three countries and the first to include 48 teams, an expansion from the previous 32-team format.[4]

Format
Expansion
The idea of expanding the tournament had been suggested as early as 2013 by then-UEFA president Michel Platini,[13][14] as well as in 2016 by current FIFA president Gianni Infantino.[15] Opponents of the proposal argued that the number of matches played was already at an unacceptable level, that the expansion would dilute the quality of the matches,[16][17] and that the decision was driven by political rather than sporting concerns, specifically that Infantino was trying to win his election by promising to bring more countries to the World Cup.[18]

Starting with the 2026 edition, the World Cup expanded to 48 teams, an increase of 16 teams compared to the previous seven tournaments.[19] As approved by the FIFA Council on March 14, 2023, the teams were split into 12 groups of 4 teams, with the top 2 teams in each group and the 8 best third-place teams progressing to a new round of 32.[20] It was the first expansion and format change since 1998.[21]

The total number of matches played increased from 64 to 104, and the number of matches played by teams reaching the semifinals increased from 7 to 8. The tournament lasted 39 days, an increase from the 32 days of the 2014 and 2018 tournaments, and from the 29 days of the 2022 tournament,[22][23] although each team still played three group matches.[24][25] The final matchday at club level for players named in the final squads was May 24, 2026; clubs had to release their players by May 25, with exceptions granted to players participating in continental club competition finals up until May 30. The combined rest, release, and tournament periods remained 56 days, the same as in the 2010, 2014, and 2018 World Cups.[20]

Other expansion formats explored
After the expansion to 48 teams was approved on January 10, 2017, it was initially decided that the tournament would include 16 groups of 3 teams, with the top two teams of each group progressing to a round of 32.[19][26] This format would have included 80 matches in total, with the maximum number of matches per team remaining at seven, although each team would have played one fewer group match than in the 2022 tournament. The tournament would still have been completed within 32 days.[27] This format was chosen over three alternatives involving 40 to 48 teams, 76 to 88 matches, and a minimum of 1 to 4 matches per team.[B]

Critics of this format argued that the use of three-team groups with two teams progressing significantly increased the risk of collusion between teams.[31] This prompted FIFA to suggest that penalty shootouts could be used to prevent draws in the group stage,[32] although even then some risk of collusion would remain, as it was possible for the two teams playing the last group match to engineer a circle of death result that would allow both of them to qualify.[31] The tournament format was finalized in 2023, with an announcement that there would be 12 groups of 4 teams.[33][34]

New rules
The 2026 FIFA World Cup introduced several rule changes, primarily designed to reduce time-wasting.[35] The new rules included:

Host cities were geographically grouped into three regions:[3]
Western Region (Vancouver, Seattle, San Francisco Bay Area, Los Angeles)
Central Region (Guadalajara, Mexico City, Monterrey, Houston, Dallas, Kansas City)
Eastern Region (Atlanta, Miami, Toronto, Boston, Philadelphia, New York/New Jersey)
```

Unit types: PASSAGE, PASSAGE, PASSAGE, PASSAGE, PASSAGE, PASSAGE, PASSAGE, PASSAGE

**Descriptive phrase inspection (not a metric)**

| Source concept | Candidate-pool passages containing phrase | Selected units containing phrase |
|---|---|---|
| 48-team expansion | P6, P8, P1 | dad23297fb1bcfc7d2b1df3acb153015, 32ddf2d0a50ce8012deaec895be7037c, 5f942ca8946eee60e72da5757d42ec32 |
| 12 groups of four | P6, P9 | 32ddf2d0a50ce8012deaec895be7037c, e2698dde51d4f3b2cb647d7b399baeb3 |
| original 16 groups of three | P8 | 5f942ca8946eee60e72da5757d42ec32 |
| 64 to 104 matches | P7 | 34563fe62063269c65ffa9e5226aaf4a |
| 7 to 8 semifinalist matches | P7 | 34563fe62063269c65ffa9e5226aaf4a |
| 39-day duration | P7 | 34563fe62063269c65ffa9e5226aaf4a |
| collusion concern | P9 | e2698dde51d4f3b2cb647d7b399baeb3 |
| penalty shootouts | P9 | e2698dde51d4f3b2cb647d7b399baeb3 |

## SFR · FIXED_SENTENCE · 256

Top-10 IDs: P6, P8, P9, P5, P1, P7, P10, P2, P15, P11

**Answer**

The 2026 World Cup expanded to 48 teams, changing the group stage from 16 groups of 3 teams to 12 groups of 4 teams, increasing the total tournament workload, and FIFA abandoned the three-team format due to logistical challenges and the need for a more balanced competition.

**Exact selected evidence**

```text
The tournament was the first FIFA World Cup to be hosted by three countries and the first to include 48 teams, an expansion from the previous 32-team format.[4]

Starting with the 2026 edition, the World Cup expanded to 48 teams, an increase of 16 teams compared to the previous seven tournaments.[19]

As approved by the FIFA Council on March 14, 2023, the teams were split into 12 groups of 4 teams, with the top 2 teams in each group and the 8 best third-place teams progressing to a new round of 32.[20]

It was the first expansion and format change since 1998.[21]

Other expansion formats explored
After the expansion to 48 teams was approved on January 10, 2017, it was initially decided that the tournament would include 16 groups of 3 teams, with the top two teams of each group progressing to a round of 32.[19][26]

The tournament format was finalized in 2023, with an announcement that there would be 12 groups of 4 teams.[33][34]
```

Unit types: SENTENCE, SENTENCE, SENTENCE, SENTENCE, SOURCE SEGMENT, SENTENCE

**Descriptive phrase inspection (not a metric)**

| Source concept | Candidate-pool passages containing phrase | Selected units containing phrase |
|---|---|---|
| 48-team expansion | P6, P8, P1 | c6913b54530f2ddd15f674403f3304c1, 11bdde878be83c3350f7fd9dea539b7a, 96c78cdd99bbdac948b82eafdfe8415c |
| 12 groups of four | P6, P9 | 3014b5368d61d594f24bcdd0e3faf8cf, 8a1d7dd2ae8742cbb6aa39549c6d397a |
| original 16 groups of three | P8 | 96c78cdd99bbdac948b82eafdfe8415c |
| 64 to 104 matches | P7 | No literal match |
| 7 to 8 semifinalist matches | P7 | No literal match |
| 39-day duration | P7 | No literal match |
| collusion concern | P9 | No literal match |
| penalty shootouts | P9 | No literal match |

## SFR · FIXED_SENTENCE · 1024

Top-10 IDs: P6, P8, P9, P5, P1, P7, P10, P2, P15, P11

**Answer**

The 2026 World Cup expanded to 48 teams, changing the group stage to 12 groups of 4 teams instead of 16 groups of 3, and increased the total number of matches from 64 to 104, while FIFA abandoned the three-team format due to concerns over collusion risks.

**Exact selected evidence**

```text
The 2026 FIFA World Cup[A] was the 23rd FIFA World Cup, the quadrennial international men's soccer championship contested by the national teams of the member associations of FIFA.

The tournament was the first FIFA World Cup to be hosted by three countries and the first to include 48 teams, an expansion from the previous 32-team format.[4]

The 2026 tournament was the second World Cup to be hosted by multiple nations, after the 2002 edition.

The event returned to its traditional June–July schedule after the 2022 World Cup in Qatar, which was uniquely held in November and December to avoid Qatar's extreme summer heat.[6]

Canada, Mexico, and the United States qualified automatically as hosts, while Cape Verde, Curaçao, Jordan, and Uzbekistan made their World Cup debuts.[5]

Format
Expansion
The idea of expanding the tournament had been suggested as early as 2013 by then-UEFA president Michel Platini,[13][14] as well as in 2016 by current FIFA president Gianni Infantino.[15]

Opponents of the proposal argued that the number of matches played was already at an unacceptable level, that the expansion would dilute the quality of the matches,[16][17] and that the decision was driven by political rather than sporting concerns, specifically that Infantino was trying to win his election by promising to bring more countries to the World Cup.[18]

Starting with the 2026 edition, the World Cup expanded to 48 teams, an increase of 16 teams compared to the previous seven tournaments.[19]

As approved by the FIFA Council on March 14, 2023, the teams were split into 12 groups of 4 teams, with the top 2 teams in each group and the 8 best third-place teams progressing to a new round of 32.[20]

The total number of matches played increased from 64 to 104, and the number of matches played by teams reaching the semifinals increased from 7 to 8.

The tournament lasted 39 days, an increase from the 32 days of the 2014 and 2018 tournaments, and from the 29 days of the 2022 tournament,[22][23] although each team still played three group matches.[24][25]

The final matchday at club level for players named in the final squads was May 24, 2026; clubs had to release their players by May 25, with exceptions granted to players participating in continental club competition finals up until May 30.

The combined rest, release, and tournament periods remained 56 days, the same as in the 2010, 2014, and 2018 World Cups.[20]

Other expansion formats explored
After the expansion to 48 teams was approved on January 10, 2017, it was initially decided that the tournament would include 16 groups of 3 teams, with the top two teams of each group progressing to a round of 32.[19][26]

This format would have included 80 matches in total, with the maximum number of matches per team remaining at seven, although each team would have played one fewer group match than in the 2022 tournament.

This format was chosen over three alternatives involving 40 to 48 teams, 76 to 88 matches, and a minimum of 1 to 4 matches per team.[B]

Critics of this format argued that the use of three-team groups with two teams progressing significantly increased the risk of collusion between teams.[31]

This prompted FIFA to suggest that penalty shootouts could be used to prevent draws in the group stage,[32] although even then some risk of collusion would remain, as it was possible for the two teams playing the last group match to engineer a circle of death result that would allow both of them to qualify.[31]

The tournament format was finalized in 2023, with an announcement that there would be 12 groups of 4 teams.[33][34]

New rules
The 2026 FIFA World Cup introduced several rule changes, primarily designed to reduce time-wasting.[35]

Another change was the tiebreaker formula used when two or more teams finish the group stage with an equal number of points.

Host cities were geographically grouped into three regions:[3]
Western Region (Vancouver, Seattle, San Francisco Bay Area, Los Angeles)
Central Region (Guadalajara, Mexico City, Monterrey, Houston, Dallas, Kansas City)
Eastern Region (Atlanta, Miami, Toronto, Boston, Philadelphia, New York/New Jersey)
```

Unit types: SENTENCE, SENTENCE, SENTENCE, SENTENCE, SENTENCE, SOURCE SEGMENT, SENTENCE, SENTENCE, SENTENCE, SENTENCE, SENTENCE, SENTENCE, SENTENCE, SOURCE SEGMENT, SENTENCE, SENTENCE, SENTENCE, SENTENCE, SENTENCE, SOURCE SEGMENT, SENTENCE, SOURCE SEGMENT

**Descriptive phrase inspection (not a metric)**

| Source concept | Candidate-pool passages containing phrase | Selected units containing phrase |
|---|---|---|
| 48-team expansion | P6, P8, P1 | c6913b54530f2ddd15f674403f3304c1, 11bdde878be83c3350f7fd9dea539b7a, 96c78cdd99bbdac948b82eafdfe8415c, 1e83b4cfab274c28383f9eb411a83641 |
| 12 groups of four | P6, P9 | 3014b5368d61d594f24bcdd0e3faf8cf, 8a1d7dd2ae8742cbb6aa39549c6d397a |
| original 16 groups of three | P8 | 96c78cdd99bbdac948b82eafdfe8415c |
| 64 to 104 matches | P7 | 168fefcea10d1d787bec4d90dbd4641d |
| 7 to 8 semifinalist matches | P7 | 168fefcea10d1d787bec4d90dbd4641d |
| 39-day duration | P7 | 0ae5336e04e67c701407003c7437ef60 |
| collusion concern | P9 | b96226b873c246e276e896b9ae850ce8, 5c05efed179e4d2781f7fbf1f34fe156 |
| penalty shootouts | P9 | 5c05efed179e4d2781f7fbf1f34fe156 |

## SFR · QBGF_FULL · 256

Top-10 IDs: P6, P8, P9, P5, P1, P7, P10, P2, P15, P11

**Answer**

The 2026 World Cup expanded to 48 teams, using 12 groups of 4 teams instead of the originally planned three-team format, increasing the total tournament workload by adding more matches and requiring more venues and logistics, leading FIFA to abandon the three-team format due to practical challenges.

**Exact selected evidence**

```text
Starting with the 2026 edition, the World Cup expanded to 48 teams, an increase of 16 teams compared to the previous seven tournaments.[19]

As approved by the FIFA Council on March 14, 2023, the teams were split into 12 groups of 4 teams, with the top 2 teams in each group and the 8 best third-place teams progressing to a new round of 32.[20]

It was the first expansion and format change since 1998.[21]

The tournament format was finalized in 2023, with an announcement that there would be 12 groups of 4 teams.[33][34]

New rules
The 2026 FIFA World Cup introduced several rule changes, primarily designed to reduce time-wasting.[35]

The new rules included:

Host cities were geographically grouped into three regions:[3]
Western Region (Vancouver, Seattle, San Francisco Bay Area, Los Angeles)
Central Region (Guadalajara, Mexico City, Monterrey, Houston, Dallas, Kansas City)
Eastern Region (Atlanta, Miami, Toronto, Boston, Philadelphia, New York/New Jersey)
```

Unit types: SENTENCE, SENTENCE, SENTENCE, SENTENCE, SOURCE SEGMENT, SOURCE SEGMENT, SOURCE SEGMENT

**Descriptive phrase inspection (not a metric)**

| Source concept | Candidate-pool passages containing phrase | Selected units containing phrase |
|---|---|---|
| 48-team expansion | P6, P8, P1 | 11bdde878be83c3350f7fd9dea539b7a |
| 12 groups of four | P6, P9 | 3014b5368d61d594f24bcdd0e3faf8cf, 8a1d7dd2ae8742cbb6aa39549c6d397a |
| original 16 groups of three | P8 | No literal match |
| 64 to 104 matches | P7 | No literal match |
| 7 to 8 semifinalist matches | P7 | No literal match |
| 39-day duration | P7 | No literal match |
| collusion concern | P9 | No literal match |
| penalty shootouts | P9 | No literal match |

**Chosen QBGF options**

| Passage | Option | Cost | Utility |
|---|---|---:|---:|
| P6 | 3 selected evidence units | 117 | 0.137774 |
| P8 | DROP | 0 | 0.000000 |
| P9 | 1 selected evidence unit | 34 | 0.045539 |
| P5 | DROP | 0 | 0.000000 |
| P1 | DROP | 0 | 0.000000 |
| P7 | DROP | 0 | 0.000000 |
| P10 | 2 selected evidence units | 33 | 0.085859 |
| P2 | DROP | 0 | 0.000000 |
| P15 | 1 selected evidence unit | 67 | 0.076976 |
| P11 | DROP | 0 | 0.000000 |

## SFR · QBGF_FULL · 1024

Top-10 IDs: P6, P8, P9, P5, P1, P7, P10, P2, P15, P11

**Answer**

The 2026 World Cup expanded to 48 teams, changing the group stage to 12 groups of 4 teams instead of 16 groups of 3, and increased the total tournament workload by adding more matches and extending the tournament duration to 39 days. FIFA abandoned the three-team format due to concerns about collusion risks and opted for the 12-group structure.

**Exact selected evidence**

```text
The 2026 FIFA World Cup[A] was the 23rd FIFA World Cup, the quadrennial international men's soccer championship contested by the national teams of the member associations of FIFA.

The tournament began on June 11, 2026, and concluded on July 19 with Spain winning the championship for the second time.[3]

It was jointly hosted by 16 cities—11 in the United States, 3 in Mexico, and 2 in Canada.

The tournament was the first FIFA World Cup to be hosted by three countries and the first to include 48 teams, an expansion from the previous 32-team format.[4]

The 2026 tournament was the second World Cup to be hosted by multiple nations, after the 2002 edition.

Mexico became the first country to host the World Cup three times, having hosted the 1970 and 1986 tournaments.[5]

The United States previously hosted the World Cup in 1994, while it was Canada's first time hosting the tournament.[5]

The event returned to its traditional June–July schedule after the 2022 World Cup in Qatar, which was uniquely held in November and December to avoid Qatar's extreme summer heat.[6]

Canada, Mexico, and the United States qualified automatically as hosts, while Cape Verde, Curaçao, Jordan, and Uzbekistan made their World Cup debuts.[5]

Format
Expansion
The idea of expanding the tournament had been suggested as early as 2013 by then-UEFA president Michel Platini,[13][14] as well as in 2016 by current FIFA president Gianni Infantino.[15]

Opponents of the proposal argued that the number of matches played was already at an unacceptable level, that the expansion would dilute the quality of the matches,[16][17] and that the decision was driven by political rather than sporting concerns, specifically that Infantino was trying to win his election by promising to bring more countries to the World Cup.[18]

Starting with the 2026 edition, the World Cup expanded to 48 teams, an increase of 16 teams compared to the previous seven tournaments.[19]

As approved by the FIFA Council on March 14, 2023, the teams were split into 12 groups of 4 teams, with the top 2 teams in each group and the 8 best third-place teams progressing to a new round of 32.[20]

It was the first expansion and format change since 1998.[21]

The tournament lasted 39 days, an increase from the 32 days of the 2014 and 2018 tournaments, and from the 29 days of the 2022 tournament,[22][23] although each team still played three group matches.[24][25]

The combined rest, release, and tournament periods remained 56 days, the same as in the 2010, 2014, and 2018 World Cups.[20]

Other expansion formats explored
After the expansion to 48 teams was approved on January 10, 2017, it was initially decided that the tournament would include 16 groups of 3 teams, with the top two teams of each group progressing to a round of 32.[19][26]

This format would have included 80 matches in total, with the maximum number of matches per team remaining at seven, although each team would have played one fewer group match than in the 2022 tournament.

The tournament would still have been completed within 32 days.[27]

This format was chosen over three alternatives involving 40 to 48 teams, 76 to 88 matches, and a minimum of 1 to 4 matches per team.[B]

Critics of this format argued that the use of three-team groups with two teams progressing significantly increased the risk of collusion between teams.[31]

The tournament format was finalized in 2023, with an announcement that there would be 12 groups of 4 teams.[33][34]

New rules
The 2026 FIFA World Cup introduced several rule changes, primarily designed to reduce time-wasting.[35]

The new rules included:

Another change was the tiebreaker formula used when two or more teams finish the group stage with an equal number of points.

Host cities were geographically grouped into three regions:[3]
Western Region (Vancouver, Seattle, San Francisco Bay Area, Los Angeles)
Central Region (Guadalajara, Mexico City, Monterrey, Houston, Dallas, Kansas City)
Eastern Region (Atlanta, Miami, Toronto, Boston, Philadelphia, New York/New Jersey)
```

Unit types: SENTENCE, SENTENCE, SENTENCE, SENTENCE, SENTENCE, SENTENCE, SENTENCE, SENTENCE, SENTENCE, SOURCE SEGMENT, SENTENCE, SENTENCE, SENTENCE, SENTENCE, SENTENCE, SENTENCE, SOURCE SEGMENT, SENTENCE, SENTENCE, SENTENCE, SENTENCE, SENTENCE, SOURCE SEGMENT, SOURCE SEGMENT, SENTENCE, SOURCE SEGMENT

**Descriptive phrase inspection (not a metric)**

| Source concept | Candidate-pool passages containing phrase | Selected units containing phrase |
|---|---|---|
| 48-team expansion | P6, P8, P1 | c6913b54530f2ddd15f674403f3304c1, 11bdde878be83c3350f7fd9dea539b7a, 96c78cdd99bbdac948b82eafdfe8415c, 1e83b4cfab274c28383f9eb411a83641 |
| 12 groups of four | P6, P9 | 3014b5368d61d594f24bcdd0e3faf8cf, 8a1d7dd2ae8742cbb6aa39549c6d397a |
| original 16 groups of three | P8 | 96c78cdd99bbdac948b82eafdfe8415c |
| 64 to 104 matches | P7 | No literal match |
| 7 to 8 semifinalist matches | P7 | No literal match |
| 39-day duration | P7 | 0ae5336e04e67c701407003c7437ef60 |
| collusion concern | P9 | b96226b873c246e276e896b9ae850ce8 |
| penalty shootouts | P9 | No literal match |

**Chosen QBGF options**

| Passage | Option | Cost | Utility |
|---|---|---:|---:|
| P6 | 3 selected evidence units | 117 | 0.137774 |
| P8 | 4 selected evidence units | 169 | 0.133861 |
| P9 | 2 selected evidence units | 63 | 0.077853 |
| P5 | 2 selected evidence units | 132 | 0.099944 |
| P1 | 4 selected evidence units | 142 | 0.098558 |
| P7 | 2 selected evidence units | 112 | 0.055411 |
| P10 | 2 selected evidence units | 33 | 0.085859 |
| P2 | 5 selected evidence units | 163 | 0.085117 |
| P15 | 1 selected evidence unit | 67 | 0.076976 |
| P11 | 1 selected evidence unit | 24 | 0.015652 |

## BM25 · RAW_SKIP · 256

Top-10 IDs: P8, P9, P1, P6, P7, P13, P3, P5, P14, P2

**Answer**

The 2026 World Cup expanded to 48 teams, using 16 groups of 3 teams with a top-two progression, resulting in 80 matches and a 32-day tournament, but FIFA abandoned this format due to logistical challenges and opted for a different structure.

**Exact selected evidence**

```text
Other expansion formats explored
After the expansion to 48 teams was approved on January 10, 2017, it was initially decided that the tournament would include 16 groups of 3 teams, with the top two teams of each group progressing to a round of 32.[19][26] This format would have included 80 matches in total, with the maximum number of matches per team remaining at seven, although each team would have played one fewer group match than in the 2022 tournament. The tournament would still have been completed within 32 days.[27] This format was chosen over three alternatives involving 40 to 48 teams, 76 to 88 matches, and a minimum of 1 to 4 matches per team.[B]
```

Unit types: PASSAGE

**Descriptive phrase inspection (not a metric)**

| Source concept | Candidate-pool passages containing phrase | Selected units containing phrase |
|---|---|---|
| 48-team expansion | P8, P1, P6 | 5f942ca8946eee60e72da5757d42ec32 |
| 12 groups of four | P9, P6 | No literal match |
| original 16 groups of three | P8 | 5f942ca8946eee60e72da5757d42ec32 |
| 64 to 104 matches | P7 | No literal match |
| 7 to 8 semifinalist matches | P7 | No literal match |
| 39-day duration | P7 | No literal match |
| collusion concern | P9 | No literal match |
| penalty shootouts | P9 | No literal match |

## BM25 · RAW_SKIP · 1024

Top-10 IDs: P8, P9, P1, P6, P7, P13, P3, P5, P14, P2

**Answer**

The 2026 World Cup expanded to 48 teams, using 12 groups of 4 teams instead of the initially planned 16 groups of 3, leading to more matches and a longer tournament, while FIFA abandoned the three-team format due to risks of collusion.

**Exact selected evidence**

```text
The 2026 FIFA World Cup[A] was the 23rd FIFA World Cup, the quadrennial international men's soccer championship contested by the national teams of the member associations of FIFA. The tournament began on June 11, 2026, and concluded on July 19 with Spain winning the championship for the second time.[3] It was jointly hosted by 16 cities—11 in the United States, 3 in Mexico, and 2 in Canada. The tournament was the first FIFA World Cup to be hosted by three countries and the first to include 48 teams, an expansion from the previous 32-team format.[4]

Starting with the 2026 edition, the World Cup expanded to 48 teams, an increase of 16 teams compared to the previous seven tournaments.[19] As approved by the FIFA Council on March 14, 2023, the teams were split into 12 groups of 4 teams, with the top 2 teams in each group and the 8 best third-place teams progressing to a new round of 32.[20] It was the first expansion and format change since 1998.[21]

The total number of matches played increased from 64 to 104, and the number of matches played by teams reaching the semifinals increased from 7 to 8. The tournament lasted 39 days, an increase from the 32 days of the 2014 and 2018 tournaments, and from the 29 days of the 2022 tournament,[22][23] although each team still played three group matches.[24][25] The final matchday at club level for players named in the final squads was May 24, 2026; clubs had to release their players by May 25, with exceptions granted to players participating in continental club competition finals up until May 30. The combined rest, release, and tournament periods remained 56 days, the same as in the 2010, 2014, and 2018 World Cups.[20]

Other expansion formats explored
After the expansion to 48 teams was approved on January 10, 2017, it was initially decided that the tournament would include 16 groups of 3 teams, with the top two teams of each group progressing to a round of 32.[19][26] This format would have included 80 matches in total, with the maximum number of matches per team remaining at seven, although each team would have played one fewer group match than in the 2022 tournament. The tournament would still have been completed within 32 days.[27] This format was chosen over three alternatives involving 40 to 48 teams, 76 to 88 matches, and a minimum of 1 to 4 matches per team.[B]

Critics of this format argued that the use of three-team groups with two teams progressing significantly increased the risk of collusion between teams.[31] This prompted FIFA to suggest that penalty shootouts could be used to prevent draws in the group stage,[32] although even then some risk of collusion would remain, as it was possible for the two teams playing the last group match to engineer a circle of death result that would allow both of them to qualify.[31] The tournament format was finalized in 2023, with an announcement that there would be 12 groups of 4 teams.[33][34]

Match schedule
The match schedule, without group assignments, was announced on February 4, 2024.[D] On June 13, FIFA released an updated schedule, with specific pairings assigned to venues for the knockout stage.[43] The full schedule was unveiled in a live broadcast on December 6, 2025, the day after the draw.[44] The opening match of the tournament included Mexico, and took place on June 11, 2026, at the Estadio Azteca in Mexico City.[45] Canada's first match occurred on June 12 at BMO Field in Toronto, while the United States' opening match took place on the same day at SoFi Stadium in Inglewood. The national team of each host nation played its three group stage matches in its own country.[41]
```

Unit types: PASSAGE, PASSAGE, PASSAGE, PASSAGE, PASSAGE, PASSAGE

**Descriptive phrase inspection (not a metric)**

| Source concept | Candidate-pool passages containing phrase | Selected units containing phrase |
|---|---|---|
| 48-team expansion | P8, P1, P6 | dad23297fb1bcfc7d2b1df3acb153015, 32ddf2d0a50ce8012deaec895be7037c, 5f942ca8946eee60e72da5757d42ec32 |
| 12 groups of four | P9, P6 | 32ddf2d0a50ce8012deaec895be7037c, e2698dde51d4f3b2cb647d7b399baeb3 |
| original 16 groups of three | P8 | 5f942ca8946eee60e72da5757d42ec32 |
| 64 to 104 matches | P7 | 34563fe62063269c65ffa9e5226aaf4a |
| 7 to 8 semifinalist matches | P7 | 34563fe62063269c65ffa9e5226aaf4a |
| 39-day duration | P7 | 34563fe62063269c65ffa9e5226aaf4a |
| collusion concern | P9 | e2698dde51d4f3b2cb647d7b399baeb3 |
| penalty shootouts | P9 | e2698dde51d4f3b2cb647d7b399baeb3 |

## BM25 · FIXED_SENTENCE · 256

Top-10 IDs: P8, P9, P1, P6, P7, P13, P3, P5, P14, P2

**Answer**

The 2026 World Cup expanded to 48 teams, changing the group stage from 16 groups of 3 teams to 12 groups of 4 teams, increasing the total tournament workload, and FIFA abandoned the three-team format due to logistical challenges and the need for a more balanced competition.

**Exact selected evidence**

```text
The tournament was the first FIFA World Cup to be hosted by three countries and the first to include 48 teams, an expansion from the previous 32-team format.[4]

Starting with the 2026 edition, the World Cup expanded to 48 teams, an increase of 16 teams compared to the previous seven tournaments.[19]

As approved by the FIFA Council on March 14, 2023, the teams were split into 12 groups of 4 teams, with the top 2 teams in each group and the 8 best third-place teams progressing to a new round of 32.[20]

It was the first expansion and format change since 1998.[21]

Other expansion formats explored
After the expansion to 48 teams was approved on January 10, 2017, it was initially decided that the tournament would include 16 groups of 3 teams, with the top two teams of each group progressing to a round of 32.[19][26]

The tournament format was finalized in 2023, with an announcement that there would be 12 groups of 4 teams.[33][34]
```

Unit types: SENTENCE, SENTENCE, SENTENCE, SENTENCE, SOURCE SEGMENT, SENTENCE

**Descriptive phrase inspection (not a metric)**

| Source concept | Candidate-pool passages containing phrase | Selected units containing phrase |
|---|---|---|
| 48-team expansion | P8, P1, P6 | c6913b54530f2ddd15f674403f3304c1, 11bdde878be83c3350f7fd9dea539b7a, 96c78cdd99bbdac948b82eafdfe8415c |
| 12 groups of four | P9, P6 | 3014b5368d61d594f24bcdd0e3faf8cf, 8a1d7dd2ae8742cbb6aa39549c6d397a |
| original 16 groups of three | P8 | 96c78cdd99bbdac948b82eafdfe8415c |
| 64 to 104 matches | P7 | No literal match |
| 7 to 8 semifinalist matches | P7 | No literal match |
| 39-day duration | P7 | No literal match |
| collusion concern | P9 | No literal match |
| penalty shootouts | P9 | No literal match |

## BM25 · FIXED_SENTENCE · 1024

Top-10 IDs: P8, P9, P1, P6, P7, P13, P3, P5, P14, P2

**Answer**

The 2026 World Cup expanded to 48 teams, changing the group stage to 12 groups of 4 teams instead of 16 groups of 3, increasing the total number of matches from 64 to 104 and extending the tournament duration to 39 days, while FIFA abandoned the three-team format due to concerns over collusion risks.

**Exact selected evidence**

```text
The 2026 FIFA World Cup[A] was the 23rd FIFA World Cup, the quadrennial international men's soccer championship contested by the national teams of the member associations of FIFA.

The tournament was the first FIFA World Cup to be hosted by three countries and the first to include 48 teams, an expansion from the previous 32-team format.[4]

The 2026 tournament was the second World Cup to be hosted by multiple nations, after the 2002 edition.

Mexico became the first country to host the World Cup three times, having hosted the 1970 and 1986 tournaments.[5]

The event returned to its traditional June–July schedule after the 2022 World Cup in Qatar, which was uniquely held in November and December to avoid Qatar's extreme summer heat.[6]

Canada, Mexico, and the United States qualified automatically as hosts, while Cape Verde, Curaçao, Jordan, and Uzbekistan made their World Cup debuts.[5]

Format
Expansion
The idea of expanding the tournament had been suggested as early as 2013 by then-UEFA president Michel Platini,[13][14] as well as in 2016 by current FIFA president Gianni Infantino.[15]

Opponents of the proposal argued that the number of matches played was already at an unacceptable level, that the expansion would dilute the quality of the matches,[16][17] and that the decision was driven by political rather than sporting concerns, specifically that Infantino was trying to win his election by promising to bring more countries to the World Cup.[18]

Starting with the 2026 edition, the World Cup expanded to 48 teams, an increase of 16 teams compared to the previous seven tournaments.[19]

As approved by the FIFA Council on March 14, 2023, the teams were split into 12 groups of 4 teams, with the top 2 teams in each group and the 8 best third-place teams progressing to a new round of 32.[20]

It was the first expansion and format change since 1998.[21]

The total number of matches played increased from 64 to 104, and the number of matches played by teams reaching the semifinals increased from 7 to 8.

The tournament lasted 39 days, an increase from the 32 days of the 2014 and 2018 tournaments, and from the 29 days of the 2022 tournament,[22][23] although each team still played three group matches.[24][25]

The final matchday at club level for players named in the final squads was May 24, 2026; clubs had to release their players by May 25, with exceptions granted to players participating in continental club competition finals up until May 30.

The combined rest, release, and tournament periods remained 56 days, the same as in the 2010, 2014, and 2018 World Cups.[20]

Other expansion formats explored
After the expansion to 48 teams was approved on January 10, 2017, it was initially decided that the tournament would include 16 groups of 3 teams, with the top two teams of each group progressing to a round of 32.[19][26]

This format would have included 80 matches in total, with the maximum number of matches per team remaining at seven, although each team would have played one fewer group match than in the 2022 tournament.

The tournament would still have been completed within 32 days.[27]

This format was chosen over three alternatives involving 40 to 48 teams, 76 to 88 matches, and a minimum of 1 to 4 matches per team.[B]

Critics of this format argued that the use of three-team groups with two teams progressing significantly increased the risk of collusion between teams.[31]

This prompted FIFA to suggest that penalty shootouts could be used to prevent draws in the group stage,[32] although even then some risk of collusion would remain, as it was possible for the two teams playing the last group match to engineer a circle of death result that would allow both of them to qualify.[31]

The tournament format was finalized in 2023, with an announcement that there would be 12 groups of 4 teams.[33][34]

On June 13, FIFA released an updated schedule, with specific pairings assigned to venues for the knockout stage.[43]

The national team of each host nation played its three group stage matches in its own country.[41]

Canada and Mexico each hosted 13 matches.
```

Unit types: SENTENCE, SENTENCE, SENTENCE, SENTENCE, SENTENCE, SENTENCE, SOURCE SEGMENT, SENTENCE, SENTENCE, SENTENCE, SENTENCE, SENTENCE, SENTENCE, SENTENCE, SENTENCE, SOURCE SEGMENT, SENTENCE, SENTENCE, SENTENCE, SENTENCE, SENTENCE, SENTENCE, SENTENCE, SENTENCE, SENTENCE

**Descriptive phrase inspection (not a metric)**

| Source concept | Candidate-pool passages containing phrase | Selected units containing phrase |
|---|---|---|
| 48-team expansion | P8, P1, P6 | c6913b54530f2ddd15f674403f3304c1, 11bdde878be83c3350f7fd9dea539b7a, 96c78cdd99bbdac948b82eafdfe8415c, 1e83b4cfab274c28383f9eb411a83641 |
| 12 groups of four | P9, P6 | 3014b5368d61d594f24bcdd0e3faf8cf, 8a1d7dd2ae8742cbb6aa39549c6d397a |
| original 16 groups of three | P8 | 96c78cdd99bbdac948b82eafdfe8415c |
| 64 to 104 matches | P7 | 168fefcea10d1d787bec4d90dbd4641d |
| 7 to 8 semifinalist matches | P7 | 168fefcea10d1d787bec4d90dbd4641d |
| 39-day duration | P7 | 0ae5336e04e67c701407003c7437ef60 |
| collusion concern | P9 | b96226b873c246e276e896b9ae850ce8, 5c05efed179e4d2781f7fbf1f34fe156 |
| penalty shootouts | P9 | 5c05efed179e4d2781f7fbf1f34fe156 |

## BM25 · QBGF_FULL · 256

Top-10 IDs: P8, P9, P1, P6, P7, P13, P3, P5, P14, P2

**Answer**

The 2026 World Cup expanded to 48 teams, using 12 groups of 4 teams instead of the originally planned three-team format, increasing the tournament workload by adding 16 teams and changing the group-stage structure to include more matches and teams. FIFA abandoned the three-team format due to logistical challenges and the need for a more manageable tournament structure.

**Exact selected evidence**

```text
The tournament was the first FIFA World Cup to be hosted by three countries and the first to include 48 teams, an expansion from the previous 32-team format.[4]

Format
Expansion
The idea of expanding the tournament had been suggested as early as 2013 by then-UEFA president Michel Platini,[13][14] as well as in 2016 by current FIFA president Gianni Infantino.[15]

Starting with the 2026 edition, the World Cup expanded to 48 teams, an increase of 16 teams compared to the previous seven tournaments.[19]

As approved by the FIFA Council on March 14, 2023, the teams were split into 12 groups of 4 teams, with the top 2 teams in each group and the 8 best third-place teams progressing to a new round of 32.[20]

It was the first expansion and format change since 1998.[21]

The tournament format was finalized in 2023, with an announcement that there would be 12 groups of 4 teams.[33][34]
```

Unit types: SENTENCE, SOURCE SEGMENT, SENTENCE, SENTENCE, SENTENCE, SENTENCE

**Descriptive phrase inspection (not a metric)**

| Source concept | Candidate-pool passages containing phrase | Selected units containing phrase |
|---|---|---|
| 48-team expansion | P8, P1, P6 | c6913b54530f2ddd15f674403f3304c1, 11bdde878be83c3350f7fd9dea539b7a |
| 12 groups of four | P9, P6 | 3014b5368d61d594f24bcdd0e3faf8cf, 8a1d7dd2ae8742cbb6aa39549c6d397a |
| original 16 groups of three | P8 | No literal match |
| 64 to 104 matches | P7 | No literal match |
| 7 to 8 semifinalist matches | P7 | No literal match |
| 39-day duration | P7 | No literal match |
| collusion concern | P9 | No literal match |
| penalty shootouts | P9 | No literal match |

**Chosen QBGF options**

| Passage | Option | Cost | Utility |
|---|---|---:|---:|
| P8 | DROP | 0 | 0.000000 |
| P9 | 1 selected evidence unit | 34 | 0.048165 |
| P1 | 1 selected evidence unit | 37 | 0.040206 |
| P6 | 3 selected evidence units | 117 | 0.145717 |
| P7 | DROP | 0 | 0.000000 |
| P13 | DROP | 0 | 0.000000 |
| P3 | DROP | 0 | 0.000000 |
| P5 | 1 selected evidence unit | 58 | 0.057428 |
| P14 | DROP | 0 | 0.000000 |
| P2 | DROP | 0 | 0.000000 |

## BM25 · QBGF_FULL · 1024

Top-10 IDs: P8, P9, P1, P6, P7, P13, P3, P5, P14, P2

**Answer**

The 2026 World Cup expanded to 48 teams, changing the group stage to 12 groups of 4 teams instead of 16 groups of 3, and increased the tournament duration to 39 days, while FIFA abandoned the three-team format due to concerns over collusion risks.

**Exact selected evidence**

```text
The 2026 FIFA World Cup[A] was the 23rd FIFA World Cup, the quadrennial international men's soccer championship contested by the national teams of the member associations of FIFA.

The tournament began on June 11, 2026, and concluded on July 19 with Spain winning the championship for the second time.[3]

It was jointly hosted by 16 cities—11 in the United States, 3 in Mexico, and 2 in Canada.

The tournament was the first FIFA World Cup to be hosted by three countries and the first to include 48 teams, an expansion from the previous 32-team format.[4]

The 2026 tournament was the second World Cup to be hosted by multiple nations, after the 2002 edition.

Mexico became the first country to host the World Cup three times, having hosted the 1970 and 1986 tournaments.[5]

The United States previously hosted the World Cup in 1994, while it was Canada's first time hosting the tournament.[5]

The event returned to its traditional June–July schedule after the 2022 World Cup in Qatar, which was uniquely held in November and December to avoid Qatar's extreme summer heat.[6]

Canada, Mexico, and the United States qualified automatically as hosts, while Cape Verde, Curaçao, Jordan, and Uzbekistan made their World Cup debuts.[5]

Format
Expansion
The idea of expanding the tournament had been suggested as early as 2013 by then-UEFA president Michel Platini,[13][14] as well as in 2016 by current FIFA president Gianni Infantino.[15]

Opponents of the proposal argued that the number of matches played was already at an unacceptable level, that the expansion would dilute the quality of the matches,[16][17] and that the decision was driven by political rather than sporting concerns, specifically that Infantino was trying to win his election by promising to bring more countries to the World Cup.[18]

Starting with the 2026 edition, the World Cup expanded to 48 teams, an increase of 16 teams compared to the previous seven tournaments.[19]

As approved by the FIFA Council on March 14, 2023, the teams were split into 12 groups of 4 teams, with the top 2 teams in each group and the 8 best third-place teams progressing to a new round of 32.[20]

It was the first expansion and format change since 1998.[21]

The tournament lasted 39 days, an increase from the 32 days of the 2014 and 2018 tournaments, and from the 29 days of the 2022 tournament,[22][23] although each team still played three group matches.[24][25]

The combined rest, release, and tournament periods remained 56 days, the same as in the 2010, 2014, and 2018 World Cups.[20]

Other expansion formats explored
After the expansion to 48 teams was approved on January 10, 2017, it was initially decided that the tournament would include 16 groups of 3 teams, with the top two teams of each group progressing to a round of 32.[19][26]

This format would have included 80 matches in total, with the maximum number of matches per team remaining at seven, although each team would have played one fewer group match than in the 2022 tournament.

The tournament would still have been completed within 32 days.[27]

This format was chosen over three alternatives involving 40 to 48 teams, 76 to 88 matches, and a minimum of 1 to 4 matches per team.[B]

Critics of this format argued that the use of three-team groups with two teams progressing significantly increased the risk of collusion between teams.[31]

This prompted FIFA to suggest that penalty shootouts could be used to prevent draws in the group stage,[32] although even then some risk of collusion would remain, as it was possible for the two teams playing the last group match to engineer a circle of death result that would allow both of them to qualify.[31]

The tournament format was finalized in 2023, with an announcement that there would be 12 groups of 4 teams.[33][34]

On June 13, FIFA released an updated schedule, with specific pairings assigned to venues for the knockout stage.[43]

The national team of each host nation played its three group stage matches in its own country.[41]
```

Unit types: SENTENCE, SENTENCE, SENTENCE, SENTENCE, SENTENCE, SENTENCE, SENTENCE, SENTENCE, SENTENCE, SOURCE SEGMENT, SENTENCE, SENTENCE, SENTENCE, SENTENCE, SENTENCE, SENTENCE, SOURCE SEGMENT, SENTENCE, SENTENCE, SENTENCE, SENTENCE, SENTENCE, SENTENCE, SENTENCE, SENTENCE

**Descriptive phrase inspection (not a metric)**

| Source concept | Candidate-pool passages containing phrase | Selected units containing phrase |
|---|---|---|
| 48-team expansion | P8, P1, P6 | c6913b54530f2ddd15f674403f3304c1, 11bdde878be83c3350f7fd9dea539b7a, 96c78cdd99bbdac948b82eafdfe8415c, 1e83b4cfab274c28383f9eb411a83641 |
| 12 groups of four | P9, P6 | 3014b5368d61d594f24bcdd0e3faf8cf, 8a1d7dd2ae8742cbb6aa39549c6d397a |
| original 16 groups of three | P8 | 96c78cdd99bbdac948b82eafdfe8415c |
| 64 to 104 matches | P7 | No literal match |
| 7 to 8 semifinalist matches | P7 | No literal match |
| 39-day duration | P7 | 0ae5336e04e67c701407003c7437ef60 |
| collusion concern | P9 | b96226b873c246e276e896b9ae850ce8, 5c05efed179e4d2781f7fbf1f34fe156 |
| penalty shootouts | P9 | 5c05efed179e4d2781f7fbf1f34fe156 |

**Chosen QBGF options**

| Passage | Option | Cost | Utility |
|---|---|---:|---:|
| P8 | 4 selected evidence units | 169 | 0.141579 |
| P9 | 3 selected evidence units | 128 | 0.113278 |
| P1 | 4 selected evidence units | 142 | 0.104240 |
| P6 | 3 selected evidence units | 117 | 0.145717 |
| P7 | 2 selected evidence units | 112 | 0.058605 |
| P13 | 2 selected evidence units | 48 | 0.029688 |
| P3 | DROP | 0 | 0.000000 |
| P5 | 2 selected evidence units | 132 | 0.105706 |
| P14 | DROP | 0 | 0.000000 |
| P2 | 5 selected evidence units | 163 | 0.090024 |

Full ranking, selected IDs/text, evidence types, exact token IDs, all QBGF options, per-unit selected status and invariant metadata are retained in each condition JSON. Previous pre-fix answers are not a method-quality baseline; any difference can result from segmentation/scoring changes.
