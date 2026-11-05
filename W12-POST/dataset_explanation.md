Yes — once you understand the **structure of the three datasets**, the result becomes much less surprising. The three tasks look superficially like “retrieve evidence and give it to an LLM,” but they require very different kinds of evidence.

The easiest mental model is:

$$
\boxed{
\text{SciFact} \approx \text{find the decisive sentence}
}
$$

$$
\boxed{
\text{HotpotQA} \approx \text{find multiple complementary facts and connect them}
}
$$

$$
\boxed{
\text{QASPER} \approx \text{search a long technical document and assemble an answer}
}
$$

That difference lines up remarkably well with your MiniLM/QBGF/BEA results.

---

# 1. What exactly is HotpotQA?

HotpotQA is a Wikipedia-based **multi-hop question answering** dataset. It was intentionally designed so that questions often require finding and reasoning over **multiple supporting documents**, and it includes sentence-level supporting facts. It also contains comparison questions. ([arXiv][1])

A simplified Hotpot question might look like:

> “The director of Movie A was born in which country?”

To answer it, you may need:

**Document 1**

> Movie A was directed by **John X**.

Then another document:

**Document 2**

> John X was born in **Canada**.

So:

$$
Movie\ A
\rightarrow
John\ X
\rightarrow
Canada
$$

The answer does not necessarily appear in the sentence most similar to the original query.

That's the critical point.

---

# 2. Why MiniLM is weaker on HotpotQA

MiniLM does:

$$
score(q,s_i)
$$

for every sentence independently.

Then:

$$
sort(s_i)
\rightarrow
pack\ highest\ scoring\ sentences.
$$

Suppose the question is:

> “What award was the screenwriter of a Denis Villeneuve thriller nominated for?”

One sentence may say:

> “Enemy is a 2013 film directed by Denis Villeneuve.”

Very high similarity to the query.

Another may say:

> “Javier Gullón wrote the screenplay for Enemy.”

Also relevant.

But the crucial third sentence could be:

> “Gullón received a Goya nomination for *Invader*.”

That third sentence might have **lower direct similarity to the original question**.

Yet it is the sentence required after establishing:

$$
Enemy\rightarrow Gullón.
$$

This is exactly the sort of task HotpotQA was constructed around. ([arXiv][1])

### Why this hurts a static reranker

MiniLM asks:

$$
\boxed{\text{How relevant is this sentence to the original query?}}
$$

It does **not** naturally ask:

$$
\boxed{
\text{Given what I've already selected, does this sentence complete a reasoning chain?}
}
$$

That is a major distinction.

---

# 3. And that makes your Hotpot result very interesting

You obtained:

| HotpotQA | MiniLM | FIXED | QBGF |   V1 |   **V2** |
| -------- | -----: | ----: | ---: | ---: | -------: |
| 256      |   .340 |  .366 | .404 | .298 | **.435** |
| 1024     |   .362 |  .389 | .402 | .344 | **.423** |

MiniLM is actually below even FIXED.

That's surprising initially because a cross-encoder is a stronger relevance estimator.

But it makes sense if:

$$
\boxed{
\text{better independent relevance ranking}
\neq
\text{better multi-hop evidence composition}.
}
$$

BEA-V2, meanwhile, does something conceptually different:

$$
U(s|E)
=
P_{\text{help}}(s)\cdot
\max(0,IG(s|E))
$$

where \(E\) is already selected evidence.

So evidence value can depend on the context already chosen.

That's much closer to the structure Hotpot requires.

### But careful

Your experiment does **not prove** this is why V2 wins.

What you can say is:

> **The result is consistent with HotpotQA requiring complementary/multi-hop evidence that cannot always be captured by static query–sentence relevance alone.**

That's a strong and safe interpretation.

---

# 4. What exactly is SciFact?

SciFact is very different.

It is a **scientific claim verification** dataset.

Given a scientific claim, the system must identify evidence from scientific literature and decide whether that evidence:

$$
\boxed{SUPPORTS}
$$

or

$$
\boxed{REFUTES}
$$

the claim. The dataset also provides evidence rationales. The original dataset contains roughly 1.4K expert-written scientific claims paired with scientific abstracts and evidence annotations. ([arXiv][2])

Your evaluation additionally uses the three-way setting:

$$
SUPPORT,\ REFUTE,\ NOT\_ENOUGH\_INFO.
$$

A simplified example:

### Claim

> “Drug X decreases blood pressure.”

Potential evidence:

> “Administration of Drug X significantly reduced systolic blood pressure.”

That's almost a direct matching task.

Or:

### Claim

> “Gene expression does not vary between genetically identical cells.”

Evidence:

> “Substantial variation in gene expression was observed among genetically identical cells.”

That directly **contradicts** the claim.

---

# 5. SciFact therefore often has a much more localized evidence structure

For many SciFact examples, the useful information may be contained in:

$$
\boxed{1-3\ decisive\ sentences}
$$

rather than requiring a chain like:

$$
A\rightarrow B\rightarrow C.
$$

So the task can resemble:

$$
\boxed{
claim\leftrightarrow evidence\ sentence
}
$$

much more closely.

And this is almost exactly the task a cross-encoder reranker is designed for.

MiniLM-L6-v2 was trained on the **MS MARCO passage-ranking task**, i.e. given a query and candidate passage, estimate how relevant that passage is and rank candidates accordingly. ([Hugging Face][3])

So although SciFact is a different domain and task, the structural form:

$$
\text{claim/query} + \text{candidate sentence}
\rightarrow relevance
$$

fits MiniLM surprisingly well.

---

# 6. And look at your SciFact result

| SciFact | FIXED | QBGF |   V1 |   V2 | **MiniLM** |
| ------- | ----: | ---: | ---: | ---: | ---------: |
| 256     |  .629 | .531 | .613 | .552 |   **.685** |
| 1024    |  .553 | .588 | .643 | .577 |   **.648** |

This is probably not random.

At 256 tokens MiniLM gets:

$$
\boxed{0.6855}
$$

while using basically the same budget as everybody else.

It isn't winning because it has more context.

FIXED:

$$
252.6\ tokens
$$

MiniLM:

$$
252.8\ tokens.
$$

The difference is largely **which sentences occupy those tokens**.

That is powerful evidence that for SciFact:

$$
\boxed{\text{sentence ranking quality matters a lot.}}
$$

---

# 7. Why can QBGF actually hurt SciFact?

Remember what QBGF optimizes.

For passage \(i\):

$$
U_i(A_i)
=
R_i\,m_i(A_i).
$$

It's trying to preserve **query-relevance mass** across retrieved sources and granularities.

But SciFact may not reward broad relevance coverage.

Imagine the claim:

> “Protein X increases Y.”

There may be:

### Sentence A

> “Protein X is involved in the regulation of Y.”

Highly relevant.

### Sentence B

> “Protein X significantly **decreases** Y.”

Also highly relevant — but **decisive**.

### Sentence C

> “Protein X is expressed in tissue Z.”

Also semantically related.

QBGF could preserve lots of relevance mass:

$$
A+C+\ldots
$$

without necessarily prioritizing the one sentence that determines:

$$
SUPPORT\ vs\ REFUTE.
$$

So:

$$
\boxed{
semantic\ relevance\ mass
\neq
label\text{-}decisive\ evidence.
}
$$

That lines up with the LOO attribution finding from your project.

---

# 8. Why did V1 sometimes do reasonably well on SciFact-1024?

This is interesting too:

$$
V1=0.6432
$$

versus:

$$
MiniLM=0.6479.
$$

They are virtually tied by point estimate.

V1 explicitly penalizes redundancy:

$$
IG
=
Rel+
0.5\Delta Cov
-
0.5Red.
$$

For scientific claims, repeatedly feeding very similar sentences may indeed be unnecessary.

So V1 can select:

> one sentence about the intervention,

> one about the outcome,

> one about the observed relation,

instead of dozens of nearly duplicate sentences.

At 1024:

$$
V1\approx696\ tokens
$$

versus:

$$
MiniLM\approx1021.
$$

Yet:

$$
0.6432\quad vs\quad0.6479.
$$

That's actually a nice result.

It suggests V1 may achieve nearly the same SciFact point quality using roughly:

$$
32\%
$$

fewer evidence tokens.

But because the uncertainty is large, don't claim V1 is “equivalent.” Just call it an interesting efficiency pattern.

---

# 9. Then what exactly is QASPER?

QASPER is quite different from both.

It is **question answering over full scientific papers**.

The dataset contains 5,049 questions over 1,585 NLP papers. Importantly, the question writer sees only the paper's **title and abstract**, then asks a question whose answer must be found in the **full text**. A different annotator reads the paper, answers the question, and provides supporting evidence. ([arXiv][4])

Examples resemble:

> “Which datasets do the authors evaluate on?”

> “What baseline methods are compared?”

> “What languages were used?”

> “What architecture did they use for encoding?”

> “What were the five classification tasks?”

The answer could appear in:

* Methods;
* Experiments;
* Dataset section;
* Results;
* table descriptions;
* multiple places in the paper.

And QASPER was deliberately designed to involve information-seeking and sometimes reasoning across multiple parts of a paper. ([arXiv][4])

---

# 10. So QASPER sits somewhere between SciFact and Hotpot

That's why I would think of the tasks like this:

### SciFact

$$
\boxed{\text{localized evidence identification}}
$$

### HotpotQA

$$
\boxed{\text{explicit multi-hop / complementary evidence}}
$$

### QASPER

$$
\boxed{\text{long-document information seeking}}
$$

QASPER can sometimes be simple:

> “What datasets were used?”

Maybe one paragraph contains the answer.

But sometimes it requires collecting several items:

> “Which seven languages?”

or:

> “What are the five classification tasks?”

Then multiple neighboring sentences or a coherent subsection may matter.

---

# 11. This explains why everyone almost ties on QASPER

Your results:

### QASPER 256

$$
FIXED=.2668
$$

$$
MiniLM=.2641
$$

$$
QBGF=.2641
$$

$$
V2=.2618.
$$

That's almost indistinguishable at point-estimate level.

And at 1024:

$$
QBGF=.2964
$$

$$
V2=.2904
$$

$$
FIXED=.2896
$$

$$
MiniLM=.2827.
$$

Again, fairly close.

Why?

One plausible explanation is that once SFR has already given a reasonably good top-10 candidate pool, **many sensible relevance-based selectors retrieve broadly similar useful content**.

So changing:

$$
embedding\ relevance
$$

to:

$$
cross\text{-}encoder\ relevance
$$

doesn't completely transform the evidence.

---

# 12. QASPER also benefits from neighboring context

This matters.

Suppose the answer is:

> “Hindi, English, Kannada, Telugu, Assamese, Bengali and Malayalam.”

One sentence might say:

> “We evaluated the system on seven Indian languages.”

Highly relevant.

But the next sentence/table may actually contain:

> “Hindi, Kannada, Telugu, Malayalam, Bengali, English and Assamese.”

If you rank isolated sentences individually, sentence 1 can score highly while not actually containing the answer.

This is exactly what happened in some of your qualitative examples.

So QASPER creates tension between:

$$
\boxed{\text{finding the correct location}}
$$

and:

$$
\boxed{\text{preserving enough local context once you find it}}.
$$

That helps explain why QBGF/V2 sometimes benefit from richer fidelity at 1024.

---

# 13. Why BEA-V1 failed so badly on QASPER

This is now much easier to explain.

At QASPER-256:

$$
V1=.187
$$

while all the other methods are around:

$$
.26.
$$

Yet V1 used:

$$
245.7/256
$$

tokens.

So:

$$
\boxed{\text{it was NOT simply using too little context.}}
$$

Instead:

* V1 selected ~20.3 units;
* from ~8.78 parents;
* with lower redundancy.

Compare FIXED:

* only ~9.34 units;
* ~5.7 parents.

V1 was spreading the same budget over **many sources and fragments**.

For a full scientific paper, that can be bad.

You may want:

> 3 adjacent sentences from the experiment section,

rather than:

> 1 sentence each from Introduction, Related Work, Methods, Results, Discussion...

So V1's:

$$
-\lambda Redundancy
$$

can inadvertently turn into:

$$
\boxed{\text{too much diversity}}
$$

and therefore:

$$
\boxed{\text{context fragmentation}}.
$$

---

# 14. QASPER-1024 exposes the second V1 problem

At 1024:

$$
V1=.208
$$

and only uses:

$$
473/1024
$$

tokens.

Now the problem becomes:

$$
\boxed{
fragmented selection
+
premature stopping
}
$$

because V1 concludes the remaining candidates have insufficient:

$$
\frac{Rel+Coverage-Redundancy}{Cost}
$$

and stops.

But additional context actually helps the reader.

V2 uses about:

$$
1009/1024
$$

and recovers to:

$$
.290.
$$

So your two QASPER budgets diagnose **two different failure mechanisms**.

---

# 15. The three datasets therefore test almost three different allocation philosophies

This is the most useful table to remember:

| Dataset      | Source                       | Task                   | Typical evidence need                          | What seems important in your experiment             |
| ------------ | ---------------------------- | ---------------------- | ---------------------------------------------- | --------------------------------------------------- |
| **SciFact**  | Scientific abstracts         | Claim verification     | Small number of decisive sentences             | **Strong ranking**                                  |
| **QASPER**   | Full NLP papers              | Information-seeking QA | Locate section + preserve useful local context | **Ranking + coherent context**                      |
| **HotpotQA** | Multiple Wikipedia documents | Multi-hop QA           | Complementary facts across documents           | **Evidence interaction / multi-source composition** |

Dataset descriptions are from the original task papers. ([arXiv][1])

And your winner pattern is:

| Dataset      | Observed winner pattern |
| ------------ | ----------------------- |
| **SciFact**  | MiniLM                  |
| **QASPER**   | FIXED/QBGF/V2 all close |
| **HotpotQA** | BEA-V2                  |

That alignment is actually quite intuitive.

---

# 16. There's also a subtle reason MiniLM fits SciFact better

MiniLM was trained on **MS MARCO passage ranking**. ([Hugging Face][3])

That training teaches it something like:

$$
(q,p)
\rightarrow
\text{Does this passage answer / relate strongly to this query?}
$$

Because it is a **cross-encoder**, it processes the query and candidate together, allowing token-level interaction.

Your original FIXED sentence scorer essentially obtains representations and compares them.

Conceptually:

$$
e_q=f(q),\quad e_s=f(s)
$$

then:

$$
sim(e_q,e_s).
$$

MiniLM instead processes:

$$
[q;\ s]
$$

jointly and predicts relevance.

So it can capture relationships such as:

* negation;
* exact subject/object matches;
* phrases interacting across query and evidence;
* potentially finer matching than cosine similarity.

That may be particularly useful when ranking short scientific evidence sentences.

But MiniLM is still **not a scientific-domain verifier**. So don't claim domain specialization; it was trained on MS MARCO, not SciFact. ([Hugging Face][3])

---

# 17. Why doesn't the same stronger MiniLM ranking improve Hotpot?

Because this:

$$
\boxed{\text{sentence relevance}}
$$

and this:

$$
\boxed{\text{joint usefulness of a set of sentences}}
$$

are fundamentally different functions.

Imagine:

$$
score(q,A)=0.95
$$

$$
score(q,B)=0.90
$$

$$
score(q,C)=0.60.
$$

MiniLM selects:

$$
A,B.
$$

But perhaps \(A\) and \(B\) say essentially the same thing.

Whereas:

$$
A+C
$$

contains the two facts needed for the bridge.

Then individually:

$$
B>C
$$

but jointly:

$$
Utility(A,C)>Utility(A,B).
$$

Static reranking cannot naturally express that because:

$$
score(C)
$$

does not depend on whether \(A\) has already been selected.

BEA explicitly tries to model this:

$$
U(s|E).
$$

That is the conceptual reason Hotpot is especially interesting for your system.

---

# 18. Your results now suggest three different regimes

I would explain this to the TA as:

### Regime 1 — **Relevance-limited**

Example: SciFact.

Main challenge:

$$
\boxed{\text{identify the decisive sentence}}
$$

A strong reranker works very well.

---

### Regime 2 — **Context-coherence-limited**

Example: QASPER.

Main challenge:

$$
\boxed{
\text{find the right region AND retain enough neighboring information}
}
$$

Too much fragmentation hurts.

---

### Regime 3 — **Interaction-limited**

Example: HotpotQA.

Main challenge:

$$
\boxed{
\text{find complementary pieces whose value depends on each other}
}
$$

Independent ranking becomes less adequate.

---

# 19. And this is probably your strongest final research insight

Originally you thought:

$$
\boxed{q\rightarrow k}
$$

“How much evidence does this query need?”

After all these experiments, the better question is:

$$
\boxed{
(q,E,s,\text{task})
\rightarrow
\Delta Utility(s|E)
}
$$

> Given the query, the evidence already selected, and the type of task, **how useful is adding this particular piece of evidence now?**

Because:

* SciFact may need one decisive sentence;
* QASPER may need a coherent local section;
* Hotpot may need a missing bridge fact.

So there cannot necessarily be one universally optimal notion of “relevant evidence.”

And that explains your empirical result:

$$
\boxed{
\text{MiniLM wins SciFact}
}
$$

$$
\boxed{
\text{QASPER methods converge}
}
$$

$$
\boxed{
\text{BEA-V2 leads Hotpot by point estimate}
}
$$

without those results being contradictory.

They are actually telling the same story:

> **Different reasoning tasks induce different evidence-utility structures.**

That's probably the cleanest final interpretation of the entire project.

[1]: https://arxiv.org/abs/1809.09600?utm_source=chatgpt.com "HotpotQA: A Dataset for Diverse, Explainable Multi-hop Question Answering"
[2]: https://arxiv.org/abs/2004.14974?utm_source=chatgpt.com "Fact or Fiction: Verifying Scientific Claims"
[3]: https://huggingface.co/cross-encoder/ms-marco-MiniLM-L6-v2/blob/main/README.md?code=true&utm_source=chatgpt.com "README.md · cross-encoder/ms-marco-MiniLM-L6-v2 at main"
[4]: https://arxiv.org/abs/2105.03011?utm_source=chatgpt.com "A Dataset of Information-Seeking Questions and Answers Anchored in Research Papers"
