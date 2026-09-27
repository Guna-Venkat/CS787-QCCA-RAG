#!/usr/bin/env python3
"""
compile_planv2_pdf.py
Generates the publication-grade PDF for Project Plan v2.0 using WeasyPrint with clean math typography.
"""

import weasyprint

html_content = """<!DOCTYPE html>
<html>
<head>
<meta charset='utf-8'>
<title>QCCA Project Plan v2.0 (Locked Specification)</title>
<style>
  @page {
    size: a4;
    margin: 18mm 16mm 20mm 16mm;
    @top-left {
      content: 'QCCA: Query-Conditioned Context Allocator';
      font-family: 'Helvetica Neue', Helvetica, Arial, sans-serif;
      font-size: 8pt;
      font-weight: bold;
      color: #1e293b;
    }
    @top-right {
      content: 'CS787 Research Project Plan v2.0 (Locked)';
      font-family: 'Helvetica Neue', Helvetica, Arial, sans-serif;
      font-size: 8pt;
      color: #64748b;
    }
    @bottom-center {
      content: 'Page ' counter(page) ' of ' counter(pages);
      font-family: 'Helvetica Neue', Helvetica, Arial, sans-serif;
      font-size: 8pt;
      color: #64748b;
    }
  }

  body {
    font-family: 'Times New Roman', Times, serif;
    font-size: 9.8pt;
    line-height: 1.42;
    color: #0f172a;
  }

  h1 {
    font-family: 'Helvetica Neue', Helvetica, Arial, sans-serif;
    font-size: 15.5pt;
    font-weight: 800;
    text-align: center;
    color: #0f2b5c;
    margin: 0 0 4px 0;
  }

  .subtitle {
    text-align: center;
    font-family: 'Helvetica Neue', Helvetica, Arial, sans-serif;
    font-size: 10.8pt;
    font-weight: 600;
    color: #0284c7;
    margin-bottom: 2px;
  }

  .meta {
    text-align: center;
    font-size: 9pt;
    color: #475569;
    margin-bottom: 12px;
    border-bottom: 1.5px solid #0f2b5c;
    padding-bottom: 6px;
  }

  h2 {
    font-family: 'Helvetica Neue', Helvetica, Arial, sans-serif;
    font-size: 11pt;
    font-weight: 700;
    color: #0f2b5c;
    border-bottom: 1px solid #cbd5e1;
    padding-bottom: 2px;
    margin-top: 13px;
    margin-bottom: 5px;
    page-break-after: avoid;
  }

  h3 {
    font-family: 'Helvetica Neue', Helvetica, Arial, sans-serif;
    font-size: 9.5pt;
    font-weight: 700;
    color: #0369a1;
    margin-top: 8px;
    margin-bottom: 3px;
    page-break-after: avoid;
  }

  .box {
    background: #f8fafc;
    border: 1px solid #cbd5e1;
    border-left: 4px solid #0f2b5c;
    padding: 7px 11px;
    margin: 8px 0;
    font-size: 9.5pt;
    page-break-inside: avoid;
  }

  .box-status {
    background: #faf5ff;
    border: 1px solid #e9d5ff;
    border-left: 4px solid #7e22ce;
    padding: 7px 11px;
    margin: 8px 0;
    font-size: 9.5pt;
    page-break-inside: avoid;
  }

  .box-hyp {
    background: #f0fdf4;
    border: 1px solid #bbf7d0;
    border-left: 4px solid #16a34a;
    padding: 6px 10px;
    margin: 6px 0;
    font-size: 9.5pt;
    page-break-inside: avoid;
  }

  .box-warn {
    background: #fef2f2;
    border: 1px solid #fecaca;
    border-left: 4px solid #dc2626;
    padding: 6px 10px;
    margin: 6px 0;
    font-size: 9.5pt;
    page-break-inside: avoid;
  }

  table {
    width: 100%;
    border-collapse: collapse;
    margin: 8px 0;
    font-size: 8.5pt;
    page-break-inside: avoid;
  }

  th, td {
    border: 1px solid #cbd5e1;
    padding: 4px 6px;
    text-align: left;
    vertical-align: top;
  }

  th {
    background: #f1f5f9;
    font-weight: bold;
    color: #0f2b5c;
  }

  tr:nth-child(even) td {
    background: #f8fafc;
  }

  code {
    background: #f1f5f9;
    padding: 1px 3px;
    font-family: 'Courier New', Courier, monospace;
    font-size: 8.5pt;
    border-radius: 2px;
  }

  ul, ol {
    margin: 3px 0 6px 0;
    padding-left: 18px;
  }

  li {
    margin-bottom: 2px;
  }

  .diagram {
    background: #f8fafc;
    border: 1px solid #e2e8f0;
    padding: 6px;
    text-align: center;
    font-family: 'Courier New', Courier, monospace;
    font-size: 8.5pt;
    margin: 8px 0;
    page-break-inside: avoid;
    line-height: 1.35;
  }

  .math-block {
    text-align: center;
    margin: 6px 0;
    font-style: italic;
    font-size: 9.5pt;
  }
</style>
</head>
<body>

<h1>Query-Conditioned Evidence Allocation for Efficient RAG</h1>
<div class='subtitle'>PROJECT PLAN v2.0 &mdash; Locked Methodological Specification</div>
<div class='meta'>
  <strong>Short Name:</strong> QCCA (Query-Conditioned Context Allocator) &bull; 
  <strong>Course:</strong> CS787 Graduate Research Project &bull; 
  <strong>Primary Benchmark:</strong> <code>allenai/qasper</code> &bull; 
  <strong>Target Hardware:</strong> 1x NVIDIA TITAN RTX (24GB VRAM)
</div>

<div class='box-status'>
  <strong>PROJECT STATUS: LOCKED EXPERIMENTAL PLAN</strong><br>
  <strong>Status:</strong> LOCKED EXPERIMENTAL SPECIFICATION (v2.0). This document serves as the authoritative methodological specification for the research project. Implementation details may be updated only when strictly required by engineering constraints or discovered bugs. Any methodological alteration must be explicitly justified and documented before final-test evaluation.
</div>

<h2>1. Executive Summary &amp; Core Research Question</h2>
<div class='box'>
  <strong>Core Question:</strong> <em>Does retrieval-side structure provide enough information to estimate how much natural-language evidence a query requires, and can a lightweight query-conditioned allocator dynamically choose the amount of uncompressed evidence while preserving answer quality and reducing context cost?</em>
  <br><br>
  <strong>Central Premise:</strong> Conventional RAG fixes the retrieved context size <em>n</em> in natural language for every query. SARA introduces context compression via a dual representation: Top-<em>k</em> passages as verbatim natural-language text, remaining 10 &minus; <em>k</em> passages compressed as dense projected representations. <strong>QCCA asks:</strong> Instead of fixing <em>k</em>=5 globally for every query, can we predict <em>k</em>(<em>q</em>) &isin; {2, 4, 6, 8} using inexpensive features available immediately after retrieval, prior to expensive neural compression or autoregressive generation?
</div>

<div class='diagram'>
Query (q) &rarr; BM25 Retriever &rarr; Top-10 Passages + Scores &rarr; <strong>QCCA Allocator</strong><br>
&darr;<br>
[Top-k(q) passages &rarr; Full Natural-Language Text] &nbsp;+&nbsp; [Remaining 10-k(q) passages &rarr; SFR Projected Embeddings (&lang;COMPRESS&rang;)]<br>
&darr;<br>
XMistral-7B Generator &rarr; <strong>Answer</strong>
</div>

<p><strong>Core Scientific Chain to Test:</strong> (1) Different queries exhibit heterogeneous sensitivity to context amount; (2) The optimal/acceptable evidence budget <em>k</em><sup>*</sup>(<em>q</em>) therefore varies across queries; (3) Retrieval-side statistics contain predictive signal about this variation; (4) A lightweight allocator can exploit this signal before compression/generation; (5) If successful, adaptive allocation improves the quality-efficiency tradeoff over the best static allocation. <em>(A negative empirical result is fully scientifically valid).</em></p>

<h2>2. Research Motivation</h2>
<p>The overarching research theme is <strong>Query-Conditioned Resource Allocation</strong>.</p>
<ul>
  <li><strong>Prior Paradigm:</strong> Internal Transformer computation allocation (predicting required Transformer layers or routing across mixture-of-experts).</li>
  <li><strong>Current Paradigm:</strong> Evidence and context allocation (predicting required uncompressed natural-language evidence vs. compressed representations).</li>
</ul>
<p>The current work focuses strictly on external evidence allocation prior to generation, testing whether costly context inflation can be mitigated using early retrieval signals.</p>

<h2>3. Research Hypotheses &amp; Falsification Conditions</h2>
<div class='box-hyp'>
  <strong>H1 &mdash; Query-Dependent Evidence Requirement:</strong> Different queries exhibit different sensitivities to the volume of natural-language evidence available to the generator. Consequently, the optimal/acceptable evidence budget <em>k</em><sup>*</sup>(<em>q</em>) varies across queries.
</div>
<div class='box-hyp'>
  <strong>H2 &mdash; Retrieval Structure Contains Predictive Signal:</strong> Features derived from the retrieval-score distribution (top-1 concentration, top-1/top-2 margin, retrieval entropy, passage score counts, query length, and surface features) can predict the required natural-language evidence.
</div>
<div class='box-hyp'>
  <strong>H3 &mdash; Adaptive Allocation Improves Quality-Efficiency Tradeoff:</strong> A query-conditioned allocator achieves comparable or superior answer quality relative to a strong fixed-<em>k</em> baseline while reducing context token overhead. The critical controlled comparison is <strong>QCCA vs. Best Static <em>k</em></strong>, not merely against parent SARA <em>k</em>=5.
</div>

<p><strong>Falsification Conditions:</strong> The central research hypothesis is falsified if any of the following occur: (a) acceptable/optimal <em>k</em><sup>*</sup> exhibits negligible variation across queries; (b) retrieval-side features provide negligible predictive signal regarding evidence sufficiency; (c) QCCA fails to improve the quality-efficiency tradeoff relative to the best static allocation; or (d) any apparent empirical advantage disappears under paper-level clustered statistical testing.</p>

<h2>4. Scientific Problem Formulation</h2>
<p>For a query <em>q</em>, the retriever returns candidate passages <em>R</em>(<em>q</em>) = {<em>p</em><sub>1</sub>, <em>p</em><sub>2</sub>, &hellip;, <em>p</em><sub>10</sub>} with descending BM25 retrieval scores <em>s</em><sub>1</sub> &ge; <em>s</em><sub>2</sub> &ge; &hellip; &ge; <em>s</em><sub>10</sub>. The SARA context representation is parameterized by <em>k</em> &isin; {0, 2, 4, 5, 6, 8, 10}:</p>
<div class='math-block'>
C<sub><em>k</em></sub>(<em>q</em>) = [ <em>p</em><sub>1</sub>, &hellip;, <em>p</em><sub><em>k</em></sub> ]<sub>text</sub> &comp; [ &pi;(SFR(<em>p</em><sub><em>k</em>+1</sub>)), &hellip;, &pi;(SFR(<em>p</em><sub>10</sub>)) ]<sub>&lang;COMPRESS&rang;</sub>
</div>
<p>The allocation policy is defined as <em>k</em>(<em>q</em>) = <em>f</em>(<em>X</em>(<em>q</em>)).</p>
<div class='box-warn'>
  <strong>Pre-Generation Information Invariant:</strong> <em>X</em>(<em>q</em>) contains <strong>strictly</strong> information available after retrieval and before compression or generation. <em>X</em>(<em>q</em>) cannot contain: gold answers, gold evidence spans, generated answers, downstream F1/ROUGE metrics, human evaluations, or any post-generation signals.
</div>

<h2>5. Primary Dataset &amp; Scope Control</h2>
<ul>
  <li><strong>Benchmark:</strong> QASPER (<code>allenai/qasper</code>), consisting of scientific paper QA over long full-text research articles.</li>
  <li><strong>Rationale:</strong> Requires evidence-grounded answers over long contexts, making it ideal for studying retrieval, compression, evidence sufficiency, and adaptive allocation.</li>
  <li><strong>Scope Control:</strong> The 8-week CS787 research project focuses exclusively on QASPER. Cross-domain benchmarks are publication extensions, not core requirements.</li>
</ul>

<h2>6. Data Splits and Leakage Control</h2>
<div class='box-warn'>
  <strong>Critical Rule &mdash; Document-Level Disjointness:</strong> All partitioning is performed strictly at the <strong>paper level</strong> (<code>paper_id</code>), never at the question level. Questions from the same paper must never appear across different development/test partitions.
</div>

<table>
  <tr><th>Partition Split</th><th>Approx. Papers</th><th>Approx. Questions</th><th>Designated Operational Scope</th></tr>
  <tr><td><strong>Discovery</strong></td><td>&sim; 11</td><td>&sim; 35</td><td>Early sensitivity checks, code debugging, and exploratory distributions. Excluded from final allocator training.</td></tr>
  <tr><td><strong>Development</strong></td><td>&sim; 90</td><td>&sim; 320</td><td>Fixed-<em>k</em> evaluation, <em>k</em><sub>dev</sub><sup>*</sup> selection, oracle construction, feature extraction, &epsilon; selection, GroupKFold validation.</td></tr>
  <tr><td><strong>Final Test</strong></td><td>&sim; 180</td><td>&sim; 650</td><td><strong>Frozen evaluation.</strong> Touched strictly once after all models, features, prompts, and code are frozen.</td></tr>
</table>

<h2>7. Existing SARA Implementation &amp; Hardware Constraints</h2>
<ul>
  <li><strong>Base Repository:</strong> <code>SARA-main</code>.</li>
  <li><strong>Retriever:</strong> BM25 through LlamaIndex, <em>n</em>=10 passages, SentenceSplitter (chunk_size=256, overlap=0).</li>
  <li><strong>Generator:</strong> <code>mistralai/Mistral-7B-Instruct-v0.2</code> with XMistral 2-layer MLP projector.</li>
  <li><strong>Compressor:</strong> <code>Salesforce/SFR-Embedding-Mistral</code> (<em>d</em>=4096).</li>
  <li><strong>Special Configurations:</strong> <em>k</em>=10 (vanilla RAG, no compression), <em>k</em>=5 (parent SARA), <em>k</em>=0 (pure compression).</li>
</ul>

<h3>Hardware Constraint (NVIDIA TITAN RTX, 24GB VRAM)</h3>
<p>Measured GPU memory requirements: Mistral-7B Generator + Projector requires &sim; 13.56 GB; SFR Compressor requires &sim; 13.24 GB. Combined footprint: 13.56 + 13.24 = 26.80 GB &gt; 24.0 GB (Causes CUDA Out-Of-Memory).
<br><strong>Execution Strategy:</strong> We execute sequential pipeline stages: (1) Load SFR, precompute/cache passage embeddings to disk, unload SFR; (2) Load Mistral/SARA, run generation using cached representations.</p>

<h2>8. Baseline Hierarchy</h2>
<ol>
  <li><strong>Baseline 1 (Vanilla RAG, <em>k</em>=10):</strong> Full natural-language context; no compression.</li>
  <li><strong>Baseline 2 (Pure Compression, <em>k</em>=0):</strong> All passages compressed; measures performance floor.</li>
  <li><strong>Baseline 3 (Parent SARA, <em>k</em>=5):</strong> Official fixed allocation reproduction.</li>
  <li><strong>Baseline 4 (Best Static SARA, <em>k</em>=<em>k</em><sub>dev</sub><sup>*</sup>):</strong> Best static <em>k</em> &isin; {2, 4, 5, 6, 8} selected on development data using the <em>same</em> quality-efficiency objective. <strong>(Primary controlled comparison)</strong>.</li>
  <li><strong>Baseline 5 (Random Allocation):</strong> Reviewer defense baseline sampling <em>k</em> ~ {2, 4, 6, 8} to verify that gains stem from intelligent allocation rather than random context variance.</li>
  <li><strong>Related Work (RECOMP):</strong> Treated as conceptual related work. RECOMP uses FLAN-UL2 (20B) with incompatible chunking/generators; forcing a non-native reproduction is avoided.</li>
</ol>

<h2>9. Fixed-<em>k</em> Experimental Matrix</h2>
<p>Every development question is evaluated across the full spectrum: <em>k</em> &isin; {0, 2, 4, 5, 6, 8, 10}.
<br>Recording per question-<em>k</em> pair: Token F1, Exact Match (EM), ROUGE-L, input tokens, output tokens, compression latency (<em>L</em><sub>comp</sub>), generation latency (<em>L</em><sub>gen</sub>), selected <em>k</em>, retrieval scores, retrieval features, and peak VRAM. This produces the complete |<em>Q</em><sub>dev</sub>| &times; |<em>K</em>| evaluation matrix.</p>

<h2>10. Primary Optimization Objective &amp; Dual Oracles</h2>
<h3>Epsilon-Constrained Quality-Efficiency Formulation</h3>
<div class='math-block'>
<em>k</em><sup>*</sup><sub>&epsilon;</sub>(<em>q</em>) = min { <em>k</em> &isin; <em>K</em> &nbsp;|&nbsp; F1(<em>q</em>, <em>k</em>) &ge; max<sub><em>j</em> &isin; <em>K</em></sub> F1(<em>q</em>, <em>j</em>) &minus; &epsilon; }
</div>
<p>Candidate values &epsilon; &isin; {0.00, 0.01, 0.02, 0.05} are evaluated on development data. Once selected, &epsilon; is frozen.</p>

<h3>Two Diagnostic Oracles</h3>
<ul>
  <li><strong>Quality Oracle:</strong> <em>k</em><sup>*</sup><sub>qual</sub>(<em>q</em>) = argmax<sub><em>k</em></sub> F1(<em>q</em>, <em>k</em>) (ties broken to smaller <em>k</em>). Theoretical upper bound diagnostic.</li>
  <li><strong>Epsilon Oracle:</strong> <em>k</em><sup>*</sup><sub>&epsilon;</sub>(<em>q</em>) as defined above. Efficiency-aware training target and benchmark.</li>
</ul>
<p><em>Note: Oracles rely on observed post-generation quality and serve exclusively as offline diagnostics.</em></p>

<h2>11. Oracle Gap Analysis</h2>
<p>We quantify three critical performance gaps:</p>
<ul>
  <li><strong>Oracle &rarr; Best Static:</strong> Amount of cross-query heterogeneity available for adaptive policies.</li>
  <li><strong>Oracle &rarr; QCCA:</strong> Remaining allocator optimization gap.</li>
  <li><strong>QCCA &rarr; Best Static:</strong> Empirical value of adaptive evidence allocation.</li>
</ul>
<p>If Oracle &approx; Best Static, the data contains little heterogeneity for any allocator to exploit.</p>

<h2>12. Evaluation Metrics &amp; Semantic Validation Audit</h2>
<ul>
  <li><strong>Primary Quality Metric:</strong> Official QASPER token-level F1.</li>
  <li><strong>Secondary Quality Metrics:</strong> ROUGE-L, Exact Match (EM).</li>
  <li><strong>Efficiency Metrics:</strong> Input tokens, context reduction percentage, compression latency, generation latency, synthesized model-resident latency, and mean selected <em>k</em>.</li>
  <li><strong>Semantic Metric Audit:</strong> Blinded human evaluation of 50 development questions at <em>k</em> &isin; {2, 5, 10} scored as 0 = incorrect, 1 = partially correct, 2 = fully correct to verify correlation with Token F1.</li>
</ul>

<h2>13. Retrieval Features &amp; BM25 Normalization</h2>
<p>Given ranked BM25 scores <em>s</em><sub>1</sub> &ge; <em>s</em><sub>2</sub> &ge; &hellip; &ge; <em>s</em><sub>10</sub>, we apply non-negative query-relative normalization: <em>s</em>&prime;<sub><em>i</em></sub> = <em>s</em><sub><em>i</em></sub> &minus; min(<em>s</em>) + &epsilon;<sub>num</sub>. Unit tests verify robustness against all-zero, equal, negative, and dominant score distributions.</p>

<ol>
  <li><strong>Top-1 Concentration:</strong> &rho;<sub>1</sub> = <em>s</em>&prime;<sub>1</sub> / (&sum;<sub><em>i</em>=1..10</sub> <em>s</em>&prime;<sub><em>i</em></sub>)</li>
  <li><strong>Top-1 / Top-2 Margin:</strong> &Delta;<sub>12</sub> = (<em>s</em>&prime;<sub>1</sub> &minus; <em>s</em>&prime;<sub>2</sub>) / (<em>s</em>&prime;<sub>1</sub> + &epsilon;<sub>num</sub>)</li>
  <li><strong>Retrieval Entropy:</strong> <em>p</em><sub><em>i</em></sub> = exp(<em>s</em>&prime;<sub><em>i</em></sub> / &tau;) / &sum;<sub><em>j</em></sub> exp(<em>s</em>&prime;<sub><em>j</em></sub> / &tau;), <em>H</em> = &minus;&sum;<sub><em>i</em>=1..10</sub> <em>p</em><sub><em>i</em></sub> ln <em>p</em><sub><em>i</em></sub>, <em>H</em><sub>norm</sub> = <em>H</em> / ln(10)</li>
  <li><strong>Query Length:</strong> Number of tokens/words in query <em>q</em>.</li>
  <li><strong>High-Score Passage Count (<em>N</em><sub>high</sub>):</strong> Number of passages where <em>s</em>&prime;<sub><em>i</em></sub> &ge; &theta;<sub>rel</sub> &middot; <em>s</em>&prime;<sub>1</sub>.</li>
  <li><strong>Query Surface Category:</strong> Heuristic indicator (Boolean, Factoid, Explanatory).</li>
  <li><strong>Raw BM25 Score (<em>s</em><sub>1</sub>):</strong> Optional ablation feature to test raw magnitude utility.</li>
</ol>

<h2>14. QCCA Architecture &amp; Grouped Validation</h2>
<p>Candidate allocation space: <em>k</em>(<em>q</em>) &isin; {2, 4, 6, 8}.</p>
<ul>
  <li><strong>QCCA-Rule:</strong> Piecewise policy mapping high concentration (&rho;<sub>1</sub> &ge; &theta;<sub>high</sub>) to smaller <em>k</em> (<em>k</em>=2), moderate concentration to <em>k</em> &isin; {4, 6}, and diffuse retrieval to <em>k</em>=8. Thresholds learned strictly on development data.</li>
  <li><strong>QCCA-Learned:</strong> L2-regularized multinomial Logistic Regression (primary) and shallow Decision Tree (secondary), targeting <em>k</em><sup>*</sup><sub>&epsilon;</sub>.</li>
  <li><strong>GroupKFold Validation:</strong> 5-fold cross-validation grouped strictly by <code>paper_id</code> to prevent document-level information leakage.</li>
</ul>

<h2>15. Retrieval-Ceiling Analysis</h2>
<p>To avoid conflating retrieval and allocation failures, we partition errors:</p>
<ul>
  <li><strong>Retrieval Failure:</strong> Golden evidence is absent from the top-10 retrieved set (determined via QASPER ground-truth evidence annotations).</li>
  <li><strong>Allocation Failure:</strong> Golden evidence was retrieved, but QCCA selected an insufficient <em>k</em>, compressing key evidence.</li>
  <li><strong>Generation Failure:</strong> Relevant evidence was presented in natural language, but the generator produced an incorrect answer.</li>
</ul>
<p>Performance is reported separately on retrieval-success and retrieval-failure subsets.</p>

<h2>16. Latency Methodology: Synthesized Model-Resident Latency</h2>
<p>To prevent hardware sequential swapping from distorting algorithmic benchmarking:</p>
<ul>
  <li>Measure <em>L</em><sub>comp</sub>(<em>k</em>) with SFR pre-loaded.</li>
  <li>Measure <em>L</em><sub>gen</sub>(<em>k</em>) with Mistral pre-loaded.</li>
  <li>Compute <strong>Synthesized Model-Resident Latency</strong>:</li>
</ul>
<div class='math-block'>
<em>L</em><sub>total</sub>(<em>q</em>) = <em>L</em><sub>comp</sub>(10 &minus; <em>k</em>(<em>q</em>)) + <em>L</em><sub>gen</sub>(<em>k</em>(<em>q</em>))
</div>
<p>Excludes model loading/unloading, disk I/O, and sequential PCIe overheads.</p>

<h2>17. Master 8-Week Execution Timeline</h2>
<table>
  <tr><th>Week</th><th>Phase Focus</th><th>Core Engineering &amp; Scientific Deliverables</th></tr>
  <tr><td><strong>W1</strong></td><td>Infrastructure</td><td>Freeze Python/library versions; reproduce SARA unit tests and inference; verify sequential GPU memory; generate paper-level split manifests; build feature extractor and SFR caching harness. <strong>[Go/No-Go Gate]</strong></td></tr>
  <tr><td><strong>W2</strong></td><td>Dev Fixed-<em>k</em> Matrix</td><td>Run complete <em>k</em> &isin; {0, 2, 4, 5, 6, 8, 10} sweep on Development set; log quality, token, and latency metrics; establish Vanilla RAG, Pure Compression, and Parent SARA baselines.</td></tr>
  <tr><td><strong>W3</strong></td><td>Oracle &amp; Metrics</td><td>Compute <em>k</em><sup>*</sup><sub>qual</sub> and <em>k</em><sup>*</sup><sub>&epsilon;</sub>; analyze optimal <em>k</em> distributions; verify hypothesis H1; conduct blinded 50-question human audit; select and freeze &epsilon;<sup>*</sup> on development data.</td></tr>
  <tr><td><strong>W4</strong></td><td>Allocator Development</td><td>Train QCCA-Rule and QCCA-Learned using GroupKFold; analyze feature importance; execute full dry-run of final test harness using a dummy random allocator.</td></tr>
  <tr><td><strong>W5</strong></td><td>Methodology Freeze</td><td><strong>LOCK ALL</strong> features, models, &epsilon;<sup>*</sup>, thresholds, static baseline <em>k</em><sub>dev</sub><sup>*</sup>, prompts, and code. Execute Final Test set evaluation across all 9 systems.</td></tr>
  <tr><td><strong>W6</strong></td><td>Statistical Analysis</td><td>Execute paper-clustered bootstrap (<em>B</em>=10,000 resamples) for 95% CIs; calculate Wilcoxon signed-rank tests; analyze isolated and synthesized latency profiles.</td></tr>
  <tr><td><strong>W7</strong></td><td>Ablations &amp; Failures</td><td>Run 3-tier ablations (static sensitivity, feature complexity, leave-one-out); Pareto frontier analysis (F1 vs. tokens/latency); deterministic 5-category failure taxonomy.</td></tr>
  <tr><td><strong>W8</strong></td><td>Manuscript &amp; Wrap-up</td><td>Finalize publication-ready LaTeX tables and Pareto figures; complete CS787 final report, presentation slides, and open-source reproducibility artifacts.</td></tr>
</table>

<h2>18. Final Test Evaluation Matrix</h2>
<table>
  <tr><th>System Arm</th><th>Retriever</th><th>Compressor</th><th>Generator</th><th>Allocation <em>k</em></th><th>Experimental Role</th></tr>
  <tr><td>1. Vanilla RAG</td><td>BM25</td><td>None</td><td>Mistral-7B</td><td>10</td><td>Full natural-language ceiling</td></tr>
  <tr><td>2. Pure Compression</td><td>BM25</td><td>SFR</td><td>Mistral-7B</td><td>0</td><td>Compression performance floor</td></tr>
  <tr><td>3. Parent SARA</td><td>BM25</td><td>SFR</td><td>Mistral-7B</td><td>5</td><td>Fixed parent architecture</td></tr>
  <tr><td>4. Best Static SARA</td><td>BM25</td><td>SFR</td><td>Mistral-7B</td><td><em>k</em><sub>dev</sub><sup>*</sup></td><td>Primary static control baseline</td></tr>
  <tr><td>5. Random Allocation</td><td>BM25</td><td>SFR</td><td>Mistral-7B</td><td><em>k</em> ~ {2,4,6,8}</td><td>Allocation variance control</td></tr>
  <tr><td>6. <strong>QCCA-Rule</strong></td><td>BM25</td><td>SFR</td><td>Mistral-7B</td><td><em>k</em>(<em>q</em>)</td><td>Interpretable rule policy</td></tr>
  <tr><td>7. <strong>QCCA-Learned</strong></td><td>BM25</td><td>SFR</td><td>Mistral-7B</td><td><em>k</em>(<em>q</em>)</td><td>Learned ML policy</td></tr>
  <tr><td>8. Quality Oracle</td><td>BM25</td><td>SFR</td><td>Mistral-7B</td><td><em>k</em><sup>*</sup><sub>qual</sub>(<em>q</em>)</td><td>Theoretical quality ceiling</td></tr>
  <tr><td>9. Epsilon Oracle</td><td>BM25</td><td>SFR</td><td>Mistral-7B</td><td><em>k</em><sup>*</sup><sub>&epsilon;</sub>(<em>q</em>)</td><td>Efficiency-aware upper bound</td></tr>
</table>

<h2>19. Reporting Schema &amp; Statistical Inference</h2>
<p><strong>Main Results Schema:</strong> System | Token F1 | ROUGE-L | EM | Input Tokens | Context Reduction % | Mean <em>k</em> | <em>L</em><sub>comp</sub> | <em>L</em><sub>gen</sub> | <em>L</em><sub>total</sub> | Peak VRAM.</p>
<ul>
  <li><strong>Paper-Clustered Bootstrap:</strong> <em>B</em>=10,000 iterations. Papers are sampled with replacement, evaluating all constituent questions to obtain paired difference distributions &Delta; = Metric(QCCA) &minus; Metric(Baseline) with 95% empirical confidence intervals.</li>
  <li><strong>Non-Parametric Validation:</strong> Wilcoxon signed-rank test on paired per-question differences.</li>
</ul>

<h2>20. Pareto Analysis &amp; Ablation Matrix</h2>
<ul>
  <li><strong>Primary Pareto Frontier:</strong> Context Input Tokens (<em>X</em>) vs. Token F1 (<em>Y</em>).</li>
  <li><strong>Secondary Pareto Frontier:</strong> Synthesized Model-Resident Latency (<em>X</em>) vs. Token F1 (<em>Y</em>).</li>
</ul>
<p><strong>Ablation Hierarchy:</strong></p>
<ul>
  <li><strong>Tier 1 (Required):</strong> Static-<em>k</em> sensitivity (<em>k</em> &isin; {0, 2, 4, 5, 6, 8, 10}), QCCA vs. Best Static, Oracle gap, Pareto analysis.</li>
  <li><strong>Tier 2 (Important):</strong> Concentration-only features vs. full feature set, leave-one-feature-out, random allocation control, query-type breakdown (Boolean, Factoid, Explanatory).</li>
  <li><strong>Tier 3 (Optional):</strong> Entropy temperature (&tau;) sensitivity, qualitative error inspection.</li>
</ul>

<h2>21. Deterministic Failure Analysis Taxonomy</h2>
<p>Failure cases are identified via quantitative rules and classified into five categories:</p>
<ol>
  <li><strong>Over-Compression:</strong> Selected <em>k</em> is smaller than <em>k</em><sup>*</sup><sub>&epsilon;</sub>; relevant evidence was excessively compressed.</li>
  <li><strong>Under-Compression:</strong> Selected <em>k</em> is larger than necessary, inflating context tokens without quality gain.</li>
  <li><strong>Retrieval Failure:</strong> Golden evidence absent from top-10 passages (retriever-bounded).</li>
  <li><strong>Generation Failure:</strong> Relevant evidence was uncompressed, but Mistral-7B produced an incorrect answer.</li>
  <li><strong>Misleading Distribution:</strong> High retrieval score concentration on distractor passages caused false allocator confidence.</li>
</ol>

<h2>22. Project Risk Matrix &amp; Guardrails</h2>
<table>
  <tr><th>Risk Factor</th><th>Root Cause / Mechanism</th><th>Engineering &amp; Methodological Mitigation</th></tr>
  <tr><td><strong>GPU OOM</strong></td><td>Concurrent SFR + Mistral footprint (26.80 GB) exceeds 24GB VRAM.</td><td>Enforce two-stage sequential execution; precompute and cache passage embeddings to disk.</td></tr>
  <tr><td><strong>Allocator Overfitting</strong></td><td>Small development set (&sim; 320 questions) with expressive models.</td><td>Enforce GroupKFold by paper; apply strong L2 regularization; restrict decision trees to shallow depth.</td></tr>
  <tr><td><strong>Best Static &approx; QCCA</strong></td><td>Retrieval statistics contain insufficient predictive signal.</td><td>Valid negative scientific finding. Report honestly, document oracle gap, and characterize limits.</td></tr>
  <tr><td><strong>Data Leakage</strong></td><td>Cross-contamination of paper topics across splits.</td><td>Strict paper-level partitioning verified by automated manifest assertions.</td></tr>
  <tr><td><strong>Metric Noise</strong></td><td>Generative scientific QA causes noise in string metrics.</td><td>Conduct blinded 50-question human audit; correlate with Token F1.</td></tr>
</table>

<div class='box-warn'>
  <strong>Project Guardrails &mdash; Prohibited Actions:</strong> Do <strong>not</strong>: fine-tune retriever or generator; introduce additional heavy model families; tune on final-test data; alter prompts after test evaluation begins; use post-generation information in <em>X</em>(<em>q</em>); or cherry-pick failure examples.
</div>

<h2>23. Summary &amp; Definitive Benchmark Standard</h2>
<div class='box'>
  <strong>Final One-Sentence Project Statement:</strong> Instead of giving every RAG query the same amount of full-context evidence, QCCA tests whether inexpensive retrieval-side signals can determine whether a query needs 2, 4, 6, or 8 natural-language passages while compressing the remaining evidence&mdash;and whether this adaptive strategy provides a better quality-efficiency tradeoff than the best fixed allocation.
</div>

</body>
</html>
"""

output_path = "/home/gunavenkat/Downloads/CS787-RAG-Project/plan/planv2.0.pdf"
weasyprint.HTML(string=html_content).write_pdf(output_path)
print(f"SUCCESS: {output_path} generated successfully!")
