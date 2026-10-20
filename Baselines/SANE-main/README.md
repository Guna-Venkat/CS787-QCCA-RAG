# Select-And-Extract: A Lightweight Plugin for Retrieval-Augmented Generation

**SANE** (Select-ANd-Extract) is a lightweight, training-free, plug-and-play
plugin for retrieval-augmented generation (RAG). It addresses the two
fundamental failure modes of RAG — **retrieval failure** (missing the right
documents) and **reading failure** (failing to synthesize from retrieved
content) — with only **three LM calls per query**.

- **Synopsis-based candidate selection** replaces the standard top-*k*
  retrieval: a wide candidate pool is retrieved, compact per-document synopses
  are constructed from cheap rule-based features, and an LM selects the most
  relevant *k* documents. This improves recall over the native retriever.
- **Blueprint-guided information extraction** structures the selected documents
  into a compact evidence table using a single LM call, guided by a small set
  of canonical schema blueprints. The answering LM sees only this structured
  evidence, not the raw documents.

SANE outperforms both vanilla RAG and heavier baselines across four diverse
benchmarks while introducing only modest overhead.

## Setup

### 1. Environment

```bash
bash scripts/setup.sh
```

Creates the conda environment, installs dependencies, and creates a `.env` template.

### 2. Benchmark Data

```bash
bash scripts/download_data.sh
```

Downloads all four benchmark datasets.  HotpotQA, MuSiQue, and LongMemEval
are fetched from HuggingFace.  FinanceBench is cloned from its official
repository (PDFs + question files), then PDF page texts are pre-extracted
into a local cache.  Users behind a firewall can set the `HF_ENDPOINT`
environment variable to use a mirror for HuggingFace downloads.

### 3. API Keys

Edit `.env` and add your API keys:

```env
DS_API_KEY=your_deepseek_api_key
VLLM_API_KEY=your_vllm_api_key
```

Model-to-provider routing is handled automatically via the registry in
``sane/core/model_registry.py`` — config YAMLs only need ``model: "name"``,
no ``base_url`` or ``api_key`` plumbing.  Supported models include
``deepseek-v4-flash`` (default), ``deepseek-v4-pro``, and vLLM-hosted models
(``qwen3.5-9b``, ``Qwen/Qwen3.5-4B``, ``qwen3.6-35b-a3b``).  Override
per-run with ``--model`` and ``--judge-model``.

## Quick Start

```bash
conda activate rag

# HotpotQA
python -m benchmarks.hotpotqa.run --method sane --config config/hotpotqa.yaml --workers 50
python -m benchmarks.hotpotqa.run --method rag --config config/hotpotqa.yaml --workers 50

# MuSiQue
python -m benchmarks.musique.run --method sane --config config/musique.yaml --workers 50
python -m benchmarks.musique.run --method rag --config config/musique.yaml --workers 50

# FinanceBench (requires PDF extraction first)
python benchmarks/financebench/setup.py --workers 8
python -m benchmarks.financebench.run --method sane --config config/financebench.yaml --workers 10 --judge-workers 16

# LongMemEval
python -m benchmarks.longmemeval.run --split m_cleaned --method sane --config config/longmemeval.yaml --workers 25 --judge-workers 16
```

Switch the backbone LM per-run:

```bash
python -m benchmarks.hotpotqa.run --config config/hotpotqa.yaml --model qwen3.5-9b --judge-model deepseek-v4-flash
```

Run all benchmarks through the unified entry point:

```bash
python -m benchmarks.run --method sane --workers 10 --judge-workers 16
```

## Library Usage

```python
import sane

config = sane.load_config("config/default.yaml")
config.method.method = "sane"
config.method.k = 5
config.method.w = 20

agent = sane.create_agent(config)

agent.ingest_text(text="The capital of France is Paris.", doc_id="doc1")
agent.ingest_text(text="Paris has over 2 million residents.", doc_id="doc2")

response = agent.send("What is the capital of France and how many people live there?")
print(response)
print(agent.get_timing_summary())
```

See `demo.py` for a complete example.

## Citation
If you find our work useful, feel free to cite our paper:

```bibtex
@misc{tang2026sane,
      title={Select-And-Extract: A Lightweight Plugin for Retrieval-Augmented Generation}, 
      author={Chenming Tang and Jiawei Han},
      year={2026},
      eprint={2608.00658},
      archivePrefix={arXiv},
      primaryClass={cs.CL},
      url={https://arxiv.org/abs/2608.00658}, 
}
```
