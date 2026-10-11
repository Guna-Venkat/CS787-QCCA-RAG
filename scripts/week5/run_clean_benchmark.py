"""Run Clean Short Benchmark on TITAN RTX under Low Contention.

Deterministic evaluation of 5 TRAIN questions (512, 745, 1633, 1221, 56)
across all k in {2, 4, 5, 6, 8} reusing cached retrieval and SFR embeddings.
Outputs runtime_estimate_clean_gpu.json comparing against the contended baseline.
"""

import json
import logging
import statistics
import sys
import time
from pathlib import Path

import torch

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(line_buffering=True)

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

REPO_ROOT = Path(__file__).resolve().parents[2]
SARA_ROOT = REPO_ROOT / "Baselines" / "SARA-main"
if str(SARA_ROOT) not in sys.path:
    sys.path.insert(0, str(SARA_ROOT))
import src
if hasattr(src, "__path__") and str(SARA_ROOT / "src") not in src.__path__:
    src.__path__.append(str(SARA_ROOT / "src"))

from src.model.loader import load_sara_for_eval
from src.model.xMistral import extract_generated_text
from src.metrics.metrics import compute_f1
from src.const import COMPRESS

TARGET_QIDS = ["512", "745", "1633", "1221", "56"]
K_GRID = [2, 4, 5, 6, 8]


def build_evaluation_prompt(question, passages, k, tokenizer):
    assert len(passages) == 10
    if k > 0:
        text_blocks = []
        for i, p in enumerate(passages[:k]):
            p_text = p.get("text", "").strip()
            if p_text:
                text_blocks.append(f"Document {i + 1}. {p_text}")
        text_context = "\n\n".join(text_blocks) if text_blocks else "No direct text context provided."
    else:
        text_context = "No direct text context provided."

    num_compressed = 10 - k
    if num_compressed > 0:
        compress_tokens_str = "\n".join([f"{idx + 1}. {COMPRESS}" for idx in range(num_compressed)])
        additional_context_str = f"## Additional Context (compression tokens)\n{compress_tokens_str}\n---"
    else:
        additional_context_str = ""

    user_prompt = f"""Answer my questions based on the given context.
---
## Context
{text_context}
---
{additional_context_str}
## Your Task
Answer the following question in a succinct manner. Use a single phrase or a short sentence if possible.
Question: {question}
Your Answer:"""

    if hasattr(tokenizer, "apply_chat_template") and tokenizer.chat_template:
        messages = [{"role": "user", "content": user_prompt}]
        formatted_prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    else:
        formatted_prompt = f"<s>[INST] {user_prompt} [/INST]"

    context_tokens = len(tokenizer.tokenize(user_prompt))
    return formatted_prompt, context_tokens


def run_benchmark():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    logger.info(f"Using device: {device}")

    # 1. Load cached retrieval
    retrieval_file = REPO_ROOT / "results/week5/train_response_surface/train_retrieval.jsonl"
    logger.info(f"Loading retrieval records from {retrieval_file}...")
    ret_records = {}
    with open(retrieval_file, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                rec = json.loads(line)
                if str(rec["question_id"]) in TARGET_QIDS:
                    ret_records[str(rec["question_id"])] = rec

    logger.info(f"Loaded {len(ret_records)} target retrieval records.")

    # 2. Load cached SFR embeddings
    sfr_file = REPO_ROOT / "results/week5/train_response_surface/train_sfr_embeddings.pt"
    logger.info(f"Loading SFR embeddings from {sfr_file}...")
    sfr_data = torch.load(sfr_file, map_location="cpu")
    cached_embeds = sfr_data["embeddings"]

    # 3. Load SARA model
    sara_ckpt_path = REPO_ROOT / "Baselines/SARA-main/checkpoints/finetune/sara_qasper_proj_lr5e4_seed42"
    logger.info(f"Loading SARA checkpoint from {sara_ckpt_path}...")
    sara_model, sara_tokenizer = load_sara_for_eval(str(sara_ckpt_path), device=device)
    pad_id = sara_tokenizer.pad_token_id if sara_tokenizer.pad_token_id is not None else sara_tokenizer.eos_token_id

    # Warmup
    logger.info("Running warmup...")
    warmup_inputs = sara_tokenizer("<s>[INST] Hello [/INST]", return_tensors="pt").to(device)
    with torch.inference_mode():
        _ = sara_model.generate(**warmup_inputs, max_new_tokens=4, do_sample=False, pad_token_id=pad_id)

    logger.info("Beginning Clean GPU Benchmark across 5 questions x 5 budgets...")
    bench_by_k = {k: [] for k in K_GRID}

    for qid in TARGET_QIDS:
        rec = ret_records[qid]
        question = rec["question"]
        passages = rec["passages"]
        gold_answers = rec.get("gold_answers", [""])

        for k in K_GRID:
            prompt, context_tokens = build_evaluation_prompt(question, passages, k, sara_tokenizer)
            inputs = sara_tokenizer(prompt, return_tensors="pt").to(device)
            input_tokens = int(inputs["input_ids"].shape[1])

            comp_embeds = cached_embeds[qid][k:10].to(device=device, dtype=sara_model.dtype)

            t0 = time.time()
            with torch.inference_mode():
                outputs = sara_model.generate(
                    **inputs,
                    retrieval_embeds=comp_embeds,
                    max_new_tokens=64,
                    do_sample=False,
                    pad_token_id=pad_id,
                    eos_token_id=sara_tokenizer.eos_token_id,
                )
            gen_latency = time.time() - t0
            total_latency = gen_latency + ((10 - k) * 0.005)

            pred_texts = extract_generated_text(sara_tokenizer, outputs)
            pred_text = pred_texts[0].strip() if pred_texts else ""
            f1 = max([compute_f1(pred_text, g)[2] for g in gold_answers]) if gold_answers else 0.0

            bench_by_k[k].append({
                "question_id": qid,
                "gen_latency": gen_latency,
                "total_latency": total_latency,
                "prompt_tokens": input_tokens,
                "context_tokens": context_tokens,
                "gen_tokens": int(outputs.shape[1]),
                "f1": f1,
            })
            logger.info(f"QID {qid} k={k}: latency={gen_latency:.3f}s, prompt_tokens={input_tokens}, f1={f1:.4f}")

    # Compile benchmark summary
    clean_k_stats = {}
    sum_clean_mean = 0.0
    sum_clean_median = 0.0

    contended_reference = {
        2: {"mean_sec": 4.29},
        4: {"mean_sec": 6.03},
        5: {"mean_sec": 7.22},
        6: {"mean_sec": 8.26},
        8: {"mean_sec": 9.45},
    }

    for k in K_GRID:
        records = bench_by_k[k]
        latencies = [r["total_latency"] for r in records]
        mean_lat = statistics.mean(latencies)
        med_lat = statistics.median(latencies)
        mean_prompt_tok = statistics.mean([r["prompt_tokens"] for r in records])
        mean_gen_tok = statistics.mean([r["gen_tokens"] for r in records])

        sum_clean_mean += mean_lat
        sum_clean_median += med_lat

        clean_k_stats[f"k{k}"] = {
            "mean_total_latency_seconds": round(mean_lat, 4),
            "median_total_latency_seconds": round(med_lat, 4),
            "mean_prompt_tokens": round(mean_prompt_tok, 1),
            "mean_generation_tokens": round(mean_gen_tok, 1),
            "sample_size": len(records),
            "contended_baseline_mean_seconds": contended_reference[k]["mean_sec"],
            "speedup_factor": round(contended_reference[k]["mean_sec"] / mean_lat, 2) if mean_lat > 0 else 1.0,
        }

    n_train_questions = 2088
    total_evals = n_train_questions * len(K_GRID)
    clean_extrap_hours_mean = round((n_train_questions * sum_clean_mean) / 3600.0, 2)
    clean_extrap_hours_median = round((n_train_questions * sum_clean_median) / 3600.0, 2)
    contended_hours = 20.45

    benchmark_result = {
        "benchmark_metadata": {
            "device": "NVIDIA TITAN RTX (24GB)",
            "gpu_contention": "Low contention (PID 1147949 ~594 MiB, ~55% util)",
            "benchmark_sample_size_questions": len(TARGET_QIDS),
            "question_ids": TARGET_QIDS,
            "budgets_evaluated": K_GRID,
            "total_evaluations_run": len(TARGET_QIDS) * len(K_GRID),
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        },
        "per_budget_statistics": clean_k_stats,
        "full_sweep_extrapolation": {
            "total_train_questions": n_train_questions,
            "total_evaluations": total_evals,
            "sum_k_mean_latency_seconds": round(sum_clean_mean, 3),
            "sum_k_median_latency_seconds": round(sum_clean_median, 3),
            "clean_estimated_hours_mean": clean_extrap_hours_mean,
            "clean_estimated_hours_median": clean_extrap_hours_median,
            "contended_estimated_hours": contended_hours,
            "effective_speedup": round(contended_hours / clean_extrap_hours_mean, 2) if clean_extrap_hours_mean > 0 else 1.0,
        },
    }

    out_file = REPO_ROOT / "results/week5/train_surface_validation/runtime_estimate_clean_gpu.json"
    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(benchmark_result, f, indent=2)

    logger.info(f"Clean benchmark complete! Saved to {out_file}")
    logger.info(json.dumps(benchmark_result, indent=2))
    return benchmark_result


if __name__ == "__main__":
    run_benchmark()
