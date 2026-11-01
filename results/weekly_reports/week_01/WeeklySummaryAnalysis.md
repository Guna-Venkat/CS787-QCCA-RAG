# Week 1 — Summary Analysis

## Objective

Reconstruct the baseline implementation, local split, actual SARA/RAG training and checkpoint qualification under Baselines/. User-provided scope resolves the previous missing Week1 mapping; code availability is separated from trained checkpoints and answer-quality evaluation.

## Experiments performed

| Part   | Question                                                                                          | Status       | Main_result                                                                                                                                                                                                                                                                                                       |
|:-------|:--------------------------------------------------------------------------------------------------|:-------------|:------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| A      | Which local examples underpin the reproduced SARA/RAG baselines?                                  | COMPLETED    | The manifest and raw files agree:2088 TRAIN queries/772 papers,231 DEV queries/86 papers, and zero TRAIN/DEV paper overlap. TEST has1309 rows according to metadata only; its questions/outcomes were not loaded for this reconstruction.                                                                         |
| B      | Was a projector-free RAG adapter actually trained rather than merely configured?                  | COMPLETED    | RAG has250 logged optimizer steps and a saved adapter. Executed notebook cell10 reports506 training examples, runtime14988.353 seconds and train_loss5.325839; the final logged step loss2.9502 is a different quantity.                                                                                          |
| C      | Was the text-plus-latent baseline trained, and how does its execution provenance differ from RAG? | COMPLETED    | The SARA checkpoint trainer state records250 optimizer steps and epoch0.501002, with last logged loss5.4034. Saved args use one epoch, unlike current YAML3. Notebook cell11 preserves an OOM-failed training attempt; that specific failure and the separately saved completed checkpoint must both be retained. |
| D      | Can both saved checkpoints load with their intended projector/tokenizer interfaces?               | SMOKE-TESTED | Executed cell13 records RAG/SARA loading in10.23/9.67 seconds, peak GPU13.51/13.58 GiB, compress token32001 and vocabulary32002. RAG is projector-free; SARA prints nonzero projector norm164.                                                                                                                    |
| E      | What do the saved loss trajectories establish about baseline training?                            | COMPLETED    | Both trainer histories contain250 finite logged losses. Their means/final losses are recomputed in the table; differences cannot be interpreted as comparative QA accuracy.                                                                                                                                       |
| F      | Is there a standalone completed Week1 answer-quality comparison in Baselines?                     | IMPLEMENTED  | Evaluation code exists, but no standalone completed local Week1 QA score table or raw evaluation result was located inside Baselines. Week2 outcomes are not copied backward into Week1.                                                                                                                          |
| G      | Which additional evidence-selection/extraction baseline is implemented?                           | IMPLEMENTED  | SANE code/configuration and benchmark interfaces are present. No completed local Week1 SANE quality run was found; current default model/config must not be assumed to be its historical Week1 frozen identity.                                                                                                   |
| H      | Does the RECOMP checkout establish a locally executed baseline comparison?                        | IMPLEMENTED  | RECOMP implementation is present with a3610-row bundled NQ completion CSV. Its local generation provenance is not established, so it is not counted as a completed Week1 experiment or a QASPER result.                                                                                                           |

## Key results

### Manifest counts independently checked against TRAIN/DEV JSONL

| split                                |   queries | papers         | source                                                        |
|:-------------------------------------|----------:|:---------------|:--------------------------------------------------------------|
| TRAIN                                |      2088 | 772            | Baselines/SARA-main/data/reformatted/QASPER_train_split.jsonl |
| DEV                                  |       231 | 86             | Baselines/SARA-main/data/reformatted/QASPER_dev.jsonl         |
| TEST metadata only; no outcomes read |      1309 | Not recomputed | Baselines/SARA-main/data/manifests/qasper_split.json          |

Source: [Baselines/SARA-main/data/manifests/qasper_split.json](../../../Baselines/SARA-main/data/manifests/qasper_split.json).

### Saved RAG training record

| method   |   global_step |   logged_steps |   recorded_epoch |   saved_num_train_epochs |   saved_num_evidence |   saved_num_additional_evidence |   model_learning_rate |   projector_learning_rate |   batch_size |   gradient_accumulation |   max_train_samples_argument |   mean_logged_training_loss |   last_logged_training_loss |   max_logged_grad_norm | best_metric   |   adapter_weight_bytes |
|:---------|--------------:|---------------:|-----------------:|-------------------------:|---------------------:|--------------------------------:|----------------------:|--------------------------:|-------------:|------------------------:|-----------------------------:|----------------------------:|----------------------------:|-----------------------:|:--------------|-----------------------:|
| RAG      |           250 |            250 |         1.972332 |                        2 |                   10 |                               0 |              0.000020 |                  0.000000 |            1 |                       4 |                         2000 |                    5.325842 |                    2.950200 |            1975.659912 |               |               27280152 |

Source: [Baselines/SARA-main/checkpoints/finetune/rag_qasper_lora_r16_seed42/checkpoint-250/trainer_state.json](../../../Baselines/SARA-main/checkpoints/finetune/rag_qasper_lora_r16_seed42/checkpoint-250/trainer_state.json).

### Executed RAG notebook training log

|   notebook_cell |   num_examples_logged |   reported_training_runtime_seconds |   reported_training_loss |
|----------------:|----------------------:|------------------------------------:|-------------------------:|
|       10.000000 |            506.000000 |                        14988.353000 |                 5.325839 |

Source: [Baselines/SARA-main/notebooks/sara_baseline_checkpoints.ipynb](../../../Baselines/SARA-main/notebooks/sara_baseline_checkpoints.ipynb).

### Saved SARA training record

| method   |   global_step |   logged_steps |   recorded_epoch |   saved_num_train_epochs |   saved_num_evidence |   saved_num_additional_evidence |   model_learning_rate |   projector_learning_rate |   batch_size |   gradient_accumulation |   max_train_samples_argument |   mean_logged_training_loss |   last_logged_training_loss |   max_logged_grad_norm | best_metric   |   adapter_weight_bytes |
|:---------|--------------:|---------------:|-----------------:|-------------------------:|---------------------:|--------------------------------:|----------------------:|--------------------------:|-------------:|------------------------:|-----------------------------:|----------------------------:|----------------------------:|-----------------------:|:--------------|-----------------------:|
| SARA     |           250 |            250 |         0.501002 |                        1 |                    5 |                               5 |              0.000020 |                  0.000500 |            1 |                       4 |                         2000 |                    6.936258 |                    5.403400 |            3262.517822 |               |               27280152 |

Source: [Baselines/SARA-main/checkpoints/finetune/sara_qasper_proj_lr5e4_seed42/checkpoint-250/trainer_state.json](../../../Baselines/SARA-main/checkpoints/finetune/sara_qasper_proj_lr5e4_seed42/checkpoint-250/trainer_state.json).

### Current configuration versus saved-run epoch identity

| method   |   current_yaml_epochs |   saved_run_epochs | source                                                                         |
|:---------|----------------------:|-------------------:|:-------------------------------------------------------------------------------|
| RAG      |                     3 |                  2 | Baselines/SARA-main/config/language_modeling/Finetune_RAG_qasper_pubbase.yaml  |
| SARA     |                     3 |                  1 | Baselines/SARA-main/config/language_modeling/Finetune_SARA_qasper_pubbase.yaml |

Source: [Baselines/SARA-main/config/language_modeling/Finetune_SARA_qasper_pubbase.yaml](../../../Baselines/SARA-main/config/language_modeling/Finetune_SARA_qasper_pubbase.yaml).

### Executed notebook cell13 checkpoint profiles

| checkpoint                    |   load_seconds |   peak_GPU_GiB |   compress_token_id |   tokenizer_vocab |
|:------------------------------|---------------:|---------------:|--------------------:|------------------:|
| rag_qasper_lora_r16_seed42    |      10.230000 |      13.510000 |               32001 |             32002 |
| sara_qasper_proj_lr5e4_seed42 |       9.670000 |      13.580000 |               32001 |             32002 |

Source: [Baselines/SARA-main/notebooks/sara_baseline_checkpoints.ipynb](../../../Baselines/SARA-main/notebooks/sara_baseline_checkpoints.ipynb).

### Offline recomputed log summaries

| method   |   logged_steps |   recorded_epoch |   mean_logged_training_loss |   last_logged_training_loss |   max_logged_grad_norm |
|:---------|---------------:|-----------------:|----------------------------:|----------------------------:|-----------------------:|
| RAG      |            250 |         1.972332 |                    5.325842 |                    2.950200 |            1975.659912 |
| SARA     |            250 |         0.501002 |                    6.936258 |                    5.403400 |            3262.517822 |

Source: [results/weekly_reports/week_01/TrainingRecomputation.csv](TrainingRecomputation.csv).

### Bundled completion-file inventory; not a locally verified run

|   rows |   question_rows |   prediction_rows | columns                                                        |
|-------:|----------------:|------------------:|:---------------------------------------------------------------|
|   3610 |            3610 |              3610 | Unnamed: 0,few_shot_examples,question,pred_answer,gold_answers |

Source: [Baselines/recomp-main/data/completions/flan_ul2_nq_5shot_no_passage.csv](../../../Baselines/recomp-main/data/completions/flan_ul2_nq_5shot_no_passage.csv).

## Main scientific conclusions

- The manifest and raw files agree:2088 TRAIN queries/772 papers,231 DEV queries/86 papers, and zero TRAIN/DEV paper overlap. TEST has1309 rows according to metadata only; its questions/outcomes were not loaded for this reconstruction. Source: [Baselines/SARA-main/data/manifests/qasper_split.json](../../../Baselines/SARA-main/data/manifests/qasper_split.json), [Baselines/SARA-main/data/reformatted/QASPER_train_split.jsonl](../../../Baselines/SARA-main/data/reformatted/QASPER_train_split.jsonl)
- RAG has250 logged optimizer steps and a saved adapter. Executed notebook cell10 reports506 training examples, runtime14988.353 seconds and train_loss5.325839; the final logged step loss2.9502 is a different quantity. Source: [Baselines/SARA-main/checkpoints/finetune/rag_qasper_lora_r16_seed42/args.json](../../../Baselines/SARA-main/checkpoints/finetune/rag_qasper_lora_r16_seed42/args.json), [Baselines/SARA-main/checkpoints/finetune/rag_qasper_lora_r16_seed42/checkpoint-250/trainer_state.json](../../../Baselines/SARA-main/checkpoints/finetune/rag_qasper_lora_r16_seed42/checkpoint-250/trainer_state.json)
- The SARA checkpoint trainer state records250 optimizer steps and epoch0.501002, with last logged loss5.4034. Saved args use one epoch, unlike current YAML3. Notebook cell11 preserves an OOM-failed training attempt; that specific failure and the separately saved completed checkpoint must both be retained. Source: [Baselines/SARA-main/checkpoints/finetune/sara_qasper_proj_lr5e4_seed42/args.json](../../../Baselines/SARA-main/checkpoints/finetune/sara_qasper_proj_lr5e4_seed42/args.json), [Baselines/SARA-main/checkpoints/finetune/sara_qasper_proj_lr5e4_seed42/checkpoint-250/trainer_state.json](../../../Baselines/SARA-main/checkpoints/finetune/sara_qasper_proj_lr5e4_seed42/checkpoint-250/trainer_state.json)
- Executed cell13 records RAG/SARA loading in10.23/9.67 seconds, peak GPU13.51/13.58 GiB, compress token32001 and vocabulary32002. RAG is projector-free; SARA prints nonzero projector norm164. Source: [Baselines/SARA-main/notebooks/sara_baseline_checkpoints.ipynb](../../../Baselines/SARA-main/notebooks/sara_baseline_checkpoints.ipynb), [Baselines/SARA-main/checkpoints/finetune/rag_qasper_lora_r16_seed42/adapter_config.json](../../../Baselines/SARA-main/checkpoints/finetune/rag_qasper_lora_r16_seed42/adapter_config.json)
- Both trainer histories contain250 finite logged losses. Their means/final losses are recomputed in the table; differences cannot be interpreted as comparative QA accuracy. Source: [Baselines/SARA-main/checkpoints/finetune/rag_qasper_lora_r16_seed42/checkpoint-250/trainer_state.json](../../../Baselines/SARA-main/checkpoints/finetune/rag_qasper_lora_r16_seed42/checkpoint-250/trainer_state.json), [Baselines/SARA-main/checkpoints/finetune/sara_qasper_proj_lr5e4_seed42/checkpoint-250/trainer_state.json](../../../Baselines/SARA-main/checkpoints/finetune/sara_qasper_proj_lr5e4_seed42/checkpoint-250/trainer_state.json)
- Evaluation code exists, but no standalone completed local Week1 QA score table or raw evaluation result was located inside Baselines. Week2 outcomes are not copied backward into Week1. Source: [Baselines/SARA-main/src/eval/eval_sara.py](../../../Baselines/SARA-main/src/eval/eval_sara.py), [Baselines/SARA-main/src/eval/calculate_qa_performance.py](../../../Baselines/SARA-main/src/eval/calculate_qa_performance.py)
- SANE code/configuration and benchmark interfaces are present. No completed local Week1 SANE quality run was found; current default model/config must not be assumed to be its historical Week1 frozen identity. Source: [Baselines/SANE-main/config/default.yaml](../../../Baselines/SANE-main/config/default.yaml), [Baselines/SANE-main/benchmarks/metrics.py](../../../Baselines/SANE-main/benchmarks/metrics.py)
- RECOMP implementation is present with a3610-row bundled NQ completion CSV. Its local generation provenance is not established, so it is not counted as a completed Week1 experiment or a QASPER result. Source: [Baselines/recomp-main/run_extractive_compressor.py](../../../Baselines/recomp-main/run_extractive_compressor.py), [Baselines/recomp-main/train_extractive_compressor.py](../../../Baselines/recomp-main/train_extractive_compressor.py)

## What did NOT work?

SARA notebook training attempt OOMs; no completed standalone Week1 QA quality comparison or local SANE/RECOMP benchmark is established.

## Limitations

Baseline training sample/epoch identities are not fully matched; quality is not established by training losses, load timing, projector norm or bundled external completions.

## Decision after this week

Saved public-base RAG and SARA checkpoints exist for later controlled evaluation. SANE/RECOMP are implemented references with no verified local Week1 performance.

## One-paragraph week takeaway

The manifest and raw files agree:2088 TRAIN queries/772 papers,231 DEV queries/86 papers, and zero TRAIN/DEV paper overlap. TEST has1309 rows according to metadata only; its questions/outcomes were not loaded for this reconstruction. RAG has250 logged optimizer steps and a saved adapter. Executed notebook cell10 reports506 training examples, runtime14988.353 seconds and train_loss5.325839; the final logged step loss2.9502 is a different quantity. The SARA checkpoint trainer state records250 optimizer steps and epoch0.501002, with last logged loss5.4034. Saved args use one epoch, unlike current YAML3. Notebook cell11 preserves an OOM-failed training attempt; that specific failure and the separately saved completed checkpoint must both be retained. Executed cell13 records RAG/SARA loading in10.23/9.67 seconds, peak GPU13.51/13.58 GiB, compress token32001 and vocabulary32002. RAG is projector-free; SARA prints nonzero projector norm164. Both trainer histories contain250 finite logged losses. Their means/final losses are recomputed in the table; differences cannot be interpreted as comparative QA accuracy. Evaluation code exists, but no standalone completed local Week1 QA score table or raw evaluation result was located inside Baselines. Week2 outcomes are not copied backward into Week1. SANE code/configuration and benchmark interfaces are present. No completed local Week1 SANE quality run was found; current default model/config must not be assumed to be its historical Week1 frozen identity. RECOMP implementation is present with a3610-row bundled NQ completion CSV. Its local generation provenance is not established, so it is not counted as a completed Week1 experiment or a QASPER result.

# Primary Sources Used

- [Baselines/SANE-main/benchmarks/metrics.py](../../../Baselines/SANE-main/benchmarks/metrics.py)
- [Baselines/SANE-main/config/default.yaml](../../../Baselines/SANE-main/config/default.yaml)
- [Baselines/SANE-main/sane/agents/sane_agent.py](../../../Baselines/SANE-main/sane/agents/sane_agent.py)
- [Baselines/SARA-main/checkpoints/finetune/rag_qasper_lora_r16_seed42/adapter_config.json](../../../Baselines/SARA-main/checkpoints/finetune/rag_qasper_lora_r16_seed42/adapter_config.json)
- [Baselines/SARA-main/checkpoints/finetune/rag_qasper_lora_r16_seed42/args.json](../../../Baselines/SARA-main/checkpoints/finetune/rag_qasper_lora_r16_seed42/args.json)
- [Baselines/SARA-main/checkpoints/finetune/rag_qasper_lora_r16_seed42/checkpoint-250/trainer_state.json](../../../Baselines/SARA-main/checkpoints/finetune/rag_qasper_lora_r16_seed42/checkpoint-250/trainer_state.json)
- [Baselines/SARA-main/checkpoints/finetune/sara_qasper_proj_lr5e4_seed42/adapter_config.json](../../../Baselines/SARA-main/checkpoints/finetune/sara_qasper_proj_lr5e4_seed42/adapter_config.json)
- [Baselines/SARA-main/checkpoints/finetune/sara_qasper_proj_lr5e4_seed42/args.json](../../../Baselines/SARA-main/checkpoints/finetune/sara_qasper_proj_lr5e4_seed42/args.json)
- [Baselines/SARA-main/checkpoints/finetune/sara_qasper_proj_lr5e4_seed42/checkpoint-250/trainer_state.json](../../../Baselines/SARA-main/checkpoints/finetune/sara_qasper_proj_lr5e4_seed42/checkpoint-250/trainer_state.json)
- [Baselines/SARA-main/checkpoints/finetune/sara_qasper_proj_lr5e4_seed42/sara_config.json](../../../Baselines/SARA-main/checkpoints/finetune/sara_qasper_proj_lr5e4_seed42/sara_config.json)
- [Baselines/SARA-main/config/language_modeling/Finetune_RAG_qasper_pubbase.yaml](../../../Baselines/SARA-main/config/language_modeling/Finetune_RAG_qasper_pubbase.yaml)
- [Baselines/SARA-main/config/language_modeling/Finetune_SARA_qasper_pubbase.yaml](../../../Baselines/SARA-main/config/language_modeling/Finetune_SARA_qasper_pubbase.yaml)
- [Baselines/SARA-main/data/manifests/qasper_split.json](../../../Baselines/SARA-main/data/manifests/qasper_split.json)
- [Baselines/SARA-main/data/reformatted/QASPER_dev.jsonl](../../../Baselines/SARA-main/data/reformatted/QASPER_dev.jsonl)
- [Baselines/SARA-main/data/reformatted/QASPER_train_split.jsonl](../../../Baselines/SARA-main/data/reformatted/QASPER_train_split.jsonl)
- [Baselines/SARA-main/notebooks/sara_baseline_checkpoints.ipynb](../../../Baselines/SARA-main/notebooks/sara_baseline_checkpoints.ipynb)
- [Baselines/SARA-main/scripts/build_qasper_dataset.py](../../../Baselines/SARA-main/scripts/build_qasper_dataset.py)
- [Baselines/SARA-main/src/data/make_qasper_splits.py](../../../Baselines/SARA-main/src/data/make_qasper_splits.py)
- [Baselines/SARA-main/src/eval/calculate_qa_performance.py](../../../Baselines/SARA-main/src/eval/calculate_qa_performance.py)
- [Baselines/SARA-main/src/eval/eval_sara.py](../../../Baselines/SARA-main/src/eval/eval_sara.py)
- [Baselines/SARA-main/src/run_sara.py](../../../Baselines/SARA-main/src/run_sara.py)
- [Baselines/SARA-main/src/train/train_generator.py](../../../Baselines/SARA-main/src/train/train_generator.py)
- [Baselines/recomp-main/data/completions/flan_ul2_nq_5shot_no_passage.csv](../../../Baselines/recomp-main/data/completions/flan_ul2_nq_5shot_no_passage.csv)
- [Baselines/recomp-main/eval_qa.py](../../../Baselines/recomp-main/eval_qa.py)
- [Baselines/recomp-main/run_extractive_compressor.py](../../../Baselines/recomp-main/run_extractive_compressor.py)
- [Baselines/recomp-main/train_extractive_compressor.py](../../../Baselines/recomp-main/train_extractive_compressor.py)
- [results/weekly_reports/week_01/CheckpointProfilingExtract.csv](CheckpointProfilingExtract.csv)
- [results/weekly_reports/week_01/NotebookExecutionChecks.json](NotebookExecutionChecks.json)
- [results/weekly_reports/week_01/SplitRecomputation.json](SplitRecomputation.json)
- [results/weekly_reports/week_01/TrainingLogExtract.csv](TrainingLogExtract.csv)
- [results/weekly_reports/week_01/TrainingRecomputation.csv](TrainingRecomputation.csv)
- [results/weekly_reports/week_01/figures/logged_training_loss.png](figures/logged_training_loss.png)

# Existing Markdown Used as Evidence

NONE
