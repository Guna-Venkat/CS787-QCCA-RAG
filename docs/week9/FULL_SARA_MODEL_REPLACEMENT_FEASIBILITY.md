# Full SARA generator replacement — future work only

SAME HIDDEN DIMENSION != COMPATIBLE LATENT SPACE.

The existing trained projector maps SFR vectors into the adapted xMistral hidden space. Llama/Qwen/DeepSeek/Gemma cannot consume those vectors as interchangeable embeddings, including the 4096-dimensional candidates. `assert_projector_compatible` and textual runtime checks reject that path. The canonical first pass remains the saved trained Mistral/SARA checkpoint; no projector or LoRA training occurred in this patch.

| Alternative | Text hidden / token embedding dimension | Tokenizer / special tokens | Projector / adaptation work | Training memory and effort |
|---|---:|---|---|---|
| Llama3.1-8B | 4096 / 4096 | Llama tokenizer, chat/header/end-of-turn tokens; add and train a compression token deliberately | New Llama-specific SFR projector; validate token insertion, train on FIT only, adapt intended attention/LoRA modules | Unmeasured; optimizer/activation memory greatly exceeds inference. Quantized LoRA plus a small projector may be feasible at bounded sequence length, but no <20 GB training claim. Substantial loader/prompt/training/parity work. |
| Qwen2.5-7B | 3584 / 3584 | Qwen tokenizer/chat delimiters; preserve EOS and vocab conventions | New 4096→3584 projector with FIT-only training; new generator-specific LoRA and compression-token validation | Unmeasured; explicit training profile required. Substantial effort, with new loss, allocation and parity checks. |
| DeepSeek distill 7B | 3584 / 3584 | Qwen-derived tokenizer with reasoning template and different stopping behavior | New projector and FIT-only adaptation, plus full reasoning-token accounting; current Turing numerical diagnostics must be resolved | Unmeasured; reasoning sequences add activation and runtime cost. High implementation/validation effort. |
| Qwen3-8B | 4096 / 4096 | Qwen3 chat template and explicit thinking mode | New Qwen3-specific projector despite matching dimension; FIT-only LoRA/token training | Unmeasured; training-only profiling required. High effort including thinking/non-thinking parity. |
| Gemma3-12B | 3840 / 3840 text tower | Gemma turn/end-of-turn tokens; conditional-generation wrapper, text-only or explicitly multimodal | New text-tower-compatible projector and token contract; FIT-only adaptation; no reuse of xMistral's embedding geometry | Native 12B inference weights already exceed the representative ceiling. Quantized training feasibility is unmeasured. Highest effort among this panel. |

Token embedding dimensions are architecture/configuration properties; usable latent geometry must be learned rather than inferred from shape. Any later replacement requires a separate protocol, train/held-out paper isolation, actual optimizer memory profiling, initial identity/parity tests, and new source/model/cache fingerprints. No alternative is labelled a superior replacement: the FIT screen has not been run. A successful textual screen may nominate a future adaptation candidate; it cannot validate latent injection.

Official configuration/model-card references are linked in [W9PRE_MODEL_RESEARCH.md](W9PRE_MODEL_RESEARCH.md).
