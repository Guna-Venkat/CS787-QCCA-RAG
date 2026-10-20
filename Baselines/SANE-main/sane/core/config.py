import os
from dataclasses import dataclass, field, is_dataclass
from pathlib import Path
from typing import Any

import yaml

from sane.core.env import load_dotenv


load_dotenv()


# ──
# Config dataclasses
# ──

@dataclass
class LLMConfig:
    base_url: str = ""                          # empty → filled from model registry
    api_key: str = ""                           # empty → filled from model registry
    model: str = "deepseek-v4-flash"
    max_tokens: int = 4096
    temperature: float = 0.0
    max_retries: int = 1
    retry_base_s: float = 1.0
    retry_max_s: float = 30.0
    chat_template_kwargs: dict | None = None
    extra_body: dict | None = None              # merged into every request


@dataclass
class EmbeddingAPIConfig:
    base_url: str = ""                          # empty → filled from model registry
    api_key: str = ""                           # empty → filled from model registry
    model: str = "Qwen3-Embedding-0.6B"
    dimension: int = 1536
    max_input_tokens: int | None = None         # set by model registry at load time


@dataclass
class EmbeddingConfig:
    provider: str = "api"
    api: EmbeddingAPIConfig = field(default_factory=EmbeddingAPIConfig)


@dataclass
class SANESettings:
    """SANE pipeline settings — nested under ``method.sane`` in config YAML."""

    # ── Evidence extraction ──────────────────────────────────────
    extraction_enabled: bool = True
    extraction_model: str = ""          # empty = reuse global llm
    extraction_base_url: str = ""       # empty = reuse global llm
    extraction_api_key: str = ""        # empty = reuse global llm
    extraction_max_tokens: int = 4096
    extraction_prompt: str = ""         # empty = built-in default

    # ── Synopsis-based wide retrieval + document selection
    selection_enabled: bool = True    # LM document selection stage (synopsis-based)
    synopsis_keyword_top_n: int = 10    # top-N TF keywords per session
    synopsis_show_keywords: bool = True  # show keywords=[...] in synopsis
    synopsis_show_labels: bool = True    # show doc labels (titles) in synopsis header
    synopsis_include_excerpt: bool = True
    synopsis_full_text: bool = False     # show full candidate text (bypass synopsis)

    # ── Evidence extraction prompt template
    schema_suite: str = "blueprint"     # "" (no-BP fallback) | "blueprint" (blueprint-guided)


@dataclass
class MethodConfig:
    method: str = "rag"
    rag_chunk_size: int = 2048
    k: int = 5                      # top-k for retrieval and pipeline selection
    rag_recent_n: int = 3
    rag_user_only: bool = True
    w: int = 20                     # wide retrieval pool (w ≥ k)
    # SANE pipeline settings (nested dataclass)
    sane: SANESettings = field(default_factory=SANESettings)


@dataclass
class Config:
    llm: LLMConfig = field(default_factory=LLMConfig)
    embedding: EmbeddingConfig = field(default_factory=EmbeddingConfig)
    method: MethodConfig = field(default_factory=MethodConfig)


# ──
# YAML loading helpers
# ──

def _resolve_env_vars(value):
    if isinstance(value, str) and value.startswith("${") and value.endswith("}"):
        return os.environ.get(value[2:-1], "")
    if isinstance(value, str):
        return os.path.expandvars(value)
    if isinstance(value, dict):
        return {k: _resolve_env_vars(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_resolve_env_vars(v) for v in value]
    return value


def _build_dataclass(cls, data: dict):
    if data is None:
        return cls()
    field_names = {f.name for f in cls.__dataclass_fields__.values()}
    kwargs = {}
    for key, value in data.items():
        if key in field_names:
            field_type = cls.__dataclass_fields__[key].type
            if is_dataclass(field_type) and isinstance(value, dict):
                kwargs[key] = _build_dataclass(field_type, value)
            else:
                kwargs[key] = value
    return cls(**kwargs)


def load_config(
    config_path: str | None = None,
    env_path: str | None = None,
) -> Config:
    """Load a YAML config file and resolve model settings from the registry.

    1. Load ``.env`` (respecting *env_path* override).
    2. Read the YAML file, capturing raw sections before env-var resolution.
    3. Resolve ``${ENV_VAR}`` placeholders.
    4. Build the ``Config`` dataclass tree.
    5. Fill missing ``LLMConfig`` / ``EmbeddingAPIConfig`` fields from the
       model registry (``sane.core.model_registry``).
    """
    load_dotenv(env_path)
    project_root = Path(__file__).parent.parent.parent

    if config_path is None:
        config_path = os.environ.get("SANE_CONFIG")
    if config_path is None:
        config_path = project_root / "config" / "default.yaml"

    with open(config_path) as f:
        raw_before_env = yaml.safe_load(f) or {}

    # Capture raw sub-dicts before env-var resolution (for explicit-field detection)
    raw_llm = raw_before_env.get("llm") or {}
    raw_emb_api = (raw_before_env.get("embedding") or {}).get("api") or {}

    raw = _resolve_env_vars(dict(raw_before_env))
    config = _build_dataclass(Config, raw)

    # Resolve through model registry
    from sane.core.model_registry import resolve_llm_via_registry, resolve_embedding_via_registry
    config.llm = resolve_llm_via_registry(config.llm, raw_llm)
    config.embedding.api = resolve_embedding_via_registry(config.embedding.api, raw_emb_api)

    return config


# ──
# Judge helpers — unified LLM-as-judge resolution for all benchmarks
# ──

def resolve_judge_config(
    main_config: Config | None = None,
    *,
    model: str | None = None,
    base_url: str | None = None,
    api_key: str | None = None,
    max_tokens: int = 10,
    temperature: float = 0.0,
    judge_overrides: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Resolve judge LLM settings from multiple sources.

    Resolution order (first wins):
    1. Explicit keyword args (*model*, *base_url*, *api_key*).
    2. *judge_overrides* dict (typically from ``benchmark.yaml`` or a config
       YAML ``judge:`` section).
    3. Model registry lookup (based on the resolved *model*).
    4. Fallback environment variables: ``JUDGE_MODEL``, ``JUDGE_BASE_URL``,
       ``JUDGE_API_KEY``, ``VLLM_API_KEY``.
    5. Fallback to *main_config*.llm settings.

    Returns a dict with keys: model, base_url, api_key, max_tokens, temperature.
    """
    from sane.core.model_registry import (
        lookup_provider_api_key,
        lookup_provider_base_url,
    )

    overrides = dict(judge_overrides or {})

    # Resolve model
    resolved_model: str | None = model
    if not resolved_model:
        resolved_model = overrides.get("model")
    if not resolved_model:
        resolved_model = os.environ.get("JUDGE_MODEL")
    if not resolved_model and main_config is not None:
        resolved_model = main_config.llm.model
    if not resolved_model:
        resolved_model = "deepseek-v4-flash"

    # Resolve base_url
    resolved_base_url: str | None = base_url
    if not resolved_base_url:
        resolved_base_url = overrides.get("base_url")
    if not resolved_base_url:
        resolved_base_url = lookup_provider_base_url(resolved_model)
    if not resolved_base_url:
        resolved_base_url = os.environ.get("JUDGE_BASE_URL")
    if not resolved_base_url and main_config is not None:
        resolved_base_url = main_config.llm.base_url
    if not resolved_base_url:
        resolved_base_url = "https://api.deepseek.com"

    # Resolve api_key
    resolved_api_key: str | None = api_key
    if not resolved_api_key:
        resolved_api_key = overrides.get("api_key")
    if not resolved_api_key:
        resolved_api_key = lookup_provider_api_key(resolved_model)
    if not resolved_api_key:
        resolved_api_key = os.environ.get("JUDGE_API_KEY", "")
    if not resolved_api_key:
        resolved_api_key = os.environ.get("VLLM_API_KEY", "")
    if not resolved_api_key and main_config is not None:
        resolved_api_key = main_config.llm.api_key
    if not resolved_api_key:
        resolved_api_key = ""

    # Resolve max_tokens / temperature from overrides
    resolved_max_tokens = max_tokens
    resolved_temperature = temperature
    if "max_tokens" in overrides:
        resolved_max_tokens = int(overrides["max_tokens"])
    if "temperature" in overrides:
        resolved_temperature = float(overrides["temperature"])

    return {
        "model": resolved_model,
        "base_url": resolved_base_url,
        "api_key": resolved_api_key,
        "max_tokens": resolved_max_tokens,
        "temperature": resolved_temperature,
    }


