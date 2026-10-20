"""Model registry — single source of truth for model → provider mapping.

To add a new model:
    Add one entry to MODELS.  If its provider is new, also add one entry to
    PROVIDERS.  That's it — no other files need changes.
"""

from __future__ import annotations

import os
from dataclasses import dataclass


# ──
# Dataclasses
# ──

@dataclass
class ProviderConfig:
    """An API provider that serves one or more models."""
    name: str                           # display name, e.g. "DeepSeek"
    base_url: str                       # "https://api.deepseek.com"
    api_key_env: str                    # env-var name, e.g. "DS_API_KEY"


@dataclass
class ModelEntry:
    """Registration of one model.

    Only *model* and *provider* are required.  Everything else has sensible
    defaults.
    """
    model: str                          # API model name, e.g. "deepseek-v4-flash"
    provider: str                       # key into PROVIDERS
    extra_body: dict | None = None      # merged into extra_body of every request
    embedding_dimension: int | None = None
    max_input_tokens: int | None = None
    max_completion_tokens: int | None = None  # clamps config max_tokens at override time
    use_max_completion_tokens: bool = False   # use "max_completion_tokens" param


# ──
# Registry data — edit HERE to add models / providers
# ──

PROVIDERS: dict[str, ProviderConfig] = {
    "deepseek": ProviderConfig(
        name="DeepSeek",
        base_url="https://api.deepseek.com",
        api_key_env="DS_API_KEY",
    ),
    "vllm": ProviderConfig(
        name="vLLM",
        base_url="https://your-vllm-host/v1",
        api_key_env="VLLM_API_KEY",
    ),
    "embedding": ProviderConfig(
        name="Embedding",
        base_url="https://your-embedding-host/v1",
        api_key_env="EMBED_API_KEY",
    ),
}


MODELS: dict[str, ModelEntry] = {
    # ── DeepSeek ──────────────────────────────────────────────────────
    "deepseek-v4-flash": ModelEntry(
        model="deepseek-v4-flash",
        provider="deepseek",
        extra_body={"thinking": {"type": "disabled"}},
    ),
    "deepseek-v4-pro": ModelEntry(
        model="deepseek-v4-pro",
        provider="deepseek",
        extra_body={"thinking": {"type": "disabled"}},
    ),

    # ── vLLM ─────────────────────────────────────────────────────────
    "Qwen/Qwen3.5-4B": ModelEntry(
        model="Qwen/Qwen3.5-4B",
        provider="vllm",
        extra_body={"enable_thinking": False},
    ),
    "qwen3.5-9b": ModelEntry(
        model="qwen3.5-9b",
        provider="vllm",
    ),
    "qwen3.6-35b-a3b": ModelEntry(
        model="qwen3.6-35b-a3b",
        provider="vllm",
    ),

    # ── Embedding ─────────────────────────────────────────────────────
    "Qwen3-Embedding-0.6B": ModelEntry(
        model="Qwen3-Embedding-0.6B",
        provider="embedding",
        embedding_dimension=1024,
        max_input_tokens=32768,
    ),
}


# ──
# Lookup helpers
# ──

def _get_entry(model: str) -> ModelEntry | None:
    """Look up a model in the registry.  Returns None for unknown models."""
    return MODELS.get(model)


def _get_provider(provider_key: str) -> ProviderConfig | None:
    return PROVIDERS.get(provider_key)


def lookup_extra_body(model: str) -> dict | None:
    """Return extra_body defaults for *model*, or None."""
    entry = _get_entry(model)
    if entry is None:
        return None
    return entry.extra_body


def lookup_provider_base_url(model: str) -> str | None:
    """Return the base_url for *model*'s provider, or None.

    The provider's default base_url can be overridden by setting the
    ``{PROVIDER_KEY}_BASE_URL`` environment variable (uppercase).
    For example, ``EMBEDDING_BASE_URL`` overrides the embedding provider.
    """
    entry = _get_entry(model)
    if entry is None:
        return None
    provider = _get_provider(entry.provider)
    if provider is None:
        return None
    # Env-var override: e.g. EMBEDDING_BASE_URL, VLLM_BASE_URL
    env_override = os.environ.get(f"{entry.provider.upper()}_BASE_URL")
    if env_override:
        return env_override
    return provider.base_url


def lookup_provider_api_key(model: str) -> str:
    """Return the API key for *model*'s provider.

    Looks up the provider's ``api_key_env`` in the process environment.
    For embedding models, also checks ``EMBED_API_KEY`` as a fallback.
    Returns ``""`` when the model is unknown or the env var is not set.
    """
    entry = _get_entry(model)
    if entry is None:
        return ""
    provider = _get_provider(entry.provider)
    if provider is None:
        return ""

    # Embedding models: try EMBED_API_KEY first, then the provider's key
    if entry.embedding_dimension is not None:
        embed_key = os.environ.get("EMBED_API_KEY", "")
        if embed_key:
            return embed_key

    return os.environ.get(provider.api_key_env, "")


def get_embedding_token_limit(model: str) -> int | None:
    """Return *max_input_tokens* for a registered embedding model, or None."""
    entry = _get_entry(model)
    if entry is None:
        return None
    return entry.max_input_tokens


# ──
# Config resolution — called from load_config()
# ──

def _yaml_explicit(raw: dict | None, key: str, resolved_value) -> bool:
    """Return True if *key* was explicitly set to a non-empty value in YAML."""
    if raw is None:
        return False
    if key not in raw:
        return False
    raw_val = raw.get(key)
    if raw_val is None:
        return False
    if isinstance(raw_val, str) and raw_val.strip() == "":
        return False
    if resolved_value is None:
        return False
    if isinstance(resolved_value, str) and not resolved_value.strip():
        return False
    return True


def resolve_llm_via_registry(
    llm: object,
    raw_llm: dict | None = None,
) -> object:
    """Fill missing LLMConfig fields from the model registry.

    *llm* is an LLMConfig dataclass instance (already built from YAML).
    *raw_llm* is the ``llm:`` section of the YAML dict *before* env-var
    resolution — used to detect which keys were explicitly set.

    Resolution rules (per field):
    1. Explicit, non-empty YAML value wins (key present in *raw_llm* with
       a non-empty value after env-var resolution).
    2. Model registry fills in.
    3. Fall back to the existing value on *llm*.
    """
    raw_llm = raw_llm or {}
    model = getattr(llm, "model", "")
    entry = _get_entry(model)

    # base_url — fill from registry if not explicit in YAML
    current_base = getattr(llm, "base_url", "")
    if not _yaml_explicit(raw_llm, "base_url", current_base):
        if not current_base and entry is not None:
            resolved_url = lookup_provider_base_url(model)
            if resolved_url:
                object.__setattr__(llm, "base_url", resolved_url)

    # api_key — fill from registry if not explicit in YAML
    current_key = getattr(llm, "api_key", "")
    if not _yaml_explicit(raw_llm, "api_key", current_key):
        if entry is not None:
            key = lookup_provider_api_key(model)
            if key:
                object.__setattr__(llm, "api_key", key)

    # extra_body — merge: registry base + YAML overrides
    registry_extra = lookup_extra_body(model)
    if "extra_body" in raw_llm:
        yaml_extra = raw_llm.get("extra_body") or {}
        if registry_extra:
            merged = dict(registry_extra)
            merged.update(yaml_extra)
            object.__setattr__(llm, "extra_body", merged)
        else:
            object.__setattr__(llm, "extra_body", yaml_extra if yaml_extra else None)
    elif registry_extra:
        yaml_extra = getattr(llm, "extra_body", None)
        if yaml_extra:
            merged = dict(registry_extra)
            merged.update(yaml_extra)
            object.__setattr__(llm, "extra_body", merged)
        else:
            object.__setattr__(llm, "extra_body", registry_extra)

    return llm


def resolve_embedding_via_registry(
    emb: object,
    raw_emb: dict | None = None,
) -> object:
    """Fill missing EmbeddingAPIConfig fields from the model registry.

    Same resolution rules as ``resolve_llm_via_registry``.
    """
    raw_emb = raw_emb or {}
    model = getattr(emb, "model", "")
    entry = _get_entry(model)

    # base_url — fill from registry if not explicit in YAML
    current_base = getattr(emb, "base_url", "")
    if not _yaml_explicit(raw_emb, "base_url", current_base):
        if not current_base and entry is not None:
            resolved_url = lookup_provider_base_url(model)
            if resolved_url:
                object.__setattr__(emb, "base_url", resolved_url)

    # api_key — fill from registry if not explicit in YAML
    current_key = getattr(emb, "api_key", "")
    if not _yaml_explicit(raw_emb, "api_key", current_key):
        if entry is not None:
            key = lookup_provider_api_key(model)
            if key:
                object.__setattr__(emb, "api_key", key)

    # dimension — fill from registry if not explicit in YAML
    current_dim = getattr(emb, "dimension", 0)
    if not _yaml_explicit(raw_emb, "dimension", current_dim):
        if entry is not None and entry.embedding_dimension is not None:
            object.__setattr__(emb, "dimension", entry.embedding_dimension)

    # max_input_tokens — from registry
    if entry is not None and entry.max_input_tokens is not None:
        object.__setattr__(emb, "max_input_tokens", entry.max_input_tokens)

    return emb


# ──
# Runtime model override — for --model / --judge-model CLI flags
# ──

def apply_model_override(llm: object, model: str) -> None:
    """Change the model on an already-resolved LLMConfig and re-resolve.

    Used when ``--model`` is passed at the CLI to override the config file.
    Caps ``max_tokens`` to the model's ``max_completion_tokens`` if set.
    """
    object.__setattr__(llm, "model", model)
    entry = _get_entry(model)
    if entry is None:
        return

    provider = _get_provider(entry.provider)
    if provider is not None:
        resolved_url = lookup_provider_base_url(model) or provider.base_url
        object.__setattr__(llm, "base_url", resolved_url)

    key = lookup_provider_api_key(model)
    if key:
        object.__setattr__(llm, "api_key", key)

    extra_body = lookup_extra_body(model)
    object.__setattr__(llm, "extra_body", extra_body)

    # Clamp max_tokens to the model's limit
    if entry.max_completion_tokens is not None:
        current_max = getattr(llm, "max_tokens", 0)
        if current_max > entry.max_completion_tokens:
            object.__setattr__(llm, "max_tokens", entry.max_completion_tokens)
