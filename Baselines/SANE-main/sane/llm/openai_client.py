import logging
import random
import re
import threading
import time
from typing import Optional

import openai
from openai import OpenAI

from sane.llm.base import LLMClient
from sane.utils.token_counter import estimate_message_tokens

logger = logging.getLogger(__name__)

_RETRYABLE_ERRORS = (
    openai.APIConnectionError,
    openai.APITimeoutError,
    openai.RateLimitError,
    openai.InternalServerError,
)


def _is_retryable(exc: Exception) -> bool:
    """Return True if the exception indicates a transient error worth retrying."""
    return isinstance(exc, _RETRYABLE_ERRORS)


# ── Context-length overflow detection ──────────────────────────────────────
# vllm returns HTTP 400 ("BadRequestError") when the request exceeds its
# max_model_len.  We detect these by matching the error message, parse the
# numeric limit, and truncate the input to retry.

_CONTEXT_LENGTH_PATTERNS = [
    r"maximum context length",
    r"context length",
    r"token count exceeds",
    r"too many tokens",
    r"reduce the length",
    r"input is too long",
    r"requested token count exceeds",
    r"maximum model length",
    r"max_model_len",
    r"exceeds the model's maximum",
    r"exceeds model maximum",
    # OpenAI-style messages
    r"reduce your prompt",
    r"context_length_exceeded",
    r"context window",
    # Generic token overflow
    r"prompt is too long",
    r"tokens? limit",
]

_CONTEXT_LENGTH_PARSE_PATTERNS = [
    r"maximum context length (?:is |of )?(\d+)",
    r"max_model_len (?:is |of )?(\d+)",
    r"maximum (?:model )?context length(?: of)? (\d+)",
    r"context length (?:of )?(\d+)",
    r"allowed (?:is )?(\d+)",
    r"maximum (?:is )?(\d+) tokens",
    r"limit (?:of )?(\d+)",
]


def _is_context_length_error(exc: openai.BadRequestError) -> bool:
    """Return True if the BadRequestError indicates context-length overflow."""
    msg = str(exc)
    return any(
        re.search(pat, msg, re.IGNORECASE) for pat in _CONTEXT_LENGTH_PATTERNS
    )


def _parse_max_context_from_error(exc: openai.BadRequestError) -> int | None:
    """Try to extract the maximum context length from the error message."""
    msg = str(exc)
    for pat in _CONTEXT_LENGTH_PARSE_PATTERNS:
        match = re.search(pat, msg, re.IGNORECASE)
        if match:
            return int(match.group(1))
    return None


# ── Message truncation helpers ─────────────────────────────────────────────

_TRUNCATION_SAFETY_MARGIN = 0.90  # fraction of max context reserved for prompt
_MIN_COMPLETION_BUDGET = 256       # minimum tokens reserved for model output


def _segment_messages(messages: list[dict]) -> list[list[dict]]:
    """Partition messages into atomic blocks for safe truncation.

    Blocks are designed so that tool-call/tool-result groups are never split
    across a truncation boundary:

    - ``system``       → solo block
    - ``user``         → solo block
    - ``assistant`` without tool_calls → solo block
    - ``assistant`` with tool_calls + all immediately-following ``tool``
      messages → multi-message block
    - orphan ``tool``  → solo block (edge case)
    """
    blocks: list[list[dict]] = []
    i = 0
    while i < len(messages):
        msg = messages[i]
        role = msg.get("role", "")
        if role == "assistant" and msg.get("tool_calls"):
            # Group assistant tool_calls with trailing tool results
            block = [msg]
            i += 1
            while i < len(messages) and messages[i].get("role") == "tool":
                block.append(messages[i])
                i += 1
            blocks.append(block)
        else:
            blocks.append([msg])
            i += 1
    return blocks


def _block_tokens(block: list[dict]) -> int:
    """Estimate total tokens for a block of messages."""
    return sum(estimate_message_tokens([m]) for m in block)


def _truncate_messages(
    messages: list[dict],
    max_model_len: int,
    max_completion_tokens: int,
) -> tuple[list[dict], int]:
    """Truncate messages to fit within the model's context window.

    Drops the oldest non-system messages first while keeping the most recent
    context intact.  Tool-call groups are treated atomically.

    Returns ``(truncated_messages, dropped_block_count)``.
    """
    if not messages:
        return messages, 0

    # Reserve space for the completion
    completion_budget = max(max_completion_tokens, _MIN_COMPLETION_BUDGET)
    target_tokens = int((max_model_len - completion_budget) * _TRUNCATION_SAFETY_MARGIN)
    if target_tokens <= 0:
        target_tokens = int(max_model_len * 0.5)

    blocks = _segment_messages(messages)

    # ── System message always preserved ──
    system_block = None
    body_blocks = blocks
    if blocks and blocks[0][0].get("role") == "system":
        system_block = blocks[0]
        body_blocks = blocks[1:]

    system_tokens = _block_tokens(system_block) if system_block else 0
    if system_tokens > target_tokens:
        # System prompt alone exceeds the budget — can't truncate usefully
        return messages, 0

    available = target_tokens - system_tokens

    # ── Keep blocks from the tail (most recent first) ──
    kept_body: list[list[dict]] = []
    for block in reversed(body_blocks):
        bt = _block_tokens(block)
        if bt <= available:
            kept_body.append(block)
            available -= bt
        else:
            # This block and all older ones are dropped
            break

    # Reassemble: system + kept body blocks (restored to original order)
    kept_body.reverse()
    kept_blocks = ([system_block] if system_block else []) + kept_body
    truncated = [m for block in kept_blocks for m in block]  # flatten
    dropped = len(body_blocks) - len(kept_body)

    return truncated, dropped


class OpenAIClient(LLMClient):
    def __init__(
        self,
        base_url: str = "https://api.deepseek.com",
        api_key: str = "",
        model: str = "deepseek-v4-flash",
        max_tokens: int = 4096,
        temperature: float = 0.0,
        max_retries: int = 1,
        retry_base_s: float = 1.0,
        retry_max_s: float = 30.0,
        chat_template_kwargs: dict | None = None,
        request_timeout: float = 300.0,
        extra_body: dict | None = None,
    ):
        self.client = OpenAI(
            base_url=base_url,
            api_key=api_key,
            timeout=request_timeout,
            max_retries=0,  # we handle retries ourselves in _call_with_retry
        )
        self.model = model
        self.max_tokens = max_tokens
        self.temperature = temperature
        self.max_retries = max_retries
        self.retry_base_s = retry_base_s
        self.retry_max_s = retry_max_s
        self._chat_template_kwargs = chat_template_kwargs
        self._extra_body = extra_body      # dict | None — lazy lookup if None
        self._metrics_lock = threading.Lock()
        self.reset_metrics()

    def chat(
        self,
        messages: list[dict],
        tools: Optional[list[dict]] = None,
        tool_choice: str = "auto",
        **kwargs,
    ) -> dict:
        max_tokens_val = kwargs.get("max_tokens", self.max_tokens)

        # Some models require max_completion_tokens instead of max_tokens
        from sane.core.model_registry import _get_entry
        entry = _get_entry(self.model)
        use_max_comp = entry is not None and entry.use_max_completion_tokens

        params = {
            "model": self.model,
            "messages": messages,
            "max_completion_tokens" if use_max_comp else "max_tokens": max_tokens_val,
            "temperature": kwargs.get("temperature", self.temperature),
        }

        if tools:
            params["tools"] = tools
            params["tool_choice"] = tool_choice

        # Build extra_body: merge registry/model defaults with overrides
        _default_extra: dict = {}
        extra_body = self._extra_body
        if extra_body is None:
            from sane.core.model_registry import lookup_extra_body
            extra_body = lookup_extra_body(self.model) or {}
        if extra_body:
            _default_extra.update(extra_body)
        if self._chat_template_kwargs:
            _default_extra["chat_template_kwargs"] = self._chat_template_kwargs
        if "extra_body" in kwargs:
            _default_extra.update(kwargs["extra_body"])
        if _default_extra:
            params["extra_body"] = _default_extra

        if "n" in kwargs:
            params["n"] = kwargs["n"]
        if "response_format" in kwargs:
            params["response_format"] = kwargs["response_format"]
        if "stop" in kwargs:
            params["stop"] = kwargs["stop"]

        estimated_prompt_tokens = estimate_message_tokens(messages)
        message_count = len(messages)
        t0 = time.perf_counter()

        response, truncated_dropped = self._call_with_retry(
            params, estimated_prompt_tokens, message_count, t0
        )

        latency_s = time.perf_counter() - t0
        choice = response.choices[0]

        result = {
            "finish_reason": choice.finish_reason,
            "content": choice.message.content,
            "tool_calls": [],
            "reasoning_content": getattr(choice.message, "reasoning_content", None),
        }

        if choice.message.tool_calls:
            result["tool_calls"] = [
                {
                    "id": tc.id,
                    "type": "function",
                    "function": {
                        "name": tc.function.name,
                        "arguments": tc.function.arguments,
                    },
                }
                for tc in choice.message.tool_calls
            ]

        usage = {}
        if hasattr(response, "usage") and response.usage:
            cached_prompt_tokens = _cached_prompt_tokens(response.usage)
            usage = {
                "prompt_tokens": response.usage.prompt_tokens,
                "completion_tokens": response.usage.completion_tokens,
                "total_tokens": response.usage.total_tokens,
                "cached_prompt_tokens": cached_prompt_tokens,
            }
            result["usage"] = {
                "prompt_tokens": usage["prompt_tokens"],
                "completion_tokens": usage["completion_tokens"],
                "total_tokens": usage["total_tokens"],
                "cached_prompt_tokens": usage["cached_prompt_tokens"],
            }

        self._record_metrics(
            latency_s=latency_s,
            estimated_prompt_tokens=estimated_prompt_tokens,
            message_count=message_count,
            usage=usage,
            truncated_dropped=truncated_dropped,
        )

        return result

    def reset_metrics(self):
        with getattr(self, "_metrics_lock", threading.Lock()):
            self._call_count = 0
            self._error_count = 0
            self._latency_s = 0.0
            self._usage = {
                "prompt_tokens": 0,
                "completion_tokens": 0,
                "total_tokens": 0,
                "cached_prompt_tokens": 0,
            }
            self._truncation_count = 0
            self._truncation_dropped_total = 0
            self._context = {
                "estimated_prompt_tokens": 0,
                "max_estimated_prompt_tokens": 0,
                "last_estimated_prompt_tokens": 0,
                "messages_total": 0,
                "max_messages": 0,
                "last_messages": 0,
            }

    def get_usage_summary(self) -> dict:
        with self._metrics_lock:
            calls = self._call_count
            context = dict(self._context)
            context["avg_estimated_prompt_tokens"] = (
                context["estimated_prompt_tokens"] / calls if calls else 0.0
            )
            context["avg_messages"] = (
                context["messages_total"] / calls if calls else 0.0
            )
            return {
                "model": self.model,
                "call_count": calls,
                "error_count": self._error_count,
                "latency_s": self._latency_s,
                "avg_latency_s": self._latency_s / calls if calls else 0.0,
                "usage": dict(self._usage),
                "prompt_cache_hit_rate": (
                    self._usage["cached_prompt_tokens"] / self._usage["prompt_tokens"]
                    if self._usage["prompt_tokens"] else 0.0
                ),
                "context": context,
                "truncation_count": self._truncation_count,
                "truncation_dropped_total": self._truncation_dropped_total,
            }

    def _call_with_retry(
        self,
        params: dict,
        estimated_prompt_tokens: int,
        message_count: int,
        t0: float,
    ):
        """Call the API with exponential backoff retry for transient errors.

        On context-length overflow (vllm returns HTTP 400 / BadRequestError),
        truncates the input messages from the front and retries once without
        consuming the retry budget.
        """
        last_exc = None
        _already_truncated = False
        _truncated_dropped = 0
        for attempt in range(self.max_retries + 1):
            try:
                return self.client.chat.completions.create(**params), _truncated_dropped
            except openai.OpenAIError as exc:
                # ── Context-length overflow fallback ──
                if (
                    not _already_truncated
                    and isinstance(exc, openai.BadRequestError)
                    and _is_context_length_error(exc)
                ):
                    max_ctx = _parse_max_context_from_error(exc)
                    # Fall back to model registry if error message doesn't
                    # contain a parseable limit
                    if max_ctx is None:
                        from sane.core.model_registry import _get_entry
                        entry = _get_entry(self.model)
                        if entry is not None and entry.max_input_tokens:
                            max_ctx = entry.max_input_tokens
                    if max_ctx is not None and max_ctx > 0:
                        max_comp = params.get("max_tokens") or params.get(
                            "max_completion_tokens", 1024
                        )
                        truncated, dropped = _truncate_messages(
                            params["messages"], max_ctx, max_comp,
                        )
                        if dropped > 0:
                            logger.warning(
                                "Context-length overflow detected (max=%d tokens). "
                                "Truncated %d message block(s): %d → %d messages, "
                                "estimated %d → %d tokens (model=%s). Retrying.",
                                max_ctx, dropped,
                                message_count, len(truncated),
                                estimated_prompt_tokens,
                                estimate_message_tokens(truncated),
                                self.model,
                            )
                            params["messages"] = truncated
                            _already_truncated = True
                            _truncated_dropped = dropped
                            estimated_prompt_tokens = estimate_message_tokens(truncated)
                            message_count = len(truncated)
                            continue  # retry WITHOUT consuming attempt budget
                        else:
                            logger.warning(
                                "Context-length overflow but truncation could not "
                                "reduce message size (system prompt may exceed limit). "
                                "Raising original error (model=%s).",
                                self.model,
                            )
                    else:
                        logger.warning(
                            "Context-length overflow but could not determine max "
                            "context length from error. Raising original error "
                            "(model=%s, error=%s).",
                            self.model, exc,
                        )
                # ── End context-length fallback ──

                if not _is_retryable(exc) or attempt >= self.max_retries:
                    if attempt >= self.max_retries and _is_retryable(exc):
                        logger.warning(
                            "LLM call failed after %d retries (max_retries=%d): %s",
                            attempt, self.max_retries, exc,
                        )
                    self._record_metrics(
                        latency_s=time.perf_counter() - t0,
                        estimated_prompt_tokens=estimated_prompt_tokens,
                        message_count=message_count,
                        error=True,
                        truncated_dropped=_truncated_dropped,
                    )
                    raise
                last_exc = exc
                wait = min(
                    self.retry_base_s * (2 ** attempt) + random.uniform(0, 1),
                    self.retry_max_s,
                )
                logger.warning(
                    "LLM call failed (attempt %d/%d, retrying in %.1fs): %s",
                    attempt + 1, self.max_retries, wait, exc,
                )
                time.sleep(wait)
            except Exception:
                self._record_metrics(
                    latency_s=time.perf_counter() - t0,
                    estimated_prompt_tokens=estimated_prompt_tokens,
                    message_count=message_count,
                    error=True,
                    truncated_dropped=_truncated_dropped,
                )
                raise
        # Should be unreachable; keep type checkers happy
        if last_exc:
            raise last_exc
        raise RuntimeError("Unexpected: retry loop exhausted without exception")

    def _record_metrics(
        self,
        *,
        latency_s: float,
        estimated_prompt_tokens: int,
        message_count: int,
        usage: dict | None = None,
        error: bool = False,
        truncated_dropped: int = 0,
    ):
        usage = usage or {}
        with self._metrics_lock:
            self._call_count += 1
            if error:
                self._error_count += 1
            self._latency_s += latency_s

            if truncated_dropped:
                self._truncation_count += 1
                self._truncation_dropped_total += truncated_dropped

            prompt_tokens = _int_or_zero(usage.get("prompt_tokens"))
            completion_tokens = _int_or_zero(usage.get("completion_tokens"))
            cached_tokens = _int_or_zero(usage.get("cached_prompt_tokens"))

            for key in (
                "prompt_tokens",
                "completion_tokens",
                "total_tokens",
                "cached_prompt_tokens",
            ):
                self._usage[key] += _int_or_zero(usage.get(key))

            self._context["estimated_prompt_tokens"] += estimated_prompt_tokens
            self._context["last_estimated_prompt_tokens"] = estimated_prompt_tokens
            self._context["max_estimated_prompt_tokens"] = max(
                self._context["max_estimated_prompt_tokens"],
                estimated_prompt_tokens,
            )
            self._context["messages_total"] += message_count
            self._context["last_messages"] = message_count
            self._context["max_messages"] = max(
                self._context["max_messages"],
                message_count,
            )


def _int_or_zero(value) -> int:
    try:
        return int(value or 0)
    except (TypeError, ValueError):
        return 0


def _cached_prompt_tokens(usage) -> int:
    details = getattr(usage, "prompt_tokens_details", None)
    if details is None and isinstance(usage, dict):
        details = usage.get("prompt_tokens_details")
    if details is None:
        return 0
    if isinstance(details, dict):
        return _int_or_zero(details.get("cached_tokens"))
    return _int_or_zero(getattr(details, "cached_tokens", 0))
