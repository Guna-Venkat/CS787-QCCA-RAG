import os
import threading
import time

from sane.embedding.base import EmbeddingModel


# Maximum texts per API call.  OpenAI allows 2048.
_API_BATCH_SIZE = int(os.environ.get("EMBED_API_BATCH_SIZE", 100))

# Concurrency: max simultaneous API calls across all threads.
_API_MAX_CONCURRENT = int(os.environ.get("EMBED_API_CONCURRENT", "50"))
_semaphore: threading.BoundedSemaphore | None = None


def _get_semaphore() -> threading.BoundedSemaphore:
    global _semaphore
    if _semaphore is None:
        _semaphore = threading.BoundedSemaphore(_API_MAX_CONCURRENT)
    return _semaphore


class APIEmbedding(EmbeddingModel):
    def __init__(self, base_url: str, api_key: str, model: str, dimension: int):
        from openai import OpenAI
        import httpx
        import ssl

        # Work around macOS Python SSL → Cloudflare TLS incompatibility.
        # The system curl (SecureTransport) works but Python's OpenSSL-based
        # httpx gets UNEXPECTED_EOF_WHILE_READING.  Forcing TLS 1.2 fixes it.
        _ssl_context = ssl.create_default_context()
        _ssl_context.maximum_version = ssl.TLSVersion.TLSv1_2
        _transport = httpx.HTTPTransport(verify=_ssl_context)
        _http_client = httpx.Client(transport=_transport, timeout=60.0)

        self._client = OpenAI(
            base_url=base_url, api_key=api_key,
            http_client=_http_client,
        )
        self._model = model
        self._dim = dimension

    def embed(self, texts: list[str]) -> list[list[float]]:
        if not texts:
            return []

        all_vectors: list[list[float]] = []
        for start in range(0, len(texts), _API_BATCH_SIZE):
            batch = texts[start:start + _API_BATCH_SIZE]

            with _get_semaphore():
                response = self._client.embeddings.create(
                    model=self._model, input=batch,
                )

            all_vectors.extend(d.embedding for d in response.data)

        return all_vectors

    def dimension(self) -> int:
        return self._dim

    def max_input_tokens(self) -> int | None:
        from sane.core.model_registry import get_embedding_token_limit
        limit = get_embedding_token_limit(self._model)
        return limit if limit is not None else 8191

    def count_tokens(self, text: str) -> int | None:
        try:
            import tiktoken
            enc = tiktoken.encoding_for_model(self._model)
            return len(enc.encode(text))
        except Exception:
            try:
                import tiktoken
                enc = tiktoken.get_encoding("cl100k_base")
                return len(enc.encode(text))
            except Exception:
                return None
