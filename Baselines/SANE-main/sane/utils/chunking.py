"""Token-based text chunking for document ingestion."""


def _chunk_text_by_tokens(
    text: str,
    max_tokens: int,
    overlap_tokens: int = 0,
    encoding_name: str = "cl100k_base",
) -> list[str]:
    """Split *text* into chunks that each fit within *max_tokens*.

    Uses tiktoken for accurate token counting, which handles Chinese,
    Japanese, and other scripts where word splitting is not meaningful.
    """
    try:
        import tiktoken
        enc = tiktoken.get_encoding(encoding_name)
        tokens = enc.encode(text, disallowed_special=())
    except Exception:
        # Fallback: approximate by characters / 2 (rough for CJK)
        words = text.split()
        if not words:
            return [text] if text.strip() else []
        chunks: list[str] = []
        step = max(1, max_tokens - overlap_tokens)
        for start in range(0, len(words), step):
            chunk_text = " ".join(words[start:start + max_tokens])
            if chunk_text.strip():
                chunks.append(chunk_text)
        return chunks

    if not tokens:
        return []

    chunks: list[str] = []
    step = max(1, max_tokens - overlap_tokens)
    for start in range(0, len(tokens), step):
        chunk_tokens = tokens[start:start + max_tokens]
        chunks.append(enc.decode(chunk_tokens))
    return chunks


def _default_chunk_encoding(model_name: str) -> str:
    """Pick a reasonable tiktoken encoding for the embedding model."""
    model_lower = model_name.lower()
    if "gte" in model_lower or "modernbert" in model_lower or "longformer" in model_lower:
        return "cl100k_base"
    if "qwen" in model_lower:
        return "cl100k_base"
    return "cl100k_base"
