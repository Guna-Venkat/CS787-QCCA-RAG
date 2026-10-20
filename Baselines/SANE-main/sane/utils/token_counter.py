def count_tokens(text: str, model: str = "deepseek-v4-flash") -> int:
    try:
        import tiktoken
        try:
            enc = tiktoken.encoding_for_model(model)
        except KeyError:
            enc = tiktoken.get_encoding("cl100k_base")
        return len(enc.encode(text))
    except Exception:
        return len(text) // 4


def estimate_message_tokens(messages: list[dict]) -> int:
    total = 0
    for msg in messages:
        content = msg.get("content", "") or ""
        total += count_tokens(content)
        if msg.get("tool_calls"):
            for tc in msg["tool_calls"]:
                total += count_tokens(tc.get("function", {}).get("arguments", ""))
    return total
