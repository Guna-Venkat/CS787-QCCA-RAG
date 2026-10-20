import json

from sane.llm.openai_client import OpenAIClient


MAX_ITERATIONS = 10


def run_tool_loop(
    llm_client: OpenAIClient,
    messages: list[dict],
    tools: list[dict] | None = None,
    tool_handlers: dict | None = None,
    max_iterations: int = MAX_ITERATIONS,
) -> str:
    """Run the LLM tool-calling inner loop.

    Appends tool_call and tool_result messages directly to the messages list
    (no context rebuild between tool calls). Returns the final response text.

    Args:
        llm_client: The LLM client to call.
        messages: The full message list (modified in-place with tool interactions).
        tools: OpenAI-format tool schemas. If None or empty, a single non-tool
               chat call is made.
        tool_handlers: dict mapping tool name -> handler(args) -> str.
        max_iterations: Maximum tool-calling iterations before giving up.

    Returns:
        The final assistant response text (may be empty on error).
    """
    handlers = tool_handlers or {}
    response_text = ""

    for _ in range(max_iterations):
        try:
            response = llm_client.chat(
                messages=messages,
                tools=tools if tools else None,
            )
        except Exception as e:
            return f"Error: {e}"

        tool_calls = response.get("tool_calls", [])

        if response.get("finish_reason") == "stop" or not tool_calls:
            response_text = response.get("content") or ""
            break

        # Append assistant message with tool_calls
        assistant_msg = {
            "role": "assistant",
            "content": response.get("content"),
            "tool_calls": tool_calls,
        }
        messages.append(assistant_msg)

        # Execute each tool call and append results
        for tc in tool_calls:
            func_name = tc["function"]["name"]
            try:
                func_args = json.loads(tc["function"]["arguments"])
            except (json.JSONDecodeError, KeyError):
                func_args = {}

            handler = handlers.get(func_name)
            if handler is not None:
                try:
                    result = handler(func_args)
                except Exception as e:
                    result = f"Tool error ({func_name}): {e}"
            else:
                result = f"Unknown tool: {func_name}"

            messages.append({
                "role": "tool",
                "tool_call_id": tc.get("id", ""),
                "content": str(result),
            })

    return response_text
