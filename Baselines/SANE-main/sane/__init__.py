from sane.core.config import Config, load_config
from sane.core.interfaces import BaseAgent
from sane.embedding.base import EmbeddingModel
from sane.agents import (
    RAGAgent,
    SANEAgent,
)


def create_agent(
    config: Config | None = None,
    extra_tools: list[dict] | None = None,
    extra_tool_handlers: dict | None = None,
    system_prompt: str | None = None,
    embedding_model: EmbeddingModel | None = None,
) -> BaseAgent:
    """Factory that instantiates the right agent based on config.method.method.

    Supported methods:
    - "rag": RAGAgent with session-level semantic retrieval
    - "sane": SANEAgent — Select-And-Extract pipeline
    """
    cfg = config or load_config()
    method = cfg.method.method

    if method == "rag":
        return RAGAgent(
            config=cfg,
            extra_tools=extra_tools,
            extra_tool_handlers=extra_tool_handlers,
            system_prompt=system_prompt,
            embedding_model=embedding_model,
        )
    elif method == "sane":
        return SANEAgent(
            config=cfg,
            extra_tools=extra_tools,
            extra_tool_handlers=extra_tool_handlers,
            system_prompt=system_prompt,
            embedding_model=embedding_model,
        )
    else:
        raise ValueError(
            f"Unknown method: {method}. "
            f"Supported: rag, sane"
        )
