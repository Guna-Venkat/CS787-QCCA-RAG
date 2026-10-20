from __future__ import annotations

from abc import ABC, abstractmethod


class BaseAgent(ABC):
    """Unified interface for all agent methods.

    All implementations accept (config, extra_tools, extra_tool_handlers, system_prompt)
    in their constructors, though the ABC does not mandate a specific __init__ signature.
    """

    @abstractmethod
    def send(self, message: str) -> str:
        """Send a message to the agent and return the response."""
        ...

    @abstractmethod
    def reset(self):
        """Reset agent state for a new conversation."""
        ...


class EmbeddingModel(ABC):
    @abstractmethod
    def embed(self, texts: list[str]) -> list[list[float]]:
        ...

    @abstractmethod
    def dimension(self) -> int:
        ...


class LLMClient(ABC):
    @abstractmethod
    def chat(
        self,
        messages: list[dict],
        tools: list[dict] | None = None,
        tool_choice: str = "auto",
        **kwargs,
    ) -> dict:
        ...
