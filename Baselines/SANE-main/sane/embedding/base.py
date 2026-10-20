from sane.core.interfaces import EmbeddingModel as BaseEmbeddingModel


class EmbeddingModel(BaseEmbeddingModel):
    """Concrete base implementation."""

    def max_input_tokens(self) -> int | None:
        """Maximum number of tokens the model can accept per text.

        Returns None if unknown (caller should use a conservative default).
        """
        return None

    def count_tokens(self, text: str) -> int | None:
        """Count tokens in *text* using the model's own tokenizer.

        Returns None if token counting is not available.
        """
        return None
