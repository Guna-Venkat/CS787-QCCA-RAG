from sane.embedding.base import EmbeddingModel
from sane.embedding.api import APIEmbedding
from sane.core.config import EmbeddingConfig


def create_embedding_model(config: EmbeddingConfig) -> EmbeddingModel:
    if config.provider == "api":
        model = APIEmbedding(
            base_url=config.api.base_url,
            api_key=config.api.api_key,
            model=config.api.model,
            dimension=config.api.dimension,
        )
    else:
        raise ValueError(f"Unknown embedding provider: {config.provider}")

    return model


__all__ = ["EmbeddingModel", "APIEmbedding", "create_embedding_model"]
