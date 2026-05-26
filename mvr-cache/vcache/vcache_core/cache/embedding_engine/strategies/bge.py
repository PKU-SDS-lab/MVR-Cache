from typing import List
from typing_extensions import override

from vcache.vcache_core.cache.embedding_engine.embedding_engine import EmbeddingEngine
from vcache.vcache_core.splitter.embedding_model import EmbeddingModel


class BGEEmbeddingEngine(EmbeddingEngine):
    """
    An embedding engine implementation that uses the BGE model via EmbeddingModel.
    """

    def __init__(self, embedding_model: EmbeddingModel):
        self.embedding_model = embedding_model

    @override
    def get_embedding(self, text: str) -> List[float]:
        """
        Get the embedding for the given text using the underlying BGE model.

        Args:
            text: The text to get the embedding for.

        Returns:
            The embedding of the text as a list of floats.
        """
        # EmbeddingModel.get_embedding returns a numpy array, we convert to list
        embedding = self.embedding_model.get_embedding(text)
        return embedding.tolist()

