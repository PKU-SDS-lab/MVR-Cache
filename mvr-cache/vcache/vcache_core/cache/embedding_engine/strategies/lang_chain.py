import os
from typing import List

# IMPORTANT: This MUST be set before importing huggingface_hub/transformers/sentence-transformers.
# Otherwise they default to https://huggingface.co and will ignore later env changes.
os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")
if "hf-mirror.com" not in os.environ.get("HF_ENDPOINT", ""):
    os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"

from langchain_community.embeddings import HuggingFaceEmbeddings

from vcache.vcache_core.cache.embedding_engine.embedding_engine import EmbeddingEngine


class LangChainEmbeddingEngine(EmbeddingEngine):
    """
    LangChain implementation of embedding engine using HuggingFace models.
    """

    def __init__(self, model_name: str = "sentence-transformers/all-mpnet-base-v2", *, hf_endpoint: str | None = None):
        """
        Initialize a LangChain embedding engine.

        Args:
            model_name: Name of the HuggingFace model to use for embeddings.
            hf_endpoint: Optional override for Hugging Face hub endpoint (e.g. "https://hf-mirror.com").
        """
        self.model_name = model_name
        if hf_endpoint:
            os.environ["HF_ENDPOINT"] = hf_endpoint
        self.embeddings = HuggingFaceEmbeddings(model_name=model_name)

    def get_embedding(self, text: str) -> List[float]:
        """
        Get embedding for the provided text using LangChain/HuggingFace.

        Args:
            text: The text to embed.

        Returns:
            The embedding vector.

        Raises:
            Exception: If there's an error getting the embedding.
        """
        try:
            embedding = self.embeddings.embed_query(text)
            return embedding
        except Exception as e:
            raise Exception(f"Error getting embedding from LangChain: {e}")