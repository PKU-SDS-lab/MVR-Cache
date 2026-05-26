import os
from typing import List, Optional

from openai import OpenAI as OpenAIClient

from vcache.vcache_core.cache.embedding_engine.embedding_engine import EmbeddingEngine


class SiliconFlowEmbeddingEngine(EmbeddingEngine):
    """
    SiliconFlow (OpenAI-compatible) embedding engine using the OpenAI Python SDK.

    By default, this targets Alibaba Cloud DashScope's OpenAI-compatible endpoint.
    """

    def __init__(
        self,
        model_name: str = "text-embedding-v2",
        api_key: Optional[str] = None,
        base_url: str = "https://dashscope.aliyuncs.com/compatible-mode/v1",
    ):
        self.model_name = model_name
        self.base_url = base_url

        self.api_key = (
            api_key
            or os.environ.get("SILICONFLOW_API_KEY")
            or os.environ.get("DASHSCOPE_API_KEY")
            or os.environ.get("OPENAI_API_KEY")
        )
        self._client: Optional[OpenAIClient] = None

    @property
    def client(self) -> OpenAIClient:
        if self._client is None:
            if not self.api_key:
                raise ValueError(
                    "No API key configured for SiliconFlowEmbeddingEngine. Set one of "
                    "SILICONFLOW_API_KEY / DASHSCOPE_API_KEY (recommended), or OPENAI_API_KEY, "
                    "or pass api_key=... when constructing the engine."
                )
            self._client = OpenAIClient(api_key=self.api_key, base_url=self.base_url)
        return self._client

    def get_embedding(self, text: str) -> List[float]:
        try:
            response = self.client.embeddings.create(input=text, model=self.model_name)
            return response.data[0].embedding
        except Exception as e:
            raise Exception(f"Error getting embedding from SiliconFlow provider: {e}")


