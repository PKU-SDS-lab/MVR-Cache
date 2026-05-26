"""
Backward-compatibility shim.

Historically, vCache shipped an `OpenAIEmbeddingEngine`. The codebase now defaults to
SiliconFlow's OpenAI-compatible provider (DashScope-compatible base_url) via
`SiliconFlowEmbeddingEngine`.

`OpenAIEmbeddingEngine` remains available as an alias so existing user code keeps working.
"""

from vcache.vcache_core.cache.embedding_engine.strategies.silicon_flow import (
    SiliconFlowEmbeddingEngine as OpenAIEmbeddingEngine,
)
