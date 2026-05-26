"""
Backward-compatibility shim.

Historically, vCache shipped an `OpenAIInferenceEngine`. The codebase now defaults to
SiliconFlow's OpenAI-compatible provider (DashScope-compatible base_url) via
`SiliconFlowInferenceEngine`.

`OpenAIInferenceEngine` remains available as an alias so existing user code keeps working.
"""

from vcache.inference_engine.strategies.silicon_flow import (
    SiliconFlowInferenceEngine as OpenAIInferenceEngine,
)
