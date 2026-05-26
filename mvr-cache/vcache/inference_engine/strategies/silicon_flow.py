import os
import time
from typing import Optional

from openai import APIError, OpenAI as OpenAIClient

from vcache.inference_engine.inference_engine import InferenceEngine


class SiliconFlowInferenceEngine(InferenceEngine):
    """
    SiliconFlow (OpenAI-compatible) inference engine using the OpenAI Python SDK.

    By default, this targets Alibaba Cloud DashScope's OpenAI-compatible endpoint.
    """

    def __init__(
        self,
        model_name: str = "deepseek-r1-distill-qwen-14b",
        temperature: float = 0.3,
        api_key: Optional[str] = None,
        base_url: str = "https://dashscope.aliyuncs.com/compatible-mode/v1",
        max_attempts: int = 4,
        backoff_factor_seconds: float = 2.0,
    ):
        super().__init__()
        self.model_name = model_name
        self.temperature = temperature
        self.base_url = base_url
        self.max_attempts = max_attempts
        self.backoff_factor_seconds = backoff_factor_seconds

        # Prefer explicit api_key, then provider-specific env vars, then OpenAI's default.
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
                    "No API key configured for SiliconFlowInferenceEngine. Set one of "
                    "SILICONFLOW_API_KEY / DASHSCOPE_API_KEY (recommended), or OPENAI_API_KEY, "
                    "or pass api_key=... when constructing the engine."
                )
            # OpenAI SDK supports OpenAI-compatible providers via base_url.
            self._client = OpenAIClient(api_key=self.api_key, base_url=self.base_url)
        return self._client

    def create(self, prompt: str, system_prompt: Optional[str] = None) -> str:
        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})

        last_err: Optional[BaseException] = None
        for attempt in range(self.max_attempts):
            try:
                completion = self.client.chat.completions.create(
                    model=self.model_name,
                    messages=messages,
                    temperature=self.temperature,
                )
                content = completion.choices[0].message.content
                return content or ""
            except APIError as e:
                last_err = e
                msg = str(e).lower()
                retriable = any(
                    s in msg
                    for s in (
                        "serviceunavailable",
                        "throttled",
                        "too many requests",
                        "internalerror",
                        "timeout",
                        "temporarily unavailable",
                    )
                )
                if not retriable or attempt == self.max_attempts - 1:
                    break
                wait_time = self.backoff_factor_seconds * (2**attempt)
                time.sleep(wait_time)
            except Exception as e:
                last_err = e
                break

        raise RuntimeError(f"Error creating completion from SiliconFlow provider: {last_err}")


