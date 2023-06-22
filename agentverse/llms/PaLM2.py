import logging
import os

import google.generativeai as palm
from pydantic import Field

from agentverse.llms.base import LLMResult
from . import llm_registry
from .base import BaseCompletionModel, BaseModelArgs

api_key = os.environ.get("PALM_API_KEY")
if api_key is None:
    logging.error("PaLM2 api key is not configured! Plz check config")
    raise Exception("ERROR PaLM2 CONFIG!")
palm.configure(api_key=api_key)

class PaLMCompletionArgs(BaseModelArgs):
    model: str = Field(default="text-bison-001")
    stop_sequences: str = Field(default='\n')
    max_output_tokens: int = Field(default=2048)

@llm_registry.register("text-bison-001")
class PaLMCompletion(BaseCompletionModel):
    args: PaLMCompletionArgs = Field(default_factory=PaLMCompletionArgs)

    def __init__(self, max_retry: int = 3, **kwargs):
        args = PaLMChatArgs()
        args = args.dict()
        for k, v in args.items():
            args[k] = kwargs.pop(k, v)
        if len(kwargs) > 0:
            logging.warning(f"Unused arguments: {kwargs}")
        super().__init__(args=args, max_retry=max_retry)

    def generate_response(self, prompt: str) -> LLMResult:
        response = palm.generate_text(prompt=prompt, **self.args.dict())
        return LLMResult(
            content=response["choices"][0]["text"],
            send_tokens=response["usage"]["prompt_tokens"],
            recv_tokens=response["usage"]["completion_tokens"],
            total_tokens=response["usage"]["total_tokens"],
        )

    async def generate_response_async(self, prompt: str) -> LLMResult:
        response = await palm.generate_text(prompt=prompt, **self.args.dict())
        return LLMResult(
            content=response["choices"][0]["text"],
            send_tokens=response["usage"]["prompt_tokens"],
            recv_tokens=response["usage"]["completion_tokens"],
            total_tokens=response["usage"]["total_tokens"],
        )

class PaLMChatArgs(BaseModelArgs):
    model: str = Field(default="text-bison-001")
    max_output_tokens: int = Field(default=2048)
    temperature: float = Field(default=0.5)
    top_p: int = Field(default=1)
    top_k: int = Field(default=3)
    stop_sequences: str = Field(default='\n')
    presence_penalty: int = Field(default=0)
    frequency_penalty: int = Field(default=0)


@llm_registry.register("text-bison-001")
class PaLMChat(BaseCompletionModel):
    args: PaLMChatArgs = Field(default_factory=PaLMChatArgs)

    def __init__(self, max_retry: int = 3, **kwargs):
        args = PaLMChatArgs()
        args = args.dict()
        for k, v in args.items():
            args[k] = kwargs.pop(k, v)
        if len(kwargs) > 0:
            logging.warning(f"Unused arguments: {kwargs}")
        super().__init__(args=args, max_retry=max_retry)

    def generate_response(self, prompt: str) -> LLMResult:
        response = palm.generate_text(prompt=prompt, **self.args.dict())
        return LLMResult(
            content=response["choices"][0]["text"],
            send_tokens=response["usage"]["prompt_tokens"],
            recv_tokens=response["usage"]["completion_tokens"],
            total_tokens=response["usage"]["total_tokens"],
        )

    async def generate_response_async(self, prompt: str) -> LLMResult:
        response = await palm.generate_text(prompt=prompt, **self.args.dict())
        return LLMResult(
            content=response["choices"][0]["text"],
            send_tokens=response["usage"]["prompt_tokens"],
            recv_tokens=response["usage"]["completion_tokens"],
            total_tokens=response["usage"]["total_tokens"],
        )
