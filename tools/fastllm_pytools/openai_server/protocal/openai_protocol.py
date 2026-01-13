# Adapted from
# https://github.com/lm-sys/FastChat/blob/v0.2.36/fastchat/protocol/openai_api_protocol.py
from __future__ import annotations

from typing import Literal, Optional, List, Dict, Any, Union, TYPE_CHECKING

import time
import uuid

import shortuuid
from pydantic import BaseModel, Field


class ErrorResponse(BaseModel):
    """OpenAI 标准错误响应格式"""
    object: str = "error"
    message: str
    type: str = "invalid_request_error"
    param: Optional[str] = None
    code: int


class ModelPermission(BaseModel):
    id: str = Field(default_factory=lambda: f"modelperm-{shortuuid.random()}")
    object: str = "model_permission"
    created: int = Field(default_factory=lambda: int(time.time()))
    allow_create_engine: bool = False
    allow_sampling: bool = True
    allow_logprobs: bool = True
    allow_search_indices: bool = True
    allow_view: bool = True
    allow_fine_tuning: bool = False
    organization: str = "*"
    group: Optional[str] = None
    is_blocking: str = False


class ModelCard(BaseModel):
    id: str
    object: str = "model"
    created: int = Field(default_factory=lambda: int(time.time()))
    owned_by: str = "fastchat"
    root: Optional[str] = None
    parent: Optional[str] = None
    permission: List[ModelPermission] = []


class ModelList(BaseModel):
    object: str = "list"
    data: List[ModelCard] = []


class UsageInfo(BaseModel):
    prompt_tokens: int = 0
    total_tokens: int = 0
    completion_tokens: Optional[int] = 0


class LogProbs(BaseModel):
    text_offset: List[int] = Field(default_factory=list)
    token_logprobs: List[Optional[float]] = Field(default_factory=list)
    tokens: List[str] = Field(default_factory=list)
    top_logprobs: List[Optional[Dict[str, float]]] = Field(default_factory=list)


class FunctionDefinition(BaseModel):
    name: str
    description: Optional[str] = None
    parameters: Optional[dict[str, Any]] = None


class ChatCompletionToolsParam(BaseModel):
    type: Literal["function"] = "function"
    function: FunctionDefinition


class ChatCompletionNamedFunction(BaseModel):
    name: str


class ChatCompletionNamedToolChoiceParam(BaseModel):
    function: ChatCompletionNamedFunction
    type: Literal["function"] = "function"


class StreamOptions(BaseModel):
    """Options for streaming responses."""
    include_usage: Optional[bool] = True  # Include usage info in final chunk
    continuous_usage_stats: Optional[bool] = False  # Include usage in every chunk


class JsonSchemaResponseFormat(BaseModel):
    """JSON Schema for structured output."""
    name: str
    description: Optional[str] = None
    schema_: Optional[Dict[str, Any]] = Field(default=None, alias="schema")
    strict: Optional[bool] = False

    class Config:
        populate_by_name = True


class ResponseFormat(BaseModel):
    """Response format configuration for structured outputs."""
    # type must be "text", "json_object", or "json_schema"
    type: Literal["text", "json_object", "json_schema"] = "text"
    json_schema: Optional[JsonSchemaResponseFormat] = None


class ChatCompletionRequest(BaseModel):
    model: str
    messages: Optional[Union[
        str,
        List[Dict[str, str]],
        List[Dict[str, Union[str, List[Dict[str, Union[str, Dict[str, str]]]]]]],
    ]] = []
    prompt: Optional[str] = ""
    temperature: Optional[float] = 0.7
    top_p: Optional[float] = 1.0
    top_k: Optional[int] = -1
    n: Optional[int] = 1
    max_tokens: Optional[int] = None
    max_completion_tokens: Optional[int] = None  # OpenAI recommended replacement for max_tokens
    min_tokens: Optional[int] = 0
    logprobs: Optional[bool] = False  # Whether to return log probabilities
    top_logprobs: Optional[int] = None  # Number of most likely tokens to return (0-20)
    stop: Optional[Union[str, List[str]]] = None
    stream: Optional[bool] = False
    stream_options: Optional[StreamOptions] = None  # Options for streaming
    response_format: Optional[ResponseFormat] = None  # Structured output format
    presence_penalty: Optional[float] = 0.0
    frequency_penalty: Optional[float] = 0.0
    logit_bias: Optional[Dict[str, float]] = None  # Bias for specific tokens
    seed: Optional[int] = None  # Random seed for reproducibility
    user: Optional[str] = None
    tools: Optional[list[ChatCompletionToolsParam]] = None
    tool_choice: Optional[Union[
        Literal["none"],
        Literal["auto"],
        Literal["required"],
        ChatCompletionNamedToolChoiceParam,
    ]] = "none"
    parallel_tool_calls: Optional[bool] = True  # Allow parallel tool calls


class ChatMessage(BaseModel):
    """Chat message with optional tool_calls and reasoning support."""
    role: str
    content: Optional[str] = None
    tool_calls: Optional[List[ToolCall]] = None  # Tool calls made by assistant
    tool_call_id: Optional[str] = None  # For tool role messages
    name: Optional[str] = None  # Function name for tool messages
    reasoning: Optional[str] = None  # Reasoning/thinking content


class ChatCompletionResponseChoice(BaseModel):
    index: int
    message: ChatMessage
    logprobs: Optional[ChatCompletionLogProbs] = None  # Token logprobs
    finish_reason: Optional[Literal["stop", "length", "tool_calls", "content_filter"]] = None


class ChatCompletionResponse(BaseModel):
    id: str = Field(default_factory=lambda: f"chatcmpl-{shortuuid.random()}")
    object: str = "chat.completion"
    created: int = Field(default_factory=lambda: int(time.time()))
    model: str
    choices: List[ChatCompletionResponseChoice]
    usage: UsageInfo
    system_fingerprint: Optional[str] = None  # System fingerprint


class DeltaFunctionCall(BaseModel):
    name: Optional[str] = None
    arguments: Optional[str] = None

class DeltaToolCall(BaseModel):
    id: Optional[str] = None
    type: Optional[Literal["function"]] = None
    index: int
    function: Optional[DeltaFunctionCall] = None

class DeltaMessage(BaseModel):
    role: Optional[str] = None
    content: Optional[str] = None
    reasoning_content: Optional[str] = None
    tool_calls: list[DeltaToolCall] = Field(default_factory=list)

class FunctionCall(BaseModel):
    name: str
    arguments: str

class ToolCall(BaseModel):
    id: str = Field(default_factory=lambda: "fastllm-tool-" + str(uuid.uuid4().hex))
    type: Literal["function"] = "function"
    function: FunctionCall

class ExtractedToolCallInformation(BaseModel):
    # indicate if tools were called
    tools_called: bool

    # extracted tool calls
    tool_calls: list[ToolCall]

    # content - per OpenAI spec, content AND tool calls can be returned rarely
    # But some models will do this intentionally
    content: Optional[str] = None


# Logprobs for Chat Completion API
class Logprob(BaseModel):
    """Log probability for a single token."""
    logprob: float
    rank: Optional[int] = None
    decoded_token: Optional[str] = None


class ChatCompletionLogProb(BaseModel):
    """Log probability information for a token in chat completion."""
    token: str
    logprob: float
    bytes: Optional[List[int]] = None


class ChatCompletionLogProbsContent(ChatCompletionLogProb):
    """Log probability with top alternatives."""
    top_logprobs: List[ChatCompletionLogProb] = Field(default_factory=list)


class ChatCompletionLogProbs(BaseModel):
    """Log probabilities for chat completion response."""
    content: Optional[List[ChatCompletionLogProbsContent]] = None


class ChatCompletionResponseStreamChoice(BaseModel):
    index: int
    delta: DeltaMessage
    logprobs: Optional[ChatCompletionLogProbs] = None  # Token logprobs
    finish_reason: Optional[Literal["stop", "length", "tool_calls"]] = None  # Added tool_calls


class ChatCompletionStreamResponse(BaseModel):
    id: str = Field(default_factory=lambda: f"chatcmpl-{shortuuid.random()}")
    object: str = "chat.completion.chunk"
    created: int = Field(default_factory=lambda: int(time.time()))
    model: str
    choices: List[ChatCompletionResponseStreamChoice]
    usage: Optional[UsageInfo] = None  # Usage info (when stream_options.include_usage=True)


class TokenCheckRequestItem(BaseModel):
    model: str
    prompt: str
    max_tokens: int


class TokenCheckRequest(BaseModel):
    prompts: List[TokenCheckRequestItem]


class TokenCheckResponseItem(BaseModel):
    fits: bool
    tokenCount: int
    contextLength: int


class TokenCheckResponse(BaseModel):
    prompts: List[TokenCheckResponseItem]


class EmbeddingsRequest(BaseModel):
    model: Optional[str] = None
    engine: Optional[str] = None
    input: Union[str, List[Any]]
    user: Optional[str] = None
    encoding_format: Optional[str] = None


class EmbeddingsResponse(BaseModel):
    object: str = "list"
    data: List[Dict[str, Any]]
    model: str
    usage: UsageInfo


class CompletionRequest(BaseModel):
    model: str
    prompt: Union[str, List[Any]]
    suffix: Optional[str] = None
    temperature: Optional[float] = 0.7
    n: Optional[int] = 1
    max_tokens: Optional[int] = 16
    stop: Optional[Union[str, List[str]]] = None
    stream: Optional[bool] = False
    top_p: Optional[float] = 1.0
    top_k: Optional[int] = -1
    logprobs: Optional[int] = None
    echo: Optional[bool] = False
    presence_penalty: Optional[float] = 0.0
    frequency_penalty: Optional[float] = 0.0
    user: Optional[str] = None
    use_beam_search: Optional[bool] = False
    best_of: Optional[int] = None


class CompletionResponseChoice(BaseModel):
    index: int
    text: str
    logprobs: Optional[LogProbs] = None
    finish_reason: Optional[Literal["stop", "length"]] = None


class CompletionResponse(BaseModel):
    id: str = Field(default_factory=lambda: f"cmpl-{shortuuid.random()}")
    object: str = "text_completion"
    created: int = Field(default_factory=lambda: int(time.time()))
    model: str
    choices: List[CompletionResponseChoice]
    usage: UsageInfo


class CompletionResponseStreamChoice(BaseModel):
    index: int
    text: str
    logprobs: Optional[LogProbs] = None
    finish_reason: Optional[Literal["stop", "length"]] = None


class CompletionStreamResponse(BaseModel):
    id: str = Field(default_factory=lambda: f"cmpl-{shortuuid.random()}")
    object: str = "text_completion"
    created: int = Field(default_factory=lambda: int(time.time()))
    model: str
    choices: List[CompletionResponseStreamChoice]

class EmbedRequest(BaseModel):
    inputs: str
    normalize: Optional[bool] = False
    prompt_name: Optional[str] = "null"
    truncate: Optional[bool] = False
    truncation_direction: Optional[str] = 'right'

class RerankRequest(BaseModel):
    query: str
    texts: List[str]
    raw_scores: Optional[bool] = True
    return_text: Optional[bool] = False
    truncate: Optional[bool] = False
    truncation_direction: Optional[str] = "right"
