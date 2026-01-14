import asyncio
import logging
import json
import traceback
import time
import shortuuid
from fastapi import Request
from http import HTTPStatus
from typing import (AsyncGenerator, AsyncIterator, Awaitable, Dict, Iterable, List,
                    Optional, Tuple, TypedDict, Union, Any, final)
import uuid
from openai.types.chat import (ChatCompletionContentPartParam,
                               ChatCompletionRole)

from .protocal.openai_protocol import *
from .. import console


class ConversationMessage:
    """
    表示对话中的一条消息，支持完整的 OpenAI Chat API 格式。
    
    参考 vLLM 的 ConversationMessage 实现，支持：
    - 基本的 role 和 content
    - tool_calls: 模型请求调用的工具
    - tool_call_id: 工具返回消息的关联 ID
    - name: 消息发送者的名称（用于 tool 角色）
    - reasoning: 推理内容（用于 thinking 模式）
    - images: 图像列表（用于多模态模型）
    """
    def __init__(
        self, 
        role: str, 
        content: Optional[str] = None,
        tool_calls: Optional[List[Dict[str, Any]]] = None,
        tool_call_id: Optional[str] = None,
        name: Optional[str] = None,
        reasoning: Optional[str] = None,
        images: Optional[List[Any]] = None,
    ):
        self.role = role
        self.content = content
        self.tool_calls = tool_calls
        self.tool_call_id = tool_call_id
        self.name = name
        self.reasoning = reasoning
        self.images = images  # PIL.Image 对象列表
    
    def to_dict(self) -> Dict[str, Any]:
        """转换为字典格式，用于传递给模型"""
        result: Dict[str, Any] = {"role": self.role}
        if self.content is not None:
            result["content"] = self.content
        if self.tool_calls is not None:
            result["tool_calls"] = self.tool_calls
        if self.tool_call_id is not None:
            result["tool_call_id"] = self.tool_call_id
        if self.name is not None:
            result["name"] = self.name
        if self.reasoning is not None:
            result["reasoning"] = self.reasoning
        # images 不放入字典，需要单独处理
        return result


def random_uuid() -> str:
    return str(uuid.uuid4().hex)


class ChatCompletionStreamResponseWithUsage(BaseModel):
    id: str = Field(default_factory=lambda: f"chatcmpl-{shortuuid.random()}")
    object: str = "chat.completion.chunk"
    created: int = Field(default_factory=lambda: int(time.time()))
    model: str
    choices: List[ChatCompletionResponseStreamChoice]
    usage: Optional[UsageInfo] = Field(default=None)
    system_fingerprint: Optional[str] = None


class FastLLmCompletion:
    def __init__(self, model_name, model, think, show_input):
        self.model_name = model_name
        self.model = model
        self.init_fast_llm_model()
        self.think = think
        self.show_input = show_input
        # Store mapping between conversation IDs and handles
        self.conversation_handles = {}
        self.tool_parser = None
        # 生成系统指纹（用于标识当前模型配置）
        self._system_fingerprint = f"fp_{shortuuid.random()[:12]}"
        # 累计请求计数器
        self._total_request_count = 0

    @property
    def system_fingerprint(self) -> str:
        """返回系统指纹，用于标识当前模型配置"""
        return self._system_fingerprint

    def init_fast_llm_model(self):
        pass

    def create_error_response(
        self,
        message: str,
        err_type: str = "invalid_request_error",
        status_code: HTTPStatus = HTTPStatus.BAD_REQUEST,
        param: Optional[str] = None
    ) -> ErrorResponse:
        """
        创建 OpenAI 标准格式的错误响应
        
        常见错误类型:
        - invalid_request_error: 请求参数无效
        - authentication_error: 认证失败
        - permission_error: 权限不足
        - not_found_error: 资源不存在
        - rate_limit_error: 速率限制
        - server_error: 服务器内部错误
        """
        return ErrorResponse(
            message=message, 
            type=err_type, 
            code=status_code.value,
            param=param
        )
    
    def _validate_request_params(self, request: ChatCompletionRequest) -> Optional[ErrorResponse]:
        """验证请求参数的有效性"""
        # 验证 temperature
        if request.temperature is not None:
            if request.temperature < 0 or request.temperature > 2:
                return self.create_error_response(
                    message="temperature must be between 0 and 2",
                    param="temperature"
                )
        
        # 验证 top_p
        if request.top_p is not None:
            if request.top_p < 0 or request.top_p > 1:
                return self.create_error_response(
                    message="top_p must be between 0 and 1",
                    param="top_p"
                )
        
        # 验证 top_logprobs
        if request.top_logprobs is not None:
            if request.top_logprobs < 0 or request.top_logprobs > 20:
                return self.create_error_response(
                    message="top_logprobs must be between 0 and 20",
                    param="top_logprobs"
                )
        
        # 验证 n (当前不支持 n > 1)
        if request.n is not None and request.n > 1:
            return self.create_error_response(
                message="n > 1 is not supported yet",
                param="n"
            )
        
        # 验证 presence_penalty
        if request.presence_penalty is not None:
            if request.presence_penalty < -2 or request.presence_penalty > 2:
                return self.create_error_response(
                    message="presence_penalty must be between -2 and 2",
                    param="presence_penalty"
                )
        
        # 验证 frequency_penalty
        if request.frequency_penalty is not None:
            if request.frequency_penalty < -2 or request.frequency_penalty > 2:
                return self.create_error_response(
                    message="frequency_penalty must be between -2 and 2",
                    param="frequency_penalty"
                )
        
        return None

    async def _check_model(self, request: ChatCompletionRequest):
        if request.model != self.model_name:
            return self.create_error_response(
                message=f"The model `{request.model}` does not exist.",
                err_type="NotFoundError",
                status_code=HTTPStatus.NOT_FOUND
            )
        else:
            return None

    def _parse_image_content(self, image_data: Dict[str, Any]) -> Optional[Any]:
        """
        解析图像内容，支持 base64 和 URL 格式。
        
        支持的格式：
        - {"type": "image_url", "image_url": {"url": "data:image/...;base64,..."}}
        - {"type": "image_url", "image_url": {"url": "https://..."}}
        - {"type": "image", "image": "base64..."}
        
        返回 PIL.Image 对象或 None（解析失败时）
        """
        try:
            from PIL import Image
            import io
            import base64
            
            part_type = image_data.get('type', '')
            
            if part_type == 'image_url':
                image_url_obj = image_data.get('image_url', {})
                if isinstance(image_url_obj, str):
                    url = image_url_obj
                else:
                    url = image_url_obj.get('url', '')
                
                if not url:
                    logging.warning("Empty image URL")
                    return None
                
                # 处理 base64 编码的图像
                if url.startswith('data:'):
                    # 格式: data:image/jpeg;base64,/9j/4AAQ...
                    try:
                        header, encoded = url.split(',', 1)
                        image_bytes = base64.b64decode(encoded)
                        image = Image.open(io.BytesIO(image_bytes)).convert('RGB')
                        logging.debug(f"Parsed base64 image: {image.size}")
                        return image
                    except Exception as e:
                        logging.warning(f"Failed to parse base64 image: {e}")
                        return None
                
                # 处理 URL 图像
                elif url.startswith('http://') or url.startswith('https://'):
                    try:
                        import requests
                        response = requests.get(url, timeout=30)
                        response.raise_for_status()
                        image = Image.open(io.BytesIO(response.content)).convert('RGB')
                        logging.debug(f"Downloaded image from URL: {image.size}")
                        return image
                    except ImportError:
                        logging.warning("requests library not available for URL image download")
                        return None
                    except Exception as e:
                        logging.warning(f"Failed to download image from URL: {e}")
                        return None
                else:
                    logging.warning(f"Unsupported image URL format: {url[:50]}...")
                    return None
            
            elif part_type == 'image':
                # 直接 base64 编码
                encoded = image_data.get('image', '')
                if encoded:
                    try:
                        image_bytes = base64.b64decode(encoded)
                        image = Image.open(io.BytesIO(image_bytes)).convert('RGB')
                        logging.debug(f"Parsed direct base64 image: {image.size}")
                        return image
                    except Exception as e:
                        logging.warning(f"Failed to parse direct base64 image: {e}")
                        return None
            
            return None
            
        except ImportError as e:
            logging.warning(f"PIL not available for image processing: {e}")
            return None
        except Exception as e:
            logging.warning(f"Error parsing image content: {e}")
            return None

    def _flatten_text_content(self, content: Optional[Union[str, List]]) -> Optional[str]:
        """
        从消息内容中提取文本部分并扁平化为单个字符串。
        参考 vLLM 的 flatten_chat_text_content 实现。
        """
        if content is None:
            return None
        if isinstance(content, str):
            return content
        if isinstance(content, list):
            text_parts = []
            for item in content:
                if isinstance(item, str):
                    text_parts.append(item)
                elif isinstance(item, dict) and item.get("type") == "text":
                    text_parts.append(item.get("text", ""))
            return "".join(text_parts) if text_parts else None
        return None

    def _parse_chat_message_content(
        self,
        role: ChatCompletionRole,
        content: Optional[Union[str, Iterable[ChatCompletionContentPartParam]]],
    ) -> Tuple[List[ConversationMessage], List[Any]]:
        """
        解析聊天消息内容，支持多种格式：
        - None: 返回空列表
        - str: 直接作为文本内容
        - list: 解析各种内容部分（text, image_url, tool_use, tool_result 等）
        
        参考 vLLM 的实现，增强对复杂消息格式的兼容性。
        返回 (消息列表, 图像列表)
        """
        if content is None:
            return [ConversationMessage(role=role, content="")], []
        if isinstance(content, str):
            return [ConversationMessage(role=role, content=content)], []
        if isinstance(content, list):
            content_str = ""
            images = []  # 收集图像
            for it in content:
                # 处理纯字符串元素
                if isinstance(it, str):
                    if content_str:
                        content_str += "\n"
                    content_str += it
                    continue
                # 处理字典格式的内容部分
                if isinstance(it, dict):
                    part_type = it.get('type', '')
                    # 处理文本类型
                    if part_type == 'text':
                        if content_str:
                            content_str += "\n"
                        content_str += it.get('text', '')
                    # 处理图像类型
                    elif part_type in ('image_url', 'image'):
                        image = self._parse_image_content(it)
                        if image is not None:
                            images.append(image)
                            logging.info(f"Parsed image content: {image.size}")
                        else:
                            logging.warning(f"Failed to parse image content")
                    # 跳过工具调用相关内容（由 _parse_chat_message 统一处理）
                    elif part_type in ('tool_use', 'tool_result', 'tool_calls', 'function'):
                        logging.debug(f"Skipping tool-related content in content array: {part_type}")
                        continue
                    # 音频和视频暂不支持
                    elif part_type in ('audio', 'audio_url', 'video', 'video_url', 'input_audio'):
                        logging.warning(f"Multimodal content type not supported: {part_type}")
                        continue
                    # 处理 refusal 类型
                    elif part_type == 'refusal':
                        refusal_text = it.get('refusal', '')
                        if refusal_text:
                            if content_str:
                                content_str += "\n"
                            content_str += refusal_text
                    # 跳过其他未知类型，记录警告但不报错
                    else:
                        logging.warning(f"Skipping unknown content type: {part_type}")
                        continue
                else:
                    # 跳过不支持的格式
                    logging.warning(f"Skipping unsupported content format: {type(it)}")
                    continue
            msg = ConversationMessage(role=role, content=content_str, images=images if images else None)
            return [msg], images
        # 暂时不支持其他格式的输入
        logging.warning(f"Unsupported content type: {type(content)}")
        raise NotImplementedError("Complex input not supported yet")

    def _parse_chat_message(
        self,
        message: Dict[str, Any],
    ) -> List[ConversationMessage]:
        """
        解析完整的聊天消息，支持 OpenAI Chat API 的所有字段。
        
        参考 vLLM 的 _parse_chat_message_content 和 parse_chat_input_to_harmony_message 实现。
        
        支持的消息格式：
        - 用户消息: {"role": "user", "content": "..."}
        - 系统消息: {"role": "system", "content": "..."}
        - 助手消息: {"role": "assistant", "content": "...", "tool_calls": [...]}
        - 工具消息: {"role": "tool", "tool_call_id": "...", "content": "..."}
        """
        role = message.get("role", "user")
        content = message.get("content")
        tool_calls = message.get("tool_calls")
        tool_call_id = message.get("tool_call_id")
        name = message.get("name")
        reasoning = message.get("reasoning") or message.get("reasoning_content")
        
        result: List[ConversationMessage] = []
        
        # 处理 assistant 消息带 tool_calls
        if role == "assistant" and tool_calls:
            # 提取文本内容
            text_content = self._flatten_text_content(content)
            
            # 创建包含 tool_calls 的消息
            result.append(ConversationMessage(
                role=role,
                content=text_content,
                tool_calls=tool_calls,
                reasoning=reasoning,
            ))
            return result
        
        # 处理 tool 角色消息（工具返回结果）
        if role == "tool":
            text_content = self._flatten_text_content(content)
            result.append(ConversationMessage(
                role=role,
                content=text_content,
                tool_call_id=tool_call_id,
                name=name,
            ))
            return result
        
        # 处理普通消息（user, system, assistant without tool_calls）
        messages, _ = self._parse_chat_message_content(role, content)
        for msg in messages:
            msg.reasoning = reasoning
            msg.name = name
        result.extend(messages)
        
        return result

    async def create_chat_completion(
        self, request: ChatCompletionRequest, raw_request: Request
    ) -> Union[ErrorResponse, AsyncGenerator[str, None],
               ChatCompletionResponse,
               Tuple[AsyncGenerator[str, None], AsyncGenerator]]:
        """Completion API similar to OpenAI's API.

        See https://platform.openai.com/docs/api-reference/chat/create
        for the API specification. This API mimics the OpenAI
        ChatCompletion API.

        NOTE: Currently we do not support the following feature:
            - function_call (Users should implement this by themselves)
        """
        # 检查模型是否存在
        error_check_ret = await self._check_model(request)
        if error_check_ret is not None:
            return error_check_ret
        
        # 验证请求参数
        param_error = self._validate_request_params(request)
        if param_error is not None:
            return param_error

        query: str = ""
        if request.prompt:
            request.messages.append({"role": "user", "content": request.prompt})
        try:
            # 使用新的 _parse_chat_message 方法解析完整消息
            conversation: List[ConversationMessage] = []
            all_images: List[Any] = []  # 收集所有图像
            for m in request.messages:
                parsed_messages = self._parse_chat_message(m)
                for msg in parsed_messages:
                    conversation.append(msg)
                    # 收集图像
                    if msg.images:
                        all_images.extend(msg.images)

            if len(conversation) == 0:
                raise Exception("Empty msg")
            
            # 转换为模型需要的格式，保留所有字段
            messages = []
            for msg in conversation:
                messages.append(msg.to_dict())
            
            # 图像信息（用于多模态模型）
            images = all_images if all_images else None
            if images:
                logging.info(f"Detected {len(images)} images in request")

        except Exception as e:
            logging.error("Error in applying chat template from request: %s", e)
            traceback.print_exc()
            return self.create_error_response(str(e))

        request_id = f"fastllm-{self.model_name}-{random_uuid()}"

        frequency_penalty = 1.0
        if request.frequency_penalty and request.frequency_penalty != 0.0:
            frequency_penalty = request.frequency_penalty

        max_length = request.max_tokens if request.max_tokens else 32768
        min_length = request.min_tokens if request.min_tokens else 0
        # logging.info(request)

        if self.show_input:
            logging.debug(f"fastllm input message: {messages}")
            # logging.debug(f"input tokens: {input_token_len}")

        input_token_len = self.model.get_input_token_len(messages)

        tools = [tool.model_dump(exclude_none=True) for tool in request.tools] if request.tools is not None else None
        # print("tools", tools)

        # 累计请求数
        self._total_request_count += 1
        console.request_start(self._total_request_count, request_id)

        # from request.tools
        handle = self.model.launch_stream_response(
            messages,
            max_length=max_length,
            min_length=min_length,
            do_sample=True,
            top_p=request.top_p,
            top_k=request.top_k,
            temperature=request.temperature,
            repeat_penalty=frequency_penalty,
            tools=tools,
            one_by_one=True,
            images=images,  # 传递图像给多模态模型
        )
        # Store the mapping between conversation ID and handle
        self.conversation_handles[request_id] = handle
        logging.debug(f"Created conversation: {request_id}, handle: {handle}")
        result_generator = self.model.stream_response_handle_async(handle)
        # Streaming response
        if request.stream:
            return (
                self.chat_completion_stream_generator(
                    request, raw_request, handle, result_generator, request_id, input_token_len, think=self.think
                ),
                None
            )
        else:
            try:
                return await self.chat_completion_full_generator(
                    request, raw_request, handle, result_generator, request_id, input_token_len
                )
            except ValueError as e:
                return self.create_error_response(str(e))

    async def chat_completion_full_generator(
        self,
        request: ChatCompletionRequest,
        raw_request: Request,
        handle: int,
        result_generator: AsyncIterator,
        request_id: str,
        input_token_len: int
    ) -> Union[ErrorResponse, ChatCompletionResponse]:
        model_name = self.model_name
        created_time = int(time.time())
        result = ""
        completion_tokens = 0
        async for res in result_generator:
            result += res
            completion_tokens += 1
            if await raw_request.is_disconnected():
                print("is_disconnected!!!")
                self.model.abort_handle(handle)
                logging.debug(f"Abort request: {request_id}")
                return self.create_error_response("Client disconnected")

        # 使用工具解析器提取工具调用（如果有）
        finish_reason = 'stop'
        tool_calls_list = None
        content = result
        
        if self.tool_parser and request.tools:
            try:
                extracted = self.tool_parser.extract_tool_calls(result, request)
                if extracted.tools_called and extracted.tool_calls:
                    # 处理 parallel_tool_calls 参数
                    if request.parallel_tool_calls is False and len(extracted.tool_calls) > 1:
                        # 只保留第一个工具调用
                        tool_calls_list = [extracted.tool_calls[0]]
                    else:
                        tool_calls_list = extracted.tool_calls
                    finish_reason = 'tool_calls'
                    content = extracted.content  # 可能为 None 或包含部分内容
            except Exception as e:
                logging.warning(f"Tool parsing failed: {e}, returning raw content")
        
        choice_data = ChatCompletionResponseChoice(
            index=0,
            message=ChatMessage(role="assistant", content=content, tool_calls=tool_calls_list),
            logprobs=None,
            finish_reason=finish_reason,
        )

        response = ChatCompletionResponse(
            id=request_id,
            created=created_time,
            model=model_name,
            choices=[choice_data],
            usage=UsageInfo(
                prompt_tokens=input_token_len,
                total_tokens=input_token_len + completion_tokens,
                completion_tokens=completion_tokens
            ),
            system_fingerprint=self.system_fingerprint
        )

        # 请求处理完成（先输出"请求完成"，再输出统计信息）
        console.request_complete(request_id)

        # 打印推理统计信息
        stats = self.model.get_handle_stats(handle)
        if stats:
            console.print_inference_stats(
                stats.prompt_tokens, stats.output_tokens,
                stats.total_time, stats.first_token_time, stats.speed
            )

        # After completion, remove the conversation from tracking dictionary
        if request_id in self.conversation_handles:
            del self.conversation_handles[request_id]
            logging.debug(f"Removed completed conversation from tracking: {request_id}")

        return response

    async def chat_completion_stream_generator(
        self,
        request: ChatCompletionRequest,
        raw_request: Request,
        handle: int,
        result_generator: AsyncIterator,
        request_id: str,
        input_token_len: int,
        think: bool
    ) -> AsyncGenerator[str, None]:
        model_name = self.model_name
        created_time = int(time.time())
        chunk_object_type = "chat.completion.chunk"

        # 解析 stream_options
        include_usage = False
        continuous_usage = False
        if request.stream_options:
            include_usage = request.stream_options.include_usage or False
            continuous_usage = request.stream_options.continuous_usage_stats or False

        # TODO: 支持request.n 和 request.echo配置
        first_iteration = True
        completion_tokens = 0
        try:
            if first_iteration:
                # 1. role部分
                choice_data = ChatCompletionResponseStreamChoice(
                    index=0,
                    delta=DeltaMessage(role="assistant"),
                    logprobs=None,
                    finish_reason=None
                )
                # 如果启用 continuous_usage，在首个 chunk 就包含 usage
                usage_info = None
                if continuous_usage:
                    usage_info = UsageInfo(
                        prompt_tokens=input_token_len,
                        total_tokens=input_token_len,
                        completion_tokens=0
                    )
                chunk = ChatCompletionStreamResponseWithUsage(
                    id=request_id,
                    object=chunk_object_type,
                    created=created_time,
                    choices=[choice_data],
                    model=model_name,
                    usage=usage_info
                )
                data = chunk.model_dump_json(exclude_unset=True, exclude_none=True)
                yield f"data: {data}\n\n"
                first_iteration = False

            # 2. content部分

            if request.tools and self.tool_parser is None:
                # tools不为空
                from .tool_parsers import ToolParser, ToolParserManager
                self.tool_parser = ToolParserManager.get_tool_parser_auto(
                    self.model.get_type(),
                    self.model.hf_tokenizer.chat_template,
                    force_chat_template=self.model.force_chat_template,
                    force_type=self.model.tool_call_parser
                )(self.model.hf_tokenizer)

            previous_token_ids = []
            current_token_ids = []
            previous_text = ""
            current_text = ""

            async for res in result_generator:
                if await raw_request.is_disconnected():
                    self.model.abort_handle(handle)
                    logging.debug(f"Abort stream request (client disconnected): {request_id}")
                    return
                completion_tokens += 1
                delta_text = res

                # print("delta_text", delta_text)

                # Send token-by-token response for each request.n
                if self.tool_parser and request.tools:
                    now_ids = self.tool_parser.get_token_ids(delta_text)
                    # print("delta_text", delta_text, "now_ids", now_ids)

                    current_text += delta_text
                    current_token_ids += now_ids

                    delta_message = self.tool_parser.extract_tool_calls_streaming(
                        previous_text=previous_text,
                        current_text=current_text,
                        delta_text=delta_text,
                        previous_token_ids=previous_token_ids,
                        current_token_ids=current_token_ids,
                        delta_token_ids=[0],
                        request=request
                    )

                    previous_text += delta_text
                    previous_token_ids += now_ids
                    # print("delta", delta_message)
                    
                    # 处理 parallel_tool_calls=False 时只保留第一个工具调用
                    if delta_message and request.parallel_tool_calls is False:
                        if delta_message.tool_calls:
                            # 过滤只保留 index=0 的工具调用
                            delta_message.tool_calls = [
                                tc for tc in delta_message.tool_calls if tc.index == 0
                            ]
                else:
                    delta_message = DeltaMessage(content=delta_text)

                if delta_message:
                    choice_data = ChatCompletionResponseStreamChoice(
                        index=0,
                        # delta = DeltaMessage(content = delta_text),
                        delta=delta_message,
                        logprobs=None,
                        finish_reason=None
                    )
                    # 如果启用 continuous_usage，每个 chunk 都包含 usage
                    usage_info = None
                    if continuous_usage:
                        usage_info = UsageInfo(
                            prompt_tokens=input_token_len,
                            total_tokens=input_token_len + completion_tokens,
                            completion_tokens=completion_tokens
                        )
                    chunk = ChatCompletionStreamResponseWithUsage(
                        id=request_id,
                        object=chunk_object_type,
                        created=created_time,
                        choices=[choice_data],
                        model=model_name,
                        usage=usage_info,
                        system_fingerprint=self.system_fingerprint
                    )
                    data = chunk.model_dump_json(exclude_unset=True, exclude_none=True)
                    yield f"data: {data}\n\n"
                # await asyncio.sleep(0)

            # 3. 结束标志
            choice_data = ChatCompletionResponseStreamChoice(
                index=0,
                delta=DeltaMessage(),
                logprobs=None,
                finish_reason='stop'
            )
            # 根据 include_usage 决定是否在最后一个 chunk 输出 usage
            final_usage = None
            if include_usage or continuous_usage:
                final_usage = UsageInfo(
                    prompt_tokens=input_token_len,
                    total_tokens=input_token_len + completion_tokens,
                    completion_tokens=completion_tokens
                )
            chunk = ChatCompletionStreamResponseWithUsage(
                id=request_id,
                object=chunk_object_type,
                created=created_time,
                choices=[choice_data],
                model=model_name,
                usage=final_usage,
                system_fingerprint=self.system_fingerprint
            )
            data = chunk.model_dump_json(exclude_unset=True, exclude_none=True)
            yield f"data: {data}\n\n"
        except ValueError as e:
            data = self.create_streaming_error_response(str(e))
            yield f"data: {data}\n\n"
            await asyncio.sleep(0)
        except asyncio.CancelledError:
            # 客户端断开通常会触发取消；确保模型侧也尽快停止
            try:
                self.model.abort_handle(handle)
                logging.debug(f"Abort stream request (cancelled): {request_id}")
            except Exception:
                pass
            raise
        finally:
            # 请求处理完成（先输出"请求完成"，再输出统计信息）
            console.request_complete(request_id)
            
            # 打印推理统计信息
            stats = self.model.get_handle_stats(handle)
            if stats:
                console.print_inference_stats(
                    stats.prompt_tokens, stats.output_tokens,
                    stats.total_time, stats.first_token_time, stats.speed
                )
            else:
                logging.debug(f"无法获取请求 {request_id} 的统计信息")
            
            # 正常结束时不要调用 abort，否则会破坏历史 KV cache 写入，导致每轮 Long Prefill。
            if request_id in self.conversation_handles:
                del self.conversation_handles[request_id]
                logging.debug(f"已移除已完成的流式对话: {request_id}")

        yield "data: [DONE]\n\n"
        await asyncio.sleep(0)

    def create_streaming_error_response(
        self,
        message: str,
        err_type: str = "BadRequestError",
        status_code: HTTPStatus = HTTPStatus.BAD_REQUEST
    ) -> str:
        json_str = json.dumps({
            "error": self.create_error_response(
                message=message,
                err_type=err_type,
                status_code=status_code
            ).model_dump()
        })
        return json_str

    def abort_conversation(self, conversation_id: str) -> bool:
        if conversation_id in self.conversation_handles:
            handle = self.conversation_handles[conversation_id]
            try:
                self.model.abort_handle(handle)
                logging.debug(f"Aborted conversation: {conversation_id}, handle: {handle}")
                # Remove the conversation from the mapping
                del self.conversation_handles[conversation_id]
                return True
            except Exception as e:
                logging.error(f"Error aborting conversation {conversation_id}: {e}")
                return False
        else:
            logging.warning(f"Conversation ID not found: {conversation_id}")
            return False

    def get_active_conversations(self) -> List[Dict[str, Any]]:
        result = []
        for conversation_id, handle in self.conversation_handles.items():
            result.append({
                "conversation_id": conversation_id,
                "handle": handle
            })
        return result

    # ==================== /v1/completions 支持 ====================
    
    async def create_completion(
        self,
        request: CompletionRequest,
        raw_request: Request
    ) -> Union[ErrorResponse, CompletionResponse, Tuple[AsyncGenerator[str, None], None]]:
        """
        处理 /v1/completions 请求（非 chat 风格的文本补全）
        """
        request_id = f"cmpl-{shortuuid.random()}"
        
        # 处理 prompt - 可能是字符串或字符串列表
        if isinstance(request.prompt, list):
            # 如果是列表，只处理第一个（简化实现）
            if len(request.prompt) == 0:
                return self.create_error_response("Prompt cannot be empty")
            prompt = request.prompt[0] if isinstance(request.prompt[0], str) else str(request.prompt[0])
        else:
            prompt = request.prompt
        
        if not prompt:
            return self.create_error_response("Prompt cannot be empty")
        
        # 计算 max_tokens
        max_length = request.max_tokens or 16
        
        # 使用模型生成
        input_token_len = self.model.get_input_token_len(prompt)
        
        # 累计请求数
        self._total_request_count += 1
        console.request_start(self._total_request_count, request_id)
        
        handle = self.model.launch_stream_response(
            prompt,
            max_length=max_length,
            do_sample=True,
            top_p=request.top_p,
            top_k=request.top_k,
            temperature=request.temperature,
            repeat_penalty=request.frequency_penalty,
            one_by_one=True,
        )
        
        self.conversation_handles[request_id] = handle
        result_generator = self.model.stream_response_handle_async(handle)
        
        if request.stream:
            return (
                self._completion_stream_generator(
                    request, raw_request, handle, result_generator, request_id, input_token_len
                ),
                None
            )
        else:
            return await self._completion_full_generator(
                request, raw_request, handle, result_generator, request_id, input_token_len
            )
    
    async def _completion_full_generator(
        self,
        request: CompletionRequest,
        raw_request: Request,
        handle: int,
        result_generator: AsyncIterator,
        request_id: str,
        input_token_len: int
    ) -> Union[ErrorResponse, CompletionResponse]:
        """非流式 completion 响应"""
        model_name = self.model_name
        created_time = int(time.time())
        result = ""
        completion_tokens = 0
        
        # 如果 echo=True，先添加原始 prompt
        if request.echo:
            result = request.prompt if isinstance(request.prompt, str) else request.prompt[0]
        
        async for res in result_generator:
            result += res
            completion_tokens += 1
            if await raw_request.is_disconnected():
                self.model.abort_handle(handle)
                logging.debug(f"Abort completion request: {request_id}")
                return self.create_error_response("Client disconnected")
        
        # 确定 finish_reason
        finish_reason = "stop"
        if completion_tokens >= (request.max_tokens or 16):
            finish_reason = "length"
        
        choice_data = CompletionResponseChoice(
            index=0,
            text=result,
            logprobs=None,
            finish_reason=finish_reason,
        )
        
        response = CompletionResponse(
            id=request_id,
            created=created_time,
            model=model_name,
            choices=[choice_data],
            usage=UsageInfo(
                prompt_tokens=input_token_len,
                total_tokens=input_token_len + completion_tokens,
                completion_tokens=completion_tokens
            )
        )
        
        # 请求处理完成（先输出“请求完成”，再输出统计信息）
        console.request_complete(request_id)
        
        # 打印推理统计信息
        stats = self.model.get_handle_stats(handle)
        if stats:
            console.print_inference_stats(
                stats.prompt_tokens, stats.output_tokens,
                stats.total_time, stats.first_token_time, stats.speed
            )
        
        if request_id in self.conversation_handles:
            del self.conversation_handles[request_id]
        
        return response
    
    async def _completion_stream_generator(
        self,
        request: CompletionRequest,
        raw_request: Request,
        handle: int,
        result_generator: AsyncIterator,
        request_id: str,
        input_token_len: int
    ) -> AsyncGenerator[str, None]:
        """流式 completion 响应"""
        model_name = self.model_name
        created_time = int(time.time())
        completion_tokens = 0
        
        try:
            # 如果 echo=True，先发送原始 prompt
            if request.echo:
                echo_text = request.prompt if isinstance(request.prompt, str) else request.prompt[0]
                choice_data = CompletionResponseStreamChoice(
                    index=0,
                    text=echo_text,
                    logprobs=None,
                    finish_reason=None
                )
                chunk = CompletionStreamResponse(
                    id=request_id,
                    created=created_time,
                    model=model_name,
                    choices=[choice_data]
                )
                yield f"data: {chunk.model_dump_json(exclude_unset=True)}\n\n"
            
            async for res in result_generator:
                if await raw_request.is_disconnected():
                    self.model.abort_handle(handle)
                    logging.debug(f"Abort completion stream: {request_id}")
                    return
                
                completion_tokens += 1
                
                choice_data = CompletionResponseStreamChoice(
                    index=0,
                    text=res,
                    logprobs=None,
                    finish_reason=None
                )
                chunk = CompletionStreamResponse(
                    id=request_id,
                    created=created_time,
                    model=model_name,
                    choices=[choice_data]
                )
                yield f"data: {chunk.model_dump_json(exclude_unset=True)}\n\n"
            
            # 发送最终 chunk
            finish_reason = "stop"
            if completion_tokens >= (request.max_tokens or 16):
                finish_reason = "length"
            
            choice_data = CompletionResponseStreamChoice(
                index=0,
                text="",
                logprobs=None,
                finish_reason=finish_reason
            )
            chunk = CompletionStreamResponse(
                id=request_id,
                created=created_time,
                model=model_name,
                choices=[choice_data]
            )
            yield f"data: {chunk.model_dump_json(exclude_unset=True)}\n\n"
            
        finally:
            # 请求处理完成（先输出“请求完成”，再输出统计信息）
            console.request_complete(request_id)
            
            # 打印推理统计信息
            stats = self.model.get_handle_stats(handle)
            if stats:
                console.print_inference_stats(
                    stats.prompt_tokens, stats.output_tokens,
                    stats.total_time, stats.first_token_time, stats.speed
                )
            
            if request_id in self.conversation_handles:
                del self.conversation_handles[request_id]
        
        yield "data: [DONE]\n\n"
