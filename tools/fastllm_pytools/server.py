import argparse
import fastapi
import logging
import sys
import uvicorn
import os
import json
from fastapi import Request
from fastapi.responses import JSONResponse, StreamingResponse
from fastapi.middleware.cors import CORSMiddleware

from .openai_server.protocal.openai_protocol import *
from .openai_server.fastllm_completion import FastLLmCompletion
from .openai_server.fastllm_embed import FastLLmEmbed
from .openai_server.fastllm_reranker import FastLLmReranker
from .openai_server.fastllm_model import FastLLmModel
from .util import make_normal_parser
from .util import add_server_args
from . import console

global fastllm_completion
global fastllm_embed
global fastllm_reranker
global fastllm_model
global dev_mode_enabled

def parse_args():
    parser = make_normal_parser("OpenAI 兼容 API 服务")
    add_server_args(parser)
    return parser.parse_args()

app = fastapi.FastAPI()
# 设置允许的请求来源, 生产环境请做对应变更
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

fastllm_completion:FastLLmCompletion
fastllm_embed:FastLLmEmbed
fastllm_model:FastLLmModel
dev_mode_enabled:bool = False

@app.post("/v1/chat/completions")
async def create_chat_completion(request: ChatCompletionRequest,
                                 raw_request: Request):
    generator = await fastllm_completion.create_chat_completion(
        request, raw_request)
    if isinstance(generator, ErrorResponse):
        return JSONResponse(content = generator.model_dump(),
                            status_code = generator.code)
    if request.stream:
        return StreamingResponse(content = generator[0],
                                 background = generator[1], 
                                 media_type = "text/event-stream")
    else:
        assert isinstance(generator, ChatCompletionResponse)
        return JSONResponse(content = generator.model_dump())


@app.post("/v1/completions")
async def create_completion(request: CompletionRequest,
                            raw_request: Request):
    """OpenAI 兼容的文本补全接口（非 chat 风格）"""
    generator = await fastllm_completion.create_completion(request, raw_request)
    if isinstance(generator, ErrorResponse):
        return JSONResponse(content = generator.model_dump(),
                            status_code = generator.code)
    if request.stream:
        return StreamingResponse(content = generator[0],
                                 background = generator[1],
                                 media_type = "text/event-stream")
    else:
        assert isinstance(generator, CompletionResponse)
        return JSONResponse(content = generator.model_dump())


@app.post("/v1/embed")
async def create_embed(request: EmbedRequest,
                       raw_request: Request):
    embedding = fastllm_embed.embedding_sentence(request, raw_request)
    return JSONResponse(embedding)


@app.post("/v1/embeddings")
async def create_embeddings(request: EmbeddingsRequest,
                            raw_request: Request):
    """OpenAI 标准的 embeddings 接口"""
    try:
        # 转换请求格式
        if isinstance(request.input, str):
            inputs = request.input
        elif isinstance(request.input, list) and len(request.input) > 0:
            inputs = request.input[0] if isinstance(request.input[0], str) else str(request.input[0])
        else:
            return JSONResponse(
                content={"error": {"message": "Input cannot be empty", "type": "invalid_request_error"}},
                status_code=400
            )
        
        # 使用现有的 embed 功能
        embed_req = EmbedRequest(inputs=inputs, normalize=True)
        result = fastllm_embed.embedding_sentence(embed_req, raw_request)
        
        # 转换为 OpenAI 格式
        embedding_data = result.get("data", [result]) if isinstance(result, dict) else [{"embedding": result}]
        
        response = EmbeddingsResponse(
            object="list",
            data=[{
                "object": "embedding",
                "embedding": item.get("embedding", item) if isinstance(item, dict) else item,
                "index": idx
            } for idx, item in enumerate(embedding_data)],
            model=request.model or fastllm_model.model_name,
            usage=UsageInfo(prompt_tokens=len(inputs.split()), total_tokens=len(inputs.split()))
        )
        return JSONResponse(content=response.model_dump())
    except Exception as e:
        logging.error(f"Embeddings error: {e}")
        return JSONResponse(
            content={"error": {"message": str(e), "type": "internal_error"}},
            status_code=500
        )


@app.post("/v1/rerank")
async def create_rerank(request: RerankRequest,
                       raw_request: Request):
    print(request)
    scores = fastllm_reranker.rerank(request, raw_request)    
    return JSONResponse(scores)


@app.get("/v1/models")
async def list_models():
    model_response = fastllm_model.response
    return JSONResponse(content = model_response)


@app.get("/health")
async def health_check():
    """健康检查接口"""
    return JSONResponse(content={"status": "healthy"})


@app.get("/v1/health")
async def v1_health_check():
    """v1 健康检查接口"""
    return JSONResponse(content={"status": "healthy"})


@app.get("/version")
async def get_version():
    """获取服务版本信息"""
    return JSONResponse(content={
        "version": "1.0.0",
        "engine": "fastllm"
    })


@app.post("/v1/cancel")
async def cancel_generation(request: Request):
    # Check if development mode is enabled
    if not dev_mode_enabled:
        return JSONResponse(content = {"error": "This API is only available in development mode"}, 
                            status_code = 403)
    
    try:
        json_data = await request.json()
        if 'conversation_id' not in json_data:
            return JSONResponse(content = {"error": "Missing required parameter: conversation_id"},
                                status_code = 400)
        
        conversation_id = json_data['conversation_id']
        success = fastllm_completion.abort_conversation(conversation_id)
        
        if success:
            return JSONResponse(content = {"message": f"Conversation {conversation_id} cancelled successfully"})
        else:
            return JSONResponse(content = {"error": f"Failed to cancel conversation {conversation_id}. Conversation not found or already finished."},
                                status_code = 404)
    except Exception as e:
        logging.error(f"Error cancelling conversation: {e}")
        return JSONResponse(content = {"error": f"Internal server error: {str(e)}"},
                            status_code = 500)

@app.get("/v1/active_conversations")
async def get_active_conversations():
    # Check if development mode is enabled
    if not dev_mode_enabled:
        return JSONResponse(content = {"error": "This API is only available in development mode"}, 
                            status_code = 403)
        
    try:
        conversations = fastllm_completion.get_active_conversations()
        return JSONResponse(content = {
            "active_conversations": conversations,
            "count": len(conversations)
        })
    except Exception as e:
        logging.error(f"Error getting active conversations: {e}")
        return JSONResponse(content = {"error": f"Internal server error: {str(e)}"},
                            status_code = 500)

def init_logging(log_level = logging.INFO, log_file:str = None):
    logging_format = '%(asctime)s %(process)d %(filename)s[line:%(lineno)d] %(levelname)s: %(message)s'
    root = logging.getLogger()
    root.setLevel(log_level)
    if log_file is not None:
        logging.basicConfig(level=log_level, filemode='a', filename=log_file, format=logging_format)
    stdout_handler = logging.StreamHandler(sys.stdout)
    stdout_handler.setFormatter(logging.Formatter(logging_format))
    root.addHandler(stdout_handler)

def fastllm_server(args):
    if args.api_key:
        @app.middleware("http")
        async def authentication(request: Request, call_next):
            print("auth")
            if request.method == "OPTIONS":
                return await call_next(request)
            url_path = request.url.path            
            if not url_path.startswith("/v1"):
                return await call_next(request)
            if request.headers.get("Authorization") != "Bearer " + args.api_key:
                return JSONResponse(content={"error": "Unauthorized"},
                                    status_code=401)
            return await call_next(request)
        
    global fastllm_completion
    global fastllm_embed
    global fastllm_reranker
    global fastllm_model
    global dev_mode_enabled
    
    console.header("API Server 配置")
    
    # Set development mode from args
    dev_mode_enabled = args.dev_mode
    if dev_mode_enabled:
        console.config("开发模式", "已启用 (会话管理 API 已激活)")
    
    init_logging()
    logging.info(args)
    
    from .util import make_normal_llm_model, format_device_map
    model = make_normal_llm_model(args)
    model.set_verbose(True)
    
    # 显示设备映射信息
    if hasattr(args, '_parsed_device_map') and args._parsed_device_map:
        console.config("设备映射", format_device_map(args._parsed_device_map))
    if hasattr(args, '_parsed_moe_device_map') and args._parsed_moe_device_map:
        console.config("MoE 设备映射", format_device_map(args._parsed_moe_device_map))
    
    if (args.model_name is None or args.model_name == ''):
        args.model_name = args.path
        if (args.model_name is None or args.model_name == ''):
            args.model_name = args.model

    fastllm_completion = FastLLmCompletion(
        model_name = args.model_name,
        model = model,
        think = (args.think.lower() != "false"),
        hide_input = args.hide_input
    )
    fastllm_embed = FastLLmEmbed(model_name = args.model_name, model = model)
    fastllm_reranker = FastLLmReranker(model_name = args.model_name, model = model)
    fastllm_model = FastLLmModel(model_name = args.model_name)
    
    console.header("服务就绪")
    console.info(f"监听地址: http://{args.host}:{args.port}")
    console.info("API 端点: /v1/chat/completions, /v1/completions, /v1/embeddings")
    print()
    
    uvicorn.run(app, host = args.host, port = args.port)

if __name__ == "__main__":
    args = parse_args()
    fastllm_server(args)
    