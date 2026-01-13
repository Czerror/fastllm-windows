"""
help_text.py - 统一帮助文本定义
提供跨 C++ 和 Python 的统一帮助信息
与 C++ include/utils/help_text.h 保持同步
"""

from dataclasses import dataclass, field
from typing import Optional, List
import argparse

# ============================================================================
# 程序信息
# ============================================================================
PROGRAM_NAME = "ftllm"
PROGRAM_DESC = "FastLLM 统一命令行工具"
PROGRAM_VERSION = "1.0"


# ============================================================================
# 数据结构定义
# ============================================================================
@dataclass
class ParamDef:
    """参数定义"""
    name: str           # 参数名 (如 "-p, --path <路径>")
    desc: str           # 描述
    py_name: Optional[str] = None      # Python argparse 名称
    py_type: Optional[str] = None      # Python 类型
    default_val: Optional[str] = None  # 默认值


@dataclass
class ParamGroup:
    """参数组定义"""
    title: str
    params: List[ParamDef]


@dataclass
class CommandDef:
    """命令定义"""
    name: str
    aliases: List[str]
    exe: Optional[str]  # Native 后端的可执行文件名 (Python 为 None)
    desc: str
    is_native: bool     # True: C++ Native, False: Python


@dataclass
class ExampleDef:
    """示例定义"""
    cmd: str
    model: str
    args: Optional[str] = None


@dataclass
class ModelFormatDef:
    """模型格式定义"""
    format: str
    desc: str


# ============================================================================
# 命令定义
# ============================================================================
COMMANDS = [
    # Native (C++) 命令
    CommandDef("serve", ["server", "api"], "apiserver.exe", "启动 OpenAI 兼容 API 服务器", True),
    CommandDef("webui", ["web"], "webui.exe", "启动 Web 界面", True),
    CommandDef("bench", ["benchmark"], "benchmark.exe", "性能测试", True),
    CommandDef("quant", ["quantize"], "quant.exe", "模型量化", True),
    # Python 命令
    CommandDef("run", ["chat"], None, "交互式聊天", False),
    CommandDef("download", [], None, "下载 HuggingFace 模型", False),
    CommandDef("ui", [], None, "启动图形界面", False),
    CommandDef("config", [], None, "生成配置文件模板", False),
    CommandDef("export", [], None, "导出模型", False),
]


# ============================================================================
# 参数组定义
# ============================================================================
PARAM_GROUPS = [
    ParamGroup("基础参数", [
        ParamDef("-p, --path <路径>", "模型路径", "path", "str"),
        ParamDef("--device <设备>", "cuda, cpu, numa", "device", "str"),
        ParamDef("--dtype <类型>", "float16, int8, int4, int4g", "dtype", "str", "auto"),
        ParamDef("-t, --threads <数量>", "CPU 线程数", "threads", "int", "-1"),
        ParamDef("--model_name <名称>", "模型显示名称 (用于 API 返回)", "model_name", "str"),
    ]),
    ParamGroup("服务器参数", [
        ParamDef("--host <地址>", "监听地址 (默认: 127.0.0.1)", "host", "str", "127.0.0.1"),
        ParamDef("--port <端口>", "监听端口 (默认: 8080)", "port", "int", "8080"),
        ParamDef("--api_key <密钥>", "API 密钥认证 (Bearer Token)", "api_key", "str"),
        ParamDef("--embedding_path <路径>", "Embedding 模型路径", "embedding_path", "str"),
        ParamDef("--dev_mode", "开发模式 (启用调试接口)", "dev_mode", "bool"),
    ]),
    ParamGroup("Batch / 并发参数", [
        ParamDef("--batch <数量>", "批处理大小", "max_batch", "int", "-1"),
        ParamDef("--max_batch <数量>", "最大批处理数量", "max_batch", "int", "-1"),
        ParamDef("--max_token <数量>", "最大生成 Token 数 (webui)", "max_token", "int", "4096"),
        ParamDef("--chunk_size <数量>", "Chunked Prefill 分块大小", "chunk_size", "int"),
    ]),
    ParamGroup("CUDA / 加速参数", [
        ParamDef("--cuda_embedding", "在 CUDA 上运行 Embedding 层", "cuda_embedding", "bool"),
        ParamDef("--cuda_shared_expert", "CUDA 共享专家优化 (MOE)", "cuda_shared_expert", "str", "true"),
        ParamDef("--cuda_se", "--cuda_shared_expert 简写", "cuda_se", "str", "true"),
        ParamDef("--enable_amx, --amx", "启用 Intel AMX 加速", "enable_amx", "str", "false"),
    ]),
    ParamGroup("MOE (混合专家) 参数", [
        ParamDef("--moe_device <设备>", "MOE 专家层设备 (cuda, cpu)", "moe_device", "str"),
        ParamDef("--moe_dtype <类型>", "MOE 专家层数据类型", "moe_dtype", "str"),
        ParamDef("--moe_experts <数量>", "启用的 MOE 专家数量", "moe_experts", "int", "-1"),
    ]),
    ParamGroup("缓存参数", [
        ParamDef("--kv_cache_limit <大小>", "KV 缓存限制 (如 8G, 4096M)", "kv_cache_limit", "str", "auto"),
        ParamDef("--cache_history", "启用历史缓存", "cache_history", "str"),
        ParamDef("--cache_fast", "启用快速缓存模式", "cache_fast", "str"),
        ParamDef("--cache_dir <路径>", "缓存目录路径", "cache_dir", "str"),
    ]),
    ParamGroup("LoRA 参数", [
        ParamDef("--lora <路径>", "LoRA 适配器路径", "lora", "str"),
        ParamDef("--custom <配置>", "自定义模型配置", "custom", "str"),
        ParamDef("--dtype_config <配置>", "数据类型配置文件", "dtype_config", "str"),
        ParamDef("--ori", "使用原始权重 (禁用量化)", "ori", "str"),
    ]),
    ParamGroup("模板 / 工具调用", [
        ParamDef("--chat_template <模板>", "对话模板 (覆盖自动检测)", "chat_template", "str"),
        ParamDef("--tool_call_parser <类型>", "工具调用解析器类型", "tool_call_parser", "str", "auto"),
        ParamDef("--enable_thinking", "启用思考模式 (<think>标签)", "enable_thinking", "str"),
        ParamDef("--think", "Python 后端思考模式", "think", "str", "false"),
        ParamDef("--show_input", "显示输入消息 (调试用)", "show_input", "bool"),
    ]),
    ParamGroup("开发 / 调试", [
        ParamDef("-v, --version", "显示版本信息"),
        ParamDef("-h, --help", "显示帮助信息"),
    ]),
]


# ============================================================================
# 示例定义
# ============================================================================
EXAMPLES = [
    ExampleDef("run", "D:\\Models\\Qwen2.5-7B", "--device cuda"),
    ExampleDef("run", "D:\\Models\\Qwen2.5-7B", "--lora ./lora"),
    ExampleDef("serve", "D:\\Models\\Qwen2.5-7B", "--port 8080 --batch 4"),
    ExampleDef("serve", "D:\\Models\\Qwen2.5-7B", "--api_key sk-xxx --dev_mode"),
    ExampleDef("webui", "D:\\Models\\Qwen2.5-7B", "--port 1616"),
    ExampleDef("download", "Qwen/Qwen2.5-7B-Instruct"),
]


# ============================================================================
# 模型格式定义
# ============================================================================
MODEL_FORMATS = [
    ModelFormatDef(".flm", "FastLLM 原生格式"),
    ModelFormatDef(".gguf", "GGUF 格式"),
    ModelFormatDef("HuggingFace 目录", "本地目录 (含 config.json)"),
    ModelFormatDef("HuggingFace Repo ID", "如 Qwen/Qwen2.5-7B (自动下载, 需 -py)"),
]


# ============================================================================
# 帮助文本生成工具
# ============================================================================
def get_commands_by_type(is_native: bool) -> List[CommandDef]:
    """按类型获取命令列表"""
    return [cmd for cmd in COMMANDS if cmd.is_native == is_native]


def find_command(name: str) -> Optional[CommandDef]:
    """根据名称或别名查找命令"""
    name_lower = name.lower()
    for cmd in COMMANDS:
        if cmd.name == name_lower or name_lower in cmd.aliases:
            return cmd
    return None


def get_param_group(title: str) -> Optional[ParamGroup]:
    """根据标题获取参数组"""
    for group in PARAM_GROUPS:
        if group.title == title:
            return group
    return None


def add_params_to_parser(parser: argparse.ArgumentParser, group_titles: Optional[List[str]] = None):
    """
    将参数组添加到 argparse 解析器
    
    Args:
        parser: argparse 解析器
        group_titles: 要添加的参数组标题列表，None 表示添加所有
    """
    for group in PARAM_GROUPS:
        if group_titles is not None and group.title not in group_titles:
            continue
        
        for param in group.params:
            if param.py_name is None:
                continue
            
            # 解析参数名
            names = []
            for part in param.name.split(','):
                part = part.strip()
                # 提取参数名，移除 <xxx> 部分
                if '<' in part:
                    part = part[:part.index('<')].strip()
                if part.startswith('-'):
                    names.append(part)
            
            if not names:
                continue
            
            # 构建 add_argument 参数
            kwargs = {'help': param.desc}
            
            if param.py_type == 'int':
                kwargs['type'] = int
                if param.default_val:
                    kwargs['default'] = int(param.default_val)
            elif param.py_type == 'bool':
                kwargs['action'] = 'store_true'
            elif param.py_type == 'str':
                kwargs['type'] = str
                if param.default_val:
                    kwargs['default'] = param.default_val
            
            try:
                parser.add_argument(*names, **kwargs)
            except argparse.ArgumentError:
                # 参数已存在，跳过
                pass


def print_help(use_color: bool = True):
    """打印统一帮助信息"""
    from . import console
    
    # 标题
    console.header(f"{PROGRAM_NAME} - {PROGRAM_DESC}")
    print()
    
    # 用法
    print(f"用法: {PROGRAM_NAME} <命令> [模型路径] [选项...]")
    print()
    
    # Native 命令
    console.info("C++ 原生命令:")
    for cmd in get_commands_by_type(True):
        aliases = f" ({', '.join(cmd.aliases)})" if cmd.aliases else ""
        print(f"  {cmd.name:12}{aliases:20} {cmd.desc}")
    print()
    
    # Python 命令
    console.info("Python 命令:")
    for cmd in get_commands_by_type(False):
        aliases = f" ({', '.join(cmd.aliases)})" if cmd.aliases else ""
        print(f"  {cmd.name:12}{aliases:20} {cmd.desc}")
    print()
    
    # 参数组
    for group in PARAM_GROUPS:
        console.info(group.title + ":")
        for param in group.params:
            print(f"  {param.name:28} {param.desc}")
        print()
    
    # 示例
    console.info("示例:")
    for ex in EXAMPLES:
        args = f" {ex.args}" if ex.args else ""
        print(f"  {PROGRAM_NAME} {ex.cmd} {ex.model}{args}")
    print()
    
    # 模型格式
    console.info("支持的模型格式:")
    for fmt in MODEL_FORMATS:
        print(f"  {fmt.format:24} {fmt.desc}")


# 测试
if __name__ == "__main__":
    print_help()
