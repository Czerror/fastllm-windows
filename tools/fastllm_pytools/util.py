import argparse
import os
import sys
from . import console

def format_device_map(device_map):
    """格式化设备映射为可读字符串，如 'cuda:1, cpu:4'"""
    if device_map is None:
        return None
    if isinstance(device_map, str):
        return f"{device_map}:1"
    elif isinstance(device_map, list):
        return ", ".join(f"{dev}:1" for dev in device_map)
    elif isinstance(device_map, dict):
        return ", ".join(f"{dev}:{cnt}" for dev, cnt in device_map.items())
    return str(device_map)


# ============================================================================
# 从 help_text 导入统一定义
# ============================================================================
from . import help_text as _ht

def make_normal_parser(des: str, add_help = True) -> argparse.ArgumentParser:
    """
    创建标准参数解析器，使用统一的帮助定义
    注意: 部分参数保留原有定义以确保兼容性
    """
    parser = argparse.ArgumentParser(description = des, add_help = add_help)
    
    # 位置参数 (保留原有定义)
    parser.add_argument('model', nargs='?', help = '模型路径，fastllm模型文件或HF模型文件夹或配置文件')
    
    # 基础参数 - 保留原有定义确保兼容性，但描述使用统一文本
    parser.add_argument('-p', '--path', type = str, required = False, default = '', help = '模型路径')
    parser.add_argument('-t', '--threads', type = int, default = -1,  help = 'CPU 线程数')
    parser.add_argument('-l', '--low', action = 'store_true', help = '低内存模式')
    parser.add_argument('--dtype', type = str, default = "auto", help = '权重类型 (float16, int8, int4, int4g)')
    parser.add_argument('--atype', type = str, default = "auto", help = '推理类型 (float32, float16)')
    parser.add_argument('--device', type = str, help = '使用的设备 (cuda, cpu, numa)')
    
    # MOE 参数
    parser.add_argument('--moe_dtype', type = str, default = "", help = 'MOE专家层数据类型')
    parser.add_argument('--moe_device', type = str, default = "", help = 'MOE专家层设备 (cuda, cpu)')
    parser.add_argument('--moe_experts', type = int, default = -1, help = '启用的MOE专家数量')
    
    # CUDA / 加速参数
    parser.add_argument('--cuda_embedding', action = 'store_true', help = '在CUDA上运行Embedding层')
    parser.add_argument("--cuda_shared_expert", "--cuda_se", type = str, default = "true", help = "CUDA共享专家优化(MOE)")
    parser.add_argument("--enable_amx", "--amx", type = str, default = "false", help = "启用Intel AMX加速")
    
    # 缓存参数
    parser.add_argument('--kv_cache_limit', type = str, default = "auto",  help = 'KV缓存限制(如8G, 4096M)')
    parser.add_argument('--max_batch', '--batch', type = int, default = -1,  help = '最大批处理数量')
    parser.add_argument("--cache_history", type = str, default = "", help = "启用历史缓存")
    parser.add_argument("--cache_fast", type = str, default = "", help = "启用快速缓存模式")
    parser.add_argument('--cache_dir', type = str, default = "", help = '缓存目录路径')
    
    # LoRA / 自定义参数
    parser.add_argument('--custom', type = str, default = "", help = '自定义模型配置')
    parser.add_argument('--lora', type = str, default = "", help = 'LoRA适配器路径')
    parser.add_argument('--dtype_config', type = str, default = "", help = '数据类型配置文件')
    parser.add_argument('--ori', type = str, default = "", help = '使用原始权重(禁用量化)')
    
    # 模板 / 工具调用
    parser.add_argument('--tool_call_parser', type = str, default = "auto", help = '工具调用解析器类型')
    parser.add_argument('--chat_template', type = str, default = "", help = '对话模板(覆盖自动检测)')
    parser.add_argument("--enable_thinking", type = str, default = "", help = "启用思考模式(<think>标签)")

    return parser


def add_server_args(parser):
    """添加服务器相关参数 - 与C++端保持一致"""
    parser.add_argument("--model_name", type = str, default = '', help = "模型显示名称(用于API返回)")
    parser.add_argument("--host", type = str, default="127.0.0.1", help = "监听地址(默认: 127.0.0.1)")
    parser.add_argument("--port", type = int, default = 8080, help = "监听端口(默认: 8080)")
    parser.add_argument("--api_key", type = str, default = "", help = "API密钥认证(Bearer Token)")
    parser.add_argument("--think", type = str, default = "false", help = "Python后端思考模式")
    parser.add_argument("--show_input", action = 'store_true', help = "显示输入消息(调试用)")
    parser.add_argument("--dev_mode", action = 'store_true', help = "开发模式(启用调试接口)")

def make_normal_llm_model(args):
    if (args.model and args.model != ''):
        if (args.model.endswith(".json") and os.path.exists(args.model)):
            import json
            with open(args.model, "r", encoding = "utf-8") as file:
                args_config = json.load(file)
                for it in args_config.keys():
                    if (it == "FASTLLM_USE_NUMA" or it == "FASTLLM_NUMA_THREADS"):
                        os.environ[it] = str(args_config[it])
                    setattr(args, it, args_config[it])
                
    usenuma = False
    try:
        env_FASTLLM_USE_NUMA = os.getenv("FASTLLM_USE_NUMA")
        if (env_FASTLLM_USE_NUMA and env_FASTLLM_USE_NUMA != '' and env_FASTLLM_USE_NUMA != "OFF" and env_FASTLLM_USE_NUMA != "0"):
            usenuma = True
    except:
        pass
    if (args.path == '' or args.path is None):
        args.path = args.model
    if (args.path == '' or args.path is None):
        print("model can't be empty. (Example: ftllm run MODELNAME)")
        exit(0)
    if not(os.path.exists(args.path)):
        if (hasattr(args, "model_name") and args.model_name == ''):
            args.model_name = args.path
            from ftllm.download import HFDNormalDownloader
            from ftllm.download import find_metadata
            from ftllm.download import search_model
        if (not(os.path.exists(get_fastllm_cache_path(args.path, args.cache_dir))) and not(find_metadata(args.path))):
            print("Can't find model \"" + args.path + "\", try to find similar one.")
            search_result = search_model(args.path)
            if (len(search_result) > 0):
                args.path = search_result[0]["id"]
                print("Replace model to \"" + args.path + "\"")
            else:
                exit(0)
        downloader = HFDNormalDownloader(args.path, local_dir = get_fastllm_cache_path(args.path, args.cache_dir))
        downloader.run()
        args.path = str(downloader.local_dir)
    
    config_path = os.path.join(args.path, "config.json")
    if (not(os.path.exists(config_path)) and args.ori != "" and os.path.exists(os.path.join(args.ori, "config.json"))):
        config_path = os.path.join(args.ori, "config.json")
    if (os.path.exists(config_path)):
        try:
            import json
            with open(config_path, "r", encoding="utf-8") as file:
                config = json.load(file)
            if (config["architectures"][0] == 'Qwen3ForCausalLM' or config["architectures"][0] == 'Qwen3MoeForCausalLM' or
                config["architectures"][0] == 'Glm4MoeForCausalLM'):
                if (args.enable_thinking == ""):
                    args.enable_thinking = "true"
            # MoE 模型自动配置 cache_history
            # 注意: device/moe_device 的自动配置已移至 C++ 底层 (basellm::ApplyAutoDeviceMap)
            if (config["architectures"][0] == 'DeepseekV3ForCausalLM' or 
                config["architectures"][0] == 'DeepseekV2ForCausalLM' or 
                config["architectures"][0] == 'Qwen3MoeForCausalLM' or 
                config["architectures"][0] == 'MiniMaxM1ForCausalLM' or 
                config["architectures"][0] == 'MiniMaxText01ForCausalLM' or 
                config["architectures"][0] == 'HunYuanMoEV1ForCausalLM' or 
                config["architectures"][0] == 'Ernie4_5_MoeForCausalLM' or 
                config["architectures"][0] == 'PanguProMoEForCausalLM' or
                config["architectures"][0] == 'Glm4MoeForCausalLM' or 
                config["architectures"][0] == 'Qwen3NextForCausalLM'):
                if (args.cache_history == ""):
                    args.cache_history = "true"
            if ("quantization_config" in config):
                quantization_config = config["quantization_config"]
                try:
                    if (args.dtype == "auto" and quantization_config['bits'] == 4 and quantization_config['group_size']):
                        args.dtype = "int4g" + str(quantization_config["group_size"])
                except:
                    pass
                try:
                    if (args.dtype == "auto" and quantization_config['quant_method'] == "fp8" and quantization_config['fmt'] == "e4m3"):
                        args.dtype = "fp8_e4m3"
                except:
                    pass
                try:
                    if (args.path.lower().find("-fp8") != -1):
                        args.dtype = "fp8_e4m3";
                except:
                    pass
        except:
            pass
    if ((args.device and args.device.find("numa") != -1) or args.moe_device.find("numa") != -1 or
        (args.device and args.device.find("tfacc") != -1) or args.moe_device.find("tfacc") != -1):
        os.environ["FASTLLM_ACTIVATE_NUMA"] = "ON"
        if (args.threads == -1):
            args.threads = 4
    if (args.threads == -1):
        try:
            available_cores = len(os.sched_getaffinity(0))  # 参数 0 表示当前进程
            args.threads = max(1, min(32, available_cores - 2))
        except:
            args.threads = max(1, min(32, os.cpu_count() - 2))
    if ("FT_THREADS" not in os.environ and "FASTLLM_NUMA_THREADS" not in os.environ):
        os.environ["FT_THREADS"] = str(args.threads)
    if (args.atype == "auto"):
        if (args.device in ["cpu", "numa", "tfacc"]):
            args.atype = "float32"
    if (args.dtype == "auto"):
        args.dtype = "float16"
    # 注意: moe_device 的默认值已移至 C++ 底层 (basellm::ApplyAutoDeviceMap)
    # 只有用户显式指定时才传递给底层
    from ftllm import llm
    
    # 用于存储解析后的设备映射（供显示用）
    args._parsed_device_map = None
    args._parsed_moe_device_map = None
    
    if (args.device and args.device != ""):
        try:
            import ast
            device_map = ast.literal_eval(args.device)
            if (isinstance(device_map, list) or isinstance(device_map, dict)):
                llm.set_device_map(device_map)
                args._parsed_device_map = device_map
            else:
                llm.set_device_map(args.device)
                args._parsed_device_map = args.device
        except:
            llm.set_device_map(args.device)
            args._parsed_device_map = args.device
    if (args.moe_device and args.moe_device != ""):
        try:
            import ast
            moe_device_map = ast.literal_eval(args.moe_device)
            if (isinstance(moe_device_map, list) or isinstance(moe_device_map, dict)):
                llm.set_device_map(moe_device_map, True)
                args._parsed_moe_device_map = moe_device_map
            else:
                llm.set_device_map(args.moe_device, True)
                args._parsed_moe_device_map = args.moe_device
        except:
            llm.set_device_map(args.moe_device, True)
            args._parsed_moe_device_map = args.moe_device
    llm.set_cpu_threads(args.threads)
    llm.set_cpu_low_mem(args.low)
    if (args.cuda_embedding):
        llm.set_cuda_embedding(True)
    if (args.cuda_shared_expert.lower() not in ["", "false", "0", "off"]):
        llm.set_cuda_shared_expert(True)
    if (args.enable_amx.lower() not in ["", "false", "0", "off"]):
        llm.set_enable_amx(True)
    graph = None
    if (args.custom != ""):
        import importlib.util
        spec = importlib.util.spec_from_file_location("custom_module", args.custom)
        if spec is None:
            raise ImportError(f"Cannot load module at {args.custom}")
        custom_module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(custom_module)
        if (hasattr(custom_module, "__model__")):
            graph = getattr(custom_module, "__model__")
    if (args.dtype_config != "" and os.path.exists(args.dtype_config)):
        with open(args.dtype_config, "r", encoding="utf-8") as file:
            args.dtype_config = file.read()
    if (args.chat_template != "" and os.path.exists(args.chat_template)):
        with open(args.chat_template, "r", encoding="utf-8") as file:
            args.chat_template = file.read()
    
    # 使用 Spinner 显示模型加载进度
    with console.spinner("加载模型", "模型加载完成"):
        model = llm.model(args.path, dtype = args.dtype, moe_dtype = args.moe_dtype, graph = graph, tokenizer_type = "auto", lora = args.lora, 
                            dtype_config = args.dtype_config, ori_model_path = args.ori, chat_template = args.chat_template, tool_call_parser = args.tool_call_parser)
    
    if (args.enable_thinking.lower() in ["", "false", "0", "off"]):
        model.enable_thinking = False
    model.set_atype(args.atype)
    if (args.cache_history.lower() not in ["", "false", "0", "off"]):
        model.set_save_history(True)
        if (args.cache_fast in ["", "false", "0", "off"]):
            llm.set_cpu_historycache(True)
    if (args.moe_experts > 0):
        model.set_moe_experts(args.moe_experts)
    if (args.max_batch > 0):
        model.set_max_batch(args.max_batch)
    if (args.kv_cache_limit != "" and args.kv_cache_limit != "auto"):
        model.set_kv_cache_limit(args.kv_cache_limit)
    return model

def make_download_parser(add_help = True):
    parser = argparse.ArgumentParser(
            description="Downloads a model or dataset from Hugging Face",
            usage="ftllm download [REPO_ID] [OPTIONS]",
            add_help = add_help
    )
        
    # 位置参数
    parser.add_argument("repo_id", nargs="?", help="Hugging Face repo ID")
    # 选项参数
    parser.add_argument("--include", nargs="+", default=[], help="Include patterns")
    parser.add_argument("--exclude", nargs="+", default=[], help="Exclude patterns")
    parser.add_argument("--hf_username", help="HF username")
    parser.add_argument("--hf_token", help="HF access token")
    parser.add_argument("--tool", choices=["aria2c", "wget"], default="aria2c", help="Download tool")
    parser.add_argument("-x", type=int, default=4, help="Threads for aria2c")
    parser.add_argument("-j", type=int, default=5, help="Concurrent downloads")
    parser.add_argument("--dataset", action="store_true", help="Download dataset")
    parser.add_argument("--local-dir", help="Local directory path")
    parser.add_argument("--revision", default="main", help="Revision to download")
    #parser.add_argument("-h", "--help", action="store_true", help="Show help")
        
    return parser

def get_fastllm_cache_path(model_name: str, cache_path = ""):
    """获取模型缓存路径
    
    优先级:
    1. 用户指定的 cache_path 参数
    2. 环境变量 FASTLLM_CACHEDIR
    3. 默认: ftllm 程序目录下的 model 文件夹
    """
    if cache_path == "":
        # 优先使用环境变量
        cache_dir = os.getenv("FASTLLM_CACHEDIR")
        if cache_dir and os.path.isdir(cache_dir):
            cache_path = cache_dir
        else:
            # 默认: ftllm 程序所在目录下的 model 文件夹
            # __file__ 是当前模块路径，向上一级即为 ftllm 目录
            ftllm_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
            cache_path = os.path.join(ftllm_dir, "models")
            
            # 确保目录存在
            if not os.path.exists(cache_path):
                os.makedirs(cache_path, exist_ok=True)

    cache_path = os.path.join(cache_path, model_name)
    return cache_path