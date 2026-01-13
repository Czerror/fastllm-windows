//
// help_text.h - 统一帮助文本定义
// 提供跨 C++ 和 Python 的统一帮助信息
//

#ifndef FASTLLM_HELP_TEXT_H
#define FASTLLM_HELP_TEXT_H

#include <string>
#include <vector>
#include <initializer_list>

namespace fastllm {
namespace help {

// ============================================================================
// 程序信息
// ============================================================================
static constexpr const char* PROGRAM_NAME = "ftllm";
static constexpr const char* PROGRAM_DESC = "FastLLM 统一命令行工具";
static constexpr const char* PROGRAM_VERSION = "1.0";

// ============================================================================
// 参数定义结构
// ============================================================================
struct ParamDef {
    const char* name;       // 参数名 (如 "-p, --path <路径>")
    const char* desc;       // 描述
    const char* py_name;    // Python argparse 名称 (可为 nullptr)
    const char* py_type;    // Python 类型 (可为 nullptr)
    const char* default_val;// 默认值 (可为 nullptr)
};

struct ParamGroup {
    const char* title;
    std::initializer_list<ParamDef> params;
};

struct CommandDef {
    const char* name;
    const char* aliases[4];
    const char* exe;        // Native 后端的可执行文件名 (Python 为 nullptr)
    const char* desc;
    bool is_native;         // true: C++ Native, false: Python
};

struct ExampleDef {
    const char* cmd;
    const char* model;
    const char* args;       // 可为 nullptr
};

struct ModelFormatDef {
    const char* format;
    const char* desc;
};

// ============================================================================
// 命令定义
// ============================================================================
static const CommandDef COMMANDS[] = {
    // Native (C++) 命令
    {"serve",    {"server", "api", nullptr, nullptr},   "apiserver.exe", "启动 OpenAI 兼容 API 服务器", true},
    {"webui",    {"web", nullptr, nullptr, nullptr},    "webui.exe",     "启动 Web 界面",               true},
    {"bench",    {"benchmark", nullptr, nullptr, nullptr}, "benchmark.exe", "性能测试",                 true},
    {"quant",    {"quantize", nullptr, nullptr, nullptr},  "quant.exe",    "模型量化",                  true},
    // Python 命令
    {"run",      {"chat", nullptr, nullptr, nullptr},   nullptr, "交互式聊天",            false},
    {"download", {nullptr, nullptr, nullptr, nullptr},  nullptr, "下载 HuggingFace 模型", false},
    {"ui",       {nullptr, nullptr, nullptr, nullptr},  nullptr, "启动图形界面",          false},
    {"config",   {nullptr, nullptr, nullptr, nullptr},  nullptr, "生成配置文件模板",      false},
    {"export",   {nullptr, nullptr, nullptr, nullptr},  nullptr, "导出模型",              false},
};
static constexpr size_t NUM_COMMANDS = sizeof(COMMANDS) / sizeof(COMMANDS[0]);

// ============================================================================
// 参数组定义
// ============================================================================
static const ParamGroup PARAM_GROUPS[] = {
    {"基础参数", {
        {"-p, --path <路径>",     "模型路径",                        "path",       "str",  nullptr},
        {"--device <设备>",       "cuda, cpu, numa",                 "device",     "str",  nullptr},
        {"--dtype <类型>",        "float16, int8, int4, int4g",      "dtype",      "str",  "auto"},
        {"-t, --threads <数量>",  "CPU 线程数",                      "threads",    "int",  "-1"},
        {"--model_name <名称>",   "模型显示名称 (用于 API 返回)",    "model_name", "str",  nullptr},
    }},
    {"服务器参数", {
        {"--host <地址>",         "监听地址 (默认: 127.0.0.1)",      "host",       "str",  "127.0.0.1"},
        {"--port <端口>",         "监听端口 (默认: 8080)",           "port",       "int",  "8080"},
        {"--api_key <密钥>",      "API 密钥认证 (Bearer Token)",     "api_key",    "str",  nullptr},
        {"--embedding_path <路径>", "Embedding 模型路径",            "embedding_path", "str", nullptr},
        {"--dev_mode",            "开发模式 (启用调试接口)",         "dev_mode",   "bool", nullptr},
    }},
    {"Batch / 并发参数", {
        {"--batch <数量>",        "批处理大小",                      "max_batch",  "int",  "-1"},
        {"--max_batch <数量>",    "最大批处理数量",                  "max_batch",  "int",  "-1"},
        {"--max_token <数量>",    "最大生成 Token 数 (webui)",       "max_token",  "int",  "4096"},
        {"--chunk_size <数量>",   "Chunked Prefill 分块大小",        "chunk_size", "int",  nullptr},
    }},
    {"CUDA / 加速参数", {
        {"--cuda_embedding",      "在 CUDA 上运行 Embedding 层",     "cuda_embedding", "bool", nullptr},
        {"--cuda_shared_expert",  "CUDA 共享专家优化 (MOE)",         "cuda_shared_expert", "str", "true"},
        {"--cuda_se",             "--cuda_shared_expert 简写",       "cuda_se",    "str",  "true"},
        {"--enable_amx, --amx",   "启用 Intel AMX 加速",             "enable_amx", "str",  "false"},
    }},
    {"MOE (混合专家) 参数", {
        {"--moe_device <设备>",   "MOE 专家层设备 (cuda, cpu)",      "moe_device", "str",  nullptr},
        {"--moe_dtype <类型>",    "MOE 专家层数据类型",              "moe_dtype",  "str",  nullptr},
        {"--moe_experts <数量>",  "启用的 MOE 专家数量",             "moe_experts", "int", "-1"},
    }},
    {"缓存参数", {
        {"--kv_cache_limit <大小>", "KV 缓存限制 (如 8G, 4096M)",    "kv_cache_limit", "str", "auto"},
        {"--cache_history",       "启用历史缓存",                    "cache_history", "str", nullptr},
        {"--cache_fast",          "启用快速缓存模式",                "cache_fast", "str",  nullptr},
        {"--cache_dir <路径>",    "缓存目录路径",                    "cache_dir",  "str",  nullptr},
    }},
    {"LoRA 参数", {
        {"--lora <路径>",         "LoRA 适配器路径",                 "lora",       "str",  nullptr},
        {"--custom <配置>",       "自定义模型配置",                  "custom",     "str",  nullptr},
        {"--dtype_config <配置>", "数据类型配置文件",                "dtype_config", "str", nullptr},
        {"--ori",                 "使用原始权重 (禁用量化)",         "ori",        "str",  nullptr},
    }},
    {"模板 / 工具调用", {
        {"--chat_template <模板>", "对话模板 (覆盖自动检测)",        "chat_template", "str", nullptr},
        {"--tool_call_parser <类型>", "工具调用解析器类型",          "tool_call_parser", "str", "auto"},
        {"--enable_thinking",     "启用思考模式 (<think>标签)",      "enable_thinking", "str", nullptr},
        {"--think",               "Python 后端思考模式",             "think",      "str",  "false"},
        {"--hide_input",          "隐藏输入内容 (隐私保护)",         "hide_input", "bool", nullptr},
    }},
    {"开发 / 调试", {
        {"-v, --version",         "显示版本信息",                    nullptr,      nullptr, nullptr},
        {"-h, --help",            "显示帮助信息",                    nullptr,      nullptr, nullptr},
    }},
};
static constexpr size_t NUM_PARAM_GROUPS = sizeof(PARAM_GROUPS) / sizeof(PARAM_GROUPS[0]);

// ============================================================================
// 示例定义
// ============================================================================
static const ExampleDef EXAMPLES[] = {
    {"run",      "D:\\Models\\Qwen2.5-7B", "--device cuda"},
    {"run",      "D:\\Models\\Qwen2.5-7B", "--lora ./lora"},
    {"serve",    "D:\\Models\\Qwen2.5-7B", "--port 8080 --batch 4"},
    {"serve",    "D:\\Models\\Qwen2.5-7B", "--api_key sk-xxx --dev_mode"},
    {"webui",    "D:\\Models\\Qwen2.5-7B", "--port 1616"},
    {"download", "Qwen/Qwen2.5-7B-Instruct", nullptr},
};
static constexpr size_t NUM_EXAMPLES = sizeof(EXAMPLES) / sizeof(EXAMPLES[0]);

// ============================================================================
// 模型格式定义
// ============================================================================
static const ModelFormatDef MODEL_FORMATS[] = {
    {".flm",                "FastLLM 原生格式"},
    {".gguf",               "GGUF 格式"},
    {"HuggingFace 目录",    "本地目录 (含 config.json)"},
    {"HuggingFace Repo ID", "如 Qwen/Qwen2.5-7B (自动下载, 需 -py)"},
};
static constexpr size_t NUM_MODEL_FORMATS = sizeof(MODEL_FORMATS) / sizeof(MODEL_FORMATS[0]);

} // namespace help
} // namespace fastllm

#endif // FASTLLM_HELP_TEXT_H
