/**
 * ftllm - FastLLM 统一命令行入口
 * 
 * 默认使用 C++ 原生程序 (apiserver/webui/benchmark 等)
 * 添加 -py 参数时使用 Python 后端
 * 添加到 PATH 后可直接使用: ftllm run/serve/webui 等
 */

#include <iostream>
#include <string>
#include <vector>
#include <algorithm>
#include <cstdlib>
#ifdef _WIN32
#include <windows.h>
#else
#include <unistd.h>
#include <sys/stat.h>
#endif

const char* FTLLM_VERSION = "1.0.1";

// ============================================================================
// 初始化
// ============================================================================

#ifdef _WIN32
/**
 * 初始化 Windows 控制台以支持 UTF-8 输出
 */
void initWindowsConsole() {
    // 设置控制台输出代码页为 UTF-8
    SetConsoleOutputCP(CP_UTF8);
    SetConsoleCP(CP_UTF8);
    
    // 启用 ANSI 转义序列（用于彩色输出）
    HANDLE hOut = GetStdHandle(STD_OUTPUT_HANDLE);
    if (hOut != INVALID_HANDLE_VALUE) {
        DWORD mode = 0;
        if (GetConsoleMode(hOut, &mode)) {
            SetConsoleMode(hOut, mode | ENABLE_VIRTUAL_TERMINAL_PROCESSING);
        }
    }
}
#endif

// ============================================================================
// 工具函数
// ============================================================================

std::string getExeDirectory() {
#ifdef _WIN32
    char path[MAX_PATH];
    GetModuleFileNameA(NULL, path, MAX_PATH);
    std::string fullPath(path);
    size_t pos = fullPath.find_last_of("\\/");
    return (pos != std::string::npos) ? fullPath.substr(0, pos) : ".";
#else
    char path[1024];
    ssize_t len = readlink("/proc/self/exe", path, sizeof(path) - 1);
    if (len != -1) {
        path[len] = '\0';
        std::string fullPath(path);
        size_t pos = fullPath.find_last_of("/");
        return (pos != std::string::npos) ? fullPath.substr(0, pos) : ".";
    }
    return ".";
#endif
}

bool fileExists(const std::string& path) {
#ifdef _WIN32
    return GetFileAttributesA(path.c_str()) != INVALID_FILE_ATTRIBUTES;
#else
    struct stat st;
    return stat(path.c_str(), &st) == 0;
#endif
}

bool isDirectory(const std::string& path) {
#ifdef _WIN32
    DWORD attr = GetFileAttributesA(path.c_str());
    return (attr != INVALID_FILE_ATTRIBUTES) && (attr & FILE_ATTRIBUTE_DIRECTORY);
#else
    struct stat st;
    return (stat(path.c_str(), &st) == 0) && S_ISDIR(st.st_mode);
#endif
}

// ============================================================================
// LoRA 自动检测
// ============================================================================

/**
 * 检测模型目录下是否存在 LoRA 配置
 * 支持的 LoRA 格式:
 *   - lora/ 子目录 (含 adapter_config.json)
 *   - adapter_config.json (PEFT/HuggingFace 格式)
 *   - *.lora 文件
 */
bool detectLoraInModelPath(const std::string& modelPath) {
    if (modelPath.empty()) return false;
    
    // 检查是否为目录
    if (!isDirectory(modelPath)) return false;
    
    // 检查常见 LoRA 文件/目录
    std::vector<std::string> loraIndicators = {
        modelPath + "\\lora\\adapter_config.json",
        modelPath + "/lora/adapter_config.json",
        modelPath + "\\adapter_config.json",
        modelPath + "/adapter_config.json",
        modelPath + "\\lora",
        modelPath + "/lora",
    };
    
    for (const auto& indicator : loraIndicators) {
        if (fileExists(indicator)) {
            return true;
        }
    }
    
    return false;
}

/**
 * 从命令行参数中提取模型路径
 */
std::string extractModelPath(int argc, char** argv) {
    for (int i = 1; i < argc; i++) {
        std::string arg = argv[i];
        
        // -p / --path 参数
        if ((arg == "-p" || arg == "--path") && i + 1 < argc) {
            return argv[i + 1];
        }
        
        // 跳过已知选项
        if (arg[0] == '-') {
            // 带值的选项，跳过下一个参数
            if (arg == "-t" || arg == "--threads" || arg == "--device" || 
                arg == "--dtype" || arg == "--host" || arg == "--port" ||
                arg == "--lora" || arg == "--system" || arg == "--api_key" ||
                arg == "--batch" || arg == "--model_name" || arg == "--atype" ||
                arg == "--moe_device" || arg == "--moe_dtype" || arg == "--moe_experts" ||
                arg == "--kv_cache_limit" || arg == "--max_batch" || arg == "--max_token" ||
                arg == "--cache_dir" || arg == "--ori" || arg == "--custom" ||
                arg == "--dtype_config" || arg == "--chat_template" || arg == "--tool_call_parser" ||
                arg == "--top_p" || arg == "--top_k" || arg == "--temperature" || arg == "--repeat_penalty" ||
                arg == "-o" || arg == "--output") {
                i++;
            }
            continue;
        }
        
        // 跳过命令名 (chat, serve, webui 等)
        static const char* commands[] = {"chat", "run", "serve", "server", "api", 
                                          "webui", "web", "bench", "benchmark", 
                                          "quant", "quantize", "download", "ui", 
                                          "config", "export", nullptr};
        bool isCommand = false;
        for (int j = 0; commands[j]; j++) {
            if (arg == commands[j]) {
                isCommand = true;
                break;
            }
        }
        if (isCommand) continue;
        
        // 可能是模型路径
        if (isDirectory(arg) || fileExists(arg)) {
            return arg;
        }
    }
    return "";
}

// ============================================================================
// C++ 原生程序执行
// ============================================================================

// 命令到原生 exe 的映射
struct CommandMapping {
    const char* command;
    const char* aliases[3];
    const char* nativeExe;
    const char* description;
};

static const CommandMapping COMMANDS[] = {
    {"serve",    {"server", "api", nullptr},    "apiserver.exe",          "OpenAI API 服务器"},
    {"webui",    {"web", nullptr, nullptr},     "webui.exe",              "Web 聊天界面"},
    {"chat",     {"run", nullptr, nullptr},     "FastllmStudio_cli.exe",  "命令行聊天"},
    {"bench",    {"benchmark", nullptr, nullptr}, "benchmark.exe",        "性能测试"},
    {"quant",    {"quantize", nullptr, nullptr}, "quant.exe",             "模型量化"},
};

const CommandMapping* findCommand(const std::string& cmd) {
    for (const auto& mapping : COMMANDS) {
        if (cmd == mapping.command) return &mapping;
        for (int i = 0; mapping.aliases[i]; i++) {
            if (cmd == mapping.aliases[i]) return &mapping;
        }
    }
    return nullptr;
}

int executeNativeProgram(const std::string& exeName, int argc, char** argv, int startArg) {
    std::string exeDir = getExeDirectory();
    
    // 搜索路径: 同级目录 + 常见子文件夹
    std::vector<std::string> searchPaths = {
        exeDir + "\\" + exeName,
        exeDir + "\\bin\\" + exeName,
    };
    
    std::string exePath;
    for (const auto& path : searchPaths) {
        if (fileExists(path)) {
            exePath = path;
            break;
        }
    }
    
    if (exePath.empty()) {
        std::cerr << "[错误] 找不到原生程序: " << exeName << std::endl;
        return 1;
    }
    
    // 构建命令行
#ifdef _WIN32
    // chcp 65001 确保子进程使用 UTF-8 代码页，避免中文乱码
    std::string cmd = "chcp 65001 >nul && \"" + exePath + "\"";
#else
    std::string cmd = "\"" + exePath + "\"";
#endif
    
    // 检查是否已有 -p/--path 参数
    bool hasPathArg = false;
    std::string positionalModelPath;
    
    for (int i = startArg; i < argc; i++) {
        std::string arg = argv[i];
        if (arg == "-p" || arg == "--path") {
            hasPathArg = true;
            break;
        }
    }
    
    // 处理参数
    for (int i = startArg; i < argc; i++) {
        std::string arg = argv[i];
        if (arg == "-py") continue;  // 跳过 -py 参数
        
        // 检测第一个位置参数（模型路径）
        if (!hasPathArg && positionalModelPath.empty() && arg[0] != '-') {
            // 检查是否像模型路径
            if (isDirectory(arg) || fileExists(arg) || 
                arg.find(':') != std::string::npos ||
                arg.find('/') != std::string::npos ||
                arg.find('\\') != std::string::npos) {
                positionalModelPath = arg;
                // 为原生程序添加 -p 参数
                cmd += " -p";
            }
        }
        
        bool needQuote = (arg.find(' ') != std::string::npos || 
                          arg.find('"') != std::string::npos ||
                          arg.find('&') != std::string::npos ||
                          arg.find('{') != std::string::npos ||
                          arg.find('}') != std::string::npos);
        if (needQuote) {
            std::string escaped;
            for (char c : arg) {
                if (c == '"') escaped += "\\\"";
                else escaped += c;
            }
            cmd += " \"" + escaped + "\"";
        } else {
            cmd += " " + arg;
        }
    }
    
    return system(cmd.c_str());
}

// ============================================================================
// Python 后端执行
// ============================================================================

std::string buildPythonCommand(int argc, char** argv, int startArg = 1) {
    std::string exeDir = getExeDirectory();
    std::string ftllmPath = exeDir + "/ftllm";
    
    std::string cmd;
    if (fileExists(ftllmPath + "/__init__.py")) {
#ifdef _WIN32
        // chcp 65001 确保子进程使用 UTF-8 代码页，避免中文乱码
        cmd = "chcp 65001 >nul && cd /d \"" + exeDir + "\" && python -m ftllm";
#else
        cmd = "cd \"" + exeDir + "\" && python -m ftllm";
#endif
    } else {
        // 安装版模式：使用 python -m ftllm
#ifdef _WIN32
        cmd = "chcp 65001 >nul && python -m ftllm";
#else
        cmd = "python -m ftllm";
#endif
    }
    
    for (int i = startArg; i < argc; i++) {
        std::string arg = argv[i];
        if (arg == "-py") continue;  // 跳过 -py 参数
        
        // 处理包含空格或特殊字符的参数
        bool needQuote = (arg.find(' ') != std::string::npos || 
                          arg.find('"') != std::string::npos ||
                          arg.find('&') != std::string::npos ||
                          arg.find('{') != std::string::npos ||
                          arg.find('}') != std::string::npos);
        if (needQuote) {
            std::string escaped;
            for (char c : arg) {
                if (c == '"') escaped += "\\\"";
                else escaped += c;
            }
            cmd += " \"" + escaped + "\"";
        } else {
            cmd += " " + arg;
        }
    }
    return cmd;
}

int executePythonBackend(int argc, char** argv, int startArg = 1) {
    std::string cmd = buildPythonCommand(argc, argv, startArg);
    return system(cmd.c_str());
}

// ============================================================================
// 帮助信息
// ============================================================================

void Usage() {
    std::cout << "Usage: ftllm <command> [options] [model_path]" << std::endl;
    std::cout << std::endl;
    std::cout << "FastLLM - 高性能大语言模型推理引擎 (v" << FTLLM_VERSION << ")" << std::endl;
    std::cout << std::endl;
    std::cout << "命令 (默认使用 C++ 原生程序):" << std::endl;
    std::cout << "  run, chat                     交互式聊天 (FastllmStudio_cli)" << std::endl;
    std::cout << "  serve, server                 启动 OpenAI API 服务器 (apiserver)" << std::endl;
    std::cout << "  webui                         启动 Web UI (webui)" << std::endl;
    std::cout << "  bench, benchmark              性能测试 (benchmark)" << std::endl;
    std::cout << "  quant, quantize               模型量化 (quant)" << std::endl;
    std::cout << std::endl;
    std::cout << "Python 专用命令 (需要 -py 或自动使用 Python):" << std::endl;
    std::cout << "  download <model>              下载 HuggingFace 模型" << std::endl;
    std::cout << "  ui                            启动图形界面" << std::endl;
    std::cout << "  config [file]                 生成配置文件模板" << std::endl;
    std::cout << "  export -o <path>              导出模型" << std::endl;
    std::cout << std::endl;
    std::cout << "模式切换:" << std::endl;
    std::cout << "  -py                           使用 Python 后端 (支持 LoRA 动态加载等)" << std::endl;
    std::cout << "  (自动)                        检测到模型目录含 lora/ 时自动切换 Python" << std::endl;
    std::cout << std::endl;
    std::cout << "快速开始:" << std::endl;
    std::cout << "  ftllm run <model_path>                   交互式聊天 (C++)" << std::endl;
    std::cout << "  ftllm run -py <model_path>               交互式聊天 (Python, 支持 LoRA)" << std::endl;
    std::cout << "  ftllm serve <model_path> --port 8080     启动 API 服务器" << std::endl;
    std::cout << std::endl;
    std::cout << "模型格式 (自动识别):" << std::endl;
    std::cout << "  .flm                          FastLLM 原生格式" << std::endl;
    std::cout << "  .gguf                         GGUF 格式" << std::endl;
    std::cout << "  HuggingFace 目录              本地目录 (含 config.json)" << std::endl;
    std::cout << "  HuggingFace Repo ID           如 Qwen/Qwen2.5-7B (自动下载, 需 -py)" << std::endl;
    std::cout << std::endl;
    std::cout << "常用参数:" << std::endl;
    std::cout << "  -p, --path <路径>             模型路径" << std::endl;
    std::cout << "  --device <设备>               cuda, cpu, numa" << std::endl;
    std::cout << "  --dtype <类型>                float16, int8, int4" << std::endl;
    std::cout << "  -t, --threads <数量>          CPU 线程数" << std::endl;
    std::cout << "  --lora <路径>                 LoRA 路径 (自动切换 Python)" << std::endl;
    std::cout << std::endl;
    std::cout << "服务器参数:" << std::endl;
    std::cout << "  --host <地址>                 监听地址 (默认: 0.0.0.0)" << std::endl;
    std::cout << "  --port <端口>                 监听端口 (默认: 8080)" << std::endl;
    std::cout << "  --api_key <密钥>              API 密钥" << std::endl;
    std::cout << std::endl;
    std::cout << "示例:" << std::endl;
    std::cout << "  ftllm run D:\\Models\\Qwen2.5-7B --device cuda" << std::endl;
    std::cout << "  ftllm run -py D:\\Models\\Qwen2.5-7B --lora ./lora" << std::endl;
    std::cout << "  ftllm serve D:\\Models\\Qwen2.5-7B --port 8080" << std::endl;
    std::cout << "  ftllm webui D:\\Models\\Qwen2.5-7B --port 1616" << std::endl;
    std::cout << "  ftllm -py download Qwen/Qwen2.5-7B-Instruct" << std::endl;
    std::cout << std::endl;
    std::cout << "详细帮助: ftllm <command> --help" << std::endl;
}

// ============================================================================
// 主函数
// ============================================================================

int main(int argc, char **argv) {
#ifdef _WIN32
    initWindowsConsole();
#endif

    // 无参数时显示帮助
    if (argc == 1) {
        Usage();
        return 0;
    }

    // 检测 -py 参数
    bool usePython = false;
    for (int i = 1; i < argc; i++) {
        if (std::string(argv[i]) == "-py") {
            usePython = true;
            break;
        }
    }

    // 处理特殊参数
    for (int i = 1; i < argc; i++) {
        std::string arg = argv[i];
        if (arg == "-py") continue;
        
        if (arg == "-h" || arg == "--help") {
            // 无子命令时显示本地帮助
            if (argc == 2 || (argc == 3 && usePython)) {
                Usage();
                return 0;
            }
            // 有子命令时继续执行
            break;
        }
        if (arg == "-v" || arg == "--version") {
            std::cout << "ftllm version " << FTLLM_VERSION << std::endl;
            return 0;
        }
    }
    
    // 获取第一个非 -py 的参数作为命令
    std::string command;
    int commandIndex = 0;
    for (int i = 1; i < argc; i++) {
        std::string arg = argv[i];
        if (arg == "-py") continue;
        if (arg[0] != '-') {
            command = arg;
            commandIndex = i;
            break;
        }
    }
    
    // Python 专用命令 (无论是否指定 -py 都使用 Python)
    static const char* pythonOnlyCommands[] = {"download", "ui", "config", "export", nullptr};
    for (int i = 0; pythonOnlyCommands[i]; i++) {
        if (command == pythonOnlyCommands[i]) {
            usePython = true;
            break;
        }
    }
    
    // 自动检测模型目录下的 LoRA 配置
    if (!usePython) {
        // 检查是否有显式 --lora 参数
        for (int i = 1; i < argc; i++) {
            if (std::string(argv[i]) == "--lora") {
                std::cout << "[自动检测] 指定 --lora 参数，切换到 Python 后端" << std::endl;
                usePython = true;
                break;
            }
        }
    }
    
    // 检测 Python 后端专用参数 (原生程序不支持)
    // 完整对比见: C++ apiserver/webui/benchmark vs Python util.py/cli.py
    std::vector<std::string> unsupportedArgs;
    if (!usePython) {
        static const char* pythonOnlyArgs[] = {
            // ===== MOE / 设备配置 =====
            "--host",               // apiserver 不支持 host 绑定
            "--moe_device",         // MOE 设备配置 (如 "{'cuda':1,'cpu':4}")
            "--moe_dtype",          // MOE 权重类型
            "--moe_experts",        // MOE 专家数
            
            // ===== CUDA 配置 =====
            "--cuda_embedding",     // CUDA embedding
            "--kv_cache_limit",     // KV 缓存最大使用量
            
            // ===== 缓存 / 内存配置 =====
            "--max_batch",          // 最大批次 (Python 用 max_batch, 原生用 batch)
            "--cache_history",      // 缓存历史对话
            "--cache_fast",         // 快速缓存（消耗显存）
            "--cache_dir",          // 模型缓存目录
            
            // ===== 模型特性 =====
            "--enable_thinking",    // 硬思考开关（需要模型支持）
            "--cuda_shared_expert", // CUDA 共享专家 (--cuda_se)
            "--cuda_se",            // 同上的别名
            "--enable_amx",         // AMX 加速
            "--amx",                // 同上的别名
            "--flash_attn",         // Flash Attention
            
            // ===== LoRA / 自定义 =====
            "--lora",               // LoRA 路径
            "--adapter",            // LoRA adapter (别名)
            "--custom",             // 自定义模型 python 文件
            "--dtype_config",       // 权重类型配置文件
            "--ori",                // 原始模型权重（GGUF）
            
            // ===== 模板 / 解析 =====
            "--tool_call_parser",   // tool_call 解析器
            "--chat_template",      // 聊天模板文件
            
            // ===== Server 专用 =====
            "--model_name",         // 模型名称
            "--api_key",            // API 密钥
            "--think",              // 处理 <think> 标签
            "--hide_input",         // 不显示请求信息
            "--dev_mode",           // 开发模式
            
            // ===== 采样参数 (部分原生支持) =====
            "--top_p",              // 采样参数
            "--top_k",              // 采样参数
            "--temperature",        // 温度参数
            "--repeat_penalty",     // 重复惩罚
            
            nullptr
        };
        
        for (int i = 1; i < argc; i++) {
            std::string arg = argv[i];
            for (int j = 0; pythonOnlyArgs[j]; j++) {
                if (arg == pythonOnlyArgs[j]) {
                    unsupportedArgs.push_back(arg);
                    break;
                }
            }
        }
        
        if (!unsupportedArgs.empty()) {
            std::cout << "[提示] 以下参数在 C++ 原生程序中不支持:" << std::endl;
            for (const auto& arg : unsupportedArgs) {
                std::cout << "  - " << arg << std::endl;
            }
            std::cout << "[自动切换] 使用 Python 后端以支持这些参数" << std::endl;
            usePython = true;
        }
    }
    
    if (!usePython) {
        std::string modelPath = extractModelPath(argc, argv);
        if (detectLoraInModelPath(modelPath)) {
            std::cout << "[自动检测] 发现 LoRA 配置，切换到 Python 后端" << std::endl;
            usePython = true;
        }
    }
    
    // 如果使用 Python 后端
    if (usePython) {
        return executePythonBackend(argc, argv);
    }
    
    // 检查是否以 - 开头但不是已知选项（可能是拼写错误）
    if (!command.empty() && command[0] == '-') {
        std::cerr << "[错误] 未知选项 '" << command << "'" << std::endl;
        std::cerr << std::endl;
        std::cerr << "常用选项:" << std::endl;
        std::cerr << "  -py              使用 Python 后端" << std::endl;
        std::cerr << "  -p, --path       模型路径" << std::endl;
        std::cerr << "  -t, --threads    线程数" << std::endl;
        std::cerr << "  -h, --help       显示帮助" << std::endl;
        std::cerr << "  -v, --version    显示版本" << std::endl;
        std::cerr << std::endl;
        std::cerr << "使用 'ftllm --help' 查看完整帮助" << std::endl;
        return 1;
    }
    
    // 查找命令映射
    const CommandMapping* mapping = findCommand(command);
    if (mapping) {
        // 执行 C++ 原生程序
        return executeNativeProgram(mapping->nativeExe, argc, argv, commandIndex + 1);
    }
    
    // 空命令时显示帮助
    if (command.empty()) {
        Usage();
        return 0;
    }
    
    // 检查是否像模型路径（包含路径分隔符或以常见模型扩展名结尾）
    bool looksLikeModelPath = (command.find('/') != std::string::npos ||
                               command.find('\\') != std::string::npos ||
                               command.find(':') != std::string::npos ||
                               command.ends_with(".flm") ||
                               command.ends_with(".gguf"));
    
    if (looksLikeModelPath) {
        // 用户可能直接输入了模型路径，提示正确用法
        std::cout << "[提示] 检测到模型路径，尝试启动聊天..." << std::endl;
        // 构造新的参数: ftllm chat <modelPath> [其他参数]
        std::string exeDir = getExeDirectory();
        std::string cliExe;
        
        // 搜索路径: 同级目录 + 常见子文件夹
        std::vector<std::string> cliPaths = {
            exeDir + "\\FastllmStudio_cli.exe",
            exeDir + "\\bin\\FastllmStudio_cli.exe",
        };
        for (const auto& p : cliPaths) {
            if (fileExists(p)) { cliExe = p; break; }
        }
        
        if (!cliExe.empty()) {
#ifdef _WIN32
            // chcp 65001 确保子进程使用 UTF-8 代码页，避免中文乱码
            std::string cmd = "chcp 65001 >nul && \"" + cliExe + "\" -p \"" + command + "\"";
#else
            std::string cmd = "\"" + cliExe + "\" -p \"" + command + "\"";
#endif
            // 添加剩余参数
            for (int i = commandIndex + 1; i < argc; i++) {
                std::string arg = argv[i];
                if (arg == "-py") continue;
                
                bool needQuote = (arg.find(' ') != std::string::npos || 
                                  arg.find('"') != std::string::npos ||
                                  arg.find('&') != std::string::npos ||
                                  arg.find('{') != std::string::npos ||
                                  arg.find('}') != std::string::npos);
                if (needQuote) {
                    std::string escaped;
                    for (char c : arg) {
                        if (c == '"') escaped += "\\\"";
                        else escaped += c;
                    }
                    cmd += " \"" + escaped + "\"";
                } else {
                    cmd += " " + arg;
                }
            }
            return system(cmd.c_str());
        }
    }
    
    // 未知命令，尝试 Python 后端
    std::cout << "[提示] 未知命令 '" << command << "'，尝试 Python 后端..." << std::endl;
    return executePythonBackend(argc, argv);
}