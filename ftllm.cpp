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
#include <process.h>
#else
#include <unistd.h>
#include <sys/stat.h>
#endif

const char* FTLLM_VERSION = "1.0.1";

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
                arg == "--lora" || arg == "--system" || arg == "--api_key") {
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
    
    // 搜索路径: bin 子目录或同级目录
    std::vector<std::string> searchPaths = {
        exeDir + "\\bin\\" + exeName,
        exeDir + "\\" + exeName,
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
        std::cerr << "搜索路径:" << std::endl;
        for (const auto& path : searchPaths) {
            std::cerr << "  - " << path << std::endl;
        }
        return 1;
    }
    
    // 构建命令行
    std::string cmd = "\"" + exePath + "\"";
    for (int i = startArg; i < argc; i++) {
        std::string arg = argv[i];
        if (arg == "-py") continue;  // 跳过 -py 参数
        
        bool needQuote = (arg.find(' ') != std::string::npos || 
                          arg.find('"') != std::string::npos ||
                          arg.find('&') != std::string::npos);
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
    // 检测便携版: pytools 目录在 exe 同级或上级
    std::string exeDir = getExeDirectory();
    std::string pytoolsPath;
    
    // 检查 exe同级/pytools 或 exe上级/pytools
    std::vector<std::string> searchPaths = {
        exeDir + "\\pytools",
        exeDir + "\\..\\pytools",
        exeDir + "/pytools",
        exeDir + "/../pytools"
    };
    
    for (const auto& path : searchPaths) {
        std::string cliPath = path + "\\cli.py";
        std::string cliPathUnix = path + "/cli.py";
#ifdef _WIN32
        if (GetFileAttributesA(cliPath.c_str()) != INVALID_FILE_ATTRIBUTES) {
            pytoolsPath = path;
            break;
        }
#else
        struct stat st;
        if (stat(cliPathUnix.c_str(), &st) == 0) {
            pytoolsPath = path;
            break;
        }
#endif
    }
    
    std::string cmd;
    if (!pytoolsPath.empty()) {
        // 便携版模式：直接运行 cli.py
        cmd = "python \"" + pytoolsPath + "\\cli.py\"";
    } else {
        // 安装版模式：使用 python -m ftllm
        cmd = "python -m ftllm";
    }
    
    for (int i = startArg; i < argc; i++) {
        std::string arg = argv[i];
        if (arg == "-py") continue;  // 跳过 -py 参数
        
        // 处理包含空格或特殊字符的参数
        bool needQuote = (arg.find(' ') != std::string::npos || 
                          arg.find('"') != std::string::npos ||
                          arg.find('&') != std::string::npos);
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
#ifdef _WIN32
    return system(cmd.c_str());
#else
    return system(cmd.c_str());
#endif
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
    // 设置 UTF-8 代码页
    SetConsoleOutputCP(65001);
    SetConsoleCP(65001);
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
                std::cerr << "[自动检测] 指定 --lora 参数，切换到 Python 后端" << std::endl;
                usePython = true;
                break;
            }
        }
    }
    
    if (!usePython) {
        std::string modelPath = extractModelPath(argc, argv);
        if (detectLoraInModelPath(modelPath)) {
            std::cerr << "[自动检测] 发现 LoRA 配置，切换到 Python 后端" << std::endl;
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
        std::cerr << "[提示] 检测到模型路径，尝试启动聊天..." << std::endl;
        // 构造新的参数: ftllm chat <modelPath> [其他参数]
        std::string exeDir = getExeDirectory();
        std::string cliExe = exeDir + "\\bin\\FastllmStudio_cli.exe";
        if (!fileExists(cliExe)) {
            cliExe = exeDir + "\\FastllmStudio_cli.exe";
        }
        
        if (fileExists(cliExe)) {
            std::string cmd = "\"" + cliExe + "\" -p \"" + command + "\"";
            // 添加剩余参数
            for (int i = commandIndex + 1; i < argc; i++) {
                std::string arg = argv[i];
                if (arg == "-py") continue;
                cmd += " " + arg;
            }
            return system(cmd.c_str());
        }
    }
    
    // 未知命令，尝试 Python 后端
    std::cerr << "[提示] 未知命令 '" << command << "'，尝试 Python 后端..." << std::endl;
    return executePythonBackend(argc, argv);
}