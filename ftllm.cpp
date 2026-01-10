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
#include <cctype>
#include <fstream>
#include <filesystem>
#include <optional>

#ifndef _WIN32
#error "ftllm.cpp is Windows-only in this repository."
#endif

#include <windows.h>
#include <shellapi.h>
#include <io.h>

static constexpr const char* FTLLM_VERSION = "1.0";
static constexpr int DEFAULT_SERVE_PORT = 8080;
static constexpr const char* REPL_PROMPT = "ftllm> ";

static constexpr const char* UI_LINE = "════════════════════════════════════════════════════════════";
static constexpr const char* UI_THIN_LINE = "────────────────────────────────────────────────────────────";
static constexpr const char* UI_STATUS_OK = "[√]";
static constexpr const char* UI_STATUS_WARN = "[!]";
static constexpr const char* UI_STATUS_ERR = "[×]";

static bool g_hasAnsi = false;

namespace ui {
static constexpr const char* RESET = "\x1b[0m";
static constexpr const char* BOLD = "\x1b[1m";
static constexpr const char* DIM = "\x1b[2m";
static constexpr const char* GREEN = "\x1b[32m";
static constexpr const char* YELLOW = "\x1b[33m";
static constexpr const char* RED = "\x1b[31m";
static constexpr const char* CYAN = "\x1b[36m";
static constexpr const char* MAGENTA = "\x1b[35m";
static constexpr const char* WHITE = "\x1b[97m";

// 便捷格式化输出：自动处理 ANSI 开关
inline std::ostream& ansi(std::ostream& os, const char* code) {
    if (g_hasAnsi && code) os << code;
    return os;
}
inline std::ostream& reset(std::ostream& os) {
    if (g_hasAnsi) os << ui::RESET;
    return os;
}
}

static void printStyled(const char* color, const std::string& text, bool newline = true) {
    ui::ansi(std::cout, color) << text;
    ui::reset(std::cout);
    if (newline) std::cout << std::endl;
}

static void printRule(const char* title = nullptr) {
    std::cout << UI_LINE << std::endl;
    if (title && *title) {
        ui::ansi(std::cout, ui::BOLD);
        ui::ansi(std::cout, ui::CYAN) << title;
        ui::reset(std::cout) << std::endl;
        std::cout << UI_THIN_LINE << std::endl;
    }
}

enum class StatusType { Ok, Warn, Err };

static void printStatus(StatusType type, const std::string& label, const std::string& detail = {}) {
    const char* icon = nullptr;
    const char* color = nullptr;
    switch (type) {
        case StatusType::Ok:   icon = UI_STATUS_OK;   color = ui::GREEN;  break;
        case StatusType::Warn: icon = UI_STATUS_WARN; color = ui::YELLOW; break;
        case StatusType::Err:  icon = UI_STATUS_ERR;  color = ui::RED;    break;
    }
    if (g_hasAnsi) std::cout << color << icon << ui::RESET;
    else std::cout << icon;
    std::cout << " " << label;
    if (!detail.empty()) std::cout << ": " << detail;
    std::cout << std::endl;
}

static inline void printStatusOk(const std::string& label, const std::string& detail = {}) {
    printStatus(StatusType::Ok, label, detail);
}
static inline void printStatusWarn(const std::string& label, const std::string& detail = {}) {
    printStatus(StatusType::Warn, label, detail);
}
static inline void printStatusErr(const std::string& label, const std::string& detail = {}) {
    printStatus(StatusType::Err, label, detail);
}

static void printKV(const std::string& key, const std::string& value) {
    std::cout << "    ";
    ui::ansi(std::cout, ui::DIM) << key;
    ui::reset(std::cout) << ": " << value << std::endl;
}

static void printLaunchConfig(const std::string& program, const std::string& backend, const std::string& modelPath, const std::vector<std::string>& /*args*/, bool background) {
    // 紧凑格式：一行显示核心信息
    std::cout << UI_LINE << std::endl;
    ui::ansi(std::cout, ui::BOLD) << "  启动";
    ui::reset(std::cout) << ": ";
    ui::ansi(std::cout, ui::CYAN) << program;
    ui::reset(std::cout) << " (" << backend;
    if (background) std::cout << ", 后台";
    std::cout << ")" << std::endl;
    
    if (!modelPath.empty()) {
        std::cout << "  ";
        ui::ansi(std::cout, ui::DIM) << "模型";
        ui::reset(std::cout) << ": " << modelPath << std::endl;
    }
    std::cout << UI_LINE << std::endl;
}

static void printChildOutputHeader(const std::string& program) {
    // 输出区域：只用一条分隔线，节省垂直空间
    (void)program; // 配置区已显示程序名，此处不再重复
    std::cout << std::endl;
}

static void printChildOutputFooter(int exitCode, bool background) {
    std::cout << std::endl;
    std::cout << UI_THIN_LINE << std::endl;
    if (background) {
        printStatusOk("后台服务已启动", "输入 stop 可停止");
    } else if (exitCode == 0) {
        printStatusOk("执行完成");
    } else {
        printStatusWarn("进程已退出");
    }
}

static std::wstring utf8ToWide(const std::string& s) {
    if (s.empty()) return std::wstring();
    const int wlen = MultiByteToWideChar(CP_UTF8, 0, s.c_str(), -1, nullptr, 0);
    if (wlen <= 0) return std::wstring();
    std::wstring ws;
    ws.resize(static_cast<size_t>(wlen));
    MultiByteToWideChar(CP_UTF8, 0, s.c_str(), -1, ws.data(), wlen);
    if (!ws.empty() && ws.back() == L'\0') ws.pop_back();
    return ws;
}

static std::string quoteWindowsArg(const std::string& arg) {
    // Windows CreateProcess commandLine quoting rules (compatible with MS CRT parsing):
    // wrap in quotes if needed; backslashes before quotes must be doubled.
    const bool needQuotes = arg.empty() || arg.find_first_of(" \t\n\v\"") != std::string::npos;
    if (!needQuotes) return arg;

    std::string out;
    out.push_back('"');
    size_t backslashes = 0;
    for (char c : arg) {
        if (c == '\\') {
            backslashes++;
            out.push_back('\\');
            continue;
        }
        if (c == '"') {
            out.append(backslashes, '\\');
            backslashes = 0;
            out.push_back('\\');
            out.push_back('"');
            continue;
        }
        backslashes = 0;
        out.push_back(c);
    }
    out.append(backslashes, '\\');
    out.push_back('"');
    return out;
}

static std::string buildWindowsCommandLine(const std::string& program, const std::vector<std::string>& args) {
    std::string cmd;
    cmd.reserve(program.size() + 1 + 64);
    cmd += quoteWindowsArg(program);
    for (const auto& a : args) {
        cmd.push_back(' ');
        cmd += quoteWindowsArg(a);
    }
    return cmd;
}

static HANDLE g_activeChildProcess = nullptr;
static HANDLE g_activeChildJob = nullptr;
static bool g_consoleHandlerInstalled = false;
static bool g_exitHandlerInstalled = false;

static constexpr DWORD CHILD_PROCESS_KILL_EXIT_CODE = 1;
static constexpr const char* FTLLM_FLAG_BACKGROUND_1 = "--bg";
static constexpr const char* FTLLM_FLAG_BACKGROUND_2 = "--detach";
static constexpr const char* FTLLM_FLAG_REPL = "--repl";
static constexpr const char* FTLLM_ENV_PSHOSTED = "FTLLM_PSHOSTED";

static void ensureExitHandlerInstalled();

// Forward declaration needed before PowerShell-host bootstrap helper
std::string getExeDirectory();

static std::string escapePowerShellSingleQuoted(std::string s) {
    // PowerShell 单引号字符串内：用 '' 表示一个单引号
    size_t pos = 0;
    while ((pos = s.find('\'', pos)) != std::string::npos) {
        s.replace(pos, 1, "''");
        pos += 2;
    }
    return s;
}

static bool hasEnvironmentVariable(const char* name) {
    if (name == nullptr || *name == '\0') return false;
    DWORD len = GetEnvironmentVariableA(name, nullptr, 0);
    return len > 0;
}

static bool tryLaunchPowerShellHostAndRunFtllm(const std::vector<std::string>& ftllmArgs) {
    const std::string exeDir = getExeDirectory();
    const std::string exePath = exeDir + "\\ftllm.exe";

    // 组装 PowerShell 脚本：设置环境变量防止自我重入；切到 exeDir；再运行 ftllm.exe。
    std::string script;
    script.reserve(512);
    script += "$env:";
    script += FTLLM_ENV_PSHOSTED;
    script += "='1'; ";
    script += "$OutputEncoding=[System.Text.UTF8Encoding]::UTF8; ";
    script += "[Console]::OutputEncoding=[System.Text.UTF8Encoding]::new($false); ";
    script += "[Console]::InputEncoding=[System.Text.UTF8Encoding]::new($false); ";
    script += "Set-Location -LiteralPath '";
    script += escapePowerShellSingleQuoted(exeDir);
    script += "'; ";
    script += "& '";
    script += escapePowerShellSingleQuoted(exePath);
    script += "'";
    for (const auto& a : ftllmArgs) {
        script += " '";
        script += escapePowerShellSingleQuoted(a);
        script += "'";
    }

    const std::vector<std::string> psArgs = {
        "-NoLogo",
        "-NoExit",
        "-ExecutionPolicy",
        "Bypass",
        "-Command",
        script,
    };

    auto launch = [&](const std::string& psExe) -> bool {
        const std::string cmdUtf8 = buildWindowsCommandLine(psExe, psArgs);
        std::wstring cmdLine = utf8ToWide(cmdUtf8);
        if (cmdLine.empty()) return false;

        STARTUPINFOW si;
        PROCESS_INFORMATION pi;
        ZeroMemory(&si, sizeof(si));
        ZeroMemory(&pi, sizeof(pi));
        si.cb = sizeof(si);

        std::vector<wchar_t> cmdBuf(cmdLine.begin(), cmdLine.end());
        cmdBuf.push_back(L'\0');

        const BOOL ok = CreateProcessW(
            nullptr,
            cmdBuf.data(),
            nullptr,
            nullptr,
            FALSE,
            CREATE_NEW_CONSOLE,
            nullptr,
            nullptr,
            &si,
            &pi
        );

        if (!ok) return false;
        CloseHandle(pi.hThread);
        CloseHandle(pi.hProcess);
        return true;
    };

    // 优先 pwsh（PowerShell 7+），失败再降级 powershell（Windows PowerShell 5.1）
    if (launch("pwsh.exe")) return true;
    if (launch("powershell.exe")) return true;
    return false;
}

static void forceKillActiveChild() {
    // 最小统一清理策略：
    // 1) 优先杀 Job（会递归杀进程树，覆盖“未响应/卡死/派生子进程”场景）
    // 2) 再兜底杀当前子进程
    if (g_activeChildJob != nullptr) {
        TerminateJobObject(g_activeChildJob, CHILD_PROCESS_KILL_EXIT_CODE);
    }
    if (g_activeChildProcess != nullptr) {
        TerminateProcess(g_activeChildProcess, CHILD_PROCESS_KILL_EXIT_CODE);
    }
}

static void killAndCloseActiveChild() {
    // 强杀 + 释放资源：用于“正常退出兜底”与“启动新子进程前清理旧子进程”。
    forceKillActiveChild();

    if (g_activeChildProcess != nullptr) {
        CloseHandle(g_activeChildProcess);
        g_activeChildProcess = nullptr;
    }
    if (g_activeChildJob != nullptr) {
        CloseHandle(g_activeChildJob);
        g_activeChildJob = nullptr;
    }
}

static bool isBackgroundFlag(const std::string& arg) {
    return arg == FTLLM_FLAG_BACKGROUND_1 || arg == FTLLM_FLAG_BACKGROUND_2;
}

static void killActiveChildProcessOnExit() {
    // 正常退出兜底：立刻强杀并释放子进程资源（含进程树）。
    killAndCloseActiveChild();
}

static BOOL WINAPI onConsoleControlEvent(DWORD ctrlType) {
    switch (ctrlType) {
        case CTRL_C_EVENT:
        case CTRL_BREAK_EVENT:
        case CTRL_CLOSE_EVENT:
        case CTRL_LOGOFF_EVENT:
        case CTRL_SHUTDOWN_EVENT:
            forceKillActiveChild();
            break;
        default:
            break;
    }

    // 让默认处理继续终止当前进程（我们只是先尽力杀掉子进程）
    return FALSE;
}

static void ensureConsoleHandlerInstalled() {
    if (g_consoleHandlerInstalled) return;
    if (SetConsoleCtrlHandler(onConsoleControlEvent, TRUE)) {
        g_consoleHandlerInstalled = true;
    }

    // 同时安装“正常退出兜底”，保证 main 正常返回也会清理子进程。
    ensureExitHandlerInstalled();
}

static void ensureExitHandlerInstalled() {
    if (g_exitHandlerInstalled) return;
    if (std::atexit(killActiveChildProcessOnExit) == 0) {
        g_exitHandlerInstalled = true;
    }
}

static HANDLE createKillOnCloseJobObject() {
    HANDLE job = CreateJobObjectW(nullptr, nullptr);
    if (job == nullptr) return nullptr;

    JOBOBJECT_EXTENDED_LIMIT_INFORMATION info;
    ZeroMemory(&info, sizeof(info));
    info.BasicLimitInformation.LimitFlags = JOB_OBJECT_LIMIT_KILL_ON_JOB_CLOSE;
    if (!SetInformationJobObject(job, JobObjectExtendedLimitInformation, &info, sizeof(info))) {
        CloseHandle(job);
        return nullptr;
    }
    return job;
}

static std::string g_currentChildProgram;
static bool g_currentChildBackground = false;

static int runChildProcessWindows(const std::string& program, const std::vector<std::string>& args, const std::optional<std::string>& workingDir, const bool shouldWaitForExit) {
    ensureConsoleHandlerInstalled();

    // 保证同一时间最多一个活跃子进程，避免资源泄露与难以回收。
    if (g_activeChildProcess != nullptr || g_activeChildJob != nullptr) {
        killAndCloseActiveChild();
    }

    // 记录当前子进程信息（用于输出区域显示）
    g_currentChildProgram = program;
    g_currentChildBackground = !shouldWaitForExit;

    // 显示子进程输出区域标题
    printChildOutputHeader(program);

    const std::string cmdUtf8 = buildWindowsCommandLine(program, args);
    std::wstring cmdLine = utf8ToWide(cmdUtf8);
    if (cmdLine.empty()) {
        std::cerr << "[错误] 启动子进程失败：命令行编码失败" << std::endl;
        return 1;
    }

    std::optional<std::wstring> cwd;
    if (workingDir.has_value()) {
        cwd = utf8ToWide(*workingDir);
    }

    STARTUPINFOW si;
    PROCESS_INFORMATION pi;
    ZeroMemory(&si, sizeof(si));
    ZeroMemory(&pi, sizeof(pi));
    si.cb = sizeof(si);

    // CreateProcessW 需要可写的 commandLine 缓冲区
    std::vector<wchar_t> cmdBuf(cmdLine.begin(), cmdLine.end());
    cmdBuf.push_back(L'\0');

    const BOOL ok = CreateProcessW(
        nullptr,
        cmdBuf.data(),
        nullptr,
        nullptr,
        TRUE,
        0,
        nullptr,
        cwd.has_value() ? cwd->c_str() : nullptr,
        &si,
        &pi
    );

    if (!ok) {
        const DWORD err = GetLastError();
        std::cerr << "[错误] 启动子进程失败，Win32Error=" << err << std::endl;
        return 1;
    }

    HANDLE job = createKillOnCloseJobObject();
    if (job != nullptr) {
        // 将子进程加入 job：确保 ftllm 退出时子进程被系统自动杀掉
        if (!AssignProcessToJobObject(job, pi.hProcess)) {
            // 可能因为父进程已在 job 且不允许嵌套等原因失败；降级为仅依赖 CtrlHandler
            CloseHandle(job);
            job = nullptr;
        }
    }

    g_activeChildProcess = pi.hProcess;
    g_activeChildJob = job;

    // 不等待：立刻返回（子进程会在 ftllm 退出时强杀；也会在下一次启动前被强杀回收）。
    if (!shouldWaitForExit) {
        CloseHandle(pi.hThread);
        printChildOutputFooter(0, true);
        return 0;
    }

    DWORD exitCode = 1;
    const DWORD waitRc = WaitForSingleObject(pi.hProcess, INFINITE);
    if (waitRc == WAIT_FAILED) {
        forceKillActiveChild();
        exitCode = CHILD_PROCESS_KILL_EXIT_CODE;
    } else {
        if (!GetExitCodeProcess(pi.hProcess, &exitCode)) {
            forceKillActiveChild();
            exitCode = CHILD_PROCESS_KILL_EXIT_CODE;
        }
    }

    g_activeChildProcess = nullptr;
    g_activeChildJob = nullptr;

    CloseHandle(pi.hThread);
    CloseHandle(pi.hProcess);
    if (job != nullptr) CloseHandle(job);

    printChildOutputFooter(static_cast<int>(exitCode), false);
    return static_cast<int>(exitCode);
}

// ============================================================================
// 初始化
// ============================================================================

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
            const DWORD desired = mode | ENABLE_VIRTUAL_TERMINAL_PROCESSING;
            if (SetConsoleMode(hOut, desired)) {
                g_hasAnsi = true;
            }
        }
    }
}

static bool shouldPauseAfterHelp() {
    // Explorer 双击启动时通常会创建一个仅包含本进程的控制台窗口。
    // 从现有 cmd/powershell/Windows Terminal 启动时，控制台一般还包含父进程。
    DWORD procList[8];
    const DWORD count = GetConsoleProcessList(procList, static_cast<DWORD>(std::size(procList)));
    return count == 1;
}

static void keepConsoleOpenUntilClose();

// Forward declaration for Usage
void Usage();

// ============================================================================
// 工具函数
// ============================================================================

// Forward declarations for helpers defined later in this file
std::string toLowerCopy(std::string s);
std::string getExeDirectory();
bool fileExists(const std::string& path);
bool isDirectory(const std::string& path);

static std::string trimCopy(std::string s);
static bool looksLikeHuggingFaceRepoId(const std::string& s);
static int handleModelInputWithChoiceMenu(const std::string& modelInput, const std::vector<std::string>& extraArgs, bool backgroundServe);
#ifdef _WIN32
static std::vector<std::string> splitCommandLineWindows(const std::string& utf8Line);
#endif
int executeNativeProgram(const std::string& exeName, int argc, char** argv, int startArg);
int executePythonBackend(int argc, char** argv, int startArg);

// ============================================================================
// 统一命令定义表（驱动命令检测、帮助输出、执行分发）
// ============================================================================

enum class Backend { Native, Python };

struct CommandDef {
    const char* name;           // 主命令名
    const char* aliases[3];     // 别名列表
    const char* nativeExe;      // C++ 原生程序名 (nullptr 表示无原生版)
    const char* description;    // 描述
    Backend defaultBackend;     // 默认后端
};

static const CommandDef ALL_COMMANDS[] = {
    // C++ 原生程序
    {"serve",    {"server", "api", nullptr},      "apiserver.exe",  "OpenAI API 服务器",   Backend::Native},
    {"webui",    {"web", nullptr, nullptr},       "webui.exe",      "Web 聊天界面",        Backend::Native},
    {"bench",    {"benchmark", nullptr, nullptr}, "benchmark.exe",  "性能测试",            Backend::Native},
    {"quant",    {"quantize", nullptr, nullptr},  "quant.exe",      "模型量化",            Backend::Native},
    // Python 专用
    {"run",      {"chat", nullptr, nullptr},      nullptr,          "交互式聊天",          Backend::Python},
    {"download", {nullptr, nullptr, nullptr},     nullptr,          "下载 HuggingFace 模型", Backend::Python},
    {"ui",       {nullptr, nullptr, nullptr},     nullptr,          "启动图形界面",        Backend::Python},
    {"config",   {nullptr, nullptr, nullptr},     nullptr,          "生成配置文件模板",    Backend::Python},
    {"export",   {nullptr, nullptr, nullptr},     nullptr,          "导出模型",            Backend::Python},
};
static constexpr size_t NUM_COMMANDS = sizeof(ALL_COMMANDS) / sizeof(ALL_COMMANDS[0]);

// 查找命令定义
inline const CommandDef* findCommandDef(const std::string& cmd);
inline bool isKnownCommand(const std::string& cmd);

static bool isInteractiveStdin() {
    return _isatty(_fileno(stdin)) != 0;
}

static std::string sanitizeFolderName(std::string name) {
    if (name.empty()) return "model";

    for (char& c : name) {
        const bool isBad = (c == '<' || c == '>' || c == ':' || c == '"' || c == '/' || c == '\\' || c == '|' || c == '?' || c == '*');
        if (isBad) c = '_';
    }

    while (!name.empty() && (name.back() == ' ' || name.back() == '.')) name.pop_back();
    if (name.empty()) return "model";
    return name;
}

static std::string trimCopy(std::string s) {
    auto isWs = [](unsigned char c) { return std::isspace(c) != 0; };
    while (!s.empty() && isWs(static_cast<unsigned char>(s.front()))) s.erase(s.begin());
    while (!s.empty() && isWs(static_cast<unsigned char>(s.back()))) s.pop_back();
    return s;
}

static bool looksLikeHuggingFaceRepoId(const std::string& s) {
    const std::string t = trimCopy(s);
    if (t.empty()) return false;
    if (t.find(' ') != std::string::npos || t.find('\t') != std::string::npos) return false;

    // Windows 本地路径/URI 排除
    if (t.find(':') != std::string::npos) return false;
    if (t.find('\\') != std::string::npos) return false;
    if (t.rfind("./", 0) == 0 || t.rfind("../", 0) == 0) return false;
    if (t.rfind("/", 0) == 0) return false;

    // 典型 repo id: owner/repo
    const size_t slash = t.find('/');
    if (slash == std::string::npos) return false;
    if (slash == 0 || slash == t.size() - 1) return false;
    if (t.find('/', slash + 1) != std::string::npos) return false;
    return true;
}

static std::string getModelDisplayNameFromPath(std::string modelPath) {
    if (modelPath.empty()) return "model";

    while (!modelPath.empty() && (modelPath.back() == '\\' || modelPath.back() == '/')) {
        modelPath.pop_back();
    }

    std::string leaf = modelPath;
    const size_t pos = leaf.find_last_of("/\\");
    if (pos != std::string::npos) leaf = leaf.substr(pos + 1);

    if (leaf.empty()) leaf = modelPath;

    const std::string lowerLeaf = toLowerCopy(leaf);
    if (lowerLeaf.size() > 4 && (lowerLeaf.ends_with(".flm") || lowerLeaf.ends_with(".gguf"))) {
        leaf = leaf.substr(0, leaf.find_last_of('.'));
    }

    return sanitizeFolderName(leaf);
}

static std::string getModelDisplayNameFromInput(const std::string& modelInput) {
    if (looksLikeHuggingFaceRepoId(modelInput)) {
        std::string name = modelInput;
        std::replace(name.begin(), name.end(), '/', '_');
        return sanitizeFolderName(name);
    }
    return getModelDisplayNameFromPath(modelInput);
}

#ifdef _WIN32
static std::vector<std::string> splitCommandLineWindows(const std::string& utf8Line) {
    std::vector<std::string> args;
    const std::string trimmed = trimCopy(utf8Line);
    if (trimmed.empty()) return args;

    // CommandLineToArgvW 需要“完整命令行”；这里补一个虚拟程序名以复用 Windows 的转义/引号规则。
    const std::string fullUtf8 = std::string("ftllm ") + trimmed;

    const int wlen = MultiByteToWideChar(CP_UTF8, 0, fullUtf8.c_str(), -1, nullptr, 0);
    if (wlen <= 0) return args;
    std::wstring wcmd;
    wcmd.resize(static_cast<size_t>(wlen));
    MultiByteToWideChar(CP_UTF8, 0, fullUtf8.c_str(), -1, wcmd.data(), wlen);

    int argcW = 0;
    LPWSTR* argvW = CommandLineToArgvW(wcmd.c_str(), &argcW);
    if (!argvW || argcW <= 1) {
        if (argvW) LocalFree(argvW);
        return args;
    }

    for (int i = 1; i < argcW; i++) {
        const std::wstring ws = argvW[i];
        const int u8len = WideCharToMultiByte(CP_UTF8, 0, ws.c_str(), -1, nullptr, 0, nullptr, nullptr);
        if (u8len <= 0) continue;
        std::string u8;
        u8.resize(static_cast<size_t>(u8len));
        WideCharToMultiByte(CP_UTF8, 0, ws.c_str(), -1, u8.data(), u8len, nullptr, nullptr);
        if (!u8.empty() && u8.back() == '\0') u8.pop_back();
        args.push_back(u8);
    }

    LocalFree(argvW);
    return args;
}
#endif

static std::vector<std::string> collectArgsAfterIndex(int argc, char** argv, int startIndex) {
    std::vector<std::string> out;
    if (startIndex < 0) startIndex = 0;
    for (int i = startIndex; i < argc; i++) {
        const std::string arg = argv[i];
        if (arg == "-py") continue;
        out.push_back(arg);
    }
    return out;
}

static bool argsContainFlag(const std::vector<std::string>& args, const std::string& flag) {
    return std::find(args.begin(), args.end(), flag) != args.end();
}

static int executePythonBackendWithArgs(const std::vector<std::string>& args) {
    std::vector<std::string> storage;
    storage.reserve(args.size() + 1);
    storage.push_back("ftllm");
    for (const auto& a : args) storage.push_back(a);

    std::vector<char*> argv2;
    argv2.reserve(storage.size());
    for (auto& s : storage) argv2.push_back(const_cast<char*>(s.c_str()));
    return executePythonBackend(static_cast<int>(argv2.size()), argv2.data(), 1);
}

static int executeNativeServeWithArgs(const std::string& modelPath, const std::vector<std::string>& extraArgs, const bool background) {
    std::vector<std::string> storage;
    storage.reserve(extraArgs.size() + 5);
    storage.push_back("ftllm");
    storage.push_back("serve");
    storage.push_back(modelPath);
    for (const auto& a : extraArgs) storage.push_back(a);

    if (background) storage.push_back(FTLLM_FLAG_BACKGROUND_1);

    if (!argsContainFlag(extraArgs, "--port")) {
        storage.push_back("--port");
        storage.push_back(std::to_string(DEFAULT_SERVE_PORT));
    }

    std::vector<char*> argv2;
    argv2.reserve(storage.size());
    for (auto& s : storage) argv2.push_back(const_cast<char*>(s.c_str()));

    // 复用既有逻辑（含 Qwen3 --nt 自动透传、参数补 -p 等）
    return executeNativeProgram("apiserver.exe", static_cast<int>(argv2.size()), argv2.data(), 2);
}

static int executePythonServeWithArgs(const std::string& modelInput, const std::vector<std::string>& extraArgs, const bool background) {
    std::vector<std::string> args;
    args.reserve(extraArgs.size() + 6);
    args.push_back("serve");
    args.push_back(modelInput);
    for (const auto& a : extraArgs) args.push_back(a);
    if (background) args.push_back(FTLLM_FLAG_BACKGROUND_1);
    if (!argsContainFlag(extraArgs, "--port")) {
        args.push_back("--port");
        args.push_back(std::to_string(DEFAULT_SERVE_PORT));
    }
    return executePythonBackendWithArgs(args);
}

// REPL 直接命令调用 (run/serve/webui/... + 模型 + 参数)
static int executeReplDirectCommand(const std::string& cmd, const std::vector<std::string>& tokens) {
    // tokens[0] 是 cmd，tokens[1...] 是模型和参数
    std::vector<std::string> storage;
    storage.reserve(tokens.size() + 2);
    storage.push_back("ftllm");
    for (const auto& t : tokens) storage.push_back(t);
    
    std::vector<char*> argv2;
    argv2.reserve(storage.size());
    for (auto& s : storage) argv2.push_back(const_cast<char*>(s.c_str()));
    
    // 使用统一命令表查找
    const CommandDef* def = findCommandDef(cmd);
    if (def && def->nativeExe) {
        return executeNativeProgram(def->nativeExe, static_cast<int>(argv2.size()), argv2.data(), 2);
    }
    // Python 后端
    return executePythonBackend(static_cast<int>(argv2.size()), argv2.data(), 1);
}

// 检测是否是 REPL 支持的直接命令（使用统一命令表）
static bool isReplCommand(const std::string& cmd) {
    return findCommandDef(cmd) != nullptr;
}

static int handleModelInputWithChoiceMenu(const std::string& modelInput, const std::vector<std::string>& extraArgs, const bool backgroundServe) {
    printRule("FastLLM 交互选择");
    ui::ansi(std::cout, ui::BOLD) << "模型输入";
    ui::reset(std::cout) << ": " << modelInput << std::endl;
    std::cout << UI_THIN_LINE << std::endl;
    std::cout << "  1) run    交互式运行/聊天 (Python)" << std::endl;
    if (looksLikeHuggingFaceRepoId(modelInput)) {
        std::cout << "  2) serve  启动 OpenAI API 服务器 (Python，支持自动下载)，默认端口 " << DEFAULT_SERVE_PORT;
    } else {
        std::cout << "  2) serve  启动 OpenAI API 服务器 (C++)，默认端口 " << DEFAULT_SERVE_PORT;
    }
    if (backgroundServe) {
        std::cout << "  (后台启动，立刻返回，可输入 stop 停止)";
    }
    std::cout << std::endl;
    std::cout << "  3) export 导出模型到 Models/<模型同名文件夹> (Python)" << std::endl;
    std::cout << std::endl;
    std::cout << "输入 1/2/3 (默认 2): ";
    std::cout.flush();

    std::string choice;
    if (isInteractiveStdin()) {
        std::getline(std::cin, choice);
        choice = trimCopy(choice);
    }

    const int selected = (choice == "1") ? 1 : (choice == "3") ? 3 : 2;
    if (selected == 2) {
        if (looksLikeHuggingFaceRepoId(modelInput)) {
            return executePythonServeWithArgs(modelInput, extraArgs, backgroundServe);
        }
        return executeNativeServeWithArgs(modelInput, extraArgs, backgroundServe);
    }

    if (selected == 3) {
        const std::string exeDir = getExeDirectory();
        const std::string modelName = getModelDisplayNameFromInput(modelInput);
        const std::filesystem::path outDir = std::filesystem::path(exeDir) / "Models" / modelName;
        std::error_code ec;
        std::filesystem::create_directories(outDir, ec);
        if (ec) {
            std::cerr << "[错误] 创建导出目录失败: " << outDir.string() << std::endl;
            return 1;
        }

        std::cout << "[提示] 导出目录: " << outDir.string() << std::endl;

        std::vector<std::string> args2;
        args2.reserve(extraArgs.size() + 6);
        args2.push_back("export");
        args2.push_back(modelInput);
        for (size_t i = 0; i < extraArgs.size(); i++) {
            const std::string& a = extraArgs[i];
            if (a == "-o" || a == "--output") {
                if (i + 1 < extraArgs.size()) i++;
                continue;
            }
            args2.push_back(a);
        }
        args2.push_back("-o");
        args2.push_back(outDir.string());
        return executePythonBackendWithArgs(args2);
    }

    std::vector<std::string> args1;
    args1.reserve(extraArgs.size() + 4);
    args1.push_back("run");
    args1.push_back(modelInput);
    for (const auto& a : extraArgs) args1.push_back(a);
    return executePythonBackendWithArgs(args1);
}

std::string getExeDirectory() {
    char path[MAX_PATH];
    GetModuleFileNameA(NULL, path, MAX_PATH);
    std::string fullPath(path);
    size_t pos = fullPath.find_last_of("\\/");
    return (pos != std::string::npos) ? fullPath.substr(0, pos) : ".";
}

bool fileExists(const std::string& path) {
    return GetFileAttributesA(path.c_str()) != INVALID_FILE_ATTRIBUTES;
}

bool isDirectory(const std::string& path) {
    DWORD attr = GetFileAttributesA(path.c_str());
    return (attr != INVALID_FILE_ATTRIBUTES) && (attr & FILE_ATTRIBUTE_DIRECTORY);
}

static void keepConsoleOpenUntilClose() {
    // 居中显示程序名
    std::cout << std::endl;
    std::cout << UI_LINE << std::endl;
    if (g_hasAnsi) {
        std::cout << ui::BOLD << ui::CYAN 
                  << "                  FastLLM - 高性能大语言模型推理引擎" << FTLLM_VERSION << ui::RESET << std::endl;
    } else {
        std::cout << "                  FastLLM - 高性能大语言模型推理引擎" << FTLLM_VERSION << std::endl;
    }
    std::cout << UI_LINE << std::endl;
    // 精简命令提示
    if (g_hasAnsi) {
        std::cout << ui::DIM << "  用法: " << ui::RESET << "<模型> [选项] " << ui::DIM << "|" << ui::RESET << " <命令> <模型> [选项]" << std::endl;
        std::cout << ui::DIM << "  命令: " << ui::RESET << "run serve webui bench quant " << ui::DIM << "|" << ui::RESET << " help stop exit" << std::endl;
        std::cout << ui::DIM << "  模式: " << ui::RESET << ui::RED << "-py" << ui::RESET << " " << ui::DIM << "|" << ui::RESET << " <自动>" << std::endl;
    } else {
        std::cout << "  用法: <模型> [选项]  |  <命令> <模型> [选项]" << std::endl;
        std::cout << "  命令: run serve webui bench quant | help stop exit" << std::endl;
        std::cout << "  模式: -py | <自动>" << std::endl;
    }
    std::cout << std::endl;

    std::string line;
    while (true) {
        if (g_hasAnsi) {
            std::cout << ui::CYAN << REPL_PROMPT << ui::RESET;
        } else {
            std::cout << REPL_PROMPT;
        }
        std::cout.flush();

        if (!std::getline(std::cin, line)) {
            break;
        }
        line = trimCopy(line);
        if (line.empty()) continue;

        const std::string lower = toLowerCopy(line);
        if (lower == "exit" || lower == "quit") {
            break;
        }
        if (lower == "stop") {
            killAndCloseActiveChild();
            printStatusOk("已停止", "子进程已强制终止并释放资源");
            continue;
        }
        // 简单帮助 (h / ?)
        if (lower == "?" || lower == "h") {
            // 紧凑版帮助（多色彩）
            std::cout << UI_LINE << std::endl;
            
            // 命令列表
            if (g_hasAnsi) std::cout << ui::BOLD << ui::CYAN;
            std::cout << "命令";
            if (g_hasAnsi) std::cout << ui::RESET;
            std::cout << std::endl;
            if (g_hasAnsi) {
                std::cout << "  " << ui::GREEN << "run" << ui::RESET << "," << ui::GREEN << "chat" << ui::RESET 
                          << "    交互聊天" << ui::DIM << "(Py)" << ui::RESET 
                          << "      " << ui::GREEN << "serve" << ui::RESET << "," << ui::GREEN << "api" << ui::RESET 
                          << "   API服务" << ui::DIM << "(C++)" << ui::RESET << std::endl;
                std::cout << "  " << ui::GREEN << "webui" << ui::RESET 
                          << "       Web界面" << ui::DIM << "(C++)" << ui::RESET 
                          << "       " << ui::GREEN << "bench" << ui::RESET 
                          << "       性能测试" << ui::DIM << "(C++)" << ui::RESET << std::endl;
                std::cout << "  " << ui::GREEN << "quant" << ui::RESET 
                          << "       模型量化" << ui::DIM << "(C++)" << ui::RESET 
                          << "       " << ui::GREEN << "export" << ui::RESET 
                          << "      导出模型" << ui::DIM << "(Py)" << ui::RESET << std::endl;
            } else {
                std::cout << "  run,chat    交互聊天(Py)      serve,api   API服务(C++)" << std::endl;
                std::cout << "  webui       Web界面(C++)       bench       性能测试(C++)" << std::endl;
                std::cout << "  quant       模型量化(C++)       export      导出模型(Py)" << std::endl;
            }
            
            std::cout << UI_THIN_LINE << std::endl;
            
            // 用法
            if (g_hasAnsi) std::cout << ui::BOLD << ui::CYAN;
            std::cout << "用法";
            if (g_hasAnsi) std::cout << ui::RESET;
            std::cout << std::endl;
            if (g_hasAnsi) {
                std::cout << "  " << ui::YELLOW << "<模型>" << ui::RESET << " [选项]           → 选择操作(run/serve/export)" << std::endl;
                std::cout << "  " << ui::GREEN << "<命令>" << ui::RESET << " " << ui::YELLOW << "<模型>" << ui::RESET << " [选项]   → 直接执行" << std::endl;
            } else {
                std::cout << "  <模型> [选项]           → 选择操作(run/serve/export)" << std::endl;
                std::cout << "  <命令> <模型> [选项]   → 直接执行" << std::endl;
            }
            
            std::cout << UI_THIN_LINE << std::endl;
            if (g_hasAnsi) std::cout << ui::DIM;
            std::cout << "  stop 停止子进程 | exit 退出 | " << ui::CYAN << "help" << ui::RESET << ui::DIM << " 详细帮助 | 模型: .flm .gguf HF";
            if (g_hasAnsi) std::cout << ui::RESET;
            std::cout << std::endl;
            continue;
        }
        // 详细帮助 (help)
        if (lower == "help") {
            // 命令 (C++ 原生程序)
            if (g_hasAnsi) std::cout << ui::BOLD << ui::CYAN;
            std::cout << "命令 (C++ 原生程序)";
            if (g_hasAnsi) std::cout << ui::RESET;
            std::cout << std::endl;
            if (g_hasAnsi) {
                std::cout << "  " << ui::GREEN << "serve" << ui::RESET << ", " << ui::GREEN << "server" << ui::RESET << "                 启动 OpenAI API 服务器 (apiserver)" << std::endl;
                std::cout << "  " << ui::GREEN << "webui" << ui::RESET << "                         启动 Web UI (webui)" << std::endl;
                std::cout << "  " << ui::GREEN << "bench" << ui::RESET << ", " << ui::GREEN << "benchmark" << ui::RESET << "              性能测试 (benchmark)" << std::endl;
                std::cout << "  " << ui::GREEN << "quant" << ui::RESET << ", " << ui::GREEN << "quantize" << ui::RESET << "               模型量化 (quant)" << std::endl;
            } else {
                std::cout << "  serve, server                 启动 OpenAI API 服务器 (apiserver)" << std::endl;
                std::cout << "  webui                         启动 Web UI (webui)" << std::endl;
                std::cout << "  bench, benchmark              性能测试 (benchmark)" << std::endl;
                std::cout << "  quant, quantize               模型量化 (quant)" << std::endl;
            }
            std::cout << std::endl;
            
            // 命令 (Python 后端)
            if (g_hasAnsi) std::cout << ui::BOLD << ui::CYAN;
            std::cout << "命令 (Python 后端)";
            if (g_hasAnsi) std::cout << ui::RESET;
            std::cout << std::endl;
            if (g_hasAnsi) {
                std::cout << "  " << ui::GREEN << "run" << ui::RESET << ", " << ui::GREEN << "chat" << ui::RESET << "                     交互式聊天" << std::endl;
                std::cout << "  " << ui::GREEN << "download" << ui::RESET << " <model>              下载 HuggingFace 模型" << std::endl;
                std::cout << "  " << ui::GREEN << "ui" << ui::RESET << "                            启动图形界面" << std::endl;
                std::cout << "  " << ui::GREEN << "config" << ui::RESET << " [file]                 生成配置文件模板" << std::endl;
                std::cout << "  " << ui::GREEN << "export" << ui::RESET << " -o <path>              导出模型" << std::endl;
            } else {
                std::cout << "  run, chat                     交互式聊天" << std::endl;
                std::cout << "  download <model>              下载 HuggingFace 模型" << std::endl;
                std::cout << "  ui                            启动图形界面" << std::endl;
                std::cout << "  config [file]                 生成配置文件模板" << std::endl;
                std::cout << "  export -o <path>              导出模型" << std::endl;
            }
            std::cout << std::endl;
            
            // 模式切换
            if (g_hasAnsi) std::cout << ui::BOLD << ui::CYAN;
            std::cout << "模式切换";
            if (g_hasAnsi) std::cout << ui::RESET;
            std::cout << std::endl;
            if (g_hasAnsi) {
                std::cout << "  " << ui::RED << "-py" << ui::RESET << "                           使用 Python 后端 (支持 LoRA 动态加载等)" << std::endl;
            } else {
                std::cout << "  -py                           使用 Python 后端 (支持 LoRA 动态加载等)" << std::endl;
            }
            std::cout << "  (自动)                        检测到 --lora / lora/ 目录时自动切换" << std::endl;
            std::cout << std::endl;
            
            // 模型格式
            if (g_hasAnsi) std::cout << ui::BOLD << ui::CYAN;
            std::cout << "模型格式 (自动识别)";
            if (g_hasAnsi) std::cout << ui::RESET;
            std::cout << std::endl;
            if (g_hasAnsi) {
                std::cout << "  " << ui::YELLOW << ".flm" << ui::RESET << "                          FastLLM 原生格式" << std::endl;
                std::cout << "  " << ui::YELLOW << ".gguf" << ui::RESET << "                         GGUF 格式" << std::endl;
                std::cout << "  " << ui::YELLOW << "HuggingFace 目录" << ui::RESET << "              本地目录 (含 config.json)" << std::endl;
                std::cout << "  " << ui::YELLOW << "HuggingFace Repo ID" << ui::RESET << "           如 Qwen/Qwen2.5-7B (自动下载, 需 -py)" << std::endl;
            } else {
                std::cout << "  .flm                          FastLLM 原生格式" << std::endl;
                std::cout << "  .gguf                         GGUF 格式" << std::endl;
                std::cout << "  HuggingFace 目录              本地目录 (含 config.json)" << std::endl;
                std::cout << "  HuggingFace Repo ID           如 Qwen/Qwen2.5-7B (自动下载, 需 -py)" << std::endl;
            }
            std::cout << std::endl;
            
            // 基础参数
            if (g_hasAnsi) std::cout << ui::BOLD << ui::CYAN;
            std::cout << "基础参数";
            if (g_hasAnsi) std::cout << ui::RESET;
            std::cout << std::endl;
            if (g_hasAnsi) {
                std::cout << "  " << ui::RED << "-p" << ui::RESET << ", " << ui::RED << "--path" << ui::RESET << " <路径>             模型路径" << std::endl;
                std::cout << "  " << ui::RED << "--device" << ui::RESET << " <设备>               cuda, cpu, numa" << std::endl;
                std::cout << "  " << ui::RED << "--dtype" << ui::RESET << " <类型>                float16, int8, int4, int4g" << std::endl;
                std::cout << "  " << ui::RED << "-t" << ui::RESET << ", " << ui::RED << "--threads" << ui::RESET << " <数量>          CPU 线程数" << std::endl;
                std::cout << "  " << ui::RED << "--model_name" << ui::RESET << " <名称>            模型显示名称 (用于 API 返回)" << std::endl;
            } else {
                std::cout << "  -p, --path <路径>             模型路径" << std::endl;
                std::cout << "  --device <设备>               cuda, cpu, numa" << std::endl;
                std::cout << "  --dtype <类型>                float16, int8, int4, int4g" << std::endl;
                std::cout << "  -t, --threads <数量>          CPU 线程数" << std::endl;
                std::cout << "  --model_name <名称>           模型显示名称 (用于 API 返回)" << std::endl;
            }
            std::cout << std::endl;
            
            // 服务器参数
            if (g_hasAnsi) std::cout << ui::BOLD << ui::CYAN;
            std::cout << "服务器参数 (serve/webui)";
            if (g_hasAnsi) std::cout << ui::RESET;
            std::cout << std::endl;
            if (g_hasAnsi) {
                std::cout << "  " << ui::RED << "--host" << ui::RESET << " <地址>                 监听地址 (默认: 127.0.0.1)" << std::endl;
                std::cout << "  " << ui::RED << "--port" << ui::RESET << " <端口>                 监听端口 (默认: 8080)" << std::endl;
                std::cout << "  " << ui::RED << "--api_key" << ui::RESET << " <密钥>              API 密钥认证" << std::endl;
                std::cout << "  " << ui::RED << "--nt" << ui::RESET << " <true|false>             输入<think>归一化 (稳定 KV 缓存)" << std::endl;
            } else {
                std::cout << "  --host <地址>                 监听地址 (默认: 127.0.0.1)" << std::endl;
                std::cout << "  --port <端口>                 监听端口 (默认: 8080)" << std::endl;
                std::cout << "  --api_key <密钥>              API 密钥认证" << std::endl;
                std::cout << "  --nt <true|false>             输入<think>归一化 (稳定 KV 缓存)" << std::endl;
            }
            std::cout << std::endl;
            
            // Batch 参数
            if (g_hasAnsi) std::cout << ui::BOLD << ui::CYAN;
            std::cout << "Batch / 并发参数";
            if (g_hasAnsi) std::cout << ui::RESET;
            std::cout << std::endl;
            if (g_hasAnsi) {
                std::cout << "  " << ui::RED << "--batch" << ui::RESET << " <数量>                批处理大小" << std::endl;
                std::cout << "  " << ui::RED << "--max_batch" << ui::RESET << " <数量>            最大批处理数量" << std::endl;
                std::cout << "  " << ui::RED << "--max_token" << ui::RESET << " <数量>            最大生成 Token 数 (webui)" << std::endl;
            } else {
                std::cout << "  --batch <数量>                批处理大小" << std::endl;
                std::cout << "  --max_batch <数量>            最大批处理数量" << std::endl;
                std::cout << "  --max_token <数量>            最大生成 Token 数 (webui)" << std::endl;
            }
            std::cout << std::endl;
            
            // CUDA / 加速参数
            if (g_hasAnsi) std::cout << ui::BOLD << ui::CYAN;
            std::cout << "CUDA / 加速参数";
            if (g_hasAnsi) std::cout << ui::RESET;
            std::cout << std::endl;
            if (g_hasAnsi) {
                std::cout << "  " << ui::RED << "--cuda_embedding" << ui::RESET << "              CUDA 运行 Embedding 层" << std::endl;
                std::cout << "  " << ui::RED << "--cuda_shared_expert" << ui::RESET << "          CUDA 共享专家优化 (MOE)" << std::endl;
                std::cout << "  " << ui::RED << "--cuda_se" << ui::RESET << "                     --cuda_shared_expert 简写" << std::endl;
                std::cout << "  " << ui::RED << "--enable_amx" << ui::RESET << ", " << ui::RED << "--amx" << ui::RESET << "           启用 Intel AMX 加速" << std::endl;
            } else {
                std::cout << "  --cuda_embedding              CUDA 运行 Embedding 层" << std::endl;
                std::cout << "  --cuda_shared_expert          CUDA 共享专家优化 (MOE)" << std::endl;
                std::cout << "  --cuda_se                     --cuda_shared_expert 简写" << std::endl;
                std::cout << "  --enable_amx, --amx           启用 Intel AMX 加速" << std::endl;
            }
            std::cout << std::endl;
            
            // MOE 参数
            if (g_hasAnsi) std::cout << ui::BOLD << ui::CYAN;
            std::cout << "MOE (混合专家) 参数";
            if (g_hasAnsi) std::cout << ui::RESET;
            std::cout << std::endl;
            if (g_hasAnsi) {
                std::cout << "  " << ui::RED << "--moe_device" << ui::RESET << " <设备>            MOE 专家层设备 (cuda, cpu)" << std::endl;
                std::cout << "  " << ui::RED << "--moe_dtype" << ui::RESET << " <类型>             MOE 专家层数据类型" << std::endl;
                std::cout << "  " << ui::RED << "--moe_experts" << ui::RESET << " <数量>           启用的 MOE 专家数量" << std::endl;
            } else {
                std::cout << "  --moe_device <设备>           MOE 专家层设备 (cuda, cpu)" << std::endl;
                std::cout << "  --moe_dtype <类型>            MOE 专家层数据类型" << std::endl;
                std::cout << "  --moe_experts <数量>          启用的 MOE 专家数量" << std::endl;
            }
            std::cout << std::endl;
            
            // 缓存参数
            if (g_hasAnsi) std::cout << ui::BOLD << ui::CYAN;
            std::cout << "缓存参数";
            if (g_hasAnsi) std::cout << ui::RESET;
            std::cout << std::endl;
            if (g_hasAnsi) {
                std::cout << "  " << ui::RED << "--kv_cache_limit" << ui::RESET << " <大小>        KV 缓存限制 (如 8G, 4096M)" << std::endl;
                std::cout << "  " << ui::RED << "--cache_history" << ui::RESET << "               启用历史缓存" << std::endl;
                std::cout << "  " << ui::RED << "--cache_fast" << ui::RESET << "                  启用快速缓存模式" << std::endl;
                std::cout << "  " << ui::RED << "--cache_dir" << ui::RESET << " <路径>            缓存目录路径" << std::endl;
            } else {
                std::cout << "  --kv_cache_limit <大小>       KV 缓存限制 (如 8G, 4096M)" << std::endl;
                std::cout << "  --cache_history               启用历史缓存" << std::endl;
                std::cout << "  --cache_fast                  启用快速缓存模式" << std::endl;
                std::cout << "  --cache_dir <路径>            缓存目录路径" << std::endl;
            }
            std::cout << std::endl;
            
            // LoRA 参数
            if (g_hasAnsi) std::cout << ui::BOLD << ui::CYAN;
            std::cout << "LoRA 参数 (自动切换 Python)";
            if (g_hasAnsi) std::cout << ui::RESET;
            std::cout << std::endl;
            if (g_hasAnsi) {
                std::cout << "  " << ui::RED << "--lora" << ui::RESET << " <路径>                 LoRA 适配器路径" << std::endl;
                std::cout << "  " << ui::RED << "--custom" << ui::RESET << " <配置>               自定义模型配置" << std::endl;
                std::cout << "  " << ui::RED << "--dtype_config" << ui::RESET << " <配置>         数据类型配置文件" << std::endl;
                std::cout << "  " << ui::RED << "--ori" << ui::RESET << "                         使用原始权重 (禁用量化)" << std::endl;
            } else {
                std::cout << "  --lora <路径>                 LoRA 适配器路径" << std::endl;
                std::cout << "  --custom <配置>               自定义模型配置" << std::endl;
                std::cout << "  --dtype_config <配置>         数据类型配置文件" << std::endl;
                std::cout << "  --ori                         使用原始权重 (禁用量化)" << std::endl;
            }
            std::cout << std::endl;
            
            // 模板 / 工具调用
            if (g_hasAnsi) std::cout << ui::BOLD << ui::CYAN;
            std::cout << "模板 / 工具调用";
            if (g_hasAnsi) std::cout << ui::RESET;
            std::cout << std::endl;
            if (g_hasAnsi) {
                std::cout << "  " << ui::RED << "--chat_template" << ui::RESET << " <模板>        对话模板 (覆盖自动检测)" << std::endl;
                std::cout << "  " << ui::RED << "--tool_call_parser" << ui::RESET << " <类型>     工具调用解析器类型" << std::endl;
                std::cout << "  " << ui::RED << "--enable_thinking" << ui::RESET << "             启用思考模式 (<think>标签)" << std::endl;
                std::cout << "  " << ui::RED << "--think" << ui::RESET << "                       Python 后端思考模式" << std::endl;
                std::cout << "  " << ui::RED << "--hide_input" << ui::RESET << "                  隐藏输入内容 (隐私保护)" << std::endl;
            } else {
                std::cout << "  --chat_template <模板>        对话模板 (覆盖自动检测)" << std::endl;
                std::cout << "  --tool_call_parser <类型>     工具调用解析器类型" << std::endl;
                std::cout << "  --enable_thinking             启用思考模式 (<think>标签)" << std::endl;
                std::cout << "  --think                       Python 后端思考模式" << std::endl;
                std::cout << "  --hide_input                  隐藏输入内容 (隐私保护)" << std::endl;
            }
            std::cout << std::endl;
            
            // 示例
            if (g_hasAnsi) std::cout << ui::BOLD << ui::CYAN;
            std::cout << "示例";
            if (g_hasAnsi) std::cout << ui::RESET;
            std::cout << std::endl;
            if (g_hasAnsi) {
                std::cout << "  " << ui::GREEN << "run" << ui::RESET << " " << ui::YELLOW << "D:\\Models\\Qwen2.5-7B" << ui::RESET << " " << ui::RED << "--device" << ui::RESET << " cuda" << std::endl;
                std::cout << "  " << ui::GREEN << "run" << ui::RESET << " " << ui::YELLOW << "D:\\Models\\Qwen2.5-7B" << ui::RESET << " " << ui::RED << "--lora" << ui::RESET << " ./lora" << std::endl;
                std::cout << "  " << ui::GREEN << "serve" << ui::RESET << " " << ui::YELLOW << "D:\\Models\\Qwen2.5-7B" << ui::RESET << " " << ui::RED << "--port" << ui::RESET << " 8080 " << ui::RED << "--batch" << ui::RESET << " 4" << std::endl;
                std::cout << "  " << ui::GREEN << "serve" << ui::RESET << " " << ui::YELLOW << "D:\\Models\\Qwen2.5-7B" << ui::RESET << " " << ui::RED << "--cuda_embedding" << ui::RESET << " " << ui::RED << "--kv_cache_limit" << ui::RESET << " 8G" << std::endl;
                std::cout << "  " << ui::GREEN << "webui" << ui::RESET << " " << ui::YELLOW << "D:\\Models\\Qwen2.5-7B" << ui::RESET << " " << ui::RED << "--port" << ui::RESET << " 1616" << std::endl;
                std::cout << "  " << ui::GREEN << "download" << ui::RESET << " " << ui::YELLOW << "Qwen/Qwen2.5-7B-Instruct" << ui::RESET << std::endl;
            } else {
                std::cout << "  run D:\\Models\\Qwen2.5-7B --device cuda" << std::endl;
                std::cout << "  run D:\\Models\\Qwen2.5-7B --lora ./lora" << std::endl;
                std::cout << "  serve D:\\Models\\Qwen2.5-7B --port 8080 --batch 4" << std::endl;
                std::cout << "  serve D:\\Models\\Qwen2.5-7B --cuda_embedding --kv_cache_limit 8G" << std::endl;
                std::cout << "  webui D:\\Models\\Qwen2.5-7B --port 1616" << std::endl;
                std::cout << "  download Qwen/Qwen2.5-7B-Instruct" << std::endl;
            }
            std::cout << std::endl;
            std::cout << UI_THIN_LINE << std::endl;
            if (g_hasAnsi) std::cout << ui::DIM;
            std::cout << "  输入 " << ui::CYAN << "h" << ui::RESET << ui::DIM << " 或 " << ui::CYAN << "?" << ui::RESET << ui::DIM << " 查看简要帮助";
            if (g_hasAnsi) std::cout << ui::RESET;
            std::cout << std::endl;
            continue;
        }

        const std::vector<std::string> tokens = splitCommandLineWindows(line);
        if (tokens.empty()) continue;

        // 检测是否是直接命令 (run/serve/webui/...)
        if (isReplCommand(tokens[0])) {
            if (tokens.size() < 2) {
                printStatusWarn("缺少模型参数", "用法: " + tokens[0] + " <模型> [选项]");
                continue;
            }
            const int rc = executeReplDirectCommand(tokens[0], tokens);
            (void)rc;
            std::cout << std::endl;
            continue;
        }

        const std::string modelInput = tokens[0];
        std::vector<std::string> extraArgs;
        if (tokens.size() > 1) {
            extraArgs.assign(tokens.begin() + 1, tokens.end());
        }

        const bool isLocal = isDirectory(modelInput) || fileExists(modelInput);
        const bool isRepoId = looksLikeHuggingFaceRepoId(modelInput);
        const bool looksLikePath = (modelInput.find('\\') != std::string::npos ||
                                   modelInput.find(':') != std::string::npos ||
                                   modelInput.ends_with(".flm") ||
                                   modelInput.ends_with(".gguf"));

        if (!isLocal && !isRepoId && !looksLikePath) {
            printStatusWarn("请输入模型路径或 Repo ID", "输入 help 查看示例");
            continue;
        }

        const int rc = handleModelInputWithChoiceMenu(modelInput, extraArgs, true);
        (void)rc;
        std::cout << std::endl;
    }
}

std::string toLowerCopy(std::string s) {
    std::transform(s.begin(), s.end(), s.begin(), [](unsigned char c) {
        return static_cast<char>(std::tolower(c));
    });
    return s;
}

bool containsQwen3Hint(const std::string& s) {
    const std::string lower = toLowerCopy(s);
    return (lower.find("qwen3") != std::string::npos) || (lower.find("qwen-3") != std::string::npos);
}

std::string readAllTextFile(const std::string& path) {
    std::ifstream ifs(path, std::ios::in | std::ios::binary);
    if (!ifs.is_open()) return "";
    return std::string(std::istreambuf_iterator<char>(ifs), std::istreambuf_iterator<char>());
}

bool shouldEnableThinkNormalizeForModelPath(const std::string& modelPath) {
    if (modelPath.empty()) return false;
    if (!isDirectory(modelPath)) return false;

    const std::string configPathWin = modelPath + "\\config.json";
    const std::string configPathPosix = modelPath + "/config.json";
    std::string configPath;
    if (fileExists(configPathWin)) configPath = configPathWin;
    else if (fileExists(configPathPosix)) configPath = configPathPosix;
    else return false;

    const std::string raw = readAllTextFile(configPath);
    if (raw.empty()) return false;
    return containsQwen3Hint(raw);
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
        
        // 跳过命令名（使用统一命令表）
        if (isKnownCommand(arg)) continue;
        
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

// 查找命令定义（实现）
inline const CommandDef* findCommandDef(const std::string& cmd) {
    const std::string lower = toLowerCopy(cmd);
    for (const auto& def : ALL_COMMANDS) {
        if (lower == def.name) return &def;
        for (int i = 0; def.aliases[i]; i++) {
            if (lower == def.aliases[i]) return &def;
        }
    }
    return nullptr;
}

// 检测是否是已知命令（实现）
inline bool isKnownCommand(const std::string& cmd) {
    return findCommandDef(cmd) != nullptr;
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

    // 构建参数（始终走 CreateProcessW 子进程）
    std::vector<std::string> childArgs;
    childArgs.reserve(static_cast<size_t>(argc - startArg + 8));

    bool shouldWaitForExit = true;
    
    // 检查是否已有 -p/--path 参数
    bool hasPathArg = false;
    std::string positionalModelPath;
    std::string pathArgValue;
    bool hasNtArg = false;
    
    for (int i = startArg; i < argc; i++) {
        std::string arg = argv[i];
        if (arg == "-p" || arg == "--path") {
            hasPathArg = true;
            if (i + 1 < argc) {
                pathArgValue = argv[i + 1];
            }
            break;
        }
    }

    for (int i = startArg; i < argc; i++) {
        std::string arg = argv[i];
        if (arg == "--nt") hasNtArg = true;
    }
    
    // 处理参数
    for (int i = startArg; i < argc; i++) {
        std::string arg = argv[i];
        if (arg == "-py") continue;  // 跳过 -py 参数
        if (isBackgroundFlag(arg)) {
            shouldWaitForExit = false;
            continue;
        }
        
        // 检测第一个位置参数（模型路径），自动补全 -p
        if (!hasPathArg && positionalModelPath.empty() && arg[0] != '-') {
            // 检查是否像模型路径
            if (isDirectory(arg) || fileExists(arg) || 
                arg.find(':') != std::string::npos ||
                arg.find('/') != std::string::npos ||
                arg.find('\\') != std::string::npos) {
                positionalModelPath = arg;
                childArgs.push_back("-p");
            }
        }

        childArgs.push_back(arg);
    }

    // 仅对 apiserver.exe：主程序根据 config.json 判断是否为 Qwen3，并透传归一化开关。
    const std::string modelPath = !pathArgValue.empty() ? pathArgValue : positionalModelPath;
    if (exeName == "apiserver.exe" && !hasNtArg) {
        if (shouldEnableThinkNormalizeForModelPath(modelPath)) {
            childArgs.push_back("--nt");
            childArgs.push_back("true");
        }
    }

    // 显示启动配置块
    printLaunchConfig(exeName, "C++ 原生", modelPath, childArgs, !shouldWaitForExit);

    return runChildProcessWindows(exePath, childArgs, std::optional<std::string>(exeDir), shouldWaitForExit);
}

// ============================================================================
// Python 后端执行
// ============================================================================

int executePythonBackend(int argc, char** argv, int startArg = 1) {
    const std::string exeDir = getExeDirectory();
    const std::string ftllmPath = exeDir + "/ftllm";
    const bool hasLocalPythonPackage = fileExists(ftllmPath + "/__init__.py");

    bool shouldWaitForExit = true;

    std::vector<std::string> args;
    args.reserve(static_cast<size_t>(argc - startArg + 4));
    args.push_back("-m");
    args.push_back("ftllm");
    for (int i = startArg; i < argc; i++) {
        const std::string a = argv[i];
        if (a == "-py") continue;
        if (isBackgroundFlag(a)) {
            shouldWaitForExit = false;
            continue;
        }
        args.push_back(a);
    }

    // 提取模型路径用于显示
    std::string modelPath;
    for (size_t i = 0; i < args.size(); i++) {
        const std::string& a = args[i];
        if (a == "-p" || a == "--path") {
            if (i + 1 < args.size()) modelPath = args[i + 1];
            break;
        }
        // 位置参数：跳过 -m ftllm 之后的子命令，取第一个非 - 开头的
        if (i >= 2 && !a.empty() && a[0] != '-' && modelPath.empty()) {
            // 检查是否为子命令
            static const char* cmds[] = {"run", "chat", "serve", "server", "export", "download", "webui", "config", nullptr};
            bool isCmd = false;
            for (int j = 0; cmds[j]; j++) {
                if (a == cmds[j]) { isCmd = true; break; }
            }
            if (!isCmd) modelPath = a;
        }
    }

    // 显示启动配置块
    printLaunchConfig("python -m ftllm", "Python", modelPath, args, !shouldWaitForExit);

    const std::optional<std::string> cwd = hasLocalPythonPackage ? std::optional<std::string>(exeDir) : std::nullopt;
    return runChildProcessWindows("python", args, cwd, shouldWaitForExit);
}

// ============================================================================
// 帮助信息（基于统一命令表）
// ============================================================================

// 打印命令列表（带颜色）
static void printCommandList(Backend filter, const char* title) {
    ui::ansi(std::cout, ui::BOLD);
    ui::ansi(std::cout, ui::CYAN) << title;
    ui::reset(std::cout) << std::endl;
    
    for (const auto& cmd : ALL_COMMANDS) {
        if (cmd.defaultBackend != filter) continue;
        
        // 格式化命令名和别名
        std::string names = cmd.name;
        for (int i = 0; cmd.aliases[i]; i++) {
            names += ", ";
            names += cmd.aliases[i];
        }
        
        // 输出：左对齐命令名（30字符），右边描述
        std::cout << "  ";
        ui::ansi(std::cout, ui::GREEN) << names;
        ui::reset(std::cout);
        // 填充空格到固定宽度
        for (size_t i = names.size(); i < 28; i++) std::cout << ' ';
        std::cout << cmd.description << std::endl;
    }
    std::cout << std::endl;
}

// 打印参数帮助行
static void printParamLine(const char* param, const char* desc) {
    std::cout << "  ";
    ui::ansi(std::cout, ui::RED) << param;
    ui::reset(std::cout);
    for (size_t i = strlen(param); i < 28; i++) std::cout << ' ';
    std::cout << desc << std::endl;
}

// 打印示例行
static void printExampleLine(const char* cmd, const char* model, const char* args = nullptr) {
    std::cout << "  ";
    ui::ansi(std::cout, ui::GREEN) << cmd;
    ui::reset(std::cout) << " ";
    ui::ansi(std::cout, ui::YELLOW) << model;
    ui::reset(std::cout);
    if (args) {
        std::cout << " ";
        ui::ansi(std::cout, ui::RED) << args;
        ui::reset(std::cout);
    }
    std::cout << std::endl;
}

void Usage() {
    std::cout << "Usage: ftllm <command> [options] [model_path]" << std::endl;
    std::cout << std::endl;
    ui::ansi(std::cout, ui::BOLD);
    std::cout << "FastLLM - 高性能大语言模型推理引擎 (v" << FTLLM_VERSION << ")";
    ui::reset(std::cout) << std::endl;
    std::cout << std::endl;
    
    printCommandList(Backend::Native, "命令 (C++ 原生程序):");
    printCommandList(Backend::Python, "命令 (Python 后端):");
    
    ui::ansi(std::cout, ui::BOLD);
    ui::ansi(std::cout, ui::CYAN) << "模式切换:";
    ui::reset(std::cout) << std::endl;
    printParamLine("-py", "使用 Python 后端 (支持 LoRA 动态加载等)");
    std::cout << "  (自动)                        检测到 --lora / lora/ 目录时自动切换" << std::endl;
    std::cout << std::endl;
    
    ui::ansi(std::cout, ui::BOLD);
    ui::ansi(std::cout, ui::CYAN) << "模型格式 (自动识别):";
    ui::reset(std::cout) << std::endl;
    std::cout << "  .flm                          FastLLM 原生格式" << std::endl;
    std::cout << "  .gguf                         GGUF 格式" << std::endl;
    std::cout << "  HuggingFace 目录              本地目录 (含 config.json)" << std::endl;
    std::cout << "  HuggingFace Repo ID           如 Qwen/Qwen2.5-7B (自动下载, 需 -py)" << std::endl;
    std::cout << std::endl;
    
    // ===== 基础参数 =====
    ui::ansi(std::cout, ui::BOLD);
    ui::ansi(std::cout, ui::CYAN) << "基础参数:";
    ui::reset(std::cout) << std::endl;
    printParamLine("-p, --path <路径>", "模型路径");
    printParamLine("--device <设备>", "cuda, cpu, numa");
    printParamLine("--dtype <类型>", "float16, int8, int4, int4g");
    printParamLine("-t, --threads <数量>", "CPU 线程数");
    printParamLine("--model_name <名称>", "模型显示名称 (用于 API 返回)");
    std::cout << std::endl;
    
    // ===== 服务器参数 =====
    ui::ansi(std::cout, ui::BOLD);
    ui::ansi(std::cout, ui::CYAN) << "服务器参数 (serve/webui):";
    ui::reset(std::cout) << std::endl;
    printParamLine("--host <地址>", "监听地址 (默认: 127.0.0.1)");
    printParamLine("--port <端口>", "监听端口 (默认: 8080)");
    printParamLine("--api_key <密钥>", "API 密钥认证");
    printParamLine("--nt <true|false>", "输入<think>归一化 (稳定 KV 缓存命中)");
    std::cout << std::endl;
    
    // ===== Batch 参数 =====
    ui::ansi(std::cout, ui::BOLD);
    ui::ansi(std::cout, ui::CYAN) << "Batch / 并发参数:";
    ui::reset(std::cout) << std::endl;
    printParamLine("--batch <数量>", "批处理大小");
    printParamLine("--max_batch <数量>", "最大批处理数量");
    printParamLine("--max_token <数量>", "最大生成 Token 数 (webui)");
    std::cout << std::endl;
    
    // ===== CUDA / 加速参数 =====
    ui::ansi(std::cout, ui::BOLD);
    ui::ansi(std::cout, ui::CYAN) << "CUDA / 加速参数:";
    ui::reset(std::cout) << std::endl;
    printParamLine("--cuda_embedding", "在 CUDA 上运行 Embedding 层");
    printParamLine("--cuda_shared_expert", "CUDA 共享专家优化 (MOE)");
    printParamLine("--cuda_se", "--cuda_shared_expert 简写");
    printParamLine("--enable_amx, --amx", "启用 Intel AMX 加速");
    std::cout << std::endl;
    
    // ===== MOE 参数 =====
    ui::ansi(std::cout, ui::BOLD);
    ui::ansi(std::cout, ui::CYAN) << "MOE (混合专家) 参数:";
    ui::reset(std::cout) << std::endl;
    printParamLine("--moe_device <设备>", "MOE 专家层设备 (cuda, cpu)");
    printParamLine("--moe_dtype <类型>", "MOE 专家层数据类型");
    printParamLine("--moe_experts <数量>", "启用的 MOE 专家数量");
    std::cout << std::endl;
    
    // ===== 缓存参数 =====
    ui::ansi(std::cout, ui::BOLD);
    ui::ansi(std::cout, ui::CYAN) << "缓存参数:";
    ui::reset(std::cout) << std::endl;
    printParamLine("--kv_cache_limit <大小>", "KV 缓存限制 (如 8G, 4096M)");
    printParamLine("--cache_history", "启用历史缓存");
    printParamLine("--cache_fast", "启用快速缓存模式");
    printParamLine("--cache_dir <路径>", "缓存目录路径");
    std::cout << std::endl;
    
    // ===== LoRA 参数 =====
    ui::ansi(std::cout, ui::BOLD);
    ui::ansi(std::cout, ui::CYAN) << "LoRA 参数 (自动切换 Python):";
    ui::reset(std::cout) << std::endl;
    printParamLine("--lora <路径>", "LoRA 适配器路径");
    printParamLine("--custom <配置>", "自定义模型配置");
    printParamLine("--dtype_config <配置>", "数据类型配置文件");
    printParamLine("--ori", "使用原始权重 (禁用量化)");
    std::cout << std::endl;
    
    // ===== 模板 / 工具调用 =====
    ui::ansi(std::cout, ui::BOLD);
    ui::ansi(std::cout, ui::CYAN) << "模板 / 工具调用:";
    ui::reset(std::cout) << std::endl;
    printParamLine("--chat_template <模板>", "对话模板 (覆盖自动检测)");
    printParamLine("--tool_call_parser <类型>", "工具调用解析器类型");
    printParamLine("--enable_thinking", "启用思考模式 (<think>标签)");
    printParamLine("--think", "Python 后端思考模式");
    printParamLine("--hide_input", "隐藏输入内容 (隐私保护)");
    std::cout << std::endl;
    
    // ===== 开发 / 调试 =====
    ui::ansi(std::cout, ui::BOLD);
    ui::ansi(std::cout, ui::CYAN) << "开发 / 调试:";
    ui::reset(std::cout) << std::endl;
    printParamLine("--dev_mode", "开发模式 (详细日志)");
    printParamLine("-v, --version", "显示版本信息");
    printParamLine("-h, --help", "显示帮助信息");
    std::cout << std::endl;
    
    // ===== 示例 =====
    ui::ansi(std::cout, ui::BOLD);
    ui::ansi(std::cout, ui::CYAN) << "示例:";
    ui::reset(std::cout) << std::endl;
    printExampleLine("run", "D:\\Models\\Qwen2.5-7B", "--device cuda");
    printExampleLine("run", "D:\\Models\\Qwen2.5-7B", "--lora ./lora");
    printExampleLine("serve", "D:\\Models\\Qwen2.5-7B", "--port 8080 --batch 4");
    printExampleLine("serve", "D:\\Models\\Qwen2.5-7B", "--cuda_embedding --kv_cache_limit 8G");
    printExampleLine("webui", "D:\\Models\\Qwen2.5-7B", "--port 1616");
    printExampleLine("download", "Qwen/Qwen2.5-7B-Instruct");
    std::cout << std::endl;
    
    ui::ansi(std::cout, ui::DIM);
    std::cout << "子命令帮助: ftllm <command> --help | 简要帮助: h 或 ?";
    ui::reset(std::cout) << std::endl;
}

// ============================================================================
// 主函数
// ============================================================================

int main(int argc, char **argv) {
    initWindowsConsole();

    // 内部参数：强制进入 REPL（用于 PowerShell 宿主重启、自定义快捷方式等）
    for (int i = 1; i < argc; i++) {
        if (std::string(argv[i]) == FTLLM_FLAG_REPL) {
            keepConsoleOpenUntilClose();
            return 0;
        }
    }

    // 无参数时显示帮助（双击运行时避免窗口一闪而过）
    if (argc == 1) {
        if (shouldPauseAfterHelp()) {
            // 默认使用 PowerShell 作为宿主，以获得更好的美化效果（颜色/分隔线/交互一致性）。
            if (!hasEnvironmentVariable(FTLLM_ENV_PSHOSTED)) {
                const bool launched = tryLaunchPowerShellHostAndRunFtllm({FTLLM_FLAG_REPL});
                if (launched) return 0;
                printStatusWarn("PowerShell 启动失败", "将使用当前控制台宿主继续");
            }
            keepConsoleOpenUntilClose();
            return 0;
        }

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
                if (shouldPauseAfterHelp()) {
                    keepConsoleOpenUntilClose();
                }
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
    
    // Python 专用命令（使用统一命令表检测）
    const CommandDef* cmdDef = findCommandDef(command);
    if (cmdDef && cmdDef->defaultBackend == Backend::Python) {
        usePython = true;
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
        const std::string cmdLower = toLowerCopy(command);
        const bool canFallbackToPython = (
            cmdLower == "serve" || cmdLower == "server" || cmdLower == "api" ||
            cmdLower == "webui" || cmdLower == "web"
        );

        // bench/quant 等仅有原生实现：不要因为“Python-only 参数”误切到 Python
        if (canFallbackToPython) {
            const char* const* pythonOnlyArgs = nullptr;

            // serve/server/api: C++ apiserver 已支持 --host / --port / --batch/--max_batch / --cuda_embedding / --model_name / --device / --nt 等
            static const char* pythonOnlyArgsServe[] = {
                "--api_key",            // Python server 支持
                "--think",              // Python server 支持
                "--hide_input",         // Python server 支持
                "--dev_mode",           // Python server 支持

                // ===== MOE / 设备配置 =====
                "--moe_device",
                "--moe_dtype",
                "--moe_experts",

                // ===== CUDA / 内存 / 缓存 =====
                "--kv_cache_limit",
                "--cache_history",
                "--cache_fast",
                "--cache_dir",

                // ===== 模型特性 =====
                "--enable_thinking",
                "--cuda_shared_expert",
                "--cuda_se",
                "--enable_amx",
                "--amx",

                // ===== LoRA / 自定义 =====
                "--lora",
                "--custom",
                "--dtype_config",
                "--ori",

                // ===== 模板 / 解析 =====
                "--tool_call_parser",
                "--chat_template",

                nullptr
            };

            // webui/web: C++ webui 参数面很窄，Python webui 额外支持很多运行参数
            static const char* pythonOnlyArgsWebui[] = {
                "--cuda_embedding",
                "--kv_cache_limit",
                "--max_batch",
                "--max_token",          // Python webui 子命令参数
                "--think",              // Python webui 子命令参数

                "--moe_device",
                "--moe_dtype",
                "--moe_experts",
                "--cache_history",
                "--cache_fast",
                "--cache_dir",

                "--enable_thinking",
                "--cuda_shared_expert",
                "--cuda_se",
                "--enable_amx",
                "--amx",

                "--lora",
                "--custom",
                "--dtype_config",
                "--ori",

                "--tool_call_parser",
                "--chat_template",

                nullptr
            };

            if (cmdLower == "serve" || cmdLower == "server" || cmdLower == "api") {
                pythonOnlyArgs = pythonOnlyArgsServe;
            } else {
                pythonOnlyArgs = pythonOnlyArgsWebui;
            }

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
                std::cout << "[提示] 以下参数在当前 C++ 原生子命令中不支持:" << std::endl;
                for (const auto& arg : unsupportedArgs) {
                    std::cout << "  - " << arg << std::endl;
                }
                std::cout << "[自动切换] 使用 Python 后端以支持这些参数" << std::endl;
                usePython = true;
            }
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
    
    // 查找命令定义（复用前面已查找的 cmdDef）
    if (cmdDef && cmdDef->nativeExe) {
        // 执行 C++ 原生程序
        return executeNativeProgram(cmdDef->nativeExe, argc, argv, commandIndex + 1);
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
        const std::string modelInput = command;
        const std::vector<std::string> extraArgs = collectArgsAfterIndex(argc, argv, commandIndex + 1);
        return handleModelInputWithChoiceMenu(modelInput, extraArgs, false);
    }
    
    // 未知命令，尝试 Python 后端
    std::cout << "[提示] 未知命令 '" << command << "'，尝试 Python 后端..." << std::endl;
    return executePythonBackend(argc, argv);
}
