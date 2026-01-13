/**
 * ftllm - FastLLM 统一命令行入口
 * 
 * 默认使用 C++ 原生程序 (apiserver/webui/benchmark 等)
 * 添加 -py 参数时使用 Python 后端
 * 添加到 PATH 后可直接使用: ftllm run/serve/webui 等
 */

#include <iostream>
#include <string>
#include <sstream>
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

// 使用统一的控制台模块和帮助文本
#include "utils/console.h"
#include "utils/help_text.h"
namespace ui = fastllm::console;
namespace help = fastllm::help;

static constexpr int DEFAULT_SERVE_PORT = 8080;
static constexpr const char* REPL_PROMPT = "ftllm> ";

// 版本号引用统一定义
#define FTLLM_VERSION help::PROGRAM_VERSION

// 引用 console 模块的常量
static constexpr const char* UI_LINE = fastllm::console::LINE_DOUBLE;
static constexpr const char* UI_THIN_LINE = fastllm::console::LINE_SINGLE;
static constexpr const char* UI_STATUS_OK = fastllm::console::STATUS_OK;
static constexpr const char* UI_STATUS_WARN = fastllm::console::STATUS_WARN;
static constexpr const char* UI_STATUS_ERR = fastllm::console::STATUS_ERR;

// g_hasAnsi 作为对统一模块的引用
#define g_hasAnsi fastllm::console::getAnsiEnabled()

// 直接使用 console 模块的函数（创建本地别名）
using ui::printSuccess;
using ui::printError;
using ui::printInfo;
using ui::printWarning;
using ui::printArrow;
using ui::printBullet;
using ui::printRule;
using ui::printKV;
using ui::printConfig;
using ui::printHeader;
using ui::printStatus;
using ui::printStatusOk;
using ui::printStatusWarn;
using ui::printStatusErr;
using ui::printStyled;
using ui::StatusType;

// 保留 ftllm 特有的扩展函数（使用 console 模块基础设施）

// 计算字符串的显示宽度（正确处理中文和 ANSI 转义序列）
// 直接使用 ui::getDisplayWidth

// 绘制带边框的框（Box Drawing）
static void printBoxTop(int width = 60) {
    std::cout << ui::BOX_TL;
    for (int i = 0; i < width - 2; ++i) std::cout << ui::BOX_H;
    std::cout << ui::BOX_TR << std::endl;
}
static void printBoxBottom(int width = 60) {
    std::cout << ui::BOX_BL;
    for (int i = 0; i < width - 2; ++i) std::cout << ui::BOX_H;
    std::cout << ui::BOX_BR << std::endl;
}
static void printBoxLine(const std::string& text, int width = 60) {
    std::cout << ui::BOX_V << " " << text;
    int displayWidth = ui::getDisplayWidth(text);
    int pad = width - 4 - displayWidth;
    for (int i = 0; i < pad; ++i) std::cout << " ";
    std::cout << " " << ui::BOX_V << std::endl;
}
static void printBoxSeparator(int width = 60) {
    std::cout << ui::BOX_L;
    for (int i = 0; i < width - 2; ++i) std::cout << ui::BOX_H;
    std::cout << ui::BOX_R << std::endl;
}

// 双线边框版本
static void printBox2Top(int width = 60) {
    std::cout << ui::BOX2_TL;
    for (int i = 0; i < width - 2; ++i) std::cout << ui::BOX2_H;
    std::cout << ui::BOX2_TR << std::endl;
}
static void printBox2Bottom(int width = 60) {
    std::cout << ui::BOX2_BL;
    for (int i = 0; i < width - 2; ++i) std::cout << ui::BOX2_H;
    std::cout << ui::BOX2_BR << std::endl;
}

static void printBox2Line(const std::string& text, int width = 60) {
    std::cout << ui::BOX2_V << " " << text;
    int displayWidth = ui::getDisplayWidth(text);
    int pad = width - 4 - displayWidth;
    for (int i = 0; i < pad; ++i) std::cout << " ";
    std::cout << " " << ui::BOX2_V << std::endl;
}

// 单行进度更新 (覆盖当前行)
static void updateProgressInline(double progress, int width = 40, const char* label = nullptr) {
    if (g_hasAnsi) std::cout << ui::CLEAR_LINE;
    if (label) {
        ui::ansi(std::cout, ui::DIM) << label << " ";
        ui::reset(std::cout);
    }
    int filled = static_cast<int>(progress * width);
    std::cout << "[";
    ui::ansi(std::cout, ui::GREEN);
    for (int i = 0; i < filled; ++i) std::cout << "█";
    ui::reset(std::cout);
    ui::ansi(std::cout, ui::DIM);
    for (int i = filled; i < width; ++i) std::cout << "░";
    ui::reset(std::cout);
    std::cout << "] " << static_cast<int>(progress * 100) << "%" << std::flush;
}

// 标题区域显示
static void printSectionTitle(const std::string& title) {
    std::cout << std::endl;
    ui::ansi(std::cout, ui::BOLD);
    ui::ansi(std::cout, ui::CYAN) << ui::ICON_PLAY << " " << title;
    ui::reset(std::cout) << std::endl;
    std::cout << UI_THIN_LINE << std::endl;
}

static void printLaunchConfig(const std::string& program, const std::string& backend, const std::string& modelPath, const std::vector<std::string>& /*args*/, bool background) {
    // 使用精美边框显示启动配置
    printBox2Top(60);
    
    std::ostringstream oss;
    ui::ansi(oss, ui::BOLD);
    ui::ansi(oss, ui::CYAN) << ui::ICON_PLAY << " 启动";
    ui::reset(oss);
    oss << ": " << program << " (" << backend;
    if (background) oss << ", 后台";
    oss << ")";
    printBox2Line(oss.str(), 60);
    
    if (!modelPath.empty()) {
        std::ostringstream mss;
        mss << ui::ICON_GEAR << " 模型: " << modelPath;
        printBox2Line(mss.str(), 60);
    }
    
    printBox2Bottom(60);
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
        printSuccess("后台服务已启动 (输入 stop 可停止)");
    } else if (exitCode == 0) {
        printSuccess("执行完成");
    } else {
        printWarning("进程已退出 (code: " + std::to_string(exitCode) + ")");
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

// ============================================================================
// 进程优先级与功耗管理
// ============================================================================

// Power Throttling 结构体（Windows 10 1709+ / SDK 可能未包含）
struct FTLLM_PROCESS_POWER_THROTTLING_STATE {
    DWORD Version;
    DWORD ControlMask;
    DWORD StateMask;
};
static const DWORD FTLLM_POWER_THROTTLING_VERSION = 1;
static const DWORD FTLLM_POWER_THROTTLING_EXECUTION_SPEED = 0x1;
static const int FTLLM_ProcessPowerThrottling = 4; // PROCESS_INFORMATION_CLASS 枚举值

// SetProcessInformation 函数指针类型（动态加载）
typedef BOOL (WINAPI *PFN_SetProcessInformation)(
    HANDLE hProcess,
    int ProcessInformationClass,
    LPVOID ProcessInformation,
    DWORD ProcessInformationSize
);

/**
 * 禁用指定进程的 Power Throttling（后台/最小化节能限频）。
 * 使用动态加载以兼容旧版 Windows。
 */
static void disablePowerThrottling(HANDLE hProcess) {
    static PFN_SetProcessInformation pfnSetProcessInformation = nullptr;
    static bool pfnChecked = false;

    if (!pfnChecked) {
        pfnChecked = true;
        HMODULE hKernel32 = GetModuleHandleW(L"kernel32.dll");
        if (hKernel32) {
            pfnSetProcessInformation = reinterpret_cast<PFN_SetProcessInformation>(
                GetProcAddress(hKernel32, "SetProcessInformation")
            );
        }
    }

    if (pfnSetProcessInformation) {
        FTLLM_PROCESS_POWER_THROTTLING_STATE throttleState = {};
        throttleState.Version = FTLLM_POWER_THROTTLING_VERSION;
        throttleState.ControlMask = FTLLM_POWER_THROTTLING_EXECUTION_SPEED;
        throttleState.StateMask = 0; // 0 = 禁用节流
        pfnSetProcessInformation(hProcess, FTLLM_ProcessPowerThrottling,
                                 &throttleState, sizeof(throttleState));
    }
}

/**
 * 提升当前进程优先级并禁用 Power Throttling（后台/最小化节能）。
 * 效果：即使窗口最小化，推理性能也不会因系统节能策略而下降。
 */
static void boostProcessPriority() {
    // 1) 设置进程为高优先级（HIGH_PRIORITY_CLASS）
    //    - 比默认 NORMAL 高，但低于 REALTIME，不影响系统稳定性
    const HANDLE hProcess = GetCurrentProcess();
    if (!SetPriorityClass(hProcess, HIGH_PRIORITY_CLASS)) {
        // 降级：尝试 ABOVE_NORMAL
        SetPriorityClass(hProcess, ABOVE_NORMAL_PRIORITY_CLASS);
    }

    // 2) 禁用 Power Throttling（Windows 10 1709+ / Windows 11）
    disablePowerThrottling(hProcess);
}

/**
 * 提升子进程优先级。在 CreateProcessW 成功后调用。
 */
static void boostChildProcessPriority(HANDLE hChildProcess) {
    if (hChildProcess == nullptr || hChildProcess == INVALID_HANDLE_VALUE) return;

    // 设置子进程为高优先级
    if (!SetPriorityClass(hChildProcess, HIGH_PRIORITY_CLASS)) {
        SetPriorityClass(hChildProcess, ABOVE_NORMAL_PRIORITY_CLASS);
    }

    // 禁用 Power Throttling
    disablePowerThrottling(hChildProcess);
}
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

    // 提升子进程优先级，防止最小化时性能下降
    boostChildProcessPriority(pi.hProcess);

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
 * 使用统一的 console 模块初始化，并设置环境变量供子进程使用
 */
void initWindowsConsole() {
    // 使用统一模块初始化
    fastllm::console::init();
    
    // 设置环境变量，供子进程检测 ANSI 支持
    if (g_hasAnsi) {
        _putenv_s("FTLLM_ANSI", "1");
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
// 使用统一帮助定义 (来自 help_text.h)
// ============================================================================
// 本地 Backend 枚举用于兼容现有代码
enum class Backend { Native, Python };

// 使用统一命令定义类型
using CommandDef = help::CommandDef;

// 从统一定义创建本地引用（保持向后兼容）
static const auto& ALL_COMMANDS = help::COMMANDS;
static constexpr size_t NUM_COMMANDS = help::NUM_COMMANDS;

// 辅助函数：将统一定义转换为本地 Backend 枚举
static Backend getBackend(const help::CommandDef& cmd) {
    return cmd.is_native ? Backend::Native : Backend::Python;
}

// ============================================================================
// 统一帮助输出函数
// ============================================================================

// 打印命令列表（带颜色）
static void printCommandList(bool is_native, const char* title) {
    ui::ansi(std::cout, ui::BOLD);
    ui::ansi(std::cout, ui::CYAN) << title;
    ui::reset(std::cout) << std::endl;
    
    for (const auto& cmd : ALL_COMMANDS) {
        if (cmd.is_native != is_native) continue;
        
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
        std::cout << cmd.desc << std::endl;
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

// 打印组标题
static void printGroupTitle(const char* title) {
    ui::ansi(std::cout, ui::BOLD);
    ui::ansi(std::cout, ui::CYAN) << title;
    ui::reset(std::cout) << std::endl;
}

// 打印单个参数组（使用统一定义）
static void printParamGroup(const help::ParamGroup& group) {
    printGroupTitle(group.title);
    for (const auto& param : group.params) {
        printParamLine(param.name, param.desc);
    }
    std::cout << std::endl;
}

// 打印所有参数组
static void printAllParamGroups() {
    for (const auto& group : help::PARAM_GROUPS) {
        printParamGroup(group);
    }
}

// 打印模型格式
static void printModelFormats() {
    printGroupTitle("模型格式 (自动识别)");
    for (const auto& fmt : help::MODEL_FORMATS) {
        std::cout << "  ";
        ui::ansi(std::cout, ui::YELLOW) << fmt.format;
        ui::reset(std::cout);
        for (size_t i = strlen(fmt.format); i < 28; i++) std::cout << ' ';
        std::cout << fmt.desc << std::endl;
    }
    std::cout << std::endl;
}

// 打印所有示例
static void printAllExamples() {
    printGroupTitle("示例");
    for (const auto& ex : help::EXAMPLES) {
        printExampleLine(ex.cmd, ex.model, ex.args);
    }
    std::cout << std::endl;
}

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

    // 复用既有逻辑（参数补 -p 等）
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
    if (def && def->exe) {
        return executeNativeProgram(def->exe, static_cast<int>(argv2.size()), argv2.data(), 2);
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
                std::cout << "  <模型> [选项]          → 选择操作(run/serve/export)" << std::endl;
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
            // 使用统一帮助系统
            printCommandList(true, "命令 (C++ 原生程序):");
            printCommandList(false, "命令 (Python 后端):");
            
            // 模式切换
            printGroupTitle("模式切换");
            printParamLine("-py", "使用 Python 后端 (支持 LoRA 动态加载等)");
            std::cout << "  (自动)                        检测到 --lora / lora/ 目录时自动切换" << std::endl;
            std::cout << std::endl;
            
            // 模型格式
            printModelFormats();
            
            // 所有参数组
            printAllParamGroups();
            
            // 示例
            printAllExamples();
            
            std::cout << UI_THIN_LINE << std::endl;
            ui::ansi(std::cout, ui::DIM);
            std::cout << "  输入 ";
            ui::ansi(std::cout, ui::CYAN) << "h";
            ui::reset(std::cout);
            ui::ansi(std::cout, ui::DIM) << " 或 ";
            ui::ansi(std::cout, ui::CYAN) << "?";
            ui::reset(std::cout);
            ui::ansi(std::cout, ui::DIM) << " 查看简要帮助";
            ui::reset(std::cout) << std::endl;
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

    // 获取模型路径用于显示
    const std::string modelPath = !pathArgValue.empty() ? pathArgValue : positionalModelPath;

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

void Usage() {
    std::cout << "Usage: ftllm <command> [options] [model_path]" << std::endl;
    std::cout << std::endl;
    ui::ansi(std::cout, ui::BOLD);
    std::cout << "FastLLM - 高性能大语言模型推理引擎 (v" << help::PROGRAM_VERSION << ")";
    ui::reset(std::cout) << std::endl;
    std::cout << std::endl;
    
    printCommandList(true, "命令 (C++ 原生程序):");
    printCommandList(false, "命令 (Python 后端):");
    
    // 模式切换
    printGroupTitle("模式切换:");
    printParamLine("-py", "使用 Python 后端 (支持 LoRA 动态加载等)");
    std::cout << "  (自动)                        检测到 --lora / lora/ 目录时自动切换" << std::endl;
    std::cout << std::endl;
    
    // 模型格式
    printModelFormats();
    
    // 所有参数组
    printAllParamGroups();
    
    // 示例
    printAllExamples();
    
    ui::ansi(std::cout, ui::DIM);
    std::cout << "子命令帮助: ftllm <command> --help | 简要帮助: h 或 ?";
    ui::reset(std::cout) << std::endl;
}

// ============================================================================
// 主函数
// ============================================================================

int main(int argc, char **argv) {
    initWindowsConsole();
    boostProcessPriority(); // 提升主进程优先级，防止最小化时推理性能下降

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
    if (cmdDef && !cmdDef->is_native) {
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

            // serve/server/api: C++ apiserver 已支持 --host / --port / --batch/--max_batch / --cuda_embedding / --model_name / --device 等
            static const char* pythonOnlyArgsServe[] = {
                "--api_key",            // Python server 支持
                "--think",              // Python server 支持
                "--hide_input",         // Python server 支持
                "--dev_mode",           // Python server 支持

                // ===== MOE / 设备配置 =====
                // "--moe_device",      // C++ apiserver 已支持
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

                // "--moe_device",      // C++ webui 已支持
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
    if (cmdDef && cmdDef->exe) {
        // 执行 C++ 原生程序
        return executeNativeProgram(cmdDef->exe, argc, argv, commandIndex + 1);
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
