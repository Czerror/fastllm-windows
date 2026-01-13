//
// console.h - 统一控制台输出模块
// 提供跨平台的 ANSI 颜色支持、Unicode 图标、格式化输出等功能
//

#ifndef FASTLLM_CONSOLE_H
#define FASTLLM_CONSOLE_H

#include <iostream>
#include <string>
#include <sstream>
#include <iomanip>
#include <cstdlib>

#ifdef _WIN32
#include <windows.h>
#endif

namespace fastllm {
namespace console {

// ============================================================================
// ANSI 样式代码
// ============================================================================
static constexpr const char* RESET = "\x1b[0m";
static constexpr const char* BOLD = "\x1b[1m";
static constexpr const char* DIM = "\x1b[2m";
static constexpr const char* UNDERLINE = "\x1b[4m";
static constexpr const char* BLINK = "\x1b[5m";
static constexpr const char* REVERSE = "\x1b[7m";

// ============================================================================
// 前景色
// ============================================================================
static constexpr const char* BLACK = "\x1b[30m";
static constexpr const char* RED = "\x1b[31m";
static constexpr const char* GREEN = "\x1b[32m";
static constexpr const char* YELLOW = "\x1b[33m";
static constexpr const char* BLUE = "\x1b[34m";
static constexpr const char* MAGENTA = "\x1b[35m";
static constexpr const char* CYAN = "\x1b[36m";
static constexpr const char* WHITE = "\x1b[37m";

// 亮色前景
static constexpr const char* BRIGHT_BLACK = "\x1b[90m";   // 灰色
static constexpr const char* BRIGHT_RED = "\x1b[91m";
static constexpr const char* BRIGHT_GREEN = "\x1b[92m";
static constexpr const char* BRIGHT_YELLOW = "\x1b[93m";
static constexpr const char* BRIGHT_BLUE = "\x1b[94m";
static constexpr const char* BRIGHT_MAGENTA = "\x1b[95m";
static constexpr const char* BRIGHT_CYAN = "\x1b[96m";
static constexpr const char* BRIGHT_WHITE = "\x1b[97m";

// ============================================================================
// 背景色
// ============================================================================
static constexpr const char* BG_BLACK = "\x1b[40m";
static constexpr const char* BG_RED = "\x1b[41m";
static constexpr const char* BG_GREEN = "\x1b[42m";
static constexpr const char* BG_YELLOW = "\x1b[43m";
static constexpr const char* BG_BLUE = "\x1b[44m";
static constexpr const char* BG_MAGENTA = "\x1b[45m";
static constexpr const char* BG_CYAN = "\x1b[46m";
static constexpr const char* BG_WHITE = "\x1b[47m";

// ============================================================================
// Unicode 图标
// ============================================================================
static constexpr const char* ICON_CHECK = "\xe2\x9c\x93";      // ✓
static constexpr const char* ICON_CROSS = "\xe2\x9c\x97";      // ✗
static constexpr const char* ICON_ARROW = "\xe2\x86\x92";      // →
static constexpr const char* ICON_BULLET = "\xe2\x97\x8f";     // ●
static constexpr const char* ICON_CIRCLE = "\xe2\x97\x8b";     // ○
static constexpr const char* ICON_PLAY = "\xe2\x96\xb6";       // ▶
static constexpr const char* ICON_STOP = "\xe2\x96\xa0";       // ■
static constexpr const char* ICON_STAR = "\xe2\x98\x85";       // ★
static constexpr const char* ICON_INFO = "\xe2\x84\xb9";       // ℹ
static constexpr const char* ICON_WARN = "\xe2\x9a\xa0";       // ⚠
static constexpr const char* ICON_GEAR = "\xe2\x9a\x99";       // ⚙

// ============================================================================
// Box Drawing 边框字符
// ============================================================================
// 单线边框
static constexpr const char* BOX_H = "\xe2\x94\x80";           // ─
static constexpr const char* BOX_V = "\xe2\x94\x82";           // │
static constexpr const char* BOX_TL = "\xe2\x94\x8c";          // ┌
static constexpr const char* BOX_TR = "\xe2\x94\x90";          // ┐
static constexpr const char* BOX_BL = "\xe2\x94\x94";          // └
static constexpr const char* BOX_BR = "\xe2\x94\x98";          // ┘
static constexpr const char* BOX_T = "\xe2\x94\xac";           // ┬
static constexpr const char* BOX_B = "\xe2\x94\xb4";           // ┴
static constexpr const char* BOX_L = "\xe2\x94\x9c";           // ├
static constexpr const char* BOX_R = "\xe2\x94\xa4";           // ┤
static constexpr const char* BOX_X = "\xe2\x94\xbc";           // ┼

// 双线边框
static constexpr const char* BOX2_H = "\xe2\x95\x90";          // ═
static constexpr const char* BOX2_V = "\xe2\x95\x91";          // ║
static constexpr const char* BOX2_TL = "\xe2\x95\x94";         // ╔
static constexpr const char* BOX2_TR = "\xe2\x95\x97";         // ╗
static constexpr const char* BOX2_BL = "\xe2\x95\x9a";         // ╚
static constexpr const char* BOX2_BR = "\xe2\x95\x9d";         // ╝

// ============================================================================
// 光标控制
// ============================================================================
static constexpr const char* CURSOR_HIDE = "\x1b[?25l";
static constexpr const char* CURSOR_SHOW = "\x1b[?25h";
static constexpr const char* CLEAR_LINE = "\x1b[2K\r";
static constexpr const char* CLEAR_SCREEN = "\x1b[2J\x1b[H";
static constexpr const char* CURSOR_UP = "\x1b[A";
static constexpr const char* CURSOR_DOWN = "\x1b[B";
static constexpr const char* CURSOR_SAVE = "\x1b[s";
static constexpr const char* CURSOR_RESTORE = "\x1b[u";

// ============================================================================
// 进度动画帧 (Spinner)
// ============================================================================
static constexpr const char* SPINNER_FRAMES[] = {
    "\xe2\xa0\x8b", // ⠋
    "\xe2\xa0\x99", // ⠙
    "\xe2\xa0\xb9", // ⠹
    "\xe2\xa0\xb8", // ⠸
    "\xe2\xa0\xbc", // ⠼
    "\xe2\xa0\xb4", // ⠴
    "\xe2\xa0\xa6", // ⠦
    "\xe2\xa0\xa7", // ⠧
    "\xe2\xa0\x87", // ⠇
    "\xe2\xa0\x8f"  // ⠏
};
static constexpr int SPINNER_FRAME_COUNT = 10;

// ============================================================================
// 预定义边框线
// ============================================================================
static constexpr const char* LINE_DOUBLE = "════════════════════════════════════════════════════════════";
static constexpr const char* LINE_SINGLE = "────────────────────────────────────────────────────────────";
static constexpr const char* LINE_THIN = "────────────────────────────────────────";

// ============================================================================
// 全局 ANSI 支持状态
// ============================================================================
inline bool& getAnsiEnabled() {
    static bool g_ansi = false;
    return g_ansi;
}

inline bool isAnsiEnabled() {
    return getAnsiEnabled();
}

// ============================================================================
// 初始化控制台（启用 ANSI 支持和 UTF-8）
// ============================================================================
inline void init() {
#ifdef _WIN32
    // 设置控制台输出为 UTF-8
    SetConsoleOutputCP(CP_UTF8);
    SetConsoleCP(CP_UTF8);
    
    // 检查父进程是否已启用 ANSI (通过环境变量)
    const char* env = std::getenv("FTLLM_ANSI");
    if (env && std::string(env) == "1") {
        getAnsiEnabled() = true;
    } else {
        // 尝试自行启用 ANSI
        HANDLE hOut = GetStdHandle(STD_OUTPUT_HANDLE);
        if (hOut != INVALID_HANDLE_VALUE) {
            DWORD mode = 0;
            if (GetConsoleMode(hOut, &mode)) {
                if (SetConsoleMode(hOut, mode | ENABLE_VIRTUAL_TERMINAL_PROCESSING)) {
                    getAnsiEnabled() = true;
                }
            }
        }
    }
#else
    // Unix 系统通常默认支持
    getAnsiEnabled() = true;
#endif
}

// ============================================================================
// ANSI 输出辅助函数
// ============================================================================
inline std::ostream& ansi(std::ostream& os, const char* code) {
    if (getAnsiEnabled() && code) os << code;
    return os;
}

inline std::ostream& reset(std::ostream& os) {
    if (getAnsiEnabled()) os << RESET;
    return os;
}

// ============================================================================
// 字符串显示宽度计算（正确处理中文和 ANSI 转义序列）
// ============================================================================
inline int getDisplayWidth(const std::string& text) {
    int width = 0;
    bool inEscape = false;
    const unsigned char* p = reinterpret_cast<const unsigned char*>(text.c_str());
    while (*p) {
        if (inEscape) {
            if (*p == 'm') inEscape = false;
            ++p;
        } else if (*p == '\x1b') {
            inEscape = true;
            ++p;
        } else if (*p >= 0x80) {
            if ((*p & 0xE0) == 0xC0) {
                width += 1;
                p += 2;
            } else if ((*p & 0xF0) == 0xE0) {
                unsigned int codepoint = ((p[0] & 0x0F) << 12) | ((p[1] & 0x3F) << 6) | (p[2] & 0x3F);
                if ((codepoint >= 0x4E00 && codepoint <= 0x9FFF) ||
                    (codepoint >= 0x3400 && codepoint <= 0x4DBF) ||
                    (codepoint >= 0xF900 && codepoint <= 0xFAFF) ||
                    (codepoint >= 0x3000 && codepoint <= 0x303F) ||
                    (codepoint >= 0xFF00 && codepoint <= 0xFFEF) ||
                    (codepoint >= 0xAC00 && codepoint <= 0xD7AF)) {
                    width += 2;
                } else {
                    width += 1;
                }
                p += 3;
            } else if ((*p & 0xF8) == 0xF0) {
                width += 2;
                p += 4;
            } else {
                ++p;
            }
        } else {
            width += 1;
            ++p;
        }
    }
    return width;
}

// ============================================================================
// 基础打印函数
// ============================================================================

// 打印带样式的文本
inline void printStyled(const char* color, const std::string& text, bool newline = true) {
    ansi(std::cout, color) << text;
    reset(std::cout);
    if (newline) std::cout << std::endl;
}

// 打印带图标的状态消息
inline void printStatusIcon(const char* icon, const char* color, const std::string& msg) {
    ansi(std::cout, color) << icon << " ";
    reset(std::cout) << msg << std::endl;
}

// 成功消息 (绿色 ✓)
inline void printSuccess(const std::string& msg) {
    if (getAnsiEnabled()) {
        std::cout << GREEN << ICON_CHECK << " " << RESET << msg << std::endl;
    } else {
        std::cout << "[OK] " << msg << std::endl;
    }
}

// 错误消息 (红色 ✗)
inline void printError(const std::string& msg) {
    if (getAnsiEnabled()) {
        std::cout << RED << ICON_CROSS << " " << RESET << msg << std::endl;
    } else {
        std::cout << "[ERROR] " << msg << std::endl;
    }
}

// 信息消息 (青色 ℹ)
inline void printInfo(const std::string& msg) {
    if (getAnsiEnabled()) {
        std::cout << CYAN << ICON_INFO << " " << RESET << msg << std::endl;
    } else {
        std::cout << "[INFO] " << msg << std::endl;
    }
}

// 警告消息 (黄色 ⚠)
inline void printWarning(const std::string& msg) {
    if (getAnsiEnabled()) {
        std::cout << YELLOW << ICON_WARN << " " << RESET << msg << std::endl;
    } else {
        std::cout << "[WARN] " << msg << std::endl;
    }
}

// 箭头消息 (蓝色 →)
inline void printArrow(const std::string& msg) {
    printStatusIcon(ICON_ARROW, BRIGHT_BLUE, msg);
}

// 项目符号消息 (灰色 ●)
inline void printBullet(const std::string& msg) {
    printStatusIcon(ICON_BULLET, DIM, msg);
}

// 配置项 (键值对)
inline void printConfig(const std::string& key, const std::string& value) {
    if (getAnsiEnabled()) {
        std::cout << "  " << DIM << key << RESET << ": " << BRIGHT_CYAN << value << RESET << std::endl;
    } else {
        std::cout << "  " << key << ": " << value << std::endl;
    }
}

// 章节标题
inline void printHeader(const std::string& title) {
    if (getAnsiEnabled()) {
        std::cout << std::endl << BOLD << CYAN << ICON_PLAY << " " << title << RESET << std::endl;
        std::cout << LINE_THIN << std::endl;
    } else {
        std::cout << std::endl << "=== " << title << " ===" << std::endl;
    }
}

// 分隔线
inline void printRule(const char* title = nullptr) {
    std::cout << LINE_DOUBLE << std::endl;
    if (title && *title) {
        ansi(std::cout, BOLD);
        ansi(std::cout, CYAN) << title;
        reset(std::cout) << std::endl;
        std::cout << LINE_SINGLE << std::endl;
    }
}

// 键值对行
inline void printKV(const std::string& key, const std::string& value) {
    std::cout << "    ";
    ansi(std::cout, DIM) << key;
    reset(std::cout) << ": " << value << std::endl;
}

// ============================================================================
// 状态消息（带状态标记）
// ============================================================================
enum class StatusType { Ok, Warn, Err };

static constexpr const char* STATUS_OK = "[√]";
static constexpr const char* STATUS_WARN = "[!]";
static constexpr const char* STATUS_ERR = "[×]";

inline void printStatus(StatusType type, const std::string& label, const std::string& detail = {}) {
    const char* icon = nullptr;
    const char* color = nullptr;
    switch (type) {
        case StatusType::Ok:   icon = STATUS_OK;   color = GREEN;  break;
        case StatusType::Warn: icon = STATUS_WARN; color = YELLOW; break;
        case StatusType::Err:  icon = STATUS_ERR;  color = RED;    break;
    }
    if (getAnsiEnabled()) std::cout << color << icon << RESET;
    else std::cout << icon;
    std::cout << " " << label;
    if (!detail.empty()) std::cout << ": " << detail;
    std::cout << std::endl;
}

inline void printStatusOk(const std::string& label, const std::string& detail = {}) {
    printStatus(StatusType::Ok, label, detail);
}
inline void printStatusWarn(const std::string& label, const std::string& detail = {}) {
    printStatus(StatusType::Warn, label, detail);
}
inline void printStatusErr(const std::string& label, const std::string& detail = {}) {
    printStatus(StatusType::Err, label, detail);
}

// ============================================================================
// Box Drawing 边框绘制
// ============================================================================

// 单线边框
inline void printBoxTop(int width = 60) {
    std::cout << BOX_TL;
    for (int i = 0; i < width - 2; ++i) std::cout << BOX_H;
    std::cout << BOX_TR << std::endl;
}
inline void printBoxBottom(int width = 60) {
    std::cout << BOX_BL;
    for (int i = 0; i < width - 2; ++i) std::cout << BOX_H;
    std::cout << BOX_BR << std::endl;
}
inline void printBoxLine(const std::string& text, int width = 60) {
    std::cout << BOX_V << " " << text;
    int displayWidth = getDisplayWidth(text);
    int pad = width - 4 - displayWidth;
    for (int i = 0; i < pad; ++i) std::cout << " ";
    std::cout << " " << BOX_V << std::endl;
}
inline void printBoxSeparator(int width = 60) {
    std::cout << BOX_L;
    for (int i = 0; i < width - 2; ++i) std::cout << BOX_H;
    std::cout << BOX_R << std::endl;
}

// 双线边框
inline void printBox2Top(int width = 60) {
    std::cout << BOX2_TL;
    for (int i = 0; i < width - 2; ++i) std::cout << BOX2_H;
    std::cout << BOX2_TR << std::endl;
}
inline void printBox2Bottom(int width = 60) {
    std::cout << BOX2_BL;
    for (int i = 0; i < width - 2; ++i) std::cout << BOX2_H;
    std::cout << BOX2_BR << std::endl;
}
inline void printBox2Line(const std::string& text, int width = 60) {
    std::cout << BOX2_V << " " << text;
    int displayWidth = getDisplayWidth(text);
    int pad = width - 4 - displayWidth;
    for (int i = 0; i < pad; ++i) std::cout << " ";
    std::cout << " " << BOX2_V << std::endl;
}

// ============================================================================
// 进度条
// ============================================================================
inline void printProgressBar(double progress, int width = 40, const char* label = nullptr) {
    if (label) {
        ansi(std::cout, DIM) << label << " ";
        reset(std::cout);
    }
    int filled = static_cast<int>(progress * width);
    std::cout << "[";
    ansi(std::cout, GREEN);
    for (int i = 0; i < filled; ++i) std::cout << "#";
    reset(std::cout);
    ansi(std::cout, DIM);
    for (int i = filled; i < width; ++i) std::cout << "-";
    reset(std::cout);
    std::cout << "] " << static_cast<int>(progress * 100) << "%" << std::endl;
}

// 单行进度更新 (覆盖当前行)
inline void updateProgressInline(double progress, int width = 40, const char* label = nullptr) {
    if (getAnsiEnabled()) std::cout << CLEAR_LINE;
    if (label) {
        ansi(std::cout, DIM) << label << " ";
        reset(std::cout);
    }
    int filled = static_cast<int>(progress * width);
    std::cout << "[";
    ansi(std::cout, GREEN);
    for (int i = 0; i < filled; ++i) std::cout << "#";
    reset(std::cout);
    ansi(std::cout, DIM);
    for (int i = filled; i < width; ++i) std::cout << "-";
    reset(std::cout);
    std::cout << "] " << static_cast<int>(progress * 100) << "%" << std::flush;
}

// ============================================================================
// 日志级别标记输出
// ============================================================================
inline void logInfo(const std::string& tag, const std::string& msg) {
    if (getAnsiEnabled()) {
        std::cout << CYAN << "[" << tag << "]" << RESET << " " << msg << std::endl;
    } else {
        std::cout << "[" << tag << "] " << msg << std::endl;
    }
}

inline void logDebug(const std::string& tag, const std::string& msg) {
    if (getAnsiEnabled()) {
        std::cout << DIM << "[" << tag << "] " << msg << RESET << std::endl;
    } else {
        std::cout << "[" << tag << "] " << msg << std::endl;
    }
}

inline void logWarn(const std::string& tag, const std::string& msg) {
    if (getAnsiEnabled()) {
        std::cout << YELLOW << "[" << tag << "]" << RESET << " " << msg << std::endl;
    } else {
        std::cout << "[" << tag << "] " << msg << std::endl;
    }
}

inline void logError(const std::string& tag, const std::string& msg) {
    if (getAnsiEnabled()) {
        std::cout << RED << "[" << tag << "]" << RESET << " " << msg << std::endl;
    } else {
        std::cout << "[" << tag << "] " << msg << std::endl;
    }
}

} // namespace console
} // namespace fastllm

#endif // FASTLLM_CONSOLE_H
