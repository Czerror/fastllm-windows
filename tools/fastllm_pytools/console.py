"""
console.py - 控制台美化输出模块

通过检测 FTLLM_ANSI 环境变量来决定是否启用 ANSI 彩色输出。
当通过 ftllm.exe 启动时，环境变量会自动设置。
"""

import os
import sys
from contextlib import contextmanager

# 检测 ANSI 支持
_ansi_enabled = os.environ.get("FTLLM_ANSI", "0") == "1"

# 如果未设置环境变量，尝试自行检测
if not _ansi_enabled and sys.platform != "win32":
    # Unix 系统默认支持
    _ansi_enabled = True
elif not _ansi_enabled and sys.platform == "win32":
    # Windows: 检查是否在支持 ANSI 的终端中
    try:
        import ctypes
        kernel32 = ctypes.windll.kernel32
        # 尝试启用 VT 模式
        STD_OUTPUT_HANDLE = -11
        ENABLE_VIRTUAL_TERMINAL_PROCESSING = 0x0004
        handle = kernel32.GetStdHandle(STD_OUTPUT_HANDLE)
        mode = ctypes.c_ulong()
        if kernel32.GetConsoleMode(handle, ctypes.byref(mode)):
            if kernel32.SetConsoleMode(handle, mode.value | ENABLE_VIRTUAL_TERMINAL_PROCESSING):
                _ansi_enabled = True
    except Exception:
        pass


class Style:
    """ANSI 样式代码"""
    RESET = "\x1b[0m"
    BOLD = "\x1b[1m"
    DIM = "\x1b[2m"
    UNDERLINE = "\x1b[4m"
    
    # 前景色
    BLACK = "\x1b[30m"
    RED = "\x1b[31m"
    GREEN = "\x1b[32m"
    YELLOW = "\x1b[33m"
    BLUE = "\x1b[34m"
    MAGENTA = "\x1b[35m"
    CYAN = "\x1b[36m"
    WHITE = "\x1b[37m"
    
    # 亮色
    BRIGHT_BLACK = "\x1b[90m"
    BRIGHT_RED = "\x1b[91m"
    BRIGHT_GREEN = "\x1b[92m"
    BRIGHT_YELLOW = "\x1b[93m"
    BRIGHT_BLUE = "\x1b[94m"
    BRIGHT_MAGENTA = "\x1b[95m"
    BRIGHT_CYAN = "\x1b[96m"
    BRIGHT_WHITE = "\x1b[97m"


class Icon:
    """Unicode 图标"""
    CHECK = "✓"
    CROSS = "✗"
    ARROW = "→"
    BULLET = "●"
    CIRCLE = "○"
    PLAY = "▶"
    STOP = "■"
    STAR = "★"
    INFO = "ℹ"
    WARN = "⚠"
    GEAR = "⚙"
    SPARKLE = "✨"


def is_ansi_enabled() -> bool:
    """检查 ANSI 是否启用"""
    return _ansi_enabled


def styled(text: str, *styles) -> str:
    """应用样式到文本"""
    if not _ansi_enabled or not styles:
        return text
    style_codes = "".join(styles)
    return f"{style_codes}{text}{Style.RESET}"


def success(msg: str) -> None:
    """打印成功消息"""
    # 先清除当前行（可能有进度或状态信息）
    if _ansi_enabled:
        sys.stdout.write("\x1b[2K\r")
        sys.stdout.flush()
        print(f"{Style.GREEN}{Icon.CHECK}{Style.RESET} {msg}")
    else:
        sys.stdout.write("\r" + " " * 60 + "\r")
        sys.stdout.flush()
        print(f"[OK] {msg}")


def error(msg: str) -> None:
    """打印错误消息"""
    if _ansi_enabled:
        print(f"{Style.RED}{Icon.CROSS}{Style.RESET} {msg}")
    else:
        print(f"[ERROR] {msg}")


def info(msg: str) -> None:
    """打印信息消息"""
    if _ansi_enabled:
        print(f"{Style.CYAN}{Icon.INFO}{Style.RESET} {msg}")
    else:
        print(f"[INFO] {msg}")


def warning(msg: str) -> None:
    """打印警告消息"""
    if _ansi_enabled:
        print(f"{Style.YELLOW}{Icon.WARN}{Style.RESET} {msg}")
    else:
        print(f"[WARN] {msg}")


def config(key: str, value: str) -> None:
    """打印配置项"""
    if _ansi_enabled:
        print(f"  {Style.DIM}{key}{Style.RESET}: {Style.BRIGHT_CYAN}{value}{Style.RESET}")
    else:
        print(f"  {key}: {value}")


def header(title: str) -> None:
    """打印章节标题"""
    if _ansi_enabled:
        print(f"\n{Style.BOLD}{Style.CYAN}{Icon.PLAY} {title}{Style.RESET}")
        print("─" * 40)
    else:
        print(f"\n=== {title} ===")


def rule(char: str = "─", width: int = 40) -> None:
    """打印分隔线"""
    print(char * width)


def request_start(total_count: int, request_id: str) -> None:
    """打印请求开始信息（与 C++ apiserver 一致）"""
    if _ansi_enabled:
        # 格式：[累计请求数](蓝色) = N
        print(f"{Style.CYAN}[累计请求数]{Style.RESET} = {total_count}")
    else:
        print(f"[累计请求数] = {total_count}")


def request_complete(request_id: str) -> None:
    """打印请求完成信息（与 C++ apiserver 一致）"""
    # 先清除当前行（可能有"生成中"状态），确保正确换行
    sys.stdout.write("\x1b[2K\r")
    sys.stdout.flush()
    if _ansi_enabled:
        # 格式：  请求 X 处理完成 (灰色，有缩进)
        print(f"{Style.DIM}  请求 {request_id} 处理完成{Style.RESET}")
    else:
        print(f"  请求 {request_id} 处理完成")


def print_inference_stats(prompt_tokens: int, output_tokens: int, total_time: float, 
                          first_token_time: float, speed: float) -> None:
    """打印推理统计信息（与 C++ apiserver 一致）"""
    # 注意：调用方（request_complete）已经清除行并换行，这里直接输出
    if _ansi_enabled:
        print(f"{Style.GREEN}{Icon.CHECK}{Style.RESET} "
              f"提示词: {Style.BRIGHT_CYAN}{prompt_tokens}{Style.RESET}, "
              f"输出: {Style.BRIGHT_CYAN}{output_tokens}{Style.RESET}, "
              f"耗时: {Style.YELLOW}{total_time:.2f}s{Style.RESET}, "
              f"首字: {Style.YELLOW}{first_token_time:.2f}s{Style.RESET}, "
              f"速度: {Style.BRIGHT_GREEN}{speed:.1f} tokens/s{Style.RESET}")
    else:
        print(f"[完成] 提示词: {prompt_tokens}, 输出: {output_tokens}, "
              f"耗时: {total_time:.2f}s, 首字: {first_token_time:.2f}s, "
              f"速度: {speed:.1f} tokens/s")


def prompt(text: str) -> str:
    """美化的输入提示"""
    if _ansi_enabled:
        return f"{Style.BOLD}{Style.CYAN}{text}{Style.RESET}"
    return text


def ai_response_start() -> None:
    """AI 响应开始标记"""
    if _ansi_enabled:
        print(f"{Style.BOLD}{Style.GREEN}AI:{Style.RESET} ", end="")
    else:
        print("AI: ", end="")


def user_prompt() -> str:
    """获取用户输入提示文本"""
    if _ansi_enabled:
        return f"\n{Style.BOLD}{Style.BLUE}User：{Style.RESET}"
    return "\nUser："


def clear_line() -> None:
    """清除当前行"""
    if _ansi_enabled:
        print("\x1b[2K\r", end="")


# === 与 C++ 端一致的接口 ===

def init() -> None:
    """初始化控制台（设置 UTF-8 编码等）"""
    global _ansi_enabled
    if sys.platform == "win32":
        try:
            import ctypes
            kernel32 = ctypes.windll.kernel32
            # 设置控制台输出代码页为 UTF-8
            kernel32.SetConsoleOutputCP(65001)
            kernel32.SetConsoleCP(65001)
            # 启用 VT 模式
            STD_OUTPUT_HANDLE = -11
            ENABLE_VIRTUAL_TERMINAL_PROCESSING = 0x0004
            handle = kernel32.GetStdHandle(STD_OUTPUT_HANDLE)
            mode = ctypes.c_ulong()
            if kernel32.GetConsoleMode(handle, ctypes.byref(mode)):
                kernel32.SetConsoleMode(handle, mode.value | ENABLE_VIRTUAL_TERMINAL_PROCESSING)
                _ansi_enabled = True
        except Exception:
            pass
    # 设置 stdout 编码
    if hasattr(sys.stdout, 'reconfigure'):
        try:
            sys.stdout.reconfigure(encoding='utf-8')
        except Exception:
            pass


def box_start(title: str, width: int = 50) -> None:
    """开始一个信息框"""
    if _ansi_enabled:
        # 顶部边框
        print(f"{Style.CYAN}┌{'─' * (width - 2)}┐{Style.RESET}")
        # 标题行
        title_display = f" {title} "
        padding = width - 4 - len(title_display.encode('utf-8').decode('utf-8'))
        # 使用中文字符宽度计算
        title_len = sum(2 if ord(c) > 127 else 1 for c in title_display)
        padding = width - 2 - title_len
        if padding < 0:
            padding = 0
        print(f"{Style.CYAN}│{Style.RESET}{Style.BOLD}{title_display}{Style.RESET}{' ' * padding}{Style.CYAN}│{Style.RESET}")
        # 分隔线
        print(f"{Style.CYAN}├{'─' * (width - 2)}┤{Style.RESET}")
    else:
        print("+" + "-" * (width - 2) + "+")
        print(f"| {title}")
        print("+" + "-" * (width - 2) + "+")


def box_end(width: int = 50) -> None:
    """结束一个信息框"""
    if _ansi_enabled:
        print(f"{Style.CYAN}└{'─' * (width - 2)}┘{Style.RESET}")
    else:
        print("+" + "-" * (width - 2) + "+")


def box_line(key: str, value: str, width: int = 50) -> None:
    """在信息框内打印一行"""
    if _ansi_enabled:
        content = f"  {key}: {Style.BRIGHT_CYAN}{value}{Style.RESET}"
        # 计算实际显示宽度（不含 ANSI 码）
        visible_len = sum(2 if ord(c) > 127 else 1 for c in f"  {key}: {value}")
        padding = width - 2 - visible_len
        if padding < 0:
            padding = 0
        print(f"{Style.CYAN}│{Style.RESET}{content}{' ' * padding}{Style.CYAN}│{Style.RESET}")
    else:
        print(f"|   {key}: {value}")


def banner(title: str, subtitle: str = None) -> None:
    """打印启动 banner"""
    if _ansi_enabled:
        print(f"\n{Style.BOLD}{Style.CYAN}╔{'═' * 48}╗{Style.RESET}")
        # 居中标题
        title_len = sum(2 if ord(c) > 127 else 1 for c in title)
        padding = (48 - title_len) // 2
        print(f"{Style.BOLD}{Style.CYAN}║{Style.RESET}{' ' * padding}{Style.BOLD}{title}{Style.RESET}{' ' * (48 - padding - title_len)}{Style.BOLD}{Style.CYAN}║{Style.RESET}")
        if subtitle:
            sub_len = sum(2 if ord(c) > 127 else 1 for c in subtitle)
            sub_padding = (48 - sub_len) // 2
            print(f"{Style.BOLD}{Style.CYAN}║{Style.RESET}{Style.DIM}{' ' * sub_padding}{subtitle}{' ' * (48 - sub_padding - sub_len)}{Style.RESET}{Style.BOLD}{Style.CYAN}║{Style.RESET}")
        print(f"{Style.BOLD}{Style.CYAN}╚{'═' * 48}╝{Style.RESET}\n")
    else:
        print(f"\n{'=' * 50}")
        print(f"  {title}")
        if subtitle:
            print(f"  {subtitle}")
        print(f"{'=' * 50}\n")


# 便捷的颜色函数
def green(text: str) -> str:
    return styled(text, Style.GREEN)

def red(text: str) -> str:
    return styled(text, Style.RED)

def yellow(text: str) -> str:
    return styled(text, Style.YELLOW)

def cyan(text: str) -> str:
    return styled(text, Style.CYAN)

def bold(text: str) -> str:
    return styled(text, Style.BOLD)

def dim(text: str) -> str:
    return styled(text, Style.DIM)
