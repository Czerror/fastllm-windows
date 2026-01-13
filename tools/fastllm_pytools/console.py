"""
console.py - 控制台美化输出模块

通过检测 FTLLM_ANSI 环境变量来决定是否启用 ANSI 彩色输出。
当通过 ftllm.exe 启动时，环境变量会自动设置。
"""

import os
import sys
import threading
import time
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


# Spinner 动画帧
SPINNER_FRAMES = ["⠋", "⠙", "⠹", "⠸", "⠼", "⠴", "⠦", "⠧", "⠇", "⠏"]
SPINNER_FRAMES_SIMPLE = ["|", "/", "-", "\\"]


class Spinner:
    """旋转动画指示器"""
    
    def __init__(self, message: str = "加载中", use_simple: bool = False):
        self.message = message
        self.frames = SPINNER_FRAMES_SIMPLE if use_simple else SPINNER_FRAMES
        self.running = False
        self.thread = None
        self.frame_idx = 0
        self.start_time = None
    
    def _animate(self):
        while self.running:
            elapsed = time.time() - self.start_time
            frame = self.frames[self.frame_idx % len(self.frames)]
            
            if _ansi_enabled:
                # 清除当前行并显示 spinner
                sys.stdout.write(f"\x1b[2K\r{Style.CYAN}{frame}{Style.RESET} {self.message} ({elapsed:.1f}s)")
            else:
                sys.stdout.write(f"\r{frame} {self.message} ({elapsed:.1f}s)")
            sys.stdout.flush()
            
            self.frame_idx += 1
            time.sleep(0.1)
    
    def start(self):
        """启动 spinner"""
        self.running = True
        self.start_time = time.time()
        self.thread = threading.Thread(target=self._animate, daemon=True)
        self.thread.start()
    
    def stop(self, final_message: str = None, success: bool = True):
        """停止 spinner"""
        self.running = False
        if self.thread:
            self.thread.join(timeout=0.2)
        
        elapsed = time.time() - self.start_time if self.start_time else 0
        
        # 清除 spinner 行
        if _ansi_enabled:
            sys.stdout.write("\x1b[2K\r")
        else:
            sys.stdout.write("\r" + " " * 60 + "\r")
        
        # 显示最终消息
        if final_message:
            if success:
                if _ansi_enabled:
                    print(f"{Style.GREEN}{Icon.CHECK}{Style.RESET} {final_message} ({elapsed:.1f}s)")
                else:
                    print(f"[OK] {final_message} ({elapsed:.1f}s)")
            else:
                if _ansi_enabled:
                    print(f"{Style.RED}{Icon.CROSS}{Style.RESET} {final_message} ({elapsed:.1f}s)")
                else:
                    print(f"[FAIL] {final_message} ({elapsed:.1f}s)")
        sys.stdout.flush()


@contextmanager
def spinner(message: str = "加载中", success_msg: str = None, fail_msg: str = None):
    """
    Spinner 上下文管理器
    
    使用方式:
        with spinner("加载模型", "模型加载完成"):
            load_model()
    """
    s = Spinner(message)
    s.start()
    try:
        yield s
        s.stop(success_msg or message + " 完成", success=True)
    except Exception as e:
        s.stop(fail_msg or f"{message} 失败: {e}", success=False)
        raise


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
    if _ansi_enabled:
        print(f"{Style.GREEN}{Icon.CHECK}{Style.RESET} {msg}")
    else:
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


# 进度条相关
def progress_bar(progress: float, width: int = 40, label: str = None) -> str:
    """生成进度条字符串"""
    filled = int(progress * width)
    bar = "█" * filled + "░" * (width - filled)
    pct = int(progress * 100)
    
    if _ansi_enabled:
        prefix = f"{Style.DIM}{label} {Style.RESET}" if label else ""
        return f"{prefix}[{Style.GREEN}{bar[:filled]}{Style.RESET}{Style.DIM}{bar[filled:]}{Style.RESET}] {pct}%"
    else:
        prefix = f"{label} " if label else ""
        return f"{prefix}[{bar}] {pct}%"


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
