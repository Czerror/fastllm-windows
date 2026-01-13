__all__ = ["llm", "LogLevel", "LogEvent", "LogHandlers", 
           "set_log_callback", "clear_log_callback",
           "enable_pretty_logging", "enable_simple_logging", "disable_logging"]

try:
    from importlib.metadata import version
    __version__ = version("ftllm")  # 从安装的元数据读取
except:
    try:
        __version__ = version("ftllm-rocm")
    except:
        __version__ = "0.1.5.1"

# 导出日志相关接口
from .llm import (
    LogLevel, LogEvent, LogHandlers,
    set_log_callback, clear_log_callback,
    enable_pretty_logging, enable_simple_logging, disable_logging
)