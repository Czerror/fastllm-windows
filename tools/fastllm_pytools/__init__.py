__all__ = ["llm"]

try:
    from importlib.metadata import version
    __version__ = version("ftllm")  # 从安装的元数据读取
except:
    try:
        __version__ = version("ftllm-rocm")
    except:
        __version__ = "0.1.5.1"