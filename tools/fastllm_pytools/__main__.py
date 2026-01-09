"""
ftllm - FastLLM 命令行入口

支持以下方式运行:
  python -m ftllm <command> [options]
  ftllm <command> [options]  (通过 ftllm.exe 路由)
"""

from .cli import main

if __name__ == "__main__":
    main()
