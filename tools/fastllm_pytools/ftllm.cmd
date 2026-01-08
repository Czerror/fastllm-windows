@echo off
chcp 65001 >nul
setlocal EnableDelayedExpansion

:: ============================================================
:: ftllm - FastLLM Python CLI Launcher
:: 使用方法: ftllm run <模型路径> [选项]
:: ============================================================

:: 获取脚本所在目录 (bin/)
set "SCRIPT_DIR=%~dp0"
set "SCRIPT_DIR=%SCRIPT_DIR:~0,-1%"

:: 设置根目录和 Python 模块路径
for %%i in ("%SCRIPT_DIR%") do set "ROOT_DIR=%%~dpi"
set "ROOT_DIR=%ROOT_DIR:~0,-1%"
set "PYTHON_DIR=%ROOT_DIR%"

:: 将 bin 目录添加到 PATH (用于 DLL 查找: CUDA/VC++ runtime)
set "PATH=%SCRIPT_DIR%;%PATH%"

:: 检查 Python 是否可用
where python >nul 2>&1
if errorlevel 1 (
    echo [错误] 未找到 Python，请确保 Python 已安装并添加到 PATH
    echo.
    echo 您可以从以下地址下载 Python:
    echo   https://www.python.org/downloads/
    exit /b 1
)

:: 显示帮助信息
if "%~1"=="" (
    echo FastLLM Python CLI
    echo.
    echo 使用方法:
    echo   ftllm run ^<模型路径^> [选项]    - 运行模型进行对话
    echo   ftllm chat ^<模型路径^> [选项]   - 运行模型进行对话
    echo   ftllm serve ^<模型路径^> [选项]  - 启动 OpenAI 兼容 API 服务
    echo   ftllm webui ^<模型路径^> [选项]  - 启动 Web UI ^(需要 streamlit^)
    echo   ftllm -v                        - 显示版本
    echo.
    echo 常用选项:
    echo   --device cuda                   - 使用 CUDA GPU
    echo   --device cpu                    - 使用 CPU
    echo   --threads N                     - CPU 线程数
    echo.
    echo 示例:
    echo   ftllm run D:\Models\Qwen3 --device cuda
    echo   ftllm serve D:\Models\Qwen3 --port 8080
    exit /b 0
)

if "%~1"=="-h" (
    python -c "import sys; sys.path.insert(0, r'%PYTHON_DIR%'); from ftllm.cli import main; main()" --help
    exit /b 0
)

if "%~1"=="--help" (
    python -c "import sys; sys.path.insert(0, r'%PYTHON_DIR%'); from ftllm.cli import main; main()" --help
    exit /b 0
)

:: 运行 ftllm CLI
python -c "import sys; sys.path.insert(0, r'%PYTHON_DIR%'); from ftllm.cli import main; main()" %*
