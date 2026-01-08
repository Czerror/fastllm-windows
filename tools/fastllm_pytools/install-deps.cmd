@echo off
chcp 65001 >nul
:: FastLLM 依赖安装脚本
:: 基础 run/chat 功能无需安装依赖，此脚本用于安装可选功能的依赖

setlocal EnableDelayedExpansion

set "SCRIPT_DIR=%~dp0"
set "SCRIPT_DIR=%SCRIPT_DIR:~0,-1%"
for %%i in ("%SCRIPT_DIR%") do set "ROOT_DIR=%%~dpi"
set "ROOT_DIR=%ROOT_DIR:~0,-1%"

echo ============================================================
echo   FastLLM 依赖安装
echo ============================================================
echo.
echo 基础功能 (ftllm run/chat) 无需安装任何依赖！
echo.
echo 可选功能依赖:
echo   [1] API Server (ftllm serve)  - fastapi, uvicorn, pydantic
echo   [2] Web UI (ftllm webui)      - streamlit
echo   [3] HuggingFace 下载          - huggingface_hub
echo   [4] 全部安装
echo   [0] 退出
echo.

set /p choice="请选择要安装的功能 (1/2/3/4/0): "

if "%choice%"=="0" exit /b 0
if "%choice%"=="1" (
    echo 正在安装 API Server 依赖...
    pip install fastapi pydantic uvicorn shortuuid openai
    goto :done
)
if "%choice%"=="2" (
    echo 正在安装 Web UI 依赖...
    pip install streamlit
    goto :done
)
if "%choice%"=="3" (
    echo 正在安装 HuggingFace 依赖...
    pip install huggingface_hub
    goto :done
)
if "%choice%"=="4" (
    echo 正在安装全部依赖...
    pip install -r "%ROOT_DIR%\ftllm\requirements.txt"
    goto :done
)

echo 无效选择
exit /b 1

:done
echo.
echo 安装完成！
pause
