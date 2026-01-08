@echo off
chcp 65001 >nul 2>&1
title FastLLM 构建工具
cd /d "%~dp0"
powershell -NoLogo -NoProfile -ExecutionPolicy Bypass -File "%~dp0build.ps1"
if errorlevel 1 (
    echo.
    echo 脚本执行出错，按任意键退出...
    pause >nul
)
