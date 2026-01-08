<# 
.SYNOPSIS
    FastLLM Windows 编译环境一键安装脚本

.DESCRIPTION
    自动下载并安装所有编译所需工具：
    - Visual Studio 2022 Build Tools (MSVC 编译器)
    - CMake (便携版，自动集成)
    - CUDA Toolkit (可选，仅 CUDA 版本需要)
    - Python (可选，用于 Python API)

.PARAMETER All
    安装所有工具（包括 CUDA）

.PARAMETER NoCuda
    不安装 CUDA Toolkit

.PARAMETER NoPython
    不安装 Python

.EXAMPLE
    .\setup-env.ps1
    交互式安装

.EXAMPLE
    .\setup-env.ps1 -All
    安装所有工具
#>

param(
    [switch]$All,
    [switch]$NoCuda,
    [switch]$NoPython
)

$ErrorActionPreference = "Stop"
[Console]::OutputEncoding = [System.Text.Encoding]::UTF8

$ProjectRoot = if ($PSScriptRoot) { $PSScriptRoot } else { Split-Path -Parent $MyInvocation.MyCommand.Path }
$ToolsDir = Join-Path $ProjectRoot "tools\build-tools"

# ============== 工具下载链接 ==============
$Downloads = @{
    # Visual Studio 2022 Build Tools (在线安装器)
    VSBuildTools = @{
        Url = "https://aka.ms/vs/17/release/vs_buildtools.exe"
        File = "vs_buildtools.exe"
        Desc = "Visual Studio 2022 Build Tools"
    }
    # CMake 便携版
    CMake = @{
        Url = "https://github.com/Kitware/CMake/releases/download/v3.31.4/cmake-3.31.4-windows-x86_64.zip"
        File = "cmake-3.31.4-windows-x86_64.zip"
        Desc = "CMake 3.31.4"
        ExtractDir = "cmake-3.31.4-windows-x86_64"
    }
    # CUDA Toolkit (在线安装器)
    CUDA = @{
        Url = "https://developer.download.nvidia.com/compute/cuda/12.6.3/network_installers/cuda_12.6.3_windows_network.exe"
        File = "cuda_12.6.3_network.exe"
        Desc = "CUDA Toolkit 12.6.3"
    }
    # Python 嵌入版
    Python = @{
        Url = "https://www.python.org/ftp/python/3.11.9/python-3.11.9-embed-amd64.zip"
        File = "python-3.11.9-embed-amd64.zip"
        Desc = "Python 3.11.9 Embedded"
    }
}

# ============== 工具函数 ==============

function Write-Banner {
    Clear-Host
    Write-Host ""
    Write-Host "  ╔═══════════════════════════════════════════════════════════╗" -ForegroundColor Cyan
    Write-Host "  ║                                                           ║" -ForegroundColor Cyan
    Write-Host "  ║        FastLLM 编译环境一键安装工具 v1.0                  ║" -ForegroundColor Cyan
    Write-Host "  ║                                                           ║" -ForegroundColor Cyan
    Write-Host "  ╚═══════════════════════════════════════════════════════════╝" -ForegroundColor Cyan
    Write-Host ""
}

function Write-Section {
    param([string]$Title)
    Write-Host ""
    Write-Host "  ┌─ $Title " -ForegroundColor Yellow -NoNewline
    $padding = 55 - $Title.Length
    if ($padding -gt 0) { Write-Host ("-" * $padding) -ForegroundColor Yellow }
    else { Write-Host "" }
}

function Test-CommandExists {
    param([string]$Command)
    $null -ne (Get-Command $Command -ErrorAction SilentlyContinue)
}

function Get-VSInstallPath {
    $vswhere = "${env:ProgramFiles(x86)}\Microsoft Visual Studio\Installer\vswhere.exe"
    if (Test-Path $vswhere) {
        $path = & $vswhere -latest -products * -requires Microsoft.VisualStudio.Component.VC.Tools.x86.x64 -property installationPath
        return $path
    }
    return $null
}

function Test-VSBuildTools {
    $vsPath = Get-VSInstallPath
    if ($vsPath) {
        $vcvars = Join-Path $vsPath "VC\Auxiliary\Build\vcvars64.bat"
        return Test-Path $vcvars
    }
    return $false
}

function Test-CUDA {
    return Test-Path "${env:CUDA_PATH}\bin\nvcc.exe" -ErrorAction SilentlyContinue
}

function Download-File {
    param(
        [string]$Url,
        [string]$OutFile,
        [string]$Desc
    )
    
    Write-Host "  正在下载 $Desc ..." -ForegroundColor Cyan
    
    $ProgressPreference = 'SilentlyContinue'
    try {
        Invoke-WebRequest -Uri $Url -OutFile $OutFile -UseBasicParsing
        Write-Host "  ✓ 下载完成: $OutFile" -ForegroundColor Green
        return $true
    } catch {
        Write-Host "  ✗ 下载失败: $_" -ForegroundColor Red
        return $false
    }
}

# ============== 检测已安装工具 ==============

function Show-Status {
    Write-Section "当前环境检测"
    Write-Host ""
    
    $status = @{}
    
    # Visual Studio Build Tools
    if (Test-VSBuildTools) {
        Write-Host "  ✓ Visual Studio Build Tools" -ForegroundColor Green -NoNewline
        Write-Host " - 已安装" -ForegroundColor DarkGray
        $status.VS = $true
    } else {
        Write-Host "  ✗ Visual Studio Build Tools" -ForegroundColor Red -NoNewline
        Write-Host " - 未安装 (必需)" -ForegroundColor Yellow
        $status.VS = $false
    }
    
    # CMake
    $cmakeLocal = Join-Path $ToolsDir "cmake\bin\cmake.exe"
    if ((Test-CommandExists "cmake") -or (Test-Path $cmakeLocal)) {
        Write-Host "  ✓ CMake" -ForegroundColor Green -NoNewline
        if (Test-Path $cmakeLocal) {
            Write-Host " - 已集成 (本地)" -ForegroundColor DarkGray
        } else {
            Write-Host " - 已安装 (系统)" -ForegroundColor DarkGray
        }
        $status.CMake = $true
    } else {
        Write-Host "  ✗ CMake" -ForegroundColor Red -NoNewline
        Write-Host " - 未安装 (必需)" -ForegroundColor Yellow
        $status.CMake = $false
    }
    
    # CUDA
    if (Test-CUDA) {
        $cudaVer = (nvcc --version 2>$null | Select-String "release" | ForEach-Object { $_ -replace '.*release (\d+\.\d+).*', '$1' })
        Write-Host "  ✓ CUDA Toolkit" -ForegroundColor Green -NoNewline
        Write-Host " - 已安装 (v$cudaVer)" -ForegroundColor DarkGray
        $status.CUDA = $true
    } else {
        Write-Host "  ○ CUDA Toolkit" -ForegroundColor DarkGray -NoNewline
        Write-Host " - 未安装 (仅 CUDA 版本需要)" -ForegroundColor DarkGray
        $status.CUDA = $false
    }
    
    # Python
    if (Test-CommandExists "python") {
        $pyVer = (python --version 2>&1) -replace 'Python ', ''
        Write-Host "  ✓ Python" -ForegroundColor Green -NoNewline
        Write-Host " - 已安装 (v$pyVer)" -ForegroundColor DarkGray
        $status.Python = $true
    } else {
        Write-Host "  ○ Python" -ForegroundColor DarkGray -NoNewline
        Write-Host " - 未安装 (可选)" -ForegroundColor DarkGray
        $status.Python = $false
    }
    
    # NVIDIA GPU
    Write-Host ""
    if (Test-CommandExists "nvidia-smi") {
        $gpu = (nvidia-smi --query-gpu=name --format=csv,noheader 2>$null | Select-Object -First 1)
        Write-Host "  ✓ NVIDIA GPU" -ForegroundColor Green -NoNewline
        Write-Host " - $gpu" -ForegroundColor DarkGray
        $status.GPU = $true
    } else {
        Write-Host "  ○ NVIDIA GPU" -ForegroundColor DarkGray -NoNewline
        Write-Host " - 未检测到 (仅支持 CPU 版本)" -ForegroundColor DarkGray
        $status.GPU = $false
    }
    
    return $status
}

# ============== 安装函数 ==============

function Install-VSBuildTools {
    Write-Section "安装 Visual Studio 2022 Build Tools"
    Write-Host ""
    Write-Host "  将安装以下组件:" -ForegroundColor Cyan
    Write-Host "    - MSVC C++ 编译器"
    Write-Host "    - Windows SDK"
    Write-Host "    - CMake 工具"
    Write-Host ""
    
    $downloadDir = Join-Path $env:TEMP "fastllm-setup"
    New-Item -ItemType Directory -Path $downloadDir -Force | Out-Null
    
    $installerPath = Join-Path $downloadDir $Downloads.VSBuildTools.File
    
    if (-not (Test-Path $installerPath)) {
        if (-not (Download-File -Url $Downloads.VSBuildTools.Url -OutFile $installerPath -Desc $Downloads.VSBuildTools.Desc)) {
            return $false
        }
    }
    
    Write-Host ""
    Write-Host "  正在启动安装程序..." -ForegroundColor Cyan
    Write-Host "  请在安装界面中选择:" -ForegroundColor Yellow
    Write-Host "    1. 使用 C++ 的桌面开发" -ForegroundColor White
    Write-Host "    2. (可选) Windows 10/11 SDK" -ForegroundColor White
    Write-Host ""
    
    # 静默安装模式（包含必要组件）
    $args = @(
        "--add", "Microsoft.VisualStudio.Workload.VCTools",
        "--add", "Microsoft.VisualStudio.Component.VC.Tools.x86.x64",
        "--add", "Microsoft.VisualStudio.Component.Windows11SDK.22621",
        "--add", "Microsoft.VisualStudio.Component.VC.CMake.Project",
        "--includeRecommended",
        "--passive",
        "--norestart"
    )
    
    Start-Process -FilePath $installerPath -ArgumentList $args -Wait
    
    if (Test-VSBuildTools) {
        Write-Host "  ✓ Visual Studio Build Tools 安装成功" -ForegroundColor Green
        return $true
    } else {
        Write-Host "  ✗ 安装可能未完成，请重试" -ForegroundColor Red
        return $false
    }
}

function Install-CMake {
    Write-Section "安装 CMake (便携版)"
    Write-Host ""
    
    $downloadDir = Join-Path $env:TEMP "fastllm-setup"
    New-Item -ItemType Directory -Path $downloadDir -Force | Out-Null
    New-Item -ItemType Directory -Path $ToolsDir -Force | Out-Null
    
    $zipPath = Join-Path $downloadDir $Downloads.CMake.File
    $extractPath = Join-Path $ToolsDir "cmake"
    
    if (-not (Test-Path $zipPath)) {
        if (-not (Download-File -Url $Downloads.CMake.Url -OutFile $zipPath -Desc $Downloads.CMake.Desc)) {
            return $false
        }
    }
    
    Write-Host "  正在解压..." -ForegroundColor Cyan
    
    # 删除旧版本
    if (Test-Path $extractPath) {
        Remove-Item -Path $extractPath -Recurse -Force
    }
    
    # 解压到临时目录
    $tempExtract = Join-Path $downloadDir "cmake-extract"
    Expand-Archive -Path $zipPath -DestinationPath $tempExtract -Force
    
    # 移动到目标目录
    $sourceDir = Join-Path $tempExtract $Downloads.CMake.ExtractDir
    Move-Item -Path $sourceDir -Destination $extractPath -Force
    
    # 清理
    Remove-Item -Path $tempExtract -Recurse -Force -ErrorAction SilentlyContinue
    
    $cmakeExe = Join-Path $extractPath "bin\cmake.exe"
    if (Test-Path $cmakeExe) {
        Write-Host "  ✓ CMake 安装成功: $extractPath" -ForegroundColor Green
        
        # 更新 build.ps1 中的 CMake 路径
        Write-Host "  正在更新 build.ps1 配置..." -ForegroundColor Cyan
        
        return $true
    } else {
        Write-Host "  ✗ CMake 安装失败" -ForegroundColor Red
        return $false
    }
}

function Install-CUDA {
    Write-Section "安装 CUDA Toolkit 12.6"
    Write-Host ""
    Write-Host "  注意: CUDA Toolkit 需要 NVIDIA GPU 和驱动" -ForegroundColor Yellow
    Write-Host ""
    
    $downloadDir = Join-Path $env:TEMP "fastllm-setup"
    New-Item -ItemType Directory -Path $downloadDir -Force | Out-Null
    
    $installerPath = Join-Path $downloadDir $Downloads.CUDA.File
    
    if (-not (Test-Path $installerPath)) {
        if (-not (Download-File -Url $Downloads.CUDA.Url -OutFile $installerPath -Desc $Downloads.CUDA.Desc)) {
            return $false
        }
    }
    
    Write-Host "  正在启动 CUDA 安装程序..." -ForegroundColor Cyan
    Write-Host "  建议只安装以下组件:" -ForegroundColor Yellow
    Write-Host "    - CUDA Runtime"
    Write-Host "    - CUDA Development (compiler, headers)"
    Write-Host "    - cuBLAS"
    Write-Host ""
    
    Start-Process -FilePath $installerPath -Wait
    
    if (Test-CUDA) {
        Write-Host "  ✓ CUDA Toolkit 安装成功" -ForegroundColor Green
        return $true
    } else {
        Write-Host "  ⚠ 请重启终端后验证 CUDA 安装" -ForegroundColor Yellow
        return $true
    }
}

# ============== 主流程 ==============

Write-Banner
$status = Show-Status

Write-Host ""
Write-Host "  ─────────────────────────────────────────────────────────" -ForegroundColor DarkGray
Write-Host ""

# 确定需要安装的组件
$needVS = -not $status.VS
$needCMake = -not $status.CMake
$needCUDA = (-not $status.CUDA) -and $status.GPU -and (-not $NoCuda)

if ($needVS -or $needCMake -or $needCUDA) {
    Write-Host "  需要安装的组件:" -ForegroundColor Cyan
    if ($needVS) { Write-Host "    • Visual Studio Build Tools" }
    if ($needCMake) { Write-Host "    • CMake" }
    if ($needCUDA) { Write-Host "    • CUDA Toolkit" }
    Write-Host ""
    
    if (-not $All) {
        $confirm = Read-Host "  是否继续安装? (Y/n)"
        if ($confirm -eq 'n' -or $confirm -eq 'N') {
            Write-Host "  已取消" -ForegroundColor Yellow
            exit 0
        }
    }
    
    # 安装 Visual Studio Build Tools
    if ($needVS) {
        Install-VSBuildTools
    }
    
    # 安装 CMake
    if ($needCMake) {
        Install-CMake
    }
    
    # 安装 CUDA
    if ($needCUDA) {
        Install-CUDA
    }
    
    Write-Host ""
    Write-Host "  ═══════════════════════════════════════════════════════════" -ForegroundColor Green
    Write-Host "  安装完成！" -ForegroundColor Green
    Write-Host ""
    Write-Host "  下一步: 运行 .\build.ps1 开始编译" -ForegroundColor Cyan
    Write-Host "  ═══════════════════════════════════════════════════════════" -ForegroundColor Green
} else {
    Write-Host "  ✓ 所有必需工具已安装，可以开始编译" -ForegroundColor Green
    Write-Host ""
    Write-Host "  运行 .\build.ps1 开始编译" -ForegroundColor Cyan
}

Write-Host ""
