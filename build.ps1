<# 
.SYNOPSIS
    FastLLM Windows 交互式构建脚本

.DESCRIPTION
    通过交互式菜单或命令行参数选择构建选项

.PARAMETER Auto
    非交互模式，使用参数或默认值

.PARAMETER Target
    构建目标: cpu, cuda, both

.PARAMETER CudaArch
    CUDA 架构: native, all, 或具体值如 "89;120"

.PARAMETER Clean
    清理后重新构建

.PARAMETER CleanCache
    清理 MSVC/CMake 编译缓存 (C:\Users\xxx\AppData\Local 下的缓存)

.PARAMETER NoPackage
    构建完成后不打包

.PARAMETER UseMmap
    使用内存映射加载模型 (默认: 开启)

.PARAMETER UseSentencePiece
    启用 SentencePiece 分词器 (默认: 开启)

.PARAMETER BuildCli
    构建命令行工具 (默认: 开启)

.PARAMETER PyApi
    构建 Python API (默认: 开启)

.PARAMETER UnitTest
    构建单元测试 (默认: 开启)

.PARAMETER CudaNoTensorCore
    针对无 Tensor Core 的旧款 GPU 优化

.EXAMPLE
    .\build.ps1
    交互式模式

.EXAMPLE
    .\build.ps1 -Auto -Target cuda -CudaArch native
    自动构建 CUDA 版本，仅编译本机 GPU 架构
#>

param(
    [switch]$Auto,
    
    [ValidateSet("cpu", "cuda", "both")]
    [string]$Target = "",
    
    [string]$CudaArch = "",
    
    [switch]$Clean,
    [switch]$CleanCache,
    [switch]$NoPackage,
    
    # CMake 选项
    [Nullable[bool]]$UseMmap = $null,
    [Nullable[bool]]$UseSentencePiece = $null,
    [Nullable[bool]]$BuildCli = $null,
    [Nullable[bool]]$PyApi = $null,
    [Nullable[bool]]$UnitTest = $null,
    [Nullable[bool]]$CudaNoTensorCore = $null
)

# ============== 全局错误处理 ==============
trap {
    Write-Host ""
    Write-Host "  [错误] $_" -ForegroundColor Red
    Write-Host ""
    if (-not $Auto) { Read-Host "  按 Enter 键退出" }
    exit 1
}

# ============== 配置 ==============
$ErrorActionPreference = "Continue"
[Console]::OutputEncoding = [System.Text.Encoding]::UTF8

try { $Host.UI.RawUI.WindowTitle = "FastLLM 构建工具" } catch {}

$ProjectRoot = if ($PSScriptRoot) { $PSScriptRoot } else { Split-Path -Parent $MyInvocation.MyCommand.Path }

# CMake 路径自动检测 (优先级: 本地集成 > VS BuildTools > 系统 PATH)
$CMakeLocalPath = Join-Path $ProjectRoot "tools\build-tools\cmake\bin\cmake.exe"
$CMakeVSPath = "C:\Program Files (x86)\Microsoft Visual Studio\2022\BuildTools\Common7\IDE\CommonExtensions\Microsoft\CMake\CMake\bin\cmake.exe"
$CMakePath = if (Test-Path $CMakeLocalPath) { $CMakeLocalPath }
             elseif (Test-Path $CMakeVSPath) { $CMakeVSPath }
             elseif (Get-Command "cmake" -ErrorAction SilentlyContinue) { "cmake" }
             else { $CMakeVSPath }

$VCRuntimePath = "C:\Program Files (x86)\Microsoft Visual Studio\2022\BuildTools\VC\Redist\MSVC\14.44.35112\x64\Microsoft.VC143.CRT"

# 子模块版本缓存文件 (用于检测子模块更新)
$SubmoduleVersionFile = Join-Path $ProjectRoot "build\.submodule_versions"

# ============== 子模块版本检测 ==============
function Get-SubmoduleVersions {
    $versions = @{}
    $submodules = @("third_party/pybind11", "third_party/sentencepiece")
    foreach ($submodule in $submodules) {
        $submodulePath = Join-Path $ProjectRoot $submodule
        if (Test-Path (Join-Path $submodulePath ".git")) {
            Push-Location $submodulePath
            try {
                $hash = git rev-parse HEAD 2>$null
                if ($hash) { $versions[$submodule] = $hash.Trim() }
            } finally { Pop-Location }
        }
    }
    return $versions
}

function Test-SubmoduleUpdated {
    $currentVersions = Get-SubmoduleVersions
    if (-not (Test-Path $SubmoduleVersionFile)) {
        return $true  # 首次构建，视为需要清理
    }
    
    $savedVersions = @{}
    Get-Content $SubmoduleVersionFile | ForEach-Object {
        $parts = $_ -split "="
        if ($parts.Count -eq 2) { $savedVersions[$parts[0]] = $parts[1] }
    }
    
    foreach ($key in $currentVersions.Keys) {
        if ($savedVersions[$key] -ne $currentVersions[$key]) {
            Write-Host "  [!] 检测到子模块更新: $key" -ForegroundColor Yellow
            Write-Host "      旧: $($savedVersions[$key])" -ForegroundColor DarkGray
            Write-Host "      新: $($currentVersions[$key])" -ForegroundColor DarkGray
            return $true
        }
    }
    return $false
}

function Save-SubmoduleVersions {
    $versions = Get-SubmoduleVersions
    $dir = Split-Path $SubmoduleVersionFile
    if (-not (Test-Path $dir)) { New-Item -ItemType Directory -Path $dir -Force | Out-Null }
    $versions.GetEnumerator() | ForEach-Object { "$($_.Key)=$($_.Value)" } | Set-Content $SubmoduleVersionFile
}

# CUDA 架构选项
$script:CudaArchPresets = @(
    @{ Name = "全架构 (RTX 20/30/40/50 全系列)"; Value = "75;80;86;89;90;120"; Desc = "兼容性最好，体积最大" },
    @{ Name = "仅本机 GPU"; Value = "native"; Desc = "编译快，体积小" },
    @{ Name = "RTX 50 系列 (Blackwell)"; Value = "120"; Desc = "sm_120" },
    @{ Name = "RTX 40 系列 (Ada Lovelace)"; Value = "89"; Desc = "sm_89" },
    @{ Name = "RTX 40 + 50 系列"; Value = "89;120"; Desc = "sm_89, sm_120" },
    @{ Name = "RTX 30 系列 (Ampere)"; Value = "80;86"; Desc = "sm_80, sm_86" },
    @{ Name = "RTX 20 系列 (Turing)"; Value = "75"; Desc = "sm_75" },
    @{ Name = "数据中心 (H100/A100)"; Value = "80;90"; Desc = "sm_80, sm_90" },
    @{ Name = "自定义输入"; Value = "custom"; Desc = "" }
)

# CMake 编译选项 (Windows 默认值)
$script:CMakeOptions = @{
    USE_MMAP = @{ Default = $true; Desc = "使用内存映射加载模型文件 (减少内存占用)" }
    USE_SENTENCEPIECE = @{ Default = $true; Desc = "启用 SentencePiece 分词器 (从子模块构建)" }
    BUILD_CLI = @{ Default = $true; Desc = "构建命令行界面工具 (FastllmStudio CLI)" }
    PY_API = @{ Default = $true; Desc = "构建 Python API 绑定 (需要 pybind11 子模块和 Python)" }
    UNIT_TEST = @{ Default = $true; Desc = "构建单元测试" }
    CUDA_NO_TENSOR_CORE = @{ Default = $false; Desc = "针对无 Tensor Core 的旧款 GPU 优化 (GTX 10系列)" }
}

# ============== 工具函数 ==============

function Write-Banner {
    Clear-Host
    Write-Host ""
    Write-Host "  ╔═══════════════════════════════════════════════════════════╗" -ForegroundColor Cyan
    Write-Host "  ║                                                           ║" -ForegroundColor Cyan
    Write-Host "  ║            FastLLM Windows 构建工具 v1.1                  ║" -ForegroundColor Cyan
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

function Show-Menu {
    param(
        [string]$Title,
        [array]$Options,
        [int]$Default = 0,
        [switch]$ShowDesc
    )
    
    Write-Section $Title
    Write-Host ""
    
    for ($i = 0; $i -lt $Options.Count; $i++) {
        $prefix = if ($i -eq $Default) { " >" } else { "  " }
        $color = if ($i -eq $Default) { "Green" } else { "White" }
        $text = "$prefix [$($i + 1)] $($Options[$i].Name)"
        Write-Host $text -ForegroundColor $color -NoNewline
        
        if ($ShowDesc -and $Options[$i].Desc) {
            Write-Host " - $($Options[$i].Desc)" -ForegroundColor DarkGray
        } else {
            Write-Host ""
        }
    }
    
    Write-Host ""
    $choice = Read-Host "  请选择 [1-$($Options.Count)] (默认: $($Default + 1))"
    
    if ([string]::IsNullOrWhiteSpace($choice)) {
        return $Default
    }
    
    $index = [int]$choice - 1
    if ($index -ge 0 -and $index -lt $Options.Count) {
        return $index
    }
    
    return $Default
}

function Show-YesNo {
    param(
        [string]$Question,
        [bool]$Default = $false
    )
    
    $defaultText = if ($Default) { "Y/n" } else { "y/N" }
    $choice = Read-Host "  $Question [$defaultText]"
    
    if ([string]::IsNullOrWhiteSpace($choice)) {
        return $Default
    }
    
    return $choice.ToLower() -eq "y" -or $choice.ToLower() -eq "yes" -or $choice -eq "是"
}

function Show-ToggleMenu {
    param(
        [string]$Title,
        [hashtable]$Options,
        [hashtable]$CurrentValues
    )
    
    Write-Section $Title
    Write-Host ""
    Write-Host "  输入编号切换开关状态，输入 0 或回车完成" -ForegroundColor DarkGray
    Write-Host ""
    
    $keys = @($Options.Keys | Sort-Object)
    
    while ($true) {
        for ($i = 0; $i -lt $keys.Count; $i++) {
            $key = $keys[$i]
            $val = $CurrentValues[$key]
            $status = if ($val) { "[开启]" } else { "[关闭]" }
            $statusColor = if ($val) { "Green" } else { "DarkGray" }
            
            Write-Host "   [$($i + 1)] " -NoNewline -ForegroundColor White
            Write-Host $status -NoNewline -ForegroundColor $statusColor
            Write-Host " $($Options[$key].Desc)" -ForegroundColor White
        }
        
        Write-Host ""
        $choice = Read-Host "  输入编号切换 (0 或回车完成)"
        
        if ([string]::IsNullOrWhiteSpace($choice) -or $choice -eq "0") {
            break
        }
        
        $index = [int]$choice - 1
        if ($index -ge 0 -and $index -lt $keys.Count) {
            $key = $keys[$index]
            $CurrentValues[$key] = -not $CurrentValues[$key]
            Write-Host ""
        }
    }
    
    return $CurrentValues
}

function Find-CMake {
    if (Test-Path $CMakePath) {
        return $CMakePath
    }
    
    $altPaths = @(
        "cmake",
        "C:\Program Files\CMake\bin\cmake.exe",
        "C:\Program Files (x86)\CMake\bin\cmake.exe"
    )
    
    foreach ($path in $altPaths) {
        if (Get-Command $path -ErrorAction SilentlyContinue) {
            return $path
        }
    }
    
    return $null
}

function Find-CudaPath {
    $versions = @("v13.1", "v13.0", "v12.8", "v12.6", "v12.4", "v12.2", "v12.0", "v11.8")
    
    foreach ($ver in $versions) {
        $testPath = "C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\$ver"
        if (Test-Path $testPath) {
            # CUDA 13.x: DLL 在 bin\x64, nvcc 在 bin
            # CUDA 12.x 及以下: DLL 和 nvcc 都在 bin
            $dllPath = if ($ver -match "v13\.") { "$testPath\bin\x64" } else { "$testPath\bin" }
            return @{
                Path = $testPath
                BinPath = "$testPath\bin"
                DllPath = $dllPath
                Version = $ver
            }
        }
    }
    
    return $null
}

function Test-Environment {
    Write-Section "检查构建环境"
    Write-Host ""
    
    $errors = @()
    
    # 检查 CMake
    $cmake = Find-CMake
    if ($cmake) {
        Write-Host "  [√] CMake: " -ForegroundColor Green -NoNewline
        Write-Host $cmake -ForegroundColor Gray
        $script:CMakePath = $cmake
    } else {
        Write-Host "  [×] CMake 未找到" -ForegroundColor Red
        $errors += "CMake"
    }
    
    # 检查 CUDA
    $cuda = Find-CudaPath
    if ($cuda) {
        Write-Host "  [√] CUDA Toolkit: " -ForegroundColor Green -NoNewline
        Write-Host "$($cuda.Version)" -ForegroundColor Gray
        $script:CudaInfo = $cuda
    } else {
        Write-Host "  [!] CUDA Toolkit 未找到 (仅支持 CPU 构建)" -ForegroundColor Yellow
        $script:CudaInfo = $null
    }
    
    # 检查 VC++ Runtime
    if (Test-Path $VCRuntimePath) {
        Write-Host "  [√] VC++ 运行时: " -ForegroundColor Green -NoNewline
        Write-Host "已找到" -ForegroundColor Gray
    } else {
        Write-Host "  [!] VC++ 运行时未找到 (打包时可能缺少 DLL)" -ForegroundColor Yellow
    }
    
    Write-Host ""
    
    if ($errors.Count -gt 0) {
        Write-Host "  缺少必要组件: $($errors -join ', ')" -ForegroundColor Red
        Write-Host "  请安装 Visual Studio 2022 BuildTools" -ForegroundColor Red
        return $false
    }
    
    return $true
}

function Build-Project {
    param(
        [string]$BuildType,
        [string]$BuildDir,
        [hashtable]$Options,
        [bool]$CleanBuild
    )
    
    $fullBuildDir = Join-Path $ProjectRoot $BuildDir
    
    Write-Host ""
    Write-Host "  ════════════════════════════════════════════════════════════" -ForegroundColor Cyan
    Write-Host "    正在构建 $BuildType 版本" -ForegroundColor Cyan
    Write-Host "  ════════════════════════════════════════════════════════════" -ForegroundColor Cyan
    Write-Host ""
    
    # 检测子模块更新，自动清理构建目录
    if ($script:SubmoduleCheckDone -ne $true) {
        if (Test-SubmoduleUpdated) {
            Write-Host "  [!] 子模块已更新，需要清理构建以确保使用新版本" -ForegroundColor Yellow
            $CleanBuild = $true
        }
        $script:SubmoduleCheckDone = $true
    }
    
    # 清理构建目录
    if ($CleanBuild -and (Test-Path $fullBuildDir)) {
        Write-Host "  [1/3] 清理构建目录..." -ForegroundColor Yellow
        Remove-Item $fullBuildDir -Recurse -Force
    }
    
    # 清理编译缓存 (仅首次构建时执行)
    if ($script:CleanCacheExecuted -ne $true -and $CleanCache.IsPresent) {
        Write-Host "  [*] 清理编译缓存..." -ForegroundColor Yellow
        $cachePaths = @(
            # CMake 缓存
            "$env:LOCALAPPDATA\CMake\cache",
            # MSVC IntelliSense 缓存
            "$env:LOCALAPPDATA\Microsoft\VisualStudio\17.0_*\ComponentModelCache",
            # 项目 .vs 目录
            (Join-Path $ProjectRoot ".vs")
        )
        foreach ($cachePath in $cachePaths) {
            $resolved = Get-Item $cachePath -ErrorAction SilentlyContinue
            foreach ($item in $resolved) {
                if (Test-Path $item) {
                    Write-Host "    删除: $item" -ForegroundColor DarkGray
                    Remove-Item $item -Recurse -Force -ErrorAction SilentlyContinue
                }
            }
        }
        $script:CleanCacheExecuted = $true
    }
    
    # 创建目录
    if (-not (Test-Path $fullBuildDir)) {
        New-Item -ItemType Directory -Path $fullBuildDir -Force | Out-Null
    }
    
    Push-Location $fullBuildDir
    
    try {
        # CMake 配置
        Write-Host "  [1/3] CMake 配置中..." -ForegroundColor Yellow
        
        $cmakeArgs = @("..", "-G", "Visual Studio 17 2022", "-A", "x64")
        foreach ($key in $Options.Keys) {
            $val = $Options[$key]
            if ($val -is [bool]) {
                $val = if ($val) { "ON" } else { "OFF" }
            }
            $cmakeArgs += "-D$key=$val"
        }
        
        Write-Host "        cmake $($cmakeArgs -join ' ')" -ForegroundColor DarkGray
        
        $configOutput = & $CMakePath $cmakeArgs 2>&1
        if ($LASTEXITCODE -ne 0) {
            Write-Host $configOutput -ForegroundColor Red
            throw "CMake 配置失败"
        }
        Write-Host "  [√] CMake 配置完成" -ForegroundColor Green
        
        # 编译
        Write-Host "  [2/3] 编译中 (可能需要几分钟)..." -ForegroundColor Yellow
        
        $buildStart = Get-Date
        $buildOutput = & $CMakePath --build . --config Release --parallel 2 2>&1
        $buildTime = (Get-Date) - $buildStart
        
        if ($LASTEXITCODE -ne 0) {
            Write-Host "编译输出:" -ForegroundColor Red
            $buildOutput | Select-Object -Last 50 | ForEach-Object { Write-Host $_ }
            throw "编译失败"
        }
        
        Write-Host "  [√] 编译完成 (耗时: $([math]::Round($buildTime.TotalMinutes, 1)) 分钟)" -ForegroundColor Green
        
        # 验证输出
        Write-Host "  [3/3] 验证输出文件..." -ForegroundColor Yellow
        $exeFiles = Get-ChildItem "$fullBuildDir\Release\*.exe" -ErrorAction SilentlyContinue
        $pydFiles = Get-ChildItem "$fullBuildDir\Release\*.pyd" -ErrorAction SilentlyContinue
        
        if ($exeFiles.Count -eq 0 -and $pydFiles.Count -eq 0) {
            throw "未找到输出文件"
        }
        
        if ($exeFiles.Count -gt 0) {
            Write-Host "  [√] 生成了 $($exeFiles.Count) 个可执行文件" -ForegroundColor Green
            foreach ($exe in $exeFiles) {
                $size = [math]::Round($exe.Length / 1MB, 2)
                Write-Host "      - $($exe.Name) ($size MB)" -ForegroundColor Gray
            }
        }
        
        if ($pydFiles.Count -gt 0) {
            Write-Host "  [√] 生成了 $($pydFiles.Count) 个 Python 模块" -ForegroundColor Green
            foreach ($pyd in $pydFiles) {
                $size = [math]::Round($pyd.Length / 1MB, 2)
                Write-Host "      - $($pyd.Name) ($size MB)" -ForegroundColor Gray
            }
        }
        
        return $true
    }
    catch {
        Write-Host "  [×] 构建失败: $($_.Exception.Message)" -ForegroundColor Red
        return $false
    }
    finally {
        Pop-Location
    }
}

function New-ReleasePackage {
    param(
        [string]$BuildType,
        [string]$BuildDir,
        [string]$PackageName,
        [hashtable]$CudaInfo = $null
    )
    
    Write-Host ""
    Write-Host "  正在打包 $BuildType 版本..." -ForegroundColor Yellow
    
    # 从 setup.py 读取版本号
    $setupPyPath = Join-Path $ProjectRoot "tools\scripts\setup.py"
    $version = "0.0.0"
    if (Test-Path $setupPyPath) {
        $content = Get-Content $setupPyPath -Raw
        if ($content -match 'version\s*=\s*"([^"]+)"') {
            $version = $matches[1]
            Write-Host "  检测到版本号: $version" -ForegroundColor Gray
        }
    }
    
    $fullBuildDir = Join-Path $ProjectRoot $BuildDir
    $zipPath = Join-Path $ProjectRoot "$PackageName.zip"
    
    # 删除旧包
    if (Test-Path $zipPath) { Remove-Item $zipPath -Force -ErrorAction SilentlyContinue }
    
    # 使用 .NET ZipArchive 直接写入压缩包
    Add-Type -AssemblyName System.IO.Compression
    Add-Type -AssemblyName System.IO.Compression.FileSystem
    $zip = [System.IO.Compression.ZipFile]::Open($zipPath, 'Create')
    
    try {
        # 辅助函数：添加文件到压缩包
        $addFile = {
            param($sourcePath, $entryName)
            if (Test-Path $sourcePath) {
                [System.IO.Compression.ZipFileExtensions]::CreateEntryFromFile($zip, $sourcePath, $entryName, 'Optimal') | Out-Null
            }
        }
        
        # 辅助函数：添加目录下所有文件
        $addDir = {
            param($sourceDir, $targetPrefix)
            if (Test-Path $sourceDir) {
                Get-ChildItem $sourceDir -Recurse -File | ForEach-Object {
                    $relativePath = $_.FullName.Substring($sourceDir.Length + 1)
                    $entryName = "$targetPrefix/$relativePath".Replace("\", "/")
                    & $addFile $_.FullName $entryName
                }
            }
        }
        
        # 1. bin/ - 可执行文件
        Get-ChildItem "$fullBuildDir\Release\*.exe" -ErrorAction SilentlyContinue | ForEach-Object {
            & $addFile $_.FullName "bin/$($_.Name)"
        }
        
        # 2. bin/ - VC++ 运行时
        if (Test-Path $VCRuntimePath) {
            Get-ChildItem "$VCRuntimePath\*.dll" -ErrorAction SilentlyContinue | ForEach-Object {
                & $addFile $_.FullName "bin/$($_.Name)"
            }
        }
        
        # 3. bin/ - CUDA DLL (仅 CUDA 版本)
        if ($BuildType -eq "cuda" -and $CudaInfo) {
            @("cublas64_*.dll", "cublasLt64_*.dll", "cudart64_*.dll") | ForEach-Object {
                Get-ChildItem $CudaInfo.DllPath -Filter $_ -ErrorAction SilentlyContinue | ForEach-Object {
                    & $addFile $_.FullName "bin/$($_.Name)"
                }
            }
        }
        
        # 4. bin/ - ftllm 命令行入口脚本和依赖安装脚本
        $ftllmCmd = Join-Path $ProjectRoot "tools\fastllm_pytools\ftllm.cmd"
        if (Test-Path $ftllmCmd) {
            & $addFile $ftllmCmd "bin/ftllm.cmd"
        }
        $installDeps = Join-Path $ProjectRoot "tools\fastllm_pytools\install-deps.cmd"
        if (Test-Path $installDeps) {
            & $addFile $installDeps "bin/install-deps.cmd"
        }
        
        # 5. ftllm/ - Python 模块
        Get-ChildItem "$fullBuildDir\Release\*.pyd" -ErrorAction SilentlyContinue | ForEach-Object {
            & $addFile $_.FullName "ftllm/$($_.Name)"
        }
        
        # 6. ftllm/ - ftllm 工具
        $ftllmSource = Join-Path $fullBuildDir "tools\ftllm"
        & $addDir $ftllmSource "ftllm"
        
        # 6.1 ftllm/__init__.py - 更新版本号
        $initPyContent = @"
__all__ = ["llm"]

try:
    from importlib.metadata import version
    __version__ = version("ftllm")
except:
    try:
        __version__ = version("ftllm-rocm")
    except:
        __version__ = "$version"
"@
        $tempInitPy = Join-Path $env:TEMP "ftllm_init_temp.py"
        Set-Content -Path $tempInitPy -Value $initPyContent -Encoding UTF8 -NoNewline
        & $addFile $tempInitPy "ftllm/__init__.py"
        Remove-Item $tempInitPy -Force -ErrorAction SilentlyContinue
        
        # 7. ftllm/ - requirements.txt
        $reqFile = Join-Path $ProjectRoot "tools\fastllm_pytools\requirements.txt"
        if (Test-Path $reqFile) {
            & $addFile $reqFile "ftllm/requirements.txt"
        }
        
        # 6. web/ - Web UI
        $webSource = Join-Path $ProjectRoot "example\webui\web"
        & $addDir $webSource "web"
        
        # 8. docs/ - 文档
        Get-ChildItem "$ProjectRoot\docs\*.md" -ErrorAction SilentlyContinue | Where-Object { $_.Name -ne "使用说明.md" } | ForEach-Object {
            & $addFile $_.FullName "docs/$($_.Name)"
        }
        Get-ChildItem "$ProjectRoot\README*.md" -ErrorAction SilentlyContinue | ForEach-Object {
            & $addFile $_.FullName "docs/$($_.Name)"
        }
        
        # 9. 根目录 - 使用说明
        $readmeFile = Join-Path $ProjectRoot "docs\使用说明.md"
        if (Test-Path $readmeFile) {
            & $addFile $readmeFile "使用说明.md"
        }
    }
    finally {
        $zip.Dispose()
    }
    
    $size = [math]::Round((Get-Item $zipPath).Length / 1MB, 2)
    Write-Host "  [√] 打包完成: $PackageName.zip ($size MB)" -ForegroundColor Green
    
    return $zipPath
}

# ============== 主程序 ==============

Write-Banner

# 检查环境
if (-not (Test-Environment)) {
    Write-Host ""
    if (-not $Auto) { Read-Host "  按 Enter 键退出" }
    exit 1
}

# 初始化 CMake 选项值
$cmakeValues = @{}
foreach ($key in $CMakeOptions.Keys) {
    $cmakeValues[$key] = $CMakeOptions[$key].Default
}

# 自动模式
if ($Auto) {
    Write-Host "  [自动模式] 使用命令行参数" -ForegroundColor Magenta
    Write-Host ""
    
    # 设置目标
    if ([string]::IsNullOrWhiteSpace($Target)) {
        $target = if ($CudaInfo) { "cuda" } else { "cpu" }
    } else {
        $target = $Target
    }
    
    # 设置 CUDA 架构
    if ([string]::IsNullOrWhiteSpace($CudaArch)) {
        $cudaArch = "native"
    } elseif ($CudaArch -eq "all") {
        $cudaArch = "75;80;86;89;90;120"
    } else {
        $cudaArch = $CudaArch
    }
    
    $clean = $Clean.IsPresent
    $package = -not $NoPackage.IsPresent
    
    # 应用命令行 CMake 选项
    if ($null -ne $UseMmap) { $cmakeValues["USE_MMAP"] = $UseMmap }
    if ($null -ne $UseSentencePiece) { $cmakeValues["USE_SENTENCEPIECE"] = $UseSentencePiece }
    if ($null -ne $BuildCli) { $cmakeValues["BUILD_CLI"] = $BuildCli }
    if ($null -ne $PyApi) { $cmakeValues["PY_API"] = $PyApi }
    if ($null -ne $UnitTest) { $cmakeValues["UNIT_TEST"] = $UnitTest }
    if ($null -ne $CudaNoTensorCore) { $cmakeValues["CUDA_NO_TENSOR_CORE"] = $CudaNoTensorCore }
    
    Write-Host "    构建目标: $target" -ForegroundColor Gray
    if ($target -ne "cpu") { Write-Host "    CUDA 架构: $cudaArch" -ForegroundColor Gray }
    Write-Host "    清理重建: $(if($clean){'是'}else{'否'})" -ForegroundColor Gray
    Write-Host "    自动打包: $(if($package){'是'}else{'否'})" -ForegroundColor Gray
    Write-Host ""
}
else {
    # ========== 交互模式 ==========
    
    # 1. 选择构建目标
    $targetOptions = @()
    if ($CudaInfo) {
        $targetOptions += @{ Name = "CUDA 版本 (GPU 加速，推荐)"; Value = "cuda"; Desc = "需要 NVIDIA GPU" }
        $targetOptions += @{ Name = "CPU 版本 (无需 GPU)"; Value = "cpu"; Desc = "兼容性好" }
        $targetOptions += @{ Name = "同时构建 CPU 和 CUDA 版本"; Value = "both"; Desc = "完整构建" }
    } else {
        $targetOptions = @(
            @{ Name = "CPU 版本 (无需 GPU)"; Value = "cpu"; Desc = "仅此选项可用" }
        )
    }
    
    $targetIndex = Show-Menu -Title "选择构建目标" -Options $targetOptions -Default 0 -ShowDesc
    $target = $targetOptions[$targetIndex].Value

    # 2. CUDA 架构选择
    $cudaArch = "75;80;86;89;90;120"
    if ($target -ne "cpu" -and $CudaInfo) {
        $archIndex = Show-Menu -Title "选择 CUDA 架构" -Options $CudaArchPresets -Default 0 -ShowDesc
        $cudaArch = $CudaArchPresets[$archIndex].Value
        
        if ($cudaArch -eq "custom") {
            Write-Host ""
            Write-Host "  架构代码参考:" -ForegroundColor Gray
            Write-Host "    75 = RTX 20 系列 (Turing)" -ForegroundColor DarkGray
            Write-Host "    80 = RTX 30 系列 / A100 (Ampere)" -ForegroundColor DarkGray
            Write-Host "    86 = RTX 30 系列笔记本 (Ampere)" -ForegroundColor DarkGray
            Write-Host "    89 = RTX 40 系列 (Ada Lovelace)" -ForegroundColor DarkGray
            Write-Host "    90 = H100 (Hopper)" -ForegroundColor DarkGray
            Write-Host "   120 = RTX 50 系列 (Blackwell)" -ForegroundColor DarkGray
            Write-Host ""
            $cudaArch = Read-Host "  请输入架构 (用分号分隔，如 89;120)"
            if ([string]::IsNullOrWhiteSpace($cudaArch)) {
                $cudaArch = "75;80;86;89;90;120"
            }
        }
    }

    # 3. 高级编译选项
    Write-Section "高级编译选项"
    Write-Host ""
    
    if (Show-YesNo -Question "是否配置高级编译选项?" -Default $false) {
        $cmakeValues = Show-ToggleMenu -Title "编译选项 (输入编号切换)" -Options $CMakeOptions -CurrentValues $cmakeValues
    } else {
        Write-Host "  使用默认配置" -ForegroundColor Gray
    }

    # 4. 其他选项
    Write-Section "构建选项"
    Write-Host ""
    
    $clean = Show-YesNo -Question "清理后重新构建?" -Default $false
    $package = $true  # 编译完成后自动打包

    # 5. 确认配置
    Write-Section "配置确认"
    Write-Host ""
    Write-Host "  构建目标:     " -NoNewline; Write-Host $target -ForegroundColor Cyan
    if ($target -ne "cpu") {
        Write-Host "  CUDA 架构:    " -NoNewline; Write-Host $cudaArch -ForegroundColor Cyan
    }
    Write-Host "  清理重建:     " -NoNewline; Write-Host $(if($clean){"是"}else{"否"}) -ForegroundColor Cyan
    Write-Host "  自动打包:     " -NoNewline; Write-Host "是" -ForegroundColor Green
    Write-Host ""
    Write-Host "  编译选项:" -ForegroundColor White
    foreach ($key in ($cmakeValues.Keys | Sort-Object)) {
        $val = $cmakeValues[$key]
        $status = if ($val) { "开启" } else { "关闭" }
        $statusColor = if ($val) { "Green" } else { "DarkGray" }
        Write-Host "    - $key : " -NoNewline -ForegroundColor Gray
        Write-Host $status -ForegroundColor $statusColor
    }
    Write-Host ""

    if (-not (Show-YesNo -Question "确认开始构建?" -Default $true)) {
        Write-Host ""
        Write-Host "  已取消构建" -ForegroundColor Yellow
        exit 0
    }
}

# ========== 开始构建 ==========
$results = @{}
$startTime = Get-Date

# 构建 CPU 版本
if ($target -eq "cpu" -or $target -eq "both") {
    $cpuOptions = @{}
    foreach ($key in $cmakeValues.Keys) {
        $cpuOptions[$key] = $cmakeValues[$key]
    }
    
    $success = Build-Project -BuildType "CPU" -BuildDir "build" -Options $cpuOptions -CleanBuild $clean
    $results["CPU"] = $success
    
    if ($success -and $package) {
        New-ReleasePackage -BuildType "cpu" -BuildDir "build" -PackageName "fastllm-win-x64-cpu-release"
    }
}

# 构建 CUDA 版本
if (($target -eq "cuda" -or $target -eq "both") -and $CudaInfo) {
    $cudaOptions = @{
        "USE_CUDA" = $true
        "CUDA_ARCH" = $cudaArch
    }
    foreach ($key in $cmakeValues.Keys) {
        $cudaOptions[$key] = $cmakeValues[$key]
    }
    
    $success = Build-Project -BuildType "CUDA" -BuildDir "build-cuda" -Options $cudaOptions -CleanBuild $clean
    $results["CUDA"] = $success
    
    if ($success -and $package) {
        $archSuffix = if ($cudaArch -eq "75;80;86;89;90;120") { "all-arch" } 
                      elseif ($cudaArch -eq "native") { "native" } 
                      else { $cudaArch.Replace(";", "-") }
        New-ReleasePackage -BuildType "cuda" -BuildDir "build-cuda" -PackageName "fastllm-win-x64-cuda-$archSuffix-release" -CudaInfo $cudaInfo
    }
}

# ========== 清理临时文件 ==========
if ($package) {
    Write-Host ""
    Write-Host "  清理临时构建文件..." -ForegroundColor Gray
    $buildDirs = @("build", "build-cuda") | ForEach-Object { Join-Path $ProjectRoot $_ }
    foreach ($dir in $buildDirs) {
        if (Test-Path $dir) {
            Remove-Item -Recurse -Force $dir -ErrorAction SilentlyContinue
            Write-Host "    已删除: $([System.IO.Path]::GetFileName($dir))" -ForegroundColor DarkGray
        }
    }
}

# ========== 显示结果 ==========
$totalTime = (Get-Date) - $startTime

# 保存子模块版本 (仅当有成功的构建时)
if ($results.Values -contains $true) {
    Save-SubmoduleVersions
}

Write-Host ""
Write-Host "  ════════════════════════════════════════════════════════════" -ForegroundColor Cyan
Write-Host "                        构建完成                              " -ForegroundColor Cyan
Write-Host "  ════════════════════════════════════════════════════════════" -ForegroundColor Cyan
Write-Host ""

foreach ($key in $results.Keys) {
    if ($results[$key]) {
        Write-Host "  [√] $key 版本: 成功" -ForegroundColor Green
    } else {
        Write-Host "  [×] $key 版本: 失败" -ForegroundColor Red
    }
}

Write-Host ""
Write-Host "  总耗时: $([math]::Round($totalTime.TotalMinutes, 1)) 分钟" -ForegroundColor Gray

# 列出生成的包
$zipFiles = Get-ChildItem $ProjectRoot -Filter "fastllm-win-*.zip" -ErrorAction SilentlyContinue | 
            Sort-Object LastWriteTime -Descending |
            Select-Object -First 5

if ($zipFiles) {
    Write-Host ""
    Write-Host "  生成的安装包:" -ForegroundColor White
    foreach ($zip in $zipFiles) {
        $size = [math]::Round($zip.Length / 1MB, 2)
        Write-Host "    - $($zip.Name) ($size MB)" -ForegroundColor Gray
    }
}

Write-Host ""
if (-not $Auto) { Read-Host "  按 Enter 键退出" }
