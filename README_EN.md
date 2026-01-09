# FastLLM Windows

[![Build Windows](https://github.com/Czerror/fastllm-windows/actions/workflows/build-windows.yml/badge.svg)](https://github.com/Czerror/fastllm-windows/actions/workflows/build-windows.yml)
[![License](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](LICENSE)

[中文文档](README.md)

> 🔱 **Fork of [ztxz16/fastllm](https://github.com/ztxz16/fastllm)** — Pre-built binaries for Windows platform

This project is a Windows fork of [fastllm](https://github.com/ztxz16/fastllm), providing:
- ✅ **Pre-compiled binaries** — Ready to use without build environment setup
- ✅ **GitHub Actions auto-build** — CPU and CUDA versions
- ✅ **One-click local build script** — Interactive PowerShell build tool

## 📦 Download

Go to [Releases](https://github.com/Czerror/fastllm-windows/releases) to download pre-built binaries:

| Version | Description |
|---------|-------------|
| `fastllm-windows-cpu-x.x.x.zip` | CPU only, no GPU required |
| `fastllm-windows-cuda-x.x.x.zip` | CUDA acceleration, requires NVIDIA GPU |

## 🚀 Quick Start

### 1. Extract the downloaded zip file

### 2. Environment Setup

1. Add the `bin` directory to system PATH
2. Ensure Python 3.8+ is installed (3.10+ recommended)

### 3. Run Models

**Method 1: Unified Entry Point (Recommended)**
```cmd
# Default uses C++ native programs
ftllm chat D:\Models\Qwen3 --device cuda

# Use Python backend
ftllm -py chat D:\Models\Qwen3 --device cuda
```

**Method 2: Direct native program call**
```cmd
FastllmStudio_cli.exe -p D:\Models\Qwen3 --device cuda
```

### 4. Supported Models

Refer to upstream documentation: [Supported Models List](https://github.com/ztxz16/fastllm/blob/master/docs/models.md)

---

## 🔧 Build Options

### Option 1: GitHub Actions Online Build

This project has complete CI/CD workflow configured, automatically triggered on push to `master` branch.

#### Trigger Conditions
- Push to `master` branch
- Create Pull Request to `master` branch
- Manual trigger (workflow_dispatch)

#### Build Artifacts
The workflow generates two versions:
- **CPU Version**: Pure CPU inference, compatible with all Windows x64 systems
- **CUDA Version**: GPU acceleration, supports RTX 20/30/40/50 series

#### How to Use

1. **Fork this repository**
2. **Push after modifying code**
3. **Check build status on Actions page**
4. **Download artifacts after build completes**

If a Release tag is created (e.g., `v0.1.5.1`), it will be automatically published to Releases page.

### Option 2: Local Build

#### One-click Environment Setup

```powershell
# Clone repository
git clone https://github.com/Czerror/fastllm-windows.git
cd fastllm-windows

# Run environment setup script (auto-detects and installs missing tools)
.\setup-env.ps1
```

The script will automatically detect and install:
- **Visual Studio 2022 Build Tools** — MSVC compiler (required)
- **CMake** — Portable version, auto-integrated into project
- **CUDA Toolkit** — Only when NVIDIA GPU is detected

#### Manual Environment Requirements

| Component | Version Requirement |
|-----------|---------------------|
| Windows | 10/11 x64 |
| Visual Studio | 2022 (with C++ Desktop Development) |
| CMake | 3.18+ |
| CUDA Toolkit | 12.0+ (only for CUDA version) |
| Python | 3.8+ (optional, for Python API) |

#### Quick Build

```powershell
# Clone repository
git clone https://github.com/Czerror/fastllm-windows.git
cd fastllm-windows

# Initialize submodules
git submodule update --init --recursive

# Run build script (interactive)
# Open "点我启动编译.bat" or run:
.\build.ps1
```

#### Build Script Options

Interactive menu guides you through:
1. **Build Target**: CPU / CUDA / Both
2. **CUDA Architecture**: All / Native GPU only / Specific architectures
3. **CMake Options**: Memory mapping, SentencePiece, Python API, etc.

#### Command Line Mode

```powershell
# Auto build CUDA version, compile only for native GPU
.\build.ps1 -Auto -Target cuda -CudaArch native

# Auto build CPU version, skip packaging
.\build.ps1 -Auto -Target cpu -NoPackage

# Clean rebuild
.\build.ps1 -Auto -Target both -Clean

# Build all CUDA architectures (compatible with RTX 20/30/40/50)
.\build.ps1 -Auto -Target cuda -CudaArch "75;80;86;89;90;120"
```

#### Build Output Location

After build completes, outputs are located at:
- Binary files: `build\x64\Release\`
- Packaged ZIP: `build\fastllm-windows-[cpu|cuda]-x.x.x.zip`

---

## 📁 Project Structure

```
fastllm-windows/
├── .github/workflows/     # GitHub Actions workflows
│   └── build-windows.yml  # Windows build configuration
├── build.ps1              # Windows local build script
├── include/               # C++ header files
├── src/                   # C++ source code
├── tools/                 # Python tools and scripts
├── example/               # Example code
└── docs/                  # Documentation
```

---

## 🔗 Related Links

- **Upstream Repository**: [ztxz16/fastllm](https://github.com/ztxz16/fastllm)
- **Documentation**: [docs/](https://github.com/ztxz16/fastllm/tree/master/docs)
- **Deploy DeepSeek**: [docs/deepseek.md](https://github.com/ztxz16/fastllm/blob/master/docs/deepseek.md)
- **Deploy Qwen3**: [docs/qwen3.md](https://github.com/ztxz16/fastllm/blob/master/docs/qwen3.md)

---

## 📄 License

This project follows [Apache License 2.0](LICENSE), consistent with upstream fastllm.

---

## 🙏 Acknowledgments

Thanks to [ztxz16/fastllm](https://github.com/ztxz16/fastllm) for the excellent LLM inference engine.
