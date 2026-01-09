# FastLLM Windows

[![Build Windows](https://github.com/Czerror/fastllm-windows/actions/workflows/build-windows.yml/badge.svg)](https://github.com/Czerror/fastllm-windows/actions/workflows/build-windows.yml)
[![License](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](LICENSE)

> 🔱 **Fork of [ztxz16/fastllm](https://github.com/ztxz16/fastllm)** — 专注于 Windows 平台的预编译版本

本项目是 [fastllm](https://github.com/ztxz16/fastllm) 的 Windows 分支，提供：
- ✅ **预编译二进制文件** — 无需配置编译环境即可使用
- ✅ **GitHub Actions 自动构建** — CPU 版本和 CUDA 版本
- ✅ **一键本地编译脚本** — 交互式 PowerShell 构建工具

## 📦 下载

前往 [Releases](https://github.com/Czerror/fastllm-windows/releases) 下载预编译版本：

| 版本 | 说明 |
|------|------|
| `fastllm-windows-cpu-x.x.x.zip` | 仅 CPU，无需 GPU |
| `fastllm-windows-cuda-x.x.x.zip` | CUDA 加速，需要 NVIDIA GPU |

## 🚀 快速使用

### 1. 解压下载的 zip 文件

### 2. 环境准备

1. 将 `bin` 目录添加到系统 PATH 环境变量
2. 确保已安装 Python 3.8+ (推荐 3.10+)

### 3. 运行模型

**方式一：统一入口（推荐）**
```cmd
# 默认使用 C++ 原生程序
ftllm chat D:\Models\Qwen3 --device cuda

# 使用 Python 后端
ftllm -py chat D:\Models\Qwen3 --device cuda
```

**方式二：直接调用原生程序**
```cmd
FastllmStudio_cli.exe -p D:\Models\Qwen3 --device cuda
```

### 4. 支持的模型

参考上游文档：[支持的模型列表](https://github.com/ztxz16/fastllm/blob/master/docs/models.md)

---

## 🔧 编译方式

### 方式一：GitHub Actions 在线编译

本项目配置了完整的 CI/CD 工作流，每次推送到 `master` 分支会自动触发编译。

#### 触发条件
- 推送到 `master` 分支
- 创建 Pull Request 到 `master` 分支
- 手动触发 (workflow_dispatch)

#### 构建产物
工作流会生成两个版本：
- **CPU 版本**: 纯 CPU 推理，兼容所有 Windows x64 系统
- **CUDA 版本**: GPU 加速，支持 RTX 20/30/40/50 全系列

#### 如何使用

1. **Fork 本仓库**
2. **修改代码后推送**
3. **前往 Actions 页面查看构建状态**
4. **构建完成后在 Artifacts 下载编译产物**

如果创建 Release 标签 (如 `v0.1.5.1`)，会自动发布到 Releases 页面。

### 方式二：本地编译

#### 一键安装编译环境

```powershell
# 克隆仓库
git clone https://github.com/Czerror/fastllm-windows.git
cd fastllm-windows

# 运行环境安装脚本 (自动检测并安装缺失的工具)
.\setup-env.ps1
```

脚本会自动检测并安装：
- **Visual Studio 2022 Build Tools** — MSVC 编译器（必需）
- **CMake** — 便携版，自动集成到项目中
- **CUDA Toolkit** — 仅当检测到 NVIDIA GPU 时安装

#### 手动安装环境要求

| 组件 | 版本要求 |
|------|----------|
| Windows | 10/11 x64 |
| Visual Studio | 2022 (含 C++ 桌面开发工具) |
| CMake | 3.18+ |
| CUDA Toolkit | 12.0+ (仅 CUDA 版本需要) |
| Python | 3.8+ (可选，用于 Python API) |

#### 快速编译

```powershell
# 克隆仓库
git clone https://github.com/Czerror/fastllm-windows.git
cd fastllm-windows

# 初始化子模块
git submodule update --init --recursive

# 运行构建脚本 (交互式)
打开点我启动编译.bat
```

#### 构建脚本选项

交互式菜单会引导你选择：
1. **构建目标**: CPU / CUDA / 两者都构建
2. **CUDA 架构**: 全架构 / 仅本机 GPU / 指定架构
3. **CMake 选项**: 内存映射、SentencePiece、Python API 等

#### 命令行模式

```powershell
# 自动构建 CUDA 版本，仅编译本机 GPU 架构
.\build.ps1 -Auto -Target cuda -CudaArch native

# 自动构建 CPU 版本，不打包
.\build.ps1 -Auto -Target cpu -NoPackage

# 清理后重新构建
.\build.ps1 -Auto -Target both -Clean

# 构建全架构 CUDA (兼容 RTX 20/30/40/50)
.\build.ps1 -Auto -Target cuda -CudaArch "75;80;86;89;90;120"
```

#### 构建产物位置

编译完成后，产物位于：
- 二进制文件: `build\x64\Release\`
- 打包 ZIP: `build\fastllm-windows-[cpu|cuda]-x.x.x.zip`

---

## 📁 项目结构

```
fastllm-windows/
├── .github/workflows/     # GitHub Actions 工作流
│   └── build-windows.yml  # Windows 编译配置
├── build.ps1              # Windows 本地编译脚本
├── include/               # C++ 头文件
├── src/                   # C++ 源码
├── tools/                 # Python 工具和脚本
├── example/               # 示例代码
└── docs/                  # 文档
```

---

## 🔗 相关链接

- **上游仓库**: [ztxz16/fastllm](https://github.com/ztxz16/fastllm)
- **文档**: [docs/](https://github.com/ztxz16/fastllm/tree/master/docs)
- **部署 DeepSeek**: [docs/deepseek.md](https://github.com/ztxz16/fastllm/blob/master/docs/deepseek.md)
- **部署 Qwen3**: [docs/qwen3.md](https://github.com/ztxz16/fastllm/blob/master/docs/qwen3.md)

---

## 📄 许可证

本项目遵循 [Apache License 2.0](LICENSE)，与上游 fastllm 保持一致。

---

## 🙏 致谢

- [ztxz16/fastllm](https://github.com/ztxz16/fastllm) — 原项目作者
- 所有贡献者和社区成员
