cmake_minimum_required(VERSION 3.18...3.31)

project(fastllm LANGUAGES CXX)

# 默认使用 C++20（Windows 已通过 /std:c++20 设置；这里用于跨平台与 IDE/IntelliSense 统一）
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

# ============== 编译选项说明 ==============
# 使用方法: cmake .. -D选项名=ON/OFF
# 例如: cmake .. -DUSE_CUDA=ON -DBUILD_STUDIO_CLI=ON

# 启用 NVIDIA CUDA GPU 加速 (需要安装 CUDA Toolkit)
option(USE_CUDA "use CUDA" OFF)

# 针对无 Tensor Core 的旧款 CUDA GPU 优化 (如 GTX 10系列)
option(CUDA_NO_TENSOR_CORE "Optimize for legacy CUDA GPUs which has Tensor Core." OFF)

# 启用 AMD ROCm GPU 加速 (主要支持 Linux，Windows 实验性支持)
option(USE_ROCM "use ROCm" OFF)

# 启用 ThinkForce TFACC 加速卡支持
option(USE_TFACC "use tfacc" OFF)

# 构建 Python API 绑定 (pyfastllm 模块)
option(PY_API "python api" ON)

# 使用内存映射加载模型文件 (减少内存占用)
option(USE_MMAP "use mmap" ON)

# 启用 SentencePiece 分词器支持
option(USE_SENTENCEPIECE "use sentencepiece" ON)

# 启用天数智芯 Iluvatar CoreX GPU 支持
option(USE_IVCOREX "use iluvatar corex gpu" OFF)

# 构建 FastllmStudio CLI (交互式 TUI，通常不需要)
option(BUILD_STUDIO_CLI "build FastllmStudio CLI" OFF)

# 构建单元测试
option(UNIT_TEST "build unit tests" ON)

# CUDA 编译的 C++ 标准（NVCC -std），默认跟随全局 CMAKE_CXX_STANDARD（本工程默认 C++20）。
# 如需显式覆盖：-DFASTLLM_CUDA_STANDARD=17 或 -DFASTLLM_CUDA_STANDARD=20
set(FASTLLM_CUDA_STANDARD "${CMAKE_CXX_STANDARD}" CACHE STRING "C++ standard for CUDA (nvcc -std=...). Default follows CMAKE_CXX_STANDARD.")

# CUDA 架构设置，默认编译全架构版本 (RTX 20/30/40/50 系列)
# 如需仅编译本机 GPU 架构以加快编译速度并减小体积，可使用:
#   cmake .. -DCUDA_ARCH="native"
# 或指定特定架构:
#   cmake .. -DCUDA_ARCH="89"      # 仅 RTX 40 系列
#   cmake .. -DCUDA_ARCH="89;120"  # RTX 40 和 RTX 50 系列
if(NOT DEFINED CUDA_ARCH)
    set(CUDA_ARCH "75;80;86;89;90;120")
endif()

# ROCm device detection function
function(detect_rocm_devices)
    if(USE_ROCM)
        # Try to find rocminfo (cross-platform paths)
        if(WIN32)
            find_program(ROCMINFO rocminfo PATHS 
                $ENV{ROCM_PATH}/bin 
                "C:/Program Files/AMD/ROCm/bin"
                "C:/opt/rocm/bin")
        else()
            find_program(ROCMINFO rocminfo PATHS ${ROCM_PATH}/bin /opt/rocm/bin)
        endif()
        
        if(ROCMINFO)
            # Execute rocminfo to get device info
            execute_process(
                COMMAND ${ROCMINFO}
                OUTPUT_VARIABLE ROCMINFO_OUTPUT
                ERROR_QUIET
                OUTPUT_STRIP_TRAILING_WHITESPACE
            )
            
            # Parse the output to find device architectures
            set(DETECTED_ARCHS "")
            set(HAS_MI50 FALSE)
            
            # Extract gfx architectures from rocminfo output
            string(REGEX MATCHALL "Name:[ \t]+gfx[0-9a-f]+" GFX_MATCHES "${ROCMINFO_OUTPUT}")
            foreach(MATCH ${GFX_MATCHES})
                string(REGEX REPLACE "Name:[ \t]+" "" ARCH ${MATCH})
                list(APPEND DETECTED_ARCHS ${ARCH})
                if(${ARCH} STREQUAL "gfx906")
                    set(HAS_MI50 TRUE)
                endif()
            endforeach()
            
            # Also check for device names containing MI50
            string(REGEX MATCH "Marketing Name:[ \t]+[^\n]*MI50" MI50_MATCH "${ROCMINFO_OUTPUT}")
            if(MI50_MATCH)
                set(HAS_MI50 TRUE)
            endif()
            
            # Remove duplicates
            if(DETECTED_ARCHS)
                list(REMOVE_DUPLICATES DETECTED_ARCHS)
            endif()
            
            # Set parent scope variables
            set(ROCM_DETECTED_ARCHS "${DETECTED_ARCHS}" PARENT_SCOPE)
            set(ROCM_HAS_MI50 ${HAS_MI50} PARENT_SCOPE)
        else()
            message(WARNING "rocminfo not found, cannot auto-detect ROCm devices")
            set(ROCM_DETECTED_ARCHS "" PARENT_SCOPE)
            set(ROCM_HAS_MI50 FALSE PARENT_SCOPE)
        endif()
    endif()
endfunction()

# Call device detection
detect_rocm_devices()

# Set ROCM_ARCH based on detection or use default
if(USE_ROCM)
    if(NOT DEFINED ROCM_ARCH)
        if(ROCM_DETECTED_ARCHS)
            # Use detected architectures
            set(ROCM_ARCH "${ROCM_DETECTED_ARCHS}")
            message(STATUS "Auto-detected ROCm architectures: ${ROCM_ARCH}")
        else()
            # Use default if detection failed
            set(ROCM_ARCH "gfx908;gfx90a;gfx1100")
            message(STATUS "ROCM_ARCH not set and auto-detection failed, using default: ${ROCM_ARCH}")
        endif()
    else()
        message(STATUS "Using user-specified ROCM_ARCH: ${ROCM_ARCH}")
        # Check if user specified gfx906
        if("gfx906" IN_LIST ROCM_ARCH)
            set(ROCM_HAS_MI50 TRUE)
            message(STATUS "gfx906 (MI50) found in user-specified ROCM_ARCH")
        endif()
    endif()
    
    # Report MI50 detection status
    if(ROCM_HAS_MI50)
        message(STATUS "MI50 device detected (gfx906)")
    endif()
endif()

message(STATUS "USE_CUDA: ${USE_CUDA}")

message(STATUS "USE_ROCM: ${USE_ROCM}")

message(STATUS "CUDA_ARCH: ${CUDA_ARCH}")

message(STATUS "USE_TFACC: ${USE_TFACC}")

message(STATUS "For legacy CUDA GPUs: ${CUDA_NO_TENSOR_CORE}")

message(STATUS "PYTHON_API: ${PY_API}")

message(STATUS "BUILD_STUDIO_CLI: ${BUILD_STUDIO_CLI}")

message(STATUS "USE_SENTENCEPIECE: ${USE_SENTENCEPIECE}")

message(STATUS "USE_IVCOREX: ${USE_IVCOREX}")

message(STATUS "MAKE_WHL_X86: ${MAKE_WHL_X86}")

set(CMAKE_BUILD_TYPE "Release")

string(REGEX REPLACE "^([0-9]+)\\.[0-9]+\\.[0-9]+$" "\\1" CXX_MAJOR_VERSION "${CMAKE_CXX_COMPILER_VERSION}")

if (CMAKE_CXX_COMPILER_ID STREQUAL "Clang")
    # 统一由 CMAKE_CXX_STANDARD 控制语言标准（本项目默认 C++20），避免手动 --std 覆盖。
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -pthread -O2")
elseif(CMAKE_CXX_COMPILER_ID STREQUAL "MSVC")
    string(REPLACE "/Ob2" "/Ob1 /Gy" CMAKE_CXX_FLAGS_RELEASE ${CMAKE_CXX_FLAGS_RELEASE})
    # Use C++20 for designated initializers support (gguf.cpp requires it)
    # Reduced /MP to /MP2 to avoid CL.exe crash (access violation / resource exhaustion)
    # /wd4244 /wd4267: Suppress type conversion warnings (size_t to int, etc.)
    # /wd4190: Suppress C linkage with UDT return type warning
    # /wd4996: Suppress deprecated std::wstring_convert warning
    # /source-charset:utf-8 /execution-charset:utf-8: Source and execution charset both UTF-8
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} /DNOMINMAX /std:c++20 /arch:AVX2 /source-charset:utf-8 /execution-charset:utf-8 /bigobj /MP2 /D_CRT_SECURE_NO_WARNINGS /wd4244 /wd4267 /wd4190 /wd4996")
    # Windows MSVC uses AMD64 instead of x86_64
    if (CXX_MAJOR_VERSION GREATER_EQUAL 19 AND (CMAKE_SYSTEM_PROCESSOR MATCHES "x86_64" OR CMAKE_SYSTEM_PROCESSOR MATCHES "AMD64"))
        # MSVC 2022 (19.30+) supports AVX512 intrinsics with /arch:AVX512
        # For MSVC 2019 and earlier, intrinsics are available but compiler flag differs
        # Check if MSVC version supports /arch:AVX512 (MSVC 2022 17.0+ = 19.30+)
        if(MSVC_VERSION GREATER_EQUAL 1930)
            set(AVX512_FLAG "/arch:AVX512")
        else()
            # Older MSVC versions: AVX512 intrinsics work with AVX2 flag + runtime detection
            # The code uses runtime CPUID checks, so this is safe
            set(AVX512_FLAG "/arch:AVX2")
            message(STATUS "MSVC < 2022: AVX512 intrinsics enabled with runtime detection")
        endif()
        
        set_source_files_properties(
            "src/devices/cpu/avx512f.cpp"
            PROPERTIES
            COMPILE_FLAGS "${AVX512_FLAG}"
        )
        set_source_files_properties(
            "src/devices/cpu/avx512bf16.cpp"
            PROPERTIES
            COMPILE_FLAGS "${AVX512_FLAG}"
        )
        set_source_files_properties(
            "src/devices/cpu/avx512vnni.cpp"
            PROPERTIES
            COMPILE_FLAGS "${AVX512_FLAG}"
        )
        set_source_files_properties(
            "src/devices/cpu/amx.cpp"
            PROPERTIES
            COMPILE_FLAGS "${AVX512_FLAG}"
        )
    endif()
else()
    if (MAKE_WHL_X86)
        set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -pthread -O2 -mavx -mavx2 -mf16c -mfma -static-libstdc++ -static-libgcc -fPIC -Wl,-Bsymbolic -Wl,--exclude-libs,ALL")
    else()
        set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -pthread -O2 -march=native")
    endif()
    message(STATUS "CMAKE_SYSTEM_PROCESSOR: ${CMAKE_SYSTEM_PROCESSOR}")
    if (CXX_MAJOR_VERSION GREATER_EQUAL 10 AND CMAKE_SYSTEM_PROCESSOR MATCHES "x86_64")
        set_source_files_properties(
            "src/devices/cpu/avx512f.cpp"
            PROPERTIES
            COMPILE_OPTIONS "-mavx512f;-mavx512bw;-mavx512vl"
        )
        
        set_source_files_properties(
            "src/devices/cpu/avx512bf16.cpp"
            PROPERTIES
            COMPILE_OPTIONS "-mavx512bf16;-mavx512f;-mavx512bw;-mavx512vl"
        )
        set_source_files_properties(
            "src/devices/cpu/avx512vnni.cpp"
            PROPERTIES
            COMPILE_OPTIONS "-mavx512vnni;-mavx512f"
        )
        set(AMX_COMPILE_OPTIONS "-mavx512bf16;-mavx512f;-mavx512bw;-mavx512vl")
        if (CXX_MAJOR_VERSION GREATER_EQUAL 11)
            list(APPEND AMX_COMPILE_OPTIONS "-mamx-tile" "-mamx-bf16")
            message(STATUS "AMX support enabled (g++ >= 11)")
        endif()
        set_source_files_properties(
            "src/devices/cpu/amx.cpp"
            PROPERTIES
            COMPILE_OPTIONS "${AMX_COMPILE_OPTIONS}"
        )
    endif()
endif()


message(STATUS "CMAKE_CXX_FLAGS: ${CMAKE_CXX_FLAGS}")
file(GLOB GRAPH_MODEL_FILES "src/models/graph/*.cpp")
file(GLOB CPU_DEVICE_FILES "src/devices/cpu/*.cpp")
set(FASTLLM_CXX_SOURCES src/fastllm.cpp src/device.cpp src/model.cpp src/executor.cpp src/template.cpp src/graph.cpp src/tokenizer.cpp
        src/devices/cpu/cpudevice.cpp src/devices/cpu/cpudevicebatch.cpp
        src/models/graphllm.cpp src/models/chatglm.cpp src/models/moss.cpp src/models/llama.cpp src/models/qwen.cpp src/models/basellm.cpp
        src/models/glm.cpp src/models/minicpm.cpp src/models/minicpm3.cpp src/models/internlm2.cpp src/models/bert.cpp src/models/moe.cpp src/models/deepseekv2.cpp
        src/models/phi3.cpp src/models/xlmroberta.cpp src/models/cogvlm.cpp src/models/qwen3.cpp src/models/qwen3_moe.cpp src/models/qwen3_next.cpp
        src/models/minimax.cpp src/models/hunyuan.cpp
        src/models/ernie4_5.cpp src/models/pangu_moe.cpp src/models/glm4_moe.cpp src/models/gpt_oss.cpp
        third_party/json11/json11.cpp
        third_party/gguf/gguf.cpp third_party/gguf/ggml-quant.cpp third_party/gguf/ggml-iqk.cpp third_party/gguf/ggml-dequantize.cpp
        third_party/gguf/gguf-adapter.cpp
        ${CPU_DEVICE_FILES}
        ${GRAPH_MODEL_FILES})

include_directories(include)
include_directories(include/utils)
include_directories(include/models)
include_directories(include/devices/cpu)
include_directories(third_party/json11)
include_directories(third_party/gguf)
include_directories(third_party/flashinfer)

# USE_MMAP: Only enable on non-Windows or when not building PY_API
# Windows lacks sys/mman.h, so mmap is not available
if (USE_MMAP)
    if(WIN32 AND PY_API)
        message(STATUS "USE_MMAP disabled on Windows with PY_API (sys/mman.h unavailable)")
        set(USE_MMAP OFF)
    elseif(WIN32)
        message(STATUS "USE_MMAP disabled on Windows (sys/mman.h unavailable)")
        set(USE_MMAP OFF)
    else()
        add_compile_definitions(USE_MMAP)
    endif()
endif()

if (USE_SENTENCEPIECE)
    add_compile_definitions(USE_SENTENCEPIECE)
    
    # Try to find sentencepiece via vcpkg/system first
    find_package(sentencepiece CONFIG QUIET)
    if(sentencepiece_FOUND)
        message(STATUS "Found sentencepiece via vcpkg/find_package")
        set(FASTLLM_LINKED_LIBS ${FASTLLM_LINKED_LIBS} sentencepiece::sentencepiece)
    else()
        # Build sentencepiece from submodule
        set(SENTENCEPIECE_SOURCE_DIR "${CMAKE_CURRENT_SOURCE_DIR}/third_party/sentencepiece")
        if(EXISTS "${SENTENCEPIECE_SOURCE_DIR}/CMakeLists.txt")
            message(STATUS "Building sentencepiece from source (submodule)")
            
            # Save current CXX flags (sentencepiece uses /utf-8 which conflicts with /source-charset:utf-8 and /execution-charset:utf-8)
            set(SAVED_CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS}")
            # Remove our charset flags for sentencepiece build (it uses /utf-8 internally)
            string(REPLACE "/source-charset:utf-8" "" CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS}")
            string(REPLACE "/execution-charset:utf-8" "" CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS}")
            
            # Sentencepiece build options
            set(SPM_ENABLE_SHARED OFF CACHE BOOL "" FORCE)
            set(SPM_ENABLE_TCMALLOC OFF CACHE BOOL "" FORCE)
            set(SPM_ENABLE_NFKC_COMPILE OFF CACHE BOOL "" FORCE)
            
            # Add sentencepiece subdirectory
            add_subdirectory(${SENTENCEPIECE_SOURCE_DIR} ${CMAKE_BINARY_DIR}/sentencepiece EXCLUDE_FROM_ALL)
            
            # Restore CXX flags for fastllm build
            set(CMAKE_CXX_FLAGS "${SAVED_CMAKE_CXX_FLAGS}")
            
            # Include directories
            include_directories(${SENTENCEPIECE_SOURCE_DIR}/src)
            include_directories(${CMAKE_BINARY_DIR}/sentencepiece)  # For generated headers
            
            # Link against static library
            set(FASTLLM_LINKED_LIBS ${FASTLLM_LINKED_LIBS} sentencepiece-static)
        else()
            message(WARNING "sentencepiece submodule not found.")
            message(WARNING "Run: git submodule add https://github.com/google/sentencepiece.git third_party/sentencepiece")
            message(WARNING "Or install via vcpkg: vcpkg install sentencepiece:x64-windows")
            set(FASTLLM_LINKED_LIBS ${FASTLLM_LINKED_LIBS} sentencepiece)
        endif()
    endif()
endif()

if (USE_CUDA)
    enable_language(CUDA)
    # CUDA 侧的 -std（NVCC）。默认跟随 FASTLLM_CUDA_STANDARD（默认与 CMAKE_CXX_STANDARD 一致）。
    set(CMAKE_CUDA_STANDARD ${FASTLLM_CUDA_STANDARD})
    set(CMAKE_CUDA_STANDARD_REQUIRED ON)
    # Pass UTF-8 charset to host compiler for Chinese comments
    if (MSVC)
        set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -Xcompiler=/source-charset:utf-8,/execution-charset:utf-8,/Zc:preprocessor -DCCCL_IGNORE_MSVC_TRADITIONAL_PREPROCESSOR_WARNING")
    endif()
    add_compile_definitions(USE_CUDA)
    if (CUDA_NO_TENSOR_CORE)
        add_compile_definitions(CUDA_NO_TENSOR_CORE)
    endif()
    include_directories(include/devices/cuda)
    #message(${CMAKE_CUDA_IMPLICIT_LINK_DIRECTORIES})
    set(FASTLLM_CUDA_SOURCES src/devices/cuda/cudadevice.cpp src/devices/cuda/cudadevicebatch.cpp 
        src/devices/cuda/fastllm-cuda.cu 
        src/devices/cuda/fastllm-ggml-cuda.cu
        src/devices/cuda/attention/fastllm-attention.cu)

    include_directories(include/devices/multicuda)
    set(FASTLLM_CUDA_SOURCES ${FASTLLM_CUDA_SOURCES} src/devices/multicuda/multicudadevice.cpp src/devices/multicuda/fastllm-multicuda.cu)

    set(FASTLLM_LINKED_LIBS ${FASTLLM_LINKED_LIBS} cublas nccl)
    set(CMAKE_CUDA_ARCHITECTURES ${CUDA_ARCH})
endif()

if(USE_ROCM)
    # ROCm is primarily supported on Linux
    if(WIN32)
        message(WARNING "ROCm support is primarily available on Linux. Windows ROCm support is experimental.")
    endif()
    
    list(APPEND CMAKE_MODULE_PATH "${PROJECT_SOURCE_DIR}/third_party/hipify_torch/cmake")
    include(Hipify)
    enable_language(HIP)

    # remove target hip dirs first (cross-platform)
    file(REMOVE_RECURSE ${PROJECT_SOURCE_DIR}/src/devices/hip)
    file(REMOVE_RECURSE ${PROJECT_SOURCE_DIR}/src/devices/multihip)
    hipify(CUDA_SOURCE_DIR ${PROJECT_SOURCE_DIR}/src/devices/cuda HIP_SOURCE_DIR "${PROJECT_SOURCE_DIR}/src/devices/hip")
    hipify(CUDA_SOURCE_DIR ${PROJECT_SOURCE_DIR}/src/devices/multicuda HIP_SOURCE_DIR "${PROJECT_SOURCE_DIR}/src/devices/multihip")
    include_directories(include/devices/cuda)
    include_directories(include/devices/multicuda)
    set(FASTLLM_CUDA_SOURCES src/devices/cuda/cudadevice.cpp src/devices/cuda/cudadevicebatch.cpp 
        src/devices/hip/fastllm-hip.hip src/devices/hip/fastllm-ggml-hip.hip 
        src/devices/multicuda/multicudadevice.cpp src/devices/multihip/fastllm-multihip.hip)
    add_compile_definitions(USE_ROCM)
    add_compile_definitions(USE_CUDA)
    if (DEFINED ENV{ROCM_PATH})
        set(ROCM_PATH $ENV{ROCM_PATH})
    elseif(WIN32)
        # Windows ROCm default path (if installed)
        if(EXISTS "C:/Program Files/AMD/ROCm")
            set(ROCM_PATH "C:/Program Files/AMD/ROCm")
        else()
            set(ROCM_PATH "C:/opt/rocm")
        endif()
    else()
        set(ROCM_PATH /opt/rocm)
    endif()
    set(CMAKE_HIP_COMPILER_ROCM_ROOT ${ROCM_PATH})
    set(CMAKE_CXX_COMPILER "${CMAKE_HIP_COMPILER_ROCM_ROOT}/bin/amdclang++")
    message(STATUS "CMAKE_HIP_COMPILER_ROCM_ROOT: ${CMAKE_HIP_COMPILER_ROCM_ROOT}")
    message(STATUS "CMAKE_HIP_COMPILER: ${CMAKE_HIP_COMPILER}")
    message(STATUS "CMAKE_CXX_COMPILER: ${CMAKE_CXX_COMPILER}")
    
    set(CMAKE_PREFIX_PATH ${ROCM_PATH})
    find_package(hip REQUIRED) 
    find_package(hipblas REQUIRED)
    find_package(rocprim REQUIRED)
    list(APPEND FASTLLM_LINKED_LIBS hip::device roc::hipblas roc::rocprim)
    set(CMAKE_HIP_ARCHITECTURES ${ROCM_ARCH})
    add_compile_definitions(HIPBLAS_V2)

    # Check if gfx906(MI50) is in the architecture list
    set(HAS_MI50_IN_TARGETS FALSE)
    foreach(ARCH IN LISTS CMAKE_HIP_ARCHITECTURES)
        if(${ARCH} STREQUAL "gfx906")
            set(HAS_MI50_IN_TARGETS TRUE)
            break()
        endif()
    endforeach()
    
    # Enable MI50 workaround if MI50 is detected or in target architectures
    if(ROCM_HAS_MI50 OR HAS_MI50_IN_TARGETS)
        add_compile_definitions(USE_MI50_WORKAROUND)
        add_compile_definitions(HIP_NO_TENSOR_CORE)
        message(STATUS "MI50 support enabled (USE_MI50_WORKAROUND defined)")
    endif()
endif()

if (USE_IVCOREX)
    set(FASTLLM_LINKED_LIBS ${FASTLLM_LINKED_LIBS} cudart)
    set(CMAKE_CUDA_ARCHITECTURES ${IVCOREX_ARCH})
endif()

if (USE_TFACC)
    #execute_process(
    #    COMMAND "./insmodTFDriver.sh"
    #    WORKING_DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}/third_party/thinkforce"
    #)

    add_compile_definitions(USE_TFACC)
    set(FASTLLM_TFACC_SOURCES src/devices/tfacc/tfaccdevice.cpp src/devices/tfacc/fastllm-tfacc.cpp)
    include_directories(include/devices/tfacc)
endif()

if (USE_NUMA)
    add_compile_definitions(USE_NUMA)
    set(FASTLLM_CXX_SOURCES ${FASTLLM_CXX_SOURCES} src/devices/numa/numadevice.cpp src/devices/numa/fastllm-numa.cpp src/devices/numa/computeserver.cpp src/devices/numa/kvcache.cpp)
    include_directories(include/devices/numa)
    set(FASTLLM_LINKED_LIBS ${FASTLLM_LINKED_LIBS} numa)
endif()

if (USE_NUMAS)
    add_compile_definitions(USE_NUMAS)
    set(FASTLLM_CXX_SOURCES ${FASTLLM_CXX_SOURCES} src/devices/numas/numasdevice.cpp src/devices/numas/numas.cpp)
    include_directories(include/devices/numas)
    set(FASTLLM_LINKED_LIBS ${FASTLLM_LINKED_LIBS} numa)
endif()

if (USE_TOPS) # For suiyuan
    add_compile_definitions(USE_TOPS)
    set(FASTLLM_TOPS_SOURCES src/devices/tops/topsdevice.cpp)
    include_directories(include/devices/tops)

    include_directories(/usr/include/gcu/)
    include_directories(/opt/tops/include)
    include_directories(/usr/include/gcu/topsdnn)
    include_directories(/usr/src/topsaten_samples/common)

    link_directories(/usr/lib)
    link_directories(/opt/tops/lib)
    set(FASTLLM_LINKED_LIBS ${FASTLLM_LINKED_LIBS} topsaten /opt/tops/lib/libtopsrt.so)
    #target_link_libraries(main rtcu)
endif()

if (PY_API)
    if(POLICY CMP0148)
        cmake_policy(SET CMP0148 NEW)
    endif()
    set(PYBIND third_party/pybind11)
    add_subdirectory(${PYBIND})
    
    # Auto-detect Python3 (cross-platform)
    # Set Python3_ROOT_DIR if you need a specific Python version
    # set(Python3_ROOT_DIR "/usr/local/python3.10.6/bin/")  # Linux example
    # set(Python3_ROOT_DIR "C:/Python310")  # Windows example
    find_package(Python3 REQUIRED COMPONENTS Interpreter Development)

    include_directories(third_party/pybind11/include)
    file(GLOB FASTLLM_CXX_HEADERS include/**/*.h)
    add_library(pyfastllm MODULE src/pybinding.cpp ${FASTLLM_CXX_SOURCES} ${FASTLLM_CXX_HEADERS} ${FASTLLM_CUDA_SOURCES} ${FASTLLM_TFACC_SOURCES} ${FASTLLM_TOPS_SOURCES})
    
    # 仅对 pyfastllm 目标定义 PY_API 宏
    target_compile_definitions(pyfastllm PRIVATE PY_API)
    
    # Ensure C++20 for pyfastllm target (required by gguf.cpp designated initializers)
    set_target_properties(pyfastllm PROPERTIES CXX_STANDARD 20 CXX_STANDARD_REQUIRED ON)
    
    target_link_libraries(pyfastllm PUBLIC pybind11::module ${FASTLLM_LINKED_LIBS})
    pybind11_extension(pyfastllm)
endif()

# 始终编译核心库和可执行文件
add_library(fastllm OBJECT
            ${FASTLLM_CXX_SOURCES}
            ${FASTLLM_CUDA_SOURCES}
            ${FASTLLM_TFACC_SOURCES}
            ${FASTLLM_TOPS_SOURCES}
            )
# Ensure C++20 for fastllm (required by gguf.cpp designated initializers)
set_target_properties(fastllm PROPERTIES CXX_STANDARD 20 CXX_STANDARD_REQUIRED ON)
target_link_libraries(fastllm PUBLIC ${FASTLLM_LINKED_LIBS})

# ftllm 统一命令行入口 (不依赖 fastllm 库，仅转发到其他 exe)
add_executable(ftllm ftllm.cpp)
set_target_properties(ftllm PROPERTIES 
    OUTPUT_NAME "ftllm"
    CXX_STANDARD 20
    CXX_STANDARD_REQUIRED ON
)

add_executable(quant tools/src/quant.cpp)
target_link_libraries(quant fastllm)

if (UNIT_TEST)
    add_executable(testOps test/ops/cppOps.cpp)
    target_link_libraries(testOps fastllm)

    add_executable(testTokenizer test/ops/tokenizerTest.cpp)
    target_link_libraries(testTokenizer fastllm)
endif()

add_executable(webui example/webui/webui.cpp)
target_link_libraries(webui fastllm)
add_custom_command(
        TARGET webui
        POST_BUILD
        COMMAND ${CMAKE_COMMAND} -E make_directory web
        COMMAND ${CMAKE_COMMAND} -E copy_directory ${CMAKE_CURRENT_SOURCE_DIR}/example/webui/web ${CMAKE_BINARY_DIR}/web
)

add_executable(benchmark example/benchmark/benchmark.cpp)
target_link_libraries(benchmark fastllm)

add_executable(apiserver example/apiserver/apiserver.cpp)
target_link_libraries(apiserver fastllm)

if (BUILD_STUDIO_CLI)
    add_executable(FastllmStudio_cli example/FastllmStudio/cli/cli.cpp example/FastllmStudio/cli/ui.cpp)
    target_link_libraries(FastllmStudio_cli fastllm)
endif()

add_library(fastllm_tools SHARED ${FASTLLM_CXX_SOURCES} ${FASTLLM_CUDA_SOURCES} ${FASTLLM_TFACC_SOURCES} ${FASTLLM_TOPS_SOURCES} tools/src/pytools.cpp)
set_target_properties(fastllm_tools PROPERTIES CXX_STANDARD 20 CXX_STANDARD_REQUIRED ON)
target_link_libraries(fastllm_tools PUBLIC ${FASTLLM_LINKED_LIBS})

if (${CMAKE_HOST_WIN32})
    add_custom_command(
            TARGET fastllm_tools
            POST_BUILD
            COMMAND ${CMAKE_COMMAND} -E make_directory tools
            COMMAND ${CMAKE_COMMAND} -E make_directory tools/ftllm
            COMMAND ${CMAKE_COMMAND} -E copy_directory ${CMAKE_CURRENT_SOURCE_DIR}/tools/fastllm_pytools ${CMAKE_BINARY_DIR}/tools/ftllm/.
            COMMAND ${CMAKE_COMMAND} -E copy_directory ${CMAKE_CURRENT_SOURCE_DIR}/tools/scripts ${CMAKE_BINARY_DIR}/tools/.
            COMMAND ${CMAKE_COMMAND} -E copy ${CMAKE_BINARY_DIR}/$(Configuration)/fastllm_tools.dll ${CMAKE_BINARY_DIR}/tools/ftllm/.
            COMMAND ${CMAKE_COMMAND} -E remove ${CMAKE_BINARY_DIR}/$(Configuration)/fastllm_tools.dll
    )
else()
    add_custom_command(
            TARGET fastllm_tools
            POST_BUILD
            COMMAND ${CMAKE_COMMAND} -E make_directory tools
            COMMAND ${CMAKE_COMMAND} -E make_directory tools/ftllm
            COMMAND ${CMAKE_COMMAND} -E copy_directory ${CMAKE_CURRENT_SOURCE_DIR}/tools/fastllm_pytools ${CMAKE_BINARY_DIR}/tools/ftllm/.
            COMMAND ${CMAKE_COMMAND} -E copy_directory ${CMAKE_CURRENT_SOURCE_DIR}/tools/scripts ${CMAKE_BINARY_DIR}/tools/.
            COMMAND ${CMAKE_COMMAND} -E copy ${CMAKE_BINARY_DIR}/libfastllm_tools.* ${CMAKE_BINARY_DIR}/tools/ftllm/.
            COMMAND ${CMAKE_COMMAND} -E remove ${CMAKE_BINARY_DIR}/libfastllm_tools.*
    )
endif()