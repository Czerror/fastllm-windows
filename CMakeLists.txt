cmake_minimum_required(VERSION 3.5)

project(fastllm LANGUAGES CXX)

# C++20 for gguf.cpp designated initializers
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

option(USE_NUMAS "use numas" OFF)
option(USE_CUDA "use CUDA" OFF)
option(CUDA_NO_TENSOR_CORE "Optimize for legacy CUDA GPUs" OFF)
option(CUDA_MEM_DEBUG "Enable CUDA memory pool debug logging" OFF)
option(USE_ROCM "use ROCm" OFF)
option(USE_TFACC "use tfacc" OFF)
option(PY_API "python api" ON)
option(USE_MMAP "use mmap" ON)
option(USE_SENTENCEPIECE "use sentencepiece" ON)
option(USE_IVCOREX "use iluvatar corex gpu" OFF)
option(USE_TOPS "use tops (suiyuan)" OFF)
option(BUILD_STUDIO_CLI "build FastllmStudio CLI" OFF)
option(UNIT_TEST "build unit tests" ON)

set(FASTLLM_CUDA_STANDARD "${CMAKE_CXX_STANDARD}" CACHE STRING "C++ standard for CUDA")

if(NOT DEFINED CUDA_ARCH)
    set(CUDA_ARCH "75;80;86;89;90;120")
endif()

function(detect_rocm_devices)
    if(USE_ROCM)
        if(WIN32)
            find_program(ROCMINFO rocminfo PATHS $ENV{ROCM_PATH}/bin "C:/Program Files/AMD/ROCm/bin" "C:/opt/rocm/bin")
        else()
            find_program(ROCMINFO rocminfo PATHS ${ROCM_PATH}/bin /opt/rocm/bin)
        endif()
        if(ROCMINFO)
            execute_process(COMMAND ${ROCMINFO} OUTPUT_VARIABLE ROCMINFO_OUTPUT ERROR_QUIET OUTPUT_STRIP_TRAILING_WHITESPACE)
            set(DETECTED_ARCHS "")
            set(HAS_MI50 FALSE)
            string(REGEX MATCHALL "Name:[ \t]+gfx[0-9a-f]+" GFX_MATCHES "${ROCMINFO_OUTPUT}")
            foreach(MATCH ${GFX_MATCHES})
                string(REGEX REPLACE "Name:[ \t]+" "" ARCH ${MATCH})
                list(APPEND DETECTED_ARCHS ${ARCH})
                if(${ARCH} STREQUAL "gfx906")
                    set(HAS_MI50 TRUE)
                endif()
            endforeach()
            string(REGEX MATCH "Marketing Name:[ \t]+[^\n]*MI50" MI50_MATCH "${ROCMINFO_OUTPUT}")
            if(MI50_MATCH)
                set(HAS_MI50 TRUE)
            endif()
            if(DETECTED_ARCHS)
                list(REMOVE_DUPLICATES DETECTED_ARCHS)
            endif()
            set(ROCM_DETECTED_ARCHS "${DETECTED_ARCHS}" PARENT_SCOPE)
            set(ROCM_HAS_MI50 ${HAS_MI50} PARENT_SCOPE)
        else()
            message(WARNING "rocminfo not found")
            set(ROCM_DETECTED_ARCHS "" PARENT_SCOPE)
            set(ROCM_HAS_MI50 FALSE PARENT_SCOPE)
        endif()
    endif()
endfunction()

detect_rocm_devices()

if(USE_ROCM)
    if(NOT DEFINED ROCM_ARCH)
        if(ROCM_DETECTED_ARCHS)
            set(ROCM_ARCH "${ROCM_DETECTED_ARCHS}")
            message(STATUS "Auto-detected ROCm architectures: ${ROCM_ARCH}")
        else()
            set(ROCM_ARCH "gfx908;gfx90a;gfx1100")
        endif()
    else()
        if("gfx906" IN_LIST ROCM_ARCH)
            set(ROCM_HAS_MI50 TRUE)
        endif()
    endif()
    if(ROCM_HAS_MI50)
        message(STATUS "MI50 device detected (gfx906)")
    endif()
endif()

set(CMAKE_BUILD_TYPE "Release")
string(REGEX REPLACE "^([0-9]+)\\.[0-9]+\\.[0-9]+$" "\\1" CXX_MAJOR_VERSION "${CMAKE_CXX_COMPILER_VERSION}")

if (CMAKE_CXX_COMPILER_ID STREQUAL "Clang")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -pthread -O2")
elseif(CMAKE_CXX_COMPILER_ID STREQUAL "MSVC")
    string(REPLACE "/Ob2" "/Ob1 /Gy" CMAKE_CXX_FLAGS_RELEASE ${CMAKE_CXX_FLAGS_RELEASE})
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} /DNOMINMAX /std:c++20 /arch:AVX2 /source-charset:utf-8 /execution-charset:utf-8 /bigobj /MP2 /D_CRT_SECURE_NO_WARNINGS /wd4244 /wd4267 /wd4190 /wd4996")
    if (CXX_MAJOR_VERSION GREATER_EQUAL 19 AND (CMAKE_SYSTEM_PROCESSOR MATCHES "x86_64" OR CMAKE_SYSTEM_PROCESSOR MATCHES "AMD64"))
        if(MSVC_VERSION GREATER_EQUAL 1940)
            set(AVX512_FLAG "${CMAKE_CXX_FLAGS} /arch:AVX10.1")
        elseif(MSVC_VERSION GREATER_EQUAL 1930)
            set(AVX512_FLAG "${CMAKE_CXX_FLAGS} /arch:AVX512")
        else()
            set(AVX512_FLAG "${CMAKE_CXX_FLAGS} /arch:AVX2")
        endif()
        set_source_files_properties("src/devices/cpu/avx512f.cpp" PROPERTIES COMPILE_FLAGS "${AVX512_FLAG}")
        set_source_files_properties("src/devices/cpu/avx512bf16.cpp" PROPERTIES COMPILE_FLAGS "${AVX512_FLAG}")
        set_source_files_properties("src/devices/cpu/avx512vnni.cpp" PROPERTIES COMPILE_FLAGS "${AVX512_FLAG}")
        set_source_files_properties("src/devices/cpu/amx.cpp" PROPERTIES COMPILE_FLAGS "${AVX512_FLAG}")
    endif()
else()
    if (MAKE_WHL_X86)
        set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -pthread -O2 -mavx -mavx2 -mf16c -mfma -static-libstdc++ -static-libgcc -fPIC -Wl,-Bsymbolic -Wl,--exclude-libs,ALL")
    else()
        set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -pthread -O2 -march=native")
    endif()
    if (CXX_MAJOR_VERSION GREATER_EQUAL 10 AND CMAKE_SYSTEM_PROCESSOR MATCHES "x86_64")
        set_source_files_properties("src/devices/cpu/avx512f.cpp" PROPERTIES COMPILE_OPTIONS "-mavx512f;-mavx512bw;-mavx512vl")
        set_source_files_properties("src/devices/cpu/avx512bf16.cpp" PROPERTIES COMPILE_OPTIONS "-mavx512bf16;-mavx512f;-mavx512bw;-mavx512vl")
        set_source_files_properties("src/devices/cpu/avx512vnni.cpp" PROPERTIES COMPILE_OPTIONS "-mavx512vnni;-mavx512f")
        set(AMX_COMPILE_OPTIONS "-mavx512bf16;-mavx512f;-mavx512bw;-mavx512vl")
        if (CXX_MAJOR_VERSION GREATER_EQUAL 11)
            list(APPEND AMX_COMPILE_OPTIONS "-mamx-tile" "-mamx-bf16")
        endif()
        set_source_files_properties("src/devices/cpu/amx.cpp" PROPERTIES COMPILE_OPTIONS "${AMX_COMPILE_OPTIONS}")
    endif()
endif()

message(STATUS "CMAKE_CXX_FLAGS: ${CMAKE_CXX_FLAGS}")
file(GLOB GRAPH_MODEL_FILES "src/models/graph/*.cpp")
file(GLOB BLOCK_FILES "src/blocks/*.cpp")
file(GLOB CPU_DEVICE_FILES "src/devices/cpu/*.cpp")
set(FASTLLM_CXX_SOURCES src/fastllm.cpp src/device.cpp src/model.cpp src/executor.cpp src/template.cpp src/graph.cpp src/tokenizer.cpp
        src/devices/disk/diskdevice.cpp
        src/models/graphllm.cpp src/models/chatglm.cpp src/models/moss.cpp src/models/llama.cpp src/models/qwen.cpp src/models/basellm.cpp
        src/models/glm.cpp src/models/minicpm.cpp src/models/minicpm3.cpp src/models/internlm2.cpp src/models/bert.cpp src/models/moe.cpp src/models/deepseekv2.cpp src/models/deepseekv4.cpp
        src/models/phi3.cpp src/models/xlmroberta.cpp src/models/cogvlm.cpp src/models/qwen3.cpp src/models/qwen3_moe.cpp src/models/qwen3_next.cpp
        src/models/minimax.cpp src/models/hunyuan.cpp
        src/models/ernie4_5.cpp src/models/pangu_moe.cpp src/models/glm4_moe.cpp src/models/gpt_oss.cpp
        src/models/minimax_m2.cpp src/models/qwen2.cpp src/models/qwen3_5.cpp src/models/gemma4.cpp src/models/step3p5.cpp
        third_party/json11/json11.cpp
        third_party/gguf/gguf.cpp third_party/gguf/ggml-quant.cpp third_party/gguf/ggml-iqk.cpp third_party/gguf/ggml-dequantize.cpp
        third_party/gguf/gguf-adapter.cpp
        ${CPU_DEVICE_FILES}
        ${BLOCK_FILES}
        ${GRAPH_MODEL_FILES})

include_directories(include)
include_directories(include/utils)
include_directories(include/models)
include_directories(include/blocks)
include_directories(include/devices/cpu)
include_directories(include/devices/disk)
include_directories(third_party/json11)
include_directories(third_party/gguf)
include_directories(third_party/flashinfer)
include_directories(third_party/nlohmann/single_include)
include_directories(third_party/minja/include)

if (USE_MMAP)
    if(WIN32)
        message(STATUS "USE_MMAP disabled on Windows")
        set(USE_MMAP OFF)
    else()
        add_compile_definitions(USE_MMAP)
    endif()
endif()

if (USE_SENTENCEPIECE)
    add_compile_definitions(USE_SENTENCEPIECE)
    find_package(sentencepiece CONFIG QUIET)
    if(sentencepiece_FOUND)
        set(FASTLLM_LINKED_LIBS ${FASTLLM_LINKED_LIBS} sentencepiece::sentencepiece)
    else()
        set(SENTENCEPIECE_SOURCE_DIR "${CMAKE_CURRENT_SOURCE_DIR}/third_party/sentencepiece")
        if(EXISTS "${SENTENCEPIECE_SOURCE_DIR}/CMakeLists.txt")
            set(SAVED_CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS}")
            string(REPLACE "/source-charset:utf-8" "" CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS}")
            string(REPLACE "/execution-charset:utf-8" "" CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS}")
            set(SPM_ENABLE_SHARED OFF CACHE BOOL "" FORCE)
            set(SPM_ENABLE_TCMALLOC OFF CACHE BOOL "" FORCE)
            set(SPM_ENABLE_NFKC_COMPILE OFF CACHE BOOL "" FORCE)
            add_subdirectory(${SENTENCEPIECE_SOURCE_DIR} ${CMAKE_BINARY_DIR}/sentencepiece EXCLUDE_FROM_ALL)
            set(CMAKE_CXX_FLAGS "${SAVED_CMAKE_CXX_FLAGS}")
            include_directories(${SENTENCEPIECE_SOURCE_DIR}/src)
            include_directories(${CMAKE_BINARY_DIR}/sentencepiece)
            set(FASTLLM_LINKED_LIBS ${FASTLLM_LINKED_LIBS} sentencepiece-static)
        else()
            set(FASTLLM_LINKED_LIBS ${FASTLLM_LINKED_LIBS} sentencepiece)
        endif()
    endif()
endif()

if (USE_CUDA)
    enable_language(CUDA)
    if (CMAKE_BUILD_TYPE STREQUAL "Release" AND CMAKE_CUDA_FLAGS_RELEASE STREQUAL "")
        set(CMAKE_CUDA_FLAGS_RELEASE "-O3 -DNDEBUG" CACHE STRING "CUDA release flags" FORCE)
    endif()
    if ("cuda_std_20" IN_LIST CMAKE_CUDA_COMPILE_FEATURES)
        set(CMAKE_CUDA_STANDARD 20)
    else()
        set(CMAKE_CUDA_STANDARD 17)
    endif()
    set(CMAKE_CUDA_STANDARD_REQUIRED ON)
    add_compile_options($<$<COMPILE_LANGUAGE:CUDA>:--default-stream=per-thread>)
    add_compile_options($<$<COMPILE_LANGUAGE:CUDA>:--expt-relaxed-constexpr>)
    add_compile_options($<$<COMPILE_LANGUAGE:CUDA>:--expt-extended-lambda>)
    if (MSVC)
        set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -Xcompiler=/source-charset:utf-8,/execution-charset:utf-8,/Zc:preprocessor -DCCCL_IGNORE_MSVC_TRADITIONAL_PREPROCESSOR_WARNING")
    endif()
    add_compile_definitions(USE_CUDA)
    if (CUDA_NO_TENSOR_CORE)
        add_compile_definitions(CUDA_NO_TENSOR_CORE)
    endif()
    if (CUDA_MEM_DEBUG)
        add_compile_definitions(CUDA_MEM_DEBUG)
    endif()
    include_directories(include/devices/cuda)
    set(FASTLLM_CUDA_SOURCES src/devices/cuda/cudadevice.cpp src/devices/cuda/cudadevicebatch.cpp 
        src/devices/cuda/fastllm-cuda.cu src/devices/cuda/fastllm-triton-cuda.cu
        src/devices/cuda/fastllm-ggml-cuda.cu
        src/devices/cuda/attention/fastllm-attention.cu
        src/devices/cuda/attention/paged/fastllm-paged-attention-native.cu
        src/devices/cuda/linear/fastllm-linear-fp32.cu
        src/devices/cuda/linear/fastllm-linear-fp16.cu
        src/devices/cuda/linear/fastllm-linear-bf16.cu
        src/devices/cuda/linear/fastllm-linear-fp8.cu
        src/devices/cuda/moe/fastllm-moe-fp8.cu
        src/devices/cuda/linear/fastllm-linear-int8.cu
        src/devices/cuda/linear/fastllm-linear-int4.cu
        src/devices/cuda/linear/fastllm-linear-int4group.cu
        src/devices/cuda/linear/fastllm-marlin.cu
        src/devices/cuda/linear/fastllm-linear-int4nozero.cu
        src/devices/cuda/models/deepseekv4-kernels.cu)

    include_directories(${CMAKE_CURRENT_SOURCE_DIR}/third_party/turbomind)
    set(TURBOMIND_ROOT ${CMAKE_CURRENT_SOURCE_DIR}/third_party/turbomind/src/turbomind)
    set(FASTLLM_CUDA_SOURCES ${FASTLLM_CUDA_SOURCES}
        src/devices/cuda/awq_sm70/fastllm-awq-sm70.cu
        src/devices/cuda/awq_sm70/tm_registry_sm70.cu
        ${TURBOMIND_ROOT}/core/check.cc ${TURBOMIND_ROOT}/core/layout.cc
        ${TURBOMIND_ROOT}/core/context.cc ${TURBOMIND_ROOT}/core/allocator.cc
        ${TURBOMIND_ROOT}/core/buffer.cc ${TURBOMIND_ROOT}/core/stream.cc
        ${TURBOMIND_ROOT}/utils/logger.cc ${TURBOMIND_ROOT}/utils/parser.cc
        ${TURBOMIND_ROOT}/kernels/gemm/gemm.cu ${TURBOMIND_ROOT}/kernels/gemm/kernel.cu
        ${TURBOMIND_ROOT}/kernels/gemm/dispatch_cache.cu ${TURBOMIND_ROOT}/kernels/gemm/context.cu
        ${TURBOMIND_ROOT}/kernels/gemm/convert_v3.cu ${TURBOMIND_ROOT}/kernels/gemm/cast.cu
        ${TURBOMIND_ROOT}/kernels/gemm/unpack.cu
        ${TURBOMIND_ROOT}/kernels/gemm/tuner/cache_utils.cu
        ${TURBOMIND_ROOT}/kernels/gemm/tuner/measurer.cu
        ${TURBOMIND_ROOT}/kernels/gemm/tuner/sampler.cu
        ${TURBOMIND_ROOT}/kernels/gemm/tuner/stopping_criterion.cc
        ${TURBOMIND_ROOT}/kernels/gemm/tuner/params.cc
        ${TURBOMIND_ROOT}/kernels/gemm/kernel/sm70_884_4.cu
        ${TURBOMIND_ROOT}/kernels/gemm/kernel/sm70_884_8.cu
        ${TURBOMIND_ROOT}/kernels/gemm/kernel/sm70_884_16.cu)
    set_source_files_properties(
        ${TURBOMIND_ROOT}/core/check.cc ${TURBOMIND_ROOT}/core/layout.cc
        ${TURBOMIND_ROOT}/core/context.cc ${TURBOMIND_ROOT}/core/allocator.cc
        ${TURBOMIND_ROOT}/core/buffer.cc ${TURBOMIND_ROOT}/core/stream.cc
        ${TURBOMIND_ROOT}/utils/logger.cc ${TURBOMIND_ROOT}/utils/parser.cc
        ${TURBOMIND_ROOT}/kernels/gemm/tuner/stopping_criterion.cc
        ${TURBOMIND_ROOT}/kernels/gemm/tuner/params.cc
        PROPERTIES LANGUAGE CUDA)

    include_directories(include/devices/multicuda)
    set(FASTLLM_CUDA_SOURCES ${FASTLLM_CUDA_SOURCES} src/devices/multicuda/multicudadevice.cpp src/devices/multicuda/fastllm-multicuda.cu)

    add_compile_definitions(FASTLLM_SOURCE_DIR="${CMAKE_CURRENT_SOURCE_DIR}")
    set(FASTLLM_LINKED_LIBS ${FASTLLM_LINKED_LIBS} cublas nccl cuda)
    set(CMAKE_CUDA_ARCHITECTURES ${CUDA_ARCH})
endif()

if(USE_ROCM)
    if(WIN32)
        message(WARNING "ROCm support is primarily available on Linux. Windows ROCm support is experimental.")
    endif()
    list(APPEND CMAKE_MODULE_PATH "${PROJECT_SOURCE_DIR}/third_party/hipify_torch/cmake")
    include(Hipify)
    enable_language(HIP)
    file(REMOVE_RECURSE ${PROJECT_SOURCE_DIR}/src/devices/hip)
    file(REMOVE_RECURSE ${PROJECT_SOURCE_DIR}/src/devices/multihip)
    hipify(CUDA_SOURCE_DIR ${PROJECT_SOURCE_DIR}/src/devices/cuda HIP_SOURCE_DIR "${PROJECT_SOURCE_DIR}/src/devices/hip")
    hipify(CUDA_SOURCE_DIR ${PROJECT_SOURCE_DIR}/src/devices/multicuda HIP_SOURCE_DIR "${PROJECT_SOURCE_DIR}/src/devices/multihip")
    include_directories(include/devices/cuda)
    include_directories(include/devices/multicuda)
    set(FASTLLM_CUDA_SOURCES src/devices/cuda/cudadevice.cpp src/devices/cuda/cudadevicebatch.cpp 
        src/devices/hip/fastllm-hip.hip src/devices/hip/fastllm-ggml-hip.hip 
        src/devices/multicuda/multicudadevice.cpp src/devices/multihip/fastllm-multihip.hip)
    add_compile_definitions(USE_ROCM USE_CUDA)
    if (DEFINED ENV{ROCM_PATH})
        set(ROCM_PATH $ENV{ROCM_PATH})
    elseif(WIN32)
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
    set(CMAKE_PREFIX_PATH ${ROCM_PATH})
    find_package(hip REQUIRED) 
    find_package(hipblas REQUIRED)
    find_package(rocprim REQUIRED)
    list(APPEND FASTLLM_LINKED_LIBS hip::device roc::hipblas roc::rocprim)
    set(CMAKE_HIP_ARCHITECTURES ${ROCM_ARCH})
    add_compile_definitions(HIPBLAS_V2)
    if(ROCM_HAS_MI50 OR "gfx906" IN_LIST CMAKE_HIP_ARCHITECTURES)
        add_compile_definitions(USE_MI50_WORKAROUND HIP_NO_TENSOR_CORE)
    endif()
endif()

if (USE_IVCOREX)
    set(FASTLLM_LINKED_LIBS ${FASTLLM_LINKED_LIBS} cudart)
    set(CMAKE_CUDA_ARCHITECTURES ${IVCOREX_ARCH})
endif()

if (USE_TFACC)
    add_compile_definitions(USE_TFACC)
    set(FASTLLM_TFACC_SOURCES src/devices/tfacc/tfaccdevice.cpp src/devices/tfacc/fastllm-tfacc.cpp)
    include_directories(include/devices/tfacc)
endif()

if (USE_NUMAS)
    add_compile_definitions(USE_NUMAS)
    set(FASTLLM_CXX_SOURCES ${FASTLLM_CXX_SOURCES} src/devices/numas/numasdevice.cpp src/devices/numas/numas.cpp)
    include_directories(include/devices/numas)
    set(FASTLLM_LINKED_LIBS ${FASTLLM_LINKED_LIBS} numa)
endif()

if (USE_TOPS)
    add_compile_definitions(USE_TOPS)
    set(FASTLLM_TOPS_SOURCES src/devices/tops/topsdevice.cpp)
    include_directories(include/devices/tops)
endif()

if (PY_API)
    if(POLICY CMP0148)
        cmake_policy(SET CMP0148 NEW)
    endif()
    set(PYBIND third_party/pybind11)
    add_subdirectory(${PYBIND})
    find_package(Python3 REQUIRED COMPONENTS Interpreter Development)
    include_directories(third_party/pybind11/include)
    file(GLOB FASTLLM_CXX_HEADERS include/**/*.h)
    add_library(pyfastllm MODULE src/pybinding.cpp ${FASTLLM_CXX_SOURCES} ${FASTLLM_CXX_HEADERS} ${FASTLLM_CUDA_SOURCES} ${FASTLLM_TFACC_SOURCES} ${FASTLLM_TOPS_SOURCES})
    target_compile_definitions(pyfastllm PRIVATE PY_API)
    set_target_properties(pyfastllm PROPERTIES CXX_STANDARD 20 CXX_STANDARD_REQUIRED ON)
    target_link_libraries(pyfastllm PUBLIC pybind11::module ${FASTLLM_LINKED_LIBS})
    pybind11_extension(pyfastllm)
endif()

add_library(fastllm OBJECT ${FASTLLM_CXX_SOURCES} ${FASTLLM_CUDA_SOURCES} ${FASTLLM_TFACC_SOURCES} ${FASTLLM_TOPS_SOURCES})
set_target_properties(fastllm PROPERTIES CXX_STANDARD 20 CXX_STANDARD_REQUIRED ON)
target_link_libraries(fastllm PUBLIC ${FASTLLM_LINKED_LIBS})

add_executable(ftllm ftllm.cpp)
set_target_properties(ftllm PROPERTIES OUTPUT_NAME "ftllm" CXX_STANDARD 20 CXX_STANDARD_REQUIRED ON)

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
add_custom_command(TARGET webui POST_BUILD
    COMMAND ${CMAKE_COMMAND} -E make_directory web
    COMMAND ${CMAKE_COMMAND} -E copy_directory ${CMAKE_CURRENT_SOURCE_DIR}/example/webui/web ${CMAKE_BINARY_DIR}/web)

add_executable(benchmark example/benchmark/benchmark.cpp)
target_link_libraries(benchmark fastllm)

add_library(chat_handler STATIC src/chat_handler.cpp)
target_include_directories(chat_handler PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/include ${CMAKE_CURRENT_SOURCE_DIR}/third_party/nlohmann/single_include ${CMAKE_CURRENT_SOURCE_DIR}/third_party/minja/include)

add_executable(apiserver example/apiserver/apiserver.cpp)
target_link_libraries(apiserver fastllm chat_handler)

if (BUILD_STUDIO_CLI)
    add_executable(FastllmStudio_cli example/FastllmStudio/cli/cli.cpp example/FastllmStudio/cli/ui.cpp)
    target_link_libraries(FastllmStudio_cli fastllm)
endif()

add_library(fastllm_tools SHARED ${FASTLLM_CXX_SOURCES} ${FASTLLM_CUDA_SOURCES} ${FASTLLM_TFACC_SOURCES} ${FASTLLM_TOPS_SOURCES} tools/src/pytools.cpp)
set_target_properties(fastllm_tools PROPERTIES CXX_STANDARD 20 CXX_STANDARD_REQUIRED ON)
target_link_libraries(fastllm_tools PUBLIC ${FASTLLM_LINKED_LIBS})

if (${CMAKE_HOST_WIN32})
    add_custom_command(TARGET fastllm_tools POST_BUILD
        COMMAND ${CMAKE_COMMAND} -E make_directory tools
        COMMAND ${CMAKE_COMMAND} -E make_directory tools/ftllm
        COMMAND ${CMAKE_COMMAND} -E copy_directory ${CMAKE_CURRENT_SOURCE_DIR}/tools/fastllm_pytools ${CMAKE_BINARY_DIR}/tools/ftllm/.
        COMMAND ${CMAKE_COMMAND} -E copy_directory ${CMAKE_CURRENT_SOURCE_DIR}/tools/scripts ${CMAKE_BINARY_DIR}/tools/.
        COMMAND ${CMAKE_COMMAND} -E copy ${CMAKE_BINARY_DIR}/$(Configuration)/fastllm_tools.dll ${CMAKE_BINARY_DIR}/tools/ftllm/.
        COMMAND ${CMAKE_COMMAND} -E remove ${CMAKE_BINARY_DIR}/$(Configuration)/fastllm_tools.dll
    )
else()
    add_custom_command(TARGET fastllm_tools POST_BUILD
        COMMAND ${CMAKE_COMMAND} -E make_directory tools
        COMMAND ${CMAKE_COMMAND} -E make_directory tools/ftllm
        COMMAND ${CMAKE_COMMAND} -E copy_directory ${CMAKE_CURRENT_SOURCE_DIR}/tools/fastllm_pytools ${CMAKE_BINARY_DIR}/tools/ftllm/.
        COMMAND ${CMAKE_COMMAND} -E copy_directory ${CMAKE_CURRENT_SOURCE_DIR}/tools/scripts ${CMAKE_BINARY_DIR}/tools/.
        COMMAND ${CMAKE_COMMAND} -E copy ${CMAKE_BINARY_DIR}/libfastllm_tools.* ${CMAKE_BINARY_DIR}/tools/ftllm/.
        COMMAND ${CMAKE_COMMAND} -E remove ${CMAKE_BINARY_DIR}/libfastllm_tools.*
    )
endif()
