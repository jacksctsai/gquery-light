cmake_minimum_required(VERSION 3.5)

project(gquery_light VERSION 0.1.0 LANGUAGES CXX)

set(CMAKE_OSX_ARCHITECTURES "arm64")

# --- CUDA Detection ---
include(CheckLanguage)
check_language(CUDA)

if(CMAKE_CUDA_COMPILER)
    enable_language(CUDA)
    message(STATUS "CUDA found: ${CMAKE_CUDA_COMPILER}")
    set(CUDA_ENABLED TRUE)
else()
    message(STATUS "CUDA NOT found. GPU mode will be disabled.")
    set(CUDA_ENABLED FALSE)
endif()

# --- Source Files ---
set(COMMON_SOURCES
    src/csv_reader.cpp
    src/filter_cpu.cpp
    src/binary_io.cpp
)

set(GQUERY_SOURCES src/main.cpp ${COMMON_SOURCES})
if(CUDA_ENABLED)
    list(APPEND GQUERY_SOURCES src/filter_gpu.cu)
endif()

# --- Executables ---
add_executable(gquery ${GQUERY_SOURCES})
target_include_directories(gquery PRIVATE include)

add_executable(test_smoke tests/test_smoke.cpp ${COMMON_SOURCES})
target_include_directories(test_smoke PRIVATE include)

# --- Compiler Options ---
if (MSVC)
    target_compile_options(gquery PRIVATE /W4 /permissive-)
    target_compile_options(test_smoke PRIVATE /W4 /permissive-)
else()
    target_compile_options(gquery PRIVATE -Wall -Wextra -Wpedantic -Wconversion -Wshadow -std=c++20)
    target_compile_options(test_smoke PRIVATE -Wall -Wextra -Wpedantic -Wconversion -Wshadow -std=c++20)
endif()

if(CUDA_ENABLED)
    target_compile_definitions(gquery PRIVATE GQUERY_USE_CUDA)
    # Needed because `gquery` C++ sources include CUDA runtime headers (pinned host allocator).
    target_include_directories(gquery PRIVATE /usr/local/cuda/include)
    # See README: match GQUERY_CUDA_ARCH to your GPU (e.g. 75 for compute capability 7.5) or use native locally.
    set(GQUERY_CUDA_ARCH "80" CACHE STRING "CUDA sm_XX (e.g. 75, 80, 89) or native")
    set_target_properties(gquery PROPERTIES CUDA_ARCHITECTURES "${GQUERY_CUDA_ARCH}")
endif()
