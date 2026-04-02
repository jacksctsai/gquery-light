cmake_minimum_required(VERSION 3.5)

project(gquery_light VERSION 0.1.0 LANGUAGES CXX)

set(CMAKE_OSX_ARCHITECTURES "arm64")

# --- CUDA Detection ---
include(CheckLanguage)
check_language(CUDA)

if(CMAKE_CUDA_COMPILER)
    enable_language(CUDA)
    message(STATUS "CUDA found: ${CMAKE_CUDA_COMPILER}")
    add_definitions(-DGQUERY_USE_CUDA)
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

# --- Executables ---
add_executable(gquery src/main.cpp ${COMMON_SOURCES})
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
    set_target_properties(gquery PROPERTIES CUDA_ARCHITECTURES "native")
    set_target_properties(test_smoke PROPERTIES CUDA_ARCHITECTURES "native")
endif()
