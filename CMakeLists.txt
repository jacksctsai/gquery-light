cmake_minimum_required(VERSION 3.5)

project(gquery_light VERSION 0.1.0 LANGUAGES CXX)

set(CMAKE_OSX_ARCHITECTURES "arm64")

add_executable(gquery
    src/main.cpp
    src/csv_reader.cpp
    src/filter_cpu.cpp
    src/binary_io.cpp
)

target_include_directories(gquery
    PRIVATE
        include
)

add_executable(test_smoke
    tests/test_smoke.cpp
    src/csv_reader.cpp
    src/filter_cpu.cpp
    src/binary_io.cpp
)

target_include_directories(test_smoke
    PRIVATE
        include
)

if (MSVC)
    target_compile_options(gquery PRIVATE /W4 /permissive-)
    target_compile_options(test_smoke PRIVATE /W4 /permissive-)
else()
    target_compile_options(gquery PRIVATE -Wall -Wextra -Wpedantic -Wconversion -Wshadow -std=c++20)
    target_compile_options(test_smoke PRIVATE -Wall -Wextra -Wpedantic -Wconversion -Wshadow -std=c++20)
endif()
