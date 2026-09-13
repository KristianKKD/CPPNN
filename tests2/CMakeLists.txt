cmake_minimum_required(VERSION 3.18)

project(CPPNNTests LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# GoogleTest
include(FetchContent)

FetchContent_Declare(
    googletest
    URL https://github.com/google/googletest/archive/refs/tags/v1.15.2.zip
)

set(gtest_force_shared_crt ON CACHE BOOL "" FORCE)
FetchContent_MakeAvailable(googletest)

enable_testing()

# Test executable
add_executable(CPPNNTests
    densetests.cpp
    ../src2/denselayer.cpp
    ../src/library.cpp
)

target_include_directories(CPPNNTests PRIVATE 
    ../include
    ../src2
)

target_link_libraries(CPPNNTests PRIVATE GTest::gtest_main)

include(GoogleTest)
gtest_discover_tests(CPPNNTests)
