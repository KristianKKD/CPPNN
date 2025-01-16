cmake_minimum_required(VERSION 3.18)

#name
project(CPPNN LANGUAGES CXX CUDA)

#standards
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

#properties
enable_language(CUDA)

#files
set(SOURCES
    src/main.cu

    src/dataloader.cpp
    src/library.cu
    src/logging.cpp
    src/neuralnetwork.cu
    src/stringtech.cpp
    src/unittests.cu
)

#executable + sources
add_executable(${PROJECT_NAME} ${SOURCES})

#include directories
target_include_directories(${PROJECT_NAME} PRIVATE include)
