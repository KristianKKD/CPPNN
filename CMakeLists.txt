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

    src/io.cpp
    src/logging.cpp
    src/library.cpp
    src/neuralnetwork.cu
    src/stringtech.cpp
    src/unittests.cu

    src/Implementations/PPOGridWorld.cu
    # src/Implementations/ChatBot.cu
)

#executable + sources
add_executable(${PROJECT_NAME} ${SOURCES})

#include directories
target_include_directories(${PROJECT_NAME} PRIVATE include)

#to prevent conflicts, set flags to use the correct dll
# set(CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE} /MD") #release
set(CMAKE_CXX_FLAGS_DEBUG "${CMAKE_CXX_FLAGS_DEBUG} /MDd") #debug