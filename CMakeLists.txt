# Specify the minimum version of CMake required
cmake_minimum_required(VERSION 3.10)

# Name of the project
project(CPPNN)

# Specify the C++ standard
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED True)

# Add the executable target and source files
add_executable(CPPNN
    src/dataloader.cpp
    src/library.cpp
    src/logging.cpp
    src/main.cpp
    src/neuralnetwork.cpp
    src/stringtech.cpp
)

# Include directories for header files
target_include_directories(CPPNN PRIVATE include)