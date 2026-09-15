cmake_minimum_required(VERSION 3.18)

project(CPPNN2 LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

set(SOURCES
    main.cpp
    neuralarchitecture.cpp
    denselayer.cpp
    activationlayer.cpp

    ${CMAKE_SOURCE_DIR}/../src/library.cpp
)

add_executable(${PROJECT_NAME} ${SOURCES})

target_include_directories(${PROJECT_NAME} PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}
    ${CMAKE_CURRENT_SOURCE_DIR}/../include
)
