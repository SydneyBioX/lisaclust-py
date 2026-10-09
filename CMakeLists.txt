cmake_minimum_required(VERSION 3.18)
project(lisaclust LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

find_package(Python COMPONENTS Interpreter Development.Module REQUIRED)
find_package(pybind11 CONFIG REQUIRED)

file(GLOB CORE_SOURCES ${CMAKE_CURRENT_SOURCE_DIR}/src/core/src/*.cpp)
pybind11_add_module(_core src/bindings.cpp ${CORE_SOURCES})
target_include_directories(_core PRIVATE src/core/include)
# M_PI on MSVC
target_compile_definitions(_core PRIVATE _USE_MATH_DEFINES)
install(TARGETS _core DESTINATION lisaclust)
