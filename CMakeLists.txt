cmake_minimum_required(VERSION 3.16)

project(Fortune)

set(CMAKE_CXX_STANDARD 20)

add_executable(Fortune Fortune/main.cpp)

find_package(Python3 COMPONENTS Interpreter Development REQUIRED)
if(WIN32 AND MSVC)
    target_link_libraries(Fortune ${Python3_LIBRARY_RELEASE})
else()
    target_link_libraries(Fortune ${Python3_LIBRARIES})
endif()

add_subdirectory(pybind11)
include_directories(${pybind11_INCLUDE_DIRS})
