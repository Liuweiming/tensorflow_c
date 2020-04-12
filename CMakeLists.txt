cmake_minimum_required(VERSION 3.0)

project(TensorflowC)

set(CMAKE_CXX_STANDARD 17)
# set(CMAKE_VERBOSE_MAKEFILE ON)
set (CMAKE_CXX_FLAGS "-Wno-everything")

add_library(tensorflow_c OBJECT
    scope_guard.h tf_utils.h tf_utils.cc
    model.h model.cc tensor.h tensor.cc)
target_include_directories(tensorflow_c PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})
link_libraries(tensorflow)

# examples
include_directories(${CMAKE_CURRENT_SOURCE_DIR})
add_subdirectory(examples/load_model)
add_subdirectory(examples/mnist)
add_subdirectory(examples/large_model)
add_subdirectory(test)