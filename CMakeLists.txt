cmake_minimum_required(VERSION 3.0)

project(TensorflowC
    VERSION 0.0.1
    DESCRIPTION "A test of tensorflow c api."
    LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_VERBOSE_MAKEFILE ON)
set (CMAKE_CXX_FLAGS "-Wno-everything")

add_executable(hello_tf 
    hello_tf.cc tf_utils.h tf_utils.cc
    tensor.h tensor.cc model.h model.cc)
target_link_libraries(hello_tf tensorflow)
