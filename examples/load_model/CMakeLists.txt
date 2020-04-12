add_executable(load_model main.cc
    $<TARGET_OBJECTS:tensorflow_c>)
target_include_directories(load_model PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})
