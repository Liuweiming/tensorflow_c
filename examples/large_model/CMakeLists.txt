add_executable(large_model main.cpp 
    $<TARGET_OBJECTS:tensorflow_c>)
target_include_directories(large_model PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})

file(COPY ${CMAKE_CURRENT_SOURCE_DIR}/graph.pb DESTINATION ${CMAKE_CURRENT_BINARY_DIR})
