add_executable(save_and_restore main.cc
    $<TARGET_OBJECTS:tensorflow_c>)
target_include_directories(save_and_restore PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})

file(COPY ${CMAKE_CURRENT_SOURCE_DIR}/graph.pb DESTINATION ${CMAKE_CURRENT_BINARY_DIR})