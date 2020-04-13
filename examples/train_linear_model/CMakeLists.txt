add_executable(example_train example.c)
target_include_directories(example_train PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})
target_link_libraries(example_train tensorflow)

add_executable(train_model main.cc
    $<TARGET_OBJECTS:tensorflow_c>)
target_include_directories(train_model PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})

file(COPY ${CMAKE_CURRENT_SOURCE_DIR}/graph.pb DESTINATION ${CMAKE_CURRENT_BINARY_DIR})