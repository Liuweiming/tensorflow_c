add_executable(hello_tf hello_tf.cpp
    $<TARGET_OBJECTS:tensorflow_c>)

add_executable(session_run session_run.cpp
    $<TARGET_OBJECTS:tensorflow_c>)

add_executable(load_graph load_graph.cpp
    $<TARGET_OBJECTS:tensorflow_c>)

add_executable(interface interface.cpp
    $<TARGET_OBJECTS:tensorflow_c>)

add_executable(graph_info graph_info.cpp
    $<TARGET_OBJECTS:tensorflow_c>)

add_executable(create_tensor create_tensor.cpp
    $<TARGET_OBJECTS:tensorflow_c>)

add_executable(tensor_info tensor_info.cpp
    $<TARGET_OBJECTS:tensorflow_c>)

add_executable(allocate_tensor allocate_tensor.cpp
    $<TARGET_OBJECTS:tensorflow_c>)

add_executable(batch_interface batch_interface.cpp
    $<TARGET_OBJECTS:tensorflow_c>)

configure_file(models/graph.pb ${CMAKE_CURRENT_BINARY_DIR}/graph.pb COPYONLY)


