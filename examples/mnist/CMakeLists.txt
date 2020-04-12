find_package( OpenCV REQUIRED )
add_executable(mnist main.cpp 
    $<TARGET_OBJECTS:tensorflow_c>)
target_include_directories(mnist PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})
target_link_libraries (mnist ${OpenCV_LIBS})

file(COPY ${CMAKE_CURRENT_SOURCE_DIR}/checkpoint DESTINATION ${CMAKE_CURRENT_BINARY_DIR})
file(COPY ${CMAKE_CURRENT_SOURCE_DIR}/images DESTINATION ${CMAKE_CURRENT_BINARY_DIR})
file(COPY ${CMAKE_CURRENT_SOURCE_DIR}/model.pb DESTINATION ${CMAKE_CURRENT_BINARY_DIR})
