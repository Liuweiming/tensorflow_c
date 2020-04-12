find_package( OpenCV REQUIRED )
add_executable(mnist main.cpp ../../src/Model.cpp ../../src/Tensor.cpp)
target_include_directories(mnist PRIVATE ../../include)
target_link_libraries (mnist -ltensorflow ${OpenCV_LIBS})