add_executable(coco main.cpp ../../src/Model.cpp ../../src/Tensor.cpp)
find_package( OpenCV REQUIRED )
target_include_directories(coco PRIVATE ../../include)
target_link_libraries (coco -ltensorflow ${OpenCV_LIBS})