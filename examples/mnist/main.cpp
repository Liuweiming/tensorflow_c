//
// Created by sergio on 12/05/19.
//

#include <algorithm>
#include <iterator>
#include <opencv2/opencv.hpp>
#include "model.h"
#include "tensor.h"

using namespace tf_cpp;

int main() {
  // Create model
  Model m("model.pb");
  m.restore("checkpoint/train.ckpt");

  std::cout << "operations: -----------" << std::endl;
  for (auto &op : m.get_operations()) {
    std::cout << op << std::endl;
  }
  std::cout << "-------------------" << std::endl;

  // Create Tensors
  Tensor input(m.get_graph(), "input", {1, 784}, TF_DOUBLE);
  std::cout << tf_cpp::to_string(input.shape()) << std::endl;
  Tensor prediction(m.get_graph(), "prediction", {1, 10}, TF_DOUBLE);
  std::cout << tf_cpp::to_string(prediction.shape()) << std::endl;

  // Read image
  for (int i = 0; i < 10; i++) {
    cv::Mat img, scaled;

    // Read image
    img = cv::imread("images/" + std::to_string(i) + ".png");

    // Scale image to range 0-1
    img.convertTo(scaled, CV_64F, 1.f / 255);

    // Put image in vector
    std::vector<double> img_data;
    img_data.assign(scaled.begin<double>(), scaled.end<double>());

    // Feed data to input tensor
    for (std::size_t i = 0; i != img_data.size(); ++i) {
      input.at<double>(0, i) = img_data[i];
    }

    // Run and show predictions
    m.run({&input}, {&prediction});

    // Get tensor with predictions
    std::vector<double> result;
    std::cout << tf_cpp::to_string(prediction.shape()) << std::endl;
    for (std::size_t i = 0; i != 10; ++i) {
      result.push_back(prediction.at<double>(0, i));
    }

    // Maximum prob
    auto max_result = std::max_element(result.begin(), result.end());

    // Print result
    std::cout << "Real label: " << i
              << ", predicted: " << std::distance(result.begin(), max_result)
              << ", Probability: " << (*max_result) << std::endl;
  }
}
