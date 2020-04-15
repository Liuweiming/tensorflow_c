//
// Created by sergio on 16/05/19.
//

#include <iomanip>
#include <numeric>

#include "model.h"
#include "tensor.h"

using namespace tf_cpp;

int main() {
  Model model("load_model.pb");

  std::cout << "operations: -----------" << std::endl;
  for (auto &op : model.get_operations()) {
    std::cout << op << std::endl;
  }
  std::cout << "-------------------" << std::endl;

  Tensor input_a(model.get_graph(), "input_a", {1, 100}, TF_FLOAT);
  Tensor input_b(model.get_graph(), "input_b", {1, 100}, TF_FLOAT);
  Tensor output(model.get_graph(), "result", {1, 100}, TF_FLOAT);

  std::vector<float> data(100);
  std::iota(data.begin(), data.end(), 0);
  for (std::size_t i = 0; i != data.size(); ++i) {
    input_a.at<float>(0, i) = data[i];
    input_b.at<float>(0, i) = data[i];
  }

  model.run({&input_a, &input_b}, {&output});
  for (std::size_t i = 0; i != data.size(); ++i) {
    std::cout << output.at<float>(0, i) << " ";
  }
  std::cout << std::endl;
}
