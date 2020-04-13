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

  Tensor input_a{model.get_graph(), "input_a"};
  Tensor input_b{model.get_graph(), "input_b"};
  Tensor output{model.get_graph(), "result"};

  std::vector<float> data(100);
  std::iota(data.begin(), data.end(), 0);

  input_a.set_data(data);
  input_b.set_data(data);

  model.run({&input_a, &input_b}, {&output});
  for (float f : output.get_data<float>()) {
    std::cout << f << " ";
  }
  std::cout << std::endl;
}
