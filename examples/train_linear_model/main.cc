//
// Created by sergio on 16/05/19.
//

#include <iomanip>
#include <numeric>

#include "model.h"

using namespace tf_cpp;

int main() {
  Model model("/mnt/c/Users/liuwe/Documents/project/tensorflow_c/graph.pb");
  model.register_operations({"init", "train"});
  model.register_tensors({"input", "target", "output", "loss"});

  std::cout << "operations: -----------" << std::endl;
  for (auto &op : model.get_operations()) {
    std::cout << op << std::endl;
  }
  std::cout << "-------------------" << std::endl;

  model.run({}, {}, {"init"});

  int bs = 3;
  std::vector<float> train_inputs(bs);
  std::vector<float> train_labels(bs);
  for (int i = 0; i != bs; ++i) {
    train_inputs[i] = (rand() % 10000) / 10000.0;
    std::cout << train_inputs[i] << " ";
    train_labels[i] = 0.5 * train_inputs[i] - 1;
  }
  std::cout << "]" << std::endl;

  model.set_data("input", train_inputs);
  model.set_data("target", train_labels);

  std::cout << "before training" << std::endl;
  model.run({"input"}, {"output"});
  for (float f : model.get_data<float>("output")) {
    std::cout << f << " ";
  }

  for (int iter = 0; iter != 1000; ++iter) {
    std::cout << "iteration " << iter << std::endl;
    model.run({"input", "label"}, {"output", "loss"}, {"train"});
    float train_loss = model.get_data<float>("loss")[0];
    std::cout << "loss: " << train_loss << std::endl;
    for (float f : train_labels) {
      std::cout << f << " ";
    }
    std::cout << std::endl;
    for (float f : model.get_data<float>("output")) {
      std::cout << f << " ";
    }
    std::cout << std::endl;
  }
}
