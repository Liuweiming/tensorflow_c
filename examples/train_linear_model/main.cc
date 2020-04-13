//
// Created by sergio on 16/05/19.
//

#include <iomanip>
#include <numeric>

#include "model.h"
#include "tensor.h"

using namespace tf_cpp;

int main() {
  Model model("/mnt/c/Users/liuwe/Documents/project/tensorflow_c/graph.pb");

  std::cout << "operations: -----------" << std::endl;
  for (auto &op : model.get_operations()) {
    std::cout << op << std::endl;
  }
  std::cout << "-------------------" << std::endl;

  TF_Operation *init = TF_GraphOperationByName(model.get_graph(), "init");
  TF_Operation *train = TF_GraphOperationByName(model.get_graph(), "train");

  Tensor input{model.get_graph(), "input"};
  Tensor label{model.get_graph(), "target"};
  Tensor predict{model.get_graph(), "output"};
  Tensor loss{model.get_graph(), "loss"};

  model.run_operation(init);

  int bs = 3;
  std::vector<float> train_inputs(bs);
  std::vector<float> train_labels(bs);
  for (int i = 0; i != bs; ++i) {
    train_inputs[i] = (rand() % 10000) / 10000.0;
    std::cout << train_inputs[i] << " ";
    train_labels[i] = 0.5 * train_inputs[i] - 1;
  }
  std::cout << "]" << std::endl;

  input.set_data(train_inputs);
  label.set_data(train_labels);

  std::cout << "before training" << std::endl;
  model.run({&input}, {&predict});
  for (float f : predict.get_data<float>()) {
    std::cout << f << " ";
  }

  for (int iter = 0; iter != 10; ++iter) {
    std::cout << "iteration " << iter << std::endl;
    model.run({&input, &label}, {&predict, &loss}, {train});
    float train_loss = loss.get_data<float>()[0];
    std::cout << "loss: " << train_loss << std::endl;
    for (float f : train_labels) {
      std::cout << f << " ";
    }
    std::cout << std::endl;
    for (float f : predict.get_data<float>()) {
      std::cout << f << " ";
    }
    std::cout << std::endl;
  }
}
