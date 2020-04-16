//
// Created by sergio on 16/05/19.
//

#include <iomanip>
#include <numeric>

#include "model.h"
#include "tensor.h"

using namespace tf_cpp;

void load_and_run(const std::string &path, bool restore, bool training,
                  bool save) {
  Model model(path);

  // std::cout << "operations: -----------" << std::endl;
  // for (auto &op : model.get_operations()) {
  //   std::cout << op << std::endl;
  // }
  // std::cout << "-------------------" << std::endl;

  TF_Operation *init = TF_GraphOperationByName(model.get_graph(), "init");
  TF_Operation *train = TF_GraphOperationByName(model.get_graph(), "train");

  Tensor input(model.get_graph(), "input", {3, 1, 1}, TF_FLOAT);
  Tensor label(model.get_graph(), "target", {3, 1, 1}, TF_FLOAT);
  Tensor predict(model.get_graph(), "output", {3, 1, 1}, TF_FLOAT);
  Tensor loss(model.get_graph(), "loss", {}, TF_FLOAT);

  model.run_operation(init);

  if (restore) {
    model.restore("save.ckpt");
  }

  int bs = 3;
  std::vector<float> train_inputs(bs);
  std::vector<float> train_labels(bs);
  for (int i = 0; i != bs; ++i) {
    train_inputs[i] = i;
    train_labels[i] = 0.5 * train_inputs[i] - 1;
  }

  for (int i = 0; i != 3; ++i) {
    input.at<float>(i, 0, 0) = train_inputs[i];
    label.at<float>(i, 0, 0) = train_labels[i];
  }

  std::cout << "before training" << std::endl;
  model.run({&input}, {&predict});
  for (int i = 0; i != 3; ++i) {
    std::cout << predict.at<float>(i, 0, 0) << " ";
  }
  std::cout << std::endl;
  if (training) {
    for (int iter = 0; iter != 100; ++iter) {
      std::cout << "iteration " << iter << std::endl;
      model.run({&input, &label}, {&predict, &loss}, {train});
      float train_loss = loss.at<float>();
      std::cout << "loss: " << train_loss << std::endl;
      for (float f : train_labels) {
        std::cout << f << " ";
      }
      std::cout << std::endl;
      for (int i = 0; i != 3; ++i) {
        std::cout << predict.at<float>(i, 0, 0) << " ";
      }
      std::cout << std::endl;
    }
  }
  if (save) {
    model.save_graph("save.pb");
    model.save("save.ckpt");
  }
}

int main() {
  load_and_run("graph.pb", false, true, true);
  // save graph, restore variables.
  load_and_run("graph.pb", true, false, false);
  // load graph, restore variables.
  load_and_run("save.pb", true, false, false);
}
