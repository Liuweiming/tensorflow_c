// Adapted from serizba/cppflow https://github.com/serizba/cppflow
// First created by sergio on 12/05/19.
// Reconstructed by Weiming Liu in 04/05/20.

#ifndef TENSORFLOW_C_MODEL_H
#define TENSORFLOW_C_MODEL_H

#include <tensorflow/c/c_api.h>

#include <algorithm>
#include <cstring>
#include <fstream>
#include <iostream>
#include <string>
#include <tuple>
#include <vector>

#include "tensor.h"

namespace tf_cpp {

class Model {
 public:
  explicit Model(const std::string&);

  Model(const Model& model) = delete;
  Model(Model&& model) = default;
  Model& operator=(const Model& model) = delete;
  Model& operator=(Model&& model) = default;

  ~Model();

  TF_Graph* get_graph() { return graph; }
  void restore(const std::string& ckpt);
  void save(const std::string& ckpt);
  void save_graph(const std::string& graph_path);
  std::vector<std::string> get_operations() const;

  // inputs should containts datas for evaluating.
  // outputs and operations will be evaluated.
  // after that, the users can access outputs' data.
  // we does not need to know the type of Tensors, so we use a std::variant
  // tensor_type
  void run(const std::vector<Tensor*>& inputs,
           const std::vector<Tensor*>& outputs,
           const std::vector<TF_Operation*>& operations = {}) {
    // Get input operations
    std::vector<TF_Output> io(inputs.size());
    std::transform(inputs.begin(), inputs.end(), io.begin(),
                   [](auto i) { return i->tf_op; });

    // Get input values
    std::vector<TF_Tensor*> iv(inputs.size());
    std::transform(inputs.begin(), inputs.end(), iv.begin(),
                   [](auto i) { return i->tf_tensor; });

    // Get output operations
    std::vector<TF_Output> oo(outputs.size());
    std::transform(outputs.begin(), outputs.end(), oo.begin(),
                   [](auto o) { return o->tf_op; });

    // Get output values
    std::vector<TF_Tensor*> ov(outputs.size());
    auto tf_code =
        tf_utils::RunSession(session, io, iv, oo, ov, operations, status);
    if (tf_code != TF_OK) {
      throw std::runtime_error(tf_utils::CodeToString(tf_code));
    }
    // Save results on outputs
    // must not delete ov, as it will be used by outputs.
    for (std::size_t i = 0; i < outputs.size(); i++) {
      outputs[i]->set_tensor(ov[i]);
    }
  }

  void run_operation(TF_Operation* op) { run({}, {}, {op}); }

 private:
  TF_Status* status;
  TF_Graph* graph;
  TF_SessionOptions* opts;
  TF_Session* session;

  // Read a file from a string
  static TF_Buffer* read(const std::string&);

  bool status_check(bool throw_exc) const;
  void error_check(bool condition, const std::string& error) const;
};
}  // namespace tf_cpp
#endif  // TENSORFLOW_C_MODEL_H
