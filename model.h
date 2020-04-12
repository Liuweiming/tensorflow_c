//
// Created by sergio on 12/05/19.
//

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

namespace tf_cpp {
class Tensor;

class Model {
 public:
  explicit Model(const std::string&);

  Model(const Model& model) = delete;
  Model(Model&& model) = default;
  Model& operator=(const Model& model) = delete;
  Model& operator=(Model&& model) = default;

  ~Model();

  void restore(const std::string& ckpt);
  void save(const std::string& ckpt);
  std::vector<std::string> get_operations() const;

  // Original Run
  void run(const std::vector<Tensor*>& inputs,
           const std::vector<Tensor*>& outputs);

  // Run with references
  void run(Tensor& input, const std::vector<Tensor*>& outputs);
  void run(const std::vector<Tensor*>& inputs, Tensor& output);
  void run(Tensor& input, Tensor& output);

  // Run with pointers
  void run(Tensor* input, const std::vector<Tensor*>& outputs);
  void run(const std::vector<Tensor*>& inputs, Tensor* output);
  void run(Tensor* input, Tensor* output);

 private:
  TF_Status* status;
  TF_Graph* graph;
  TF_SessionOptions* opts;
  TF_Session* session;

  // Read a file from a string
  static TF_Buffer* read(const std::string&);

  bool status_check(bool throw_exc) const;
  void error_check(bool condition, const std::string& error) const;

 public:
  friend class Tensor;
};
}  // namespace tf_cpp
#endif  // TENSORFLOW_C_MODEL_H
