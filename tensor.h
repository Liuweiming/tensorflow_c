// Wrapper of tf.tensor.
// Adapted from sergio.
//
// Created by sergio on 13/05/19.
//

#ifndef CPPFLOW_TENSOR_H
#define CPPFLOW_TENSOR_H

#include <tensorflow/c/c_api.h>

#include <algorithm>
#include <cstring>
#include <memory>
#include <numeric>
#include <string>
#include <vector>

#include "model.h"

class Model;

class Tensor {
 public:
  Tensor(const Model &model, const std::string &operation);

  Tensor(const Tensor &tensor) = delete;
  Tensor(Tensor &&tensor) = default;
  Tensor &operator=(const Tensor &tensor) = delete;
  Tensor &operator=(Tensor &&tensor) = default;

  ~Tensor();

  void clean();

  template <typename T>
  void set_data(std::vector<T> new_data);

  template <typename T>
  void set_data(std::vector<T> new_data, const std::vector<int64_t> &new_shape);

  template <typename T>
  std::vector<T> get_data();

  std::vector<int64_t> get_shape();

 private:
  TF_Tensor *tf_tensor;
  TF_Status *status;
  TF_Output op;
  TF_DataType type;
  std::vector<int64_t> shape;
  std::unique_ptr<std::vector<int64_t>> actual_shape;
  template <typename T>
  static TF_DataType deduce_type();

  void deduce_shape();

 public:
  friend class Model;
};

#endif  // CPPFLOW_TENSOR_H
