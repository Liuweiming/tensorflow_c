// Wrapper of tf.tensor.
// Adapted from sergio.
//
// Created by sergio on 13/05/19.
//

#ifndef TENSORFLOW_C_TENSOR_H
#define TENSORFLOW_C_TENSOR_H

#include <tensorflow/c/c_api.h>

#include <algorithm>
#include <cstring>
#include <memory>
#include <numeric>
#include <string>
#include <vector>

#include "tf_utils.h"
#define MAX_DIMS 10

namespace tf_cpp {

class Model;

class Tensor {
 public:
  Tensor(const Model &model, const std::string &operation);

  Tensor(const Tensor &tensor) = delete;
  Tensor(Tensor &&tensor) = default;
  Tensor &operator=(const Tensor &tensor) = delete;
  Tensor &operator=(Tensor &&tensor) = default;

  ~Tensor();

  // create tf_tensor, should be called only once.
  // could be called to reset the shape of tf_tensor.
  template <typename T>
  void create_tensor(const std::vector<T> &new_data) {
    if (tf_tensor != nullptr) {
      TF_DeleteTensor(tf_tensor);
    }
    // calcualate actual_shape based on new_data.
    auto exp_size = std::abs(std::accumulate(shape.begin(), shape.end(), 1,
                                             std::multiplies<int64_t>()));
    actual_shape = shape;
    std::replace_if(
        actual_shape.begin(), actual_shape.end(),
        [](int64_t r) { return r == -1; }, new_data.size() / exp_size);
    tf_tensor = tf_utils::CreateTensor(type, actual_shape, new_data);
    if (tf_tensor == nullptr) {
      throw std::runtime_error("tf_utils::CreateTensor error");
    }
  }

  // set tf_tensor from new_tensor.
  // useful for accessing data from session out.
  void set_tensor(TF_Tensor *new_tensor);

  // set tf_tensor from new_data.
  // new_data should be compatible with the shape of tf_tensor.
  template <typename T>
  void set_data(const std::vector<T> &new_data) {
    tf_utils::SetTensorData<T>(tf_tensor, new_data);
  }

  template <typename T>
  std::vector<T> get_data() {
    return tf_utils::GetTensorData<T>(tf_tensor);
  }

  std::vector<int64_t> get_shape();

 private:
  TF_Status *status;
  TF_Tensor *tf_tensor;
  TF_Output op;
  TF_DataType type;
  int n_dims;
  int64_t *dims;
  std::vector<int64_t> shape;
  std::vector<int64_t> actual_shape;

 public:
  friend class Model;
};
}  // namespace tf_cpp
#endif  // TENSORFLOW_C_TENSOR_H
