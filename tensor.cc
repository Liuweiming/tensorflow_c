//
// Created by sergio on 13/05/19.
//

#include "tensor.h"

#include <sstream>
#include <string>
#include <utility>
#include <vector>

#include "model.h"
#include "tf_utils.h"

namespace tf_cpp {
Tensor::Tensor(const Model &model, const std::string &oper_name)
    : status(nullptr), tf_tensor(nullptr), data_size(0) {
  status = TF_NewStatus();
  auto tf_code = tf_utils::GetTGraphOperation(
      model.graph, oper_name.c_str(), &op, &type, &n_dims, dims, status);
  if (tf_code != TF_OK) {
    throw std::runtime_error("tf_utils::GetTGraphOperation error.");
  }
  shape = std::vector<int64_t>(dims, dims + n_dims);
  tf_tensor = nullptr;
}

Tensor::~Tensor() {
  if (tf_tensor != nullptr) {
    TF_DeleteTensor(tf_tensor);
  }
  if (status != nullptr) {
    TF_DeleteStatus(status);
  }
}

void Tensor::set_tensor(TF_Tensor *new_tensor) {
  if (tf_tensor != nullptr) {
    TF_DeleteTensor(tf_tensor);
  }
  // some operations may have null tensor.
  // for example, the training_op.
  if (new_tensor == nullptr) {
    tf_tensor = nullptr;
    n_dims = 0;
    shape.clear();
    actual_shape.clear();
    return;
  }
  // new_tensors must not be deleted.
  tf_tensor = new_tensor;
  if (tf_tensor == nullptr) {
    throw std::runtime_error("TF_TensorMaybeMove error");
  }
  // op should not be modified.
  // set type.
  type = TF_TensorType(tf_tensor);
  // set n_dims, dims and shape.
  int n_dims = TF_NumDims(tf_tensor);
  if (n_dims <= 0 || n_dims > MAX_DIMS) {
    throw std::runtime_error("TF_NumDims error");
  }
  shape.clear();
  if (n_dims > 0) {
    shape.reserve(n_dims);
    for (int i = 0; i < n_dims; i++) {
      dims[i] = TF_Dim(tf_tensor, i);
      shape[i] = dims[i];
    }
    actual_shape = shape;
  }
}

std::vector<int64_t> Tensor::get_shape() { return shape; }
}  // namespace tf_cpp