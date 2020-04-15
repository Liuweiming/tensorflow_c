#include "tensor.h"

#include <sstream>
#include <string>
#include <utility>
#include <vector>

#include "model.h"
#include "tf_utils.h"

namespace tf_cpp {

Tensor::Tensor(TF_Graph *graph, const std::string &oper_name,
               const std::vector<int64_t> &shape, const TF_DataType &dtype)
    : status(nullptr), tf_tensor(nullptr) {
  status = TF_NewStatus();
  int n_dims;
  int64_t dims[MAX_DIMS];
  auto tf_code = tf_utils::GetTGraphOperation(graph, oper_name.c_str(), &tf_op,
                                              &tf_type, &n_dims, dims, status);
  if (tf_code != TF_OK) {
    throw std::runtime_error("tf_utils::GetTGraphOperation error.");
  }
  if (tf_type != dtype) {
    throw std::runtime_error(
        "dtype is incompatible with tf_tensor data type. [" +
        tf_utils::DataTypeToString(dtype) + " vs. " +
        tf_utils::DataTypeToString(tf_type) + "].");
  }
  tf_shape = std::vector<int64_t>(dims, dims + n_dims);
  if (shape.size() != n_dims) {
    throw std::runtime_error(
        std::string("data's dimension is incompatible with tf_tensor "
                    "dimensions: [") +
        std::to_string(shape.size()) + " vs. " + std::to_string(n_dims) + "]," +
        "shape: [" + to_string(shape) + " vs. " + to_string(tf_shape) + "].");
  }
  for (int i = 0; i != n_dims; ++i) {
    if (tf_shape[i] != shape[i] && tf_shape[i] != -1) {
      throw std::runtime_error(
          std::string("data's shape is incompatible with tf_tensor shape. [") +
          to_string(shape) + " vs. " + to_string(tf_shape) + "].");
    }
    tf_shape[i] = shape[i];
  }
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
    tf_shape.clear();
    return;
  }
  // new_tensors must not be deleted.
  tf_tensor = new_tensor;
  if (tf_tensor == nullptr) {
    throw std::runtime_error("TF_TensorMaybeMove error");
  }
  // op should not be modified.
  // set type.
  tf_type = TF_TensorType(tf_tensor);
  // set n_dims, dims and shape.
  int n_dims = TF_NumDims(tf_tensor);
  if (n_dims < 0 || n_dims > MAX_DIMS) {
    throw std::runtime_error("TF_NumDims error");
  }
  tf_shape.clear();
  if (n_dims > 0) {
    tf_shape.resize(n_dims);
    for (int i = 0; i < n_dims; i++) {
      tf_shape[i] = TF_Dim(tf_tensor, i);
    }
  }
  // std::cout << n_dims << " " << to_string(tf_shape) << std::endl;
}
}  // namespace tf_cpp