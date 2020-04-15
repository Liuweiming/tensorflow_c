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
#include <typeinfo>
#include <variant>
#include <vector>

#include "tf_utils.h"

#define MAX_DIMS 10

namespace tf_cpp {

class Model;
using tensor_type =
    std::variant<float, double, int8_t, int16_t, int32_t, int64_t, uint8_t,
                 uint16_t, uint32_t, uint64_t, bool>;

template <typename T>
std::string to_string(const std::vector<T> &vec) {
  std::string ret = "[";
  for (int i = 0; i != vec.size(); ++i) {
    ret += std::to_string(vec[i]) + ((i == vec.size() - 1) ? "]" : " ");
  }
  return ret;
}

template <typename T>
class Tensor {
 public:
  // shape and type are used to verify the shape and dtype of tf_tensor.
  Tensor(TF_Graph *graph, const std::string &oper_name,
         const std::vector<int64_t> &shape, const TF_DataType &dtype)
      : status(nullptr), tf_tensor(nullptr) {
    status = TF_NewStatus();
    int n_dims;
    int64_t dims[MAX_DIMS];
    auto tf_code = tf_utils::GetTGraphOperation(
        graph, oper_name.c_str(), &tf_op, &tf_type, &n_dims, dims, status);
    if (tf_code != TF_OK) {
      throw std::runtime_error("tf_utils::GetTGraphOperation error.");
    }
    if (tf_type != dtype) {
      throw std::runtime_error(
          "dtype is incompatible with tf_tensor data type. [" +
          tf_utils::DataTypeToString(dtype) + " vs. " +
          tf_utils::DataTypeToString(tf_type) + "].");
    }
    if (deduce_type() != tf_type) {
      throw std::runtime_error(
          "can not access tf_tensor in this type. tf_tensor type is " +
          tf_utils::DataTypeToString(tf_type) + ".");
    }
    tf_shape = std::vector<int64_t>(dims, dims + n_dims);
    if (shape.size() != n_dims) {
      throw std::runtime_error(
          std::string(
              "data's dimension is incompatible with tf_tensor dimensions. [") +
          std::to_string(shape.size()) + " vs. " + std::to_string(n_dims) +
          "].");
    }
    for (int i = 0; i != n_dims; ++i) {
      if (tf_shape[i] != shape[i] && tf_shape[i] != -1) {
        throw std::runtime_error(
            std::string(
                "data's shape is incompatible with tf_tensor shape. [") +
            to_string(shape) + " vs. " + to_string(tf_shape) + "].");
      }
      tf_shape[i] = shape[i];
    }
  }
  // move only.
  Tensor(const Tensor &tensor) = delete;
  Tensor(Tensor &&tensor) = default;
  Tensor &operator=(const Tensor &tensor) = delete;
  Tensor &operator=(Tensor &&tensor) = default;

  ~Tensor() {
    if (tf_tensor != nullptr) {
      TF_DeleteTensor(tf_tensor);
    }
    if (status != nullptr) {
      TF_DeleteStatus(status);
    }
  }

  // access tf_tensor as type T.
  T &operator()(const std::vector<std::size_t> &indexs) {
    if (tf_tensor == nullptr) {
      create_tensor();
    }
    if (indexs.size() > tf_shape.size()) {
      throw std::runtime_error("indexs dimension is larger than tf_tensor. [" +
                               std::to_string(indexs.size()) + " vs. " +
                               std::to_string(tf_shape.size()) + "].");
    }
    std::size_t linear_index = 0;
    for (int i = 0; i != indexs.size(); ++i) {
      if (indexs[i] >= tf_shape[i]) {
        throw std::runtime_error(
            "index at dimention " + std::to_string(i) +
            " is out of range the size of the dimension. [" +
            std::to_string(indexs[i]) + " .vs " + std::to_string(tf_shape[i]) +
            "].");
      }
      for (int j = i + 1; j < tf_shape.size(); ++j) {
        linear_index += indexs[i] * tf_shape[j];
      }
    }
    return *(static_cast<T *>(TF_TensorData(tf_tensor)) + linear_index);
  }

  std::vector<int64_t> get_shape() { return tf_shape; }

 private:
  // create tf_tensor, should be called only once.
  // could be called to reset the shape of tf_tensor.
  void create_tensor() {
    if (tf_tensor != nullptr) {
      TF_DeleteTensor(tf_tensor);
    }
    int data_size = 1;
    for (auto &s : tf_shape) {
      data_size *= abs(s);
    }
    tf_tensor =
        tf_utils::CreateEmptyTensor(tf_type, tf_shape, data_size * sizeof(T));
    if (tf_tensor == nullptr) {
      throw std::runtime_error("tf_utils::CreateTensor error");
    }
  }

  TF_DataType deduce_type() {
    // we don't support bool type, please do not use TF_BOOL in tensorflow.
    // (std::is_same<T, bool>::value) return TF_BOOL;
    if (std::is_same<T, float>::value) return TF_FLOAT;
    if (std::is_same<T, double>::value) return TF_DOUBLE;
    if (std::is_same<T, int8_t>::value) return TF_INT8;
    if (std::is_same<T, int16_t>::value) return TF_INT16;
    if (std::is_same<T, int32_t>::value) return TF_INT32;
    if (std::is_same<T, int64_t>::value) return TF_INT64;
    if (std::is_same<T, uint8_t>::value) return TF_UINT8;
    if (std::is_same<T, uint16_t>::value) return TF_UINT16;
    if (std::is_same<T, uint32_t>::value) return TF_UINT32;
    if (std::is_same<T, uint64_t>::value) return TF_UINT64;

    throw std::runtime_error{"Could not deduce type!"};
  }

  // set tf_tensor from new_tensor.
  // useful for accessing data from session out.
  // should only be called by Model.
  void set_tensor(TF_Tensor *new_tensor) {
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
      tf_shape.reserve(n_dims);
      for (int i = 0; i < n_dims; i++) {
        tf_shape[i] = TF_Dim(tf_tensor, i);
      }
    }
  }

 private:
  TF_Status *status;
  TF_Tensor *tf_tensor;
  TF_Output tf_op;
  TF_DataType tf_type;
  std::vector<int64_t> tf_shape;

 public:
  friend class Model;
};
}  // namespace tf_cpp
#endif  // TENSORFLOW_C_TENSOR_H
