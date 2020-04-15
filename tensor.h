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

template <typename T>
std::string to_string(const std::vector<T> &vec) {
  std::string ret = "[";
  for (int i = 0; i != vec.size(); ++i) {
    ret += std::to_string(vec[i]) + ((i == vec.size() - 1) ? "]" : " ");
  }
  return ret;
}

class Tensor {
 public:
  // shape and type are used to verify the shape and dtype of tf_tensor.
  Tensor(TF_Graph *graph, const std::string &oper_name,
         const std::vector<int64_t> &shape, const TF_DataType &dtype);
  // move only.
  Tensor(const Tensor &tensor) = delete;
  Tensor(Tensor &&tensor) = default;
  Tensor &operator=(const Tensor &tensor) = delete;
  Tensor &operator=(Tensor &&tensor) = default;

  ~Tensor();

  // access tf_tensor as type T.
  // access by vector or {ix0, ix1, ... ixn}.
  template <typename T>
  T &at(const std::vector<int> &indexs) {
    if (tf_tensor == nullptr) {
      create_tensor<T>();
    }
    if (indexs.size() > tf_shape.size()) {
      throw std::runtime_error("indexs dimension is larger than tf_tensor. [" +
                               std::to_string(indexs.size()) + " vs. " +
                               std::to_string(tf_shape.size()) + "].");
    }
    if (deduce_type<T>() != tf_type) {
      throw std::runtime_error(
          "can not access tf_tensor in this type. tf_tensor type is " +
          tf_utils::DataTypeToString(tf_type) + ".");
    }
    int linear_index = 0;
    for (int i = 0; i != indexs.size(); ++i) {
      if (indexs[i] >= tf_shape[i]) {
        throw std::runtime_error(
            "index at dimention " + std::to_string(i) +
            " is out of range the size of the dimension. [" +
            std::to_string(indexs[i]) + " .vs " + std::to_string(tf_shape[i]) +
            "].");
      }
      int prod = 1;
      for (int j = i + 1; j < tf_shape.size(); ++j) {
        prod *= tf_shape[j];
      }
      linear_index += indexs[i] * prod;
    }
    return *(static_cast<T *>(TF_TensorData(tf_tensor)) + linear_index);
  }

  template <typename T, typename... Types>
  T &at(std::vector<int> indexs, int ixn, Types... rest) {
    indexs.push_back(ixn);
    // std::cout << to_string(indexs) << std::endl;
    return at<T>(indexs, rest...);
  }

  // access tf_tensor by (ix0, ix1, ... ixn)
  template <typename T, typename... Types>
  T &at(int ix0, Types... rest) {
    // std::cout << ix0 << std::endl;
    return at<T>(std::vector<int>{ix0}, rest...);
  }

  template <typename T>
  T &at() {
    return at<T>(std::vector<int>());
  }

  template <typename T>
  T &at(int idx0) {
    return at<T>(std::vector<int>{idx0});
  }

  template <typename T>
  T &at(int idx0, int idx1) {
    return at<T>(std::vector<int>{idx0, idx1});
  }

  template <typename T>
  T &at(int idx0, int idx1, int idx2) {
    return at<T>(std::vector<int>{idx0, idx1, idx2});
  }

  template <typename T>
  T &at(int idx0, int idx1, int idx2, int idx3) {
    return at<T>(std::vector<int>{idx0, idx1, idx2, idx3});
  }

  std::vector<int64_t> shape() { return tf_shape; }
  std::size_t dim() { return tf_shape.size(); }

 private:
  // create tf_tensor, should be called only once.
  // could be called to reset the shape of tf_tensor.
  template <typename T>
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

  template <typename T>
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
  void set_tensor(TF_Tensor *new_tensor);

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
