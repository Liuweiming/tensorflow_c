//
// Created by sergio on 12/05/19.
//

#ifndef TENSORFLOW_C_MODEL_H
#define TENSORFLOW_C_MODEL_H

#include <tensorflow/c/c_api.h>

#include <algorithm>
#include <cstring>
#include <fstream>
#include <functional>
#include <iostream>
#include <map>
#include <string>
#include <tuple>
#include <variant>
#include <vector>

#define MAX_DIMS 10

namespace tf_cpp {

class Model {
 private:
  class Tensor {
   public:
    Tensor(TF_Graph* graph, const std::string& operation);

    Tensor(const Tensor& tensor) = delete;
    Tensor(Tensor&& tensor) = default;
    Tensor& operator=(const Tensor& tensor) = delete;
    Tensor& operator=(Tensor&& tensor) = default;

    ~Tensor();

    // set tf_tensor from new_data.
    // new_data should be compatible with the shape of tf_tensor.
    template <typename T>
    void set_data(const std::vector<T>& new_data, bool reset = false) {
      // If it is the first time to call set_data, i.e., tf_tensor is nullptr,
      // try to create a tensor by create_tensor. May recreate the tensor if the
      // user wants. It is recreated if the data_size is changed, because of,
      // for example, the batch size is changed.
      if (tf_tensor == nullptr || reset || new_data.size() != data_size) {
        create_tensor<T>(new_data);
      } else {
        tf_utils::SetTensorData<T>(tf_tensor, new_data);
      }
    }

    template <typename T>
    std::vector<T> get_data() {
      return tf_utils::GetTensorData<T>(tf_tensor);
    }

    template <typename T>
    void get_data(std::vector<T> ret_data) {
      return tf_utils::GetTensorData<T>(tf_tensor, ret_data);
    }

    std::vector<int64_t> get_shape();

   private:
    // create tf_tensor, should be called only once.
    // could be called to reset the shape of tf_tensor.
    template <typename T>
    void create_tensor(const std::vector<T>& new_data) {
      if (tf_tensor != nullptr) {
        TF_DeleteTensor(tf_tensor);
      }
      data_size = new_data.size();
      // calcualate actual_shape based on new_data.
      auto exp_size = std::abs(std::accumulate(shape.begin(), shape.end(), 1,
                                               std::multiplies<int64_t>()));
      actual_shape = shape;
      std::replace_if(actual_shape.begin(), actual_shape.end(),
                      [](int64_t r) { return r == -1; },
                      new_data.size() / exp_size);
      tf_tensor = tf_utils::CreateTensor(type, actual_shape, new_data);
      for (auto& as : actual_shape) {
        std::cout << as << " ";
      }
      std::cout << "]" << std::endl;
      if (tf_tensor == nullptr) {
        throw std::runtime_error("tf_utils::CreateTensor error");
      }
    }

    // set tf_tensor from new_tensor.
    // useful for accessing data from session out.
    // should only be called by Model.
    void set_tensor(TF_Tensor* new_tensor);

   private:
    TF_Status* status;
    TF_Tensor* tf_tensor;
    TF_Output op;
    TF_DataType type;
    int n_dims;
    int64_t dims[MAX_DIMS];
    size_t data_size;
    std::vector<int64_t> shape;
    std::vector<int64_t> actual_shape;

   public:
    friend class Model;
  };

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
  Model& register_tensor(const std::string& op_name);
  Model& register_operation(const std::string& op_name);
  Model& register_tensors(const std::vector<std::string>& op_names) {
    for (auto& op : op_names) {
      register_tensor(op);
    }
    return *this;
  }
  Model& register_operations(const std::vector<std::string>& op_names) {
    for (auto& op : op_names) {
      register_operation(op);
    }
    return *this;
  }

  template <typename T>
  std::vector<T> get_data(const std::string& op_name) {
    return tensor_map[op_name].get_data<T>();
  }
  template <typename T>
  void set_data(const std::string& op_name, const std::vector<T>& data) {
    tensor_map[op_name].set_data<T>(data);
  }
  // Original Run
  void run(const std::vector<std::string>& inputs,
           const std::vector<std::string>& outputs,
           const std::vector<std::string>& operations = {});

 private:
  TF_Status* status;
  TF_Graph* graph;
  TF_SessionOptions* opts;
  TF_Session* session;

  std::map<std::string, Tensor> tensor_map;
  std::map<std::string, TF_Operation*> operation_map;

  TF_Graph* get_graph() { return graph; }

  // Read a file from a string
  static TF_Buffer* read(const std::string&);

  bool status_check(bool throw_exc) const;
  void error_check(bool condition, const std::string& error) const;

 public:
  friend class Tensor;
};
}  // namespace tf_cpp
#endif  // TENSORFLOW_C_MODEL_H
