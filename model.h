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

// all types that are possible in Tensorflow.
using var_t =
    std::variant<std::vector<float>, std::vector<double>, std::vector<int8_t>,
                 std::vector<int16_t>, std::vector<int32_t>,
                 std::vector<int64_t>, std::vector<uint8_t>,
                 std::vector<uint16_t>, std::vector<uint32_t>,
                 std::vector<uint64_t>, std::vector<bool>>;
class Model;

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
    // user wants. It is recreated if the data_size is changed, because of, for
    // example, the batch size is changed.
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

  void Model::run(const std::vector<std::string>& outputs,
                  const std::vector<std::string>& operations,
                  const std::map<std::string, var_t>& feed_dict,
                  std::map<std::string, var_t>& output_dict) {
    for (auto& op_name : outputs) {
      register_tensor(op_name);
    }
    for (auto& op_name : operations) {
      register_operation(op_name);
    }

    for (auto& feed : feed_dict) {
      register_tensor(feed.first);
      auto& tm = tensor_map[feed.first].tf_tensor;
      std::visit([tm](auto& x) { tm.set_data(x); }, feed.second);
    }
    std::vector<TF_Output> input_ops(feed_dict.size());
    std::vector<TF_Tensor*> input_tensors(feed_dict.size());
    std::vector<TF_Output> output_ops(outputs.size());
    std::vector<TF_Operation*> ops(operations.size());
    int i = 0;
    for (auto feed_iter = feed_dict.begin(); feed_iter != feed_dict.end();
         ++i, ++feed_iter) {
      input_ops[i] = tensor_map[feed_iter->first].op;
      input_tensors[i] = tensor_map[feed_iter->first].tf_tensor;
    }
    for (std::size_t i = 0; i != outputs.size(); ++i) {
      output_ops[i] = tensor_map[outputs[i]].op;
    }
    for (std::size_t i = 0; i != operations.size(); ++i) {
      ops[i] = operation_map[operations[i]];
    }

    // Get output values
    std::vector<TF_Tensor*> output_tensors(outputs.size());
    auto tf_code =
        tf_utils::RunSession(session, input_ops, input_tensors, output_ops,
                             output_tensors, ops, status);
    if (tf_code != TF_OK) {
      throw std::runtime_error(tf_utils::CodeToString(tf_code));
    }
    // Save results on outputs
    // Must not delete ov, as it will be used by outputs.
    std::map<std::string, std::vector<var_t>> ret_outputs;
    for (std::size_t i = 0; i < outputs.size(); i++) {
      tensor_map[outputs[i]].set_tensor(output_tensors[i]);
      tensor_map[outputs[i]].get_data(output_dict[outputs[i]]);
    }
  }

 private:
  TF_Status* status;
  TF_Graph* graph;
  TF_SessionOptions* opts;
  TF_Session* session;
  std::map<std::string, Tensor> tensor_map;
  std::map<std::string, TF_Operation*> operation_map;

  inline void register_tensor(const std::string& op_name) {
    if (tensor_map.find(op_name) != tensor_map.end()) {
      return;
    }
    tensor_map.emplace(op_name, Tensor{graph, op_name});
  }

  inline void register_operation(const std::string& op_name) {
    if (operation_map.find(op_name) != operation_map.end()) {
      return;
    }
    operation_map.emplace(op_name,
                          TF_GraphOperationByName(graph, op_name.c_str()));
  }

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
