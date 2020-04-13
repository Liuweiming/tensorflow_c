//
// Created by sergio on 12/05/19.
//

#include "model.h"
#include "tf_utils.h"

namespace tf_cpp {

Model::Tensor::Tensor(TF_Graph* graph, const std::string& oper_name)
    : status(nullptr), tf_tensor(nullptr), data_size(0) {
  status = TF_NewStatus();
  auto tf_code = tf_utils::GetTGraphOperation(graph, oper_name.c_str(), &op,
                                              &type, &n_dims, dims, status);
  if (tf_code != TF_OK) {
    throw std::runtime_error("tf_utils::GetTGraphOperation error.");
  }
  shape = std::vector<int64_t>(dims, dims + n_dims);
  tf_tensor = nullptr;
}

Model::Tensor::~Tensor() {
  if (tf_tensor != nullptr) {
    TF_DeleteTensor(tf_tensor);
  }
  if (status != nullptr) {
    TF_DeleteStatus(status);
  }
}

void Model::Tensor::set_tensor(TF_Tensor* new_tensor) {
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
  if (n_dims < 0 || n_dims > MAX_DIMS) {
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

std::vector<int64_t> Model::Tensor::get_shape() { return shape; }

Model::Model(const std::string& model_filename)
    : status(nullptr), graph(nullptr), opts(nullptr), session(nullptr) {
  status = TF_NewStatus();
  graph = tf_utils::LoadGraph(model_filename.c_str(), status);
  if (graph == nullptr) {
    throw std::runtime_error("tf_utils::LoadGraph error");
  }

  // Create the session.
  opts = tf_utils::CreateSessionOptions(1, 1, status);
  if (opts == nullptr) {
    throw std::runtime_error("tf_utils::CreateSessionOptions error");
  }

  session = tf_utils::CreateSession(graph, opts, status);
  if (session == nullptr) {
    throw std::runtime_error("tf_utils::CreateSession error");
  }
}

Model::~Model() {
  if (opts != nullptr) {
    TF_DeleteSessionOptions(opts);
  }
  if (graph != nullptr) {
    TF_DeleteGraph(graph);
  }
  if (session != nullptr) {
    TF_DeleteSession(session, status);
  }
  TF_DeleteStatus(status);
}

void Model::save(const std::string& ckpt) {
  auto tf_code = tf_utils::Save(graph, session, ckpt.c_str(), status);
  if (tf_code != TF_OK) {
    throw std::runtime_error("tf_utils::Save error");
  }
}

void Model::restore(const std::string& ckpt) {
  auto tf_code = tf_utils::Restore(graph, session, ckpt.c_str(), status);
  if (tf_code != TF_OK) {
    throw std::runtime_error("tf_utils::Restore error");
  }
}

std::vector<std::string> Model::get_operations() const {
  std::vector<std::string> result;
  size_t pos = 0;
  TF_Operation* oper;

  // Iterate through the operations of a graph
  while ((oper = TF_GraphNextOperation(graph, &pos)) != nullptr) {
    result.emplace_back(TF_OperationName(oper));
  }

  return result;
}

Model& Model::register_tensor(const std::string& op_name) {
  if (tensor_map.find(op_name) != tensor_map.end()) {
    return;
  }
  tensor_map.emplace(op_name, Tensor{graph, op_name});
  return *this;
}
Model& Model::register_operation(const std::string& op_name) {
  if (operation_map.find(op_name) != operation_map.end()) {
    return;
  }
  operation_map.emplace(op_name,
                        TF_GraphOperationByName(graph, op_name.c_str()));
  return *this;
}

void Model::run(const std::vector<std::string>& inputs,
                const std::vector<std::string>& outputs,
                const std::vector<std::string>& operations) {
  register_tensors(inputs);
  register_tensors(outputs);
  register_operations(operations);
  std::vector<TF_Output> input_ops(inputs.size());
  std::vector<TF_Tensor*> input_tensors(inputs.size());
  std::vector<TF_Output> output_ops(outputs.size());
  std::vector<TF_Operation*> ops(operations.size());
  for (std::size_t i = 0; i != inputs.size(); ++i) {
    input_ops[i] = tensor_map[inputs[i]].op;
    input_tensors[i] = tensor_map[inputs[i]].tf_tensor;
  }
  for (std::size_t i = 0; i != outputs.size(); ++i) {
    output_ops[i] = tensor_map[outputs[i]].op;
  }
  for (std::size_t i = 0; i != operations.size(); ++i) {
    ops[i] = operation_map[operations[i]];
  }

  // Get output values
  std::vector<TF_Tensor*> output_tensors(outputs.size());
  auto tf_code = tf_utils::RunSession(session, input_ops, input_tensors,
                                      output_ops, output_tensors, ops, status);
  if (tf_code != TF_OK) {
    throw std::runtime_error(tf_utils::CodeToString(tf_code));
  }
  // Save results on outputs
  // Must not delete ov, as it will be used by outputs.
  for (std::size_t i = 0; i < outputs.size(); i++) {
    tensor_map[outputs[i]].set_tensor(output_tensors[i]);
  }
}

void Model::run(const std::vector<Tensor*>& inputs,
                const std::vector<Tensor*>& outputs,
                const std::vector<TF_Operation*>& operations) {
  // Get input operations
  std::vector<TF_Output> io(inputs.size());
  std::transform(inputs.begin(), inputs.end(), io.begin(),
                 [](const Tensor* i) { return i->op; });

  // Get input values
  std::vector<TF_Tensor*> iv(inputs.size());
  std::transform(inputs.begin(), inputs.end(), iv.begin(),
                 [](const Tensor* i) { return i->tf_tensor; });

  // Get output operations
  std::vector<TF_Output> oo(outputs.size());
  std::transform(outputs.begin(), outputs.end(), oo.begin(),
                 [](const Tensor* o) { return o->op; });

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

void Model::error_check(bool condition, const std::string& error) const {
  if (!condition) {
    throw std::runtime_error(error);
  }
}
}  // namespace tf_cpp