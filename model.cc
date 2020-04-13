//
// Created by sergio on 12/05/19.
//

#include "model.h"

#include "tensor.h"
#include "tf_utils.h"

namespace tf_cpp {

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