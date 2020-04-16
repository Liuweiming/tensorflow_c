// Adapted from serizba/cppflow https://github.com/serizba/cppflow
// First created by sergio on 12/05/19.
// Reconstructed by Weiming Liu in 04/05/20.

#include "model.h"
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

void Model::save_graph(const std::string& graph_path) {
  auto tf_code =
      tf_utils::DumpGraph(graph, session, graph_path.c_str(), status);
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

void Model::error_check(bool condition, const std::string& error) const {
  if (!condition) {
    throw std::runtime_error(error);
  }
}
}  // namespace tf_cpp