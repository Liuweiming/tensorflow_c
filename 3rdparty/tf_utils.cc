#include "tf_utils.h"

#include <tensorflow/c/c_api.h>

#include <cstring>
#include <fstream>
#include <iostream>

namespace tf_tuils {
namespace {

static void DeallocateBuffer(void *data, size_t) { std::free(data); }

static TF_Buffer *ReadBufferFromFile(const char *file) {
  std::ifstream f(file, std::ios::binary);
  if (f.fail() || !f.is_open()) {
    return nullptr;
  }

  if (f.seekg(0, std::ios::end).fail()) {
    return nullptr;
  }
  auto fsize = f.tellg();
  if (f.seekg(0, std::ios::beg).fail()) {
    return nullptr;
  }

  if (fsize <= 0) {
    return nullptr;
  }

  auto data = static_cast<char *>(std::malloc(fsize));
  if (f.read(data, fsize).fail()) {
    return nullptr;
  }

  auto buf = TF_NewBuffer();
  buf->data = data;
  buf->length = fsize;
  buf->data_deallocator = DeallocateBuffer;

  f.close();
  return buf;
}

TF_Tensor *ScalarStringTensor(const std::string &str, TF_Status *status) {
  auto str_len = str.size();
  auto nbytes =
      8 + TF_StringEncodedSize(str_len);  // 8 extra bytes - for start_offset.
  auto tensor = TF_AllocateTensor(TF_STRING, nullptr, 0, nbytes);
  auto data = static_cast<char *>(TF_TensorData(tensor));
  std::memset(data, 0, 8);
  TF_StringEncode(str.c_str(), str_len, data + 8, nbytes - 8, status);
  return tensor;
}

TF_Graph *LoadGraph(const std::string &path, const std::string &check_point) {
  auto buffer = ReadBufferFromFile(path.c_str());
  if (buffer == nullptr) {
    return nullptr;
  }
  auto status = TF_NewStatus();
  auto graph = TF_NewGraph();
  auto opts = TF_NewImportGraphDefOptions();

  TF_GraphImportGraphDef(graph, buffer, opts, status);
  TF_DeleteImportGraphDefOptions(opts);
  TF_DeleteBuffer(buffer);

  if (TF_GetCode(status) != TF_OK) {
    DeleteGraph(graph);
    graph = nullptr;
  }
  TF_DeleteStatus(status);
  return graph;

  // if (check_point.size() == 0) {
  //   return graph;
  // }

  // auto checkpoint_tensor = ScalarStringTensor(check_point, status);

  // if (TF_GetCode(status) != TF_OK) {
  //   DeleteGraph(graph);
  //   return nullptr;
  // }

  // TF_DeleteTensor(checkpoint_tensor);
}

TF_Graph *LoadGraph(const std::string &path) { return LoadGraph(path, ""); }

void DeleteGraph(TF_Graph *graph) {
  if (graph != nullptr) {
    TF_DeleteGraph(graph);
  }
}

TF_Session *CreateSession(TF_Graph *graph, TF_SessionOptions *options) {
  if (graph == nullptr) {
    return nullptr;
  }

  auto status = TF_NewStatus();

  bool need_delete_option = false;
  if (options == nullptr) {
    options = TF_NewSessionOptions();
    need_delete_option = true;
  }
  auto session = TF_NewSession(graph, options, status);
  if (TF_GetCode(status) != TF_OK) {
    DeleteSession(session);
    session = nullptr;
  }

  TF_DeleteStatus(status);
  if (need_delete_option) {
    TF_DeleteSessionOptions(options);
  }
  return session;
}

TF_Session *CreateSession(TF_Graph *graph) {
  return CreateSession(graph, nullptr);
}

TF_Code DeleteSession(TF_Session *session) {
  if (session == nullptr) {
    return TF_INVALID_ARGUMENT;
  }
  TF_Code ret;
  auto status = TF_NewStatus();
  TF_CloseSession(session, status);
  if (TF_GetCode(status) != TF_OK) {
    ret = TF_GetCode(status);
  } else {
    TF_DeleteSession(session, status);
    if (TF_GetCode(status) != TF_OK) {
      ret = TF_GetCode(status);
    }
  }
  TF_DeleteStatus(status);
  return ret;
}

TF_Code RunSession(TF_Session *session, const TF_Output *inputs,
                   TF_Tensor *const *input_tensors, std::size_t ninputs,
                   const TF_Output *outputs, TF_Tensor **output_tensors,
                   std::size_t noutputs) {
  if (session == nullptr || inputs == nullptr || input_tensors == nullptr ||
      outputs == nullptr || output_tensors == nullptr) {
    return TF_INVALID_ARGUMENT;
  }

  auto status = TF_NewStatus();

  TF_SessionRun(
      session,
      nullptr,  // Run options.
      inputs, input_tensors,
      static_cast<int>(
          ninputs),  // Input tensors, input tensor values, number of inputs.
      outputs, output_tensors,
      static_cast<int>(noutputs),  // Output tensors, output tensor values,
                                   // number of outputs.
      nullptr, 0,                  // Target operations, number of targets.
      nullptr,                     // Run metadata.
      status                       // Output status.
  );
  TF_DeleteStatus(status);
  return TF_GetCode(status);
}

TF_Code RunSession(TF_Session *session, const std::vector<TF_Output> &inputs,
                   const std::vector<TF_Tensor *> &input_tensors,
                   const std::vector<TF_Output> &outputs,
                   std::vector<TF_Tensor *> &output_tensors) {
  return RunSession(session, inputs.data(), input_tensors.data(),
                    input_tensors.size(), outputs.data(), output_tensors.data(),
                    output_tensors.size());
}

TF_Tensor *CreateTensor(TF_DataType data_type, const std::int64_t *dims,
                        std::size_t num_dims, const void *data,
                        std::size_t len);
}  // namespace
}  // namespace tf_tuils