#include <tensorflow/c/c_api.h>

#include <string>
#include <vector>

namespace tf_utils {

TF_Graph *LoadGraph(const std::string &path, const std::string &check_point);

TF_Graph *LoadGraph(const std::string &path);

void DeleteGraph(TF_Graph *graph);

TF_Session *CreateSession(TF_Graph *graph, TF_SessionOptions *options);

TF_Session *CreateSession(TF_Graph *graph);

TF_Session *CreateSession();

TF_Code DeleteSession(TF_Session *session);

TF_Code RunSession(TF_Session *session, const std::vector<TF_Output> &inputs,
                   const std::vector<TF_Tensor *> &input_tensors,
                   const std::vector<TF_Output> &outputs,
                   std::vector<TF_Tensor *> &output_tensors);

TF_Tensor *CreateTensor(TF_DataType data_type, const std::int64_t *dims,
                        std::size_t num_dims, const void *data,
                        std::size_t len);

template <typename T>
TF_Tensor *CreateTensor(TF_DataType data_type,
                        const std::vector<std::int64_t> &dims,
                        const std::vector<T> &data) {
  return CreateTensor(data_type, dims.data(), dims.size(), data.data(),
                      data.size() * sizeof(T));
}

TF_Tensor *CreateEmptyTensor(TF_DataType data_type, const std::int64_t *dims,
                             std::size_t num_dims, std::size_t len = 0);

TF_Tensor *CreateEmptyTensor(TF_DataType data_type,
                             const std::vector<std::int64_t> &dims,
                             std::size_t len = 0);

void DeleteTensor(TF_Tensor *tensor);

void DeleteTensors(const std::vector<TF_Tensor *> &tensors);

bool SetTensorData(TF_Tensor *tensor, const void *data, std::size_t len);

template <typename T>
void SetTensorData(TF_Tensor *tensor, const std::vector<T> &data) {
  SetTensorsData(tensor, data.data(), data.size() * sizeof(T));
}

template <typename T>
std::vector<T> GetTensorData(const TF_Tensor *tensor) {
  if (tensor == nullptr) {
    return {};
  }
  auto data = static_cast<T *>(TF_TensorData(tensor));
  auto size =
      TF_TensorByteSize(tensor) / TF_DataTypeSize(TF_TensorType(tensor));
  if (data == nullptr || size <= 0) {
    return {};
  }
  return {data, data + size};
}

std::vector<std::int64_t> GetTensorShape(TF_Graph *graph,
                                         const TF_Output &output);

TF_SessionOptions *CreateSessionOptions();

void DeleteSessionOptions(TF_SessionOptions *options);

std::string DataTypeToString(TF_DataType data_type);

}  // namespace tf_utils