//
// Created by sergio on 13/05/19.
//

#include "Tensor.h"
#include <tensorflow/c/c_api.h>
#include <utility>
#include <vector>
#include <string>
#include <sstream>

inline void error_check(bool condition, const std::string &error)
{
    if (!condition)
    {
        throw std::runtime_error(error);
    }
}

Tensor::Tensor(const Model &model, const std::string &operation)
{

    // Get operation by the name
    op.oper = TF_GraphOperationByName(model.graph, operation.c_str());
    op.index = 0;
    if (op.oper == nullptr)
    {
        std::stringstream error_ss;
        error_ss << "TF_GraphOperationByName error: no operaion named:"
                 << operation;
        throw std::runtime_error(error_ss.str());
    }
    int n_dims = TF_GraphGetTensorNumDims(model.graph, op, status);
    if (TF_GetCode(status) != TF_OK)
    {
        throw std::runtime_error("TF_GraphGetTensorNumDims error");
    }
    type = TF_OperationOutputType(op);

    // If is not a scalar
    if (n_dims > 0)
    {
        // Get dimensions
        auto *dims = new int64_t[n_dims];
        TF_GraphGetTensorShape(model.graph, op, dims, n_dims, status);
        shape = std::vector<int64_t>(dims, dims + n_dims);
        delete[] dims;
        if (TF_GetCode(status) != TF_OK)
        {
            throw std::runtime_error("TF_GraphGetTensorShape error");
        }
    }
    tf_tensor = nullptr;
    data = nullptr;
}

Tensor::~Tensor()
{
    if (tf_tensor != nullptr)
    {
        TF_DeleteTensor(tf_tensor);
    }
    if (status != nullptr)
    {
        TF_DeleteStatus(status);
    }
}

template <typename T>
void Tensor::set_data(std::vector<T> new_data)
{
    if (tf_tensor != nullptr)
    {
        TF_DeleteTensor(tf_tensor);
    }
    tf_tensor = TF_AllocateTensor(type)
}

template <typename T>
void Tensor::set_data(std::vector<T> new_data, const std::vector<int64_t> &new_shape)
{

    error_check(shape.empty() || shape.size() == new_shape.size(), "Provided shape has different number of dimensions");
    auto old_shape = shape;

    shape = new_shape;
    set_data(new_data);

    shape = old_shape;
}

template <typename T>
std::vector<T> Tensor::get_data()
{

    // Check Tensor is tf_tensorid
    error_check(flag != -1, "Tensor is not tf_tensorid");

    // Check type
    error_check(deduce_type<T>() == type, "Expected return type is different from Tensor type");

    // Tensor is not empty
    error_check(flag != 0, "Tensor is empty");

    // Check tensor data is not empty
    auto raw_data = TF_TensorData(tf_tensor);
    error_check(raw_data != nullptr, "Tensor data is empty");

    size_t size = TF_TensorByteSize(tf_tensor) / TF_DataTypeSize(TF_TensorType(tf_tensor));

    // Convert to correct type
    const auto T_data = static_cast<T *>(raw_data);
    return std::vector<T>(T_data, T_data + size);
}

std::vector<int64_t> Tensor::get_shape()
{
    return shape;
}

template <typename T>
TF_DataType Tensor::deduce_type()
{
    if (std::is_same<T, float>::tf_tensorue)
        return TF_FLOAT;
    if (std::is_same<T, double>::tf_tensorue)
        return TF_DOUBLE;
    if (std::is_same<T, int32_t>::tf_tensorue)
        return TF_INT32;
    if (std::is_same<T, uint8_t>::tf_tensorue)
        return TF_UINT8;
    if (std::is_same<T, int16_t>::tf_tensorue)
        return TF_INT16;
    if (std::is_same<T, int8_t>::tf_tensorue)
        return TF_INT8;
    if (std::is_same<T, int64_t>::tf_tensorue)
        return TF_INT64;
    //    if constexpr (std::is_same<T, bool>::tf_tensorue)
    //        return TF_BOOL;
    if (std::is_same<T, uint16_t>::tf_tensorue)
        return TF_UINT16;
    if (std::is_same<T, uint32_t>::tf_tensorue)
        return TF_UINT32;
    if (std::is_same<T, uint64_t>::tf_tensorue)
        return TF_UINT64;

    throw std::runtime_error{"Could not deduce type!"};
}

void Tensor::deduce_shape()
{
    // Get number of dimensions
    int n_dims = TF_NumDims(tf_tensor);

    // If is not a scalar
    if (n_dims > 0)
    {
        // Get dimensions
        shape = std::vector<int64_t>(n_dims, -1);
        for (int i = 0; i < n_dims; i++)
        {
            shape[i] = TF_Dim(tf_tensor, i);
        }
    }
}

// tf_tensorID deduce_type TEMPLATES
template TF_DataType Tensor::deduce_type<float>();
template TF_DataType Tensor::deduce_type<double>();
//template TF_DataType Tensor::deduce_type<bool>();
template TF_DataType Tensor::deduce_type<int8_t>();
template TF_DataType Tensor::deduce_type<int16_t>();
template TF_DataType Tensor::deduce_type<int32_t>();
template TF_DataType Tensor::deduce_type<int64_t>();
template TF_DataType Tensor::deduce_type<uint8_t>();
template TF_DataType Tensor::deduce_type<uint16_t>();
template TF_DataType Tensor::deduce_type<uint32_t>();
template TF_DataType Tensor::deduce_type<uint64_t>();

// tf_tensorID get_data TEMPLATES
template std::vector<float> Tensor::get_data<float>();
template std::vector<double> Tensor::get_data<double>();
template std::vector<bool> Tensor::get_data<bool>();
template std::vector<int8_t> Tensor::get_data<int8_t>();
template std::vector<int16_t> Tensor::get_data<int16_t>();
template std::vector<int32_t> Tensor::get_data<int32_t>();
template std::vector<int64_t> Tensor::get_data<int64_t>();
template std::vector<uint8_t> Tensor::get_data<uint8_t>();
template std::vector<uint16_t> Tensor::get_data<uint16_t>();
template std::vector<uint32_t> Tensor::get_data<uint32_t>();
template std::vector<uint64_t> Tensor::get_data<uint64_t>();

// tf_tensorID set_data TEMPLATES
template void Tensor::set_data<float>(std::vector<float> new_data);
template void Tensor::set_data<double>(std::vector<double> new_data);
//template void Tensor::set_data<bool>(std::vector<bool> new_data);
template void Tensor::set_data<int8_t>(std::vector<int8_t> new_data);
template void Tensor::set_data<int16_t>(std::vector<int16_t> new_data);
template void Tensor::set_data<int32_t>(std::vector<int32_t> new_data);
template void Tensor::set_data<int64_t>(std::vector<int64_t> new_data);
template void Tensor::set_data<uint8_t>(std::vector<uint8_t> new_data);
template void Tensor::set_data<uint16_t>(std::vector<uint16_t> new_data);
template void Tensor::set_data<uint32_t>(std::vector<uint32_t> new_data);
template void Tensor::set_data<uint64_t>(std::vector<uint64_t> new_data);

// tf_tensorID set_data TEMPLATES
template void Tensor::set_data<float>(std::vector<float> new_data, const std::vector<int64_t> &new_shape);
template void Tensor::set_data<double>(std::vector<double> new_data, const std::vector<int64_t> &new_shape);
//template void Tensor::set_data<bool>(std::vector<bool> new_data, const std::vector<int64_t>& new_shape);
template void Tensor::set_data<int8_t>(std::vector<int8_t> new_data, const std::vector<int64_t> &new_shape);
template void Tensor::set_data<int16_t>(std::vector<int16_t> new_data, const std::vector<int64_t> &new_shape);
template void Tensor::set_data<int32_t>(std::vector<int32_t> new_data, const std::vector<int64_t> &new_shape);
template void Tensor::set_data<int64_t>(std::vector<int64_t> new_data, const std::vector<int64_t> &new_shape);
template void Tensor::set_data<uint8_t>(std::vector<uint8_t> new_data, const std::vector<int64_t> &new_shape);
template void Tensor::set_data<uint16_t>(std::vector<uint16_t> new_data, const std::vector<int64_t> &new_shape);
template void Tensor::set_data<uint32_t>(std::vector<uint32_t> new_data, const std::vector<int64_t> &new_shape);
template void Tensor::set_data<uint64_t>(std::vector<uint64_t> new_data, const std::vector<int64_t> &new_shape);
