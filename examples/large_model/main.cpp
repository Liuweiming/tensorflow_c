#include <algorithm>
#include <iostream>
#include <iterator>
#include <vector>
#include "model.h"
#include "tensor.h"

using namespace tf_cpp;

int main() {
  Model model("graph.pb");
  std::cout << "operations: -----------" << std::endl;
  for (auto &op : model.get_operations()) {
    std::cout << op << std::endl;
  }
  std::cout << "-------------------" << std::endl;
  Tensor input(model.get_graph(), "input_4", {2, 5, 12}, TF_FLOAT);
  Tensor out(model.get_graph(), "output_node0", {2, 4}, TF_FLOAT);

  const std::vector<std::int64_t> input_dims = {2, 5, 12};  // batch 2

  const std::vector<float> input_vals_1 = {
      -0.4809832f, -0.3770838f, 0.1743573f,  0.7720509f,  -0.4064746f,
      0.0116595f,  0.0051413f,  0.9135732f,  0.7197526f,  -0.0400658f,
      0.1180671f,  -0.6829428f, -0.4810135f, -0.3772099f, 0.1745346f,
      0.7719303f,  -0.4066443f, 0.0114614f,  0.0051195f,  0.9135003f,
      0.7196983f,  -0.0400035f, 0.1178188f,  -0.6830465f, -0.4809143f,
      -0.3773398f, 0.1746384f,  0.7719052f,  -0.4067171f, 0.0111654f,
      0.0054433f,  0.9134697f,  0.7192584f,  -0.0399981f, 0.1177435f,
      -0.6835230f, -0.4808300f, -0.3774327f, 0.1748246f,  0.7718700f,
      -0.4070232f, 0.0109549f,  0.0059128f,  0.9133330f,  0.7188759f,
      -0.0398740f, 0.1181437f,  -0.6838635f, -0.4807833f, -0.3775733f,
      0.1748378f,  0.7718275f,  -0.4073670f, 0.0107582f,  0.0062978f,
      0.9131795f,  0.7187147f,  -0.0394935f, 0.1184392f,  -0.6840039f,
  };
  const std::vector<float> input_vals_2 = {
      -0.5807833f, -0.3775733f, 0.1748378f,  0.7718275f,  -0.4073670f,
      0.0107582f,  0.0062978f,  0.9131795f,  0.7187147f,  -0.0394935f,
      0.1184392f,  -0.6840039f, -0.5809832f, -0.3770838f, 0.1743573f,
      0.7720509f,  -0.4064746f, 0.0116595f,  0.0051413f,  0.9135732f,
      0.7197526f,  -0.0400658f, 0.1180671f,  -0.6829428f, -0.5810135f,
      -0.3772099f, 0.1745346f,  0.7719303f,  -0.4066443f, 0.0114614f,
      0.0051195f,  0.9135003f,  0.7196983f,  -0.0400035f, 0.1178188f,
      -0.6830465f, -0.5809143f, -0.3773398f, 0.1746384f,  0.7719052f,
      -0.4067171f, 0.0111654f,  0.0054433f,  0.9134697f,  0.7192584f,
      -0.0399981f, 0.1177435f,  -0.6835230f, -0.5808300f, -0.3774327f,
      0.1748246f,  0.7718700f,  -0.4070232f, 0.0109549f,  0.0059128f,
      0.9133330f,  0.7188759f,  -0.0398740f, 0.1181437f,  -0.6838635f,
  };

  std::vector<float> input_vals_batch;
  input_vals_batch.reserve(input_vals_1.size() + input_vals_2.size());
  input_vals_batch.insert(input_vals_batch.end(), input_vals_1.begin(),
                          input_vals_1.end());
  input_vals_batch.insert(input_vals_batch.end(), input_vals_2.begin(),
                          input_vals_2.end());
  memcpy((&input.at<float>(0)), input_vals_1.data(),
         sizeof(float) * input_vals_1.size());
  memcpy((&input.at<float>(1)), input_vals_2.data(),
         sizeof(float) * input_vals_2.size());
  // memcpy((&input.at<float>({0})), input_vals_batch.data(),
  //        sizeof(float) * input_vals_batch.size());
  model.run({&input}, {&out});

  std::vector<float> result(&out.at<float>(0), &out.at<float>(0) + 8);
  std::cout << "Output vals_1: " << result[0] << ", " << result[1] << ", "
            << result[2] << ", " << result[3] << std::endl;
  std::cout << "Output vals_2: " << result[4] << ", " << result[5] << ", "
            << result[6] << ", " << result[7] << std::endl;
}
