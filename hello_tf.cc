#include <tensorflow/c/c_api.h>

#include <iostream>

#include "tf_utils.h"

int main(int argc, char **argv) {
  std::cout << "Tensorflow Version :" << TF_Version() << std::endl;
  return 0;
}