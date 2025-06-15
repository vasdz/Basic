#pragma once
#include <string>
#include <vector>
#include <torch/script.h>

class DNNModel {
public:
    DNNModel(const std::string& model_path);
    bool Classify(const std::vector<float>& input_data, std::string& result);
private:
    torch::jit::Module model;
    bool LoadModel(const std::string& model_path);
};
