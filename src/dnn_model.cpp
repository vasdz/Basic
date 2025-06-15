#include "dnn_model.h"
#include <fstream>

DNNModel::DNNModel(const std::string& model_path) {
    if (!LoadModel(model_path)) {
        throw std::runtime_error("Failed to load DNN model");
    }
}

bool DNNModel::LoadModel(const std::string& model_path) {
    try {
        model = torch::jit::load(model_path);
        return true;
    } catch (const c10::Error& e) {
        std::cerr << "Error loading DNN model\n";
        return false;
    }
}

bool DNNModel::Classify(const std::vector<float>& input_data, std::string& result) {
    try {
        torch::Tensor input_tensor = torch::from_blob(
            const_cast<float*>(input_data.data()), {1, 224, 224, 3});
        input_tensor = input_tensor.permute({0, 3, 1, 2}); // NHWC -> NCHW

        at::AutoGradMode guard(false); // disable autograd
        torch::Tensor output_tensor = model.forward({input_tensor}).toTensor();

        auto [max_prob, max_index] = output_tensor.max(1);
        int index = max_index.item<int>();

        static std::vector<std::string> class_names = [](){
            std::vector<std::string> classes;
            std::ifstream file("data/imagenet_classes.txt");
            std::string line;
            while (std::getline(file, line)) classes.push_back(line);
            return classes;
        }();

        result = class_names[index];
        return true;
    } catch (const std::exception& ex) {
        result = "DNN: Classification error";
        std::cerr << "DNN error: " << ex.what() << "\n";
        return false;
    }
}
