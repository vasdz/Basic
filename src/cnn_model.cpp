#include "cnn_model.h"
#include <fstream>

CNNModel::CNNModel(const std::string& model_path) {
    if (!LoadModel(model_path)) {
        throw std::runtime_error("Failed to load CNN model");
    }
}

bool CNNModel::LoadModel(const std::string& model_path) {
    try {
        model = torch::jit::load(model_path);
        return true;
    } catch (const c10::Error& e) {
        std::cerr << "Error loading CNN model\n";
        return false;
    }
}

bool CNNModel::Classify(const std::vector<float>& input_data, std::string& result) {
    try {
        torch::Tensor input_tensor = torch::from_blob(
            const_cast<float*>(input_data.data()), {1, 224, 224, 3});
        input_tensor = input_tensor.permute({0, 3, 1, 2});

        at::AutoGradMode guard(false);
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
        result = "CNN: Classification error";
        std::cerr << "CNN error: " << ex.what() << "\n";
        return false;
    }
}
