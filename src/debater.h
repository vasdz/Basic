#pragma once
#include <string>

class Debater {
public:
    static void ResolveDispute(
        const std::string& dnn_result,
        const std::string& cnn_result,
        const std::string& lstm_result);
};
