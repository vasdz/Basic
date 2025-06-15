#include "dnn_model.h"
#include "cnn_model.h"
#include "lstm_model.h"
#include "debater.h"
#include "voice.h"

#include <iostream>
#include <thread>
#include <chrono>

// Функция-заглушка для получения данных изображения
std::vector<float> GetFakeInputData() {
    return std::vector<float>(224 * 224 * 3, 0.5f); // Серое изображение
}

int main() {
    try {
        DNNModel dnn("models/dnn_titanic.pt");
        CNNModel cnn("models/cnn_titanic.pt");
        LSTMModel lstm("models/lstm_titanic.pt");

        std::cout << "Система готова. Нажмите ESC для выхода." << std::endl;

        while (true) {
            auto input_data = GetFakeInputData();

            std::string dnn_res, cnn_res, lstm_res;
            if (dnn.Classify(input_data, dnn_res) &&
                cnn.Classify(input_data, cnn_res) &&
                lstm.Classify(input_data, lstm_res)) {
                std::cout << "DNN: " << dnn_res << std::endl;
                std::cout << "CNN: " << cnn_res << std::endl;
                std::cout << "LSTM: " << lstm_res << std::endl;
                Debater::ResolveDispute(dnn_res, cnn_res, lstm_res);
                }

            std::this_thread::sleep_for(std::chrono::seconds(5));
        }
    } catch (const std::exception& ex) {
        std::cerr << "Ошибка: " << ex.what() << std::endl;
        return -1;
    }

    return 0;
}
