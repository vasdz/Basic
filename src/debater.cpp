#include "debater.h"
#include "voice.h"
#include <iostream>
#include <string>

void Debater::ResolveDispute(
    const std::string& dnn_result,
    const std::string& cnn_result,
    const std::string& lstm_result) {
    std::cout << "=== СПОР МОДЕЛЕЙ ===" << std::endl;
    std::cout << "DNN говорит: \"" << dnn_result << "\" потому что это похоже на объект с прямыми линиями!" << std::endl;
    Speak("DNN говорит: " + dnn_result + " потому что это похоже на объект с прямыми линиями");
    
    std::cout << "CNN говорит: \"" << cnn_result << "\" потому что я вижу характерные признаки!" << std::endl;
    Speak("CNN говорит: " + cnn_result + " потому что я вижу характерные признаки");
    
    std::cout << "LSTM говорит: \"" << lstm_result << "\" потому что последовательность признаков важна!" << std::endl;
    Speak("LSTM говорит: " + lstm_result + " потому что последовательность признаков важна");
    
    std::cout << "Кто прав? Введите: 1 - DNN, 2 - CNN, 3 - LSTM" << std::endl;
    int choice;
    std::cin >> choice;
    
    switch (choice) {
        case 1: std::cout << "Вы выбрали DNN" << std::endl; break;
        case 2: std::cout << "Вы выбрали CNN" << std::endl; break;
        case 3: std::cout << "Вы выбрали LSTM" << std::endl; break;
        default: std::cout << "Неверный выбор" << std::endl; break;
    }
}
