cmake_minimum_required(VERSION 3.25)
project(ai_debate_torch VERSION 1.0 LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Укажите путь к установленному LibTorch
set(TORCH_DIR "D:/09/overflow/libtorch")  # Убедитесь, что здесь лежит torchConfig.cmake

# Поиск LibTorch
find_package(Torch REQUIRED PATHS ${TORCH_DIR} NO_DEFAULT_PATH)

add_executable(ai_debate_torch
        src/main.cpp
        src/dnn_model.cpp
        src/dnn_model.h
        src/cnn_model.cpp
        src/cnn_model.h
        src/lstm_model.cpp
        src/lstm_model.h
        src/debater.cpp
        src/debater.h
        src/voice.cpp
        src/voice.h
)

# Подключаем библиотеки
target_link_libraries(ai_debate_torch PRIVATE ${TORCH_LIBRARIES})

# Подключаем заголовочные файлы
target_include_directories(ai_debate_torch PRIVATE ${TORCH_INCLUDE_DIRS})

# Чтобы работала динамическая линковка (DLL), укажите флаг:
if(WIN32)
    target_compile_definitions(ai_debate_torch PRIVATE _USE_MATH_DEFINES _CRT_SECURE_NO_WARNINGS)
endif()
