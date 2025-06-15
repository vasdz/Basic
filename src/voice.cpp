#include <windows.h>
#include <sapi.h>
#include <sphelper.h>
#include <string>
#pragma comment(lib, "sapi.lib")

void Speak(const std::string& text) {
    HRESULT hr = CoInitialize(NULL);
    ISpVoice* pVoice = nullptr;
    hr = CoCreateInstance(CLSID_SpVoice, NULL, CLSCTX_ALL, IID_ISpVoice, (void**)&pVoice);
    if (SUCCEEDED(hr)) {
        std::wstring wide_text(text.begin(), text.end());
        pVoice->Speak(wide_text.c_str(), SPF_DEFAULT, NULL);
        pVoice->Release();
    }
    CoUninitialize();
}
