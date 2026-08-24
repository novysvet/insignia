@echo off
call "C:\Program Files\Microsoft Visual Studio\18\Community\VC\Auxiliary\Build\vcvars64.bat"
if errorlevel 1 exit /b 1
cl /nologo /std:c++20 /O2 /EHsc /Iinclude src\model_file.cpp src\test_model.cpp /Fe:build\test-model.exe
if errorlevel 1 exit /b 1
build\test-model.exe build\qwen35.insignia-index
