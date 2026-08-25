@echo off
call "C:\Program Files\Microsoft Visual Studio\18\Community\VC\Auxiliary\Build\vcvars64.bat" >nul 2>&1
set NVCC=C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v13.3\bin\nvcc.exe
set HOST=C:\Program Files\Microsoft Visual Studio\18\Community\VC\Tools\MSVC\14.51.36231\bin\Hostx64\x64
"%NVCC%" -ccbin "%HOST%" -arch=sm_89 -O3 -shared "%TMP%\tiny.cu" -o "%TMP%\tiny.dll"
