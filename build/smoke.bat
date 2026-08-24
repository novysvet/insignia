@echo off
call "C:\Program Files\Microsoft Visual Studio\18\Community\VC\Auxiliary\Build\vcvars64.bat"
if errorlevel 1 exit /b 1
set NVCC=C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v13.3\bin\nvcc.exe
set HOST=C:\Program Files\Microsoft Visual Studio\18\Community\VC\Tools\MSVC\14.51.36231\bin\Hostx64\x64
"%NVCC%" -ccbin "%HOST%" -arch=sm_89 -O3 -std=c++20 src\smoke.cu -o build\insignia-smoke.exe
if errorlevel 1 exit /b 1
build\insignia-smoke.exe
if errorlevel 1 exit /b 1
"%NVCC%" -ccbin "%HOST%" -arch=sm_89 -O3 -std=c++20 -Iinclude src\mxfp4.cu src\test_mxfp4.cu -o build\test-mxfp4.exe
if errorlevel 1 exit /b 1
build\test-mxfp4.exe
"%NVCC%" -ccbin "%HOST%" -arch=sm_89 -O3 -std=c++20 -Iinclude src\mxfp4.cu src\bench_mxfp4_mlx.cu -o build\bench-mxfp4-mlx.exe
if errorlevel 1 exit /b 1
build\bench-mxfp4-mlx.exe
