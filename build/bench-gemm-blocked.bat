@echo off
call "C:\Program Files\Microsoft Visual Studio\18\Community\VC\Auxiliary\Build\vcvars64.bat" >nul 2>&1
set NVCC=C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v13.3\bin\nvcc.exe
set HOST=C:\Program Files\Microsoft Visual Studio\18\Community\VC\Tools\MSVC\14.51.36231\bin\Hostx64\x64
"%NVCC%" -ccbin "%HOST%" -arch=sm_89 -O3 --use_fast_math -std=c++20 -Iinclude -shared -Xlinker /IMPLIB:build\bgx.lib src\mxfp4.cu src\gemm.cu src\bench_gemm.cu src\dllshim_c.cu -o build\bench-mxfp4.dll
if errorlevel 1 exit /b 1
