@echo off
call "C:\Program Files\Microsoft Visual Studio\18\Community\VC\Auxiliary\Build\vcvars64.bat" >nul 2>&1
set NVCC=C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v13.3\bin\nvcc.exe
set HOST=C:\Program Files\Microsoft Visual Studio\18\Community\VC\Tools\MSVC\14.51.36231\bin\Hostx64\x64
"%NVCC%" -ccbin "%HOST%" -arch=sm_89 -O3 --use_fast_math -std=c++20 -Iinclude -shared -Xlinker /IMPLIB:build\tpx.lib src\model_file.cpp src\storage.cu src\mxfp4.cu src\qwen35.cu src\qwen_kernels.cu src\ops.cu src\attention.cu src\deltanet.cu src\prefill.cu src\gemm.cu src\decode.cu src\test_pair.cu src\dllshim.cu -o build\test-pair.dll
