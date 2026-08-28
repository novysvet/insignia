@echo off
call "C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v13.3\bin\compute-sanitizer.bat" --tool memcheck --launch-timeout 120 "E:\coding\Insignia\build\generate27.exe" %*
