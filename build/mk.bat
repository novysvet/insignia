@echo off
rem Insignia build driver: mk.bat <target> [args...]   (see: python tools\mk.py --list)
call "C:\Program Files\Microsoft Visual Studio\18\Community\VC\Auxiliary\Build\vcvars64.bat" >nul 2>&1
python tools\mk.py %*
