// Generic DLL entry used when Smart App Control blocks unsigned process creation:
// python.exe (signed) loads this DLL and calls dll_run, which forwards to the tool's wmain.
#include <cstdio>
int wmain(int argc, wchar_t **argv);
extern "C" __declspec(dllexport) int dll_run(int argc, wchar_t **argv) {
    const int rc = wmain(argc, argv);
    std::fflush(nullptr);
    return rc;
}
