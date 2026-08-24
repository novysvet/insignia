// DLL shim for tools whose entry is main(argc, argv).
int main(int argc, char **argv);
extern "C" __declspec(dllexport) int dll_run_c(int argc, char **argv) { return main(argc, argv); }
