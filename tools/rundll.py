#!/usr/bin/env python3
"""Run an Insignia tool DLL: rundll.py <tool.dll> [args...]"""
import ctypes, sys, time

def load(path, tries=12):
    for i in range(tries):
        try:
            return ctypes.CDLL(path)
        except OSError as e:
            if "4551" not in str(e) or i == tries - 1:
                raise
            time.sleep(10)  # Smart App Control briefly blocks freshly written DLLs

def main():
    if len(sys.argv) < 2:
        raise SystemExit("usage: rundll.py <tool.dll> [args...]")
    dll = load(sys.argv[1])
    args = [sys.argv[1]] + sys.argv[2:]  # argv[0] dummy keeps tool argc conventions intact
    n = len(args)
    arr = (ctypes.c_wchar_p * (n + 1))(*args)
    try:
        rc = dll.dll_run(n, arr)
    except AttributeError:
        arrc = (ctypes.c_char_p * (n + 1))(*[a.encode("utf-8") for a in args])
        rc = dll.dll_run_c(n, arrc)
    sys.exit(rc)

if __name__ == "__main__":
    main()
