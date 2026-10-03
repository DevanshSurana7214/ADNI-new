#!/usr/bin/env python3
"""Step 0 environment report.

Prints: OS, Python version, CPU count, RAM, GPU name/VRAM (if any), torch version,
torch.cuda.is_available(), and the SHA256 of every file in data/raw/.

Run from the project root:  python scripts/env_report.py
"""
from __future__ import annotations

import ctypes
import hashlib
import os
import platform
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
RAW_DIR = PROJECT_ROOT / "data" / "raw"


def total_ram_bytes() -> int | None:
    """Best-effort total physical RAM in bytes (works on Windows and POSIX)."""
    if sys.platform.startswith("win"):
        class MEMORYSTATUSEX(ctypes.Structure):
            _fields_ = [
                ("dwLength", ctypes.c_ulong),
                ("dwMemoryLoad", ctypes.c_ulong),
                ("ullTotalPhys", ctypes.c_ulonglong),
                ("ullAvailPhys", ctypes.c_ulonglong),
                ("ullTotalPageFile", ctypes.c_ulonglong),
                ("ullAvailPageFile", ctypes.c_ulonglong),
                ("ullTotalVirtual", ctypes.c_ulonglong),
                ("ullAvailVirtual", ctypes.c_ulonglong),
                ("ullAvailExtendedVirtual", ctypes.c_ulonglong),
            ]

        stat = MEMORYSTATUSEX()
        stat.dwLength = ctypes.sizeof(MEMORYSTATUSEX)
        if ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(stat)):
            return int(stat.ullTotalPhys)
        return None
    try:
        return os.sysconf("SC_PAGE_SIZE") * os.sysconf("SC_PHYS_PAGES")
    except (ValueError, OSError, AttributeError):
        return None


def sha256_file(path: Path, chunk: int = 1 << 20) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for block in iter(lambda: fh.read(chunk), b""):
            h.update(block)
    return h.hexdigest()


def gpu_report() -> list[str]:
    lines: list[str] = []
    try:
        import torch
    except ImportError:
        return ["torch: NOT INSTALLED"]
    lines.append(f"torch version: {torch.__version__}")
    lines.append(f"torch.cuda.is_available(): {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        n = torch.cuda.device_count()
        lines.append(f"CUDA device count: {n}")
        for i in range(n):
            props = torch.cuda.get_device_properties(i)
            vram_gb = props.total_memory / 1024**3
            lines.append(f"  GPU {i}: {props.name}  VRAM={vram_gb:.1f} GiB  (capability {props.major}.{props.minor})")
    else:
        lines.append("GPU: none available to torch")
    return lines


def main() -> None:
    print("=" * 72)
    print("ENVIRONMENT REPORT")
    print("=" * 72)
    print(f"OS: {platform.system()} {platform.release()} ({platform.version()})")
    print(f"machine: {platform.machine()}")
    print(f"Python version: {sys.version.split()[0]} ({platform.python_implementation()})")
    print(f"Python executable: {sys.executable}")
    print(f"CPU count (logical): {os.cpu_count()}")
    ram = total_ram_bytes()
    print(f"RAM: {ram / 1024**3:.1f} GiB" if ram else "RAM: unknown")
    for line in gpu_report():
        print(line)

    print("-" * 72)
    print(f"SHA256 of files in {RAW_DIR.relative_to(PROJECT_ROOT)}:")
    if not RAW_DIR.is_dir():
        print("  (data/raw/ does not exist)")
        return
    files = sorted(p for p in RAW_DIR.iterdir() if p.is_file())
    if not files:
        print("  (no files)")
    for p in files:
        writable = os.access(p, os.W_OK)
        print(f"  {p.name:45s} {sha256_file(p)}  readonly={not writable}")
    print("=" * 72)


if __name__ == "__main__":
    main()
