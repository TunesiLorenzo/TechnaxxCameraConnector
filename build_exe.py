"""Build a standalone Windows executable with PyInstaller."""

from __future__ import annotations

import shutil
import subprocess
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parent
APPLICATION_NAME = "TechnaxxCameraConnector"


def build(one_file: bool = True, clean: bool = False) -> int:
    if clean:
        for directory in ("build", "dist"):
            shutil.rmtree(PROJECT_ROOT / directory, ignore_errors=True)

    command = [
        sys.executable,
        "-m",
        "PyInstaller",
        "--noconfirm",
        "--onefile" if one_file else "--onedir",
        "--console",
        "--name",
        APPLICATION_NAME,
        "--runtime-hook",
        str(PROJECT_ROOT / "build_hooks" / "opencv_env.py"),
        # main.py imports this lazily, so it is not found by static analysis.
        "--hidden-import",
        "tx158_capture",
        # Nothing in this project uses these, and they pull in large trees.
        "--exclude-module",
        "matplotlib",
        "--exclude-module",
        "tkinter",
        str(PROJECT_ROOT / "main.py"),
    ]

    print(" ".join(command), "\n")
    result = subprocess.run(command, cwd=PROJECT_ROOT, check=False)

    if result.returncode == 0:
        suffix = ".exe" if sys.platform == "win32" else ""
        executable = PROJECT_ROOT / "dist" / f"{APPLICATION_NAME}{suffix}"

        if not one_file:
            executable = (
                PROJECT_ROOT
                / "dist"
                / APPLICATION_NAME
                / f"{APPLICATION_NAME}{suffix}"
            )

        size = executable.stat().st_size / (1024 * 1024) if executable.exists() else 0
        print(f"\nBuilt {executable} ({size:.1f} MB)")

    return result.returncode


if __name__ == "__main__":
    raise SystemExit(
        build(
            one_file="--onedir" not in sys.argv,
            clean="--clean" in sys.argv,
        )
    )
