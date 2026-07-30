"""Entry point that configures OpenCV before its native libraries load."""

import os
import subprocess
import sys


def relaunch_command() -> list[str]:
    """
    Build the command that re-runs this program with OpenCV configured.

    A PyInstaller executable is its own interpreter, so it is relaunched
    directly instead of through a script path.
    """
    if getattr(sys, "frozen", False):
        return [sys.executable, *sys.argv[1:]]

    return [sys.executable, "-B", __file__, *sys.argv[1:]]


def owns_console() -> bool:
    """
    True when this process is the only one attached to the console.

    That is the case when the executable is started from Explorer, where
    Windows closes the window the moment the program exits and any error
    message would disappear with it.
    """
    try:
        import ctypes

        processes = (ctypes.c_uint * 4)()
        count = ctypes.windll.kernel32.GetConsoleProcessList(processes, 4)
        return count <= 1
    except Exception:
        return False


def run() -> int:
    from tx158_capture import main

    try:
        return main()
    except KeyboardInterrupt:
        return 130
    except Exception:
        import traceback

        traceback.print_exc()
        return 1


if __name__ == "__main__":
    if os.environ.get("TX158_CAPTURE_CHILD") != "1":
        child_environment = os.environ.copy()
        child_environment["TX158_CAPTURE_CHILD"] = "1"
        child_environment["OPENCV_FFMPEG_CAPTURE_OPTIONS"] = "rtsp_transport;udp"
        result = subprocess.run(
            relaunch_command(),
            env=child_environment,
            check=False,
        )
        raise SystemExit(result.returncode)

    exit_code = run()

    if owns_console():
        input("\nPress Enter to close this window...")

    raise SystemExit(exit_code)
