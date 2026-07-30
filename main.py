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

    from tx158_capture import main

    raise SystemExit(main())
