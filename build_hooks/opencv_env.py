"""
PyInstaller runtime hook: configure OpenCV before its libraries load.

This runs before any application code, which is early enough for the
FFmpeg backend to pick the option up. The packaged executable therefore
does not need the subprocess relaunch that main.py performs when it runs
as a plain script, and it avoids unpacking the onefile bundle twice.
"""

import os

os.environ["OPENCV_FFMPEG_CAPTURE_OPTIONS"] = "rtsp_transport;udp"
os.environ["TX158_CAPTURE_CHILD"] = "1"
