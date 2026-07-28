from __future__ import annotations

import argparse
import os
import re
import socket
import subprocess
import sys
import time
from pathlib import Path

import cv2


COMMON_PORTS = [
    80,
    81,
    554,
    1935,
    5000,
    8000,
    8001,
    8080,
    8081,
    8554,
    8888,
]


def get_default_gateway_windows() -> str | None:
    """
    Find the IPv4 default gateway used by Windows.

    This normally corresponds to the microscope IP when the computer
    is connected directly to the Cam-XXXXXX Wi-Fi network.
    """
    try:
        result = subprocess.run(
            ["ipconfig"],
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="ignore",
            check=False,
        )
    except OSError:
        return None

    # Matches both English and localized ipconfig output because only
    # the IPv4 value after a colon is relevant.
    gateway_pattern = re.compile(
        r"Default Gateway[ .]*:\s*(\d{1,3}(?:\.\d{1,3}){3})",
        re.IGNORECASE,
    )

    matches = gateway_pattern.findall(result.stdout)

    for address in matches:
        if address != "0.0.0.0":
            return address

    # More generic fallback for localized Windows installations.
    lines = result.stdout.splitlines()

    for index, line in enumerate(lines):
        if "gateway" not in line.lower():
            continue

        possible_lines = [line]

        if index + 1 < len(lines):
            possible_lines.append(lines[index + 1])

        for candidate_line in possible_lines:
            match = re.search(
                r"(\d{1,3}(?:\.\d{1,3}){3})",
                candidate_line,
            )

            if match:
                return match.group(1)

    return None


def is_port_open(ip_address: str, port: int, timeout: float = 0.4) -> bool:
    """Return True when a TCP port accepts a connection."""
    try:
        with socket.create_connection(
            (ip_address, port),
            timeout=timeout,
        ):
            return True
    except (TimeoutError, ConnectionRefusedError, OSError):
        return False


def scan_ports(ip_address: str) -> list[int]:
    """Scan common camera streaming ports."""
    open_ports: list[int] = []

    print(f"\nScanning {ip_address}...")

    for port in COMMON_PORTS:
        if is_port_open(ip_address, port):
            print(f"  Open TCP port: {port}")
            open_ports.append(port)

    if not open_ports:
        print("  No common TCP ports were found.")

    return open_ports


def build_candidate_urls(ip_address: str) -> list[str]:
    """Generate common video stream addresses."""
    return [
        # Common MJPEG/HTTP endpoints
        f"http://{ip_address}/",
        f"http://{ip_address}/video",
        f"http://{ip_address}/stream",
        f"http://{ip_address}/mjpeg",
        f"http://{ip_address}/videostream.cgi",
        f"http://{ip_address}:81/stream",
        f"http://{ip_address}:81/video",
        f"http://{ip_address}:8080/video",
        f"http://{ip_address}:8080/stream",
        f"http://{ip_address}:8080/?action=stream",
        f"http://{ip_address}:8081/video",
        f"http://{ip_address}:8081/stream",
        f"http://{ip_address}:8888/video",
        f"http://{ip_address}:8888/stream",

        # Common RTSP endpoints
        f"rtsp://{ip_address}:554/live",
        f"rtsp://{ip_address}:554/live.sdp",
        f"rtsp://{ip_address}:554/stream",
        f"rtsp://{ip_address}:554/stream1",
        f"rtsp://{ip_address}:554/ch0",
        f"rtsp://{ip_address}:554/ch0_0.h264",
        f"rtsp://{ip_address}:8554/live",
        f"rtsp://{ip_address}:8554/stream",
        f"rtsp://{ip_address}:8554/stream1",
    ]


def test_stream(url: str, timeout_seconds: float = 4.0):
    """
    Try opening a stream and reading at least one valid frame.

    Returns the active VideoCapture and first frame when successful.
    """
    print(f"Testing: {url}")

    capture = cv2.VideoCapture(url, cv2.CAP_FFMPEG)

    try:
        capture.set(cv2.CAP_PROP_OPEN_TIMEOUT_MSEC, timeout_seconds * 1000)
        capture.set(cv2.CAP_PROP_READ_TIMEOUT_MSEC, timeout_seconds * 1000)
        capture.set(cv2.CAP_PROP_BUFFERSIZE, 1)
    except Exception:
        pass

    deadline = time.monotonic() + timeout_seconds

    while time.monotonic() < deadline:
        success, frame = capture.read()

        if success and frame is not None and frame.size > 0:
            print(f"\nStream found:\n{url}\n")
            return capture, frame

        time.sleep(0.05)

    capture.release()
    return None, None


def find_stream(ip_address: str):
    """Test candidate URLs and return the first working stream."""
    candidates = build_candidate_urls(ip_address)

    for url in candidates:
        capture, frame = test_stream(url)

        if capture is not None:
            return url, capture, frame

    return None, None, None


def create_video_writer(
    frame,
    output_directory: Path,
    frames_per_second: float = 20.0,
):
    """Create an MP4 writer matching the current video frame."""
    output_directory.mkdir(parents=True, exist_ok=True)

    timestamp = time.strftime("%Y%m%d_%H%M%S")
    output_path = output_directory / f"tx158_{timestamp}.mp4"

    height, width = frame.shape[:2]

    writer = cv2.VideoWriter(
        str(output_path),
        cv2.VideoWriter_fourcc(*"mp4v"),
        frames_per_second,
        (width, height),
    )

    if not writer.isOpened():
        raise RuntimeError("OpenCV could not create the video file.")

    return writer, output_path


def show_stream(
    capture,
    first_frame,
    output_directory: Path,
) -> None:
    """
    Display the camera stream.

    Controls:
        Q or Esc: quit
        S: save screenshot
        R: start/stop recording
    """
    frame = first_frame
    video_writer = None
    recording_path = None

    output_directory.mkdir(parents=True, exist_ok=True)

    print("Controls:")
    print("  Q or Esc  Quit")
    print("  S         Save screenshot")
    print("  R         Start or stop recording")

    while True:
        if frame is None:
            success, frame = capture.read()

            if not success or frame is None:
                print("The video stream ended or stopped responding.")
                break

        displayed_frame = frame.copy()

        if video_writer is not None:
            video_writer.write(frame)

            cv2.putText(
                displayed_frame,
                "REC",
                (20, 40),
                cv2.FONT_HERSHEY_SIMPLEX,
                1.0,
                (0, 0, 255),
                2,
                cv2.LINE_AA,
            )

        cv2.imshow("Technaxx TX-158", displayed_frame)

        key = cv2.waitKey(1) & 0xFF

        if key in (27, ord("q")):
            break

        if key == ord("s"):
            timestamp = time.strftime("%Y%m%d_%H%M%S")
            screenshot_path = output_directory / f"tx158_{timestamp}.png"

            if cv2.imwrite(str(screenshot_path), frame):
                print(f"Screenshot saved: {screenshot_path}")
            else:
                print("Could not save the screenshot.")

        if key == ord("r"):
            if video_writer is None:
                try:
                    video_writer, recording_path = create_video_writer(
                        frame,
                        output_directory,
                    )
                    print(f"Recording started: {recording_path}")
                except RuntimeError as error:
                    print(error)
            else:
                video_writer.release()
                video_writer = None
                print(f"Recording saved: {recording_path}")
                recording_path = None

        success, next_frame = capture.read()

        if not success:
            print("Could not read another frame.")
            break

        frame = next_frame

    if video_writer is not None:
        video_writer.release()

        if recording_path:
            print(f"Recording saved: {recording_path}")

    capture.release()
    cv2.destroyAllWindows()


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Connect to a Technaxx TX-158 Wi-Fi microscope."
    )

    parser.add_argument(
        "--ip",
        help=(
            "Microscope IP address. When omitted, the Windows default "
            "gateway is used."
        ),
    )

    parser.add_argument(
        "--url",
        help="Test a specific HTTP or RTSP stream URL.",
    )

    parser.add_argument(
        "--output",
        type=Path,
        default=Path("captures"),
        help="Directory used for screenshots and recordings.",
    )

    parser.add_argument(
        "--scan-only",
        action="store_true",
        help="Only scan common ports without testing video URLs.",
    )

    return parser.parse_args()


def main() -> int:
    args = parse_arguments()

    # OpenCV FFmpeg settings: prefer TCP for RTSP and limit buffering.
    os.environ.setdefault(
        "OPENCV_FFMPEG_CAPTURE_OPTIONS",
        "rtsp_transport;tcp",
    )

    ip_address = args.ip or get_default_gateway_windows()
    ip_address = "192.168.25.1"
    if not ip_address:
        print(
            "Could not determine the microscope IP address.\n"
            "Connect Windows to Cam-XXXXXX and run:\n\n"
            "    ipconfig\n\n"
            "Then start this script with:\n\n"
            "    py main.py --ip 192.168.x.x"
        )
        return 1

    print(f"Using camera IP: {ip_address}")

    scan_ports(ip_address)

    if args.scan_only:
        return 0

    if args.url:
        url = args.url
        capture, first_frame = test_stream(url)

        if capture is None:
            print(f"The specified stream did not work: {url}")
            return 2
    else:
        url, capture, first_frame = find_stream(ip_address)

        if capture is None:
            print(
                "\nNo standard HTTP or RTSP stream was found.\n"
                "The TX-158 probably uses a proprietary UDP protocol."
            )
            return 2

    show_stream(
        capture=capture,
        first_frame=first_frame,
        output_directory=args.output,
    )

    return 0


if __name__ == "__main__":
    sys.exit(main())