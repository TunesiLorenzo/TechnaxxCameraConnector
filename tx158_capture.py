from __future__ import annotations

import argparse
import json
import os
import socket
import sys
import threading
import time
from datetime import datetime
from pathlib import Path
from http.client import BadStatusLine
from urllib.error import URLError
from urllib.request import urlopen

# OpenCV reads this option while its FFmpeg backend is initialized, so it
# must be present before importing cv2 (setting it just before VideoCapture
# is too late on Windows).
os.environ.setdefault("OPENCV_FFMPEG_CAPTURE_OPTIONS", "rtsp_transport;udp")

import cv2
import numpy as np


DEFAULT_IP = "192.168.25.1"
CTP_PORT = 3333
TCP_STREAM_PORT = 2229
UDP_STREAM_PORT = 2228
DISCOVERY_PORT = 3889
HTTP_PORT = 8080
WINDOW_NAME = "Technaxx TX-158"

# Display zoom limits, applied to the preview only. Saved images and
# screenshots always keep the camera's native resolution.
MIN_ZOOM = 0.1
MAX_ZOOM = 8.0
ZOOM_STEP = 1.25


def ctp_packet(topic: str, operation: str, parameters: dict | None = None) -> bytes:
    """Build the text packet used by the Jieli CTP command socket."""
    content: dict[str, object] = {"op": operation}
    if parameters is not None:
        content["param"] = parameters

    encoded = json.dumps(content, separators=(",", ":"))
    return f"CTP:{topic}\r\nContent: {encoded}\r\n".encode("utf-8")


class JpegExtractor:
    """Extract JPEG images from an arbitrarily packetized byte stream."""

    SOI = b"\xff\xd8"
    EOI = b"\xff\xd9"

    def __init__(self, maximum_buffer: int = 8 * 1024 * 1024) -> None:
        self.buffer = bytearray()
        self.maximum_buffer = maximum_buffer

    def feed(self, data: bytes) -> list[bytes]:
        self.buffer.extend(data)
        images: list[bytes] = []

        while True:
            start = self.buffer.find(self.SOI)
            if start < 0:
                if len(self.buffer) > self.maximum_buffer:
                    del self.buffer[:-1]
                break

            if start:
                del self.buffer[:start]

            end = self.buffer.find(self.EOI, 2)
            if end < 0:
                if len(self.buffer) > self.maximum_buffer:
                    del self.buffer[:-2]
                break

            end += len(self.EOI)
            images.append(bytes(self.buffer[:end]))
            del self.buffer[:end]

        return images


def is_open(ip_address: str, port: int, timeout: float = 0.5) -> bool:
    try:
        with socket.create_connection((ip_address, port), timeout=timeout):
            return True
    except OSError:
        return False


def diagnose(ip_address: str) -> bool:
    print(f"Camera: {ip_address}")
    expected_ports = [80, HTTP_PORT, 8081, CTP_PORT, TCP_STREAM_PORT]
    any_open = False

    for port in expected_ports:
        open_now = is_open(ip_address, port)
        print(f"  TCP {port:5}: {'open' if open_now else 'closed'}")
        any_open |= open_now

    description_url = (
        f"http://{ip_address}:{HTTP_PORT}/mnt/spiflash/res/dev_desc.txt"
    )
    print(f"\nDevice description: {description_url}")

    try:
        with urlopen(description_url, timeout=2.0) as response:
            body = response.read(256 * 1024)
        print(body.decode("utf-8", errors="replace"))
    except (OSError, URLError, BadStatusLine) as error:
        print(f"  unavailable: {error}")

    if not any_open:
        print(
            "\nThe microscope is not accepting connections. Make sure it is "
            "powered on and Windows is connected to its Cam-XXXXXX Wi-Fi."
        )

    return any_open


class CtpControl:
    def __init__(self, ip_address: str, verbose: bool = False) -> None:
        self.ip_address = ip_address
        self.verbose = verbose
        self.socket: socket.socket | None = None
        self.stop_event = threading.Event()
        self.reader: threading.Thread | None = None

    def connect(self) -> None:
        self.socket = socket.create_connection(
            (self.ip_address, CTP_PORT),
            timeout=3.0,
        )
        self.socket.settimeout(0.5)
        self.reader = threading.Thread(
            target=self._read_responses,
            name="CTP response reader",
            daemon=True,
        )
        self.reader.start()

    def send(
        self,
        topic: str,
        operation: str = "PUT",
        parameters: dict | None = None,
    ) -> None:
        if self.socket is None:
            raise RuntimeError("The CTP command socket is not connected.")

        packet = ctp_packet(topic, operation, parameters)
        if self.verbose:
            print(f"CTP -> {packet.decode('utf-8').rstrip()}")
        self.socket.sendall(packet)

    def _read_responses(self) -> None:
        assert self.socket is not None
        pending = bytearray()

        while not self.stop_event.is_set():
            try:
                block = self.socket.recv(4096)
            except socket.timeout:
                continue
            except OSError:
                break

            if not block:
                break

            pending.extend(block)
            while b"\r\n" in pending:
                line, _, remainder = pending.partition(b"\r\n")
                pending = bytearray(remainder)
                if self.verbose and line:
                    print(f"CTP <- {line.decode('utf-8', errors='replace')}")

    def close(self) -> None:
        self.stop_event.set()
        if self.socket is not None:
            try:
                self.socket.shutdown(socket.SHUT_RDWR)
            except OSError:
                pass
            self.socket.close()
            self.socket = None


def create_stream_socket(
    ip_address: str,
    transport: str,
) -> socket.socket:
    if transport == "tcp":
        stream = socket.create_connection(
            (ip_address, TCP_STREAM_PORT),
            timeout=4.0,
        )
        stream.settimeout(1.0)
        return stream

    stream = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    stream.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    stream.bind(("", UDP_STREAM_PORT))
    stream.settimeout(1.0)
    return stream


def receive_block(stream: socket.socket, transport: str) -> bytes:
    if transport == "tcp":
        return stream.recv(64 * 1024)
    block, _source = stream.recvfrom(64 * 1024)
    return block


def save_frame(frame, output_directory: Path) -> Path:
    output_directory.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")[:-3]
    output_path = output_directory / f"tx158_{timestamp}.jpg"

    if not cv2.imwrite(str(output_path), frame):
        raise RuntimeError(f"Could not save {output_path}")

    return output_path


def get_screen_size() -> tuple[int, int]:
    """Return the primary screen size in pixels."""
    try:
        import ctypes

        user32 = ctypes.windll.user32

        # Without this the metrics come back in scaled coordinates on
        # displays that use a Windows scaling factor above 100%.
        try:
            user32.SetProcessDPIAware()
        except Exception:
            pass

        width = int(user32.GetSystemMetrics(0))
        height = int(user32.GetSystemMetrics(1))
        if width > 0 and height > 0:
            return width, height
    except Exception:
        pass

    return 1920, 1080


def fit_zoom(frame, margin: float = 0.9) -> float:
    """Zoom that makes a frame fill most of the screen without overflowing."""
    height, width = frame.shape[:2]
    if width <= 0 or height <= 0:
        return 1.0

    screen_width, screen_height = get_screen_size()
    zoom = min(
        (screen_width * margin) / width,
        (screen_height * margin) / height,
    )
    return max(MIN_ZOOM, min(MAX_ZOOM, zoom))


class LiveWindow:
    """Resizable OpenCV preview with a clickable screenshot control."""

    def __init__(
        self,
        output_directory: Path,
        zoom: float | None = None,
    ) -> None:
        self.output_directory = output_directory
        # None means "fit to the screen once the frame size is known".
        self.zoom = zoom
        self.applied_zoom: float | None = None
        self.fullscreen = False
        self.screenshot_requested = False
        self.button = (16, 16, 236, 66)

    def open(self) -> None:
        # WINDOW_NORMAL lets the user drag the window edges to resize.
        cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL)
        cv2.setMouseCallback(WINDOW_NAME, self._on_mouse)

    def _on_mouse(
        self,
        event: int,
        x: int,
        y: int,
        _flags: int,
        _parameter: object,
    ) -> None:
        left, top, right, bottom = self.button

        if (
            event == cv2.EVENT_LBUTTONUP
            and left <= x <= right
            and top <= y <= bottom
        ):
            self.screenshot_requested = True

    def _set_zoom(self, zoom: float) -> None:
        self.zoom = max(MIN_ZOOM, min(MAX_ZOOM, zoom))

    def _toggle_fullscreen(self) -> None:
        self.fullscreen = not self.fullscreen
        cv2.setWindowProperty(
            WINDOW_NAME,
            cv2.WND_PROP_FULLSCREEN,
            cv2.WINDOW_FULLSCREEN if self.fullscreen else cv2.WINDOW_NORMAL,
        )

        # Restore the zoomed window size when leaving fullscreen.
        if not self.fullscreen:
            self.applied_zoom = None

    @staticmethod
    def _was_closed() -> bool:
        """
        Detect that the user closed the window with the title bar X.

        cv2.imshow silently recreates a destroyed window, so without this
        check the capture loop reopens it forever.
        """
        try:
            return cv2.getWindowProperty(WINDOW_NAME, cv2.WND_PROP_VISIBLE) < 1
        except cv2.error:
            return True

    def _scale(self, frame):
        """Resize the frame for display and size the window to match."""
        if self.zoom is None:
            self.zoom = fit_zoom(frame)

        height, width = frame.shape[:2]
        new_width = max(1, int(round(width * self.zoom)))
        new_height = max(1, int(round(height * self.zoom)))

        # Only resize on an actual zoom change, otherwise a window the
        # user dragged to a new size would snap back every frame.
        if self.zoom != self.applied_zoom:
            self.applied_zoom = self.zoom
            if not self.fullscreen:
                cv2.resizeWindow(WINDOW_NAME, new_width, new_height)

        if (new_width, new_height) == (width, height):
            return frame.copy()

        interpolation = cv2.INTER_AREA if self.zoom < 1.0 else cv2.INTER_LINEAR
        return cv2.resize(
            frame,
            (new_width, new_height),
            interpolation=interpolation,
        )

    def _draw_overlay(self, display) -> None:
        """Draw the screenshot button and status text at a readable size."""
        height, width = display.shape[:2]

        left = top = margin = min(16, width // 10, height // 10)
        button_width = min(220, width - 2 * margin)
        button_height = min(50, height - 2 * margin)

        # A heavily zoomed-out frame can be smaller than the button, in
        # which case there is nothing sensible to draw or click.
        if button_width < 24 or button_height < 12:
            self.button = (0, 0, -1, -1)
            return

        right, bottom = left + button_width, top + button_height
        self.button = (left, top, right, bottom)

        cv2.rectangle(display, (left, top), (right, bottom), (35, 145, 60), -1)
        cv2.rectangle(display, (left, top), (right, bottom), (255, 255, 255), 2)

        label = "Save screenshot"
        relative = button_height / 50.0
        font_scale = 0.65 * relative
        thickness = max(1, int(round(2 * relative)))
        text_size, _baseline = cv2.getTextSize(
            label,
            cv2.FONT_HERSHEY_SIMPLEX,
            font_scale,
            thickness,
        )
        cv2.putText(
            display,
            label,
            (
                left + (button_width - text_size[0]) // 2,
                top + (button_height + text_size[1]) // 2,
            ),
            cv2.FONT_HERSHEY_SIMPLEX,
            font_scale,
            (255, 255, 255),
            thickness,
            cv2.LINE_AA,
        )

        status_lines = [
            "Battery: unavailable",
            f"Zoom: {round((self.zoom or 1.0) * 100)}%",
        ]

        for index, text in enumerate(status_lines):
            text_size, _baseline = cv2.getTextSize(
                text,
                cv2.FONT_HERSHEY_SIMPLEX,
                0.55,
                1,
            )
            cv2.putText(
                display,
                text,
                (max(16, width - text_size[0] - 16), 42 + index * 26),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.55,
                (255, 255, 255),
                1,
                cv2.LINE_AA,
            )

    def show(self, frame) -> bool:
        """Display one frame, save on click/S, and return False when exiting."""
        display = self._scale(frame)
        self._draw_overlay(display)

        cv2.imshow(WINDOW_NAME, display)
        key = cv2.waitKey(1) & 0xFF

        if self._was_closed():
            return False

        if self.screenshot_requested or key == ord("s"):
            self.screenshot_requested = False
            # Always save the untouched frame, never the scaled overlay.
            print(f"Image saved: {save_frame(frame, self.output_directory)}")

        if key in (ord("+"), ord("=")):
            self._set_zoom((self.zoom or 1.0) * ZOOM_STEP)
        elif key in (ord("-"), ord("_")):
            self._set_zoom((self.zoom or 1.0) / ZOOM_STEP)
        elif key == ord("0"):
            self.zoom = None
        elif key == ord("1"):
            self._set_zoom(1.0)
        elif key == ord("f"):
            self._toggle_fullscreen()

        return key not in (27, ord("q"))


def print_controls() -> None:
    print(
        "Controls:\n"
        "  Q or Esc  Quit          S  Save screenshot\n"
        "  + / -     Zoom          0  Fit to screen\n"
        "  1         100% size     F  Toggle fullscreen\n"
        "The window edges can also be dragged to resize."
    )


def capture_rtsp(
    ip_address: str,
    output_directory: Path,
    timeout: float,
    one_frame: bool,
    scale: float | None = None,
) -> int:
    """
    Capture the Generalplus/GoPlus MJPEG stream.

    This firmware accepts RTSP control on TCP 8080, but it only offers UDP
    media transport. Asking FFmpeg for RTSP-over-TCP produces:
    "Nonmatching transport in server reply".
    """
    os.environ["OPENCV_FFMPEG_CAPTURE_OPTIONS"] = "rtsp_transport;udp"
    url = f"rtsp://{ip_address}:8080/?action=stream"
    print(f"Opening {url}")
    print("Media transport: UDP")
    capture_device = cv2.VideoCapture(url, cv2.CAP_FFMPEG)

    if not capture_device.isOpened():
        print(
            "Could not open the stream. Check that the microscope is powered "
            "on, no phone app is connected, and Windows is on Cam-XXXXXX Wi-Fi."
        )
        return 2

    live_window = None if one_frame else LiveWindow(output_directory, scale)
    if live_window is not None:
        live_window.open()
        print_controls()

    try:
        while True:
            success, frame = capture_device.read()
            if not success or frame is None:
                print("The camera stopped sending frames.")
                return 2

            if one_frame:
                saved = save_frame(frame, output_directory)
                print(
                    f"Received {frame.shape[1]}x{frame.shape[0]} image.\n"
                    f"Image saved: {saved}"
                )
                return 0

            assert live_window is not None
            if not live_window.show(frame):
                return 0
    finally:
        capture_device.release()
        cv2.destroyAllWindows()


def capture(
    ip_address: str,
    transport: str,
    output_directory: Path,
    width: int,
    height: int,
    fps: int,
    timeout: float,
    one_frame: bool,
    verbose: bool,
    dump_path: Path | None,
    scale: float | None = None,
) -> int:
    control = CtpControl(ip_address, verbose=verbose)
    stream: socket.socket | None = None
    dump_file = None

    try:
        # Jieli requires the data channel to exist before OPEN_RT_STREAM.
        print(
            f"Connecting {transport.upper()} video channel "
            f"{ip_address}:{TCP_STREAM_PORT if transport == 'tcp' else UDP_STREAM_PORT}..."
        )
        stream = create_stream_socket(ip_address, transport)

        print(f"Connecting CTP control channel {ip_address}:{CTP_PORT}...")
        control.connect()
        control.send(
            "APP_ACCESS",
            parameters={"type": "0", "ver": "3.1"},
        )
        time.sleep(0.15)

        # format/type 0 asks the firmware for independently decodable JPEG frames.
        control.send(
            "OPEN_RT_STREAM",
            parameters={
                "format": "0",
                "type": "0",
                "w": str(width),
                "h": str(height),
                "fps": str(fps),
                "rate": "8000",
            },
        )

        extractor = JpegExtractor()
        deadline = time.monotonic() + timeout
        received = 0
        decoded = 0
        live_window = None if one_frame else LiveWindow(output_directory, scale)
        if live_window is not None:
            live_window.open()
            print_controls()

        if dump_path is not None:
            dump_path.parent.mkdir(parents=True, exist_ok=True)
            dump_file = dump_path.open("wb")
            print(f"Saving raw protocol bytes to {dump_path}")

        print("Waiting for JPEG frames...")

        while time.monotonic() < deadline:
            try:
                block = receive_block(stream, transport)
            except socket.timeout:
                control.send("CTP_KEEP_ALIVE", parameters=None)
                continue

            if not block:
                break

            received += len(block)
            if dump_file is not None:
                dump_file.write(block)

            for jpeg in extractor.feed(block):
                data = np.frombuffer(jpeg, dtype=np.uint8)
                frame = cv2.imdecode(data, cv2.IMREAD_COLOR)
                if frame is None:
                    continue

                decoded += 1
                deadline = time.monotonic() + timeout

                if decoded == 1:
                    print(
                        f"Received {frame.shape[1]}x{frame.shape[0]} image "
                        f"({len(jpeg):,} bytes)."
                    )

                if one_frame:
                    saved = save_frame(frame, output_directory)
                    print(f"Image saved: {saved}")
                    return 0

                assert live_window is not None
                if not live_window.show(frame):
                    return 0

        if received:
            print(
                f"Received {received:,} protocol bytes but found no JPEG. "
                "Re-run with --verbose --dump captures/stream.bin; the firmware "
                "may have selected H.264 despite the JPEG request."
            )
        else:
            print(
                "No video bytes arrived. Re-run with --verbose. If the CTP "
                "response reports an error, include that output when reporting "
                "the camera firmware variant."
            )
        return 2
    except ConnectionRefusedError as error:
        print(f"Connection refused: {error}")
        print(
            "The microscope may still be booting or this firmware may use UDP. "
            "Run --diagnose, then try --transport udp."
        )
        return 2
    except OSError as error:
        print(f"Network error: {error}")
        return 2
    finally:
        try:
            control.send("CLOSE_RT_STREAM", parameters={})
        except (OSError, RuntimeError):
            pass
        control.close()
        if stream is not None:
            stream.close()
        if dump_file is not None:
            dump_file.close()
        cv2.destroyAllWindows()


def parse_scale(value: str) -> float | None:
    """Parse --scale, where 'fit' means 'size to the current screen'."""
    if value.strip().lower() in ("fit", "auto"):
        return None

    try:
        zoom = float(value)
    except ValueError:
        raise argparse.ArgumentTypeError(
            f"expected 'fit' or a number such as 1.5, not {value!r}"
        ) from None

    if not MIN_ZOOM <= zoom <= MAX_ZOOM:
        raise argparse.ArgumentTypeError(
            f"zoom must be between {MIN_ZOOM} and {MAX_ZOOM}"
        )

    return zoom


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Capture JPEG frames from a Technaxx TX-158/Jieli Wi-Fi microscope."
    )
    parser.add_argument("--ip", default=DEFAULT_IP, help="Camera IP address.")
    parser.add_argument(
        "--transport",
        choices=("rtsp", "tcp", "udp"),
        default="rtsp",
        help=(
            "Camera protocol. 'rtsp' is correct for the TX-158 tested here; "
            "tcp/udp are fallbacks for Jieli CTP firmware."
        ),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("captures"),
        help="Screenshot directory.",
    )
    parser.add_argument(
        "--scale",
        type=parse_scale,
        default=None,
        help=(
            "Preview zoom: 'fit' (default, sizes the window to your screen) "
            "or a factor such as 0.5 or 2. Saved images are unaffected."
        ),
    )
    parser.add_argument("--width", type=int, default=640)
    parser.add_argument("--height", type=int, default=480)
    parser.add_argument("--fps", type=int, default=20)
    parser.add_argument(
        "--timeout",
        type=float,
        default=8.0,
        help="Seconds to wait for a frame.",
    )
    parser.add_argument(
        "--once",
        action="store_true",
        help="Save the first frame and exit instead of opening a window.",
    )
    parser.add_argument(
        "--diagnose",
        action="store_true",
        help="Check Jieli ports and fetch the device description.",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Print CTP commands and camera responses.",
    )
    parser.add_argument(
        "--dump",
        type=Path,
        help="Optionally save raw video-channel bytes for protocol analysis.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_arguments()
    if args.diagnose:
        return 0 if diagnose(args.ip) else 2

    if args.transport == "rtsp":
        return capture_rtsp(
            ip_address=args.ip,
            output_directory=args.output,
            timeout=args.timeout,
            one_frame=args.once,
            scale=args.scale,
        )

    return capture(
        ip_address=args.ip,
        transport=args.transport,
        output_directory=args.output,
        width=args.width,
        height=args.height,
        fps=args.fps,
        timeout=args.timeout,
        one_frame=args.once,
        verbose=args.verbose,
        dump_path=args.dump,
        scale=args.scale,
    )


if __name__ == "__main__":
    sys.exit(main())
