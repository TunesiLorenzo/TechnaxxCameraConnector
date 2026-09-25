from __future__ import annotations

import argparse
import json
import math
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

from rtp_jpeg import RtpJpegStream


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

# Controls are drawn at a fixed size so they keep their shape and stay
# readable whatever the zoom level or window proportions are.
BUTTON_HEIGHT = 30
BUTTON_FONT_SCALE = 0.5
BUTTON_TEXT_THICKNESS = 1
PANEL_CLOSE_SIZE = 28
PANEL_CLOSE_MARGIN = 6

# Angle tool: endpoint marker size and how near a click has to land to
# pick one up, both in displayed pixels.
HANDLE_RADIUS = 6
HANDLE_GRAB_RADIUS = 15

# Highest USB camera index probed by --list-cameras and --usb auto.
MAX_USB_INDEX = 7
USB_OPEN_TIMEOUT = 3.0
USB_OPEN_RETRY_DELAY = 0.25

# Camera 0 is known not to work on this installation. Keep it out of manual
# selection, startup auto-discovery, camera listing, and in-window rescans.
IGNORED_USB_INDEXES = {0}


def usb_backend() -> int:
    """DirectShow opens far faster than the default backend on Windows."""
    return cv2.CAP_DSHOW if sys.platform == "win32" else cv2.CAP_ANY


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


def find_usb_cameras(
    maximum_index: int = MAX_USB_INDEX,
    skip: set[int] | None = None,
) -> list[tuple[int, int, int]]:
    """
    Return (index, width, height) for every USB camera that opens.

    Indexes in `skip` are left untouched, because probing a camera that
    is already streaming can steal frames from it.
    """
    found: list[tuple[int, int, int]] = []

    # Probing empty indexes is normal here, so hide OpenCV's backend warnings.
    previous_log_level = cv2.utils.logging.getLogLevel()
    cv2.utils.logging.setLogLevel(cv2.utils.logging.LOG_LEVEL_ERROR)

    try:
        for index in range(maximum_index + 1):
            if index in IGNORED_USB_INDEXES or (skip and index in skip):
                continue

            device = cv2.VideoCapture(index, usb_backend())
            try:
                if not device.isOpened():
                    continue

                success, frame = device.read()
                if not success or frame is None:
                    continue

                found.append((index, frame.shape[1], frame.shape[0]))
            finally:
                device.release()
    finally:
        cv2.utils.logging.setLogLevel(previous_log_level)

    return found


def list_cameras() -> int:
    ignored = ", ".join(str(index) for index in sorted(IGNORED_USB_INDEXES))
    print(
        f"Probing USB camera indexes 0-{MAX_USB_INDEX} "
        f"(always ignoring {ignored})..."
    )
    cameras = find_usb_cameras()

    if not cameras:
        print("No USB cameras responded.")
        return 2

    for index, width, height in cameras:
        print(f"  --usb {index}   {width}x{height}")

    return 0


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


def save_frame(frame, output_directory: Path, prefix: str = "tx158") -> Path:
    output_directory.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")[:-3]
    output_path = output_directory / f"{prefix}_{timestamp}.jpg"

    if not cv2.imwrite(str(output_path), frame):
        raise RuntimeError(f"Could not save {output_path}")

    return output_path


class FrameSource:
    """
    One camera, decoded on its own thread.

    Every camera runs independently so that a slow or stalled one cannot
    hold up the preview of the others.
    """

    def __init__(self, name: str, slug: str) -> None:
        self.name = name
        self.slug = slug
        self.error: str | None = None
        self.frame_count = 0
        self._frame = None
        self._lock = threading.Lock()
        self._stop_event = threading.Event()
        self._finished = threading.Event()
        self._thread: threading.Thread | None = None

    def start(self) -> None:
        self._thread = threading.Thread(
            target=self._run_guarded,
            name=f"{self.slug} reader",
            daemon=True,
        )
        self._thread.start()

    def _run_guarded(self) -> None:
        try:
            self._run()
        except Exception as error:  # Surfaced in the preview and on exit.
            self.error = f"{type(error).__name__}: {error}"
        finally:
            self._finished.set()

    def _run(self) -> None:
        raise NotImplementedError

    def _publish(self, frame) -> None:
        with self._lock:
            self._frame = frame
            self.frame_count += 1

    def latest(self):
        with self._lock:
            return self._frame

    @property
    def running(self) -> bool:
        return self._thread is not None and not self._finished.is_set()

    def status(self) -> str:
        if self.running:
            return "live" if self.frame_count else "connecting..."
        return self.error or "stopped"

    def stop(self) -> None:
        self._stop_event.set()
        if self._thread is not None:
            self._thread.join(timeout=3.0)


class VideoCaptureSource(FrameSource):
    """A camera OpenCV can open directly: a USB index or a stream URL."""

    def __init__(
        self,
        name: str,
        slug: str,
        target: int | str,
        backend: int = cv2.CAP_ANY,
    ) -> None:
        super().__init__(name, slug)
        self.target = target
        self.backend = backend

    def _run(self) -> None:
        # Auto-discovery has just opened and released a USB device to verify
        # that it works. Some Windows UVC drivers need a short time before the
        # device can be opened again for the real stream. Direct startup does
        # not hit that race, which is why `--usb 1` can work while Add USB
        # initially fails for the same camera.
        deadline = time.monotonic() + USB_OPEN_TIMEOUT

        while True:
            device = cv2.VideoCapture(self.target, self.backend)
            if device.isOpened():
                break

            device.release()
            if (
                not isinstance(self.target, int)
                or self._stop_event.is_set()
                or time.monotonic() >= deadline
            ):
                raise RuntimeError(f"could not open {self.target}")

            self._stop_event.wait(USB_OPEN_RETRY_DELAY)

        try:
            while not self._stop_event.is_set():
                success, frame = device.read()
                if not success or frame is None:
                    raise RuntimeError("the camera stopped sending frames")
                self._publish(frame)
        finally:
            device.release()


class RtpJpegSource(FrameSource):
    """
    The TX-158 network camera, driven without FFmpeg.

    OpenCV cannot open this firmware at all: it answers every SETUP with
    a UDP transport and a malformed "RTSP/1,0" status line, so FFmpeg
    stops with "Nonmatching transport in server reply". Speaking RTSP
    and RTP directly sidesteps that.
    """

    def __init__(
        self,
        name: str,
        slug: str,
        ip_address: str,
        timeout: float,
    ) -> None:
        super().__init__(name, slug)
        self.ip_address = ip_address
        self.timeout = timeout

    def _run(self) -> None:
        stream = RtpJpegStream(self.ip_address, timeout=max(self.timeout, 5.0))
        stream.open()

        # A short media timeout keeps the thread responsive to stop(), so the
        # RTSP TEARDOWN still reaches the camera when the window closes.
        if stream.media is not None:
            stream.media.settimeout(1.0)

        try:
            deadline = time.monotonic() + self.timeout

            while not self._stop_event.is_set():
                jpeg = stream.read_jpeg()
                if jpeg is None:
                    if time.monotonic() >= deadline:
                        raise RuntimeError(
                            "the camera stopped sending frames"
                            if self.frame_count
                            else "no RTP video arrived. Check that the camera "
                            "is powered on, that Windows is on its Cam-XXXXXX "
                            "Wi-Fi, and that no phone app is connected."
                        )
                    continue

                data = np.frombuffer(jpeg, dtype=np.uint8)
                frame = cv2.imdecode(data, cv2.IMREAD_COLOR)
                if frame is None:
                    continue

                deadline = time.monotonic() + self.timeout
                if not self.frame_count:
                    print(
                        f"Received {frame.shape[1]}x{frame.shape[0]} image "
                        f"({len(jpeg):,} bytes)."
                    )
                self._publish(frame)
        finally:
            stream.close()


class CtpSource(FrameSource):
    """A Jieli CTP camera: JPEG frames arriving over a raw TCP/UDP socket."""

    def __init__(
        self,
        name: str,
        slug: str,
        ip_address: str,
        transport: str,
        width: int,
        height: int,
        fps: int,
        timeout: float,
        verbose: bool,
        dump_path: Path | None,
    ) -> None:
        super().__init__(name, slug)
        self.ip_address = ip_address
        self.transport = transport
        self.width = width
        self.height = height
        self.fps = fps
        self.timeout = timeout
        self.verbose = verbose
        self.dump_path = dump_path

    def _run(self) -> None:
        control = CtpControl(self.ip_address, verbose=self.verbose)
        stream: socket.socket | None = None
        dump_file = None

        try:
            # Jieli requires the data channel to exist before OPEN_RT_STREAM.
            port = TCP_STREAM_PORT if self.transport == "tcp" else UDP_STREAM_PORT
            print(
                f"Connecting {self.transport.upper()} video channel "
                f"{self.ip_address}:{port}..."
            )
            stream = create_stream_socket(self.ip_address, self.transport)

            print(f"Connecting CTP control channel {self.ip_address}:{CTP_PORT}...")
            control.connect()
            control.send("APP_ACCESS", parameters={"type": "0", "ver": "3.1"})
            time.sleep(0.15)

            # format/type 0 asks the firmware for independently decodable JPEG frames.
            control.send(
                "OPEN_RT_STREAM",
                parameters={
                    "format": "0",
                    "type": "0",
                    "w": str(self.width),
                    "h": str(self.height),
                    "fps": str(self.fps),
                    "rate": "8000",
                },
            )

            if self.dump_path is not None:
                self.dump_path.parent.mkdir(parents=True, exist_ok=True)
                dump_file = self.dump_path.open("wb")
                print(f"Saving raw protocol bytes to {self.dump_path}")

            extractor = JpegExtractor()
            deadline = time.monotonic() + self.timeout
            received = 0

            print("Waiting for JPEG frames...")

            while not self._stop_event.is_set():
                if time.monotonic() >= deadline:
                    raise RuntimeError(self._stall_message(received))

                try:
                    block = receive_block(stream, self.transport)
                except socket.timeout:
                    control.send("CTP_KEEP_ALIVE", parameters=None)
                    continue

                if not block:
                    raise RuntimeError("the camera closed the video channel")

                received += len(block)
                if dump_file is not None:
                    dump_file.write(block)

                for jpeg in extractor.feed(block):
                    data = np.frombuffer(jpeg, dtype=np.uint8)
                    frame = cv2.imdecode(data, cv2.IMREAD_COLOR)
                    if frame is None:
                        continue

                    deadline = time.monotonic() + self.timeout
                    if not self.frame_count:
                        print(
                            f"Received {frame.shape[1]}x{frame.shape[0]} image "
                            f"({len(jpeg):,} bytes)."
                        )
                    self._publish(frame)
        except ConnectionRefusedError as error:
            raise RuntimeError(
                f"connection refused ({error}). The microscope may still be "
                "booting, or this firmware may use UDP: try --transport udp."
            ) from None
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

    def _stall_message(self, received: int) -> str:
        if self.frame_count:
            return "the camera stopped sending frames"

        if received:
            return (
                f"received {received:,} protocol bytes but found no JPEG. "
                "Re-run with --verbose --dump captures/stream.bin; the firmware "
                "may have selected H.264 despite the JPEG request."
            )

        return (
            "no video bytes arrived. Re-run with --verbose; if the CTP response "
            "reports an error, include that output when reporting the firmware."
        )


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


def make_placeholder(width: int, height: int, lines: list[str]):
    """A dark panel standing in for a camera that has no frame yet."""
    panel = np.zeros((max(1, height), max(1, width), 3), dtype=np.uint8)
    panel[:] = (40, 40, 40)

    for index, text in enumerate(lines):
        text_size, _baseline = cv2.getTextSize(
            text,
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            1,
        )
        origin = (
            max(8, (panel.shape[1] - text_size[0]) // 2),
            panel.shape[0] // 2 + index * 28,
        )
        cv2.putText(
            panel,
            text,
            origin,
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (200, 200, 200),
            1,
            cv2.LINE_AA,
        )

    return panel


def label_panel(panel, text: str, active: bool) -> None:
    """Name a panel in the combined view and outline the active camera."""
    height, width = panel.shape[:2]
    text_size, baseline = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 1)
    box_height = text_size[1] + baseline + 12

    if height > box_height + 8 and width > text_size[0] + 20:
        cv2.rectangle(
            panel,
            (0, height - box_height),
            (text_size[0] + 20, height),
            (0, 0, 0),
            -1,
        )
        cv2.putText(
            panel,
            text,
            (10, height - box_height // 2 + text_size[1] // 2),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (255, 255, 255),
            1,
            cv2.LINE_AA,
        )

    if active:
        cv2.rectangle(panel, (0, 0), (width - 1, height - 1), (35, 200, 90), 3)


class LiveWindow:
    """Resizable OpenCV preview of one or more cameras, with click controls."""

    def __init__(
        self,
        sources: list[FrameSource],
        output_directory: Path,
        zoom: float | None = None,
        combined: bool | None = None,
    ) -> None:
        self.sources = sources
        self.output_directory = output_directory
        # None means "fit to the screen once the frame size is known".
        self.zoom = zoom
        self.applied_zoom: float | None = None
        self.applied_shape: tuple[int, int] | None = None
        self.fullscreen = False
        self.vertical = False
        self.active = 0
        self.combined = len(sources) > 1 if combined is None else combined
        self.pending_action: str | None = None
        self.scanning = False
        self.buttons: list[tuple[str, str, tuple[int, int, int, int]]] = []
        self.panel_close_buttons: list[
            tuple[str, tuple[int, int, int, int]]
        ] = []
        self.pending_close_slug: str | None = None

        # Angle tool. Every camera carries its own pair of lines, keyed by
        # slug so they survive a camera being removed and added back.
        # Endpoints are kept in that camera's own unscaled coordinates, so
        # zooming, restacking or resizing never moves a line relative to
        # what it is measuring.
        self.measuring = False
        self.lines_by_slug: dict[str, list[list[list[float]]]] = {}
        self.dragging: tuple[str, int, int] | None = None

        # Where each visible camera sits in the composed image this frame:
        # slug -> (offset x, offset y, panel scale, frame width, frame height)
        self.placements: dict[str, tuple[float, float, float, int, int]] = {}

    def open(self) -> None:
        # WINDOW_NORMAL lets the user drag the window edges to resize.
        cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL)
        cv2.setMouseCallback(WINDOW_NAME, self._on_mouse)

    def _button_at(self, x: int, y: int) -> str | None:
        for _label, action, (left, top, right, bottom) in self.buttons:
            if left <= x <= right and top <= y <= bottom:
                return action
        return None

    def _close_button_at(self, x: int, y: int) -> str | None:
        for slug, (left, top, right, bottom) in self.panel_close_buttons:
            if left <= x <= right and top <= y <= bottom:
                return slug
        return None

    def _handle_at(self, x: int, y: int) -> tuple[str, int, int] | None:
        """Find the angle-tool endpoint under a click, on any visible camera."""
        if not self.measuring:
            return None

        closest = None
        closest_distance = float(HANDLE_GRAB_RADIUS)

        for slug in self.placements:
            lines = self._lines_for(slug)
            if lines is None:
                continue

            for line_index, line in enumerate(lines):
                for point_index, point in enumerate(line):
                    display_x, display_y = self._to_display(slug, point)
                    distance = math.hypot(display_x - x, display_y - y)
                    if distance <= closest_distance:
                        closest_distance = distance
                        closest = (slug, line_index, point_index)

        return closest

    def _on_mouse(
        self,
        event: int,
        x: int,
        y: int,
        _flags: int,
        _parameter: object,
    ) -> None:
        if event == cv2.EVENT_LBUTTONDOWN:
            # Buttons win over endpoints so the two never fight over a click.
            if (
                self._button_at(x, y) is None
                and self._close_button_at(x, y) is None
            ):
                self.dragging = self._handle_at(x, y)
            return

        if event == cv2.EVENT_MOUSEMOVE:
            if self.dragging is not None:
                slug, line_index, point_index = self.dragging
                lines = self.lines_by_slug.get(slug)

                # The camera can disappear mid-drag if it is removed.
                if lines is None or slug not in self.placements:
                    self.dragging = None
                    return

                lines[line_index][point_index] = self._from_display(slug, x, y)
            return

        if event == cv2.EVENT_LBUTTONUP:
            if self.dragging is not None:
                self.dragging = None
                return

            action = self._button_at(x, y)
            if action is not None:
                self.pending_action = action
                return

            close_slug = self._close_button_at(x, y)
            if close_slug is not None:
                self.pending_close_slug = close_slug

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

    def _next_source(self) -> None:
        if len(self.sources) > 1:
            self.active = (self.active + 1) % len(self.sources)
            # Switching cameras usually changes the frame size.
            self.applied_zoom = None

    def _toggle_combined(self) -> None:
        if len(self.sources) > 1:
            self.combined = not self.combined

    @staticmethod
    def _default_lines(width: int, height: int) -> list[list[list[float]]]:
        """Two crossing lines across the middle of one camera's frame."""
        centre_x, centre_y = width / 2.0, height / 2.0
        span = min(width, height) * 0.3

        return [
            [[centre_x - span, centre_y], [centre_x + span, centre_y]],
            [[centre_x, centre_y - span], [centre_x, centre_y + span]],
        ]

    def _lines_for(self, slug: str) -> list[list[list[float]]] | None:
        """The measuring lines of one visible camera, created on demand."""
        placement = self.placements.get(slug)
        if placement is None:
            return None

        lines = self.lines_by_slug.get(slug)
        if lines is None:
            _offset_x, _offset_y, _scale, width, height = placement
            lines = self._default_lines(width, height)
            self.lines_by_slug[slug] = lines

        return lines

    def _reset_lines(self) -> None:
        """Recentre the lines of every camera currently on screen."""
        for slug, placement in self.placements.items():
            _offset_x, _offset_y, _scale, width, height = placement
            self.lines_by_slug[slug] = self._default_lines(width, height)

    def _to_display(self, slug: str, point: list[float]) -> tuple[float, float]:
        """Camera coordinates to displayed pixels."""
        offset_x, offset_y, scale, _width, _height = self.placements[slug]
        zoom = self.zoom or 1.0
        return (
            (point[0] * scale + offset_x) * zoom,
            (point[1] * scale + offset_y) * zoom,
        )

    def _from_display(self, slug: str, x: float, y: float) -> list[float]:
        """Displayed pixels back to camera coordinates, clamped to the frame."""
        offset_x, offset_y, scale, width, height = self.placements[slug]
        zoom = self.zoom or 1.0
        camera_x = (x / zoom - offset_x) / scale
        camera_y = (y / zoom - offset_y) / scale
        return [
            min(max(camera_x, 0.0), width - 1.0),
            min(max(camera_y, 0.0), height - 1.0),
        ]

    def _toggle_measure(self) -> None:
        self.measuring = not self.measuring
        self.dragging = None

    @staticmethod
    def _line_direction(line: list[list[float]]) -> float:
        (start_x, start_y), (end_x, end_y) = line
        return math.atan2(end_y - start_y, end_x - start_x)

    @classmethod
    def _angle_between_lines(cls, lines: list[list[list[float]]]) -> float:
        """
        Angle between one camera's two lines in degrees, within [0, 180).

        Lines have no direction, so the result is independent of which
        end of each line is dragged where.
        """
        first = cls._line_direction(lines[0])
        second = cls._line_direction(lines[1])
        return math.degrees(second - first) % 180.0

    def _name_for(self, slug: str) -> str:
        for source in self.sources:
            if source.slug == slug:
                return source.name
        return slug

    @staticmethod
    def _draw_lines(image, lines, mapper, thickness: int, radius: int) -> None:
        """Draw one camera's two lines through a coordinate mapper."""
        colours = ((255, 200, 60), (80, 120, 255))

        for index, line in enumerate(lines):
            points = [
                tuple(int(round(value)) for value in mapper(point))
                for point in line
            ]
            cv2.line(image, points[0], points[1], colours[index], thickness, cv2.LINE_AA)

            for point in points:
                cv2.circle(image, point, radius, colours[index], -1)
                cv2.circle(image, point, radius, (255, 255, 255), 1, cv2.LINE_AA)

    @classmethod
    def _angle_text(cls, lines) -> str:
        angle = cls._angle_between_lines(lines)
        text = f"{angle:.1f} deg"

        # The complement only says something new past a right angle.
        if angle > 90.0:
            text += f"  ({180.0 - angle:.1f} acute)"

        return text

    def _draw_measure(self, display) -> None:
        """Draw each visible camera's measuring lines and its angle."""
        if not self.measuring:
            return

        for slug in self.placements:
            lines = self._lines_for(slug)
            if lines is None:
                continue

            self._draw_lines(
                display,
                lines,
                lambda point, slug=slug: self._to_display(slug, point),
                2,
                HANDLE_RADIUS,
            )
            self._draw_panel_angle(display, slug, lines)

    def _draw_panel_angle(self, display, slug: str, lines) -> None:
        """Put one camera's angle along the bottom of its own panel."""
        display_height, display_width = display.shape[:2]
        offset_x, offset_y, scale, width, height = self.placements[slug]
        zoom = self.zoom or 1.0

        panel_left = offset_x * zoom
        panel_top = offset_y * zoom
        panel_right = panel_left + width * scale * zoom
        panel_bottom = panel_top + height * scale * zoom

        text = self._angle_text(lines)
        text_size, baseline = cv2.getTextSize(
            text,
            cv2.FONT_HERSHEY_SIMPLEX,
            BUTTON_FONT_SCALE,
            BUTTON_TEXT_THICKNESS,
        )
        box_width = text_size[0] + 20
        box_height = text_size[1] + baseline + 12

        if panel_right - panel_left < box_width:
            return
        if panel_bottom - panel_top < box_height:
            return

        # Bottom right, so it never lands on the camera name label that
        # the combined view draws in the bottom left corner.
        right = (
            min(int(round(panel_right)), display_width)
            - PANEL_CLOSE_SIZE
            - 2 * PANEL_CLOSE_MARGIN
        )
        bottom = min(int(round(panel_bottom)), display_height)
        left = right - box_width
        top = bottom - box_height

        if left < panel_left:
            return

        cv2.rectangle(display, (left, top), (right, bottom), (0, 0, 0), -1)
        cv2.putText(
            display,
            text,
            (left + 10, bottom - (box_height - text_size[1]) // 2),
            cv2.FONT_HERSHEY_SIMPLEX,
            BUTTON_FONT_SCALE,
            (255, 255, 255),
            BUTTON_TEXT_THICKNESS,
            cv2.LINE_AA,
        )

    def _annotated(self, frame, slug: str):
        """Burn the measuring lines into a screenshot while the tool is on."""
        lines = self.lines_by_slug.get(slug)
        if not self.measuring or lines is None:
            return frame

        image = frame.copy()
        height, width = image.shape[:2]

        # Scale the marks with the frame so a 1280-wide capture and a
        # 640-wide one come out looking the same.
        thickness = max(2, round(width / 640))
        radius = max(4, round(width / 160))
        self._draw_lines(image, lines, lambda point: point, thickness, radius)

        text = self._angle_text(lines)
        scale = max(0.6, width / 1600)
        text_thickness = max(1, round(width / 800))
        text_size, baseline = cv2.getTextSize(
            text, cv2.FONT_HERSHEY_SIMPLEX, scale, text_thickness
        )
        box_height = text_size[1] + baseline + 16

        if height > box_height and width > text_size[0] + 24:
            cv2.rectangle(
                image,
                (0, height - box_height),
                (text_size[0] + 24, height),
                (0, 0, 0),
                -1,
            )
            cv2.putText(
                image,
                text,
                (12, height - box_height // 2 + text_size[1] // 2),
                cv2.FONT_HERSHEY_SIMPLEX,
                scale,
                (255, 255, 255),
                text_thickness,
                cv2.LINE_AA,
            )

        return image

    def _usb_indexes_in_use(self) -> set[int]:
        return {
            source.target
            for source in self.sources
            if isinstance(source, VideoCaptureSource)
            and isinstance(source.target, int)
        }

    def _scan_usb_cameras(self) -> None:
        """
        Look for USB cameras and show any that are not open yet.

        Probing runs on its own thread because opening several camera
        indexes takes seconds and would otherwise freeze the preview.
        """
        if self.scanning:
            return

        self.scanning = True

        def worker() -> None:
            try:
                cameras = find_usb_cameras(skip=self._usb_indexes_in_use())

                for index, width, height in cameras:
                    source = VideoCaptureSource(
                        name=f"USB camera {index}",
                        slug=f"usb{index}",
                        target=index,
                        backend=usb_backend(),
                    )
                    source.start()
                    self.sources.append(source)
                    print(f"Added USB camera {index} ({width}x{height}).")

                if not cameras:
                    print("No new USB camera responded.")
                elif len(self.sources) > 1:
                    self.combined = True
                    self.applied_zoom = None
            finally:
                self.scanning = False

        threading.Thread(
            target=worker,
            name="USB camera scan",
            daemon=True,
        ).start()

    def _drop_active_source(self) -> None:
        """Stop and remove the highlighted camera, keeping at least one."""
        if len(self.sources) <= 1:
            print("The last camera cannot be removed.")
            return

        self._remove_source(self.sources[self.active].slug)

    def _remove_source(self, slug: str, allow_last: bool = False) -> bool:
        """Remove one camera by slug; return True when none remain."""
        if len(self.sources) <= 1 and not allow_last:
            print("The last camera cannot be removed.")
            return False

        source_index = next(
            (index for index, source in enumerate(self.sources) if source.slug == slug),
            None,
        )
        if source_index is None:
            return not self.sources

        source = self.sources.pop(source_index)
        if source_index < self.active:
            self.active -= 1
        elif source_index == self.active:
            self.active = min(self.active, max(0, len(self.sources) - 1))

        self.combined = self.combined and len(self.sources) > 1
        self.applied_zoom = None
        source.stop()
        print(f"Removed {source.name}.")
        return not self.sources

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

    def _panel_for(self, source: FrameSource, width: int, height: int):
        frame = source.latest()
        if frame is None:
            return make_placeholder(width, height, [source.name, source.status()])
        return frame

    def _compose(self):
        """Build the unscaled image shown this iteration."""
        self.placements = {}

        if not self.combined or len(self.sources) == 1:
            source = self.sources[self.active]
            frame = source.latest()
            if frame is None:
                frame = make_placeholder(640, 480, [source.name, source.status()])

            self.placements[source.slug] = (
                0.0,
                0.0,
                1.0,
                frame.shape[1],
                frame.shape[0],
            )
            return frame.copy()

        available = [
            frame
            for frame in (source.latest() for source in self.sources)
            if frame is not None
        ]
        if available:
            reference_height = min(frame.shape[0] for frame in available)
            reference_width = min(frame.shape[1] for frame in available)
        else:
            reference_width, reference_height = 640, 480

        panels = []
        offset = 0
        for index, source in enumerate(self.sources):
            panel = self._panel_for(source, reference_width, reference_height)
            height, width = panel.shape[:2]

            # Normalize the axis the panels are joined along so that no
            # camera is upscaled beyond the smallest one.
            if self.vertical:
                target_width = reference_width
                target_height = max(1, round(height * target_width / width))
            else:
                target_height = reference_height
                target_width = max(1, round(width * target_height / height))

            interpolation = (
                cv2.INTER_AREA
                if target_width * target_height < width * height
                else cv2.INTER_LINEAR
            )
            panel = cv2.resize(
                panel,
                (target_width, target_height),
                interpolation=interpolation,
            )

            # Panels keep their aspect ratio, so one scale covers both axes.
            self.placements[source.slug] = (
                0.0 if self.vertical else float(offset),
                float(offset) if self.vertical else 0.0,
                target_width / width,
                width,
                height,
            )
            offset += target_height if self.vertical else target_width

            label_panel(panel, source.name, active=index == self.active)
            panels.append(panel)

        return np.vstack(panels) if self.vertical else np.hstack(panels)

    def _scale(self, frame):
        """Resize the frame for display and size the window to match."""
        if self.zoom is None:
            self.zoom = fit_zoom(frame)

        height, width = frame.shape[:2]
        new_width = max(1, int(round(width * self.zoom)))
        new_height = max(1, int(round(height * self.zoom)))

        # Only resize on an actual zoom or layout change, otherwise a window
        # the user dragged to a new size would snap back every frame.
        if self.zoom != self.applied_zoom or (width, height) != self.applied_shape:
            self.applied_zoom = self.zoom
            self.applied_shape = (width, height)
            if not self.fullscreen:
                cv2.resizeWindow(WINDOW_NAME, new_width, new_height)

        if (new_width, new_height) == (width, height):
            return frame

        interpolation = cv2.INTER_AREA if self.zoom < 1.0 else cv2.INTER_LINEAR
        return cv2.resize(
            frame,
            (new_width, new_height),
            interpolation=interpolation,
        )

    def _button_labels(self) -> list[tuple[str, str]]:
        labels = [("Save screenshot", "screenshot")]
        labels.append(("Hide angle" if self.measuring else "Measure angle", "measure"))

        if self.measuring:
            labels.append(("Reset lines", "reset_lines"))

        if len(self.sources) > 1:
            show_all = "Both cameras" if len(self.sources) == 2 else "All cameras"
            labels.append(("Single view" if self.combined else show_all, "combined"))
            labels.append(("Next camera", "next"))

            if self.combined:
                labels.append(
                    ("Side by side" if self.vertical else "Stack", "layout")
                )

        labels.append(("Scanning..." if self.scanning else "Add USB", "scan_usb"))

        if len(self.sources) > 1:
            labels.append(("Remove camera", "drop"))

        return labels

    def _draw_overlay(self, display) -> None:
        """Draw the control buttons and status text at a readable size."""
        height, width = display.shape[:2]

        margin = min(12, width // 20, height // 20)
        self.buttons = []

        # A heavily zoomed-out frame can be smaller than the buttons, in
        # which case there is nothing sensible to draw or click.
        if height >= BUTTON_HEIGHT + 2 * margin:
            left = margin
            for label, action in self._button_labels():
                text_size, _baseline = cv2.getTextSize(
                    label,
                    cv2.FONT_HERSHEY_SIMPLEX,
                    BUTTON_FONT_SCALE,
                    BUTTON_TEXT_THICKNESS,
                )
                button_width = text_size[0] + 18

                if left + button_width + margin > width:
                    break

                rectangle = (
                    left,
                    margin,
                    left + button_width,
                    margin + BUTTON_HEIGHT,
                )
                self.buttons.append((label, action, rectangle))
                self._draw_button(display, label, rectangle)
                left = rectangle[2] + max(4, margin // 2)

        status_lines = ["Battery: unavailable", f"Zoom: {round((self.zoom or 1.0) * 100)}%"]
        if not self.combined and len(self.sources) > 1:
            status_lines.insert(0, f"Camera: {self.sources[self.active].name}")
        for source in self.sources:
            if not source.running:
                status_lines.append(f"{source.name}: {source.status()}")

        # Sits below the button row, which can otherwise reach far enough
        # right to run underneath this text.
        first_line = margin + BUTTON_HEIGHT + 22

        for index, text in enumerate(status_lines):
            text_size, _baseline = cv2.getTextSize(
                text,
                cv2.FONT_HERSHEY_SIMPLEX,
                BUTTON_FONT_SCALE,
                BUTTON_TEXT_THICKNESS,
            )
            cv2.putText(
                display,
                text,
                (max(10, width - text_size[0] - 10), first_line + index * 22),
                cv2.FONT_HERSHEY_SIMPLEX,
                BUTTON_FONT_SCALE,
                (255, 255, 255),
                BUTTON_TEXT_THICKNESS,
                cv2.LINE_AA,
            )

    def _draw_panel_close_buttons(self, display) -> None:
        """Draw a clickable close control in every visible camera pane."""
        display_height, display_width = display.shape[:2]
        zoom = self.zoom or 1.0
        self.panel_close_buttons = []

        for slug, (offset_x, offset_y, scale, width, height) in self.placements.items():
            panel_left = int(round(offset_x * zoom))
            panel_top = int(round(offset_y * zoom))
            panel_right = min(
                int(round((offset_x + width * scale) * zoom)), display_width
            )
            panel_bottom = min(
                int(round((offset_y + height * scale) * zoom)), display_height
            )

            if (
                panel_right - panel_left < PANEL_CLOSE_SIZE + 2 * PANEL_CLOSE_MARGIN
                or panel_bottom - panel_top
                < PANEL_CLOSE_SIZE + 2 * PANEL_CLOSE_MARGIN
            ):
                continue

            right = panel_right - PANEL_CLOSE_MARGIN
            bottom = panel_bottom - PANEL_CLOSE_MARGIN
            left = right - PANEL_CLOSE_SIZE
            top = bottom - PANEL_CLOSE_SIZE
            rectangle = (left, top, right, bottom)
            self.panel_close_buttons.append((slug, rectangle))

            cv2.rectangle(display, (left, top), (right, bottom), (45, 45, 190), -1)
            cv2.rectangle(display, (left, top), (right, bottom), (255, 255, 255), 1)
            inset = 8
            cv2.line(
                display,
                (left + inset, top + inset),
                (right - inset, bottom - inset),
                (255, 255, 255),
                2,
                cv2.LINE_AA,
            )
            cv2.line(
                display,
                (right - inset, top + inset),
                (left + inset, bottom - inset),
                (255, 255, 255),
                2,
                cv2.LINE_AA,
            )

    @staticmethod
    def _draw_button(display, label: str, rectangle: tuple[int, int, int, int]) -> None:
        left, top, right, bottom = rectangle
        cv2.rectangle(display, (left, top), (right, bottom), (35, 145, 60), -1)
        cv2.rectangle(display, (left, top), (right, bottom), (255, 255, 255), 2)

        text_size, _baseline = cv2.getTextSize(
            label,
            cv2.FONT_HERSHEY_SIMPLEX,
            BUTTON_FONT_SCALE,
            BUTTON_TEXT_THICKNESS,
        )
        cv2.putText(
            display,
            label,
            (
                left + (right - left - text_size[0]) // 2,
                top + (bottom - top + text_size[1]) // 2,
            ),
            cv2.FONT_HERSHEY_SIMPLEX,
            BUTTON_FONT_SCALE,
            (255, 255, 255),
            BUTTON_TEXT_THICKNESS,
            cv2.LINE_AA,
        )

    def _save_screenshot(self) -> None:
        """
        Save camera frames at their native resolution.

        The preview's zoom and buttons are never included. The measuring
        lines are, whenever the angle tool is switched on.
        """
        chosen = (
            self.sources
            if self.combined and len(self.sources) > 1
            else [self.sources[self.active]]
        )

        for source in chosen:
            frame = source.latest()
            if frame is None:
                print(f"{source.name}: no frame to save ({source.status()}).")
                continue
            image = self._annotated(frame, source.slug)
            print(
                f"Image saved: "
                f"{save_frame(image, self.output_directory, source.slug)}"
            )

    def update(self) -> bool:
        """Display one iteration and return False when the user exits."""
        # _compose also records where each camera landed, which the angle
        # tool needs before it can map its lines onto the display.
        display = self._scale(self._compose())
        self._draw_measure(display)
        self._draw_panel_close_buttons(display)
        self._draw_overlay(display)

        cv2.imshow(WINDOW_NAME, display)
        key = cv2.waitKey(10) & 0xFF

        if self._was_closed():
            return False

        action, self.pending_action = self.pending_action, None
        close_slug, self.pending_close_slug = self.pending_close_slug, None
        if key == ord("s"):
            action = "screenshot"
        elif key in (9, ord("c")):
            action = "next"
        elif key == ord("b"):
            action = "combined"
        elif key == ord("u"):
            action = "scan_usb"
        elif key == ord("r"):
            action = "drop"
        elif key == ord("m"):
            action = "measure"
        elif key == ord("n"):
            action = "reset_lines"

        if action == "screenshot":
            self._save_screenshot()
        elif action == "next":
            self._next_source()
        elif action == "combined":
            self._toggle_combined()
        elif action == "layout":
            self.vertical = not self.vertical
        elif action == "scan_usb":
            self._scan_usb_cameras()
        elif action == "drop":
            self._drop_active_source()
        elif action == "measure":
            self._toggle_measure()
        elif action == "reset_lines":
            self._reset_lines()

        if close_slug is not None and self._remove_source(close_slug, allow_last=True):
            return False

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
        elif key == ord("v"):
            self.vertical = not self.vertical

        return key not in (27, ord("q"))


def print_controls(sources: list[FrameSource]) -> None:
    # Cameras can be added while running, so the multi-camera keys are
    # always listed even when only one source is open right now.
    lines = [
        "Controls:",
        "  Q or Esc  Quit             S  Save screenshot",
        "  + / -     Zoom             0  Fit to screen",
        "  1         100% size        F  Toggle fullscreen",
        "  U         Add USB cameras  R  Remove the highlighted camera",
        "  Tab or C  Next camera      B  Show one or all cameras",
        "  V         Stack the combined view vertically",
        "  M         Angle tool       N  Reset the measuring lines",
        "Click the X in a camera pane to stop and remove that camera.",
        "Drag the round handles to move the measuring lines. Each camera",
        "shows its own angle along the bottom of its panel, and keeps its",
        "lines in the screenshots taken while the tool is on.",
        "The same actions are available as buttons in the window, and the",
        "window edges can be dragged to resize.",
    ]
    print("\n".join(lines))


def build_sources(args: argparse.Namespace) -> list[FrameSource]:
    """Create one source per requested camera, network camera first."""
    sources: list[FrameSource] = []

    if not args.no_network:
        if args.transport == "rtp":
            print(
                f"Opening rtsp://{args.ip}:8080/?action=stream "
                "(direct RTP/JPEG over UDP)"
            )
            sources.append(
                RtpJpegSource(
                    name="Technaxx TX-158",
                    slug="tx158",
                    ip_address=args.ip,
                    timeout=args.timeout,
                )
            )
        elif args.transport == "rtsp":
            # This firmware accepts RTSP control on TCP 8080 but only offers
            # UDP media transport; asking FFmpeg for RTSP-over-TCP fails with
            # "Nonmatching transport in server reply".
            os.environ["OPENCV_FFMPEG_CAPTURE_OPTIONS"] = "rtsp_transport;udp"
            url = f"rtsp://{args.ip}:8080/?action=stream"
            print(f"Opening {url} (media transport: UDP)")
            sources.append(
                VideoCaptureSource(
                    name="Technaxx TX-158",
                    slug="tx158",
                    target=url,
                    backend=cv2.CAP_FFMPEG,
                )
            )
        else:
            sources.append(
                CtpSource(
                    name="Technaxx TX-158",
                    slug="tx158",
                    ip_address=args.ip,
                    transport=args.transport,
                    width=args.width,
                    height=args.height,
                    fps=args.fps,
                    timeout=args.timeout,
                    verbose=args.verbose,
                    dump_path=args.dump,
                )
            )

    for index in resolve_usb_indexes(args.usb):
        sources.append(
            VideoCaptureSource(
                name=f"USB camera {index}",
                slug=f"usb{index}",
                target=index,
                backend=usb_backend(),
            )
        )

    return sources


def resolve_usb_indexes(requested: list[str] | None) -> list[int]:
    """Turn --usb values into indexes, expanding 'auto' by probing."""
    if not requested:
        return []

    indexes: list[int] = []
    for value in requested:
        if value == "auto":
            detected = [index for index, _w, _h in find_usb_cameras()]
            if not detected:
                print("No USB camera responded to --usb auto.")
            indexes.extend(detected)
        else:
            index = int(value)
            if index in IGNORED_USB_INDEXES:
                print(f"Ignoring disabled USB camera {index}.")
                continue
            indexes.append(index)

    # Opening the same device twice fails, so keep the first mention only.
    unique: list[int] = []
    for index in indexes:
        if index not in unique:
            unique.append(index)
    return unique


def run_live(
    sources: list[FrameSource],
    output_directory: Path,
    timeout: float,
    one_frame: bool,
    scale: float | None,
) -> int:
    for source in sources:
        source.start()

    try:
        if not wait_for_first_frame(sources, timeout, require_all=one_frame):
            report_failures(sources)
            return 2

        if one_frame:
            for source in sources:
                frame = source.latest()
                if frame is None:
                    print(f"{source.name}: no frame ({source.status()}).")
                    continue
                print(
                    f"{source.name}: received {frame.shape[1]}x{frame.shape[0]} image.\n"
                    f"Image saved: {save_frame(frame, output_directory, source.slug)}"
                )
            return 0

        window = LiveWindow(sources, output_directory, scale)
        window.open()
        print_controls(sources)

        while window.update():
            if all(not source.running for source in sources):
                print("Every camera stopped sending frames.")
                report_failures(sources)
                return 2

        return 0
    finally:
        for source in sources:
            source.stop()
        cv2.destroyAllWindows()


def wait_for_first_frame(
    sources: list[FrameSource],
    timeout: float,
    require_all: bool = False,
) -> bool:
    """
    Wait until a camera delivers a frame.

    The preview starts as soon as one camera is ready, but saving with
    --once waits for all of them so no camera is missed by a moment.
    """
    deadline = time.monotonic() + max(timeout, 5.0)

    while time.monotonic() < deadline:
        ready = [source for source in sources if source.latest() is not None]
        if ready and (not require_all or len(ready) == len(sources)):
            return True
        if all(not source.running for source in sources):
            break
        time.sleep(0.05)

    return any(source.latest() is not None for source in sources)


def report_failures(sources: list[FrameSource]) -> None:
    for source in sources:
        if source.latest() is None:
            print(f"{source.name}: {source.status()}")

    if any(isinstance(source, VideoCaptureSource) for source in sources):
        print(
            "Check that the microscope is powered on, no phone app is "
            "connected, and Windows is on its Cam-XXXXXX Wi-Fi. For USB "
            "cameras, run --list-cameras to see which indexes exist."
        )


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


def parse_usb(value: str) -> str:
    """Parse --usb, where 'auto' means 'every USB camera that responds'."""
    if value.strip().lower() in ("auto", "all"):
        return "auto"

    try:
        index = int(value)
    except ValueError:
        raise argparse.ArgumentTypeError(
            f"expected 'auto' or a camera index such as 0, not {value!r}"
        ) from None

    if index < 0:
        raise argparse.ArgumentTypeError("a camera index cannot be negative")

    return str(index)


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Capture JPEG frames from a Technaxx TX-158/Jieli Wi-Fi microscope "
            "and, optionally, USB cameras alongside it."
        )
    )
    parser.add_argument("--ip", default=DEFAULT_IP, help="Camera IP address.")
    parser.add_argument(
        "--transport",
        choices=("rtp", "rtsp", "tcp", "udp"),
        default="rtp",
        help=(
            "Camera protocol. 'rtp' talks to the TX-158 directly and is the "
            "only one that works on the tested firmware; 'rtsp' routes through "
            "OpenCV/FFmpeg; tcp/udp are fallbacks for Jieli CTP firmware."
        ),
    )
    parser.add_argument(
        "--usb",
        type=parse_usb,
        action="append",
        nargs="?",
        const="auto",
        metavar="INDEX",
        help=(
            "Also show a USB camera. Repeat for several, or pass no value "
            "(or 'auto') to use every USB camera that responds. USB camera "
            "index 0 is always ignored."
        ),
    )
    parser.add_argument(
        "--no-network",
        action="store_true",
        help="Skip the Wi-Fi microscope and show only the USB cameras.",
    )
    parser.add_argument(
        "--list-cameras",
        action="store_true",
        help="Print the USB camera indexes this machine can open, then exit.",
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
        help="Save the first frame of each camera and exit.",
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

    if args.list_cameras:
        return list_cameras()

    if args.diagnose:
        return 0 if diagnose(args.ip) else 2

    sources = build_sources(args)
    if not sources:
        print("No cameras selected: --no-network needs at least one --usb.")
        return 2

    return run_live(
        sources=sources,
        output_directory=args.output,
        timeout=args.timeout,
        one_frame=args.once,
        scale=args.scale,
    )


if __name__ == "__main__":
    sys.exit(main())
