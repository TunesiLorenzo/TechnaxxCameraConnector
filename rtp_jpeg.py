"""
Receive RTP/JPEG video (RFC 2435) straight from the camera.

The TX-158 firmware answers RTSP on TCP 8080 but always replies with a
UDP transport and a malformed "RTSP/1,0" status line, so FFmpeg aborts
with "Nonmatching transport in server reply". Doing the handshake here
avoids FFmpeg entirely.

RTP/JPEG sends only the entropy-coded scan data; the quantization and
Huffman tables have to be rebuilt from the 8-byte payload header, which
is what make_headers() does.
"""

from __future__ import annotations

import socket
import struct

# How many RTP ports to try before giving up. The Windows Firewall rule
# created for this program covers the whole resulting range.
PORT_ATTEMPTS = 10

# Table sources: RFC 2435 appendices A and B.
LUMA_QUANTIZER = (
    16, 11, 12, 14, 12, 10, 16, 14,
    13, 14, 18, 17, 16, 19, 24, 40,
    26, 24, 22, 22, 24, 49, 35, 37,
    29, 40, 58, 51, 61, 60, 57, 51,
    56, 55, 64, 72, 92, 78, 64, 68,
    87, 69, 55, 56, 80, 109, 81, 87,
    95, 98, 103, 104, 103, 62, 77, 113,
    121, 112, 100, 120, 92, 101, 103, 99,
)

CHROMA_QUANTIZER = (
    17, 18, 18, 24, 21, 24, 47, 26,
    26, 47, 99, 66, 56, 66, 99, 99,
    99, 99, 99, 99, 99, 99, 99, 99,
    99, 99, 99, 99, 99, 99, 99, 99,
    99, 99, 99, 99, 99, 99, 99, 99,
    99, 99, 99, 99, 99, 99, 99, 99,
    99, 99, 99, 99, 99, 99, 99, 99,
    99, 99, 99, 99, 99, 99, 99, 99,
)

LUMA_DC_CODE_LENGTHS = bytes(
    (0, 1, 5, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0)
)
LUMA_DC_SYMBOLS = bytes(range(12))

CHROMA_DC_CODE_LENGTHS = bytes(
    (0, 3, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0)
)
CHROMA_DC_SYMBOLS = bytes(range(12))

LUMA_AC_CODE_LENGTHS = bytes(
    (0, 2, 1, 3, 3, 2, 4, 3, 5, 5, 4, 4, 0, 0, 1, 0x7D)
)
LUMA_AC_SYMBOLS = bytes((
    0x01, 0x02, 0x03, 0x00, 0x04, 0x11, 0x05, 0x12,
    0x21, 0x31, 0x41, 0x06, 0x13, 0x51, 0x61, 0x07,
    0x22, 0x71, 0x14, 0x32, 0x81, 0x91, 0xA1, 0x08,
    0x23, 0x42, 0xB1, 0xC1, 0x15, 0x52, 0xD1, 0xF0,
    0x24, 0x33, 0x62, 0x72, 0x82, 0x09, 0x0A, 0x16,
    0x17, 0x18, 0x19, 0x1A, 0x25, 0x26, 0x27, 0x28,
    0x29, 0x2A, 0x34, 0x35, 0x36, 0x37, 0x38, 0x39,
    0x3A, 0x43, 0x44, 0x45, 0x46, 0x47, 0x48, 0x49,
    0x4A, 0x53, 0x54, 0x55, 0x56, 0x57, 0x58, 0x59,
    0x5A, 0x63, 0x64, 0x65, 0x66, 0x67, 0x68, 0x69,
    0x6A, 0x73, 0x74, 0x75, 0x76, 0x77, 0x78, 0x79,
    0x7A, 0x83, 0x84, 0x85, 0x86, 0x87, 0x88, 0x89,
    0x8A, 0x92, 0x93, 0x94, 0x95, 0x96, 0x97, 0x98,
    0x99, 0x9A, 0xA2, 0xA3, 0xA4, 0xA5, 0xA6, 0xA7,
    0xA8, 0xA9, 0xAA, 0xB2, 0xB3, 0xB4, 0xB5, 0xB6,
    0xB7, 0xB8, 0xB9, 0xBA, 0xC2, 0xC3, 0xC4, 0xC5,
    0xC6, 0xC7, 0xC8, 0xC9, 0xCA, 0xD2, 0xD3, 0xD4,
    0xD5, 0xD6, 0xD7, 0xD8, 0xD9, 0xDA, 0xE1, 0xE2,
    0xE3, 0xE4, 0xE5, 0xE6, 0xE7, 0xE8, 0xE9, 0xEA,
    0xF1, 0xF2, 0xF3, 0xF4, 0xF5, 0xF6, 0xF7, 0xF8,
    0xF9, 0xFA,
))

CHROMA_AC_CODE_LENGTHS = bytes(
    (0, 2, 1, 2, 4, 4, 3, 4, 7, 5, 4, 4, 0, 1, 2, 0x77)
)
CHROMA_AC_SYMBOLS = bytes((
    0x00, 0x01, 0x02, 0x03, 0x11, 0x04, 0x05, 0x21,
    0x31, 0x06, 0x12, 0x41, 0x51, 0x07, 0x61, 0x71,
    0x13, 0x22, 0x32, 0x81, 0x08, 0x14, 0x42, 0x91,
    0xA1, 0xB1, 0xC1, 0x09, 0x23, 0x33, 0x52, 0xF0,
    0x15, 0x62, 0x72, 0xD1, 0x0A, 0x16, 0x24, 0x34,
    0xE1, 0x25, 0xF1, 0x17, 0x18, 0x19, 0x1A, 0x26,
    0x27, 0x28, 0x29, 0x2A, 0x35, 0x36, 0x37, 0x38,
    0x39, 0x3A, 0x43, 0x44, 0x45, 0x46, 0x47, 0x48,
    0x49, 0x4A, 0x53, 0x54, 0x55, 0x56, 0x57, 0x58,
    0x59, 0x5A, 0x63, 0x64, 0x65, 0x66, 0x67, 0x68,
    0x69, 0x6A, 0x73, 0x74, 0x75, 0x76, 0x77, 0x78,
    0x79, 0x7A, 0x82, 0x83, 0x84, 0x85, 0x86, 0x87,
    0x88, 0x89, 0x8A, 0x92, 0x93, 0x94, 0x95, 0x96,
    0x97, 0x98, 0x99, 0x9A, 0xA2, 0xA3, 0xA4, 0xA5,
    0xA6, 0xA7, 0xA8, 0xA9, 0xAA, 0xB2, 0xB3, 0xB4,
    0xB5, 0xB6, 0xB7, 0xB8, 0xB9, 0xBA, 0xC2, 0xC3,
    0xC4, 0xC5, 0xC6, 0xC7, 0xC8, 0xC9, 0xCA, 0xD2,
    0xD3, 0xD4, 0xD5, 0xD6, 0xD7, 0xD8, 0xD9, 0xDA,
    0xE2, 0xE3, 0xE4, 0xE5, 0xE6, 0xE7, 0xE8, 0xE9,
    0xEA, 0xF2, 0xF3, 0xF4, 0xF5, 0xF6, 0xF7, 0xF8,
    0xF9, 0xFA,
))


def make_quantization_tables(quality: int) -> tuple[bytes, bytes]:
    """Derive the luma and chroma tables from a Q factor (RFC 2435 A)."""
    factor = min(99, max(1, quality))
    scale = 5000 // factor if quality < 50 else 200 - factor * 2

    def scaled(table: tuple[int, ...]) -> bytes:
        return bytes(
            min(255, max(1, (value * scale + 50) // 100)) for value in table
        )

    return scaled(LUMA_QUANTIZER), scaled(CHROMA_QUANTIZER)


def make_headers(
    jpeg_type: int,
    width: int,
    height: int,
    luma_table: bytes,
    chroma_table: bytes,
    restart_interval: int = 0,
) -> bytes:
    """Build the JPEG headers that RTP/JPEG strips from the wire format."""
    header = bytearray(b"\xff\xd8")  # SOI

    # DQT, one segment per table.
    for identifier, table in ((0, luma_table), (1, chroma_table)):
        header += b"\xff\xdb" + struct.pack(">HB", 67, identifier) + table

    if restart_interval:
        header += b"\xff\xdd" + struct.pack(">HH", 4, restart_interval)

    # Type 0 is 4:2:2 and type 1 is 4:2:0; both use one chroma block.
    luma_sampling = 0x21 if (jpeg_type & 0x3F) == 0 else 0x22

    header += b"\xff\xc0" + struct.pack(
        ">HBHHB",
        17,      # segment length
        8,       # sample precision
        height,
        width,
        3,       # component count
    )
    header += bytes((1, luma_sampling, 0, 2, 0x11, 1, 3, 0x11, 1))

    # DHT, the four standard baseline tables.
    for identifier, lengths, symbols in (
        (0x00, LUMA_DC_CODE_LENGTHS, LUMA_DC_SYMBOLS),
        (0x10, LUMA_AC_CODE_LENGTHS, LUMA_AC_SYMBOLS),
        (0x01, CHROMA_DC_CODE_LENGTHS, CHROMA_DC_SYMBOLS),
        (0x11, CHROMA_AC_CODE_LENGTHS, CHROMA_AC_SYMBOLS),
    ):
        segment_length = 3 + len(lengths) + len(symbols)
        header += b"\xff\xc4" + struct.pack(">HB", segment_length, identifier)
        header += lengths + symbols

    # SOS
    header += b"\xff\xda" + struct.pack(">HB", 12, 3)
    header += bytes((1, 0x00, 2, 0x11, 3, 0x11, 0, 63, 0))

    return bytes(header)


class RtpJpegStream:
    """RTSP control plus RTP/JPEG reassembly for the TX-158 firmware."""

    def __init__(
        self,
        ip_address: str,
        rtsp_port: int = 8080,
        client_port: int = 51234,
        timeout: float = 5.0,
    ) -> None:
        self.ip_address = ip_address
        self.rtsp_port = rtsp_port
        self.client_port = client_port
        self.timeout = timeout
        self.url = f"rtsp://{ip_address}:{rtsp_port}/?action=stream"
        self.control: socket.socket | None = None
        self.media: socket.socket | None = None
        self.session = ""
        self.sequence = 0

        self.fragments = bytearray()
        self.current_timestamp: int | None = None
        self.expected_offset = 0
        self.header: bytes | None = None
        self.frame_size = (0, 0)

    def _request(self, method: str, extra: str = "") -> str:
        assert self.control is not None
        self.sequence += 1
        message = (
            f"{method} {self.url} RTSP/1.0\r\n"
            f"CSeq: {self.sequence}\r\n"
            f"{extra}\r\n"
        )
        self.control.sendall(message.encode("utf-8"))
        return self.control.recv(4096).decode("utf-8", "replace")

    def _bind_media_socket(self) -> None:
        """
        Bind the RTP port, stepping forward when one is already taken.

        A second instance of the program, or one still shutting down,
        would otherwise fail outright. RTP keeps the data port even, so
        the candidates advance in twos.
        """
        first = self.client_port
        last_error: OSError | None = None

        for candidate in range(first, first + 2 * PORT_ATTEMPTS, 2):
            media = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
            media.setsockopt(socket.SOL_SOCKET, socket.SO_RCVBUF, 1 << 20)
            media.settimeout(self.timeout)

            try:
                media.bind(("0.0.0.0", candidate))
            except OSError as error:
                media.close()
                last_error = error
                continue

            self.media = media
            self.client_port = candidate
            return

        raise OSError(
            f"No free RTP port between {first} and "
            f"{first + 2 * PORT_ATTEMPTS - 2}. Another copy of this program "
            f"is probably already running. ({last_error})"
        )

    def open(self) -> None:
        """Run the RTSP handshake and start the UDP media flow."""
        # Bind the media port before SETUP so the first packets are not lost.
        self._bind_media_socket()

        self.control = socket.create_connection(
            (self.ip_address, self.rtsp_port),
            timeout=self.timeout,
        )
        self.control.settimeout(self.timeout)

        self._request("OPTIONS")
        self._request("DESCRIBE", "Accept: application/sdp\r\n")

        reply = self._request(
            "SETUP",
            "Transport: RTP/AVP;unicast;"
            f"client_port={self.client_port}-{self.client_port + 1}\r\n",
        )

        for line in reply.splitlines():
            if line.lower().startswith("session:"):
                self.session = line.split(":", 1)[1].strip().split(";")[0]

        if not self.session:
            raise RuntimeError(f"The camera did not open a session:\n{reply}")

        self._request(
            "PLAY",
            f"Session: {self.session}\r\nRange: npt=0.000-\r\n",
        )

    def _handle_packet(self, packet: bytes) -> bytes | None:
        """Add one RTP packet, returning a complete JPEG when finished."""
        if len(packet) < 20:
            return None

        csrc_count = packet[0] & 0x0F
        has_extension = (packet[0] >> 4) & 1
        marker = (packet[1] >> 7) & 1
        timestamp = struct.unpack(">I", packet[4:8])[0]

        start = 12 + 4 * csrc_count
        if has_extension:
            extension_words = struct.unpack(">H", packet[start + 2:start + 4])[0]
            start += 4 + 4 * extension_words

        payload = packet[start:]
        if len(payload) < 8:
            return None

        fragment_offset = int.from_bytes(payload[1:4], "big")
        jpeg_type = payload[4]
        quality = payload[5]
        width = payload[6] * 8
        height = payload[7] * 8
        cursor = 8

        restart_interval = 0
        if jpeg_type >= 64:
            restart_interval = struct.unpack(">H", payload[cursor:cursor + 2])[0]
            cursor += 4

        if fragment_offset == 0:
            if quality >= 128:
                # Tables travel inline; skip the 4-byte table header.
                table_length = struct.unpack(">H", payload[cursor + 2:cursor + 4])[0]
                cursor += 4
                tables = payload[cursor:cursor + table_length]
                cursor += table_length
                luma = bytes(tables[:64])
                chroma = bytes(tables[64:128]) or luma
            else:
                luma, chroma = make_quantization_tables(quality)

            self.header = make_headers(
                jpeg_type, width, height, luma, chroma, restart_interval
            )
            self.frame_size = (width, height)
            self.fragments = bytearray()
            self.current_timestamp = timestamp
            self.expected_offset = 0

        if self.header is None or timestamp != self.current_timestamp:
            # A fragment from a frame whose start we missed.
            return None

        if fragment_offset != self.expected_offset:
            # Lost a UDP packet, so this frame can no longer be decoded.
            self.header = None
            return None

        scan = payload[cursor:]
        self.fragments += scan
        self.expected_offset += len(scan)

        if not marker:
            return None

        payload_data = bytes(self.fragments)
        header = self.header
        self.header = None
        self.fragments = bytearray()

        # This firmware ignores the RTP/JPEG wire format and sends whole
        # JPEG files, headers included. Only rebuild headers when the
        # payload really is bare scan data, as RFC 2435 requires.
        if payload_data[:2] != b"\xff\xd8":
            payload_data = header + payload_data

        if payload_data[-2:] != b"\xff\xd9":
            payload_data += b"\xff\xd9"

        return payload_data

    def read_jpeg(self) -> bytes | None:
        """Block until one complete JPEG arrives, or return None on timeout."""
        assert self.media is not None

        while True:
            try:
                packet, _address = self.media.recvfrom(65536)
            except socket.timeout:
                return None

            image = self._handle_packet(packet)
            if image is not None:
                return image

    def close(self) -> None:
        if self.control is not None:
            try:
                if self.session:
                    self._request("TEARDOWN", f"Session: {self.session}\r\n")
            except OSError:
                pass
            self.control.close()
            self.control = None

        if self.media is not None:
            self.media.close()
            self.media = None
