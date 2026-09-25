# Technaxx TX-158 camera capture

The tested TX-158 exposes MJPEG through an unusual combination:

- URL: `rtsp://192.168.25.1:8080/?action=stream`
- RTSP control: TCP port 8080
- Video transport: UDP only
- Video codec: MJPEG

The `HVCamwifi` APK contains this exact RTSP URL. FFmpeg/OpenCV must be told
to use UDP: forcing RTSP-over-TCP fails with `Nonmatching transport in server
reply`.

Connect Windows to the microscope's `Cam-XXXXXX` Wi-Fi network and power the
microscope on. First check that the Jieli services are available:

```powershell
py main.py --diagnose
```

Capture one image:

```powershell
py main.py --once --verbose
```

Show the live image:

```powershell
py main.py
```

Click **Save screenshot** in the preview, or press `S`, to save a clean camera
frame in `captures`. Press `Q` or Escape to exit. Screenshot filenames include
milliseconds, so rapid clicks do not overwrite one another.

## USB cameras

USB cameras can be shown next to the microscope in the same window. List the
ones this machine can open:

```powershell
py main.py --list-cameras
```

Then add them by index, or let the program find them:

```powershell
py main.py --usb 1          # microscope plus USB camera 1
py main.py --usb            # microscope plus every USB camera found
py main.py --usb 1 --usb 2  # several USB cameras
py main.py --no-network --usb 1
```

USB camera index `0` is disabled for this installation and is never opened,
including during automatic scans. Each visible camera pane has an **X** in its
lower-right corner that stops and removes just that camera.

With more than one camera the preview starts side by side. **Both cameras** /
**Single view** switches between the combined and single view, **Next camera**
picks which one the single view shows, and `V` stacks the combined view
vertically instead. The keyboard equivalents are `B`, `Tab` (or `C`) and `V`.

Every camera runs on its own thread, so a camera that is slow or missing never
holds up the others; one that is not ready yet shows as a grey panel. In the
combined view a screenshot saves one full-resolution image per camera, named
after it (`tx158_...jpg`, `usb1_...jpg`).

The preview currently reports `Battery: unavailable`. The vendor GoPlus app
does read a five-level battery value over TCP 8081, but its request packet is
implemented in a proprietary native library. Sending an unverified packet
could invoke a different camera command, so this project does not guess it.

Some newer Jieli-based firmware variants instead expose CTP control on TCP
3333 and live video on TCP 2229 or UDP 2228. The script retains this as a
fallback:

```powershell
py main.py --transport tcp --once --verbose
py main.py --transport udp --once --verbose
```

For a camera firmware variant that returns data but does not produce JPEG,
save a small raw sample:

```powershell
py main.py --once --verbose --dump captures/stream.bin
```

Do not run the official phone application at the same time. Many versions of
the firmware allow only one CTP client.
