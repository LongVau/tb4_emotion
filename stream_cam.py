#!/usr/bin/env python3
"""Windows MJPEG webcam streaming server for WSL2 (port 5005).

Copy or run this script on Windows:
    python stream_cam.py

Stream URL for WSL2:
    http://<WINDOWS_HOST_IP>:5005/video
"""

from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import time
import cv2

cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)


class StreamHandler(BaseHTTPRequestHandler):
    def do_GET(self):
        self.send_response(200)
        self.send_header(
            "Content-Type", "multipart/x-mixed-replace; boundary=frame"
        )
        self.send_header("Cache-Control", "no-cache, private")
        self.send_header("Pragma", "no-cache")
        self.end_headers()
        try:
            while True:
                ret, frame = cap.read()
                if not ret or frame is None:
                    time.sleep(0.02)
                    continue
                ok, jpeg = cv2.imencode(
                    ".jpg", frame, [int(cv2.IMWRITE_JPEG_QUALITY), 80]
                )
                if not ok:
                    continue
                self.wfile.write(
                    b"--frame\r\nContent-Type: image/jpeg\r\n\r\n"
                    + jpeg.tobytes()
                    + b"\r\n"
                )
        except (BrokenPipeError, ConnectionResetError):
            pass

    def log_message(self, format, *args):
        return


if __name__ == "__main__":
    print("Streaming webcam on http://0.0.0.0:5005/video ... (Ctrl+C to stop)")
    ThreadingHTTPServer(("0.0.0.0", 5005), StreamHandler).serve_forever()
