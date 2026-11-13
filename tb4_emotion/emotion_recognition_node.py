#!/usr/bin/env python3
"""ROS 2 Node: emotion_recognition_node.

Reads webcam or Windows-to-WSL2 HTTP MJPEG stream (stream_cam.py), detects
faces with OpenCV YuNet (or Haar fallback), classifies 6 facial expressions
using the pretrained ONNX FER+ model (emotion-ferplus-8.onnx), draws real-time
overlay, and publishes std_msgs/msg/String to /detected_emotion.
"""

import math
import os
from pathlib import Path
import subprocess
import threading
import time
from typing import List, Optional, Sequence, Tuple

EMOTION_LABELS = {
    "Happy": "Happy (Vui ve -> Cong sang trai)",
    "Sad": "Sad (Buon -> Cong sang phai)",
    "Neutral": "Neutral (Binh thuong -> Dung tai cho)",
    "Angry": "Angry (Tuc gian -> Lui nhe ve phia sau)",
    "Surprise": "Surprise (Ngac nhien -> Xoay 1 vong tai cho)",
    "Fear": "Fear (So hai -> Tien nhe ve phia truoc)",
    "None": "None (Mat mat -> Cho 2s deadzone roi dung)",
}

# FER+ 8 classes: [neutral, happiness, surprise, sadness, anger, disgust, fear, contempt]
FERPLUS_TARGET_INDICES = (
    (0, "Neutral"),
    (1, "Happy"),
    (2, "Surprise"),
    (3, "Sad"),
    (4, "Angry"),
    (6, "Fear"),
)


def classify_ferplus_logits(logits: Sequence[float]) -> Tuple[str, float]:
    """Compute normalized probability over the 6 target emotions from FER+ logits."""
    if len(logits) < 7:
        raise ValueError(f"Expected at least 7 logits, got {len(logits)}")
    target_vals = [float(logits[idx]) for idx, _ in FERPLUS_TARGET_INDICES]
    max_v = max(target_vals)
    exps = [math.exp(v - max_v) for v in target_vals]
    total = sum(exps) or 1.0
    probs = [e / total for e in exps]
    best_i = max(range(len(probs)), key=probs.__getitem__)
    return FERPLUS_TARGET_INDICES[best_i][1], float(probs[best_i])


class EmotionFilter:
    """Debounces frame-by-frame emotion predictions while reporting face loss immediately."""

    def __init__(
        self,
        min_confidence: float = 0.35,
        confirm_frames: int = 3,
    ) -> None:
        self.min_confidence = min_confidence
        self.confirm_frames = confirm_frames
        self._candidate: str = "None"
        self._streak: int = 0
        self._current: str = "None"
        self._confidence: float = 0.0

    def update(
        self, raw_label: Optional[str], score: float
    ) -> Tuple[str, float]:
        if not raw_label or raw_label == "None" or score < self.min_confidence:
            self._candidate = "None"
            self._streak = 0
            self._current = "None"
            self._confidence = 0.0
            return "None", 0.0

        if raw_label == self._candidate:
            self._streak += 1
        else:
            self._candidate = raw_label
            self._streak = 1

        if self._streak >= self.confirm_frames:
            self._current = raw_label
            self._confidence = float(score)
        elif raw_label == self._current:
            self._confidence = float(score)

        return self._current, self._confidence


def parse_camera_source(val: object) -> object:
    if isinstance(val, int):
        return val
    s = str(val).strip()
    return int(s) if s.isdigit() else s


def detect_wsl_host_ip() -> Optional[str]:
    """Detects Windows host gateway IP from inside WSL2 via `ip route show`."""
    try:
        out = subprocess.check_output(
            ["ip", "route", "show", "default"],
            text=True,
            timeout=1.0,
            stderr=subprocess.DEVNULL,
        )
        parts = out.strip().split()
        if "via" in parts:
            idx = parts.index("via")
            if idx + 1 < len(parts):
                return parts[idx + 1]
    except Exception:
        pass
    return None


def resolve_camera_candidates(
    source: object,
    wsl_host_ip: Optional[str] = None,
    stream_port: int = 5005,
) -> List[object]:
    """Returns ordered camera sources to try (supports Windows-to-WSL2 HTTP fallback)."""
    parsed = parse_camera_source(source)
    host_ip = wsl_host_ip if wsl_host_ip is not None else detect_wsl_host_ip()
    fallback_url = f"http://{host_ip}:{stream_port}/video" if host_ip else None

    if parsed in ("auto", "wsl", 0):
        cands: List[object] = [0]
        if fallback_url:
            cands.append(fallback_url)
        return cands
    return [parsed]


def _default_model_dir() -> Path:
    try:
        from ament_index_python.packages import get_package_share_directory

        d = Path(get_package_share_directory("tb4_emotion")) / "models"
        if d.is_dir():
            return d
    except Exception:
        pass
    return Path(__file__).resolve().parent.parent / "models"


class LatestFrameGrabber:
    """Continuously drains VideoCapture in a background thread to eliminate MJPEG stream lag."""

    def __init__(self, cap) -> None:
        self.cap = cap
        self._lock = threading.Lock()
        self._frame = None
        self._ok = False
        self._running = True
        self._thread = threading.Thread(target=self._loop, daemon=True)
        self._thread.start()

    def _loop(self) -> None:
        while self._running and self.cap.isOpened():
            ok, frame = self.cap.read()
            with self._lock:
                self._ok = bool(ok and frame is not None)
                if self._ok:
                    self._frame = frame
            if not ok:
                break

    def read(self):
        with self._lock:
            if not self._ok or self._frame is None:
                return False, None
            return True, self._frame.copy()

    def release(self) -> None:
        self._running = False
        try:
            self._thread.join(timeout=0.5)
        except Exception:
            pass
        try:
            self.cap.release()
        except Exception:
            pass


def main(args=None) -> None:
    import cv2
    import numpy as np
    from rcl_interfaces.msg import ParameterDescriptor
    import rclpy
    from rclpy.node import Node
    from std_msgs.msg import String

    class EmotionRecognitionNode(Node):
        def __init__(self) -> None:
            super().__init__("emotion_recognition_node")
            self.declare_parameter(
                "camera_source",
                "0",
                ParameterDescriptor(dynamic_typing=True),
            )
            model_dir = _default_model_dir()
            self.declare_parameter(
                "emotion_model_path",
                str(model_dir / "emotion-ferplus-8.onnx"),
            )
            self.declare_parameter(
                "face_model_path",
                str(model_dir / "face_detection_yunet_2023mar.onnx"),
            )
            self.declare_parameter("min_confidence", 0.35)
            self.declare_parameter("confirm_frames", 3)
            self.declare_parameter("show_window", True)

            self.camera_source = parse_camera_source(
                self.get_parameter("camera_source").value
            )
            emotion_model_path = str(
                self.get_parameter("emotion_model_path").value
            )
            face_model_path = str(self.get_parameter("face_model_path").value)
            self.show_window = bool(self.get_parameter("show_window").value)

            if not os.path.isfile(emotion_model_path):
                raise FileNotFoundError(
                    f"Emotion model not found: {emotion_model_path}"
                )

            self.emotion_net = cv2.dnn.readNetFromONNX(emotion_model_path)
            self.face_detector = None
            if os.path.isfile(face_model_path) and hasattr(
                cv2, "FaceDetectorYN"
            ):
                self.face_detector = cv2.FaceDetectorYN.create(
                    face_model_path, "", (320, 320), 0.6, 0.3, 5000
                )
            self.haar_cascade = cv2.CascadeClassifier(
                os.path.join(
                    cv2.data.haarcascades, "haarcascade_frontalface_default.xml"
                )
            )

            self.filter = EmotionFilter(
                min_confidence=float(
                    self.get_parameter("min_confidence").value
                ),
                confirm_frames=int(self.get_parameter("confirm_frames").value),
            )
            self.pub = self.create_publisher(String, "/detected_emotion", 10)
            self.grabber: Optional[LatestFrameGrabber] = None
            self.active_source: Optional[object] = None
            self.last_open_try = 0.0
            self.last_logged = ""

            self._open_camera()
            self.timer = self.create_timer(0.05, self._tick)
            self.get_logger().info(
                f"emotion_recognition_node ready (camera_source={self.camera_source})"
            )

        def _open_camera(self) -> None:
            now = time.monotonic()
            if now - self.last_open_try < 2.0:
                return
            self.last_open_try = now
            if self.grabber is not None:
                self.grabber.release()
                self.grabber = None

            candidates = resolve_camera_candidates(self.camera_source)
            for src in candidates:
                if isinstance(src, int):
                    cap = cv2.VideoCapture(src, cv2.CAP_V4L2)
                    if not cap.isOpened():
                        cap = cv2.VideoCapture(src)
                else:
                    cap = cv2.VideoCapture(str(src))
                if cap.isOpened():
                    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
                    self.grabber = LatestFrameGrabber(cap)
                    self.active_source = src
                    self.get_logger().info(f"Camera connected: {src}")
                    return

            self.get_logger().warn(
                f"Cannot open camera (tried {candidates}). Waiting for webcam or Windows stream_cam.py..."
            )

        def _detect_largest_face(
            self, frame: np.ndarray
        ) -> Optional[Tuple[int, int, int, int]]:
            h, w = frame.shape[:2]
            if self.face_detector is not None:
                self.face_detector.setInputSize((w, h))
                _, faces = self.face_detector.detect(frame)
                if faces is not None and len(faces) > 0:
                    best = max(faces, key=lambda f: float(f[2] * f[3]))
                    x, y, bw, bh = (
                        int(best[0]),
                        int(best[1]),
                        int(best[2]),
                        int(best[3]),
                    )
                    x = max(0, x)
                    y = max(0, y)
                    bw = min(w - x, bw)
                    bh = min(h - y, bh)
                    if bw > 20 and bh > 20:
                        return x, y, bw, bh

            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            rects = self.haar_cascade.detectMultiScale(
                gray, scaleFactor=1.1, minNeighbors=5, minSize=(48, 48)
            )
            if rects is not None and len(rects) > 0:
                best = max(rects, key=lambda r: int(r[2]) * int(r[3]))
                return int(best[0]), int(best[1]), int(best[2]), int(best[3])
            return None

        def _classify_face(
            self, frame: np.ndarray, box: Tuple[int, int, int, int]
        ) -> Tuple[str, float]:
            x, y, bw, bh = box
            roi = frame[y : y + bh, x : x + bw]
            gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
            resized = cv2.resize(gray, (64, 64), interpolation=cv2.INTER_AREA)
            blob = resized.astype(np.float32).reshape(1, 1, 64, 64)
            self.emotion_net.setInput(blob)
            logits = self.emotion_net.forward().reshape(-1)
            return classify_ferplus_logits(logits.tolist())

        def _tick(self) -> None:
            if self.grabber is None:
                self._open_camera()
                return
            ok, frame = self.grabber.read()
            if not ok or frame is None:
                if time.monotonic() - self.last_open_try > 2.5:
                    self._open_camera()
                return

            frame = cv2.flip(frame, 1)
            face_box = self._detect_largest_face(frame)
            raw_label: Optional[str] = None
            raw_score: float = 0.0

            if face_box is not None:
                raw_label, raw_score = self._classify_face(frame, face_box)

            emotion, conf = self.filter.update(raw_label, raw_score)
            msg = String()
            msg.data = emotion
            self.pub.publish(msg)

            if emotion != self.last_logged:
                self.last_logged = emotion
                self.get_logger().info(
                    f"Emotion: {emotion} ({conf * 100:.1f}%)"
                )

            if self.show_window:
                h, w = frame.shape[:2]
                if face_box is not None:
                    fx, fy, fw, fh = face_box
                    color = (
                        (0, 255, 0)
                        if emotion not in ("None", "Neutral")
                        else (0, 200, 255)
                    )
                    cv2.rectangle(
                        frame, (fx, fy), (fx + fw, fy + fh), color, 2
                    )
                cv2.rectangle(frame, (0, 0), (w, 76), (25, 25, 25), -1)
                conf_txt = f"{conf * 100:.1f}%" if emotion != "None" else "--"
                cv2.putText(
                    frame,
                    f"Emotion: {emotion} ({conf_txt})",
                    (12, 30),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.8,
                    (0, 255, 128) if emotion != "None" else (80, 180, 255),
                    2,
                )
                cv2.putText(
                    frame,
                    EMOTION_LABELS.get(emotion, emotion),
                    (12, 62),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    (230, 230, 230),
                    1,
                )
                try:
                    cv2.imshow("TurtleBot 4 Emotion Recognition", frame)
                    cv2.waitKey(1)
                except cv2.error:
                    self.show_window = False

        def destroy_node(self) -> bool:
            if self.grabber is not None:
                self.grabber.release()
                self.grabber = None
            try:
                cv2.destroyAllWindows()
            except Exception:
                pass
            return super().destroy_node()

    rclpy.init(args=args)
    node = EmotionRecognitionNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()
