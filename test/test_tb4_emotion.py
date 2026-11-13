import math
import pytest

from tb4_emotion.emotion_recognition_node import (
    EMOTION_LABELS,
    EmotionFilter,
    classify_ferplus_logits,
    detect_wsl_host_ip,
    parse_camera_source,
    resolve_camera_candidates,
)


def test_parse_camera_source():
    assert parse_camera_source(0) == 0
    assert parse_camera_source("0") == 0
    assert parse_camera_source("/dev/video0") == "/dev/video0"
    assert (
        parse_camera_source("http://172.23.64.1:5005/video")
        == "http://172.23.64.1:5005/video"
    )


def test_resolve_camera_candidates_for_wsl_stream():
    cands = resolve_camera_candidates("auto", wsl_host_ip="172.23.64.1")
    assert cands == [0, "http://172.23.64.1:5005/video"]

    cands_zero = resolve_camera_candidates(0, wsl_host_ip="172.23.64.1")
    assert cands_zero == [0, "http://172.23.64.1:5005/video"]

    cands_url = resolve_camera_candidates(
        "http://192.168.1.10:5005/video", wsl_host_ip="172.23.64.1"
    )
    assert cands_url == ["http://192.168.1.10:5005/video"]


def test_classify_ferplus_logits_masks_non_target_classes():
    # FER+ 8 classes: [neutral, happiness, surprise, sadness, anger, disgust, fear, contempt]
    logits = [0.1, 0.2, 0.1, 0.1, 0.5, 10.0, 2.5, 9.0]
    label, prob = classify_ferplus_logits(logits)
    assert label == "Fear"
    assert 0.5 < prob <= 1.0

    # Check all 6 target emotions exist in EMOTION_LABELS
    for target in ("Happy", "Sad", "Neutral", "Angry", "Surprise", "Fear", "None"):
        assert target in EMOTION_LABELS


def test_emotion_filter_debounce_and_none():
    ef = EmotionFilter(min_confidence=0.40, confirm_frames=3)
    assert ef.update("Happy", 0.80) == ("None", 0.0)
    assert ef.update("Happy", 0.82) == ("None", 0.0)
    assert ef.update("Happy", 0.85) == ("Happy", 0.85)

    # Switching emotion requires confirm_frames consecutive frames; keeps previous confirmed emotion until new one confirms
    assert ef.update("Sad", 0.90) == ("Happy", 0.85)
    assert ef.update("Sad", 0.90) == ("Happy", 0.85)
    assert ef.update("Sad", 0.92) == ("Sad", 0.92)

    # Face lost (None) or low confidence immediately reports ("None", 0.0) so controller manages the 2.0s deadzone
    assert ef.update(None, 0.0) == ("None", 0.0)
    assert ef.update("Sad", 0.20) == ("None", 0.0)


def test_onnx_models_load_and_infer():
    import cv2
    import numpy as np
    from pathlib import Path

    model_dir = Path(__file__).resolve().parent.parent / "models"
    fer_path = model_dir / "emotion-ferplus-8.onnx"
    yunet_path = model_dir / "face_detection_yunet_2023mar.onnx"
    assert fer_path.is_file() and fer_path.stat().st_size > 100_000
    assert yunet_path.is_file() and yunet_path.stat().st_size > 100_000

    net = cv2.dnn.readNetFromONNX(str(fer_path))
    dummy = np.zeros((1, 1, 64, 64), dtype=np.float32)
    net.setInput(dummy)
    out = net.forward().reshape(-1)
    assert out.shape == (8,)
    label, prob = classify_ferplus_logits(out.tolist())
    assert label in ("Neutral", "Happy", "Surprise", "Sad", "Angry", "Fear")
    assert 0.0 < prob <= 1.0


def test_emotion_controller_six_trajectories():
    from tb4_emotion.emotion_control_node import EmotionController

    ctrl = EmotionController(face_loss_timeout=2.0)

    ctrl.on_emotion("Happy", 0.0)
    assert ctrl.step(0.1) == pytest.approx((0.18, 0.45))

    ctrl.on_emotion("Sad", 1.0)
    assert ctrl.step(1.1) == pytest.approx((0.18, -0.45))

    ctrl.on_emotion("Neutral", 2.0)
    assert ctrl.step(2.1) == pytest.approx((0.0, 0.0))

    ctrl.on_emotion("Angry", 3.0)
    assert ctrl.step(3.1) == pytest.approx((-0.12, 0.0))

    ctrl.on_emotion("Fear", 4.0)
    assert ctrl.step(4.1) == pytest.approx((0.12, 0.0))


def test_emotion_controller_surprise_one_full_turn_and_hold():
    from tb4_emotion.emotion_control_node import EmotionController

    ctrl = EmotionController(face_loss_timeout=2.0, spin_angular=1.0)
    ctrl.on_emotion("Surprise", 10.0)
    assert ctrl.step(10.5) == pytest.approx((0.0, 1.0))

    # Brief face flicker (< 2.0s) during Surprise does not reset spin timer
    ctrl.on_emotion("None", 11.0)
    assert ctrl.step(12.0) == pytest.approx((0.0, 1.0))
    ctrl.on_emotion("Surprise", 15.0)
    assert ctrl.step(10.0 + 2 * math.pi - 0.1) == pytest.approx((0.0, 1.0))

    # After 2*pi seconds, stops even if Surprise is still held
    ctrl.on_emotion("Surprise", 10.0 + 2 * math.pi + 0.1)
    assert ctrl.step(10.0 + 2 * math.pi + 0.1) == pytest.approx((0.0, 0.0))


def test_emotion_controller_two_second_face_loss_deadzone():
    from tb4_emotion.emotion_control_node import EmotionController

    ctrl = EmotionController(face_loss_timeout=2.0)
    ctrl.on_emotion("Fear", 0.0)
    assert ctrl.step(0.5) == pytest.approx((0.12, 0.0))

    # Face lost at t=1.0, last valid face at t=0.0 -> still moving at t=1.9, stops after t=2.0
    ctrl.on_emotion("None", 1.0)
    assert ctrl.step(1.9) == pytest.approx((0.12, 0.0))
    assert ctrl.step(2.1) == pytest.approx((0.0, 0.0))
    assert ctrl.active_emotion == "None"

