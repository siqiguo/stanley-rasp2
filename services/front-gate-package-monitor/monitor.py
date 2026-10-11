#!/usr/bin/env python3
"""Low-resource, non-AI package-presence monitor for a fixed camera.

The monitor compares a configured region of interest against an operator-captured
empty reference image.  It intentionally detects a newly persistent object, not
the semantic category "parcel".
"""

from __future__ import annotations

import json
import os
import subprocess
import threading
import time
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np
from flask import Flask, Response, jsonify, request, send_file


DATA_DIR = Path(os.getenv("DATA_DIR", "/var/lib/front-gate-package-monitor"))
CAMERA_URL = os.environ["CAMERA_URL"]
CALIBRATION_TOKEN = os.environ["CALIBRATION_TOKEN"]
BIND_HOST = os.getenv("BIND_HOST", "0.0.0.0")
PORT = int(os.getenv("PORT", "8791"))
CHECK_INTERVAL = float(os.getenv("CHECK_INTERVAL", "10"))
BURST_INTERVAL = float(os.getenv("BURST_INTERVAL", "10"))
BURST_DURATION = float(os.getenv("BURST_DURATION", "90"))
TRIGGER_ALLOWED_IP = os.getenv("TRIGGER_ALLOWED_IP", "192.168.3.254")
FRAME_WIDTH = int(os.getenv("FRAME_WIDTH", "960"))
ROI = tuple(float(v) for v in os.getenv("ROI", "0.02,0.10,0.47,0.58").split(","))
PRESENT_THRESHOLD = float(os.getenv("PRESENT_THRESHOLD", "0.035"))
CLEAR_THRESHOLD = float(os.getenv("CLEAR_THRESHOLD", "0.016"))
PRESENT_LARGEST_THRESHOLD = float(os.getenv("PRESENT_LARGEST_THRESHOLD", "0.18"))
CLEAR_LARGEST_THRESHOLD = float(os.getenv("CLEAR_LARGEST_THRESHOLD", "0.13"))
CONFIRM_SAMPLES = int(os.getenv("CONFIRM_SAMPLES", "3"))

BASELINE_FILE = DATA_DIR / "baseline.jpg"
MAX_BASELINES = int(os.getenv("MAX_BASELINES", "24"))
AUTO_BASELINE_INTERVAL = float(os.getenv("AUTO_BASELINE_INTERVAL", "1800"))
LATEST_FILE = DATA_DIR / "latest.jpg"
ANNOTATED_FILE = DATA_DIR / "annotated.jpg"
STATE_FILE = DATA_DIR / "state.json"

app = Flask(__name__)
lock = threading.Lock()
started_at = time.time()
stop_event = threading.Event()
wake_event = threading.Event()
burst_until = 0.0

state: dict = {
    "ok": False,
    "state": "starting",
    "baseline_ready": False,
    "package_present": False,
    "changed_area_ratio": 0.0,
    "largest_region_ratio": 0.0,
    "last_check": None,
    "last_change": None,
    "baseline_at": None,
    "baseline_count": 0,
    "candidate_samples": 0,
    "clear_samples": 0,
    "error": None,
}


def atomic_write_json(path: Path, payload: dict) -> None:
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    tmp.replace(path)


def capture_frame() -> np.ndarray:
    command = [
        "ffmpeg",
        "-hide_banner",
        "-loglevel",
        "error",
        "-rtsp_transport",
        "tcp",
        "-i",
        CAMERA_URL,
        "-frames:v",
        "1",
        "-vf",
        f"scale={FRAME_WIDTH}:-2",
        "-q:v",
        "3",
        "-f",
        "image2pipe",
        "-vcodec",
        "mjpeg",
        "pipe:1",
    ]
    result = subprocess.run(command, check=True, capture_output=True, timeout=20)
    frame = cv2.imdecode(np.frombuffer(result.stdout, dtype=np.uint8), cv2.IMREAD_COLOR)
    if frame is None:
        raise RuntimeError("ffmpeg returned an unreadable frame")
    return frame


def roi_bounds(frame: np.ndarray) -> tuple[int, int, int, int]:
    height, width = frame.shape[:2]
    x1, y1, x2, y2 = ROI
    return (
        max(0, min(width - 1, round(x1 * width))),
        max(0, min(height - 1, round(y1 * height))),
        max(1, min(width, round(x2 * width))),
        max(1, min(height, round(y2 * height))),
    )


def normalized_gray(image: np.ndarray) -> np.ndarray:
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (9, 9), 0)
    return cv2.createCLAHE(clipLimit=1.5, tileGridSize=(8, 8)).apply(gray)


def compare(frame: np.ndarray, baseline: np.ndarray) -> tuple[float, float, np.ndarray]:
    x1, y1, x2, y2 = roi_bounds(frame)
    current_roi = frame[y1:y2, x1:x2]
    base_roi = baseline[y1:y2, x1:x2]
    if current_roi.shape != base_roi.shape:
        raise RuntimeError("baseline dimensions no longer match the camera frame")

    current_gray = normalized_gray(current_roi)
    base_gray = normalized_gray(base_roi)
    illumination_shift = int(np.median(current_gray.astype(np.int16) - base_gray.astype(np.int16)))
    adjusted = np.clip(current_gray.astype(np.int16) - illumination_shift, 0, 255).astype(np.uint8)
    diff = cv2.absdiff(adjusted, base_gray)
    _, mask = cv2.threshold(diff, 24, 255, cv2.THRESH_BINARY)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=2)

    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    roi_area = float(mask.shape[0] * mask.shape[1])
    min_contour_area = roi_area * 0.0025
    significant = [c for c in contours if cv2.contourArea(c) >= min_contour_area]
    significant_area = sum(cv2.contourArea(c) for c in significant)
    largest = max((cv2.contourArea(c) for c in significant), default=0.0)

    annotated = frame.copy()
    cv2.rectangle(annotated, (x1, y1), (x2, y2), (75, 220, 75), 3)
    for contour in significant:
        rx, ry, rw, rh = cv2.boundingRect(contour)
        cv2.rectangle(annotated, (x1 + rx, y1 + ry), (x1 + rx + rw, y1 + ry + rh), (0, 80, 255), 3)
    return significant_area / roi_area, largest / roi_area, annotated


def save_jpeg(path: Path, frame: np.ndarray) -> None:
    ok, encoded = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 88])
    if not ok:
        raise RuntimeError(f"failed to encode {path.name}")
    tmp = path.with_suffix(".tmp")
    tmp.write_bytes(encoded.tobytes())
    tmp.replace(path)


def set_baseline(frame: np.ndarray) -> None:
    now = datetime.now().astimezone().isoformat(timespec="seconds")
    if not BASELINE_FILE.exists():
        save_jpeg(BASELINE_FILE, frame)
    else:
        stamp = datetime.now().astimezone().strftime("%Y%m%d-%H%M%S")
        save_jpeg(DATA_DIR / f"baseline-{stamp}.jpg", frame)
        extra = sorted(DATA_DIR.glob("baseline-*.jpg"), key=lambda path: path.stat().st_mtime)
        for stale in extra[:-max(0, MAX_BASELINES - 1)]:
            stale.unlink(missing_ok=True)
    with lock:
        state.update(
            baseline_ready=True,
            package_present=False,
            state="clear",
            changed_area_ratio=0.0,
            largest_region_ratio=0.0,
            baseline_at=now,
            baseline_count=len(baseline_files()),
            last_change=now,
            candidate_samples=0,
            clear_samples=0,
            error=None,
        )
        atomic_write_json(STATE_FILE, state)


def maybe_add_clear_baseline(frame: np.ndarray, changed_ratio: float) -> None:
    """Gradually follow daylight changes without ever learning a detected object."""
    if AUTO_BASELINE_INTERVAL <= 0 or changed_ratio > CLEAR_THRESHOLD:
        return
    references = baseline_files()
    if not references:
        return
    newest_mtime = max(path.stat().st_mtime for path in references)
    if time.time() - newest_mtime < AUTO_BASELINE_INTERVAL:
        return
    stamp = datetime.now().astimezone().strftime("%Y%m%d-%H%M%S")
    save_jpeg(DATA_DIR / f"baseline-{stamp}.jpg", frame)
    extra = sorted(DATA_DIR.glob("baseline-*.jpg"), key=lambda path: path.stat().st_mtime)
    for stale in extra[:-max(0, MAX_BASELINES - 1)]:
        stale.unlink(missing_ok=True)


def baseline_files() -> list[Path]:
    paths = [BASELINE_FILE] if BASELINE_FILE.exists() else []
    paths.extend(sorted(DATA_DIR.glob("baseline-*.jpg")))
    return paths


def monitor_once() -> None:
    frame = capture_frame()
    save_jpeg(LATEST_FILE, frame)
    now = datetime.now().astimezone().isoformat(timespec="seconds")
    references = baseline_files()
    if not references:
        with lock:
            state.update(
                ok=True,
                state="calibration_required",
                baseline_ready=False,
                last_check=now,
                error=None,
            )
            atomic_write_json(STATE_FILE, state)
        return

    comparisons = []
    for reference in references:
        baseline = cv2.imread(str(reference), cv2.IMREAD_COLOR)
        if baseline is None:
            continue
        try:
            comparisons.append((*compare(frame, baseline), reference))
        except RuntimeError:
            continue
    if not comparisons:
        raise RuntimeError("no readable baseline matches the camera frame")
    changed_ratio, largest_ratio, annotated, matched_reference = min(
        comparisons, key=lambda item: (item[0], item[1])
    )
    save_jpeg(ANNOTATED_FILE, annotated)

    with lock:
        previous = bool(state["package_present"])
        if not previous:
            present_candidate = (
                changed_ratio >= PRESENT_THRESHOLD
                and largest_ratio >= PRESENT_LARGEST_THRESHOLD
            )
            state["candidate_samples"] = state["candidate_samples"] + 1 if present_candidate else 0
            state["clear_samples"] = 0
            if state["candidate_samples"] >= CONFIRM_SAMPLES:
                state["package_present"] = True
                state["last_change"] = now
        else:
            clear_candidate = largest_ratio <= CLEAR_LARGEST_THRESHOLD
            state["clear_samples"] = state["clear_samples"] + 1 if clear_candidate else 0
            state["candidate_samples"] = 0
            if state["clear_samples"] >= CONFIRM_SAMPLES:
                state["package_present"] = False
                state["last_change"] = now

        state.update(
            ok=True,
            state="package_present" if state["package_present"] else "clear",
            baseline_ready=True,
            baseline_count=len(references),
            matched_baseline=matched_reference.name,
            changed_area_ratio=round(changed_ratio, 5),
            largest_region_ratio=round(largest_ratio, 5),
            last_check=now,
            error=None,
        )
        atomic_write_json(STATE_FILE, state)
    if not state["package_present"]:
        maybe_add_clear_baseline(frame, changed_ratio)


def monitor_loop() -> None:
    global burst_until
    while not stop_event.is_set():
        started = time.monotonic()
        try:
            monitor_once()
        except Exception as exc:  # keep the service observable after camera failures
            with lock:
                state.update(
                    ok=False,
                    state="error",
                    last_check=datetime.now().astimezone().isoformat(timespec="seconds"),
                    error=f"{type(exc).__name__}: {exc}",
                )
                atomic_write_json(STATE_FILE, state)
        with lock:
            burst_active = time.time() < burst_until
        interval = BURST_INTERVAL if burst_active else CHECK_INTERVAL
        wake_event.wait(max(1.0, interval - (time.monotonic() - started)))
        wake_event.clear()


@app.get("/api/status")
def status_endpoint() -> Response:
    with lock:
        payload = dict(state)
    payload["uptime_seconds"] = round(time.time() - started_at)
    return jsonify(payload)


@app.post("/api/check")
def event_check_endpoint() -> Response:
    global burst_until
    if TRIGGER_ALLOWED_IP and request.remote_addr != TRIGGER_ALLOWED_IP:
        return jsonify({"ok": False, "error": "forbidden"}), 403
    with lock:
        burst_until = max(burst_until, time.time() + BURST_DURATION)
        state["burst_until"] = datetime.fromtimestamp(burst_until).astimezone().isoformat(timespec="seconds")
    wake_event.set()
    return jsonify({"ok": True, "burst_seconds": BURST_DURATION})


@app.get("/snapshot.jpg")
def snapshot_endpoint() -> Response:
    path = ANNOTATED_FILE if ANNOTATED_FILE.exists() else LATEST_FILE
    if not path.exists():
        return Response(status=404)
    return send_file(path, mimetype="image/jpeg", max_age=0)


@app.get("/baseline.jpg")
def baseline_endpoint() -> Response:
    if not BASELINE_FILE.exists():
        return Response(status=404)
    return send_file(BASELINE_FILE, mimetype="image/jpeg", max_age=0)


@app.post("/api/calibrate")
def calibrate_endpoint() -> Response:
    if request.headers.get("X-Calibration-Token") != CALIBRATION_TOKEN:
        return jsonify({"ok": False, "error": "unauthorized"}), 401
    frame = capture_frame()
    save_jpeg(LATEST_FILE, frame)
    set_baseline(frame)
    return jsonify({"ok": True, "baseline_at": state["baseline_at"]})


def load_previous_state() -> None:
    if STATE_FILE.exists():
        try:
            previous = json.loads(STATE_FILE.read_text(encoding="utf-8"))
            state.update(previous)
        except (OSError, ValueError):
            pass
    state["baseline_ready"] = bool(baseline_files())
    state["baseline_count"] = len(baseline_files())


def main() -> None:
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    load_previous_state()
    worker = threading.Thread(target=monitor_loop, name="package-monitor", daemon=True)
    worker.start()
    app.run(host=BIND_HOST, port=PORT, threaded=True, use_reloader=False)


if __name__ == "__main__":
    main()
