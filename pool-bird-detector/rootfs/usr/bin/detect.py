#!/usr/bin/env python3
"""
Pool Bird Detector — watches a directory for new .mp4 files, extracts a frame,
runs TFLite MobileNet SSD COCO inference, and publishes results to MQTT.
"""

import argparse
import json
import logging
import os
import re
import subprocess
import tempfile
import time
from pathlib import Path

import numpy as np
import paho.mqtt.client as mqtt
import tflite_runtime.interpreter as tflite

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [bird-detector] %(levelname)s %(message)s",
    datefmt="%Y-%m-%dT%H:%M:%S",
)
log = logging.getLogger(__name__)

BIRD_LABELS = {"bird", "duck", "goose", "swan", "heron", "seagull"}


def load_labels(path: str) -> list[str]:
    with open(path) as f:
        return [line.strip() for line in f.readlines()]


def extract_frame(video_path: str, offset_seconds: int, out_path: str) -> bool:
    """Extract a single JPEG frame from the video at the given offset."""
    # Clamp offset so we don't seek past end of a short clip
    probe = subprocess.run(
        [
            "ffprobe", "-v", "error",
            "-show_entries", "format=duration",
            "-of", "default=noprint_wrappers=1:nokey=1",
            video_path,
        ],
        capture_output=True, text=True,
    )
    try:
        duration = float(probe.stdout.strip())
    except ValueError:
        duration = float("inf")

    seek = min(offset_seconds, max(0, int(duration) - 1))

    result = subprocess.run(
        [
            "ffmpeg", "-y",
            "-ss", str(seek),
            "-i", video_path,
            "-vframes", "1",
            "-q:v", "2",
            out_path,
        ],
        capture_output=True,
    )
    return result.returncode == 0


def preprocess_frame(image_path: str, input_size: tuple[int, int]) -> np.ndarray:
    """Load a JPEG and resize to model input size, returning uint8 array."""
    import struct, zlib

    # Use ffmpeg to resize to exact input dimensions and output raw RGB
    w, h = input_size
    result = subprocess.run(
        [
            "ffmpeg", "-y",
            "-i", image_path,
            "-vf", f"scale={w}:{h}",
            "-pix_fmt", "rgb24",
            "-f", "rawvideo",
            "pipe:1",
        ],
        capture_output=True,
    )
    if result.returncode != 0 or len(result.stdout) < w * h * 3:
        raise RuntimeError(f"ffmpeg resize failed: {result.stderr.decode()}")
    arr = np.frombuffer(result.stdout, dtype=np.uint8).reshape((1, h, w, 3))
    return arr


def run_inference(
    interpreter: tflite.Interpreter,
    frame: np.ndarray,
    labels: list[str],
    confidence_threshold: float,
) -> list[dict]:
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()

    interpreter.set_tensor(input_details[0]["index"], frame)
    interpreter.invoke()

    # SSD MobileNet outputs: boxes, class_ids, scores, count
    boxes = interpreter.get_tensor(output_details[0]["index"])[0]
    class_ids = interpreter.get_tensor(output_details[1]["index"])[0]
    scores = interpreter.get_tensor(output_details[2]["index"])[0]
    count = int(interpreter.get_tensor(output_details[3]["index"])[0])

    detections = []
    for i in range(count):
        score = float(scores[i])
        if score < confidence_threshold:
            continue
        label_idx = int(class_ids[i])
        label = labels[label_idx] if label_idx < len(labels) else f"class_{label_idx}"
        detections.append({"label": label, "confidence": round(score, 4), "box": boxes[i].tolist()})

    return detections


def publish_result(client: mqtt.Client, topic: str, video_path: str, detections: list[dict]):
    bird_detections = [d for d in detections if d["label"].lower() in BIRD_LABELS]
    has_bird = len(bird_detections) > 0
    top_confidence = max((d["confidence"] for d in bird_detections), default=0.0)

    payload = {
        "file": os.path.basename(video_path),
        "bird_detected": has_bird,
        "confidence": top_confidence,
        "detections": detections,
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }
    client.publish(topic, json.dumps(payload), retain=False)
    log.info(
        "Published: bird_detected=%s confidence=%.2f file=%s",
        has_bird, top_confidence, os.path.basename(video_path),
    )


def watch_and_process(
    watch_dir: str,
    interpreter: tflite.Interpreter,
    labels: list[str],
    mqtt_client: mqtt.Client,
    mqtt_topic: str,
    confidence_threshold: float,
    frame_offset: int,
):
    input_details = interpreter.get_input_details()
    h = input_details[0]["shape"][1]
    w = input_details[0]["shape"][2]

    # Use inotifywait to stream new .mp4 close-write events
    proc = subprocess.Popen(
        [
            "inotifywait",
            "-m",
            "-e", "close_write",
            "--format", "%f",
            watch_dir,
        ],
        stdout=subprocess.PIPE,
        text=True,
    )

    log.info("Watching %s (input size %dx%d)", watch_dir, w, h)

    for line in proc.stdout:
        filename = line.strip()
        if not filename.lower().endswith(".mp4"):
            continue

        video_path = os.path.join(watch_dir, filename)
        log.info("New clip: %s", filename)

        # Small delay — camera may still be flushing
        time.sleep(1)

        with tempfile.NamedTemporaryFile(suffix=".jpg", delete=False) as tmp:
            frame_path = tmp.name

        try:
            if not extract_frame(video_path, frame_offset, frame_path):
                log.warning("Frame extraction failed for %s", filename)
                publish_result(mqtt_client, mqtt_topic, video_path, [])
                continue

            frame = preprocess_frame(frame_path, (w, h))
            detections = run_inference(interpreter, frame, labels, confidence_threshold)
            publish_result(mqtt_client, mqtt_topic, video_path, detections)
        except Exception as exc:
            log.error("Error processing %s: %s", filename, exc)
            try:
                publish_result(mqtt_client, mqtt_topic, video_path, [])
            except Exception:
                pass
        finally:
            try:
                os.unlink(frame_path)
            except OSError:
                pass


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--watch-dir", required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--labels", required=True)
    parser.add_argument("--mqtt-host", required=True)
    parser.add_argument("--mqtt-port", type=int, default=1883)
    parser.add_argument("--mqtt-user", default="")
    parser.add_argument("--mqtt-password", default="")
    parser.add_argument("--mqtt-topic", default="pool_motion/classification")
    parser.add_argument("--confidence", type=float, default=0.30)
    parser.add_argument("--frame-offset", type=int, default=3)
    args = parser.parse_args()

    labels = load_labels(args.labels)
    log.info("Loaded %d labels", len(labels))

    interpreter = tflite.Interpreter(model_path=args.model)
    interpreter.allocate_tensors()
    log.info("Model loaded: %s", args.model)

    client = mqtt.Client(client_id="pool_bird_detector")
    if args.mqtt_user:
        client.username_pw_set(args.mqtt_user, args.mqtt_password)
    client.connect(args.mqtt_host, args.mqtt_port, keepalive=60)
    client.loop_start()
    log.info("MQTT connected to %s:%d", args.mqtt_host, args.mqtt_port)

    try:
        watch_and_process(
            watch_dir=args.watch_dir,
            interpreter=interpreter,
            labels=labels,
            mqtt_client=client,
            mqtt_topic=args.mqtt_topic,
            confidence_threshold=args.confidence,
            frame_offset=args.frame_offset,
        )
    finally:
        client.loop_stop()
        client.disconnect()


if __name__ == "__main__":
    main()
