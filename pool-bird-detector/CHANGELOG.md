# Changelog

## 1.2.0
- Add MQTT Discovery: add-on now self-registers `binary_sensor`, `confidence`, and `last_file` entities in HA on startup
- Fix `bird_detected` state payload to use `ON`/`OFF` strings for correct binary sensor behaviour

## 1.1.0
- Upgrade detection model from MobileNet SSD v1 quant to EfficientDet-Lite2 (448×448 input, significantly better accuracy)
- Scan 3 frames per clip (25/50/75%) and publish best result
- Load COCO labels from model metadata automatically

## 1.0.0
- Initial release: inotifywait file watcher, ffmpeg frame extraction, TFLite inference, MQTT publish
