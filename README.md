# Pool Bird Detector — Home Assistant Add-on

Watches `/media/pool_motion/` for new `.mp4` clips (written by a Blink camera automation), extracts frames with ffmpeg, runs EfficientDet-Lite2 COCO inference via TFLite, and publishes bird/duck detection results to your existing Mosquitto MQTT broker.

## Installation

1. In HA, go to **Settings → Add-ons → Add-on Store → ⋮ → Repositories**
2. Add: `https://github.com/inverted29/ha-pool-bird-detector`
3. Find **Pool Bird Detector** and install it
4. Configure options (see below) and start

## Configuration

| Option | Default | Description |
|---|---|---|
| `mqtt_host` | `core-mosquitto` | Mosquitto broker hostname |
| `mqtt_port` | `1883` | Broker port |
| `mqtt_user` | `""` | MQTT username (leave blank if unauthenticated) |
| `mqtt_password` | `""` | MQTT password |
| `mqtt_topic` | `pool_motion/classification` | Topic to publish results to |
| `watch_dir` | `/media/pool_motion` | Directory to watch for new `.mp4` files |
| `confidence_threshold` | `0.30` | Minimum detection confidence (0–1) |
| `frame_offset_seconds` | `0` | Fixed frame offset in seconds (0 = auto-sample at 25/50/75% of clip) |

## MQTT Topics

### `pool_motion/classification`
Published on every new clip:
```json
{
  "file": "clip_20240615_023412.mp4",
  "bird_detected": "ON",
  "confidence": 0.85,
  "detections": [
    {"label": "bird", "confidence": 0.85, "box": [0.1, 0.2, 0.6, 0.8]}
  ],
  "timestamp": "2024-06-15T02:34:15Z"
}
```

### `pool_motion/bird_count`
Running total of clips where a bird was detected. Published retained so HA always has the current value.

## Auto-discovered Entities

The add-on registers the following entities in HA automatically via MQTT Discovery — no manual setup required:

| Entity | Type | Description |
|---|---|---|
| Pool Bird Detected | Binary sensor | `ON` when bird detected in latest clip |
| Pool Bird Detection Confidence | Sensor | Confidence score (0–1) of latest detection |
| Pool Bird Detection Last File | Sensor | Filename of last processed clip |
| Pool Bird Total Detections | Sensor | Cumulative count of bird detections (persisted across restarts) |

## Example HA Automation (trigger on bird detection)

```yaml
alias: Alert on pool bird detection
trigger:
  - platform: state
    entity_id: binary_sensor.pool_bird_detected
    to: "on"
action:
  - service: notify.mobile_app_your_phone
    data:
      title: "Bird at the pool!"
      message: >
        {{ states('sensor.pool_bird_detection_last_file') }} —
        confidence {{ (states('sensor.pool_bird_detection_confidence') | float * 100) | round }}%
```

## Notes

- The TFLite model (EfficientDet-Lite2, 448×448) is downloaded on first start and cached in `/data/model/`
- The model covers 90 COCO classes; ducks are detected as `bird`
- Each clip is sampled at 25%, 50%, and 75% of its duration; the highest-confidence detection across all frames is published
- Processing takes ~3–6 seconds per clip on a Raspberry Pi 4
- Bird count is persisted to `/data/bird_count.json` and survives add-on restarts
