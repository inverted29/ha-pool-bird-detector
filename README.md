# Pool Bird Detector — Home Assistant Add-on

Watches `/media/pool_motion/` for new `.mp4` clips (written by a Blink camera automation), extracts a frame with ffmpeg, runs MobileNet SSD COCO inference via TFLite, and publishes bird/duck detection results to your existing Mosquitto MQTT broker.

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
| `frame_offset_seconds` | `3` | Seconds into the clip to extract the frame |

## MQTT Payload

```json
{
  "file": "clip_20240615_023412.mp4",
  "bird_detected": true,
  "confidence": 0.82,
  "detections": [
    {"label": "bird", "confidence": 0.82, "box": [0.1, 0.2, 0.6, 0.8]}
  ],
  "timestamp": "2024-06-15T02:34:15Z"
}
```

## Example HA Automation (trigger on bird detection)

```yaml
alias: Alert on pool bird detection
trigger:
  - platform: mqtt
    topic: pool_motion/classification
condition:
  - condition: template
    value_template: "{{ trigger.payload_json.bird_detected }}"
action:
  - service: notify.mobile_app_your_phone
    data:
      title: "Bird at the pool!"
      message: >
        {{ trigger.payload_json.file }} —
        confidence {{ (trigger.payload_json.confidence * 100) | round }}%
```

## Notes

- The TFLite model (`coco_ssd_mobilenet_v1_1.0_quant`) is downloaded on first start and cached in `/data/model/`
- The model knows 90 COCO classes; birds are detected as `bird` (class 16)
- Duck is not a COCO class — it will be detected as `bird` which is correct for this use case
- Processing takes ~1–3 seconds per clip on a Raspberry Pi 4
