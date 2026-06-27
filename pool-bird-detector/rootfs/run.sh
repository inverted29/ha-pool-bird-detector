#!/usr/bin/env bash
set -e

CONFIG_PATH=/data/options.json

MQTT_HOST=$(jq -r '.mqtt_host' "$CONFIG_PATH")
MQTT_PORT=$(jq -r '.mqtt_port' "$CONFIG_PATH")
MQTT_USER=$(jq -r '.mqtt_user' "$CONFIG_PATH")
MQTT_PASSWORD=$(jq -r '.mqtt_password' "$CONFIG_PATH")
MQTT_TOPIC=$(jq -r '.mqtt_topic' "$CONFIG_PATH")
WATCH_DIR=$(jq -r '.watch_dir' "$CONFIG_PATH")
CONFIDENCE=$(jq -r '.confidence_threshold' "$CONFIG_PATH")
FRAME_OFFSET=$(jq -r '.frame_offset_seconds' "$CONFIG_PATH")

MODEL_DIR=/data/model
MODEL_PATH="$MODEL_DIR/detect.tflite"
LABELS_PATH="$MODEL_DIR/labelmap.txt"

if [ ! -f "$MODEL_PATH" ]; then
    echo "[bird-detector] Downloading EfficientDet-Lite2 model..."
    mkdir -p "$MODEL_DIR"
    wget -q -O "$MODEL_PATH" \
        "https://storage.googleapis.com/download.tensorflow.org/models/tflite/task_library/object_detection/android/lite-model_efficientdet_lite2_detection_metadata_1.tflite"
    if [ ! -f "$LABELS_PATH" ]; then
        cat > "$LABELS_PATH" << 'LABELS'
???
person
bicycle
car
motorcycle
airplane
bus
train
truck
boat
traffic light
fire hydrant
???
stop sign
parking meter
bench
bird
cat
dog
horse
sheep
cow
elephant
bear
zebra
giraffe
???
backpack
umbrella
???
???
handbag
tie
suitcase
frisbee
skis
snowboard
sports ball
kite
baseball bat
baseball glove
skateboard
surfboard
tennis racket
bottle
???
wine glass
cup
fork
knife
spoon
bowl
banana
apple
sandwich
orange
broccoli
carrot
hot dog
pizza
donut
cake
chair
couch
potted plant
bed
???
dining table
???
???
toilet
???
tv
laptop
mouse
remote
keyboard
cell phone
microwave
oven
toaster
sink
refrigerator
???
book
clock
vase
scissors
teddy bear
hair drier
toothbrush
LABELS
    fi
    echo "[bird-detector] Model ready."
fi

mkdir -p "$WATCH_DIR"

echo "[bird-detector] Watching $WATCH_DIR for new .mp4 files..."

exec python3 /usr/bin/detect.py \
    --watch-dir "$WATCH_DIR" \
    --model "$MODEL_PATH" \
    --labels "$LABELS_PATH" \
    --mqtt-host "$MQTT_HOST" \
    --mqtt-port "$MQTT_PORT" \
    --mqtt-user "$MQTT_USER" \
    --mqtt-password "$MQTT_PASSWORD" \
    --mqtt-topic "$MQTT_TOPIC" \
    --confidence "$CONFIDENCE" \
    --frame-offset "$FRAME_OFFSET"
