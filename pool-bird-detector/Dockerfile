ARG BUILD_FROM=ghcr.io/home-assistant/amd64-base-debian:bookworm
FROM $BUILD_FROM

RUN apt-get update && apt-get install -y --no-install-recommends \
    ffmpeg \
    inotify-tools \
    python3 \
    python3-pip \
    python3-numpy \
    wget \
    unzip \
    ca-certificates \
    && rm -rf /var/lib/apt/lists/*

RUN pip3 install --no-cache-dir --break-system-packages \
    tflite-runtime \
    paho-mqtt

COPY rootfs /

RUN chmod a+x \
    /usr/bin/detect.py \
    /etc/services.d/pool-bird-detector/run
