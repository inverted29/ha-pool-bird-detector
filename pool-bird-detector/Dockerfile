ARG BUILD_FROM=ghcr.io/home-assistant/amd64-base-debian:bookworm
FROM $BUILD_FROM

# Install system deps: ffmpeg for frame extraction, inotify-tools for file watching
RUN apt-get update && apt-get install -y --no-install-recommends \
    ffmpeg \
    inotify-tools \
    python3 \
    python3-pip \
    python3-numpy \
    wget \
    ca-certificates \
    && rm -rf /var/lib/apt/lists/*

# Install Python packages
RUN pip3 install --no-cache-dir --break-system-packages \
    tflite-runtime \
    paho-mqtt

# Copy add-on files
COPY rootfs /

RUN chmod +x /usr/bin/detect.py

CMD ["/bin/bash", "/run.sh"]
