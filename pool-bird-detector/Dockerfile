ARG BUILD_FROM=ghcr.io/home-assistant/amd64-base:3.19
FROM $BUILD_FROM

# Install system deps: ffmpeg for frame extraction, inotify-tools for file watching
RUN apk add --no-cache \
    ffmpeg \
    inotify-tools \
    python3 \
    py3-pip \
    py3-numpy \
    wget \
    ca-certificates

# Install Python packages
RUN pip3 install --no-cache-dir --break-system-packages \
    tflite-runtime \
    paho-mqtt

# Copy add-on files
COPY rootfs /

RUN chmod +x /usr/bin/detect.py

CMD ["/bin/bash", "/run.sh"]
