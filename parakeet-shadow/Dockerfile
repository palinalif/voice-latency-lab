FROM nvidia/cuda:12.8.1-cudnn-runtime-ubuntu22.04

RUN apt-get update && \
    DEBIAN_FRONTEND=noninteractive apt-get install -y --no-install-recommends \
        python3 \
        python3-pip \
    && rm -rf /var/lib/apt/lists/*

# onnxruntime-gpu is pinned to a CUDA 12 build to match this base image.
# onnx-asr's own [gpu] extra would grab the newest ORT, which targets CUDA 13.
RUN python3 -m pip install --no-cache-dir \
    "onnxruntime-gpu==1.23.2" \
    "onnx-asr[hub]" \
    fastapi \
    uvicorn \
    numpy

WORKDIR /app
COPY server.py /app/server.py

CMD ["python3", "/app/server.py"]
