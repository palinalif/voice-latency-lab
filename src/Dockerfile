# Voice Latency Lab
#
# Self-contained build: web stack + NOMO target-speaker VAD.
#
# The base is CUDA-flavoured because the historical image was (it also ships
# libgomp/libsndfile). NOMO itself is kept strictly CPU-side, so if you want a
# much smaller image, swap the base for python:3.11-slim and keep the same apt
# packages and pip layers.
FROM nvidia/cuda:12.8.1-cudnn-runtime-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive \
    PIP_NO_CACHE_DIR=1 \
    PYTHONUNBUFFERED=1

RUN apt-get update && \
    apt-get install -y --no-install-recommends \
        python3 \
        python3-venv \
        python3-dev \
        build-essential \
        ca-certificates \
        libgomp1 \
        libsndfile1 \
    && rm -rf /var/lib/apt/lists/*

RUN python3 -m venv /opt/venv
ENV PATH=/opt/venv/bin:$PATH

WORKDIR /app

# Web stack first: changes rarely, so this layer caches well.
COPY requirements.txt .
RUN pip install --upgrade pip setuptools wheel && \
    pip install -r requirements.txt

# NOMO target-speaker VAD. CPU-only torch on purpose: the VAD runs on CPU and
# this keeps several GB of CUDA wheels out of the image.
RUN pip install \
        --index-url https://download.pytorch.org/whl/cpu \
        'torch>=2.0,<3.0' \
        'torchaudio>=2.0,<3.0' && \
    pip install \
        'numpy>=1.24,<3.0' \
        'soundfile>=0.12' \
        'librosa>=0.10' \
        'modelscope[audio]>=1.10'

COPY server.py .
COPY static ./static

EXPOSE 8090

CMD ["uvicorn", "server:app", "--host", "0.0.0.0", "--port", "8090"]
