import io
import os
import time
import wave

import numpy as np
import onnxruntime as ort
import onnx_asr

from fastapi import FastAPI, Request
import uvicorn


PORT = int(os.getenv("PARAKEET_PORT", "3048"))
THREADS = int(os.getenv("PARAKEET_THREADS", "8"))

# "cuda" (CUDAExecutionProvider, CPU fallback) or "cpu".
DEVICE = os.getenv("PARAKEET_DEVICE", "cuda").strip().lower()

# Local model directory. Empty -> download the official model from HF.
# The fp16 dir must contain encoder-model.onnx (+ .data),
# decoder_joint-model.onnx, nemo128.onnx, config.json, vocab.txt.
MODEL_DIR = os.getenv("PARAKEET_MODEL_DIR", "").strip()

# None | int8 | ... ; fp16 files are named without a suffix.
_quant = os.getenv("PARAKEET_QUANTIZATION", "none").strip().lower()
QUANTIZATION = None if _quant in ("", "none", "fp16") else _quant


opts = ort.SessionOptions()
opts.intra_op_num_threads = THREADS
opts.inter_op_num_threads = 1
opts.graph_optimization_level = (
    ort.GraphOptimizationLevel.ORT_ENABLE_ALL
)
# Frugal memory behaviour: no pattern arena, sequential execution.
opts.enable_mem_pattern = False
opts.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL


def _providers(device):
    if device == "cuda":
        # Prefer CUDA, fall back to CPU for any unsupported node.
        return [
            (
                "CUDAExecutionProvider",
                {
                    "device_id": 0,
                    # Grow the arena exactly as requested instead of doubling it.
                    # GPU 1 has very little headroom (~1.7 GB free).
                    "arena_extend_strategy": "kSameAsRequested",
                    # HEURISTIC avoids the large cuDNN workspace of EXHAUSTIVE.
                    "cudnn_conv_algo_search": "HEURISTIC",
                    "do_copy_in_default_stream": True,
                },
            ),
            "CPUExecutionProvider",
        ]
    return ["CPUExecutionProvider"]


def _load(device):
    kwargs = {
        "quantization": QUANTIZATION,
        "providers": _providers(device),
        "sess_options": opts,
    }
    if MODEL_DIR:
        kwargs["path"] = MODEL_DIR
    return onnx_asr.load_model(
        "nemo-parakeet-tdt-0.6b-v3",
        **kwargs,
    )


print(
    f"Loading Parakeet TDT 0.6B v3 "
    f"({_quant or 'none'}) on {DEVICE}"
    + (f" from {MODEL_DIR}" if MODEL_DIR else "")
    + "...",
    flush=True,
)

t0 = time.perf_counter()

ACTIVE_DEVICE = DEVICE
try:
    model = _load(DEVICE)
except Exception as exc:
    if DEVICE != "cuda":
        raise
    print(f"CUDA load failed ({exc}); falling back to CPU", flush=True)
    ACTIVE_DEVICE = "cpu"
    model = _load("cpu")

ACTIVE_PROVIDER = (
    "CUDAExecutionProvider" if ACTIVE_DEVICE == "cuda" else "CPUExecutionProvider"
)

try:
    ACTIVE_PROVIDER = model._model.get_providers()[0]  # type: ignore[attr-defined]
except Exception:
    pass

print(
    f"Parakeet loaded in {time.perf_counter() - t0:.2f}s "
    f"(provider={ACTIVE_PROVIDER})",
    flush=True,
)

# Pay the one-time CUDA/PTX/cuDNN initialisation here rather than on the
# user's first turn. The first GPU inference took ~36 s; warm ones are ~14 ms.
print("Warming up Parakeet...", flush=True)
t_warm = time.perf_counter()
for _secs in (1.0, 5.0):
    try:
        model.recognize(
            np.zeros(int(16000 * _secs), dtype=np.float32),
            sample_rate=16000,
        )
    except Exception as exc:  # warm-up must never be fatal
        print(f"warmup {_secs}s failed: {exc}", flush=True)
print(
    f"Parakeet warm-up complete in "
    f"{time.perf_counter() - t_warm:.2f}s",
    flush=True,
)

app = FastAPI()


@app.get("/health")
async def health():
    return {
        "ok": True,
        "model": "parakeet-tdt-0.6b-v3",
        "quantization": _quant or "none",
        "provider": ACTIVE_PROVIDER,
        "device": ACTIVE_DEVICE,
        "model_dir": MODEL_DIR or "(hf)",
        "threads": THREADS,
    }


def decode_audio(data: bytes):
    # Convenience for benchmarks: accept a normal WAV file too.
    if data[:4] == b"RIFF":
        with wave.open(io.BytesIO(data), "rb") as w:
            channels = w.getnchannels()
            sample_width = w.getsampwidth()
            sample_rate = w.getframerate()

            if sample_width != 2:
                raise ValueError(
                    f"Expected 16-bit WAV, got {sample_width * 8}-bit"
                )

            raw = w.readframes(w.getnframes())

        pcm = np.frombuffer(raw, dtype="<i2")

        if channels > 1:
            pcm = pcm.reshape(-1, channels).mean(axis=1)

        audio = pcm.astype(np.float32) / 32768.0

        return audio, sample_rate

    # Live voice-lab format: raw mono PCM s16le @ 16 kHz.
    pcm = np.frombuffer(data, dtype="<i2")
    audio = pcm.astype(np.float32) / 32768.0

    return audio, 16000


@app.post("/transcribe")
async def transcribe(request: Request):
    data = await request.body()

    if not data:
        return {
            "text": "",
            "latency_ms": 0.0,
            "audio_sec": 0.0,
            "rtf": 0.0,
        }

    audio, sample_rate = decode_audio(data)

    audio_sec = len(audio) / sample_rate

    started = time.perf_counter()

    text = model.recognize(
        audio,
        sample_rate=sample_rate,
    )

    elapsed = time.perf_counter() - started

    return {
        "text": str(text).strip(),
        "latency_ms": round(elapsed * 1000.0, 1),
        "audio_sec": round(audio_sec, 3),
        "rtf": round(
            elapsed / audio_sec,
            4,
        ) if audio_sec else 0.0,
    }


if __name__ == "__main__":
    uvicorn.run(
        app,
        host="0.0.0.0",
        port=PORT,
        log_level="warning",
    )
