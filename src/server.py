import asyncio
import base64
import io
import json
import os
import re
import sys
import time
import wave
from collections import deque

import aiohttp
import numpy as np
import webrtcvad

NOMO_REPO = os.getenv("NOMO_REPO_PATH", "/nomo-pvad")
if NOMO_REPO not in sys.path:
    sys.path.insert(0, NOMO_REPO)

from nomo_pvad import NomoPVAD, NomoPVADSession
from fastapi import FastAPI, Request, WebSocket, WebSocketDisconnect
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

SAMPLE_RATE = 16000
FRAME_MS = 20
FRAME_SAMPLES = SAMPLE_RATE * FRAME_MS // 1000
FRAME_BYTES = FRAME_SAMPLES * 2

VAD_MODE = 3
VAD_HANGOVER_FRAMES = 25      # 500 ms correctness-first test
MIN_SPEECH_FRAMES = 8         # 160 ms; reject clicks/noise

NOMO_CKPT = os.getenv(
    "NOMO_CKPT",
    os.path.join(NOMO_REPO, "weights/nomo_pvad.pt"),
)
NOMO_PROFILE_PATH = os.getenv(
    "NOMO_PROFILE_PATH",
    "/profiles/pali.npy",
)
NOMO_MIN_ENROLL_SEC = 8.0
NOMO_ENROLL_SEC = 12.0
NOMO_ENROLL_FRAMES = int(NOMO_ENROLL_SEC * 1000 / FRAME_MS)

NOMO_THRESHOLD = 0.70
NOMO_ENTER_CHUNKS = 2
NOMO_EXIT_CHUNKS = 2
NOMO_RESUME_CHUNKS = 2
NOMO_CHUNK_FRAMES = 160 // FRAME_MS       # 8 x 20 ms
NOMO_CHUNK_SEC = NOMO_CHUNK_FRAMES * FRAME_MS / 1000.0
NOMO_END_GRACE_SEC = 0.00

PARTIAL_INTERVAL_SEC = 0.45
MIN_PARTIAL_AUDIO_SEC = 0.35
ENABLE_PARTIALS = False

QWEN_URL = os.getenv(
    "QWEN_URL",
    "http://127.0.0.1:18020/v1/chat/completions",
)
QWEN_MODEL = os.getenv("QWEN_MODEL", "qwen3.8-27b")

VOICE_LLM_BACKEND_FILE = os.getenv(
    "VOICE_LLM_BACKEND_FILE",
    "/profiles/voice_llm_backend.txt",
)

_voice_backend_env = os.getenv(
    "VOICE_LLM_BACKEND",
    "",
).strip().lower()

if _voice_backend_env:
    VOICE_LLM_BACKEND = _voice_backend_env
elif os.path.isfile(VOICE_LLM_BACKEND_FILE):
    VOICE_LLM_BACKEND = open(
        VOICE_LLM_BACKEND_FILE,
    ).read().strip().lower()
else:
    VOICE_LLM_BACKEND = "qwen"

if VOICE_LLM_BACKEND not in {"qwen", "gemma4"}:
    raise RuntimeError(
        "VOICE_LLM_BACKEND must be 'qwen' or 'gemma4'"
    )

GEMMA4_URL = os.getenv(
    "GEMMA4_URL",
    "http://127.0.0.1:18021/v1/chat/completions",
)
GEMMA4_MODEL = os.getenv(
    "GEMMA4_MODEL",
    "gemma4-12b",
)
GEMMA4_MAX_FRAMES = int(
    os.getenv("GEMMA4_MAX_FRAMES", "1")
)
GEMMA4_FRAME_WINDOW_SEC = float(
    os.getenv("GEMMA4_FRAME_WINDOW_SEC", "1.5")
)

VOXCPM_URL = os.getenv(
    "VOXCPM_URL",
    "http://127.0.0.1:3042/generate",
)
VOXCPM_STABLE_URL = os.getenv(
    "VOXCPM_STABLE_URL",
    "http://127.0.0.1:3042/generate-stable",
)

# On by default: shorter replies + the stable TTS endpoint.
STABLE_DEMO_MODE = os.getenv(
    "STABLE_DEMO_MODE",
    "1",
).strip().lower() not in {"0", "false", "no", "off"}
VOXCPM_SAMPLE_RATE = int(os.getenv("VOXCPM_SAMPLE_RATE", "48000"))

# Optional shadow integrations; each fails soft if unreachable.
EMOTION_URL = os.getenv(
    "EMOTION_URL",
    "http://127.0.0.1:3043/analyze",
)
SENSEVOICE_URL = os.getenv(
    "SENSEVOICE_URL",
    "http://127.0.0.1:3047/analyze",
)
KEV_URL = os.getenv(
    "KEV_URL",
    "http://127.0.0.1:3046/v1/systemone",
)

# Local Parakeet STT sidecar (parakeet-tdt-0.6b-v3, int8 on CPU).
# Replaces faster-whisper for both the live transcript and turn history.
# Contract: POST raw 16 kHz mono s16le to /transcribe -> {"text": ...}.
PARAKEET_URL = os.getenv(
    "PARAKEET_URL",
    "http://127.0.0.1:3048/transcribe",
)
PARAKEET_MODEL = os.getenv(
    "PARAKEET_MODEL",
    "parakeet-tdt-0.6b-v3",
)

# Per-persona TTS delivery descriptions, filled at startup by
# load_builtin_personas(). See BUILTIN_PERSONA_FILE.
VOXCPM_CONTROLS: dict[str, str] = {}

# Only needed for VOICE_LLM_BACKEND=qwen. Self-hosted backends (gemma4) never
# read it, so a missing file must not take the whole server down at import.
QWEN_API_KEY_FILE = os.getenv(
    "QWEN_API_KEY_FILE",
    "/secrets/qwen_api_key.txt",
)
QWEN_KEY = ""

if os.path.isfile(QWEN_API_KEY_FILE):
    QWEN_KEY = open(QWEN_API_KEY_FILE).read().strip()


# Built-in personas: id -> system prompt, filled at startup by
# load_builtin_personas(). File order is preserved, so the first entry in
# the JSON file is also the default persona. Character work ships as data,
# not code: see personas.builtin.example.json.
PERSONAS: dict[str, str] = {}


# ---------------------------------------------------------------------------
# Parakeet STT (local sidecar)
# ---------------------------------------------------------------------------

class ParakeetEngine:
    """
    STT via the local Parakeet sidecar.

    Same async ``transcribe(pcm_bytes)`` interface the old Whisper engines
    had, but the model lives in the parakeet-shadow container instead of this
    process — the voice container loads no Whisper model at all.
    """

    async def transcribe(self, pcm_bytes: bytes) -> str:
        if not pcm_bytes:
            return ""

        async with http.post(
            PARAKEET_URL,
            data=pcm_bytes,
            headers={"Content-Type": "application/octet-stream"},
        ) as resp:
            if resp.status != 200:
                raise RuntimeError(
                    f"Parakeet HTTP {resp.status}: {await resp.text()}"
                )

            payload = await resp.json()

        return str(payload.get("text", "")).strip()


# ---------------------------------------------------------------------------
# Nomo target-speaker VAD
# ---------------------------------------------------------------------------

class NomoEngine:
    def __init__(self):
        print("Loading nomo-pvad on CPU...", flush=True)
        t0 = time.perf_counter()

        self.model = NomoPVAD(
            NOMO_CKPT,
            device="cpu",
        )

        self.embedding = None

        if os.path.isfile(NOMO_PROFILE_PATH):
            emb = np.load(NOMO_PROFILE_PATH, allow_pickle=False).astype(np.float32)
            emb = emb.reshape(-1)

            if emb.shape == (192,):
                self.embedding = emb
                print("Loaded voice profile: pali", flush=True)
            else:
                print(
                    f"Ignoring invalid pali profile shape: {emb.shape}",
                    flush=True,
                )

        print(
            f"nomo-pvad loaded in {(time.perf_counter() - t0):.2f}s "
            f"(profile={'ready' if self.embedding is not None else 'missing'})",
            flush=True,
        )

    async def enroll_pcm16(self, pcm_bytes: bytes):
        pcm = np.frombuffer(pcm_bytes, dtype="<i2")
        audio = pcm.astype(np.float32) / 32768.0

        if audio.size < int(SAMPLE_RATE * NOMO_MIN_ENROLL_SEC):
            raise ValueError("Enrollment audio is too short")

        emb = await asyncio.to_thread(
            self.model.extract_emb,
            audio,
        )

        emb = emb.detach().cpu().numpy().astype(np.float32).reshape(-1)

        if emb.shape != (192,):
            raise RuntimeError(f"Unexpected embedding shape: {emb.shape}")

        os.makedirs(os.path.dirname(NOMO_PROFILE_PATH), exist_ok=True)

        tmp = NOMO_PROFILE_PATH + ".tmp.npy"
        np.save(tmp, emb)
        os.replace(tmp, NOMO_PROFILE_PATH)

        self.embedding = emb
        return emb


# ---------------------------------------------------------------------------
# Imported personas (TomoriBot preset JSON + uploaded VoxCPM voice)
# ---------------------------------------------------------------------------

PERSONA_DIR = os.getenv("PERSONA_DIR", "/profiles/personas")

# Built-in personas live here as data, so character work stays out of the repo.
# Defaults to <parent of PERSONA_DIR>/personas.builtin.json, i.e.
# /profiles/personas.builtin.json - deliberately NOT inside PERSONA_DIR, which
# is scanned for imported personas.
BUILTIN_PERSONA_FILE = os.getenv(
    "BUILTIN_PERSONA_FILE",
    os.path.join(os.path.dirname(PERSONA_DIR), "personas.builtin.json"),
)

VOXCPM_VOICES_URL = os.getenv(
    "VOXCPM_VOICES_URL",
    "http://127.0.0.1:3042/v1/audio/voices",
)
PERSONA_SAMPLE_EXCHANGES = -1
PERSONA_VOICE_PREFIX = "lab-"

imported_personas: dict[str, dict] = {}


def persona_slug(text: str) -> str:
    text = re.sub(r"[^a-z0-9]+", "-", (text or "").strip().lower())
    return text.strip("-") or "persona"


def unique_persona_id(base: str) -> str:
    candidate = base
    n = 2

    while candidate in PERSONAS or candidate in imported_personas:
        candidate = f"{base}-{n}"
        n += 1

    return candidate


def _expand_bot(text, name: str) -> str:
    return str(text or "").replace("{bot}", name).replace("{persona}", name)


def build_persona_prompt(data: dict, sample_exchanges: int = PERSONA_SAMPLE_EXCHANGES) -> str:
    """Turn a TomoriBot preset export into a compact spoken-voice prompt.

    sample_exchanges: -1 = every example in the preset, 0 = none, N = first N.
    """
    name = (data.get("tomori_nickname") or "").strip() or "Assistant"

    parts = [_expand_bot(data.get("persona_prompt"), name).strip()]

    for attr in data.get("attribute_list") or []:
        attr = _expand_bot(attr, name).strip()

        if attr:
            parts.append(attr)

    terms = (data.get("naming_config") or {}).get("addressTerms") or {}
    terms = ", ".join(f'"{v}" for {k}' for k, v in terms.items() if v)

    if terms:
        parts.append(
            "When addressing the user casually you may use these terms: "
            f"{terms}."
        )

    examples = []

    for q, a in zip(
        data.get("sample_dialogues_in") or [],
        data.get("sample_dialogues_out") or [],
    ):
        q = _expand_bot(q, name).strip()
        a = _expand_bot(a, name).strip()

        if q and a:
            examples.append((q, a))

    if examples and sample_exchanges != 0:
        picked = (
            examples
            if sample_exchanges < 0
            else examples[:sample_exchanges]
        )

        parts.append(
            f"Example exchanges ({len(picked)} of {len(examples)}), for tone and "
            "wording only; never reuse them verbatim:\n"
            + "\n".join(
                f"User: {q}\n{name}: {a}"
                for q, a in picked
            )
        )

    parts.insert(0, f"You are {name}.")
    parts.append(
        f"Stay as {name}. Never sound corporate, therapeutic, or like a generic assistant."
    )

    return "\n\n".join(p for p in parts if p) + "\n"


def normalize_reference_wav(raw: bytes) -> tuple[bytes, float]:
    """Best-effort 16 kHz mono PCM16 WAV for a VoxCPM reference voice."""
    try:
        with wave.open(io.BytesIO(raw), "rb") as w:
            channels = w.getnchannels()
            width = w.getsampwidth()
            rate = w.getframerate()
            frames = w.readframes(w.getnframes())
    except Exception:
        # Not a WAV we can decode (mp3/ogg/webm); hand it to VoxCPM untouched.
        return raw, 0.0

    if width == 2:
        data = np.frombuffer(frames, dtype="<i2").astype(np.float32) / 32768.0
    elif width == 1:
        data = (np.frombuffer(frames, dtype=np.uint8).astype(np.float32) - 128.0) / 128.0
    elif width == 4:
        data = np.frombuffer(frames, dtype="<f4").astype(np.float32)
    else:
        return raw, 0.0

    if channels > 1 and len(data):
        keep = len(data) - (len(data) % channels)
        data = data[:keep].reshape(-1, channels).mean(axis=1)

    if rate not in (0, 16000) and len(data):
        n_out = max(int(round(len(data) * 16000 / rate)), 1)
        idx = np.linspace(0, len(data) - 1, num=n_out)
        lo = np.floor(idx).astype(np.int64)
        hi = np.minimum(lo + 1, len(data) - 1)
        frac = (idx - lo).astype(np.float32)
        data = data[lo] * (1.0 - frac) + data[hi] * frac

    duration = len(data) / 16000.0
    pcm = (np.clip(data, -1.0, 1.0) * 32767.0).astype("<i2").tobytes()

    out = io.BytesIO()

    with wave.open(out, "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(16000)
        w.writeframes(pcm)

    return out.getvalue(), duration


def load_builtin_personas() -> None:
    """Fill PERSONAS / VOXCPM_CONTROLS from BUILTIN_PERSONA_FILE.

    Accepts {"id": {"prompt": "...", "control": "..."}} and the shorter
    {"id": "prompt text"}. A missing file just means no built-in personas, which
    is a valid state - imported personas still work.
    """
    PERSONAS.clear()
    VOXCPM_CONTROLS.clear()

    if not os.path.isfile(BUILTIN_PERSONA_FILE):
        return

    try:
        with open(BUILTIN_PERSONA_FILE, "r", encoding="utf-8") as fh:
            raw = json.load(fh)

        for pid, rec in raw.items():
            if isinstance(rec, str):
                PERSONAS[pid] = rec
                continue

            prompt = rec.get("prompt") or ""

            if not prompt.strip():
                continue

            PERSONAS[pid] = prompt

            control = (rec.get("control") or "").strip()

            if control:
                VOXCPM_CONTROLS[pid] = control
    except Exception as exc:
        print(
            f"Builtin persona load failed ({BUILTIN_PERSONA_FILE}): {exc}",
            flush=True,
        )


def default_persona() -> str:
    # File order decides, so JSON order is meaningful: the first entry is the
    # persona a fresh session starts with.
    if PERSONAS:
        return next(iter(PERSONAS))

    if imported_personas:
        return next(iter(imported_personas))

    return ""


def load_personas() -> None:
    imported_personas.clear()

    if not os.path.isdir(PERSONA_DIR):
        return

    for fn in sorted(os.listdir(PERSONA_DIR)):
        if not fn.endswith(".json"):
            continue

        try:
            with open(os.path.join(PERSONA_DIR, fn), "r", encoding="utf-8") as fh:
                rec = json.load(fh)

            pid = rec.get("id") or fn[:-5]
            rec.setdefault("label", pid)
            rec.setdefault("voice_id", pid)
            imported_personas[pid] = rec
        except Exception as exc:
            print(f"Persona load failed ({fn}): {exc}", flush=True)


def persona_ids() -> set[str]:
    return set(PERSONAS) | set(imported_personas)


def persona_prompt_for(pid: str) -> str:
    rec = imported_personas.get(pid)

    if rec:
        return rec.get("prompt") or ""

    return PERSONAS.get(pid, "")


def voice_id_for(pid: str) -> str:
    rec = imported_personas.get(pid)

    if rec:
        return rec.get("voice_id") or pid

    if pid in PERSONAS:
        return pid

    return default_persona()


def persona_catalog() -> list[dict]:
    out = [
        {
            "id": pid,
            "label": pid.capitalize(),
            "builtin": True,
            "voice_id": pid,
            "prompt_chars": len(prompt),
        }
        for pid, prompt in PERSONAS.items()
    ]

    for pid, rec in sorted(imported_personas.items()):
        out.append(
            {
                "id": pid,
                "label": rec.get("label") or pid,
                "builtin": False,
                "voice_id": rec.get("voice_id"),
                "created_at": rec.get("created_at"),
                "prompt_chars": len(rec.get("prompt") or ""),
            }
        )

    return out


# ---------------------------------------------------------------------------
# Application
# ---------------------------------------------------------------------------

app = FastAPI()

@app.middleware("http")
async def disable_cache(request, call_next):
    response = await call_next(request)

    if request.url.path == "/" or request.url.path.startswith("/static/"):
        response.headers["Cache-Control"] = "no-store, no-cache, must-revalidate, max-age=0"
        response.headers["Pragma"] = "no-cache"
        response.headers["Expires"] = "0"

    return response

app.mount("/static", StaticFiles(directory="/app/static"), name="static")

stt_engine = None
history_stt_engine = None
nomo = None
http = None


@app.on_event("startup")
async def startup():
    global stt_engine, history_stt_engine, nomo, http

    load_builtin_personas()
    load_personas()

    print(
        f"Voice LLM backend: {VOICE_LLM_BACKEND}",
        flush=True,
    )
    print(
        f"Personas: {len(PERSONAS)} built-in, {len(imported_personas)} imported",
        flush=True,
    )

    if not PERSONAS and not imported_personas:
        print(
            f"WARNING: no personas available. Add one to {BUILTIN_PERSONA_FILE} "
            "(see personas.builtin.example.json) or import one from the page.",
            flush=True,
        )

    if VOICE_LLM_BACKEND == "qwen" and not QWEN_KEY:
        print(
            f"WARNING: VOICE_LLM_BACKEND=qwen but {QWEN_API_KEY_FILE} is "
            "missing or empty; LLM requests will fail.",
            flush=True,
        )

    nomo = NomoEngine()

    if VOICE_LLM_BACKEND == "qwen":
        stt_engine = ParakeetEngine()
    else:
        stt_engine = None
        print(
            "Live STT disabled; Gemma 4 consumes raw audio directly",
            flush=True,
        )

    if VOICE_LLM_BACKEND == "gemma4":
        history_stt_engine = ParakeetEngine()
    else:
        history_stt_engine = None

    http = aiohttp.ClientSession(
        timeout=aiohttp.ClientTimeout(total=None)
    )


@app.on_event("shutdown")
async def shutdown():
    if http:
        await http.close()


@app.get("/")
async def root():
    return FileResponse("/app/static/index.html")


@app.get("/health")
async def health():
    return {
        "ok": True,
        "llm_backend": VOICE_LLM_BACKEND,
        "stt": (
            "parakeet"
            if VOICE_LLM_BACKEND == "qwen"
            else "gemma4-native-audio"
        ),
        "model": (
            PARAKEET_MODEL
            if VOICE_LLM_BACKEND == "qwen"
            else GEMMA4_MODEL
        ),
        "history_stt": (
            PARAKEET_MODEL
            if VOICE_LLM_BACKEND == "gemma4"
            else None
        ),
        "target_vad": "nomo-pvad",
        "voice_profile": "ready" if nomo and nomo.embedding is not None else "missing",
        "personas": len(persona_ids()),
    }


@app.get("/personas")
async def list_personas():
    return {"personas": persona_catalog()}


@app.post("/personas")
async def create_persona(request: Request):
    """Import a TomoriBot preset + voice sample as a new selectable persona.

    Multipart is unavailable in this image (no python-multipart), so the
    browser posts one JSON envelope with the preset object and the voice
    sample base64-encoded.
    """
    if http is None:
        return JSONResponse({"error": "Server is still starting."}, status_code=503)

    try:
        body = await request.json()
    except Exception:
        return JSONResponse({"error": "Expected a JSON body."}, status_code=400)

    preset = body.get("preset")

    if isinstance(preset, str):
        try:
            preset = json.loads(preset)
        except Exception:
            preset = None

    if not isinstance(preset, dict):
        return JSONResponse({"error": "Preset is not valid JSON."}, status_code=400)

    data = preset.get("data") if isinstance(preset.get("data"), dict) else preset

    if not (data.get("tomori_nickname") or data.get("persona_prompt")):
        return JSONResponse(
            {
                "error": "Not a TomoriBot persona preset "
                "(no tomori_nickname or persona_prompt).",
            },
            status_code=400,
        )

    try:
        raw_voice = base64.b64decode(body.get("voice_b64") or "")
    except Exception:
        raw_voice = b""

    if not raw_voice:
        return JSONResponse({"error": "Voice sample was empty."}, status_code=400)

    name = (
        (body.get("label") or "").strip()
        or (data.get("tomori_nickname") or "").strip()
        or "persona"
    )

    # Imported personas and their VoxCPM voices share one namespaced id, so an
    # import can never shadow an existing built-in persona.
    pid = unique_persona_id(
        f"{PERSONA_VOICE_PREFIX}{persona_slug(name)}"
    )

    requested_voice = persona_slug(body.get("voice_id") or "")
    voice_id = requested_voice or pid

    wav_bytes, seconds = normalize_reference_wav(raw_voice)

    form = aiohttp.FormData()
    form.add_field("name", voice_id)
    form.add_field("consent", "true")
    form.add_field(
        "ref_text",
        (body.get("sample_text") or "").strip()
        or "Reference voice sample.",
    )
    form.add_field(
        "audio_sample",
        io.BytesIO(wav_bytes),
        filename="reference.wav",
        content_type="audio/wav",
    )

    try:
        async with http.post(VOXCPM_VOICES_URL, data=form) as resp:
            detail = await resp.text()

            if resp.status >= 300:
                return JSONResponse(
                    {
                        "error": f"Voice registration failed ({resp.status}): "
                        f"{detail[:300]}",
                    },
                    status_code=502,
                )
    except Exception as exc:
        return JSONResponse(
            {"error": f"Could not reach VoxCPM voice API: {exc}"},
            status_code=502,
        )

    prompt_text = build_persona_prompt(data)

    rec = {
        "id": pid,
        "label": name,
        "voice_id": voice_id,
        "prompt": prompt_text,
        "source": "tomoribot",
        "created_at": time.time(),
        "sample_seconds": round(seconds, 2),
    }

    try:
        os.makedirs(PERSONA_DIR, exist_ok=True)

        with open(
            os.path.join(PERSONA_DIR, f"{pid}.json"),
            "w",
            encoding="utf-8",
        ) as fh:
            json.dump(rec, fh, indent=2, ensure_ascii=False)
    except Exception as exc:
        return JSONResponse(
            {"error": f"Could not save persona: {exc}"},
            status_code=500,
        )

    imported_personas[pid] = rec

    print(
        f"Imported persona '{name}' (id={pid}, voice={voice_id}, "
        f"prompt={len(prompt_text)} chars, "
        f"sample={rec['sample_seconds']}s)",
        flush=True,
    )

    return {
        "persona": {
            "id": pid,
            "label": name,
            "builtin": False,
            "voice_id": voice_id,
            "prompt_chars": len(prompt_text),
        }
    }


@app.delete("/personas/{pid}")
async def delete_persona(pid: str):
    rec = imported_personas.pop(pid, None)

    if not rec:
        return JSONResponse(
            {"error": "Unknown persona (built-ins cannot be deleted)."},
            status_code=404,
        )

    voice_id = rec.get("voice_id")

    if voice_id and http:
        try:
            async with http.delete(f"{VOXCPM_VOICES_URL}/{voice_id}") as resp:
                await resp.read()
        except Exception:
            pass

    try:
        os.remove(os.path.join(PERSONA_DIR, f"{pid}.json"))
    except Exception:
        pass

    return {"ok": True}


# ---------------------------------------------------------------------------
# Session state
# ---------------------------------------------------------------------------

class State:
    def __init__(self, ws: WebSocket):
        self.ws = ws
        self.send_lock = asyncio.Lock()

        self.persona = default_persona()

        self.vad = webrtcvad.Vad(VAD_MODE)

        # A little pre-roll stops aggressive VAD from clipping first consonants.
        self.pre_roll = deque(maxlen=25)  # 500 ms; covers nomo entry hysteresis

        self.audio_frames = []
        self.turn_active = False
        self.speech_frames = 0

        self.last_speech_at = None
        self.silence_frames = 0

        self.candidate_fired = False
        self.response_started = False

        self.serial = 0
        self.generation = None

        self.partial_task = None
        self.partial_epoch = 0
        self.last_partial_launch = 0.0
        self.latest_partial = ""

        # Speculative final STT pass.
        #
        # We launch this after NOMO's first negative 160 ms chunk,
        # while NOMO is still waiting for the second negative chunk
        # required to actually end the turn.
        self.speculative_whisper_task = None
        self.speculative_whisper_valid = False
        self.speculative_whisper_snapshot_len = 0
        self.speculative_whisper_epoch = -1

        # Qwen may begin from the speculative final STT result
        # before NOMO's second negative chunk commits the turn.
        #
        # TTS remains gated until speculative_qwen_commit is set.
        self.speculative_qwen_task = None
        self.speculative_qwen_commit = None
        self.speculative_qwen_transcript = ""
        self.speculative_qwen_epoch = -1

        # Gemma 4 can start directly from the speculative raw-audio
        # snapshot, without waiting for STT.
        self.speculative_gemma_task = None
        self.speculative_gemma_commit = None
        self.speculative_gemma_snapshot_len = 0
        self.speculative_gemma_epoch = -1

        self.playback_active = False
        self.active_mouth_stop = None

        self.history = []

        # Gemma transcribes completed turns for text history only after
        # response/TTS work has been launched, keeping transcription out
        # of the mouth-stop -> first-audio critical path.
        # Keep strong references to all queued history transcription
        # jobs. history_lock serializes CPU STT so they remain ordered.
        self.gemma_history_task = None
        self.history_tasks = set()
        self.history_lock = asyncio.Lock()

        # Per-connection target-speaker VAD session.
        self.nomo_session = None
        self.nomo_chunk_frames = []

        self.target_speaking = False
        self.target_positive_chunks = 0
        self.target_negative_chunks = 0
        self.target_end_pending_at = None
        self.target_silence_started_at = None

        self.refresh_nomo_session()

        # Temporary one-profile enrollment path.
        self.enrollment_active = False
        self.enrollment_frames = []

    def refresh_nomo_session(self):
        self.nomo_session = None
        self.nomo_chunk_frames = []

        if not nomo or nomo.embedding is None:
            return

        sess = NomoPVADSession(nomo.model)
        sess.set_enrollment_emb(nomo.embedding)

        self.nomo_session = sess

        self.target_speaking = False
        self.target_positive_chunks = 0
        self.target_negative_chunks = 0
        self.target_end_pending_at = None
        self.target_silence_started_at = None

    async def json(self, obj):
        async with self.send_lock:
            await self.ws.send_text(json.dumps(obj))

    async def audio(self, data: bytes):
        async with self.send_lock:
            await self.ws.send_bytes(data)

    async def clear_audio(self):
        await self.json({"type": "clear_audio"})

    async def cancel_generation(self):
        self.serial += 1

        if self.generation and not self.generation.done():
            self.generation.cancel()

        self.generation = None
        self.response_started = False

        # A generation can be cancelled after VoxCPM has already gated
        # the mic. Always recover the gate here or the call can become
        # permanently deaf.
        was_playback_active = self.playback_active
        self.playback_active = False

        await self.clear_audio()

        if was_playback_active:
            try:
                await self.json({"type": "tts_end"})
            except Exception:
                pass

            try:
                await self.json({
                    "type": "log",
                    "message": (
                        "SERVER MIC GATE FAILSAFE OFF — "
                        "generation cancelled"
                    ),
                })
            except Exception:
                pass

    def reset_turn_audio(self):
        self.pre_roll.clear()
        self.audio_frames = []
        self.turn_active = False
        self.speech_frames = 0

        self.last_speech_at = None
        self.silence_frames = 0

        self.candidate_fired = False
        self.response_started = False

        self.latest_partial = ""
        self.partial_epoch += 1

        self.speculative_whisper_valid = False
        self.speculative_whisper_snapshot_len = 0
        self.speculative_whisper_epoch = -1

        if (
            self.speculative_whisper_task is not None
            and self.speculative_whisper_task.done()
        ):
            self.speculative_whisper_task = None

        self.nomo_chunk_frames = []
        self.target_speaking = False
        self.target_positive_chunks = 0
        self.target_negative_chunks = 0
        self.target_end_pending_at = None
        self.target_silence_started_at = None

        if self.nomo_session is not None:
            self.nomo_session.reset()


async def begin_enrollment(state: State):
    if state.enrollment_active:
        return

    # Enrollment owns the mic until its 12 seconds are captured.
    await state.cancel_generation()

    state.playback_active = False
    state.reset_turn_audio()

    state.enrollment_frames = []
    state.enrollment_active = True

    await state.json({
        "type": "state",
        "value": "ENROLLING",
    })

    await state.json({
        "type": "enrollment_status",
        "status": "recording",
        "duration_sec": NOMO_ENROLL_SEC,
    })

    await state.json({
        "type": "log",
        "message": f"Voice enrollment started ({NOMO_ENROLL_SEC:.0f}s)",
    })


async def process_enrollment_frame(
    state: State,
    data: bytes,
):
    if len(data) != FRAME_BYTES:
        return

    state.enrollment_frames.append(data)

    if len(state.enrollment_frames) < NOMO_ENROLL_FRAMES:
        return

    # Use exactly the requested enrollment window.
    pcm_bytes = b"".join(
        state.enrollment_frames[:NOMO_ENROLL_FRAMES]
    )

    # Raw enrollment PCM exists only in memory.
    state.enrollment_frames = []

    await state.json({
        "type": "enrollment_status",
        "status": "processing",
    })

    try:
        await nomo.enroll_pcm16(pcm_bytes)
        state.refresh_nomo_session()

    except Exception as e:
        state.enrollment_active = False
        state.reset_turn_audio()

        await state.json({
            "type": "enrollment_status",
            "status": "error",
            "message": str(e),
        })

        await state.json({
            "type": "state",
            "value": "LISTENING",
        })

        return

    state.enrollment_active = False
    state.reset_turn_audio()

    await state.json({
        "type": "enrollment_status",
        "status": "ready",
    })

    await state.json({
        "type": "log",
        "message": "Voice profile saved: pali (192-d)",
    })

    await state.json({
        "type": "state",
        "value": "LISTENING",
    })


async def timing(state: State, event: str, t: float, mouth_stop: float):
    await state.json({
        "type": "timing",
        "event": event,
        "ms": round((t - mouth_stop) * 1000.0, 1),
    })


# ---------------------------------------------------------------------------
# STT rolling partials
# ---------------------------------------------------------------------------

async def run_partial(
    state: State,
    pcm_bytes: bytes,
    epoch: int,
):
    try:
        text = await stt_engine.transcribe(pcm_bytes)

        if epoch != state.partial_epoch:
            return

        if state.candidate_fired:
            return

        if not text:
            return

        if text == state.latest_partial:
            return

        state.latest_partial = text

        await state.json({
            "type": "transcript",
            "text": text,
            "partial": True,
        })

    except asyncio.CancelledError:
        raise

    except Exception as e:
        await state.json({
            "type": "log",
            "message": f"STT partial error: {e}",
        })


def maybe_launch_partial(state: State, now: float):
    if VOICE_LLM_BACKEND != "qwen":
        return

    if not ENABLE_PARTIALS:
        return

    if not state.turn_active:
        return

    if state.candidate_fired:
        return

    duration = (
        len(state.audio_frames) * FRAME_MS / 1000.0
    )

    if duration < MIN_PARTIAL_AUDIO_SEC:
        return

    if (
        now - state.last_partial_launch
        < PARTIAL_INTERVAL_SEC
    ):
        return

    if (
        state.partial_task
        and not state.partial_task.done()
    ):
        return

    state.last_partial_launch = now

    snapshot = b"".join(state.audio_frames)
    epoch = state.partial_epoch

    state.partial_task = asyncio.create_task(
        run_partial(state, snapshot, epoch)
    )


# ---------------------------------------------------------------------------
# Qwen
# ---------------------------------------------------------------------------

def first_speakable_sentence(text: str):
    # Find the earliest natural boundary safe to send to TTS.

    abbreviations = {
        "mr", "mrs", "ms", "dr", "prof",
        "sr", "jr", "st", "vs", "etc",
    }

    boundary_re = re.compile(
        r'(?P<sentence>[.!?]+(?:["\'”’)\]]*)?(?:\s|$))'
        r'|(?P<clause>[,;:](?:["\'”’)\]]*)?(?:\s|$)|[—–](?:\s|$))'
    )

    for m in boundary_re.finditer(text):
        end = m.end()
        candidate = text[:end].strip()

        # Performance tags do not count as spoken words.
        visible = re.sub(
            r"\[[^\]]+\]",
            "",
            candidate,
        )

        words = re.findall(
            r"\b[\w'-]+\b",
            visible,
        )

        if not words:
            continue

        if m.group("sentence") is not None:
            # Completed multi-word sentence.
            if len(words) >= 2:
                return candidate, end

            # Conversational one-word sentence:
            # "Sure.", "Cool.", "Bet.", "Exactly.", etc.
            word = words[0].lower().strip("-'")

            if word not in abbreviations:
                return candidate, end

        else:
            # Clause boundary after enough real speech.
            if len(words) >= 4:
                return candidate, end

    return None


async def stream_fish_text(
    state: State,
    text: str,
    mouth_stop: float,
    serial: int,
    tts_state: dict,
):
    """
    Legacy function name; the active backend is VoxCPM2.

    Keeping this name temporarily avoids touching the existing first-phrase
    and continuation scheduling code while we compare VoxCPM against Fish.
    """
    text = text.strip()

    if not text:
        return

    if serial != state.serial:
        return

    # Gate the mic once for the entire multi-request spoken response.
    if not tts_state["gate_on"]:
        tts_state["gate_on"] = True

        state.playback_active = True
        state.reset_turn_audio()

        await state.json({
            "type": "log",
            "message": "SERVER MIC GATE ON",
        })

    tts_start = time.perf_counter()

    # Keep the existing diagnostics event names for this A/B test so
    # the frontend timing table continues to work unchanged.
    if not tts_state["request_seen"]:
        tts_state["request_seen"] = True

        await timing(
            state,
            "s2_request",
            tts_start,
            mouth_stop,
        )

        await state.json({
            "type": "log",
            "message": "VoxCPM request",
        })
    else:
        await state.json({
            "type": "log",
            "message": "VoxCPM continuation request",
        })

    payload = {
        "voice": voice_id_for(state.persona),
        "text": text,
        "cfg_value": 1.5,
        "inference_timesteps": 20,
    }

    control = VOXCPM_CONTROLS.get(state.persona)

    if control:
        payload["control"] = control

    tts_url = (
        VOXCPM_STABLE_URL
        if STABLE_DEMO_MODE
        else VOXCPM_URL
    )

    async with http.post(
        tts_url,
        json=payload,
    ) as resp:

        if resp.status != 200:
            raise RuntimeError(
                f"VoxCPM HTTP {resp.status}: "
                f"{await resp.text()}"
            )

        sample_rate = resp.headers.get(
            "X-Audio-Sample-Rate"
        )

        if (
            sample_rate is not None
            and int(sample_rate) != VOXCPM_SAMPLE_RATE
        ):
            raise RuntimeError(
                "Unexpected VoxCPM sample rate: "
                f"{sample_rate}"
            )

        async for chunk in resp.content.iter_chunked(16384):
            if serial != state.serial:
                return

            if not chunk:
                continue

            if not tts_state["first_pcm_seen"]:
                tts_state["first_pcm_seen"] = True

                now = time.perf_counter()

                await timing(
                    state,
                    "first_pcm",
                    now,
                    mouth_stop,
                )

                await state.json({
                    "type": "state",
                    "value": "SPEAKING",
                })

            await state.audio(chunk)


def pcm16_wav_b64(pcm_bytes: bytes) -> str:
    """
    Wrap the existing 16 kHz mono s16le microphone snapshot in a WAV
    container for vLLM's OpenAI-compatible input_audio field.
    """
    buf = io.BytesIO()

    with wave.open(buf, "wb") as wav:
        wav.setnchannels(1)
        wav.setsampwidth(2)
        wav.setframerate(SAMPLE_RATE)
        wav.writeframes(pcm_bytes)

    return base64.b64encode(buf.getvalue()).decode("ascii")


def select_gemma_vision_frames() -> list[str]:
    """
    Pick a few recent browser-provided JPEGs from the existing camera ring
    buffer. Keep selection deterministic and spread across the recent window.
    """
    if GEMMA4_MAX_FRAMES <= 0:
        return []

    snapshot = list(vision_frames)

    if not snapshot:
        return []

    newest_ms = snapshot[-1][0]
    window_ms = max(0.0, GEMMA4_FRAME_WINDOW_SEC) * 1000.0

    recent = [
        item
        for item in snapshot
        if newest_ms - item[0] <= window_ms
    ]

    if not recent:
        recent = [snapshot[-1]]

    if len(recent) <= GEMMA4_MAX_FRAMES:
        selected = recent
    elif GEMMA4_MAX_FRAMES == 1:
        selected = [recent[-1]]
    else:
        last = len(recent) - 1
        indices = [
            round(i * last / (GEMMA4_MAX_FRAMES - 1))
            for i in range(GEMMA4_MAX_FRAMES)
        ]
        selected = [recent[i] for i in indices]

    return [jpeg for _, jpeg in selected]


async def history_transcribe_audio(
    pcm_bytes: bytes,
) -> str:
    """
    Produce reliable text history with the Parakeet sidecar.

    Gemma still consumes the raw audio directly for the live response;
    this runs afterward and never delays first-token/TTS latency.
    """
    if history_stt_engine is None:
        raise RuntimeError("History STT is not loaded")

    return (
        await history_stt_engine.transcribe(pcm_bytes)
    ).strip()


async def transcribe_turn_history(
    state: State,
    pcm_bytes: bytes,
):
    """
    Transcribe a completed user turn and surface it immediately.

    Started as soon as the turn commits, in parallel with the response.
    Parakeet on the GPU sidecar is ~10-20 ms, so the user transcript no
    longer has to wait for the entire spoken reply to finish streaming.

    Returns the transcript (or None), so the caller can append it to
    text history once the assistant text is known.
    """
    async with state.history_lock:
        started = time.perf_counter()

        try:
            transcript = await history_transcribe_audio(pcm_bytes)
        except asyncio.CancelledError:
            raise
        except Exception as exc:
            print(f"[STT HISTORY] transcript failed: {exc}", flush=True)
            return None

        if not transcript:
            return None

        elapsed_ms = (time.perf_counter() - started) * 1000.0

        print(
            f'[STT HISTORY] {elapsed_ms:.1f}ms "{transcript}"',
            flush=True,
        )

        try:
            await state.json({
                "type": "transcript",
                "text": transcript,
                "partial": False,
            })
        except Exception:
            pass

        try:
            await state.json({
                "type": "log",
                "message": (
                    f"YOU TRANSCRIPT ({elapsed_ms:.1f}ms): "
                    f"{transcript}"
                ),
            })
        except Exception:
            pass

        return transcript


async def run_response(
    state: State,
    transcript: str,
    mouth_stop: float,
    serial: int | None,
    commit_event: asyncio.Event | None = None,
    speculative: bool = False,
    audio_snapshot: bytes | None = None,
    vision_frame_b64: list[str] | None = None,
):
    persona_prompt = persona_prompt_for(state.persona)

    system = f"""
{persona_prompt}

You are currently in a live voice call.

Answer naturally and conversationally.

Make your FIRST SENTENCE extremely short, complete, and immediately meaningful,
usually 2-5 spoken words. A quick reaction, joke, or punchline is ideal when natural.
Avoid a one-word first sentence when you intend to continue speaking; give the first
phrase just enough substance to flow naturally into the rest of the response.
After that first sentence, continue naturally if you have more to say.

Keep your actual personality, humor, attitude, flirting, sarcasm,
hesitation, or warmth as appropriate.

Do not use markdown.

You may occasionally use short natural-language performance tags in square
brackets when the way something is spoken materially matters.

Performance tags are semantic and do NOT have a fixed vocabulary. Describe the
delivery naturally, for example:
[whisper]
[deadpan]
[quietly smug]
[trying not to laugh]
[tired and annoyed]
[embarrassed but pretending not to be]
[soft and sincere]

Place a performance tag immediately BEFORE the sentence or phrase it applies to.
A tag may appear before the very first spoken words. If the delivery changes
later, add a new tag immediately before that part.

Examples:
[deadpan] Wow. Incredible detective work.
[trying not to laugh] No, no, keep explaining. This is getting better.
[whisper] Okay, don't freak out. [quietly amused] I may have broken it.

Use performance tags sparingly. Most normal speech should have no tag at all.
Usually zero to two style tags in a response is enough. Do not tag every sentence
just because tags are available.

For actual nonverbal actions, simple tags such as [sigh], [chuckle], [laughter],
[short pause], and [long pause] are appropriate.

Do not use key=value syntax, XML, markdown, or technical TTS instructions.

For this showcase recording, prioritize natural, consistent speech over maximum
response speed. Keep replies concise, usually one or two short sentences and
roughly under 35 spoken words.

Do not use square-bracket performance/style tags in this showcase mode. Speak
naturally and let the cloned reference voice carry the delivery.

"""

    if STABLE_DEMO_MODE:
        system = f"""
{persona_prompt}

You are currently in a live voice call.

Answer naturally, conversationally, and in character.

Keep the reply concise. Aim for roughly 8-20 spoken words. One complete
conversational sentence is ideal; use a second short sentence only when useful.

Keep your actual personality, humor, attitude, flirting, sarcasm,
hesitation, or warmth as appropriate.

You may optionally put ONE natural-language performance tag at the very beginning
of your reply when it meaningfully improves the delivery.

The tag describes the overall delivery of the whole spoken reply. It is semantic,
not a fixed vocabulary: use whatever short description naturally fits, for example:
[deadpan], [softly amused], [whisper], [quietly smug], [trying not to laugh],
[tired and annoyed], [embarrassed but pretending not to be], [soft and sincere].

Usually use no performance tag. Do not force one into every response.

You may also use these non-verbal events naturally inside the reply when appropriate:
[sigh], [chuckle], [laughter], [short pause], [long pause].

Do not use more than one overall style tag and do not switch speaking styles midway
through the reply. Do not use other markdown.
"""

    history_task = None

    if VOICE_LLM_BACKEND == "gemma4":
        if not audio_snapshot:
            raise RuntimeError(
                "Gemma 4 backend requires a raw audio snapshot"
            )

        # Transcribe the user's turn immediately, in parallel with
        # generation. Parakeet on the GPU is ~10-20 ms, so the transcript
        # no longer waits for the spoken reply to finish streaming (it used
        # to be scheduled only after the whole TTS stream ended).
        history_task = asyncio.create_task(
            transcribe_turn_history(state, audio_snapshot)
        )
        state.gemma_history_task = history_task
        state.history_tasks.add(history_task)
        history_task.add_done_callback(state.history_tasks.discard)

        audio_item = {
            "type": "input_audio",
            "input_audio": {
                "data": pcm16_wav_b64(audio_snapshot),
                "format": "wav",
            },
        }

        current_content = [{
            "type": "text",
            "text": (
                "The SPOKEN AUDIO is the user's message and is always "
                "the primary thing you must respond to. Listen carefully "
                "to what the user actually says and answer that. "
                "The camera image is secondary supporting context only. "
                "Do NOT initiate comments about the user's appearance, "
                "face, expression, clothing, pose, room, or visible objects "
                "unless the user explicitly asks about something visual or "
                "the visual information is genuinely necessary to answer "
                "what they said. If audio and vision compete for your "
                "attention, follow the audio."
            ),
        }, audio_item]

        for frame in (vision_frame_b64 or []):
            current_content.append({
                "type": "image_url",
                "image_url": {
                    "url": (
                        "data:image/jpeg;base64,"
                        + frame
                    ),
                },
            })

        # Previous turns are CONTEXT, not active chat messages.
        # Feeding them back as normal user/assistant roles can make Gemma
        # occasionally answer the previous text turn instead of the current
        # raw-audio turn.
        history_lines = []

        for item in state.history[-6:]:
            role = item.get("role")
            content = str(item.get("content") or "").strip()

            if not content:
                continue

            if role == "user":
                history_lines.append(
                    f"User previously said: {content}"
                )
            elif role == "assistant":
                history_lines.append(
                    f"You already replied: {content}"
                )

        history_context = ""

        if history_lines:
            history_context = (
                "\n\n[RECENT CONVERSATION HISTORY]\n"
                "These are COMPLETED earlier turns. "
                "They are background context only. "
                "Do NOT answer any of them again.\n"
                + "\n".join(history_lines)
                + "\n[END HISTORY]\n"
            )

        current_turn_instruction = (
            "\n\nThe FINAL user message below is the CURRENT turn. "
            "Respond to the current spoken audio, not to anything in "
            "RECENT CONVERSATION HISTORY. If the current audio is unclear, "
            "say you didn't catch it rather than answering an older turn."
        )

        messages = [
            {
                "role": "system",
                "content": (
                    system
                    + history_context
                    + current_turn_instruction
                ),
            },
            {
                "role": "user",
                "content": current_content,
            },
        ]

        payload = {
            "model": GEMMA4_MODEL,
            "stream": True,
            "max_tokens": 64 if STABLE_DEMO_MODE else 120,
            "temperature": 0.8,
            "top_p": 0.9,
            "chat_template_kwargs": {
                "enable_thinking": False,
            },
            "messages": messages,
        }

        request_url = GEMMA4_URL
        request_headers = {
            "Content-Type": "application/json",
        }
        backend_label = "GEMMA"
        request_event = "gemma_request"
        first_token_event = "gemma_first_token"

    else:
        # Existing Qwen path: inject the latest asynchronously-generated
        # visual summary. Never wait for vision here.
        visual_context = ""

        try:
            if (
                latest_visual_context
                and latest_visual_context_at > 0
            ):
                visual_age = max(
                    0.0,
                    time.time() - latest_visual_context_at,
                )

                if visual_age <= 10.0:
                    visual_context = (
                        "\n\n[Live visual context — automatically generated "
                        "and may be imperfect]\n"
                        f"Visual description ({visual_age:.1f}s old): "
                        f"{latest_visual_context}\n"
                        "Use this only when relevant to the user's current "
                        "utterance. Do not assume details that are not stated "
                        "in the description."
                    )
        except Exception as exc:
            print(
                f"[VISION CONTEXT] skipped: {exc}",
                flush=True,
            )

        if visual_context:
            print(
                "[VISION CONTEXT] injected "
                + visual_context.split(
                    "Visual description (",
                    1,
                )[-1][:220],
                flush=True,
            )

        messages = [
            {
                "role": "system",
                "content": system + visual_context,
            },
        ]

        if STABLE_DEMO_MODE:
            messages.extend(state.history[-6:])
        else:
            messages.extend(state.history[-10:])

        messages.append({
            "role": "user",
            "content": transcript,
        })

        payload = {
            "model": QWEN_MODEL,
            "stream": True,
            "max_tokens": 64 if STABLE_DEMO_MODE else 120,
            "temperature": 0.8,
            "top_p": 0.9,
            "chat_template_kwargs": {
                "enable_thinking": False,
                "preserve_thinking": False,
            },
            "messages": messages,
        }

        request_url = QWEN_URL
        request_headers = {
            "Authorization": f"Bearer {QWEN_KEY}",
            "Content-Type": "application/json",
        }
        backend_label = "QWEN"
        request_event = "qwen_request"
        first_token_event = "qwen_first_token"

    llm_start = time.perf_counter()

    if not speculative:
        await state.json({
            "type": "state",
            "value": "GENERATING",
        })

    await timing(
        state,
        request_event,
        llm_start,
        mouth_stop,
    )

    first_token_seen = False

    buf = ""
    first_end = None
    first_tts_task = None

    continuation_tts_task = None
    continuation_end = None

    tts_state = {
        "gate_on": False,
        "request_seen": False,
        "first_pcm_seen": False,
    }


    def stable_demo_ready_prefix(text: str):
        if not STABLE_DEMO_MODE:
            return None

        for m in re.finditer(
            r'[.!?]+(?:["\'”’)\]]*)?(?:\s|$)',
            text,
        ):
            end = m.end()
            candidate = text[:end].strip()

            # Ignore any accidental performance tags for word count.
            visible = re.sub(
                r"\[[^\]]+\]",
                "",
                candidate,
            )

            words = re.findall(
                r"\b[\w'-]+\b",
                visible,
            )

            if len(words) >= 6:
                return candidate

        return None

    def first_speakable_continuation(text: str):
        for m in re.finditer(
            r'[.!?]+(?:["\'”’)\]]*)?(?:\s|$)',
            text,
        ):
            end = m.end()
            candidate = text[:end].strip()

            visible = re.sub(
                r"\[[^\]]+\]",
                "",
                candidate,
            )

            words = re.findall(
                r"\b[\w'-]+\b",
                visible,
            )

            if len(words) >= 5:
                return candidate, end

        return None


    async def stream_queued_continuation(
        previous_task,
        text: str,
    ):
        await previous_task

        if serial is not None and serial != state.serial:
            return

        await state.json({
            "type": "log",
            "message": f"S2 queued continuation starting: {text}",
        })

        await stream_fish_text(
            state,
            text,
            mouth_stop,
            serial,
            tts_state,
        )


    try:
        async with http.post(
            request_url,
            json=payload,
            headers=request_headers,
        ) as resp:

            if resp.status != 200:
                raise RuntimeError(
                    f"{backend_label} HTTP {resp.status}: "
                    f"{await resp.text()}"
                )

            async for raw in resp.content:
                if serial is not None and serial != state.serial:
                    return

                # Surface a Fish failure promptly even while Qwen is
                # still generating.
                if (
                    first_tts_task is not None
                    and first_tts_task.done()
                ):
                    exc = first_tts_task.exception()

                    if exc is not None:
                        raise exc

                line = raw.decode(
                    "utf-8",
                    errors="replace",
                ).strip()

                if not line.startswith("data:"):
                    continue

                data = line[5:].strip()

                if data == "[DONE]":
                    break

                try:
                    obj = json.loads(data)
                except Exception:
                    continue

                choices = obj.get("choices") or []

                if not choices:
                    continue

                delta = choices[0].get("delta") or {}
                text = delta.get("content") or ""

                if not text:
                    continue

                now = time.perf_counter()

                if not first_token_seen:
                    first_token_seen = True

                    await timing(
                        state,
                        first_token_event,
                        now,
                        mouth_stop,
                    )

                buf += text

                stable_prefix = stable_demo_ready_prefix(buf)

                if stable_prefix is not None:
                    buf = stable_prefix

                    await state.json({
                        "type": "log",
                        "message": (
                            f"{backend_label} STABLE EARLY STOP: "
                            f"{stable_prefix}"
                        ),
                    })

                    break

                if (
                    commit_event is None
                    or commit_event.is_set()
                ):
                    await state.json({
                        "type": "assistant_partial",
                        "text": buf,
                    })

                # Start TTS as soon as the first complete sentence exists,
                # but DO NOT stop Qwen generation.
                if first_tts_task is None and not STABLE_DEMO_MODE:
                    found = first_speakable_sentence(buf)

                    if found is not None:
                        first_phrase, first_end = found

                        await timing(
                            state,
                            "first_phrase",
                            now,
                            mouth_stop,
                        )

                        if commit_event is not None:
                            await commit_event.wait()

                            # Speculative Qwen had no authoritative turn serial yet.
                            # NOMO has committed now, so bind it before starting Fish.
                            if serial is None:
                                serial = state.serial
                        await state.json({
                            "type": "assistant_phrase",
                            "text": first_phrase,
                        })

                        first_tts_task = asyncio.create_task(
                            stream_fish_text(
                                state,
                                first_phrase,
                                mouth_stop,
                                serial,
                                tts_state,
                            )
                        )

                # Queue one substantial continuation while Qwen is
                # still generating. This does not delay Fish #1.
                if (
                    first_tts_task is not None
                    and continuation_tts_task is None
                    and first_end is not None
                    and first_end < len(buf)
                ):
                    pending = buf[first_end:]

                    next_found = first_speakable_continuation(
                        pending
                    )

                    if next_found is not None:
                        (
                            continuation_phrase,
                            continuation_rel_end,
                        ) = next_found

                        continuation_end = (
                            first_end
                            + continuation_rel_end
                        )

                        await state.json({
                            "type": "log",
                            "message": (
                                "S2 continuation queued: "
                                f"{continuation_phrase}"
                            ),
                        })

                        continuation_tts_task = asyncio.create_task(
                            stream_queued_continuation(
                                first_tts_task,
                                continuation_phrase,
                            )
                        )

        full_response = buf.strip()

        if not full_response:
            return

        await state.json({
            "type": "log",
            "message": f"{backend_label} FULL: {full_response}",
        })

        # The LLM may have ended before producing a punctuation boundary.
        # In that case the entire response becomes the first Fish request.
        if first_tts_task is None:
            first_end = len(buf)
            first_phrase = full_response

            await timing(
                state,
                "first_phrase",
                time.perf_counter(),
                mouth_stop,
            )

            if commit_event is not None:
                await commit_event.wait()

                # Speculative Qwen had no authoritative turn serial yet.
                # NOMO has committed now, so bind it before starting Fish.
                if serial is None:
                    serial = state.serial
            await state.json({
                "type": "assistant_phrase",
                "text": first_phrase,
            })

            first_tts_task = asyncio.create_task(
                stream_fish_text(
                    state,
                    first_phrase,
                    mouth_stop,
                    serial,
                    tts_state,
                )
            )

        # The LLM is finished. A second TTS request may already be queued
        # behind the first one.
        spoken_end = first_end

        if continuation_tts_task is not None:
            await continuation_tts_task
            continuation_tts_task = None

            if continuation_end is not None:
                spoken_end = continuation_end
        else:
            await first_tts_task

        first_tts_task = None

        if serial is not None and serial != state.serial:
            return

        remainder = ""

        if (
            spoken_end is not None
            and spoken_end < len(buf)
        ):
            remainder = buf[spoken_end:].strip()

        if remainder:
            await state.json({
                "type": "log",
                "message": f"S2 REMAINDER: {remainder}",
            })

            await stream_fish_text(
                state,
                remainder,
                mouth_stop,
                serial,
                tts_state,
            )

        if serial is not None and serial != state.serial:
            return

        # Only now tell the worklet that the complete spoken response ended.
        await state.json({
            "type": "tts_end",
        })

        if VOICE_LLM_BACKEND == "gemma4":
            # Transcription was started at the top of this turn and has
            # almost certainly finished by now. Append the completed turn
            # to text history for the next request.
            user_text = None

            if history_task is not None:
                user_text = await history_task
                history_task = None

            if user_text:
                state.history.append({
                    "role": "user",
                    "content": user_text,
                })

                state.history.append({
                    "role": "assistant",
                    "content": full_response,
                })

                state.history = state.history[-10:]

        else:
            state.history.append({
                "role": "user",
                "content": transcript,
            })

            state.history.append({
                "role": "assistant",
                "content": full_response,
            })

    except asyncio.CancelledError:
        if (
            continuation_tts_task is not None
            and not continuation_tts_task.done()
        ):
            continuation_tts_task.cancel()

            try:
                await continuation_tts_task
            except asyncio.CancelledError:
                pass

        if (
            first_tts_task is not None
            and not first_tts_task.done()
        ):
            first_tts_task.cancel()

            try:
                await first_tts_task
            except asyncio.CancelledError:
                pass

        raise

    except Exception as e:
        if (
            continuation_tts_task is not None
            and not continuation_tts_task.done()
        ):
            continuation_tts_task.cancel()

            try:
                await continuation_tts_task
            except asyncio.CancelledError:
                pass

        if (
            first_tts_task is not None
            and not first_tts_task.done()
        ):
            first_tts_task.cancel()

            try:
                await first_tts_task
            except asyncio.CancelledError:
                pass

        await state.json({
            "type": "error",
            "message": str(e),
        })


# ---------------------------------------------------------------------------
# Candidate turn completion
# ---------------------------------------------------------------------------

async def handle_candidate(
    state: State,
    mouth_stop: float,
    serial: int,
    snapshot: bytes,
):
    try:
        if VOICE_LLM_BACKEND == "gemma4":
            speculative_gemma_task = state.speculative_gemma_task

            use_speculative_gemma = (
                speculative_gemma_task is not None
                and not speculative_gemma_task.done()
                and state.speculative_gemma_commit is not None
                and state.speculative_gemma_epoch == state.partial_epoch
                and state.speculative_gemma_snapshot_len == len(snapshot)
            )

            if serial != state.serial:
                return

            if not state.candidate_fired:
                return

            if use_speculative_gemma:
                commit = state.speculative_gemma_commit

                # This request is authoritative now. Remove it from the
                # speculative slot BEFORE waking it. Otherwise a later NOMO
                # positive chunk can mistake the committed response for
                # disposable speculative work and cancel VoxCPM with it.
                state.speculative_gemma_task = None
                state.speculative_gemma_commit = None
                state.speculative_gemma_snapshot_len = 0
                state.speculative_gemma_epoch = -1

                commit.set()

                await state.json({
                    "type": "log",
                    "message": "GEMMA speculative committed",
                })

                await speculative_gemma_task

            else:
                await cancel_speculative_gemma(state)

                frames = select_gemma_vision_frames()

                await state.json({
                    "type": "log",
                    "message": (
                        "GEMMA authoritative request: "
                        f"{len(snapshot) / (SAMPLE_RATE * 2):.2f}s audio, "
                        f"{len(frames)} frame(s)"
                    ),
                })

                await run_response(
                    state,
                    "",
                    mouth_stop,
                    serial,
                    audio_snapshot=snapshot,
                    vision_frame_b64=frames,
                )

            return

        # Final high-quality STT pass over the complete utterance.
        #
        # If NOMO's first negative chunk launched STT speculatively,
        # the snapshot here should be byte-for-byte the same length after
        # the second negative chunk is trimmed.
        use_speculative = (
            state.speculative_whisper_task is not None
            and state.speculative_whisper_valid
            and state.speculative_whisper_epoch == state.partial_epoch
            and state.speculative_whisper_snapshot_len == len(snapshot)
        )

        if use_speculative:
            transcript = await state.speculative_whisper_task

            await state.json({
                "type": "log",
                "message": "STT speculative final reused",
            })
        else:
            transcript = await stt_engine.transcribe(snapshot)

        speculative_qwen_task = state.speculative_qwen_task

        use_speculative_qwen = (
            speculative_qwen_task is not None
            and not speculative_qwen_task.done()
            and state.speculative_qwen_commit is not None
            and state.speculative_qwen_transcript == transcript
            and state.speculative_qwen_epoch == state.partial_epoch
        )

        invalidate_speculative_whisper(state)

        now = time.perf_counter()

        if serial != state.serial:
            return

        if not state.candidate_fired:
            return

        if not transcript:
            state.response_started = False
            await cancel_speculative_qwen(state)
            return

        await timing(
            state,
            "whisper_final",
            now,
            mouth_stop,
        )

        state.latest_partial = transcript

        await state.json({
            "type": "transcript",
            "text": transcript,
            "partial": False,
        })

        await state.json({
            "type": "candidate",
            "text": transcript,
        })

        if use_speculative_qwen:
            state.speculative_qwen_commit.set()

            await state.json({
                "type": "log",
                "message": "QWEN speculative committed",
            })

            try:
                await speculative_qwen_task
            finally:
                state.speculative_qwen_task = None
                state.speculative_qwen_commit = None
                state.speculative_qwen_transcript = ""
                state.speculative_qwen_epoch = -1

        else:
            await cancel_speculative_qwen(state)

            await run_response(
                state,
                transcript,
                mouth_stop,
                serial,
            )

    except asyncio.CancelledError:
        raise

    except Exception as e:
        state.response_started = False

        await state.json({
            "type": "error",
            "message": str(e),
        })


async def start_candidate(
    state: State,
    mouth_stop: float,
):
    if state.response_started:
        return

    if not state.audio_frames:
        return

    state.response_started = True

    state.serial += 1
    serial = state.serial

    snapshot = b"".join(state.audio_frames)

    state.active_mouth_stop = mouth_stop

    state.generation = asyncio.create_task(
        handle_candidate(
            state,
            mouth_stop,
            serial,
            snapshot,
        )
    )


# ---------------------------------------------------------------------------
# Microphone frames
# ---------------------------------------------------------------------------

# ---------------------------------------------------------------------------
# Speculative final STT
# ---------------------------------------------------------------------------

def invalidate_speculative_whisper(
    state: State,
):
    state.speculative_whisper_valid = False
    state.speculative_whisper_snapshot_len = 0
    state.speculative_whisper_epoch = -1


async def launch_speculative_whisper(
    state: State,
    mouth_stop: float,
):
    if not state.audio_frames:
        return

    # Avoid running two GPU STT passes concurrently. A cancelled
    # asyncio.to_thread task would still leave its worker thread running.
    if (
        state.speculative_whisper_task is not None
        and not state.speculative_whisper_task.done()
    ):
        return

    snapshot = b"".join(state.audio_frames)

    state.speculative_whisper_valid = True
    state.speculative_whisper_snapshot_len = len(snapshot)
    state.speculative_whisper_epoch = state.partial_epoch

    state.speculative_whisper_task = asyncio.create_task(
        stt_engine.transcribe(snapshot)
    )

    whisper_task = state.speculative_whisper_task
    epoch = state.speculative_whisper_epoch
    snapshot_len = state.speculative_whisper_snapshot_len

    emotion_task = asyncio.create_task(
        run_emotion_shadow(
            state,
            snapshot,
            epoch,
            snapshot_len,
            mouth_stop,
        )
    )

    sensevoice_task = asyncio.create_task(
        run_sensevoice_shadow(
            state,
            snapshot,
            epoch,
            snapshot_len,
            mouth_stop,
        )
    )

    asyncio.create_task(
        run_tone_fusion_shadow(
            state,
            whisper_task,
            emotion_task,
            sensevoice_task,
            epoch,
            snapshot_len,
            mouth_stop,
        )
    )

    asyncio.create_task(
        launch_qwen_after_speculative_whisper(
            state,
            whisper_task,
            epoch,
            snapshot_len,
            mouth_stop,
        )
    )

    await state.json({
        "type": "log",
        "message": "STT speculative final started",
    })


async def run_emotion_shadow(
    state: State,
    pcm_bytes: bytes,
    epoch: int,
    snapshot_len: int,
    mouth_stop: float,
):
    # Don't even launch work for an already-stale speculative turn.
    if (
        not state.speculative_whisper_valid
        or state.speculative_whisper_epoch != epoch
        or state.speculative_whisper_snapshot_len != snapshot_len
    ):
        return

    def do_request():
        import json
        import urllib.request

        req = urllib.request.Request(
            EMOTION_URL,
            data=pcm_bytes,
            headers={
                "Content-Type": "application/octet-stream",
            },
            method="POST",
        )

        with urllib.request.urlopen(
            req,
            timeout=0.25,
        ) as resp:
            return json.loads(
                resp.read().decode("utf-8")
            )

    t0 = time.perf_counter()

    try:
        result = await asyncio.to_thread(
            do_request,
        )
    except Exception as exc:
        # Shadow mode must never interfere with the voice-call path.
        print(
            f"[EMOTION SHADOW] request failed: {exc}",
            flush=True,
        )
        return

    wall_ms = (
        time.perf_counter() - t0
    ) * 1000.0

    # User resumed speaking while emotion2vec was running.
    # Discard this result.
    if (
        not state.speculative_whisper_valid
        or state.speculative_whisper_epoch != epoch
        or state.speculative_whisper_snapshot_len != snapshot_len
    ):
        print(
            "[EMOTION SHADOW] stale result discarded",
            flush=True,
        )
        return

    emotion = str(
        result.get("emotion", "unknown")
    )

    confidence = float(
        result.get("confidence", 0.0)
    )

    margin = float(
        result.get("margin", 0.0)
    )

    inference_ms = float(
        result.get("latency_ms", 0.0)
    )

    mouth_stop_ms = (
        time.perf_counter() - mouth_stop
    ) * 1000.0

    scores = result.get("scores") or []

    top3 = ", ".join(
        f"{x.get('emotion', '?')}={float(x.get('score', 0.0)):.2f}"
        for x in scores[:3]
    )

    message = (
        f"EMOTION shadow: {emotion} "
        f"p={confidence:.2f} "
        f"margin={margin:.2f} "
        f"infer={inference_ms:.1f}ms "
        f"wall={wall_ms:.1f}ms "
        f"mouth_stop={mouth_stop_ms:.1f}ms"
    )

    if top3:
        message += f" [{top3}]"

    print(
        f"[EMOTION SHADOW] {message}",
        flush=True,
    )

    await state.json({
        "type": "log",
        "message": message,
    })

    await state.json({
        "type": "emotion",
        "emotion": emotion,
        "confidence": confidence,
        "margin": margin,
        "scores": scores[:3],
    })

    return result



async def run_sensevoice_shadow(
    state: State,
    pcm_bytes: bytes,
    epoch: int,
    snapshot_len: int,
    mouth_stop: float,
):
    # Shadow work must never affect the authoritative voice path.
    if (
        not state.speculative_whisper_valid
        or state.speculative_whisper_epoch != epoch
        or state.speculative_whisper_snapshot_len != snapshot_len
    ):
        return None

    def do_request():
        import json
        import urllib.request

        req = urllib.request.Request(
            SENSEVOICE_URL,
            data=pcm_bytes,
            headers={
                "Content-Type": "application/octet-stream",
            },
            method="POST",
        )

        with urllib.request.urlopen(
            req,
            timeout=0.5,
        ) as resp:
            return json.loads(
                resp.read().decode("utf-8")
            )

    t0 = time.perf_counter()

    try:
        result = await asyncio.to_thread(
            do_request,
        )
    except Exception as exc:
        print(
            f"[SENSEVOICE SHADOW] request failed: {exc}",
            flush=True,
        )
        return None

    wall_ms = (
        time.perf_counter() - t0
    ) * 1000.0

    if (
        not state.speculative_whisper_valid
        or state.speculative_whisper_epoch != epoch
        or state.speculative_whisper_snapshot_len != snapshot_len
    ):
        print(
            "[SENSEVOICE SHADOW] stale result discarded",
            flush=True,
        )
        return None

    emotion = str(
        result.get("emotion") or "unknown"
    )

    events = result.get("events") or []

    inference_ms = float(
        result.get("inference_ms", 0.0)
    )

    mouth_stop_ms = (
        time.perf_counter() - mouth_stop
    ) * 1000.0

    event_text = (
        ", ".join(str(x) for x in events)
        if events
        else "none"
    )

    message = (
        f"SENSEVOICE shadow: {emotion} "
        f"events={event_text} "
        f"infer={inference_ms:.1f}ms "
        f"wall={wall_ms:.1f}ms "
        f"mouth_stop={mouth_stop_ms:.1f}ms"
    )

    print(
        f"[SENSEVOICE SHADOW] {message}",
        flush=True,
    )

    await state.json({
        "type": "log",
        "message": message,
    })

    await state.json({
        "type": "sensevoice",
        "emotion": emotion,
        "events": events,
        "language": result.get("language"),
        "inference_ms": inference_ms,
    })

    return result


async def run_tone_fusion_shadow(
    state: State,
    whisper_task,
    emotion_task,
    sensevoice_task,
    epoch: int,
    snapshot_len: int,
    mouth_stop: float,
):
    # This task is entirely diagnostic. Qwen/TTS never wait for it.
    results = await asyncio.gather(
        whisper_task,
        emotion_task,
        sensevoice_task,
        return_exceptions=True,
    )

    transcript, emotion_result, sensevoice_result = results

    if isinstance(transcript, BaseException):
        print(
            f"[TONE FUSION] STT failed: {transcript}",
            flush=True,
        )
        return

    if not isinstance(transcript, str):
        return

    transcript = transcript.strip()

    if not transcript:
        return

    if (
        not state.speculative_whisper_valid
        or state.speculative_whisper_epoch != epoch
        or state.speculative_whisper_snapshot_len != snapshot_len
    ):
        print(
            "[TONE FUSION] stale result discarded",
            flush=True,
        )
        return

    if not isinstance(emotion_result, dict):
        emotion_result = None

    if not isinstance(sensevoice_result, dict):
        sensevoice_result = None

    # Don't silently turn Kev into a text-only sentiment classifier.
    if (
        emotion_result is None
        and sensevoice_result is None
    ):
        return

    emotion_scores = {}

    if emotion_result is not None:
        for item in (
            emotion_result.get("scores") or []
        ):
            try:
                emotion_scores[
                    str(item.get("emotion"))
                ] = float(
                    item.get("score", 0.0)
                )
            except Exception:
                pass

    acoustic_evidence = {}

    if emotion_result is not None:
        acoustic_evidence["emotion2vec"] = {
            "top_emotion": emotion_result.get(
                "emotion"
            ),
            "confidence": emotion_result.get(
                "confidence"
            ),
            "margin": emotion_result.get(
                "margin"
            ),
            "scores": emotion_scores,
        }

    if sensevoice_result is not None:
        acoustic_evidence["sensevoice"] = {
            "emotion": (
                sensevoice_result.get("emotion")
                or "unknown"
            ),
            "events": (
                sensevoice_result.get("events")
                or []
            ),
        }

    payload = {
        "state": {
            "transcript": transcript,
            "acoustic_evidence": acoustic_evidence,
        },
        "model": "kev-latest",
        "questions": {
            "delivery": {
                "type": "choice",
                "instructions": (
                    "Classify the speaker's delivery style using both "
                    "the transcript and the acoustic evidence. "
                    "The acoustic classifiers are imperfect and may "
                    "disagree. SenseVoice 'unknown' means it did not "
                    "detect a clear canonical emotion; do not treat "
                    "unknown as neutral."
                ),
                "criteria": {
                    "neutral": (
                        "Ordinary matter-of-fact delivery with no "
                        "strong affect."
                    ),
                    "deadpan": (
                        "Dry, flat, sarcastic, ironic, or "
                        "intentionally understated delivery."
                    ),
                    "amused": (
                        "Entertained, playful, smiling, chuckling, "
                        "or trying not to laugh."
                    ),
                    "annoyed": (
                        "Irritated, impatient, exasperated, "
                        "or frustrated."
                    ),
                    "sad": (
                        "Downcast, unhappy, hurt, or subdued."
                    ),
                    "surprised": (
                        "Startled, shocked, or strongly reacting "
                        "to something unexpected."
                    ),
                },
            }
        },
    }

    def do_request():
        import json
        import urllib.request

        req = urllib.request.Request(
            KEV_URL,
            data=json.dumps(payload).encode(
                "utf-8"
            ),
            headers={
                "Content-Type": "application/json",
            },
            method="POST",
        )

        with urllib.request.urlopen(
            req,
            timeout=0.5,
        ) as resp:
            return json.loads(
                resp.read().decode("utf-8")
            )

    t0 = time.perf_counter()

    try:
        result = await asyncio.to_thread(
            do_request,
        )
    except Exception as exc:
        print(
            f"[TONE FUSION] Kev request failed: {exc}",
            flush=True,
        )
        return

    wall_ms = (
        time.perf_counter() - t0
    ) * 1000.0

    if (
        not state.speculative_whisper_valid
        or state.speculative_whisper_epoch != epoch
        or state.speculative_whisper_snapshot_len != snapshot_len
    ):
        print(
            "[TONE FUSION] stale Kev result discarded",
            flush=True,
        )
        return

    try:
        answer = result["answers"]["delivery"]
    except Exception:
        print(
            f"[TONE FUSION] malformed response: {result}",
            flush=True,
        )
        return

    choice = str(
        answer.get("choice", "unknown")
    )

    confidence = float(
        answer.get("confidence", 0.0)
    )

    probabilities = {
        str(k): float(v)
        for k, v in (
            answer.get("probabilities") or {}
        ).items()
    }

    model_ms = float(
        result.get("latency_ms", 0.0)
    )

    top3 = sorted(
        probabilities.items(),
        key=lambda item: item[1],
        reverse=True,
    )[:3]

    top_text = ", ".join(
        f"{name}={prob:.2f}"
        for name, prob in top3
    )

    mouth_stop_ms = (
        time.perf_counter() - mouth_stop
    ) * 1000.0

    message = (
        f"TONE FUSION: {choice} "
        f"conf={confidence:.2f} "
        f"kev={model_ms:.1f}ms "
        f"wall={wall_ms:.1f}ms "
        f"mouth_stop={mouth_stop_ms:.1f}ms "
        f"[{top_text}]"
    )

    print(
        f"[TONE FUSION] {message}",
        flush=True,
    )

    await state.json({
        "type": "log",
        "message": message,
    })

    await state.json({
        "type": "tone_fusion",
        "choice": choice,
        "confidence": confidence,
        "probabilities": probabilities,
        "kev_latency_ms": model_ms,
        "wall_ms": wall_ms,
        "transcript": transcript,
    })


async def cancel_speculative_qwen(
    state: State,
):
    task = state.speculative_qwen_task

    if task is not None:
        if not task.done():
            task.cancel()

        try:
            await task
        except asyncio.CancelledError:
            pass
        except Exception:
            # A failed speculative request is harmless; the authoritative
            # path will fall back to a normal request.
            pass

    state.speculative_qwen_task = None
    state.speculative_qwen_commit = None
    state.speculative_qwen_transcript = ""
    state.speculative_qwen_epoch = -1


async def cancel_speculative_gemma(
    state: State,
):
    task = state.speculative_gemma_task

    if task is not None:
        if not task.done():
            task.cancel()

        try:
            await task
        except asyncio.CancelledError:
            pass
        except Exception:
            # A failed speculative request is harmless; the authoritative
            # path can issue a fresh Gemma request after NOMO commits.
            pass

    state.speculative_gemma_task = None
    state.speculative_gemma_commit = None
    state.speculative_gemma_snapshot_len = 0
    state.speculative_gemma_epoch = -1


async def launch_speculative_gemma(
    state: State,
    mouth_stop: float,
):
    if not state.audio_frames:
        return

    old = state.speculative_gemma_task

    if old is not None:
        if not old.done():
            return

        try:
            old.result()
        except asyncio.CancelledError:
            pass
        except Exception:
            pass

    snapshot = b"".join(state.audio_frames)
    epoch = state.partial_epoch
    snapshot_len = len(snapshot)
    frames = select_gemma_vision_frames()

    commit = asyncio.Event()

    state.speculative_gemma_commit = commit
    state.speculative_gemma_snapshot_len = snapshot_len
    state.speculative_gemma_epoch = epoch

    task = asyncio.create_task(
        run_response(
            state,
            "",
            mouth_stop,
            None,
            commit_event=commit,
            speculative=True,
            audio_snapshot=snapshot,
            vision_frame_b64=frames,
        )
    )

    state.speculative_gemma_task = task

    await state.json({
        "type": "log",
        "message": (
            "GEMMA speculative started: "
            f"{snapshot_len / (SAMPLE_RATE * 2):.2f}s audio, "
            f"{len(frames)} frame(s)"
        ),
    })


async def launch_qwen_after_speculative_whisper(
    state: State,
    whisper_task: asyncio.Task,
    epoch: int,
    snapshot_len: int,
    mouth_stop: float,
):
    try:
        transcript = await whisper_task

        # The user may have resumed speaking while STT was running.
        if (
            not state.speculative_whisper_valid
            or state.speculative_whisper_task is not whisper_task
            or state.speculative_whisper_epoch != epoch
            or state.speculative_whisper_snapshot_len != snapshot_len
        ):
            return

        if not transcript:
            return

        # There should only ever be one live speculative Qwen request.
        old = state.speculative_qwen_task

        if old is not None:
            if not old.done():
                return

            try:
                old.result()
            except asyncio.CancelledError:
                pass
            except Exception:
                pass

        commit = asyncio.Event()

        state.speculative_qwen_commit = commit
        state.speculative_qwen_transcript = transcript
        state.speculative_qwen_epoch = epoch

        state.speculative_qwen_task = asyncio.create_task(
            run_response(
                state,
                transcript,
                mouth_stop,
                None,
                commit_event=commit,
                speculative=True,
            )
        )

        await state.json({
            "type": "log",
            "message": (
                "QWEN speculative started: "
                f"{transcript}"
            ),
        })

    except asyncio.CancelledError:
        raise

    except Exception as e:
        await state.json({
            "type": "log",
            "message": (
                "QWEN speculative launch skipped: "
                f"{e}"
            ),
        })


# ---------------------------------------------------------------------------
# Target-speaker turn detection
# ---------------------------------------------------------------------------

async def start_target_speech(
    state: State,
    prob: float,
):
    if state.playback_active:
        await state.json({
            "type": "log",
            "message": (
                "NOMO TARGET ON ignored during playback "
                f"p={prob:.3f}"
            ),
        })
        return

    if state.speculative_gemma_task is not None:
        await cancel_speculative_gemma(state)

    if state.speculative_qwen_task is not None:
        await cancel_speculative_qwen(state)

    # A new real user turn supersedes any candidate/generation
    # that was already in flight.
    if (
        state.candidate_fired
        or (
            state.generation
            and not state.generation.done()
        )
    ):
        await state.cancel_generation()

    state.candidate_fired = False
    state.response_started = False

    state.turn_active = True
    state.target_speaking = True
    state.target_end_pending_at = None
    state.target_silence_started_at = None

    # The pre-roll contains the speech that occurred while the
    # two-positive hysteresis was being established.
    state.audio_frames = list(state.pre_roll)
    state.pre_roll.clear()

    state.speech_frames = len(state.audio_frames)
    state.silence_frames = 0

    state.latest_partial = ""
    state.partial_epoch += 1
    state.last_partial_launch = 0.0

    await state.json({
        "type": "log",
        "message": f"NOMO TARGET ON p={prob:.3f}",
    })


async def resume_target_speech(
    state: State,
    prob: float,
):
    if state.playback_active:
        await state.json({
            "type": "log",
            "message": (
                "NOMO TARGET RESUME ignored during playback "
                f"p={prob:.3f}"
            ),
        })
        return

    # Once Pali has already been established for this turn,
    # one positive chunk during the short end grace is enough
    # to treat it as a pause rather than a new utterance.
    state.target_speaking = True
    state.target_end_pending_at = None
    state.target_negative_chunks = 0
    state.target_silence_started_at = None

    if state.pre_roll:
        state.audio_frames.extend(state.pre_roll)
        state.pre_roll.clear()

    await state.json({
        "type": "log",
        "message": f"NOMO TARGET RESUME p={prob:.3f}",
    })


async def finish_target_turn(
    state: State,
):
    if state.candidate_fired:
        return

    if not state.audio_frames:
        return

    mouth_stop = state.last_speech_at

    if mouth_stop is None:
        mouth_stop = time.perf_counter()

    state.candidate_fired = True
    state.target_end_pending_at = None

    candidate_at = time.perf_counter()

    await timing(
        state,
        "vad_candidate",
        candidate_at,
        mouth_stop,
    )

    await state.json({
        "type": "log",
        "message": "NOMO target turn complete",
    })

    await start_candidate(
        state,
        mouth_stop,
    )

    # start_candidate already took its own immutable snapshot.
    # Start collecting clean pre-roll for a possible interruption.
    state.turn_active = False
    state.target_speaking = False
    state.audio_frames = []
    state.pre_roll.clear()


async def process_target_audio_frame(
    state: State,
    data: bytes,
):
    now = time.perf_counter()

    # Preserve 20 ms granularity for STT even though nomo
    # itself emits one probability per 160 ms.
    if state.target_speaking:
        state.audio_frames.append(data)
        state.speech_frames += 1
    else:
        state.pre_roll.append(data)

    state.nomo_chunk_frames.append(data)

    # While we're inside the short post-target grace period,
    # candidate creation happens independently of background speech.
    async def maybe_finish():
        if (
            state.target_end_pending_at is not None
            and not state.target_speaking
            and (
                time.perf_counter()
                - state.target_end_pending_at
                >= NOMO_END_GRACE_SEC
            )
        ):
            await finish_target_turn(state)

    if len(state.nomo_chunk_frames) < NOMO_CHUNK_FRAMES:
        await maybe_finish()
        return

    chunk_frames = state.nomo_chunk_frames[:NOMO_CHUNK_FRAMES]
    del state.nomo_chunk_frames[:NOMO_CHUNK_FRAMES]

    pcm = np.frombuffer(
        b"".join(chunk_frames),
        dtype="<i2",
    ).astype(np.float32) / 32768.0

    prob = await asyncio.to_thread(
        state.nomo_session.process_chunk,
        pcm,
    )

    # process_chunk() runs off-thread. VoxCPM may have begun playback
    # while we were waiting for this score. Never let that stale score
    # start/cancel a turn after the mic gate has gone up.
    if state.playback_active:
        await state.json({
            "type": "log",
            "message": (
                "NOMO stale result ignored during playback "
                f"p={prob:.3f}"
            ),
        })
        return

    positive = prob >= NOMO_THRESHOLD

    if positive:
        state.target_positive_chunks += 1

        # Once an end candidate has started, do not let one isolated
        # positive NOMO chunk reset the entire endpoint detector.
        #
        # This matters with music/background vocals: occasional false
        # target-speaker spikes are possible. Genuine resumed speech
        # should produce sustained positive chunks.
        pending_end = (
            state.target_speaking
            and state.target_negative_chunks > 0
        )

        confirmed_resume = (
            not pending_end
            or state.target_positive_chunks >= NOMO_RESUME_CHUNKS
        )

        if pending_end and not confirmed_resume:
            await state.json({
                "type": "log",
                "message": (
                    "NOMO END HOLD "
                    f"p={prob:.3f} "
                    f"resume={state.target_positive_chunks}/"
                    f"{NOMO_RESUME_CHUNKS}"
                ),
            })

        else:
            # Genuine/sustained target speech resumed. Only now
            # invalidate speculative work from the apparent endpoint.
            if state.speculative_whisper_valid:
                invalidate_speculative_whisper(state)

            if state.speculative_gemma_task is not None:
                await cancel_speculative_gemma(state)

                await state.json({
                    "type": "log",
                    "message": (
                        "GEMMA speculative cancelled — "
                        "target genuinely resumed"
                    ),
                })

            if state.speculative_qwen_task is not None:
                await cancel_speculative_qwen(state)

                await state.json({
                    "type": "log",
                    "message": (
                        "QWEN speculative cancelled — "
                        "target genuinely resumed"
                    ),
                })

            state.target_negative_chunks = 0
            state.target_silence_started_at = None

            if state.target_end_pending_at is not None:
                await resume_target_speech(
                    state,
                    prob,
                )

            elif (
                not state.target_speaking
                and state.target_positive_chunks
                >= NOMO_ENTER_CHUNKS
            ):
                await start_target_speech(
                    state,
                    prob,
                )

    else:
        state.target_positive_chunks = 0
        state.target_negative_chunks += 1

        # First negative 160 ms target-speaker chunk.
        if (
            state.target_speaking
            and state.target_negative_chunks == 1
            and state.speech_frames >= MIN_SPEECH_FRAMES
        ):
            # process_chunk() runs at the END of this 160 ms block,
            # so its beginning is our best estimate of true mouth-stop.
            state.target_silence_started_at = (
                time.perf_counter() - NOMO_CHUNK_SEC
            )

            if VOICE_LLM_BACKEND == "gemma4":
                await launch_speculative_gemma(
                    state,
                    state.target_silence_started_at,
                )
            else:
                await launch_speculative_whisper(
                    state,
                    state.target_silence_started_at,
                )

        if (
            state.target_speaking
            and state.target_negative_chunks
            >= NOMO_EXIT_CHUNKS
        ):
            state.target_speaking = False
            state.target_end_pending_at = time.perf_counter()

            # We already appended both negative chunks. Keep one short
            # trailing tail but discard the newest background-only chunk.
            trim = NOMO_CHUNK_FRAMES

            if len(state.audio_frames) > trim:
                del state.audio_frames[-trim:]

            state.last_speech_at = (
                state.target_silence_started_at
                if state.target_silence_started_at is not None
                else time.perf_counter() - NOMO_CHUNK_SEC
            )

            state.target_silence_started_at = None

            await state.json({
                "type": "log",
                "message": f"NOMO TARGET OFF p={prob:.3f}",
            })

    maybe_launch_partial(
        state,
        time.perf_counter(),
    )

    await maybe_finish()


async def process_audio_frame(
    state: State,
    data: bytes,
):
    if len(data) != FRAME_BYTES:
        return

    # Half-duplex while we're fixing/testing playback.
    # This prevents assistant speaker leakage from becoming fake barge-in.
    if state.playback_active:
        return

    if state.nomo_session is not None:
        await process_target_audio_frame(
            state,
            data,
        )
        return

    # No enrolled speaker profile: preserve the old WebRTC
    # behavior as a fallback.
    now = time.perf_counter()
    speech = state.vad.is_speech(
        data,
        SAMPLE_RATE,
    )

    if speech:
        was_candidate = state.candidate_fired

        if not state.turn_active:
            state.turn_active = True
            state.audio_frames = list(state.pre_roll)
            state.pre_roll.clear()

            state.latest_partial = ""
            state.partial_epoch += 1
            state.last_partial_launch = 0.0

            await state.json({
                "type": "state",
                "value": "LISTENING",
            })

        state.audio_frames.append(data)
        state.speech_frames += 1

        state.last_speech_at = now
        state.silence_frames = 0
        state.candidate_fired = False

        if was_candidate or (
            state.generation
            and not state.generation.done()
        ):
            await state.cancel_generation()

        state.response_started = False

        maybe_launch_partial(state, now)

        return

    # Non-speech frame.
    if not state.turn_active:
        state.pre_roll.append(data)
        return

    state.audio_frames.append(data)

    if state.last_speech_at is None:
        return

    state.silence_frames += 1

    maybe_launch_partial(state, now)

    if (
        state.silence_frames >= VAD_HANGOVER_FRAMES
        and state.speech_frames >= MIN_SPEECH_FRAMES
        and not state.candidate_fired
    ):
        state.candidate_fired = True

        candidate_at = time.perf_counter()

        await timing(
            state,
            "vad_candidate",
            candidate_at,
            state.last_speech_at,
        )

        await start_candidate(
            state,
            state.last_speech_at,
        )


# ---------------------------------------------------------------------------
# VISION SHADOW SIDECAR
# ---------------------------------------------------------------------------

VISION_URL = os.getenv(
    "VISION_URL",
    "http://127.0.0.1:3045/v1/chat/completions",
)
VISION_MODEL = "Qwen/Qwen3-VL-4B-Instruct-FP8"

VISION_CAPTURE_FPS = 10.0
VISION_MAX_FRAMES = 200
VISION_MIN_FRAMES = 20
VISION_MIN_LAUNCH_INTERVAL = 1.5

# Single-user voice-lab shadow state.
#
# Frames are already-resized JPEGs supplied by the browser.
# The main Qwen path does NOT consume this context yet.
vision_frames = deque(maxlen=VISION_MAX_FRAMES)
vision_task = None
vision_last_launch = 0.0

latest_visual_context = ""
latest_visual_context_at = 0.0


async def run_vision_shadow(ws: WebSocket):
    global latest_visual_context
    global latest_visual_context_at

    snapshot = list(vision_frames)

    if len(snapshot) < VISION_MIN_FRAMES:
        return

    frame_b64 = [item[1] for item in snapshot]
    frame_times = [item[0] for item in snapshot]

    count = len(frame_b64)

    if count > 1:
        represented_seconds = max(
            0.1,
            (frame_times[-1] - frame_times[0]) / 1000.0,
        )
    else:
        represented_seconds = count / VISION_CAPTURE_FPS

    # Keep temporal metadata simple and deterministic.
    # Frames arrive at our own ~10 FPS capture cadence.
    effective_fps = count / represented_seconds

    video_url = (
        "data:video/jpeg;base64,"
        + ",".join(frame_b64)
    )

    prompt = (
        "Describe only what is visibly supported in this video window. "
        "Summarize the current scene and important changes over time in "
        "1-3 short factual sentences. Focus on visible people, objects, "
        "actions, gestures, movement, and text when it is clearly readable. "
        "Do not identify people. Do not infer feelings, intentions, "
        "relationships, or unseen causes. If text is too unclear to read, "
        "say that rather than guessing."
    )

    payload = {
        "model": VISION_MODEL,
        "messages": [
            {
                "role": "user",
                "content": [
                    {
                        "type": "video_url",
                        "video_url": {
                            "url": video_url,
                        },
                    },
                    {
                        "type": "text",
                        "text": prompt,
                    },
                ],
            }
        ],
        "media_io_kwargs": {
            "video": {
                "num_frames": count,
                "fps": effective_fps,
                "frames_indices": list(range(count)),
                "total_num_frames": count,
                "duration": represented_seconds,
                "do_sample_frames": False,
            }
        },
        "temperature": 0,
        "max_tokens": 80,
    }

    started = time.perf_counter()

    try:
        async with http.post(
            VISION_URL,
            json=payload,
        ) as resp:
            body = await resp.text()

            if resp.status != 200:
                raise RuntimeError(
                    f"Vision HTTP {resp.status}: {body[:500]}"
                )

        result = json.loads(body)

        description = (
            result["choices"][0]["message"]["content"]
        ).strip()

        elapsed_ms = (
            time.perf_counter() - started
        ) * 1000.0

        latest_visual_context = description
        latest_visual_context_at = time.time()

        usage = result.get("usage") or {}

        try:
            await ws.send_text(json.dumps({
                "type": "vision",
                "description": description,
                "frames": count,
                "window_sec": round(
                    represented_seconds,
                    2,
                ),
                "inference_ms": round(
                    elapsed_ms,
                    1,
                ),
                "prompt_tokens": usage.get(
                    "prompt_tokens"
                ),
                "frame_timestamp_ms": (
                    frame_times[-1]
                    if frame_times
                    else None
                ),
            }))
        except Exception:
            pass

        print(
            "[VISION SHADOW] "
            f"frames={count} "
            f"window={represented_seconds:.2f}s "
            f"inference={elapsed_ms:.1f}ms "
            f"tokens={usage.get('prompt_tokens')} "
            f'description="{description}"',
            flush=True,
        )

    except Exception as exc:
        print(
            f"[VISION SHADOW] ERROR: {exc}",
            flush=True,
        )

        try:
            await ws.send_text(json.dumps({
                "type": "vision_error",
                "message": str(exc),
            }))
        except Exception:
            pass


def maybe_launch_vision_shadow(ws: WebSocket):
    global vision_task
    global vision_last_launch

    # In Gemma mode the camera ring buffer is consumed directly by the
    # conversational model; do not spend GPU on the Qwen3-VL sidecar.
    if VOICE_LLM_BACKEND == "gemma4":
        return

    if len(vision_frames) < VISION_MIN_FRAMES:
        return

    if (
        vision_task is not None
        and not vision_task.done()
    ):
        return

    now = time.perf_counter()

    if (
        now - vision_last_launch
        < VISION_MIN_LAUNCH_INTERVAL
    ):
        return

    vision_last_launch = now

    vision_task = asyncio.create_task(
        run_vision_shadow(ws)
    )


@app.websocket("/vision-ws")
async def vision_websocket_endpoint(ws: WebSocket):
    await ws.accept()

    # Do not retain stale frames across browser reconnects.
    vision_frames.clear()

    try:
        await ws.send_text(json.dumps({
            "type": "vision_status",
            "value": "CONNECTED",
        }))

        while True:
            msg = await ws.receive()

            if msg.get("type") == "websocket.disconnect":
                break

            text_data = msg.get("text")

            if text_data is None:
                continue

            try:
                obj = json.loads(text_data)
            except Exception:
                continue

            if obj.get("type") != "camera_frame":
                continue

            jpeg = obj.get("jpeg")
            timestamp_ms = obj.get("timestamp_ms")

            if not isinstance(jpeg, str):
                continue

            # A 320x180 JPEG should be nowhere near this large.
            if len(jpeg) > 250_000:
                continue

            if not isinstance(
                timestamp_ms,
                (int, float),
            ):
                timestamp_ms = time.time() * 1000.0

            vision_frames.append((
                float(timestamp_ms),
                jpeg,
            ))

            maybe_launch_vision_shadow(ws)

    except Exception as exc:
        print(
            f"[VISION WS] closed: {exc}",
            flush=True,
        )


# ---------------------------------------------------------------------------
# WebSocket
# ---------------------------------------------------------------------------

@app.websocket("/ws")
async def websocket_endpoint(ws: WebSocket):
    await ws.accept()

    state = State(ws)

    await state.json({
        "type": "state",
        "value": "LISTENING",
    })

    await state.json({
        "type": "enrollment_status",
        "status": (
            "ready"
            if nomo and nomo.embedding is not None
            else "missing"
        ),
    })

    try:
        while True:
            msg = await ws.receive()

            if msg.get("type") == "websocket.disconnect":
                break

            if msg.get("bytes") is not None:
                data = msg["bytes"]

                for i in range(
                    0,
                    len(data),
                    FRAME_BYTES,
                ):
                    frame = data[
                        i:i + FRAME_BYTES
                    ]

                    if len(frame) == FRAME_BYTES:
                        if state.enrollment_active:
                            await process_enrollment_frame(
                                state,
                                frame,
                            )
                        else:
                            await process_audio_frame(
                                state,
                                frame,
                            )

            elif msg.get("text") is not None:
                try:
                    obj = json.loads(msg["text"])
                except Exception:
                    continue

                typ = obj.get("type")

                if typ == "enroll_start":
                    await begin_enrollment(state)

                elif typ == "config":
                    persona = (
                        obj.get("persona") or default_persona()
                    ).lower()

                    if persona in persona_ids():
                        state.persona = persona

                        await state.json({
                            "type": "log",
                            "message": (
                                f"Persona: {persona} "
                                f"({len(persona_prompt_for(persona))} chars)"
                            ),
                        })

                elif typ == "playback_state":
                    active = bool(
                        obj.get("active")
                    )

                    state.playback_active = active

                    if active:
                        # New listening turn will start after playback.
                        state.reset_turn_audio()

                        await state.json({
                            "type": "log",
                            "message": (
                                "mic gated while assistant is speaking"
                            ),
                        })

                    else:
                        # Browser AudioWorklet has actually drained.
                        # Start target-VAD from a completely clean state.
                        state.reset_turn_audio()

                        await state.json({
                            "type": "state",
                            "value": "LISTENING",
                        })

                        await state.json({
                            "type": "log",
                            "message": "SERVER MIC GATE OFF — browser playback drained",
                        })

                elif typ == "playback_started":
                    if state.active_mouth_stop is not None:
                        await timing(
                            state,
                            "playback_start",
                            time.perf_counter(),
                            state.active_mouth_stop,
                        )

    except WebSocketDisconnect:
        pass

    finally:
        if (
            state.speculative_gemma_task is not None
            and not state.speculative_gemma_task.done()
        ):
            await cancel_speculative_gemma(state)

        if (
            state.speculative_qwen_task is not None
            and not state.speculative_qwen_task.done()
        ):
            await cancel_speculative_qwen(state)

        if (
            state.generation
            and not state.generation.done()
        ):
            state.generation.cancel()

        if (
            state.partial_task
            and not state.partial_task.done()
        ):
            state.partial_task.cancel()
