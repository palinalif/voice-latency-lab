# Voice Latency Lab — architecture

Source: `src/` in this repository.
Runtime: one Docker container listening on `:8090`, normally put behind HTTPS (a
reverse proxy or tunnel) because browsers only allow microphone capture on secure origins.

A single-user, microphone-driven **end-to-end speech-latency bench** for a realtime voice
assistant. It streams browser audio in, runs target-speaker VAD → STT → LLM → TTS, streams
the reply PCM back, and instruments every stage of the round trip.

`server.py` is ~4,100 lines and is the entire backend. `static/` is the whole front-end
(no framework).

---

## 1. Runtime topology

The container normally runs with `network_mode: host`, so every dependency is a loopback
service on the host (or an explicit URL via the env vars in `.env.example`):

| Service | Endpoint | Role |
|---|---|---|
| VoxCPM2 TTS | `127.0.0.1:3042/generate`, `/generate-stable` | assistant speech, 48 kHz |
| emotion2vec | `127.0.0.1:3043/analyze` | emotion "shadow" (diagnostic) |
| Vision LLM | `127.0.0.1:3045/v1/chat/completions` (`Qwen/Qwen3-VL-4B-Instruct-FP8`) | webcam "vision shadow" |
| KEV tone | `127.0.0.1:3046/v1/systemone` | tone fusion input |
| SenseVoice | `127.0.0.1:3047/analyze` | emotion/event "shadow" |
| Qwen LLM | `127.0.0.1:18020/v1/chat/completions` (`qwen3.8-27b`) | text backend ("qwen" mode) |
| Gemma 4 LLM | `127.0.0.1:18021/v1/chat/completions` (`gemma4-12b`) | native-audio backend ("gemma4" mode) |
| NOMO PVAD | in-process (`/nomo-pvad`) | target-speaker VAD |
| Parakeet STT | `127.0.0.1:3048/transcribe` (`parakeet-tdt-0.6b-v3`, fp16 on CUDA) | STT (replaced faster-whisper on 2026-10-15) |

GPU: the container reserves NVIDIA device id `1`. NOMO/torch is deliberately **CPU-only**.

## 2. HTTP + WebSocket surface

| Route | Kind | Notes |
|---|---|---|
| `/` | GET | serves `/app/static/index.html` |
| `/health` | GET | pipeline advertisement (see below) |
| `/ws` | WS | audio + control + telemetry |
| `/vision-ws` | WS | webcam JPEG frames |
| `/docs`, `/redoc`, `/openapi.json` | GET | FastAPI defaults (only `/` and `/health` documented) |

`/health`:

```json
{"ok":true,"llm_backend":"gemma4","stt":"gemma4-native-audio","model":"gemma4-12b",
 "history_stt":"parakeet-tdt-0.6b-v3","target_vad":"nomo-pvad",
 "voice_profile":"ready"}
```

The `llm_backend` field is switchable between `qwen` and `gemma4` (env `VOICE_LLM_BACKEND`,
else the file `/profiles/voice_llm_backend.txt`; default `qwen`).

### `/ws` protocol

Client → server:
- **binary**: 20 ms Int16 mono PCM frames (16 kHz, 640 bytes each; larger buffers are split).
- `{"type":"config","persona":"…"}`
- `{"type":"enroll_start"}`
- `{"type":"playback_state","active":bool}` (browser playback worklet gating)
- `{"type":"playback_started"}` (latency instrumentation)

Server → client (JSON): `state` (`LISTENING`/`GENERATING`/`SPEAKING`), `enrollment_status`
(`missing`/`recording`/`processing`/`ready`/`error`), `timing {event,ms}`, `log`,
`emotion`, `sensevoice`, `tone_fusion`, `transcript`, `assistant_partial`,
`assistant_phrase`, `candidate`, `tts_end`, `clear_audio`, `error`; binary frames are
assistant TTS PCM.

### `/vision-ws` protocol

Client sends `{"type":"camera_frame","jpeg":"<base64>","timestamp_ms":…}` (JPEG > 250 KB
rejected); server replies `{"type":"vision_status","value":"CONNECTED"}` and pushes
`vision` / `vision_error` summaries.

## 3. The turn pipeline (`process_audio_frame` → `process_target_audio_frame`)

Frame constants: 16 kHz, 20 ms (`FRAME_SAMPLES=320`, `FRAME_BYTES=640`).

1. **Target VAD (NOMO).** Audio is buffered into 160 ms chunks and scored by a per-connection
   `NomoPVADSession` with the enrolled speaker embedding (`/profiles/pali.npy`). Enter/exit/
   resume hysteresis is 2 chunks each; threshold `NOMO_THRESHOLD=0.70`. Whenever the profile
   is missing it falls back to the original `webrtcvad` (mode 3, 25-frame hangover, min 8
   speech frames).
2. **Speculative start.** On the **first negative** NOMO chunk after speech, speech is
   considered "probably done": the server launches either a speculative Parakeet STT final
   pass or (in gemma4 mode) a speculative Gemma audio generation from the raw-audio snapshot —
   *before* the second negative chunk commits the turn. A short `[transcript]`/`candidate`
   can therefore begin early.
3. **Endpoint commit.** The second negative chunk ends the target turn (`target_end_pending`,
   trailing background chunk trimmed, `last_speech_at` set to estimated mouth-stop).
4. **Speculation invalidation.** If sustained target speech resumes (≥2 positive chunks)
   before commit, the speculative Whisper/Gemma/Qwen work is cancelled — background vocals
   produce isolated false positives and must not hijack the turn.
5. **LLM.** `run_response()` builds the persona system prompt and asks Qwen
   (`qwen3.8-27b`) or Gemma 4 (raw audio + optional vision context). Prompts enforce a very
   short opening sentence, no markdown, and optional natural-language performance tags
   (`[deadpan]`, `[whisper]`, …). A `STABLE_DEMO_MODE` prompt variant targets 8–20 words.
6. **TTS streaming.** `stream_fish_text()` (legacy name; **VoxCPM2 is the active backend**)
   POSTs `{voice, text, cfg_value:1.5, inference_timesteps:20, control?}` to
   `3042/generate-stable`, expects 48 kHz, and streams PCM chunks back over `/ws` as they
   arrive (`first_pcm` timing → state `SPEAKING`). Per-persona delivery controls are
   optional and come from the persona file (`control` field).
7. **Half-duplex gating.** The server gates the mic while the assistant speaks
   (`playback_active`), and the browser confirms when the playback worklet has drained
   (`playback_state`), preventing speaker leakage from being treated as barge-in.

### Latency instrumentation

`timing()` emits events the UI tabulates: `vad_candidate`, `whisper_final`, `qwen_request`,
`qwen_first_token`, `first_phrase`, `s2_request`, `first_pcm`, `playback_start` — all
measured from the estimated mouth-stop.

## 4. Diagnostic "shadow" sidecars

These run alongside the turn and **never block** the voice path; each discards results that
arrive after their turn is invalidated:

- **Emotion shadow** → emotion2vec (`3043`) on the turn audio (0.25 s timeout).
- **SenseVoice shadow** → SenseVoice (`3047`) emotion + acoustic events.
- **Tone fusion shadow** → joins Whisper + emotion2vec + SenseVoice (+ KEV `3046`) into a
  fused tone label shown in the UI.
- **Vision shadow** → browser webcam JPEGs (~10 fps, 20–200 frames, ≥1.5 s between launches)
  summarized by `Qwen3-VL-4B`; context is **not** yet consumed by the main reply path.

## 5. STT engines and history

As of **2026-10-15**, STT is the local **`parakeet-shadow`** sidecar
(`parakeet-tdt-0.6b-v3`, fp16 ONNX, `CUDAExecutionProvider`, port `3048`), reached over HTTP
by `ParakeetEngine` in `server.py`. It replaced faster-whisper entirely:

| Old | New |
|---|---|
| `WhisperEngine` (`large-v3-turbo`, GPU) — live transcript | `ParakeetEngine` |
| `HistoryWhisperEngine` (`small.en`, CPU int8) — history | `ParakeetEngine` |

The old `WhisperEngine` / `HistoryWhisperEngine` classes, the `faster-whisper` import, the
`WHISPER_MODEL` constants and the superseded `finalize_gemma_history` helpers were **removed
on 2026-10-15**; the engine globals are now `stt_engine` / `history_stt_engine`, both
`ParakeetEngine`. `faster-whisper` was dropped from `requirements.txt` (the running image
still has the package installed until the next image rebuild, but nothing imports it). The
legacy `speculative_whisper_*` state names and the `whisper_final` timing key were left as-is
to avoid churn; the UI label now reads "Parakeet final".

Measured RTF on a 3.7 s clip: int8 CPU `0.14` → fp16 CPU `0.054` → **fp16 CUDA warm `0.0038`**
(~263× real-time). The first GPU call costs ~36 s of CUDA/PTX/cuDNN init, which the sidecar
now absorbs at startup with a dummy warm-up.

**Transcript timing.** The user-turn transcription is started at the top of `run_response`
(`transcribe_turn_history`) **in parallel with generation**, so the browser transcript appears
within ~10–20 ms instead of waiting for the whole spoken reply to finish streaming. History is
still appended at the end for the next request's prompt. Streaming partials remain disabled
(`ENABLE_PARTIALS=False`).

## 6. Enrollment

Single-profile flow: `enroll_start` records ~8–12 s of audio, extracts a speaker embedding,
and writes `/profiles/pali.npy` (override with `NOMO_PROFILE_PATH`). That embedding drives
NOMO target-speaker VAD.

### Persona registry

Two sources, merged at startup by `load_builtin_personas()` then `load_personas()`:

| Source | Location | Shape |
|---|---|---|
| Built-in | `BUILTIN_PERSONA_FILE`, default `/profiles/personas.builtin.json` | `{id: {prompt, control?}}` — file order decides the default persona |
| Imported | `PERSONA_DIR/*.json`, default `/profiles/personas/` | one record per persona, written by the import UI |

`persona_prompt_for()` and `voice_id_for()` resolve an id against both sets. The built-in file is
deliberately outside `PERSONA_DIR` (which is scanned for imports) and out of git: character work
ships as data, and `personas.builtin.example.json` shows the format. With neither present the lab
still starts — empty dropdown, one warning line.

## 7. Front-end (`static/`)

Vanilla JS, no build step:

| File | Role |
|---|---|
| `index.html` | the `Voice Latency Lab` page (personas, controls, panels, timing table) |
| `app.js` | mic capture, WS plumbing, playback, enrollment, WAV test, diagnostics copy |
| `capture-worklet.js` | AudioWorklet: accumulates Int16 320-sample frames → `/ws` |
| `playback-worklet.js` | AudioWorklet: 48 kHz PCM queue with 80 ms start buffer, 500 ms rebuffer, fade-in |

Controls: persona select (populated at runtime from `GET /personas`), Start/Stop
mic, Enroll my voice, Stream test WAV, Copy diagnostics. Panels: transcript, emotion2vec,
SenseVoice, tone fusion, vision shadow (320×180 canvas), assistant text, event log.

## 8. Build

| File | Contents |
|---|---|
| `Dockerfile` | `nvidia/cuda:12.8.1-cudnn-runtime-ubuntu22.04`, venv, `requirements.txt`, `COPY server.py` + `static`, CMD `uvicorn server:app --port 8090` |
| `Dockerfile.nomo` | `FROM voice-latency-lab:whisper`; adds CPU-only torch/torchaudio + `numpy/soundfile/librosa/modelscope[audio]` for NOMO. This produces the `:nomo` image. |
| `requirements.txt` | `fastapi`, `uvicorn[standard]`, `aiohttp`, `numpy`, `webrtcvad-wheels` |

The reference deployment runs a prebuilt image with `pull_policy: never` and bind-mounts
`server.py` and `static/` read-only (so a container recreate picks up source edits), plus the
`nomo-pvad/` model (71 MB), a rw `profiles/` volume, and any secrets.

To rebuild after source changes:
```sh
docker compose build voice-lab
docker compose up -d voice-lab
```

Or without compose:
```sh
docker build -t voice-latency-lab:latest src/
docker run --rm -p 8090:8090 --env-file .env voice-latency-lab:latest
```

## 9. Source history

The tree carries the working history of the build: 45 `server.py.pre-*` snapshots and the
front-end `.pre-*` variants. Their names map the development timeline, e.g. `pre-enroll`,
`pre-nomo`, `pre-target-vad`, `pre-wav-test`, `pre-diagnostics`, `pre-emotion-ui`,
`pre-emotion-shadow`, `pre-voxcpm*`, `pre-vision-shadow`, `pre-vision-context`,
`pre-tone-fusion`, `pre-gemma4`, `pre-history-whisper`, `pre-nonblocking-history`,
`pre-gemma-cancel-fix`, `pre-cancel-recovery`, `known-good-single-shot-2026-09-27`.

Notable naming quirks: the TTS streaming function is still `stream_fish_text()` and the UI
timing label is still `s2_request`, though the live backend is VoxCPM2; `QWEN_URL`/`QWEN_MODEL`
remain the "qwen" branch even when Gemma 4 is selected.

## 10. Ops notes

- Reference container: `network_mode: host`, one NVIDIA device, port `8090`,
  `restart: unless-stopped`.
- **No auth**: the page and `/health` are unauthenticated, and `/health` discloses the whole
  model stack. Keep it on a trusted network or behind access control.
- The `nomo-pvad/` model is mounted read-only at `/nomo-pvad` (`NOMO_REPO_PATH`); it is not
  in this repo (see README for how to fetch it).
