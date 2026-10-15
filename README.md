# Voice Latency Lab

A single-user, microphone-driven **end-to-end speech-latency bench** for a realtime voice
assistant. Audio streams from the browser, runs through target-speaker VAD → STT → LLM → TTS,
the reply streams back as PCM, and **every stage of the round trip is timestamped** so you can
see exactly where the seconds go.

It is a research bench, not a product: one speaker, one tab, no auth, no accounts. Its value is
the instrumentation — 13 named timing points from "sound left the mouth" to "sound reached the
ear", plus a realtime-factor readout.

---

## What you get

After each turn the page shows a timing table, one row per stage:

| Stage | Measures |
|---|---|
| `mic capture` | audio timestamp in the browser |
| `client send` | time the frame batch left the browser |
| `server receive` | arrival at the WebSocket |
| `VAD candidate` | end-of-turn detected (NOMO target-speaker VAD) |
| `Parakeet final` | transcript of the turn available |
| `LLM request` | request sent to the response model |
| `LLM first token` | first token of the reply |
| `first phrase` | first sentence ready for speech synthesis |
| `S2 request` | TTS request sent (legacy label) |
| `first PCM` | first audio byte came back from TTS |
| `actual playback` | first audio reached the speaker (client-reported) |
| `end to end` | turn end → first playback |
| `realtime factor` | STT cost ÷ audio duration |

Copy-diagnostics dumps the whole set as text for pasting into a bug report or a before/after
comparison.

## Architecture

```
Browser (static/, vanilla JS)
  mic -> AudioWorklet -> 16 kHz PCM16 -> WebSocket
     ^                                        |
  playback <- 48 kHz PCM <- TTS stream <-  server.py (FastAPI, ~4.1k lines)
  latency report                               |
                                  +------------+------------+------------+
                                  v            v            v            v
                             NOMO-PVAD     Parakeet       LLM       VoxCPM2 TTS
                             (in-proc,    (sidecar,    (vLLM or     (sidecar,
                              CPU)          :3048)      hosted)       :3042)
```

The pipeline is **speculative**: as soon as a turn ends, STT, the LLM and TTS are all kicked off
against partial results, and later stages cancel and supersede earlier ones rather than waiting.
That is where most of the latency win comes from, and also why the timing table is worth reading
stage by stage.

See [ARCHITECTURE.md](ARCHITECTURE.md) for the full pipeline, cancellation semantics, and the
shadow features (emotion, vision, tone fusion) that run alongside without affecting latency.

## Requirements

**Host**
- Docker with Compose v2.
- The lab itself is CPU-only and light (a few hundred MB resident, plus torch).
- Optional NVIDIA GPU + [NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html)
  for the bundled Parakeet sidecar. It falls back to CPU (roughly 18x slower, still usable) and
  you can point `PARAKEET_URL` at any compatible STT service instead.

**Browser**
- Chrome / Edge / Firefox with microphone support.
- **HTTPS or `localhost` is mandatory** — browsers refuse `getUserMedia` on a plain-HTTP origin
  reached over the network. Put it behind a reverse proxy or tunnel for remote use.

**Models (not in this repo)**

| Model | Size | Role |
|---|---|---|
| NOMO-PVAD | ~71 MB | target-speaker VAD (only reacts to the enrolled voice) |
| Parakeet TDT 0.6b v3 (ONNX) | ~1.2 GB fp16 | speech to text |
| VoxCPM2 | — | text to speech, 48 kHz, reference-audio voices |
| A response model | — | vLLM with native audio input, or any hosted chat endpoint |

## Quick start

```sh
git clone <your-fork-url> voice-lab
cd voice-lab
cp .env.example .env
mkdir -p profiles models
cp personas.builtin.example.json profiles/personas.builtin.json
```

### Fetch the models

**NOMO-PVAD** — obtain the upstream project and place the checkout (containing the `nomo_pvad`
Python package) plus its weights at `./models/nomo-pvad`:

```
models/nomo-pvad/
├── nomo_pvad/              # the Python package the server imports
└── weights/nomo_pvad.pt    # ~71 MB
```

**Parakeet** — five files into `./models/parakeet`:

```sh
pip install -U "huggingface_hub[cli]"
huggingface-cli download rleo/parakeet-tdt-0.6b-v3-onnx \
  --include "fp16/*" --local-dir models/parakeet --repo-type model
mv models/parakeet/fp16/* models/parakeet/ && rmdir models/parakeet/fp16
huggingface-cli download istupakov/parakeet-tdt-0.6b-v3-onnx nemo128.onnx \
  --local-dir models/parakeet
```

Needed: `encoder-model.onnx` (plus its `.data` shard), `decoder_joint-model.onnx`, `config.json`,
`vocab.txt`, and `nemo128.onnx` — the feature extractor, without which the sidecar refuses to
start.

### Run

```sh
docker compose up -d --build
docker compose logs -f voice-lab      # wait for "Application startup complete"
```

Open <http://localhost:8090>, click **Enroll**, speak about ten seconds of your own voice, then
talk to it.

The first GPU transcription is slow, so the Parakeet sidecar warms itself at startup — allow a
couple of minutes before trusting a cold health check.

> **The lab starts but stays silent without TTS and an LLM.** Those two are not bundled; see the
> next section for the endpoints to point `.env` at.

## Wiring up TTS and the LLM

The lab talks to everything over HTTP. To run it against your own stack, implement:

| Endpoint | Contract |
|---|---|
| `VOXCPM_URL` | POST `{"text","voice","control","cfg_value","inference_timesteps"}` → SSE stream of base64 PCM16 at `VOXCPM_SAMPLE_RATE` |
| `VOXCPM_STABLE_URL` | same, lower-variance variant used when `STABLE_DEMO_MODE=1` |
| `VOXCPM_VOICES_URL` | POST multipart `name,consent,ref_text,audio_sample` registers a reference voice; `DELETE /{voice_id}` removes it |
| `PARAKEET_URL` | POST raw 16 kHz mono s16le body → `{"text": "..."}` |
| `GEMMA4_URL` | OpenAI-style `chat/completions` that accepts **audio content parts**, streaming tokens |
| `QWEN_URL` | OpenAI-style `chat/completions`, text only, streaming |

Two response modes, switched by `VOICE_LLM_BACKEND`:

- **`gemma4`** — speech goes to the model *raw*, skipping a text round trip. Lower latency, and
  Parakeet then only runs for the on-screen transcript and conversation history.
- **`qwen`** — Parakeet transcribes first and text goes to a hosted chat endpoint. Works with any
  OpenAI-compatible API, at the cost of one extra serial STT step per turn.

The optional shadow endpoints (emotion, events, tone, vision) all fail soft — leave them pointed
at nothing and the lab just reports them unavailable.

## Configuration

Everything is environment variables; see [.env.example](.env.example) for the annotated set.

| Variable | Default | Notes |
|---|---|---|
| `VOICE_LLM_BACKEND` | `qwen` | `qwen` or `gemma4` |
| `GEMMA4_URL` / `GEMMA4_MODEL` | `127.0.0.1:18021` | used in `gemma4` mode |
| `QWEN_URL` / `QWEN_MODEL` | `127.0.0.1:18020` | used in `qwen` mode |
| `QWEN_API_KEY_FILE` | `/secrets/qwen_api_key.txt` | only matters in `qwen` mode; missing is a warning |
| `PARAKEET_URL` | `127.0.0.1:3048/transcribe` | STT sidecar |
| `VOXCPM_URL` / `VOXCPM_STABLE_URL` | `127.0.0.1:3042` | TTS |
| `VOXCPM_VOICES_URL` | `127.0.0.1:3042/v1/audio/voices` | persona voice registration |
| `STABLE_DEMO_MODE` | `1` | shorter replies + stable TTS endpoint |
| `NOMO_REPO_PATH` | `/nomo-pvad` | NOMO checkout |
| `NOMO_PROFILE_PATH` | `/profiles/pali.npy` | enrolled speaker embedding |
| `PERSONA_DIR` | `/profiles/personas` | imported personas (persist here) |
| `BUILTIN_PERSONA_FILE` | `/profiles/personas.builtin.json` | built-in personas, as data |

The compose file bind-mounts `src/server.py` and `src/static/` read-only, so a source edit needs
only `docker compose up -d --force-recreate voice-lab` — no rebuild. Delete those two lines for a
pure image deployment.

## Personas

Personas are **data, not code** — this repo ships no characters. Copy the example and write your
own:

```sh
cp personas.builtin.example.json profiles/personas.builtin.json
```

The file maps an id to a system prompt, plus an optional TTS delivery description:

```json
{
  "echo": {
    "prompt": "\nYou are Echo. Calm, plain-spoken and direct.\n",
    "control": "Neutral, natural conversational delivery, steady pace, close-mic and dry."
  }
}
```

File order matters: the **first** entry is the persona a fresh session starts with. Restart the
lab after editing it — it is read once at startup.

You can also **import a persona preset** from the page: upload a TomoriBot-style preset JSON plus a
short reference recording of the target voice and its transcript. The lab folds the preset's
persona prompt, attributes, naming terms and **all** of its sample dialogues into a system prompt,
registers the reference audio as a VoxCPM voice, and the persona appears in the dropdown.

- Imported personas are namespaced `lab-<slug>` (both persona id and voice id) so they can never
  shadow a built-in one.
- They persist as JSON under `PERSONA_DIR` — back that with a volume or they vanish on recreate.
- Voice registration needs the TTS server's voice directory to be **writable**. If it is mounted
  read-only, registration fails with a 500 and an `Errno 30` in the TTS server log.
- Sample dialogues are prompt tokens: one real preset went from ~1.1k tokens (3 exchanges) to
  ~1.9k (all 15). The dropdown shows the assembled prompt size and the **LLM first token** row
  tells you what it costs. On vLLM, prefix caching can hide the cost after the first turn —
  compare on turn 2 or later.

## Repo layout

```
src/server.py            the entire backend
src/static/              the entire front-end (no framework, no build step)
src/Dockerfile           lab image (web stack + NOMO VAD)
parakeet-shadow/         STT sidecar: Dockerfile + server.py
personas.builtin.example.json   persona file format (the real one stays local)
docker-compose.yml       lab + Parakeet
.env.example             annotated configuration
ARCHITECTURE.md          pipeline, cancellation semantics, shadow features
```

## Troubleshooting

| Symptom | Cause |
|---|---|
| Mic button does nothing | Not on HTTPS or `localhost` |
| Persona dropdown is empty | No persona file — copy the example to `profiles/personas.builtin.json` |
| Empty replies in `qwen` mode | `QWEN_API_KEY_FILE` missing — startup prints a warning |
| Transcript appears seconds late | STT too slow for the audio length; check `realtime factor` (want well under 0.1) |
| First request after boot is very slow | Cold GPU context; the sidecar warms up at startup, wait for healthy |
| `No module named 'nomo_pvad'` | `models/nomo-pvad` missing, or not the checkout root |
| `Parakeet model is missing nemo128.onnx` | Preprocessor not downloaded (see above) |
| Persona import returns 500 | TTS voice directory is read-only |
| `AUDIO UNDERRUN` in the console | TTS cannot keep ahead of playback; raise `START_BUFFER_MS` in `static/playback-worklet.js` |
| VAD ignores you | Re-enroll — the profile is voice-specific, and quiet rooms fool it |

## Security

There is **no authentication**, and `/health` discloses the full model stack. It is built to sit
on a trusted network or behind a tunnel with access control. Speaker embeddings and imported
persona audio are personal data — keep `profiles/` out of version control (already gitignored).

## Status

Actively used research bench; expect rough edges and internals renamed in place over time. Some
identifiers still say `whisper` or `qwen` for historical reasons — see the naming notes in
[ARCHITECTURE.md](ARCHITECTURE.md).
