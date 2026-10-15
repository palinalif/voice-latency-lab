const startButton = document.querySelector("#start");
const stopButton = document.querySelector("#stop");
const persona = document.querySelector("#persona");
const enrollButton = document.querySelector("#enroll");
const voiceProfileEl = document.querySelector("#voice_profile");
const testWavButton = document.querySelector("#test_wav");
const testWavInput = document.querySelector("#test_wav_input");
const copyDiagnosticsButton =
    document.querySelector("#copy_diagnostics");

const stateEl = document.querySelector("#state");
const transcriptEl = document.querySelector("#transcript");
const emotionEl = document.querySelector("#emotion");
const senseVoiceEl = document.querySelector("#sensevoice");
const toneFusionEl = document.querySelector("#tone_fusion");
const visionSummaryEl =
    document.querySelector("#vision_summary");
const visionPreviewEl =
    document.querySelector("#vision_preview");
const assistantEl = document.querySelector("#assistant");
const logEl = document.querySelector("#log");

let ws = null;
let latestEmotion = null;
let latestSenseVoice = null;
let latestToneFusion = null;


function renderEmotion() {
    if (!latestEmotion) {
        emotionEl.textContent = "";
        return;
    }

    const raw = String(
        latestEmotion.emotion || "unknown"
    );

    const label =
        raw.charAt(0).toUpperCase()
        + raw.slice(1);

    const confidence = Math.round(
        Number(latestEmotion.confidence || 0) * 100
    );

    emotionEl.textContent =
        `emotion2vec: ${label} · ${confidence}%`;
}


function renderSenseVoice() {
    if (!senseVoiceEl) {
        return;
    }

    if (!latestSenseVoice) {
        senseVoiceEl.textContent = "";
        return;
    }

    const raw = String(
        latestSenseVoice.emotion || "unknown"
    );

    const label =
        raw.charAt(0).toUpperCase()
        + raw.slice(1);

    const events = Array.isArray(
        latestSenseVoice.events
    )
        ? latestSenseVoice.events
        : [];

    senseVoiceEl.textContent =
        `SenseVoice: ${label}`
        + (
            events.length
                ? ` · ${events.join(", ")}`
                : ""
        );
}


function renderToneFusion() {
    if (!toneFusionEl) {
        return;
    }

    if (!latestToneFusion) {
        toneFusionEl.textContent = "";
        return;
    }

    const choice = String(
        latestToneFusion.choice || "unknown"
    );

    const label =
        choice.charAt(0).toUpperCase()
        + choice.slice(1);

    const probabilities =
        latestToneFusion.probabilities || {};

    const ordered = Object.entries(
        probabilities
    )
        .sort((a, b) => Number(b[1]) - Number(a[1]))
        .slice(0, 3);

    const top = ordered.map(
        ([name, value]) =>
            `${name} ${Math.round(Number(value) * 100)}%`
    ).join(" · ");

    const selected =
        Number(probabilities[choice] || 0);

    toneFusionEl.textContent =
        `Kev fusion: ${label} · `
        + `${Math.round(selected * 100)}%`
        + (top ? ` · ${top}` : "");
}


let micStream = null;
let captureContext = null;
let captureNode = null;

let playbackContext = null;
let playbackNode = null;

let playbackActive = false;

let injectingTestAudio = false;
let testAudioRunId = 0;

const TEST_SAMPLE_RATE = 16000;
const TEST_FRAME_MS = 20;
const TEST_FRAME_SAMPLES = 320;



// ---------------------------------------------------------------------------
// VISION SHADOW CAMERA
// ---------------------------------------------------------------------------

let visionWs = null;
let visionStream = null;
let visionVideo = null;
let visionTimer = null;
let visionEncoding = false;
let visionRunning = false;

const VISION_WIDTH = 320;
const VISION_HEIGHT = 180;
const VISION_FPS = 10;
const VISION_JPEG_QUALITY = 0.72;


function setVisionSummary(text) {
    if (visionSummaryEl) {
        visionSummaryEl.textContent = text;
    }
}


function connectVisionSocket() {
    if (
        visionWs
        && (
            visionWs.readyState === WebSocket.OPEN
            || visionWs.readyState
                === WebSocket.CONNECTING
        )
    ) {
        return;
    }

    const scheme =
        window.location.protocol === "https:"
            ? "wss"
            : "ws";

    visionWs = new WebSocket(
        `${scheme}://${window.location.host}/vision-ws`
    );

    visionWs.onopen = () => {
        log("vision websocket connected");
    };

    visionWs.onclose = () => {
        log("vision websocket disconnected");
    };

    visionWs.onerror = () => {
        log("vision websocket error");
    };

    visionWs.onmessage = event => {
        let msg;

        try {
            msg = JSON.parse(event.data);
        }

        catch {
            return;
        }

        if (msg.type === "vision") {
            const seconds = (
                Number(msg.inference_ms || 0)
                / 1000
            ).toFixed(2);

            const frames =
                Number(msg.frames || 0);

            const windowSec =
                Number(msg.window_sec || 0)
                    .toFixed(1);

            const tokens =
                msg.prompt_tokens ?? "?";

            setVisionSummary(
                `Vision: ${msg.description} `
                + `· ${frames}f/${windowSec}s `
                + `· ${seconds}s `
                + `· ${tokens} tok`
            );
        }

        else if (msg.type === "vision_error") {
            setVisionSummary(
                `Vision error: ${msg.message}`
            );
        }
    };
}


function drawVisionFrame() {
    if (
        !visionRunning
        || !visionVideo
        || visionVideo.readyState < 2
        || !visionPreviewEl
    ) {
        return;
    }

    const ctx = visionPreviewEl.getContext(
        "2d",
        { alpha: false }
    );

    const sw = visionVideo.videoWidth;
    const sh = visionVideo.videoHeight;

    if (!sw || !sh) {
        return;
    }

    const targetAspect =
        VISION_WIDTH / VISION_HEIGHT;

    const sourceAspect = sw / sh;

    let sx = 0;
    let sy = 0;
    let cropW = sw;
    let cropH = sh;

    // Center-crop rather than distort non-16:9 webcams.
    if (sourceAspect > targetAspect) {
        cropW = sh * targetAspect;
        sx = (sw - cropW) / 2;
    }

    else if (sourceAspect < targetAspect) {
        cropH = sw / targetAspect;
        sy = (sh - cropH) / 2;
    }

    ctx.drawImage(
        visionVideo,
        sx,
        sy,
        cropW,
        cropH,
        0,
        0,
        VISION_WIDTH,
        VISION_HEIGHT
    );
}


function sendVisionFrame() {
    drawVisionFrame();

    if (
        visionEncoding
        || !visionWs
        || visionWs.readyState !== WebSocket.OPEN
        || !visionPreviewEl
    ) {
        return;
    }

    visionEncoding = true;

    visionPreviewEl.toBlob(
        blob => {
            if (!blob) {
                visionEncoding = false;
                return;
            }

            const reader = new FileReader();

            reader.onloadend = () => {
                try {
                    if (
                        visionWs
                        && visionWs.readyState
                            === WebSocket.OPEN
                    ) {
                        const result = String(
                            reader.result || ""
                        );

                        const comma =
                            result.indexOf(",");

                        if (comma >= 0) {
                            visionWs.send(
                                JSON.stringify({
                                    type: "camera_frame",
                                    timestamp_ms: Date.now(),
                                    jpeg: result.slice(
                                        comma + 1
                                    )
                                })
                            );
                        }
                    }
                }

                finally {
                    visionEncoding = false;
                }
            };

            reader.onerror = () => {
                visionEncoding = false;
            };

            reader.readAsDataURL(blob);
        },
        "image/jpeg",
        VISION_JPEG_QUALITY
    );
}


async function startVisionCamera() {
    if (visionRunning) {
        return;
    }

    visionRunning = true;

    setVisionSummary(
        "Vision: requesting camera…"
    );

    connectVisionSocket();

    try {
        visionStream =
            await navigator.mediaDevices.getUserMedia({
                video: {
                    width: {
                        ideal: 640
                    },
                    height: {
                        ideal: 360
                    },
                    frameRate: {
                        ideal: 10,
                        max: 15
                    }
                },
                audio: false
            });

        visionVideo =
            document.createElement("video");

        visionVideo.autoplay = true;
        visionVideo.muted = true;
        visionVideo.playsInline = true;
        visionVideo.srcObject = visionStream;

        await visionVideo.play();

        setVisionSummary(
            "Vision: warming rolling buffer…"
        );

        visionTimer = setInterval(
            sendVisionFrame,
            1000 / VISION_FPS
        );

        log(
            "vision camera started "
            + `${VISION_WIDTH}×${VISION_HEIGHT} `
            + `@ ${VISION_FPS} FPS`
        );
    }

    catch (err) {
        visionRunning = false;

        setVisionSummary(
            `Vision camera error: ${err}`
        );

        log(
            `vision camera failed: ${err}`
        );
    }
}


function stopVisionCamera() {
    visionRunning = false;
    visionEncoding = false;

    if (visionTimer) {
        clearInterval(visionTimer);
        visionTimer = null;
    }

    if (visionStream) {
        for (
            const track
            of visionStream.getTracks()
        ) {
            track.stop();
        }

        visionStream = null;
    }

    if (visionVideo) {
        visionVideo.srcObject = null;
        visionVideo = null;
    }

    if (visionWs) {
        try {
            visionWs.close();
        }

        catch {
        }

        visionWs = null;
    }

    setVisionSummary(
        "Camera inactive"
    );

    log("vision camera stopped");
}


startButton.addEventListener(
    "click",
    startVisionCamera
);

stopButton.addEventListener(
    "click",
    stopVisionCamera
);



function log(text) {
    logEl.textContent =
        `${new Date().toLocaleTimeString()}  ${text}\n`
        + logEl.textContent;
}


function setState(value) {
    stateEl.textContent = value;
}


function setPlaybackActive(active) {
    if (playbackActive === active) {
        return;
    }

    playbackActive = active;

    if (
        ws
        && ws.readyState === WebSocket.OPEN
    ) {
        ws.send(JSON.stringify({
            type: "playback_state",
            active
        }));
    }

    log(
        active
            ? "assistant speaking — mic gated"
            : "assistant finished — mic live"
    );
}


function clearPlayback() {
    if (playbackNode) {
        playbackNode.port.postMessage({
            type: "clear"
        });
    }

    setPlaybackActive(false);

    log("audio queue cleared");
}


function resetTimings() {
    for (const id of [
        "vad_candidate",
        "whisper_final",
        "llm_request",
        "llm_first_token",
        "first_phrase",
        "s2_request",
        "first_pcm",
        "playback_start"
    ]) {
        const el = document.querySelector(`#${id}`);

        if (el) {
            el.textContent = "—";
        }
    }
}


function handleJSON(msg) {
    if (msg.type === "state") {
        setState(msg.value);

        if (msg.value === "LISTENING") {
            latestEmotion = null;
            renderEmotion();
        }

        if (
            msg.value === "LISTENING"
            && !injectingTestAudio
            && !playbackActive
        ) {
            testWavButton.disabled = false;
        }

        else if (
            msg.value === "GENERATING"
            || msg.value === "SPEAKING"
        ) {
            testWavButton.disabled = true;
        }

        // Gate mic as soon as PCM begins arriving from Fish.
        if (msg.value === "SPEAKING") {
            setPlaybackActive(true);
        }
    }

    else if (msg.type === "enrollment_status") {
        if (msg.status === "missing") {
            voiceProfileEl.textContent =
                "Voice profile: missing";

            // Mic lifecycle controls whether enrollment is clickable.
        }

        else if (msg.status === "recording") {
            voiceProfileEl.textContent =
                `Voice profile: recording ${msg.duration_sec}s…`;

            enrollButton.disabled = true;

            log("voice enrollment recording started");
        }

        else if (msg.status === "processing") {
            voiceProfileEl.textContent =
                "Voice profile: processing…";

            enrollButton.disabled = true;

            log("extracting speaker embedding");
        }

        else if (msg.status === "ready") {
            voiceProfileEl.textContent =
                "Voice profile: ready";

            // Mic lifecycle controls whether enrollment is clickable.

            log("voice profile ready");
        }

        else if (msg.status === "error") {
            voiceProfileEl.textContent =
                "Voice profile: error";

            // Mic lifecycle controls whether enrollment is clickable.

            log(
                `ENROLLMENT ERROR: `
                + `${msg.message || "unknown error"}`
            );
        }
    }

    else if (msg.type === "transcript") {
        transcriptEl.textContent = msg.text;
        renderEmotion();
    }

    else if (msg.type === "emotion") {
        latestEmotion = msg;
        renderEmotion();
    }

    else if (msg.type === "sensevoice") {
        latestSenseVoice = msg;
        renderSenseVoice();
    }

    else if (msg.type === "tone_fusion") {
        latestToneFusion = msg;
        renderToneFusion();
    }

    else if (msg.type === "candidate") {
        log(`candidate: ${msg.text}`);
    }

    else if (msg.type === "assistant_partial") {
        assistantEl.textContent = msg.text;
    }

    else if (msg.type === "assistant_phrase") {
        assistantEl.textContent = msg.text;

        log(`phrase: ${msg.text}`);
    }

    else if (msg.type === "timing") {
        if (msg.event === "vad_candidate") {
            resetTimings();
        }

        const aliases = {
            gemma_request: "llm_request",
            qwen_request: "llm_request",
            gemma_first_token: "llm_first_token",
            qwen_first_token: "llm_first_token"
        };

        const el = document.querySelector(
            `#${aliases[msg.event] || msg.event}`
        );

        if (el) {
            el.textContent =
                `${msg.ms.toFixed(1)} ms`;
        }

        log(
            `${msg.event}: ${msg.ms.toFixed(1)} ms`
        );
    }

    else if (msg.type === "tts_end") {
        if (playbackNode) {
            playbackNode.port.postMessage({
                type: "end"
            });
        }
    }

    else if (msg.type === "clear_audio") {
        clearPlayback();
    }

    else if (msg.type === "log") {
        log(msg.message);
    }

    else if (msg.type === "error") {
        log(`ERROR: ${msg.message}`);

        console.error(msg.message);
    }
}


async function openSocket() {
    const scheme =
        location.protocol === "https:"
            ? "wss"
            : "ws";

    const url =
        `${scheme}://${location.host}/ws`;

    log(`connecting websocket: ${url}`);

    ws = new WebSocket(url);

    ws.binaryType = "arraybuffer";

    await new Promise((resolve, reject) => {
        ws.onopen = () => {
            log("websocket connected");
            resolve();
        };

        ws.onerror = () => {
            reject(
                new Error(
                    `WebSocket connection failed: ${url}`
                )
            );
        };
    });

    ws.onmessage = event => {
        if (typeof event.data === "string") {
            handleJSON(
                JSON.parse(event.data)
            );
            return;
        }

        if (playbackNode) {
            playbackNode.port.postMessage(
                {
                    type: "pcm",
                    buffer: event.data
                },
                [event.data]
            );
        }
    };

    ws.onclose = event => {
        setState("OFFLINE");

        log(
            `websocket closed: `
            + `code=${event.code} `
            + `reason=${event.reason || "(none)"}`
        );
    };

    ws.send(JSON.stringify({
        type: "config",
        persona: persona.value
    }));
}


async function setupPlayback() {
    playbackContext = new AudioContext({
        sampleRate: 48000
    });

    await playbackContext.resume();

    if (playbackContext.sampleRate !== 48000) {
        throw new Error(
            `Playback context is `
            + `${playbackContext.sampleRate} Hz; `
            + `expected 48000 Hz`
        );
    }

    await playbackContext.audioWorklet.addModule(
        "/static/playback-worklet.js?v=ring-v17-voxcpm"
    );

    playbackNode = new AudioWorkletNode(
        playbackContext,
        "pcm-playback-processor",
        {
            numberOfInputs: 0,
            numberOfOutputs: 1,
            outputChannelCount: [1]
        }
    );

    playbackNode.connect(
        playbackContext.destination
    );

    playbackNode.port.onmessage = event => {
        const msg = event.data;

        if (msg.type === "started") {
            log(
                `playback started `
                + `(${msg.bufferedMs.toFixed(1)} ms buffered)`
            );

            if (
                ws
                && ws.readyState === WebSocket.OPEN
            ) {
                ws.send(JSON.stringify({
                    type: "playback_started"
                }));
            }
        }

        else if (msg.type === "underrun") {
            log(
                "AUDIO UNDERRUN — rebuffering"
            );
        }

        else if (msg.type === "rebuffered") {
            log(
                `playback recovered `
                + `(${msg.bufferedMs.toFixed(1)} ms buffered)`
            );
        }

        else if (msg.type === "ended") {
            setPlaybackActive(false);
        }
    };

    log(
        `playback: ${playbackContext.sampleRate} Hz`
    );
}


async function setupCapture() {
    micStream =
        await navigator.mediaDevices.getUserMedia({
            audio: {
                channelCount: 1,
                echoCancellation: true,
                noiseSuppression: true,
                autoGainControl: true
            }
        });

    captureContext = new AudioContext({
        sampleRate: 16000
    });

    await captureContext.audioWorklet.addModule(
        "/static/capture-worklet.js?v=ring-v4"
    );

    if (captureContext.sampleRate !== 16000) {
        throw new Error(
            `Capture context is `
            + `${captureContext.sampleRate} Hz; `
            + `expected 16000 Hz`
        );
    }

    const source =
        captureContext.createMediaStreamSource(
            micStream
        );

    captureNode = new AudioWorkletNode(
        captureContext,
        "capture-processor"
    );

    captureNode.port.onmessage = event => {
        if (injectingTestAudio) {
            return;
        }

        if (
            ws
            && ws.readyState === WebSocket.OPEN
        ) {
            ws.send(event.data);
        }
    };

    // Keep worklet processing without monitoring mic locally.
    const mute =
        captureContext.createGain();

    mute.gain.value = 0;

    source.connect(captureNode);
    captureNode.connect(mute);
    mute.connect(captureContext.destination);

    log(
        `capture: ${captureContext.sampleRate} Hz`
    );
}



function sleep(ms) {
    return new Promise(resolve =>
        setTimeout(resolve, ms)
    );
}


async function ensureTestSession() {
    if (!playbackContext) {
        await setupPlayback();
    }

    if (
        !ws
        || ws.readyState !== WebSocket.OPEN
    ) {
        await openSocket();
    }

    startButton.disabled = true;
    stopButton.disabled = false;

    setState("LISTENING");
}


async function decodeTestWav(file) {
    const bytes = await file.arrayBuffer();

    // Decode at the file's native rate first.
    const decodeContext = new AudioContext();

    let decoded;

    try {
        decoded = await decodeContext.decodeAudioData(
            bytes.slice(0)
        );
    }
    finally {
        await decodeContext.close();
    }

    // Resample/downmix to exactly the same format as the mic:
    // mono float32 @ 16 kHz.
    const frames = Math.ceil(
        decoded.duration * TEST_SAMPLE_RATE
    );

    const offline = new OfflineAudioContext(
        1,
        frames,
        TEST_SAMPLE_RATE
    );

    const source = offline.createBufferSource();

    source.buffer = decoded;
    source.connect(offline.destination);
    source.start();

    const rendered =
        await offline.startRendering();

    return rendered.getChannelData(0);
}


function makePcm16Frame(samples, offset) {
    const buffer = new ArrayBuffer(
        TEST_FRAME_SAMPLES * 2
    );

    const view = new DataView(buffer);

    for (
        let i = 0;
        i < TEST_FRAME_SAMPLES;
        ++i
    ) {
        let x = 0;

        const index = offset + i;

        if (samples && index < samples.length) {
            x = samples[index];
        }

        x = Math.max(-1, Math.min(1, x));

        const value =
            x < 0
                ? Math.round(x * 32768)
                : Math.round(x * 32767);

        view.setInt16(
            i * 2,
            value,
            true
        );
    }

    return buffer;
}


async function streamTestWav(file) {
    await ensureTestSession();

    if (playbackActive) {
        throw new Error(
            "Wait for assistant playback to finish first"
        );
    }

    const samples =
        await decodeTestWav(file);

    const duration =
        samples.length / TEST_SAMPLE_RATE;

    log(
        `test WAV: ${file.name} `
        + `(${duration.toFixed(2)} s)`
    );

    injectingTestAudio = true;
    testWavButton.disabled = true;

    const runId = ++testAudioRunId;

    // 500 ms before/after gives NOMO a clean negative region.
    const silenceFrames = 25;

    let nextFrameAt = performance.now();

    async function sendFrame(buffer) {
        if (runId !== testAudioRunId) {
            return false;
        }

        if (
            !ws
            || ws.readyState !== WebSocket.OPEN
        ) {
            return false;
        }

        const now = performance.now();
        const delay = nextFrameAt - now;

        if (delay > 0) {
            await sleep(delay);
        }

        if (runId !== testAudioRunId) {
            return false;
        }

        ws.send(buffer);

        nextFrameAt += TEST_FRAME_MS;

        // Don't burst dozens of frames if the browser was suspended.
        if (
            performance.now() - nextFrameAt
            > 100
        ) {
            nextFrameAt = performance.now();
        }

        return true;
    }

    try {
        log(
            "streaming test WAV as 16 kHz mic input"
        );

        const silence = makePcm16Frame(
            null,
            0
        );

        for (
            let i = 0;
            i < silenceFrames;
            ++i
        ) {
            if (!await sendFrame(silence)) {
                return;
            }
        }

        for (
            let offset = 0;
            offset < samples.length;
            offset += TEST_FRAME_SAMPLES
        ) {
            const frame = makePcm16Frame(
                samples,
                offset
            );

            if (!await sendFrame(frame)) {
                return;
            }
        }

        for (
            let i = 0;
            i < silenceFrames;
            ++i
        ) {
            if (!await sendFrame(silence)) {
                return;
            }
        }

        log("test WAV stream finished");
    }
    finally {
        injectingTestAudio = false;

        testWavInput.value = "";

        // If NOMO didn't trigger at all, don't leave the button
        // disabled forever. Normal successful runs will instead
        // re-enable it when the server returns to LISTENING.
        setTimeout(() => {
            if (
                !injectingTestAudio
                && !playbackActive
                && stateEl.textContent === "LISTENING"
            ) {
                testWavButton.disabled = false;
            }
        }, 1000);
    }
}



async function copyDiagnostics() {
    const timingIds = [
        "vad_candidate",
        "whisper_final",
        "llm_request",
        "llm_first_token",
        "first_phrase",
        "s2_request",
        "first_pcm",
        "playback_start"
    ];

    const timingLabels = {
        vad_candidate: "VAD candidate",
        whisper_final: "Parakeet final",
        llm_request: "LLM request",
        llm_first_token: "LLM first token",
        first_phrase: "First phrase",
        s2_request: "S2 request",
        first_pcm: "First PCM",
        playback_start: "Actual playback"
    };

    const timings = timingIds.map(id => {
        const value =
            document.querySelector(`#${id}`)
                ?.textContent
            || "—";

        return `${timingLabels[id]}: ${value}`;
    });

    let health = "(unavailable)";

    try {
        const response = await fetch(
            "/health",
            {
                cache: "no-store"
            }
        );

        if (response.ok) {
            health = JSON.stringify(
                await response.json(),
                null,
                2
            );
        }
        else {
            health =
                `HTTP ${response.status}`;
        }
    }
    catch (err) {
        health = String(err);
    }

    const text = [
        "=== Voice Latency Lab diagnostics ===",
        `Captured: ${new Date().toISOString()}`,
        `URL: ${location.href}`,
        "",
        "=== Session ===",
        `State: ${stateEl.textContent}`,
        `Persona: ${persona.value}`,
        `Voice profile: ${voiceProfileEl.textContent}`,
        `Playback active: ${playbackActive}`,
        `Test WAV injecting: ${injectingTestAudio}`,
        "",
        "=== Audio ===",
        `Capture context: ${
            captureContext
                ? `${captureContext.sampleRate} Hz`
                : "not running"
        }`,
        `Playback context: ${
            playbackContext
                ? `${playbackContext.sampleRate} Hz`
                : "not running"
        }`,
        "",
        "=== Timing from last speech frame ===",
        ...timings,
        "",
        "=== Transcript ===",
        transcriptEl.textContent || "(empty)",
        "",
        "=== Assistant ===",
        assistantEl.textContent || "(empty)",
        "",
        "=== Server health ===",
        health,
        "",
        "=== Browser ===",
        navigator.userAgent,
        "",
        "=== Events ===",
        logEl.textContent || "(empty)"
    ].join("\n");

    try {
        await navigator.clipboard.writeText(text);
    }
    catch {
        // Fallback for browsers that decide clipboard APIs
        // are far too exciting to permit today.
        const area = document.createElement("textarea");

        area.value = text;
        area.style.position = "fixed";
        area.style.opacity = "0";

        document.body.appendChild(area);

        area.focus();
        area.select();

        document.execCommand("copy");

        area.remove();
    }

    const oldText =
        copyDiagnosticsButton.textContent;

    copyDiagnosticsButton.textContent =
        "Copied!";

    log("diagnostics copied to clipboard");

    setTimeout(() => {
        copyDiagnosticsButton.textContent =
            oldText;
    }, 1200);
}


async function start() {
    startButton.disabled = true;

    await setupPlayback();
    await openSocket();
    await setupCapture();

    setState("LISTENING");

    stopButton.disabled = false;
    enrollButton.disabled = false;
}


async function stop() {
    stopButton.disabled = true;
    enrollButton.disabled = true;
    testWavButton.disabled = true;

    ++testAudioRunId;
    injectingTestAudio = false;

    clearPlayback();

    if (ws) {
        ws.close();
        ws = null;
    }

    if (micStream) {
        for (
            const track
            of micStream.getTracks()
        ) {
            track.stop();
        }
    }

    if (captureContext) {
        await captureContext.close();
    }

    if (playbackContext) {
        await playbackContext.close();
    }

    micStream = null;
    captureContext = null;
    captureNode = null;

    playbackContext = null;
    playbackNode = null;

    setState("OFFLINE");

    startButton.disabled = false;
}


startButton.onclick = () =>
    start().catch(err => {
        console.error(err);

        log(`START ERROR: ${err}`);

        startButton.disabled = false;
    });


stopButton.onclick = stop;


copyDiagnosticsButton.onclick = () => {
    copyDiagnostics().catch(err => {
        console.error(err);
        log(`COPY ERROR: ${err}`);
    });
};


testWavButton.onclick = () => {
    if (playbackActive) {
        log(
            "wait for assistant playback to finish"
        );
        return;
    }

    testWavInput.click();
};


testWavInput.onchange = () => {
    const file = testWavInput.files?.[0];

    if (!file) {
        return;
    }

    streamTestWav(file).catch(err => {
        console.error(err);

        injectingTestAudio = false;
        testWavButton.disabled = false;

        log(`TEST WAV ERROR: ${err}`);
    });
};


enrollButton.onclick = () => {
    if (
        !ws
        || ws.readyState !== WebSocket.OPEN
    ) {
        return;
    }

    enrollButton.disabled = true;

    ws.send(JSON.stringify({
        type: "enroll_start"
    }));
};


persona.onchange = () => {
    if (
        ws
        && ws.readyState === WebSocket.OPEN
    ) {
        ws.send(JSON.stringify({
            type: "config",
            persona: persona.value
        }));
    }
};



// ---------------------------------------------------------------------------
// Persona registry: dynamic dropdown + TomoriBot preset/voice import
// ---------------------------------------------------------------------------

const personaJsonInput = document.querySelector("#persona_json");
const personaVoiceInput = document.querySelector("#persona_voice");
const personaTextInput = document.querySelector("#persona_text");
const personaImportButton =
    document.querySelector("#persona_import_btn");
const personaDeleteButton =
    document.querySelector("#persona_delete_btn");
const personaStatusEl =
    document.querySelector("#persona_import_status");
const personaInfoEl = document.querySelector("#persona_info");

let knownPersonas = [];

function personaStatus(text, tone) {
    personaStatusEl.textContent = text || "";

    personaStatusEl.style.color =
        tone === "error"
            ? "#ff7b72"
            : tone === "ok"
                ? "#7ee787"
                : "";
}

function syncPersonaUi() {
    const rec = knownPersonas.find(
        (p) => p.id === persona.value
    );

    personaDeleteButton.disabled = !rec || rec.builtin;

    const chars = rec && rec.prompt_chars;

    personaInfoEl.textContent = chars
        ? `${chars.toLocaleString()} chars \u2248 ${Math.round(chars / 4).toLocaleString()} tokens`
        : "";
}

async function loadPersonas() {
    try {
        const resp = await fetch("/personas");

        if (!resp.ok) return;

        const data = await resp.json();
        const list = data.personas || [];

        if (!list.length) return;

        const previous = persona.value;

        knownPersonas = list;
        persona.innerHTML = "";

        for (const p of list) {
            const opt = document.createElement("option");

            opt.value = p.id;
            opt.textContent = p.builtin
                ? p.label
                : `${p.label} \u2605`;

            persona.appendChild(opt);
        }

        if (list.some((p) => p.id === previous)) {
            persona.value = previous;
        }

        syncPersonaUi();
    } catch (err) {
        // Older server or offline: keep the built-in options in the markup.
    }
}

function fileToBase64(file) {
    return file.arrayBuffer().then((buf) => {
        const bytes = new Uint8Array(buf);
        const chunk = 0x8000;
        let binary = "";

        for (let i = 0; i < bytes.length; i += chunk) {
            binary += String.fromCharCode(
                ...bytes.subarray(i, i + chunk)
            );
        }

        return btoa(binary);
    });
}

personaImportButton.onclick = async () => {
    const presetFile = personaJsonInput.files[0];
    const voiceFile = personaVoiceInput.files[0];

    if (!presetFile || !voiceFile) {
        personaStatus(
            "Pick both a preset JSON and a voice sample.",
            "error"
        );
        return;
    }

    personaImportButton.disabled = true;
    personaStatus("Importing; voice cloning takes a moment…");

    try {
        const body = JSON.stringify({
            preset: JSON.parse(await presetFile.text()),
            voice_b64: await fileToBase64(voiceFile),
            voice_filename: voiceFile.name,
            sample_text: personaTextInput.value || "",
        });

        const resp = await fetch("/personas", {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body,
        });

        const data = await resp.json().catch(() => ({}));

        if (!resp.ok) {
            personaStatus(
                data.error || `Import failed (${resp.status}).`,
                "error"
            );
            return;
        }

        await loadPersonas();

        persona.value = data.persona.id;
        persona.onchange();
        syncPersonaUi();

        personaStatus(`Added ${data.persona.label}.`, "ok");

        personaJsonInput.value = "";
        personaVoiceInput.value = "";
        personaTextInput.value = "";
    } catch (err) {
        personaStatus(`Import failed: ${err.message}`, "error");
    } finally {
        personaImportButton.disabled = false;
    }
};

personaDeleteButton.onclick = async () => {
    const rec = knownPersonas.find((p) => p.id === persona.value);

    if (!rec || rec.builtin) return;

    if (!confirm(`Delete "${rec.label}" and its VoxCPM voice?`)) {
        return;
    }

    personaDeleteButton.disabled = true;
    personaStatus(`Deleting ${rec.label}…`);

    try {
        const resp = await fetch(
            `/personas/${encodeURIComponent(rec.id)}`,
            { method: "DELETE" }
        );

        if (!resp.ok) {
            const data = await resp.json().catch(() => ({}));

            personaStatus(
                data.error || `Delete failed (${resp.status}).`,
                "error"
            );
            return;
        }

        await loadPersonas();
        persona.onchange();

        personaStatus(`Deleted ${rec.label}.`, "ok");
    } catch (err) {
        personaStatus(`Delete failed: ${err.message}`, "error");
    }
};

persona.addEventListener("change", syncPersonaUi);

loadPersonas();

log("frontend build: ring-v23-personas");
