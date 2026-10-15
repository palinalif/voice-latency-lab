const SOURCE_RATE = 48000;

const START_BUFFER_MS = 80;
const REBUFFER_MS = 500;
const FADE_IN_MS = 5;

class PCMPlaybackProcessor extends AudioWorkletProcessor {
    constructor() {
        super();

        this.queue = [];
        this.chunkOffset = 0;
        this.bufferedSamples = 0;

        this.started = false;
        this.ended = false;
        this.rebuffering = false;

        this.fadeInRemaining = 0;

        this.startThreshold = Math.round(
            SOURCE_RATE * START_BUFFER_MS / 1000
        );

        this.rebufferThreshold = Math.round(
            SOURCE_RATE * REBUFFER_MS / 1000
        );

        this.fadeInSamples = Math.max(
            1,
            Math.round(
                SOURCE_RATE * FADE_IN_MS / 1000
            )
        );

        this.port.onmessage = event => {
            const msg = event.data;

            if (msg.type === "pcm") {
                const chunk =
                    new Int16Array(msg.buffer);

                if (chunk.length > 0) {
                    this.queue.push(chunk);

                    this.bufferedSamples +=
                        chunk.length;
                }
            }

            else if (msg.type === "end") {
                this.ended = true;
            }

            else if (msg.type === "clear") {
                this.reset();

                this.port.postMessage({
                    type: "cleared"
                });
            }
        };
    }

    reset() {
        this.queue = [];
        this.chunkOffset = 0;
        this.bufferedSamples = 0;

        this.started = false;
        this.ended = false;
        this.rebuffering = false;

        this.fadeInRemaining = 0;
    }

    popSample() {
        while (this.queue.length > 0) {
            const chunk = this.queue[0];

            if (
                this.chunkOffset
                < chunk.length
            ) {
                const value =
                    chunk[this.chunkOffset++];

                this.bufferedSamples--;

                return value / 32768;
            }

            this.queue.shift();
            this.chunkOffset = 0;
        }

        return null;
    }

    enterRebuffer() {
        if (this.rebuffering) {
            return;
        }

        this.rebuffering = true;

        this.port.postMessage({
            type: "underrun"
        });
    }

    leaveRebuffer() {
        this.rebuffering = false;

        this.fadeInRemaining =
            this.fadeInSamples;

        this.port.postMessage({
            type: "rebuffered",
            bufferedMs:
                this.bufferedSamples
                / SOURCE_RATE
                * 1000
        });
    }

    process(inputs, outputs) {
        const output = outputs[0];

        if (
            !output
            || output.length === 0
        ) {
            return true;
        }

        const frames =
            output[0].length;

        for (const channel of output) {
            channel.fill(0);
        }

        // Initial startup.
        if (!this.started) {
            if (
                this.bufferedSamples
                    >= this.startThreshold
                || (
                    this.ended
                    && this.bufferedSamples > 0
                )
            ) {
                this.started = true;

                this.port.postMessage({
                    type: "started",
                    bufferedMs:
                        this.bufferedSamples
                        / SOURCE_RATE
                        * 1000
                });
            }
            else {
                return true;
            }
        }

        // After a genuine starvation, don't resume
        // merely because a handful of samples arrived.
        if (this.rebuffering) {
            if (
                this.bufferedSamples
                    >= this.rebufferThreshold
            ) {
                this.leaveRebuffer();
            }
            else if (
                this.ended
                && this.bufferedSamples > 0
            ) {
                // Drain the final partial tail.
                this.leaveRebuffer();
            }
            else if (
                this.ended
                && this.bufferedSamples === 0
            ) {
                this.port.postMessage({
                    type: "ended"
                });

                this.reset();

                return true;
            }
            else {
                return true;
            }
        }

        // Don't consume a partial render quantum and
        // then abruptly fall into zeros. If we already
        // know there isn't enough PCM for this block,
        // hold the samples and enter rebuffer cleanly.
        if (
            !this.ended
            && this.bufferedSamples < frames
        ) {
            this.enterRebuffer();
            return true;
        }

        for (let i = 0; i < frames; i++) {
            const sample =
                this.popSample();

            if (sample === null) {
                if (!this.ended) {
                    this.enterRebuffer();
                }

                break;
            }

            let gain = 1.0;

            if (this.fadeInRemaining > 0) {
                const progress =
                    this.fadeInSamples
                    - this.fadeInRemaining;

                gain =
                    progress
                    / this.fadeInSamples;

                this.fadeInRemaining--;
            }

            const value =
                sample * gain;

            for (const channel of output) {
                channel[i] = value;
            }
        }

        if (
            this.ended
            && this.bufferedSamples === 0
        ) {
            this.port.postMessage({
                type: "ended"
            });

            this.reset();
        }

        return true;
    }
}

registerProcessor(
    "pcm-playback-processor",
    PCMPlaybackProcessor
);
