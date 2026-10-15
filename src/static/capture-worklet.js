class CaptureProcessor extends AudioWorkletProcessor {
    constructor() {
        super();
        this.frame = new Int16Array(320);
        this.offset = 0;
    }

    process(inputs, outputs) {
        const input = inputs[0]?.[0];

        if (input) {
            for (let i = 0; i < input.length; i++) {
                const x = Math.max(-1, Math.min(1, input[i]));

                this.frame[this.offset++] =
                    x < 0 ? x * 32768 : x * 32767;

                if (this.offset === 320) {
                    const buffer = this.frame.buffer;

                    this.port.postMessage(buffer, [buffer]);

                    this.frame = new Int16Array(320);
                    this.offset = 0;
                }
            }
        }

        // Keep processor alive without playing the microphone.
        for (const output of outputs) {
            for (const channel of output) {
                channel.fill(0);
            }
        }

        return true;
    }
}

registerProcessor("capture-processor", CaptureProcessor);
