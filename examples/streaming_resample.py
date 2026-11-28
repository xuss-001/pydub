"""
Example: stream-resample PCM audio chunk by chunk with PCMResampler.

Usage:
    python examples/streaming_resample.py [input.wav] [output.wav]

Without arguments a 2-second stereo test signal at 48 kHz is synthesized,
streamed through the resampler in 20 ms chunks (as if arriving from the
network), converted to 16 kHz and written to /tmp/resampled.wav.

Key rules demonstrated here:
  * the first chunk locks in the stream format (sample width, channels,
    frame rate); later chunks must match or ValueError is raised;
  * call flush() exactly once at end of input to drain the tail;
  * total output frames == round(input_frames * out_rate / in_rate),
    independent of how the input was chunked.
"""

import sys

import numpy as np

from pydub import AudioSegment, PCMResampler

IN_RATE = 48000
OUT_RATE = 16000
CHUNK_MS = 20


def synth_input():
    t = np.arange(2 * IN_RATE) / IN_RATE
    left = 12000 * np.sin(2 * np.pi * 1000 * t)
    right = 12000 * np.sin(2 * np.pi * 500 * t)
    stereo = np.column_stack([left, right]).astype(np.int16)
    return AudioSegment(data=stereo.tobytes(), sample_width=2,
                        channels=2, frame_rate=IN_RATE)


def frame_chunks(seg, frames_per_chunk):
    """Slice an AudioSegment into whole-frame chunks (no ms rounding)."""
    frame_width = seg.frame_width
    data = seg.raw_data
    step = frames_per_chunk * frame_width
    for pos in range(0, len(data), step):
        yield AudioSegment(data=data[pos:pos + step],
                           sample_width=seg.sample_width,
                           channels=seg.channels,
                           frame_rate=seg.frame_rate)


def main():
    if len(sys.argv) > 1:
        source = AudioSegment.from_wav(sys.argv[1])
    else:
        source = synth_input()
    out_path = sys.argv[2] if len(sys.argv) > 2 else "/tmp/resampled.wav"

    resampler = PCMResampler(output_rate=OUT_RATE)  # rate taken from 1st chunk
    frames_per_chunk = source.frame_rate * CHUNK_MS // 1000

    out_frames = 0
    result = AudioSegment.empty()
    for chunk in frame_chunks(source, frames_per_chunk):
        piece = resampler.process(chunk)   # may be empty; latency is FIR/2
        result += piece
        out_frames += piece.frame_count()
    tail = resampler.flush()               # drain remaining samples, once
    result += tail
    out_frames += tail.frame_count()

    expected = round(source.frame_count() * OUT_RATE / source.frame_rate)
    print("input:  %d frames @ %d Hz, %d ch, %d bytes/sample"
          % (source.frame_count(), source.frame_rate, source.channels,
             source.sample_width))
    print("output: %d frames @ %d Hz (expected %d), latency %d input frames"
          % (out_frames, OUT_RATE, expected, resampler.latency))
    assert out_frames == expected

    result.export(out_path, format="wav")
    print("written to %s" % out_path)


if __name__ == "__main__":
    main()
