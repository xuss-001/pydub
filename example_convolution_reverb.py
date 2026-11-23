"""
Streaming convolution reverb example.

Run from the repository root:

    python example_convolution_reverb.py

The example synthesizes a short "room" impulse response (a few decaying
echoes), processes a long tone one block at a time, and verifies that the
block-wise output is identical (within one integer sample) to convolving the
whole recording at once.
"""

import numpy as np

from pydub import AudioSegment
from pydub.convolution import ConvolutionReverb, convolve
from pydub.generators import Sine
from pydub.utils import make_chunks


FRAME_RATE = 8000


def build_impulse_response():
    # A synthetic room response: direct sound plus three decaying, delayed
    # echoes. Samples are normalized against PCM full scale internally, so
    # the absolute amplitude here only has to fit in the 16-bit range.
    taps = {0: 1.0, 3: 0.6, 7: 0.35, 12: 0.18}
    samples = np.zeros(max(taps) + 1, dtype=np.int16)
    for tap, coefficient in taps.items():
        samples[tap] = int(round(coefficient * 32767))
    return AudioSegment(
        samples.tobytes(),
        metadata={
            "channels": 1,
            "sample_width": 2,
            "frame_rate": FRAME_RATE,
            "frame_width": 2,
        })


def main():
    recording = Sine(440, sample_rate=FRAME_RATE).to_audio_segment(
        duration=250)
    impulse_response = build_impulse_response()

    # One-shot helper: useful when the whole signal fits in memory.
    whole_output = convolve(
        recording, impulse_response, dry_gain=0.9, wet_gain=0.6)

    # Streaming API: constant memory regardless of the recording length.
    reverb = ConvolutionReverb(
        impulse_response, dry_gain=0.9, wet_gain=0.6)
    streamed_output = AudioSegment.empty()
    for block in make_chunks(recording, 25):
        streamed_output += reverb.process(block)
    # The reverb tail must be emitted after the last input block.
    streamed_output += reverb.flush()

    whole = np.asarray(
        whole_output.get_array_of_samples(), dtype=np.int64)
    streamed = np.asarray(
        streamed_output.get_array_of_samples(), dtype=np.int64)

    print("input frames:      {0}".format(int(recording.frame_count())))
    print("impulse taps:      {0}".format(
        int(impulse_response.frame_count())))
    print("one-shot frames:   {0}".format(len(whole)))
    print("streaming frames:  {0}".format(len(streamed)))
    print("max sample diff:   {0}".format(np.max(np.abs(whole - streamed))))

    assert len(whole) == len(streamed)
    assert np.max(np.abs(whole - streamed)) <= 1


if __name__ == "__main__":
    main()
