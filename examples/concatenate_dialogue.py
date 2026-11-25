"""
Join several game-dialogue lines with per-join equal-power crossfades.

Run with:  python examples/concatenate_dialogue.py

The script generates three short tones in memory (no input files needed),
concatenates them using AudioSegment.concatenate(), prints what happens at
each join and writes the result to dialogue.wav.
"""
import math
import os

from pydub import AudioSegment
from pydub.generators import Sine


def main():
    # three dialogue-like lines with different native formats
    line1 = Sine(440, sample_rate=32000, bit_depth=16).to_audio_segment(500)
    line2 = Sine(550, sample_rate=44100, bit_depth=16).to_audio_segment(400)
    line3 = Sine(660, sample_rate=22050, bit_depth=32).to_audio_segment(600)

    # 10 ms between lines 1-2, 20 ms between lines 2-3, expressed in frames
    # of the requested 48 kHz / 16-bit / stereo output
    output_rate = 48000
    crossfades = [
        int(0.010 * output_rate),
        int(0.020 * output_rate),
    ]

    dialogue = AudioSegment.concatenate(
        [line1, line2, line3],
        crossfades=crossfades,
        frame_rate=output_rate,
        sample_width=2,
        channels=2,
    )

    print("output format: {0} Hz, {1}-bit, {2} channel(s)".format(
        dialogue.frame_rate, dialogue.sample_width * 8, dialogue.channels))
    print("output frames: {0} ({1:.3f} s)".format(
        int(dialogue.frame_count()), dialogue.duration_seconds))
    print("crossfades in output frames: {0}".format(crossfades))

    # equal-power property of a one-frame crossfade
    single_frame = AudioSegment.concatenate(
        [line1[:5], line2[:5]], crossfades=1)
    print("one-frame fade gain per side: sqrt(0.5) = {0:.4f}".format(
        math.sqrt(0.5)))
    print("one-frame fade output frames: {0}".format(
        int(single_frame.frame_count())))

    out_path = os.path.join(os.path.dirname(__file__), "dialogue.wav")
    dialogue.export(out_path, format="wav")
    print("wrote {0}".format(out_path))


if __name__ == "__main__":
    main()
