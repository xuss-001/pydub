"""
Demo of the pitch-preserving `time_stretch` effect using a short
voice-surrogate signal (no audio assets or external encoders needed).

Run from the repository root:

    python examples/time_stretch_demo.py [output_dir]

It synthesizes a harmonic, formant-weighted buzz with vibrato and a
syllable-like envelope (a stand-in for a speech snippet), stretches it to
0.5x (longer) and 2.0x (shorter) and writes WAV files (pure stdlib WAV
export, no ffmpeg involved).
"""
import os
import sys

import numpy as np

from pydub import AudioSegment
import pydub.effects  # noqa: F401  (registers AudioSegment.time_stretch)

FRAME_RATE = 22050
F0 = 120.0


def voice_surrogate(duration=0.8, frame_rate=FRAME_RATE, f0=F0):
    """Short voice-like surrogate: harmonic stack with formant weighting,
    gentle vibrato and syllable-like amplitude modulation."""
    n = int(round(duration * frame_rate))
    t = np.arange(n, dtype=np.float64) / frame_rate
    phase_dev = 0.015 * f0 / 5.0 * np.sin(2 * np.pi * 5.0 * t)
    signal = np.zeros(n, dtype=np.float64)
    for harmonic in range(1, 9):
        freq = f0 * harmonic
        if freq >= frame_rate / 2:
            break
        weight = np.exp(-((freq - 600.0) ** 2) / (2 * 500.0 ** 2)) / harmonic
        signal += weight * np.sin(2 * np.pi * freq * t + phase_dev * harmonic)
    signal *= 0.6 + 0.4 * np.sin(2 * np.pi * 3.0 * t) ** 2
    signal *= 0.4 / max(1e-12, np.abs(signal).max())
    data = np.round(signal * 32767).astype('<i2').tobytes()
    return AudioSegment(data=data, sample_width=2, frame_rate=frame_rate,
                        channels=1)


def dominant_frequency(seg):
    samples = np.array(seg.get_array_of_samples(), dtype=np.float64)
    n = len(samples)
    spectrum = np.abs(np.fft.rfft(samples * np.hanning(n)))
    peak = int(np.argmax(spectrum))
    left, center, right = spectrum[peak - 1], spectrum[peak], spectrum[peak + 1]
    offset = 0.5 * (left - right) / (left - 2 * center + right)
    return (peak + offset) * seg.frame_rate / n


def main():
    out_dir = sys.argv[1] if len(sys.argv) > 1 else 'time_stretch_out'
    os.makedirs(out_dir, exist_ok=True)

    voice = voice_surrogate()
    print('input:      %d frames @ %d Hz, f0 ~= %.1f Hz'
          % (voice.frame_count(), voice.frame_rate,
             dominant_frequency(voice)))

    for speed in (0.5, 2.0):
        stretched = voice.time_stretch(speed)
        path = os.path.join(out_dir, 'voice_%gx.wav' % speed)
        stretched.export(path, format='wav')
        print('%.3gx -> %d frames, f0 ~= %.1f Hz, saved %s'
              % (speed, stretched.frame_count(),
                 dominant_frequency(stretched), path))


if __name__ == '__main__':
    main()
