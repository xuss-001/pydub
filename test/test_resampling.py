import unittest
from fractions import Fraction

import numpy as np

from pydub import AudioSegment, PCMResampler


def make_segment(samples, frame_rate, sample_width=2):
    """Build an AudioSegment from a (frames,) or (frames, channels) array."""
    samples = np.asarray(samples)
    dtype = {1: np.int8, 2: np.int16, 4: np.int32}[sample_width]
    samples = samples.astype(dtype)
    channels = 1 if samples.ndim == 1 else samples.shape[1]
    return AudioSegment(data=samples.tobytes(), sample_width=sample_width,
                        channels=channels, frame_rate=frame_rate)


def to_array(segment):
    dtype = {1: np.int8, 2: np.int16, 4: np.int32}[segment.sample_width]
    samples = np.frombuffer(segment.raw_data, dtype=dtype)
    return samples.reshape(-1, segment.channels).astype(np.float64)


def sine(freq, rate, frames, amplitude=10000):
    t = np.arange(frames) / rate
    return np.rint(amplitude * np.sin(2 * np.pi * freq * t))


def rms_db(samples):
    rms = max(np.sqrt(np.mean(samples ** 2)), 1e-12)
    return 20 * np.log10(rms)


def run_stream(chunks, output_rate, input_rate=None):
    resampler = PCMResampler(output_rate=output_rate, input_rate=input_rate)
    pieces = [resampler.process(chunk) for chunk in chunks]
    pieces.append(resampler.flush())
    return b"".join(p.raw_data for p in pieces), pieces, resampler


class PCMResamplerTest(unittest.TestCase):

    def test_frame_count_non_integer_ratio(self):
        in_rate, out_rate = 44100, 16000
        for frames in (1, 2, 7, 100, 441, 1000, 12345, 44100):
            seg = make_segment(sine(440, in_rate, frames), in_rate)
            data, pieces, _ = run_stream([seg], out_rate)
            total = sum(p.frame_count() for p in pieces)
            expected = int(Fraction(frames * out_rate, in_rate) + Fraction(1, 2))
            self.assertEqual(total, expected)
            self.assertEqual(total, len(data) // 2)

    def test_frame_count_non_integer_ratio_chunked(self):
        in_rate, out_rate = 44100, 16000
        frames = 10000
        arr = to_array(make_segment(sine(440, in_rate, frames), in_rate))
        rng = np.random.RandomState(42)
        chunks, pos = [], 0
        while pos < frames:
            size = int(rng.randint(1, 700))
            chunks.append(make_segment(arr[pos:pos + size], in_rate))
            pos += size
        _, pieces, _ = run_stream(chunks, out_rate)
        total = sum(p.frame_count() for p in pieces)
        expected = int(Fraction(frames * out_rate, in_rate) + Fraction(1, 2))
        self.assertEqual(total, expected)

    def test_chunked_output_identical_to_whole(self):
        in_rate, out_rate = 48000, 16000
        frames = 8000
        t = np.arange(frames) / in_rate
        left = 8000 * np.sin(2 * np.pi * 1000 * t)
        right = 5000 * np.sin(2 * np.pi * 2500 * t)
        noise = np.random.RandomState(0).randint(-3000, 3000, frames)
        stereo = np.column_stack([left + noise, right + noise])
        arr = to_array(make_segment(stereo, in_rate))

        ref, _, _ = run_stream([make_segment(arr, in_rate)], out_rate)

        # arbitrary chunk sizes, including 1-frame chunks
        chunks = []
        pos = 0
        for size in [1, 1, 1, 7, 1000, 2, 4096, 13, 500]:
            if pos >= frames:
                break
            chunks.append(make_segment(arr[pos:pos + size], in_rate))
            pos += size
        if pos < frames:
            chunks.append(make_segment(arr[pos:], in_rate))
        chunked, _, _ = run_stream(chunks, out_rate)
        self.assertEqual(ref, chunked)

        # frame by frame
        small = arr[:300]
        ref_small, _, _ = run_stream([make_segment(small, in_rate)], out_rate)
        fbf, _, _ = run_stream(
            [make_segment(small[i:i + 1], in_rate) for i in range(300)],
            out_rate)
        self.assertEqual(ref_small, fbf)

    def test_impulse_across_chunks_has_no_extra_seam(self):
        in_rate, out_rate = 48000, 16000
        frames = 4000
        arr = np.zeros((frames, 1))
        arr[2000, 0] = 20000  # impulse exactly at a chunk boundary
        ref, _, _ = run_stream([make_segment(arr, in_rate)], out_rate)
        chunks = [make_segment(arr[:2000], in_rate),
                  make_segment(arr[2000:], in_rate)]
        chunked, _, _ = run_stream(chunks, out_rate)
        self.assertEqual(ref, chunked)

        out = np.abs(np.frombuffer(ref, dtype=np.int16).astype(np.float64))
        peak = out.max()
        above = np.where(out > 0.5 * peak)[0]
        # one compact main lobe only: no duplicated impulse / seam artifact
        self.assertLessEqual(len(above), 3)
        self.assertTrue(np.all(np.diff(above) == 1))

    def test_no_channel_crosstalk(self):
        in_rate, out_rate = 48000, 16000
        frames = 9600
        stereo = np.column_stack([sine(1000, in_rate, frames),
                                  np.zeros(frames)])
        data, _, resampler = run_stream([make_segment(stereo, in_rate)],
                                        out_rate)
        out = to_array(AudioSegment(data=data, sample_width=2, channels=2,
                                    frame_rate=out_rate))
        lat = int(np.ceil(resampler.latency * out_rate / in_rate)) + 2
        left = out[lat:-lat, 0]
        right = out[lat:-lat, 1]
        self.assertGreater(rms_db(left) - rms_db(right), 60)

    def test_downsampling_suppresses_aliasing(self):
        in_rate, out_rate = 48000, 16000
        frames = 48000
        levels = {}
        for freq in (1000, 10000):
            seg = make_segment(sine(freq, in_rate, frames), in_rate)
            data, _, resampler = run_stream([seg], out_rate)
            out = to_array(AudioSegment(data=data, sample_width=2, channels=1,
                                        frame_rate=out_rate))[:, 0]
            lat = int(np.ceil(resampler.latency * out_rate / in_rate)) + 2
            levels[freq] = rms_db(out[lat:-lat])
        self.assertGreater(levels[1000] - levels[10000], 40)

    def test_format_mismatch_raises(self):
        resampler = PCMResampler(output_rate=16000)
        resampler.process(make_segment(np.zeros((100, 2)), 48000))
        # different channel count
        with self.assertRaises(ValueError):
            resampler.process(make_segment(np.zeros(100), 48000))
        # different sample width
        with self.assertRaises(ValueError):
            resampler.process(make_segment(np.zeros((100, 2)), 48000,
                                           sample_width=1))
        # different frame rate
        with self.assertRaises(ValueError):
            resampler.process(make_segment(np.zeros((100, 2)), 44100))

    def test_constructor_input_rate_mismatch_raises(self):
        resampler = PCMResampler(output_rate=16000, input_rate=48000)
        with self.assertRaises(ValueError):
            resampler.process(make_segment(np.zeros(100), 44100))

    def test_flush_twice_does_not_repeat_output(self):
        resampler = PCMResampler(output_rate=16000)
        resampler.process(make_segment(sine(440, 48000, 4800), 48000))
        tail1 = resampler.flush()
        self.assertGreater(tail1.frame_count(), 0)
        tail2 = resampler.flush()
        self.assertEqual(tail2.frame_count(), 0)
        with self.assertRaises(RuntimeError):
            resampler.process(make_segment(np.zeros(10), 48000))

    def test_flush_before_any_input(self):
        resampler = PCMResampler(output_rate=16000)
        tail = resampler.flush()
        self.assertEqual(tail.frame_count(), 0)
        self.assertEqual(tail.frame_rate, 16000)

    def test_input_is_not_modified(self):
        seg = make_segment(sine(440, 48000, 4800), 48000)
        before = bytes(seg.raw_data)
        resampler = PCMResampler(output_rate=16000)
        resampler.process(seg)
        resampler.flush()
        self.assertEqual(seg.raw_data, before)
        self.assertEqual(seg.frame_count(), 4800)

    def test_upsampling_frame_count(self):
        in_rate, out_rate = 16000, 48000
        frames = 8000
        seg = make_segment(sine(440, in_rate, frames), in_rate)
        _, pieces, _ = run_stream([seg], out_rate)
        total = sum(p.frame_count() for p in pieces)
        self.assertEqual(total, 24000)

    def test_sample_widths(self):
        for sample_width in (1, 2, 4):
            seg = make_segment(sine(440, 48000, 4800, amplitude=100),
                               48000, sample_width=sample_width)
            data, pieces, _ = run_stream([seg], 16000)
            total = sum(p.frame_count() for p in pieces)
            self.assertEqual(len(data), total * sample_width)


if __name__ == "__main__":
    unittest.main()
