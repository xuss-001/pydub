import array
import math
import random
import unittest

from pydub import AudioSegment, LookaheadLimiter
from pydub.exceptions import InvalidAudioFormat, InvalidLimiterSettings


def pcm(values, sample_width=2):
    if sample_width == 1:
        data = array.array("b", values)
    elif sample_width == 2:
        data = array.array("h", values)
    else:
        data = array.array("i", values)
    try:
        return data.tobytes()
    except AttributeError:  # pragma: no cover - Python 2 compatibility
        return data.tostring()


def samples(data, sample_width=2):
    if sample_width == 1:
        return array.array("b", data)
    if sample_width == 2:
        return array.array("h", data)
    return array.array("i", data)


class LookaheadLimiterTests(unittest.TestCase):
    def make_limiter(self, **kwargs):
        settings = {
            "sample_width": 2,
            "frame_rate": 1000,
            "channels": 1,
            "ceiling": 0.5,
            "lookahead_ms": 10,
            "release_ms": 25,
        }
        settings.update(kwargs)
        return LookaheadLimiter(**settings)

    def test_chunk_boundaries_do_not_change_output(self):
        random.seed(7)
        frames = []
        for i in range(600):
            if 200 <= i < 215:
                frames.append(random.choice((-32000, 32000, 28000, -28000)))
            else:
                frames.append(int(2500 * math.sin(2 * math.pi * 17 * i / 1000)))
        raw = pcm(frames)

        def run(block_sizes):
            limiter = self.make_limiter()
            output = []
            position = 0
            block_index = 0
            while position < len(frames):
                block_length = block_sizes[block_index % len(block_sizes)]
                block_length = min(block_length, len(frames) - position)
                start = position * 2
                end = (position + block_length) * 2
                output.append(limiter.process(raw[start:end]))
                position += block_length
                block_index += 1
            output.append(limiter.finish())
            return b"".join(output), limiter

        whole, whole_limiter = run([len(frames)])
        split, split_limiter = run([1, 2, 3, 5, 8, 13, 21, 34, 55])

        self.assertEqual(whole, split)
        self.assertEqual(whole_limiter.frames_in, len(frames))
        self.assertEqual(whole_limiter.frames_out, len(frames))
        self.assertEqual(split_limiter.pending_frames, 0)
        self.assertEqual(len(split) // 2, len(frames))

    def test_cross_chunk_peak_is_limited_without_clipping(self):
        frames = [4000] * 30 + [31000] + [4000] * 80
        raw = pcm(frames)
        limiter = self.make_limiter()

        output = []
        for index, frame in enumerate(frames):
            output.append(limiter.process(raw[index * 2:(index + 1) * 2]))
        output.append(limiter.finish())
        result = samples(b"".join(output))

        self.assertEqual(len(result), len(frames))
        self.assertLessEqual(max(abs(value) for value in result), 16384)
        self.assertEqual(result[30], 16384)
        # The gain ramps down before the peak instead of flattening the event.
        attack = result[20:30]
        self.assertTrue(all(a > b for a, b in zip(attack, attack[1:])))
        self.assertGreater(attack[-1], 0)

    def test_short_stream_uses_available_attack_time(self):
        frames = [1000, 1000, 1000, 30000, 1000]
        limiter = self.make_limiter(lookahead_ms=10, release_ms=10)
        output = limiter.process(pcm(frames)) + limiter.finish()
        result = samples(output)

        self.assertEqual(len(result), len(frames))
        self.assertLessEqual(max(abs(value) for value in result), 16384)
        self.assertTrue(all(abs(a) > abs(b)
                            for a, b in zip(result[:3], result[1:3])))
        # Only the actual peak reaches the ceiling; no adjacent hard-clipped run.
        self.assertEqual(sum(1 for value in result if abs(value) == 16384), 1)

    def test_sustained_overload_and_channel_relationship(self):
        frames = []
        for i in range(300):
            left = int(30000 * math.sin(2 * math.pi * 23 * i / 1000))
            right = int(left / 2)
            frames.extend((left, right))

        segment = AudioSegment(
            pcm(frames), sample_width=2, frame_rate=1000, channels=2)
        output = segment.lookahead_limiter(
            ceiling=0.5, lookahead_ms=8, release_ms=20)
        result = list(output.get_array_of_samples())

        self.assertEqual(output.frame_count(), segment.frame_count())
        self.assertLessEqual(max(abs(value) for value in result), 16384)
        for index in range(0, len(result), 2):
            self.assertLessEqual(abs(result[index] - 2 * result[index + 1]), 2)

    def test_release_recovers_normal_audio(self):
        normal = [for_sample for for_sample in (1200, -1200) * 120]
        frames = normal[:100] + [30000] + normal[100:]
        raw = pcm(frames)
        limiter = self.make_limiter(lookahead_ms=5, release_ms=30)
        output = limiter.process(raw) + limiter.finish()
        result = samples(output)

        post_peak = result[101:150]
        envelope = [abs(value) for value in post_peak]
        self.assertTrue(all(a <= b for a, b in zip(envelope, envelope[1:])))
        self.assertEqual(output[-80:], raw[-80:])
        self.assertEqual(result[-1], frames[-1])

    def test_finish_drains_once_and_never_duplicates_frames(self):
        raw = pcm([1000] * 50)
        limiter = self.make_limiter()

        output = limiter.process(raw)
        self.assertEqual(len(output), (50 - limiter.lookahead_frames) * 2)
        tail = limiter.finish()
        self.assertEqual(len(tail), limiter.lookahead_frames * 2)
        self.assertEqual(len(output) + len(tail), len(raw))
        self.assertEqual(limiter.frames_out, 50)
        self.assertEqual(limiter.finish(), b"")
        self.assertEqual(limiter.flush(), b"")
        with self.assertRaises(RuntimeError):
            limiter.process(b"\x00\x00")

    def test_format_change_and_incomplete_frames_raise(self):
        limiter = self.make_limiter()
        limiter.process(b"\x00\x00")
        with self.assertRaises(InvalidAudioFormat):
            limiter.process(b"\x00\x00", sample_width=1)
        with self.assertRaises(InvalidAudioFormat):
            limiter.process(b"\x00")

        stereo = AudioSegment(
            pcm([0, 0]), sample_width=2, frame_rate=1000, channels=2)
        with self.assertRaises(InvalidAudioFormat):
            limiter.process(stereo)
        with self.assertRaises(InvalidAudioFormat):
            limiter.process(b"\x00\x00", format={"sample_rate": 2000})

    def test_invalid_settings_raise_clear_errors(self):
        invalid_settings = [
            {"ceiling": 100000},
            {"lookahead_ms": 0},
            {"release_ms": 0},
            {"sample_width": 5},
            {"frame_rate": 0},
            {"channels": 0},
            {"ceiling_db": 1.0},
        ]
        for settings in invalid_settings:
            with self.assertRaises((InvalidLimiterSettings, InvalidAudioFormat)):
                self.make_limiter(**settings)

    def test_input_is_unchanged_and_buffer_is_bounded(self):
        frames = [20000, -20000] * 500
        raw = pcm(frames)
        original = bytes(raw)
        limiter = self.make_limiter(lookahead_ms=12, release_ms=30)

        for offset in range(0, len(raw), 20):
            limiter.process(raw[offset:offset + 20])
            self.assertLessEqual(limiter.pending_frames,
                                 limiter.lookahead_frames + 1)
            self.assertLessEqual(len(limiter._peak_queue),
                                 limiter.pending_frames)

        self.assertEqual(raw, original)
        limiter.finish()
        self.assertEqual(raw, original)

    def test_segment_effect_and_streaming_share_identical_result(self):
        frames = []
        for i in range(400):
            value = 30000 if 190 <= i < 210 else int(
                2000 * math.sin(2 * math.pi * 11 * i / 1000))
            frames.append(value)
        raw = pcm(frames)
        segment = AudioSegment(
            raw, sample_width=2, frame_rate=1000, channels=1)

        limiter = self.make_limiter(lookahead_ms=7, release_ms=18)
        streamed = limiter.process(raw) + limiter.finish()
        effect = segment.limiter(
            ceiling=0.5, lookahead_ms=7, release_ms=18)

        self.assertEqual(streamed, effect.raw_data)
        self.assertEqual(segment.raw_data, raw)
        self.assertEqual(effect.frame_count(), segment.frame_count())


if __name__ == "__main__":
    unittest.main()
