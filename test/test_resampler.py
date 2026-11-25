import math
import struct
import unittest

from pydub import AudioSegment
from pydub.resampler import (
    PCMFormatError,
    PCMFormatMismatchError,
    PCMResampler,
)


def pack_pcm(values, sample_width=2):
    if sample_width == 1:
        return struct.pack("<%db" % len(values), *values)
    if sample_width == 2:
        return struct.pack("<%dh" % len(values), *values)
    if sample_width == 4:
        return struct.pack("<%di" % len(values), *values)
    raise ValueError("test helper supports 1, 2 and 4 byte samples")


def unpack_pcm(data, sample_width=2):
    if sample_width == 1:
        return struct.unpack("<%db" % (len(data) // 1), data)
    if sample_width == 2:
        return struct.unpack("<%dh" % (len(data) // 2), data)
    if sample_width == 4:
        return struct.unpack("<%di" % (len(data) // 4), data)
    raise ValueError("test helper supports 1, 2 and 4 byte samples")


def decode_mono(data, sample_width=2):
    return unpack_pcm(data, sample_width)


def rms(values):
    return math.sqrt(sum(float(value) * value for value in values) / len(values))


class PCMResamplerTests(unittest.TestCase):

    def test_non_integer_ratio_frame_counts(self):
        cases = [
            (44100, 48000, 1003, 1092),
            (48000, 44100, 1001, 920),
            (16000, 44100, 777, 2142),
            (8000, 4000, 1, 1),  # exact half rounds up
        ]
        for input_rate, output_rate, input_frames, expected_frames in cases:
            resampler = PCMResampler(
                output_frame_rate=output_rate,
                input_frame_rate=input_rate,
                sample_width=2,
                channels=1,
            )
            data = pack_pcm([(index * 29) % 32768 - 16384 for index in range(input_frames)])
            output = resampler.process(data) + resampler.finish()
            self.assertEqual(len(output) // 2, expected_frames)
            self.assertEqual(resampler.output_frame_count, expected_frames)

    def test_whole_and_chunked_output_are_identical(self):
        input_frames = 1003
        input_rate = 44100
        output_rate = 48000
        channels = 2
        values = [
            ((index * 97) % 65536) - 32768
            for index in range(input_frames * channels)
        ]
        data = pack_pcm(values)

        whole = PCMResampler(
            output_frame_rate=output_rate,
            input_frame_rate=input_rate,
            sample_width=2,
            channels=channels,
        )
        expected = whole.process(data) + whole.finish()

        chunked = PCMResampler(
            output_frame_rate=output_rate,
            input_frame_rate=input_rate,
            sample_width=2,
            channels=channels,
        )
        pieces = []
        frame = 0
        sizes = [1, 2, 5, 3, 17, 11, 23, 7, 31, 13, 41, 19]
        size_index = 0
        while frame < input_frames:
            size = sizes[size_index % len(sizes)]
            size_index += 1
            end = min(input_frames, frame + size)
            pieces.append(chunked.process(data[frame * 4:end * 4]))
            frame = end
        pieces.append(chunked.finish())
        actual = b"".join(pieces)

        self.assertEqual(actual, expected)
        self.assertEqual(len(actual) // 4, 1092)

    def test_frame_at_a_time_matches_one_pass(self):
        input_frames = 257
        channels = 2
        values = [
            ((index * 53) % 65536) - 32768
            for index in range(input_frames * channels)
        ]
        data = pack_pcm(values)

        whole = PCMResampler(
            output_frame_rate=48000,
            input_frame_rate=44100,
            sample_width=2,
            channels=channels,
        )
        expected = whole.process(data) + whole.finish()

        per_frame = PCMResampler(
            output_frame_rate=48000,
            input_frame_rate=44100,
            sample_width=2,
            channels=channels,
        )
        pieces = [
            per_frame.process(data[index * 4:(index + 1) * 4])
            for index in range(input_frames)
        ]
        pieces.append(per_frame.finish())
        self.assertEqual(b"".join(pieces), expected)

    def test_finish_is_idempotent(self):
        resampler = PCMResampler(
            output_frame_rate=16000,
            input_frame_rate=48000,
            sample_width=1,
            channels=1,
        )
        data = pack_pcm([0] * 480, sample_width=1)
        output = resampler.process(data)
        tail = resampler.finish()
        self.assertEqual(len(output) + len(tail), 160)
        self.assertEqual(resampler.finish(), b"")
        self.assertEqual(resampler.finish(), b"")

    def test_first_chunk_sets_format_and_later_changes_fail(self):
        first = AudioSegment(
            data=pack_pcm([0] * 8),
            sample_width=2,
            frame_rate=8,
            channels=1,
        )
        resampler = PCMResampler(output_frame_rate=16)
        resampler.process(first)
        self.assertEqual(resampler.format.sample_width, 2)
        self.assertEqual(resampler.format.channels, 1)
        self.assertEqual(resampler.format.frame_rate, 8)

        changed = AudioSegment(
            data=pack_pcm([0] * 16),
            sample_width=2,
            frame_rate=8,
            channels=2,
        )
        with self.assertRaises(PCMFormatMismatchError):
            resampler.process(changed)

    def test_raw_pcm_requires_format(self):
        with self.assertRaises(PCMFormatError):
            PCMResampler(output_frame_rate=16000).process(b"\0" * 200)

    def test_chunks_must_contain_complete_frames(self):
        resampler = PCMResampler(
            output_frame_rate=16000,
            input_frame_rate=8000,
            sample_width=2,
            channels=2,
        )
        with self.assertRaises(ValueError):
            resampler.process(b"\0\0\0")

    def test_input_is_not_modified(self):
        data = bytearray(pack_pcm(list(range(-20, 20))))
        original = bytes(data)
        resampler = PCMResampler(
            output_frame_rate=16000,
            input_frame_rate=8000,
            sample_width=2,
            channels=1,
        )
        resampler.process(data)
        resampler.finish()
        self.assertEqual(bytes(data), original)

        segment = AudioSegment(
            data=pack_pcm(list(range(-20, 20))),
            sample_width=2,
            frame_rate=8000,
            channels=1,
        )
        original_segment_data = segment.raw_data
        PCMResampler(output_frame_rate=16000).process(segment)
        self.assertEqual(segment.raw_data, original_segment_data)

    def test_history_stays_bounded(self):
        resampler = PCMResampler(
            output_frame_rate=48000,
            input_frame_rate=44100,
            sample_width=2,
            channels=1,
            filter_half_width=64,
        )
        block = pack_pcm([100] * 1000)
        for _ in range(100):
            resampler.process(block)
            self.assertLessEqual(resampler.buffered_input_frames, 8 * 64 + 2)

    def test_cross_chunk_impulse_has_no_seam(self):
        input_frames = 300
        input_rate = 48000
        output_rate = 16000
        impulse_frame = 149
        values = [0] * input_frames
        values[impulse_frame] = 30000
        data = pack_pcm(values)

        whole = PCMResampler(
            output_frame_rate=output_rate,
            input_frame_rate=input_rate,
            sample_width=2,
            channels=1,
            filter_half_width=64,
        )
        expected = whole.process(data) + whole.finish()

        # The impulse sits on one side of the boundary, while the filter
        # support spans both chunks.
        chunked = PCMResampler(
            output_frame_rate=output_rate,
            input_frame_rate=input_rate,
            sample_width=2,
            channels=1,
            filter_half_width=64,
        )
        actual = (
            chunked.process(data[:150 * 2])
            + chunked.process(data[150 * 2:])
            + chunked.finish()
        )
        self.assertEqual(actual, expected)

        samples = decode_mono(actual)
        self.assertGreater(max(samples), 0)
        # Byte-for-byte equality with the one-pass result proves that the
        # chunk boundary did not inject a second impulse or a seam transient.
        self.assertEqual(len(samples), len(decode_mono(expected)))

    def test_channels_do_not_mix(self):
        input_frames = 2000
        for left, right in ((12000, 0), (0, -12000)):
            frames = [(left, right)] * input_frames
            interleaved = [value for frame in frames for value in frame]
            resampler = PCMResampler(
                output_frame_rate=16000,
                input_frame_rate=48000,
                sample_width=2,
                channels=2,
            )
            output = resampler.process(pack_pcm(interleaved)) + resampler.finish()
            samples = decode_mono(output)
            left_out = samples[0::2]
            right_out = samples[1::2]
            if left:
                self.assertEqual(max(abs(value) for value in right_out), 0)
                self.assertGreater(max(abs(value) for value in left_out), 0)
            else:
                self.assertEqual(max(abs(value) for value in left_out), 0)
                self.assertGreater(max(abs(value) for value in right_out), 0)

    def test_downsampling_suppresses_aliasing(self):
        input_rate = 48000
        output_rate = 16000
        duration = 1.0
        input_frames = int(input_rate * duration)
        amplitude = 0.8 * 32767

        def tone_rms(frequency):
            values = [
                int(round(amplitude * math.sin(2 * math.pi * frequency * index / input_rate)))
                for index in range(input_frames)
            ]
            resampler = PCMResampler(
                output_frame_rate=output_rate,
                input_frame_rate=input_rate,
                sample_width=2,
                channels=1,
            )
            output = resampler.process(pack_pcm(values)) + resampler.finish()
            samples = decode_mono(output)
            edge = int(output_rate * 0.05)
            return rms(samples[edge:-edge])

        passband = tone_rms(1000)
        aliased = tone_rms(10000)
        attenuation_db = 20 * math.log10(passband / aliased)
        self.assertGreaterEqual(attenuation_db, 40.0)


if __name__ == "__main__":
    unittest.main()
