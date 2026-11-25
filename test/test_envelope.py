import math
import struct
import unittest

from pydub import AudioSegment
from pydub.envelope import VolumeEnvelope
from pydub.utils import db_to_float


def encode_pcm(samples, sample_width):
    if sample_width == 1:
        return bytes(sample + 128 for sample in samples)
    if sample_width == 2:
        return struct.pack("<%dh" % len(samples), *samples)
    if sample_width == 3:
        out = bytearray()
        for sample in samples:
            if sample < 0:
                sample += 1 << 24
            out += bytes((sample & 0xFF, (sample >> 8) & 0xFF, (sample >> 16) & 0xFF))
        return bytes(out)
    if sample_width == 4:
        return struct.pack("<%di" % len(samples), *samples)
    raise ValueError(sample_width)


def decode_pcm(data, sample_width):
    if sample_width == 1:
        return [byte - 128 for byte in data]
    if sample_width == 2:
        return list(struct.unpack("<%dh" % (len(data) // 2), data))
    if sample_width == 3:
        samples = []
        for i in range(0, len(data), 3):
            val = data[i] | (data[i + 1] << 8) | (data[i + 2] << 16)
            if val >= 1 << 23:
                val -= 1 << 24
            samples.append(val)
        return samples
    if sample_width == 4:
        return list(struct.unpack("<%di" % (len(data) // 4), data))
    raise ValueError(sample_width)


def clip(sample, sample_width):
    bits = sample_width * 8
    return min(max(sample, -(1 << (bits - 1))), (1 << (bits - 1)) - 1)


def make_samples(count, sample_width):
    bits = sample_width * 8
    peak = (1 << (bits - 1)) - 1
    return [int(peak * 0.6 * math.sin(i * 0.7)) for i in range(count)]


def chunk_sizes(total, size):
    return [size] * (total // size) + ([total % size] if total % size else [])


class VolumeEnvelopeConsistencyTests(unittest.TestCase):
    keypoints = [(100, -12.0), (500, 6.0), (900, -3.0), (2000, 0.0)]

    def make_pcm(self, frames, channels=2, sample_width=2):
        samples = make_samples(frames * channels, sample_width)
        return encode_pcm(samples, sample_width)

    def process_chunked(self, pcm, sizes, sample_width=2, channels=2, **kwargs):
        envelope = VolumeEnvelope(self.keypoints, **kwargs)
        frame_size = sample_width * channels
        out = []
        offset = 0
        for size in sizes:
            nbytes = size * frame_size
            out.append(
                envelope.process(
                    pcm[offset:offset + nbytes],
                    sample_width=sample_width,
                    channel_count=channels,
                )
            )
            offset += nbytes
        self.assertEqual(offset, len(pcm))
        return b"".join(out)

    def test_arbitrary_chunking_matches_whole(self):
        pcm = self.make_pcm(2500)
        whole = VolumeEnvelope(self.keypoints).process(
            pcm, sample_width=2, channel_count=2
        )
        for size in (1, 2, 3, 7, 64, 333, 1000, 2500):
            chunked = self.process_chunked(pcm, chunk_sizes(2500, size))
            self.assertEqual(chunked, whole, "chunk size %d differs" % size)

    def test_keypoints_on_chunk_boundaries(self):
        frames = 1000
        pcm = self.make_pcm(frames)
        keypoints = [(0, -6.0), (250, 6.0), (500, -6.0), (750, 6.0)]
        whole = VolumeEnvelope(keypoints).process(pcm, sample_width=2, channel_count=2)
        envelope = VolumeEnvelope(keypoints)
        out = []
        frame_size = 4
        for i in range(0, frames, 250):
            chunk = pcm[i * frame_size:(i + 250) * frame_size]
            out.append(envelope.process(chunk, sample_width=2, channel_count=2))
        self.assertEqual(b"".join(out), whole)

    def test_single_frame_chunks_match_whole(self):
        frames = 1200
        pcm = self.make_pcm(frames)
        whole = VolumeEnvelope(self.keypoints).process(
            pcm, sample_width=2, channel_count=2
        )
        chunked = self.process_chunked(pcm, [1] * frames)
        self.assertEqual(chunked, whole)

    def test_chunked_matches_whole_for_all_sample_widths(self):
        for sample_width in (1, 2, 3, 4):
            pcm = self.make_pcm(600, sample_width=sample_width)
            whole = VolumeEnvelope(self.keypoints).process(
                pcm, sample_width=sample_width, channel_count=2
            )
            chunked = self.process_chunked(
                pcm, chunk_sizes(600, 7), sample_width=sample_width
            )
            self.assertEqual(
                chunked, whole, "sample width %d differs" % sample_width
            )


class VolumeEnvelopeBehaviorTests(unittest.TestCase):
    def test_output_frame_count_matches_input(self):
        envelope = VolumeEnvelope([(0, -6.0), (100, 6.0)])
        total_in = total_out = 0
        for frames in (1, 5, 33, 128, 1000):
            pcm = encode_pcm(make_samples(frames * 2, 2), 2)
            out = envelope.process(pcm, sample_width=2, channel_count=2)
            self.assertEqual(len(out), len(pcm))
            total_in += len(pcm) // 4
            total_out += len(out) // 4
        self.assertEqual(total_in, total_out)
        self.assertEqual(envelope.position, total_in)

    def test_unselected_channels_are_unchanged(self):
        frames = 500
        samples = make_samples(frames * 2, 2)
        pcm = encode_pcm(samples, 2)
        envelope = VolumeEnvelope([(0, -24.0), (frames, 24.0)], channels=1)
        out = envelope.process(pcm, sample_width=2, channel_count=2)
        result = decode_pcm(out, 2)
        for frame in range(frames):
            self.assertEqual(result[frame * 2], samples[frame * 2])
        self.assertNotEqual(
            [result[1]], [samples[1]],
            "selected channel should have been modified",
        )

    def test_all_channels_processed_by_default(self):
        samples = [1000] * 20
        pcm = encode_pcm(samples, 2)
        envelope = VolumeEnvelope([(0, 6.0)])
        out = envelope.process(pcm, sample_width=2, channel_count=2)
        expected = int(1000 * db_to_float(6.0))
        self.assertEqual(decode_pcm(out, 2), [expected] * 20)

    def test_gain_correctness_per_sample_width(self):
        gain_db = 6.0
        factor = db_to_float(gain_db)
        for sample_width in (1, 2, 3, 4):
            samples = make_samples(64, sample_width)
            pcm = encode_pcm(samples, sample_width)
            envelope = VolumeEnvelope([(0, gain_db)])
            out = envelope.process(pcm, sample_width=sample_width, channel_count=1)
            expected = [clip(int(s * factor), sample_width) for s in samples]
            self.assertEqual(
                decode_pcm(out, sample_width), expected,
                "wrong gain for sample width %d" % sample_width,
            )

    def test_unity_gain_is_identity(self):
        for sample_width in (1, 2, 3, 4):
            pcm = encode_pcm(make_samples(64, sample_width), sample_width)
            envelope = VolumeEnvelope([(0, 0.0)])
            out = envelope.process(pcm, sample_width=sample_width, channel_count=1)
            self.assertEqual(out, pcm)

    def test_linear_db_interpolation(self):
        frames = 11
        samples = [10000] * frames
        pcm = encode_pcm(samples, 2)
        envelope = VolumeEnvelope([(0, 0.0), (10, 10.0)])
        out = envelope.process(pcm, sample_width=2, channel_count=1)
        result = decode_pcm(out, 2)
        for frame in range(frames):
            expected = int(10000 * db_to_float(float(frame)))
            self.assertEqual(result[frame], expected, "frame %d" % frame)

    def test_gain_held_outside_keypoint_range(self):
        frames = 100
        samples = [1000] * frames
        pcm = encode_pcm(samples, 2)
        envelope = VolumeEnvelope([(20, -6.0), (40, 6.0)])
        out = envelope.process(pcm, sample_width=2, channel_count=1)
        result = decode_pcm(out, 2)
        first = int(1000 * db_to_float(-6.0))
        last = int(1000 * db_to_float(6.0))
        self.assertEqual(result[:20], [first] * 20)
        self.assertEqual(result[40:], [last] * (frames - 40))
        self.assertEqual(result[20], first)
        self.assertEqual(result[40], last)

    def test_samples_saturate(self):
        pcm = encode_pcm([30000, -30000], 2)
        envelope = VolumeEnvelope([(0, 24.0)])
        out = envelope.process(pcm, sample_width=2, channel_count=1)
        self.assertEqual(decode_pcm(out, 2), [32767, -32768])

    def test_input_is_not_modified(self):
        pcm = bytearray(encode_pcm(make_samples(64, 2), 2))
        original = bytes(pcm)
        envelope = VolumeEnvelope([(0, 12.0)])
        envelope.process(pcm, sample_width=2, channel_count=2)
        self.assertEqual(bytes(pcm), original)

    def test_process_audio_segment(self):
        segment = AudioSegment(
            encode_pcm(make_samples(200, 2), 2),
            sample_width=2,
            channels=1,
            frame_rate=44100,
        )
        envelope = VolumeEnvelope([(0, 6.0)])
        out = envelope.process(segment)
        self.assertEqual(len(out), len(segment.raw_data))
        self.assertEqual(envelope.sample_width, 2)
        self.assertEqual(envelope.channel_count, 1)

    def test_gain_db_at(self):
        envelope = VolumeEnvelope([(10, 0.0), (20, 10.0)])
        self.assertEqual(envelope.gain_db_at(0), 0.0)
        self.assertEqual(envelope.gain_db_at(10), 0.0)
        self.assertEqual(envelope.gain_db_at(15), 5.0)
        self.assertEqual(envelope.gain_db_at(20), 10.0)
        self.assertEqual(envelope.gain_db_at(99), 10.0)


class VolumeEnvelopeValidationTests(unittest.TestCase):
    def test_empty_keypoints_rejected(self):
        with self.assertRaises(ValueError):
            VolumeEnvelope([])

    def test_duplicate_keypoint_positions_rejected(self):
        with self.assertRaises(ValueError):
            VolumeEnvelope([(10, 0.0), (10, 6.0)])

    def test_descending_keypoint_positions_rejected(self):
        with self.assertRaises(ValueError):
            VolumeEnvelope([(20, 0.0), (10, 6.0)])

    def test_negative_keypoint_position_rejected(self):
        with self.assertRaises(ValueError):
            VolumeEnvelope([(-1, 0.0)])

    def test_non_finite_gains_rejected(self):
        for bad in (float("inf"), float("-inf"), float("nan")):
            with self.assertRaises(ValueError):
                VolumeEnvelope([(0, bad)])

    def test_format_mismatch_rejected(self):
        envelope = VolumeEnvelope([(0, 0.0)])
        envelope.process(b"\x00" * 8, sample_width=2, channel_count=2)
        with self.assertRaises(ValueError):
            envelope.process(b"\x00" * 4, sample_width=1, channel_count=2)
        with self.assertRaises(ValueError):
            envelope.process(b"\x00" * 4, sample_width=2, channel_count=1)

    def test_partial_frame_chunk_rejected(self):
        envelope = VolumeEnvelope([(0, 0.0)])
        with self.assertRaises(ValueError):
            envelope.process(b"\x00" * 5, sample_width=2, channel_count=2)

    def test_invalid_sample_width_rejected(self):
        envelope = VolumeEnvelope([(0, 0.0)])
        with self.assertRaises(ValueError):
            envelope.process(b"\x00" * 10, sample_width=5, channel_count=2)

    def test_selected_channel_out_of_range_rejected(self):
        envelope = VolumeEnvelope([(0, 0.0)], channels=2)
        with self.assertRaises(ValueError):
            envelope.process(b"\x00" * 8, sample_width=2, channel_count=2)

    def test_missing_format_rejected(self):
        envelope = VolumeEnvelope([(0, 0.0)])
        with self.assertRaises(ValueError):
            envelope.process(b"\x00" * 8)


if __name__ == "__main__":
    unittest.main()
