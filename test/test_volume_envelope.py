import math
import random
import unittest

from pydub import AudioSegment
from pydub.volume_envelope import (
    InvalidVolumeEnvelope,
    VolumeEnvelopeProcessor,
)


def gain_db_at(frame, keypoints):
    if frame <= keypoints[0][0]:
        return keypoints[0][1]
    if frame >= keypoints[-1][0]:
        return keypoints[-1][1]
    for (start_frame, start_gain), (end_frame, end_gain) in zip(
            keypoints, keypoints[1:]):
        if start_frame <= frame <= end_frame:
            fraction = (frame - start_frame) / (end_frame - start_frame)
            return start_gain + (end_gain - start_gain) * fraction
    raise AssertionError("frame is outside the envelope")


def expected_pcm(data, sample_width, channels, keypoints, selected_channels=None):
    frame_width = sample_width * channels
    output = bytearray(data)
    bits = sample_width * 8
    minimum = -(1 << (bits - 1))
    maximum = (1 << (bits - 1)) - 1

    if selected_channels is None:
        selected_channels = range(channels)

    for frame in range(len(data) // frame_width):
        factor = 10 ** (gain_db_at(frame, keypoints) / 20.0)
        for channel in selected_channels:
            offset = frame * frame_width + channel * sample_width
            sample = int.from_bytes(
                data[offset:offset + sample_width], "little", signed=True
            )
            sample = max(minimum, min(maximum, math.floor(sample * factor)))
            output[offset:offset + sample_width] = sample.to_bytes(
                sample_width, "little", signed=True
            )

    return bytes(output)


def pcm_from_samples(samples, sample_width):
    return b"".join(
        sample.to_bytes(sample_width, "little", signed=True)
        for sample in samples
    )


class VolumeEnvelopeProcessorTests(unittest.TestCase):

    def test_linear_db_interpolation_and_edge_holds(self):
        samples = [10000, 10000, 10000, 10000, 10000, 10000, 10000, 10000]
        data = pcm_from_samples(samples, 2)
        keypoints = [(2, -6.0), (4, 0.0), (6, -6.0)]
        processor = VolumeEnvelopeProcessor(
            keypoints, sample_width=2, channels=1
        )

        output = processor.process(data)

        self.assertEqual(output, expected_pcm(data, 2, 1, keypoints))
        values = [
            int.from_bytes(output[i:i + 2], "little", signed=True)
            for i in range(0, len(output), 2)
        ]
        self.assertEqual(values[0], values[1])
        self.assertGreater(values[3], values[1])
        self.assertGreater(values[3], values[2])
        self.assertEqual(values[4], 10000)
        self.assertEqual(values[5], values[3])
        self.assertEqual(values[6], values[2])
        self.assertEqual(values[7], values[6])

    def test_keypoints_on_chunk_boundaries_match_whole_buffer(self):
        keypoints = [(0, -6.0), (10, 0.0), (20, -3.0), (30, 3.0)]
        data = pcm_from_samples([1000 * (i + 1) for i in range(32)], 2)

        whole = VolumeEnvelopeProcessor(
            keypoints, sample_width=2, channels=1
        ).process(data)

        processor = VolumeEnvelopeProcessor(
            keypoints, sample_width=2, channels=1
        )
        chunked = b"".join([
            processor.process(data[:20]),
            processor.process(data[20:40]),
            processor.process(data[40:60]),
            processor.process(data[60:]),
        ])

        self.assertEqual(chunked, whole)
        self.assertEqual(expected_pcm(data, 2, 1, keypoints), whole)

    def test_one_frame_per_chunk_matches_whole_buffer(self):
        keypoints = [(0, -12.0), (1, 6.0), (5, -6.0), (9, 0.0)]
        data = pcm_from_samples(list(range(-10, 10)), 2)

        whole = VolumeEnvelopeProcessor(
            keypoints, sample_width=2, channels=1
        ).process(data)

        processor = VolumeEnvelopeProcessor(
            keypoints, sample_width=2, channels=1
        )
        chunked = b"".join(
            processor.process(data[offset:offset + 2])
            for offset in range(0, len(data), 2)
        )

        self.assertEqual(chunked, whole)
        self.assertEqual(processor.frames_processed, len(data) // 2)

    def test_arbitrary_chunking_matches_whole_buffer(self):
        random.seed(7)
        keypoints = [(0, -9.0), (13, 4.5), (37, -2.0), (61, 1.0)]
        channels = 2
        frames = 80
        samples = [
            random.randint(-20000, 20000)
            for _ in range(frames * channels)
        ]
        data = pcm_from_samples(samples, 2)

        whole = VolumeEnvelopeProcessor(
            keypoints, sample_width=2, channels=channels
        ).process(data)

        processor = VolumeEnvelopeProcessor(
            keypoints, sample_width=2, channels=channels
        )
        output = []
        offset = 0
        frame_sizes = [1, 2, 5, 13, 3, 8]
        index = 0
        while offset < frames:
            count = min(frame_sizes[index % len(frame_sizes)], frames - offset)
            byte_count = count * 2 * channels
            output.append(processor.process(data[offset * 4:offset * 4 + byte_count]))
            offset += count
            index += 1

        self.assertEqual(b"".join(output), whole)

    def test_unselected_channel_is_byte_for_byte_unchanged(self):
        frames = 25
        samples = []
        for frame in range(frames):
            samples.extend([frame * 101 - 1000, frame * -37 + 900])
        data = pcm_from_samples(samples, 2)
        keypoints = [(0, -6.0), (24, 6.0)]

        left_processor = VolumeEnvelopeProcessor(
            keypoints, channel=1, sample_width=2, channels=2
        )
        left_output = left_processor.process(data)
        self.assertEqual(left_output, expected_pcm(data, 2, 2, keypoints, (0,)))

        right_processor = VolumeEnvelopeProcessor(
            keypoints, channel=2, sample_width=2, channels=2
        )
        right_output = right_processor.process(data)
        self.assertEqual(right_output, expected_pcm(data, 2, 2, keypoints, (1,)))

        for frame in range(frames):
            input_offset = frame * 4
            self.assertEqual(
                left_output[input_offset + 2:input_offset + 4],
                data[input_offset + 2:input_offset + 4],
            )
            self.assertEqual(
                right_output[input_offset:input_offset + 2],
                data[input_offset:input_offset + 2],
            )

    def test_gain_is_correct_at_every_supported_sample_width(self):
        keypoints = [(0, 6.0)]
        for sample_width in (1, 2, 3, 4):
            bits = sample_width * 8
            minimum = -(1 << (bits - 1))
            maximum = (1 << (bits - 1)) - 1
            samples = [minimum, minimum + 1, -100, -1, 0, 1, 100, maximum]
            data = pcm_from_samples(samples, sample_width)

            processor = VolumeEnvelopeProcessor(
                keypoints, sample_width=sample_width, channels=1
            )
            output = processor.process(data)

            self.assertEqual(
                output, expected_pcm(data, sample_width, 1, keypoints)
            )
            self.assertEqual(
                len(output) // sample_width,
                len(data) // sample_width,
            )

    def test_constant_envelope_matches_existing_apply_gain(self):
        for sample_width in (1, 2, 4):
            bits = sample_width * 8
            minimum = -(1 << (bits - 1))
            maximum = (1 << (bits - 1)) - 1
            magnitude = min(1000, maximum)
            samples = [
                minimum, minimum + 1, -magnitude, -1, 0, 1, magnitude, maximum
            ]
            data = pcm_from_samples(samples, sample_width)
            segment = AudioSegment(
                data=data,
                sample_width=sample_width,
                frame_rate=8000,
                channels=1,
            )

            for gain in (-12.0, -6.0, 0.0, 3.5, 6.0):
                self.assertEqual(
                    segment.apply_volume_envelope([(0, gain)]).raw_data,
                    segment.apply_gain(gain).raw_data,
                )

    def test_output_frame_count_always_matches_input(self):
        keypoints = [(0, -3.0), (7, 3.0)]
        data = pcm_from_samples(list(range(60)), 2)
        processor = VolumeEnvelopeProcessor(
            keypoints, sample_width=2, channels=1
        )

        total = 0
        sizes = (3, 1, 8, 2, 11, 5)
        index = 0
        while total < len(data) // 2:
            size = sizes[index % len(sizes)]
            start = total * 2
            end = min(start + size * 2, len(data))
            output = processor.process(data[start:end])
            self.assertEqual(len(output), len(data[start:end]))
            total += len(output) // 2
            index += 1

        self.assertEqual(total, len(data) // 2)
        self.assertEqual(processor.position, len(data) // 2)

    def test_saturation_matches_pydub_rules(self):
        data = pcm_from_samples([30000, -30000, 0], 2)
        processor = VolumeEnvelopeProcessor(
            [(0, 6.0)], sample_width=2, channels=1
        )
        output = processor.process(data)
        values = [
            int.from_bytes(output[i:i + 2], "little", signed=True)
            for i in range(0, len(output), 2)
        ]
        self.assertEqual(values, [32767, -32768, 0])

    def test_input_buffer_is_not_modified(self):
        data = bytearray(pcm_from_samples([1000, -2000, 3000], 2))
        original = bytes(data)
        processor = VolumeEnvelopeProcessor(
            [(0, 6.0)], sample_width=2, channels=1
        )
        processor.process(data)
        self.assertEqual(bytes(data), original)

    def test_invalid_keypoints_are_rejected(self):
        with self.assertRaises(InvalidVolumeEnvelope):
            VolumeEnvelopeProcessor([(10, 0.0), (5, -3.0)])
        with self.assertRaises(InvalidVolumeEnvelope):
            VolumeEnvelopeProcessor([(10, 0.0), (10, -3.0)])
        with self.assertRaises(InvalidVolumeEnvelope):
            VolumeEnvelopeProcessor([(0, float("nan"))])
        with self.assertRaises(InvalidVolumeEnvelope):
            VolumeEnvelopeProcessor([(0, float("inf"))])
        with self.assertRaises(InvalidVolumeEnvelope):
            VolumeEnvelopeProcessor([(0, float("-inf"))])

    def test_first_chunk_locks_format_and_later_mismatch_is_rejected(self):
        processor = VolumeEnvelopeProcessor([(0, 0.0)])
        processor.process(b"\x00\x00", sample_width=2, channels=1)

        with self.assertRaises(InvalidVolumeEnvelope):
            processor.process(b"\x00", sample_width=1, channels=1)
        with self.assertRaises(InvalidVolumeEnvelope):
            processor.process(b"\x00\x00\x00\x00", sample_width=2, channels=2)

    def test_first_chunk_can_supply_format_as_a_tuple(self):
        data = pcm_from_samples([100, -200, 300], 2)
        processor = VolumeEnvelopeProcessor([(0, 0.0)])
        self.assertEqual(processor.process(data, format=(2, 1, 8000)), data)
        self.assertEqual(processor.sample_width, 2)
        self.assertEqual(processor.channels, 1)
        self.assertEqual(processor.frame_rate, 8000)

    def test_zero_based_channel_index_alias(self):
        data = pcm_from_samples([1000, 2000, -1000, -2000], 2)
        processor = VolumeEnvelopeProcessor(
            [(0, 0.0)], channel_index=1, sample_width=2, channels=2
        )
        output = processor.process(data)
        self.assertEqual(output[:2], data[:2])
        self.assertEqual(output[2:], data[2:])

    def test_chunk_must_contain_whole_frames(self):
        processor = VolumeEnvelopeProcessor(
            [(0, 0.0)], sample_width=2, channels=2
        )
        with self.assertRaises(InvalidVolumeEnvelope):
            processor.process(b"\x00\x00\x00")

    def test_audio_segment_convenience_matches_streaming_processor(self):
        samples = list(range(-200, 200))
        segment = AudioSegment(
            data=pcm_from_samples(samples, 2),
            sample_width=2,
            frame_rate=8000,
            channels=1,
        )
        keypoints = [(0, -6.0), (100, 0.0), (250, -3.0)]

        envelope_applied = segment.apply_volume_envelope(keypoints)
        streamed = VolumeEnvelopeProcessor(
            keypoints,
            sample_width=segment.sample_width,
            channels=segment.channels,
            frame_rate=segment.frame_rate,
        ).process(segment)

        self.assertEqual(envelope_applied.raw_data, streamed.raw_data)
        self.assertEqual(envelope_applied.frame_count(), segment.frame_count())


if __name__ == "__main__":
    unittest.main()
