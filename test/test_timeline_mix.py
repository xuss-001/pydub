import array
import math
import unittest

from pydub import AudioSegment, TimelineTrack, mix_timeline


def make_segment(samples, frame_rate=8000, channels=1, sample_width=2):
    type_code = {1: 'b', 2: 'h', 4: 'i'}[sample_width]
    return AudioSegment(
        array.array(type_code, samples),
        sample_width=sample_width,
        frame_rate=frame_rate,
        channels=channels,
    )


def samples_of(segment):
    return segment.get_array_of_samples().tolist()


class TimelineMixTests(unittest.TestCase):

    def test_zero_frame_and_interleaved_starts(self):
        first = make_segment([1000, 1000, 1000, 1000])
        second = make_segment([200, 200])
        result = AudioSegment.mix_timeline(
            [
                TimelineTrack(first, 0, [[1]]),
                (second, 2, [[1]]),
            ],
            frame_rate=8000,
            sample_width=2,
            channels=1,
        )

        self.assertEqual(result.frame_count(), 4)
        self.assertEqual(samples_of(result), [1000, 1000, 1200, 1200])

    def test_negative_weights_cancel_in_overlap(self):
        positive = make_segment([3000, -4000, 5000, -6000])
        negative = make_segment([3000, -4000, 5000, -6000])
        result = AudioSegment.mix_timeline(
            [
                (positive, 1, [[1]]),
                (negative, 1, [[-1]]),
            ],
            frame_rate=8000,
            sample_width=2,
            channels=1,
        )

        self.assertEqual(samples_of(result), [0, 0, 0, 0, 0])

    def test_three_track_overlap_saturates_only_after_summing(self):
        loud = make_segment([30000])
        # Two tracks alone would clip if each were saturated. The aggregate
        # signal is 30000, so the negative contribution must cancel first.
        result = AudioSegment.mix_timeline(
            [
                (loud, 0, [[2]]),
                (loud, 0, [[-1]]),
            ],
            frame_rate=8000,
            sample_width=2,
            channels=1,
        )
        self.assertEqual(samples_of(result), [30000])

        positive = make_segment([20000])
        result = AudioSegment.mix_timeline(
            [
                (positive, 0, [[1]]),
                (positive, 0, [[1]]),
                (positive, 0, [[1]]),
            ],
            frame_rate=8000,
            sample_width=2,
            channels=1,
        )
        self.assertEqual(samples_of(result), [32767])

    def test_tail_frames_and_input_order_independence(self):
        a = make_segment([11, 12, 13])
        b = make_segment([21, 22, 23, 24])
        c = make_segment([31, 32])

        tracks = [
            (a, 7, [[0.5]]),
            (b, 2, [[1.25]]),
            (c, 13, [[-0.25]]),
        ]
        forward = AudioSegment.mix_timeline(
            tracks, frame_rate=8000, sample_width=2, channels=1)
        reverse = AudioSegment.mix_timeline(
            list(reversed(tracks)), frame_rate=8000, sample_width=2,
            channels=1)
        rotated = AudioSegment.mix_timeline(
            tracks[1:] + tracks[:1], frame_rate=8000, sample_width=2,
            channels=1)

        expected_length = 13 + 2  # last track starts at frame 13
        self.assertEqual(forward.frame_count(), expected_length)
        self.assertEqual(forward.raw_data, reverse.raw_data)
        self.assertEqual(forward.raw_data, rotated.raw_data)
        self.assertEqual(samples_of(forward)[:2], [0, 0])
        self.assertEqual(samples_of(forward)[13:], [-8, -8])

    def test_different_rates_and_channel_layouts(self):
        mono_8k = make_segment([1000] * 80, frame_rate=8000, channels=1)
        stereo_16k = make_segment(
            [100, 200] * 10, frame_rate=16000, channels=2)

        result = AudioSegment.mix_timeline(
            [
                (mono_8k, 5, [[0.5], [0.5]]),
                (stereo_16k, 0, [[1, 0], [0, 1]]),
            ],
            frame_rate=16000,
            sample_width=2,
            channels=2,
        )

        self.assertEqual(result.frame_rate, 16000)
        self.assertEqual(result.sample_width, 2)
        self.assertEqual(result.channels, 2)
        self.assertEqual(result.frame_count(), 164)
        left = result.get_array_of_samples()[0::2]
        right = result.get_array_of_samples()[1::2]
        self.assertEqual(left[:2].tolist(), [100, 100])
        self.assertEqual(right[:2].tolist(), [200, 200])

    def test_sample_width_conversion_and_bit_depth_alias(self):
        source_16k = make_segment([1000, -1000], frame_rate=16000)
        result = mix_timeline(
            [(source_16k, 0, [[1]])],
            sample_rate=8000,
            bit_depth=8,
            channels=1,
        )

        self.assertEqual(result.frame_rate, 8000)
        self.assertEqual(result.sample_width, 1)
        self.assertEqual(result.channels, 1)
        self.assertEqual(result.frame_count(), 1)

    def test_matrix_and_start_validation(self):
        stereo = make_segment([1, 2], channels=2)

        with self.assertRaises(ValueError):
            AudioSegment.mix_timeline(
                [(stereo, 0, [[1, 0]])], frame_rate=8000,
                sample_width=2, channels=2)
        with self.assertRaises(ValueError):
            AudioSegment.mix_timeline(
                [(stereo, 0, [[1], [0]])], frame_rate=8000,
                sample_width=2, channels=2)
        with self.assertRaises(ValueError):
            AudioSegment.mix_timeline(
                [(stereo, -1, [[1, 0], [0, 1]])], frame_rate=8000,
                sample_width=2, channels=2)
        with self.assertRaises(TypeError):
            AudioSegment.mix_timeline(
                [(stereo, 0, [["invalid", 0], [0, 1]])], frame_rate=8000,
                sample_width=2, channels=2)
        with self.assertRaises(ValueError):
            AudioSegment.mix_timeline(
                [(stereo, 0, [[math.nan, 0], [0, 1]])], frame_rate=8000,
                sample_width=2, channels=2)

    def test_empty_timeline_uses_requested_format(self):
        result = AudioSegment.mix_timeline(
            [], frame_rate=44100, sample_width=2, channels=1)
        self.assertEqual(result.frame_rate, 44100)
        self.assertEqual(result.sample_width, 2)
        self.assertEqual(result.channels, 1)
        self.assertEqual(result.frame_count(), 0)


if __name__ == '__main__':
    unittest.main()
