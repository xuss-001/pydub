import unittest

from pydub import AudioSegment
from pydub.generators import Sine
from pydub.silence import (
    StreamingSilenceTrimmer,
    trim_silence,
)


class StreamingSilenceTrimmerTests(unittest.TestCase):

    frame_rate = 44100

    @classmethod
    def setUpClass(cls):
        cls.tone = Sine(
            440, sample_rate=cls.frame_rate
        ).to_audio_segment(duration=1000, volume=-6)

    def silence(self, duration):
        return AudioSegment.silent(
            duration=duration, frame_rate=self.frame_rate)

    def frames(self, duration):
        return int(self.frame_rate * duration / 1000.0)

    def feed_in_chunks(self, audio, chunk_sizes, **kwargs):
        trimmer = StreamingSilenceTrimmer(**kwargs)
        output = AudioSegment.empty()
        ranges = []
        position = 0
        chunk_index = 0
        total_frames = int(audio.frame_count())

        while position < total_frames:
            size = chunk_sizes[chunk_index % len(chunk_sizes)]
            size = min(size, total_frames - position)
            piece, piece_ranges = trimmer.feed(
                audio.get_sample_slice(position, position + size))
            output += piece
            ranges.extend(piece_ranges)
            position += size
            chunk_index += 1

        piece, piece_ranges = trimmer.flush()
        output += piece
        ranges.extend(piece_ranges)
        return output, ranges

    def test_silence_across_chunks_has_stable_frame_ranges(self):
        audio = (
            self.tone +
            self.silence(2000) +
            self.tone
        )
        params = dict(
            min_silence_len=500,
            silence_thresh=-30,
            keep_silence=100,
            seek_step=10,
        )

        result = trim_silence(audio, **params)

        self.assertEqual(result.ranges, [
            [0, self.frames(1100), 0, self.frames(1100)],
            [
                self.frames(2900),
                self.frames(4000),
                self.frames(1100),
                self.frames(2200),
            ],
        ])
        self.assertEqual(
            result.audio.raw_data,
            (
                audio.get_sample_slice(0, self.frames(1100)).raw_data +
                audio.get_sample_slice(
                    self.frames(2900), self.frames(4000)).raw_data
            ),
        )

    def test_chunk_boundaries_do_not_change_audio_or_ranges(self):
        audio = (
            self.tone +
            self.silence(1500) +
            self.tone +
            self.silence(700) +
            self.tone
        )
        params = dict(
            min_silence_len=400,
            silence_thresh=-30,
            keep_silence=50,
            seek_step=7,
        )
        expected = trim_silence(audio, **params)

        for chunk_sizes in (
            [1],
            [17, 3, 101, 2],
            [441, 4410, 13, 999],
            [int(audio.frame_count())],
        ):
            output, ranges = self.feed_in_chunks(
                audio, chunk_sizes, **params)
            self.assertEqual(output.raw_data, expected.audio.raw_data)
            self.assertEqual(ranges, expected.ranges)

    def test_silence_shorter_than_minimum_is_not_removed(self):
        audio = self.tone + self.silence(200) + self.tone

        result = trim_silence(
            audio,
            min_silence_len=500,
            silence_thresh=-30,
            keep_silence=100,
            seek_step=10,
        )

        self.assertEqual(result.audio.raw_data, audio.raw_data)
        self.assertEqual(
            result.ranges,
            [[0, int(audio.frame_count()),
              0, int(audio.frame_count())]],
        )

    def test_short_sound_surrounded_by_silence_is_preserved(self):
        sound = Sine(
            440, sample_rate=self.frame_rate
        ).to_audio_segment(duration=20, volume=-6)
        audio = self.silence(1000) + sound + self.silence(1000)

        result = trim_silence(
            audio,
            min_silence_len=200,
            silence_thresh=-30,
            keep_silence=20,
            seek_step=10,
        )

        expected = audio.get_sample_slice(
            self.frames(980), self.frames(1040))
        self.assertEqual(result.audio.raw_data, expected.raw_data)
        self.assertEqual(result.ranges, [[
            self.frames(980),
            self.frames(1040),
            0,
            self.frames(60),
        ]])

    def test_tail_shorter_than_detection_window_is_kept(self):
        audio = self.tone[:300] + self.silence(100)

        output, ranges = self.feed_in_chunks(
            audio,
            [700, 31, 5000],
            min_silence_len=500,
            silence_thresh=-30,
            keep_silence=100,
            seek_step=10,
        )

        self.assertEqual(output.raw_data, audio.raw_data)
        self.assertEqual(
            ranges,
            [[0, int(audio.frame_count()),
              0, int(audio.frame_count())]],
        )

    def test_short_tone_after_long_silence_is_kept(self):
        audio = self.tone[:100] + self.silence(10000) + self.tone[:100]

        result = trim_silence(
            audio,
            min_silence_len=500,
            silence_thresh=-30,
            keep_silence=100,
            seek_step=10,
        )

        self.assertEqual(result.ranges, [
            [0, self.frames(200), 0, self.frames(200)],
            [
                self.frames(10000),
                self.frames(10200),
                self.frames(200),
                self.frames(400),
            ],
        ])
        self.assertEqual(
            result.audio.raw_data,
            (
                audio.get_sample_slice(0, self.frames(200)).raw_data +
                audio.get_sample_slice(
                    self.frames(10000), self.frames(10200)).raw_data
            ),
        )

    def test_all_silence_is_removed_and_flush_is_complete(self):
        audio = self.silence(2000)

        trimmer = StreamingSilenceTrimmer(
            min_silence_len=500,
            silence_thresh=-30,
            keep_silence=100,
            seek_step=10,
        )
        first, first_ranges = trimmer.feed(audio[:350])
        trimmer.feed(audio[350:])
        last, last_ranges = trimmer.flush()

        self.assertEqual(first.frame_count(), 0)
        self.assertEqual(last.frame_count(), 0)
        self.assertEqual(first_ranges + last_ranges, [])
        self.assertEqual(trimmer.total_output_frames, 0)

    def test_all_silence_shorter_than_window_is_unchanged(self):
        audio = self.silence(300)

        result = trim_silence(
            audio,
            min_silence_len=500,
            silence_thresh=-30,
            keep_silence=100,
            seek_step=10,
        )

        self.assertEqual(result.audio.raw_data, audio.raw_data)
        self.assertEqual(
            result.ranges,
            [[0, int(audio.frame_count()),
              0, int(audio.frame_count())]],
        )

    def test_44100_hz_frame_mapping_is_half_open(self):
        audio = (
            self.tone.get_sample_slice(0, 44100) +
            self.silence(1000) +
            self.tone.get_sample_slice(0, 22050)
        )
        result = trim_silence(
            audio,
            min_silence_len=200,
            silence_thresh=-30,
            keep_silence=20,
            seek_step=10,
        )

        self.assertEqual(int(audio.frame_count()), 110250)
        self.assertEqual(result.ranges, [
            [0, 44982, 0, 44982],
            [87318, 110250, 44982, 67914],
        ])
        self.assertEqual(int(result.audio.frame_count()), 67914)
        self.assertEqual(
            result.audio.get_sample_slice(0, 44982).raw_data,
            audio.get_sample_slice(0, 44982).raw_data,
        )
        self.assertEqual(
            result.audio.get_sample_slice(44982, 67914).raw_data,
            audio.get_sample_slice(87318, 110250).raw_data,
        )

    def test_internal_buffer_stays_bounded_for_long_speech(self):
        trimmer = StreamingSilenceTrimmer(
            min_silence_len=500,
            silence_thresh=-30,
            keep_silence=100,
            seek_step=10,
        )
        max_buffer_bytes = 0

        for _ in range(200):
            trimmer.feed(self.tone[:100])
            max_buffer_bytes = max(max_buffer_bytes, len(trimmer._buffer))

        trimmer.flush()
        self.assertLess(
            max_buffer_bytes,
            12 * self.frame_rate * self.tone.frame_width,
        )

    def test_internal_buffer_stays_bounded_for_long_silence(self):
        trimmer = StreamingSilenceTrimmer(
            min_silence_len=500,
            silence_thresh=-30,
            keep_silence=100,
            seek_step=10,
        )
        max_buffer_bytes = 0

        for _ in range(200):
            trimmer.feed(self.silence(100))
            max_buffer_bytes = max(max_buffer_bytes, len(trimmer._buffer))

        result = trimmer.flush()
        self.assertEqual(result.audio.frame_count(), 0)
        self.assertEqual(result.ranges, [])
        self.assertLess(
            max_buffer_bytes,
            3 * self.frame_rate * self.tone.frame_width,
        )


if __name__ == "__main__":
    unittest.main()
