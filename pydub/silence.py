"""
Various functions for finding/manipulating silence in AudioSegments
"""
import itertools

from .audio_segment import AudioSegment
from .utils import audioop, db_to_float


class FrameRange(object):
    """
    A retained section, expressed in sample frames.

    The ranges are half-open: the start frame is included and the end frame is
    not. ``original_*`` refers to the complete input audio and ``output_*``
    refers to the trimmed audio produced so far.
    """

    __slots__ = (
        "original_start",
        "original_end",
        "output_start",
        "output_end",
    )

    _ALIASES = {
        "start": "original_start",
        "end": "original_end",
        "start_frame": "original_start",
        "end_frame": "original_end",
        "original_start_frame": "original_start",
        "original_end_frame": "original_end",
        "input_start": "original_start",
        "input_end": "original_end",
        "input_start_frame": "original_start",
        "input_end_frame": "original_end",
        "source_start": "original_start",
        "source_end": "original_end",
        "source_start_frame": "original_start",
        "source_end_frame": "original_end",
        "from_start": "original_start",
        "from_end": "original_end",
        "trimmed_start": "output_start",
        "trimmed_end": "output_end",
        "output_start_frame": "output_start",
        "output_end_frame": "output_end",
        "out_start": "output_start",
        "out_end": "output_end",
    }

    def __init__(self, original_start, original_end, output_start, output_end):
        self.original_start = int(original_start)
        self.original_end = int(original_end)
        self.output_start = int(output_start)
        self.output_end = int(output_end)

    def __iter__(self):
        return iter((
            self.original_start,
            self.original_end,
            self.output_start,
            self.output_end,
        ))

    def __len__(self):
        return 4

    def __getitem__(self, item):
        if isinstance(item, str):
            if item in ("original", "input", "source"):
                return self.original
            if item in ("output", "trimmed"):
                return self.output
            item = self._ALIASES.get(item, item)
            if item in self.__slots__:
                return getattr(self, item)
            raise KeyError(item)
        return tuple(self)[item]

    def __eq__(self, other):
        if isinstance(other, FrameRange):
            return tuple(self) == tuple(other)
        if isinstance(other, dict):
            canonical = self.as_dict()
            if len(other) != len(canonical):
                return False
            for key, value in canonical.items():
                aliases = [
                    name for name, target in self._ALIASES.items()
                    if target == key
                ]
                found = False
                for candidate in [key] + aliases:
                    if candidate in other:
                        if other[candidate] != value:
                            return False
                        found = True
                        break
                if not found:
                    return False
            return True
        try:
            return tuple(self) == tuple(other)
        except TypeError:
            return False

    def __ne__(self, other):
        return not self == other

    def __hash__(self):
        return hash(tuple(self))

    def __repr__(self):
        return (
            "FrameRange(original_start={0}, original_end={1}, "
            "output_start={2}, output_end={3})"
        ).format(*tuple(self))

    @property
    def original(self):
        return (self.original_start, self.original_end)

    @property
    def output(self):
        return (self.output_start, self.output_end)

    @property
    def input(self):
        return self.original

    @property
    def start(self):
        return self.original_start

    @property
    def end(self):
        return self.original_end

    @property
    def start_frame(self):
        return self.original_start

    @property
    def end_frame(self):
        return self.original_end

    @property
    def input_start(self):
        return self.original_start

    @property
    def input_end(self):
        return self.original_end

    @property
    def source_start(self):
        return self.original_start

    @property
    def source_end(self):
        return self.original_end

    @property
    def trimmed(self):
        return self.output

    @property
    def trimmed_start(self):
        return self.output_start

    @property
    def trimmed_end(self):
        return self.output_end

    def as_dict(self):
        return {
            "original_start": self.original_start,
            "original_end": self.original_end,
            "output_start": self.output_start,
            "output_end": self.output_end,
        }

    to_dict = as_dict

    def keys(self):
        return self.as_dict().keys()

    def items(self):
        return self.as_dict().items()

    def get(self, key, default=None):
        try:
            return self[key]
        except KeyError:
            return default


class TrimResult(object):
    """
    Audio completed during one streaming call and the matching frame ranges.

    The object is intentionally unpackable as ``audio, ranges = result``.
    """

    __slots__ = ("audio", "ranges")

    def __init__(self, audio, ranges):
        self.audio = audio
        self.ranges = list(ranges)

    def __iter__(self):
        return iter((self.audio, self.ranges))

    def __len__(self):
        return 2

    def __getitem__(self, item):
        if isinstance(item, str):
            if item in ("audio", "segment", "output"):
                return self.audio
            if item in ("ranges", "segments", "frame_ranges"):
                return self.ranges
            raise KeyError(item)
        return (self.audio, self.ranges)[item]

    def __repr__(self):
        return "TrimResult(audio={0!r}, ranges={1!r})".format(
            self.audio, self.ranges)

    def __eq__(self, other):
        if isinstance(other, TrimResult):
            return (
                self.audio.raw_data == other.audio.raw_data and
                self.ranges == other.ranges
            )
        try:
            return (self.audio, self.ranges) == tuple(other)
        except TypeError:
            return False

    def __ne__(self, other):
        return not self == other

    @property
    def segment(self):
        return self.audio

    @property
    def segments(self):
        return self.ranges

    @property
    def frame_ranges(self):
        return self.ranges


class StreamingSilenceTrimmer(object):
    """
    Incrementally remove silence from same-format AudioSegment chunks.

    Silence is identified with the same sliding-window rule used by
    :func:`detect_silence`: a window is silent when its RMS is at or below
    ``silence_thresh``. Candidate windows overlap and advance by
    ``seek_step``. Detection state is kept in absolute sample frames, so a
    silence spanning chunk boundaries is treated exactly like a silence in a
    single segment.

    ``feed()`` returns a :class:`TrimResult` containing audio that is complete
    at that point. ``flush()`` evaluates the final available window and returns
    the remaining audio. The returned ranges are half-open sample-frame ranges
    in both the original input and the concatenated output.
    """

    def __init__(self, min_silence_len=1000, silence_thresh=-16,
                 keep_silence=100, seek_step=1, sample_width=None,
                 frame_rate=None, channels=None, **kwargs):
        min_silence_len = self._alias(
            kwargs, ("min_silence", "min_silence_length", "window_len",
                     "window_ms", "min_silence_ms"), min_silence_len)
        silence_thresh = self._alias(
            kwargs, ("threshold", "threshold_dbfs", "silence_threshold",
                     "silence_thresh_dbfs", "threshold_db", "rms_threshold",
                     "rms_threshold_db"), silence_thresh)
        keep_silence = self._alias(
            kwargs, ("padding", "keep", "keep_silence_len",
                     "keep_silence_ms"), keep_silence)
        seek_step = self._alias(
            kwargs, ("step", "seek", "seek_step_ms"), seek_step)
        frame_rate = self._alias(
            kwargs, ("rate", "sample_rate"), frame_rate)

        if kwargs:
            unexpected = sorted(kwargs)
            raise TypeError(
                "Unexpected keyword argument(s): {0}".format(
                    ", ".join(unexpected)))

        if min_silence_len <= 0:
            raise ValueError("min_silence_len must be greater than zero")
        if seek_step <= 0:
            raise ValueError("seek_step must be greater than zero")

        if isinstance(keep_silence, bool):
            keep_silence = float("inf") if keep_silence else 0
        elif keep_silence < 0:
            raise ValueError("keep_silence must not be negative")

        self.min_silence_len = min_silence_len
        self.silence_thresh = silence_thresh
        self.keep_silence = keep_silence
        self.seek_step = seek_step
        self.sample_width = sample_width
        self.frame_rate = frame_rate
        self.channels = channels

        self._template = None
        self._window_frames = None
        self._step_frames = None
        self._keep_frames = None
        self._silence_amplitude = None
        self._block_frames = None

        self._buffer = bytearray()
        self._buffer_start = 0
        self._sample_count = 0

        self._next_window_start = 0
        self._last_evaluated_start = None
        self._active_silence_start = None
        self._last_silent_window_start = None

        self._run_start = 0
        self._emit_frontier = 0
        self._output_frames = 0
        self._pending_audio = []
        self._pending_ranges = []
        self._finalized = False

    @staticmethod
    def _alias(kwargs, names, default):
        value = default
        for name in names:
            if name in kwargs:
                value = kwargs.pop(name)
        return value

    @property
    def total_input_frames(self):
        return self._sample_count

    @property
    def total_output_frames(self):
        return self._output_frames

    @property
    def output_frames(self):
        return self._output_frames

    @property
    def input_frames(self):
        return self._sample_count

    def _init_format(self, chunk):
        if self._template is None:
            self._template = chunk._spawn(b"")

            if self.sample_width is not None and \
                    self.sample_width != chunk.sample_width:
                raise ValueError("sample_width does not match the input")
            if self.frame_rate is not None and \
                    self.frame_rate != chunk.frame_rate:
                raise ValueError("frame_rate does not match the input")
            if self.channels is not None and \
                    self.channels != chunk.channels:
                raise ValueError("channels does not match the input")

            self.sample_width = chunk.sample_width
            self.frame_rate = chunk.frame_rate
            self.channels = chunk.channels

            self._window_frames = int(
                chunk.frame_count(ms=self.min_silence_len))
            if self._window_frames < 1:
                raise ValueError(
                    "min_silence_len must be at least one sample frame")

            self._step_frames = max(
                1, int(chunk.frame_count(ms=self.seek_step)))
            self._keep_frames = (
                float("inf") if self.keep_silence == float("inf")
                else int(chunk.frame_count(ms=self.keep_silence))
            )
            self._silence_amplitude = (
                db_to_float(self.silence_thresh) *
                chunk.max_possible_amplitude
            )

            # Keeping a fixed block size makes emitted range boundaries
            # independent of how the caller happens to split its input. The
            # cap also bounds the amount held while a long, uninterrupted
            # sound is still waiting for its ending boundary.
            self._block_frames = max(
                self._window_frames,
                int(chunk.frame_count(ms=10000)),
            )

    def _check_format(self, chunk):
        if self._template is None:
            return
        if chunk.sample_width != self.sample_width or \
                chunk.frame_rate != self.frame_rate or \
                chunk.channels != self.channels:
            raise ValueError(
                "All audio chunks must have the same sample_width, "
                "frame_rate, and channels")

    def _get_data(self, start, end):
        start_byte = (start - self._buffer_start) * self._template.frame_width
        end_byte = (end - self._buffer_start) * self._template.frame_width
        if start_byte < 0 or end_byte > len(self._buffer):
            raise RuntimeError("internal audio buffer was pruned too early")
        return bytes(self._buffer[start_byte:end_byte])

    def _emit_span(self, start, end):
        while start < end:
            if start < self._emit_frontier:
                raise RuntimeError("retained ranges must not overlap")

            # Split at absolute block boundaries. That makes the range list
            # independent of when feed() happens to make audio safe to emit.
            block_end = (
                (start // self._block_frames) + 1
            ) * self._block_frames
            piece_end = min(end, block_end)

            data = self._get_data(start, piece_end)
            length = piece_end - start
            frame_range = FrameRange(
                start, piece_end,
                self._output_frames,
                self._output_frames + length,
            )

            self._pending_audio.append(data)
            self._pending_ranges.append(frame_range)
            self._output_frames += length
            self._emit_frontier = piece_end
            start = piece_end

    def _discard_to(self, frame):
        if frame > self._sample_count:
            frame = self._sample_count
        if frame > self._emit_frontier:
            self._emit_frontier = frame
        self._run_start = frame

    def _close_silence(self, end, final=False):
        start = self._active_silence_start
        self._active_silence_start = None
        self._last_silent_window_start = None

        if start is None:
            return

        keep = self._keep_frames

        if final:
            # There is no following non-silent range, so only the silence
            # adjacent to the preceding retained range can be kept.
            if start > self._run_start:
                left_end = min(start + keep, end)
                if left_end > self._emit_frontier:
                    self._emit_span(self._emit_frontier, left_end)
            return

        if start <= self._run_start:
            # Leading silence, or silence immediately following a cut. There
            # is no retained audio to the left.
            next_start = max(self._run_start, end - keep)
            self._discard_to(next_start)
            return

        left_end = start + keep
        next_start = end - keep

        if next_start <= left_end:
            # The two padded ranges overlap. Keeping the union avoids both a
            # duplicate sample and an unnecessary split.
            return

        if left_end > self._emit_frontier:
            self._emit_span(self._emit_frontier, left_end)

        # The middle of this silence is removed. The right padded range will
        # start at next_start after the current run is closed.
        self._discard_to(next_start)

    def _advance_active_silence(self):
        """
        Release the middle of an active silence as soon as it cannot overlap
        either of the padded retained ranges.
        """
        if self._active_silence_start is None or \
                self._last_silent_window_start is None or \
                self._keep_frames == float("inf"):
            return

        start = self._active_silence_start
        known_end = (
            self._last_silent_window_start + self._window_frames)
        keep = self._keep_frames

        if known_end - start <= 2 * keep:
            return

        if start > self._run_start:
            left_end = start + keep
            if left_end > self._emit_frontier:
                self._emit_span(self._emit_frontier, left_end)

        # Only these trailing frames can become part of a following retained
        # range. Everything before them is definitely removed.
        self._discard_to(known_end - keep)

    def _evaluate_window(self, start):
        end = start + self._window_frames
        data = self._get_data(start, end)
        silent = audioop.rms(data, self.sample_width) <= self._silence_amplitude
        self._last_evaluated_start = start

        if silent:
            if self._active_silence_start is None:
                self._active_silence_start = start
                self._last_silent_window_start = start
            elif start > self._last_silent_window_start + self._window_frames:
                self._close_silence(
                    self._last_silent_window_start + self._window_frames)
                self._active_silence_start = start
                self._last_silent_window_start = start
            else:
                self._last_silent_window_start = start
        elif self._active_silence_start is not None and \
                start > self._last_silent_window_start + self._window_frames:
            self._close_silence(
                self._last_silent_window_start + self._window_frames)

    def _analyse_available(self):
        while self._sample_count >= \
                self._next_window_start + self._window_frames:
            start = self._next_window_start
            self._evaluate_window(start)
            self._next_window_start += self._step_frames

    def _analyse_tail(self):
        if self._sample_count < self._window_frames:
            return

        tail_start = self._sample_count - self._window_frames
        if tail_start != self._last_evaluated_start:
            self._evaluate_window(tail_start)

    def _safe_output_end(self):
        if self._active_silence_start is not None:
            return min(self._active_silence_start, self._sample_count)
        return min(self._next_window_start, self._sample_count)

    def _emit_safe_audio(self):
        safe_end = self._safe_output_end()
        if safe_end <= self._emit_frontier:
            return

        limit = (safe_end // self._block_frames) * self._block_frames
        if limit > self._emit_frontier:
            self._emit_span(self._emit_frontier, limit)

    def _prune_buffer(self):
        keep_from = min(self._emit_frontier, self._next_window_start)
        if self._sample_count >= self._window_frames:
            keep_from = min(
                keep_from, self._sample_count - self._window_frames)
        keep_from = max(0, keep_from)

        if keep_from > self._buffer_start:
            byte_count = (
                (keep_from - self._buffer_start) *
                self._template.frame_width
            )
            del self._buffer[:byte_count]
            self._buffer_start = keep_from

    def _take_result(self):
        if self._template is None:
            audio = AudioSegment.empty()
        else:
            audio = self._template._spawn(b"".join(self._pending_audio))

        result = TrimResult(audio, self._pending_ranges)
        self._pending_audio = []
        self._pending_ranges = []
        return result

    def feed(self, chunk):
        """
        Accept the next same-format AudioSegment chunk.

        Returns the audio and half-open source/output frame ranges that became
        safe to emit while processing this chunk.
        """
        if self._finalized:
            raise RuntimeError("feed() cannot be used after flush()")
        if not isinstance(chunk, AudioSegment):
            raise TypeError("chunk must be an AudioSegment")

        self._check_format(chunk)
        self._init_format(chunk)
        self._check_format(chunk)

        if chunk._data:
            self._buffer.extend(chunk._data)
            self._sample_count += int(chunk.frame_count())
            self._analyse_available()
            self._advance_active_silence()
            self._emit_safe_audio()
            self._prune_buffer()

        return self._take_result()

    def flush(self):
        """
        Finish the final window, emit the remaining tail, and close the stream.
        """
        if self._finalized:
            return self._take_result()

        self._finalized = True

        if self._template is not None:
            self._analyse_available()
            self._advance_active_silence()
            self._analyse_tail()

            ended_in_trailing_silence = False
            if self._active_silence_start is not None:
                silence_end = (
                    self._last_silent_window_start + self._window_frames)
                if silence_end >= self._sample_count:
                    self._close_silence(self._sample_count, final=True)
                    ended_in_trailing_silence = True
                else:
                    self._close_silence(silence_end)

            if self._active_silence_start is None and \
                    not ended_in_trailing_silence and \
                    self._emit_frontier < self._sample_count:
                self._emit_span(self._emit_frontier, self._sample_count)

            self._prune_buffer()

        return self._take_result()

    finish = flush
    finalize = flush
    close = flush
    process = feed
    push = feed
    write = feed
    add_chunk = feed
    append = feed


def trim_silence(audio_segment, min_silence_len=1000, silence_thresh=-16,
                 keep_silence=100, seek_step=1):
    """
    One-shot convenience wrapper around :class:`StreamingSilenceTrimmer`.

    Returns a :class:`TrimResult` with the trimmed audio and half-open
    original/output sample-frame ranges.
    """
    trimmer = StreamingSilenceTrimmer(
        min_silence_len=min_silence_len,
        silence_thresh=silence_thresh,
        keep_silence=keep_silence,
        seek_step=seek_step,
    )
    first = trimmer.feed(audio_segment)
    last = trimmer.flush()
    return TrimResult(
        first.audio + last.audio,
        first.ranges + last.ranges,
    )


def trim_silence_stream(chunks, min_silence_len=1000, silence_thresh=-16,
                        keep_silence=100, seek_step=1, chunk_length=1000):
    """
    Yield completed :class:`TrimResult` objects as chunks are consumed.

    ``chunks`` may be an iterable of AudioSegments or one AudioSegment. In the
    latter case it is split into ``chunk_length`` millisecond slices.
    """
    if isinstance(chunks, AudioSegment):
        audio = chunks
        chunks = (
            audio[start:start + chunk_length]
            for start in range(0, len(audio), chunk_length)
        )

    trimmer = StreamingSilenceTrimmer(
        min_silence_len=min_silence_len,
        silence_thresh=silence_thresh,
        keep_silence=keep_silence,
        seek_step=seek_step,
    )
    for chunk in chunks:
        result = trimmer.feed(chunk)
        if result.audio or result.ranges:
            yield result
    result = trimmer.flush()
    if result.audio or result.ranges:
        yield result


SilenceTrimmer = StreamingSilenceTrimmer
StreamingSilenceSplitter = StreamingSilenceTrimmer
SegmentRange = FrameRange
SilenceTrimResult = TrimResult
trim_silence_with_ranges = trim_silence
streaming_trim_silence = trim_silence_stream


def detect_silence(audio_segment, min_silence_len=1000, silence_thresh=-16, seek_step=1):
    """
    Returns a list of all silent sections [start, end] in milliseconds of audio_segment.
    Inverse of detect_nonsilent()

    audio_segment - the segment to find silence in
    min_silence_len - the minimum length for any silent section
    silence_thresh - the upper bound for how quiet is silent in dFBS
    seek_step - step size for interating over the segment in ms
    """
    seg_len = len(audio_segment)

    # you can't have a silent portion of a sound that is longer than the sound
    if seg_len < min_silence_len:
        return []

    # convert silence threshold to a float value (so we can compare it to rms)
    silence_thresh = db_to_float(silence_thresh) * audio_segment.max_possible_amplitude

    # find silence and add start and end indicies to the to_cut list
    silence_starts = []

    # check successive (1 sec by default) chunk of sound for silence
    # try a chunk at every "seek step" (or every chunk for a seek step == 1)
    last_slice_start = seg_len - min_silence_len
    slice_starts = range(0, last_slice_start + 1, seek_step)

    # guarantee last_slice_start is included in the range
    # to make sure the last portion of the audio is searched
    if last_slice_start % seek_step:
        slice_starts = itertools.chain(slice_starts, [last_slice_start])

    for i in slice_starts:
        audio_slice = audio_segment[i:i + min_silence_len]
        if audio_slice.rms <= silence_thresh:
            silence_starts.append(i)

    # short circuit when there is no silence
    if not silence_starts:
        return []

    # combine the silence we detected into ranges (start ms - end ms)
    silent_ranges = []

    prev_i = silence_starts.pop(0)
    current_range_start = prev_i

    for silence_start_i in silence_starts:
        continuous = (silence_start_i == prev_i + seek_step)

        # sometimes two small blips are enough for one particular slice to be
        # non-silent, despite the silence all running together. Just combine
        # the two overlapping silent ranges.
        silence_has_gap = silence_start_i > (prev_i + min_silence_len)

        if not continuous and silence_has_gap:
            silent_ranges.append([current_range_start,
                                  prev_i + min_silence_len])
            current_range_start = silence_start_i
        prev_i = silence_start_i

    silent_ranges.append([current_range_start,
                          prev_i + min_silence_len])

    return silent_ranges


def detect_nonsilent(audio_segment, min_silence_len=1000, silence_thresh=-16, seek_step=1):
    """
    Returns a list of all nonsilent sections [start, end] in milliseconds of audio_segment.
    Inverse of detect_silent()

    audio_segment - the segment to find silence in
    min_silence_len - the minimum length for any silent section
    silence_thresh - the upper bound for how quiet is silent in dFBS
    seek_step - step size for interating over the segment in ms
    """
    silent_ranges = detect_silence(audio_segment, min_silence_len, silence_thresh, seek_step)
    len_seg = len(audio_segment)

    # if there is no silence, the whole thing is nonsilent
    if not silent_ranges:
        return [[0, len_seg]]

    # short circuit when the whole audio segment is silent
    if silent_ranges[0][0] == 0 and silent_ranges[0][1] == len_seg:
        return []

    prev_end_i = 0
    nonsilent_ranges = []
    for start_i, end_i in silent_ranges:
        nonsilent_ranges.append([prev_end_i, start_i])
        prev_end_i = end_i

    if end_i != len_seg:
        nonsilent_ranges.append([prev_end_i, len_seg])

    if nonsilent_ranges[0] == [0, 0]:
        nonsilent_ranges.pop(0)

    return nonsilent_ranges


def split_on_silence(audio_segment, min_silence_len=1000, silence_thresh=-16, keep_silence=100,
                     seek_step=1):
    """
    Returns list of audio segments from splitting audio_segment on silent sections

    audio_segment - original pydub.AudioSegment() object

    min_silence_len - (in ms) minimum length of a silence to be used for
        a split. default: 1000ms

    silence_thresh - (in dBFS) anything quieter than this will be
        considered silence. default: -16dBFS

    keep_silence - (in ms or True/False) leave some silence at the beginning
        and end of the chunks. Keeps the sound from sounding like it
        is abruptly cut off.
        When the length of the silence is less than the keep_silence duration
        it is split evenly between the preceding and following non-silent
        segments.
        If True is specified, all the silence is kept, if False none is kept.
        default: 100ms

    seek_step - step size for interating over the segment in ms
    """

    # from the itertools documentation
    def pairwise(iterable):
        "s -> (s0,s1), (s1,s2), (s2, s3), ..."
        a, b = itertools.tee(iterable)
        next(b, None)
        return zip(a, b)

    if isinstance(keep_silence, bool):
        keep_silence = len(audio_segment) if keep_silence else 0

    output_ranges = [
        [ start - keep_silence, end + keep_silence ]
        for (start,end)
            in detect_nonsilent(audio_segment, min_silence_len, silence_thresh, seek_step)
    ]

    for range_i, range_ii in pairwise(output_ranges):
        last_end = range_i[1]
        next_start = range_ii[0]
        if next_start < last_end:
            range_i[1] = (last_end+next_start)//2
            range_ii[0] = range_i[1]

    return [
        audio_segment[ max(start,0) : min(end,len(audio_segment)) ]
        for start,end in output_ranges
    ]


def detect_leading_silence(sound, silence_threshold=-50.0, chunk_size=10):
    """
    Returns the millisecond/index that the leading silence ends.

    audio_segment - the segment to find silence in
    silence_threshold - the upper bound for how quiet is silent in dFBS
    chunk_size - chunk size for interating over the segment in ms
    """
    trim_ms = 0 # ms
    assert chunk_size > 0 # to avoid infinite loop
    while sound[trim_ms:trim_ms+chunk_size].dBFS < silence_threshold and trim_ms < len(sound):
        trim_ms += chunk_size

    # if there is no end it should return the length of the segment
    return min(trim_ms, len(sound))


