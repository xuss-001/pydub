"""
Streaming look-ahead peak limiter.

``LookaheadLimiter`` consumes signed PCM bytes.  Callers declare the PCM
format once, feed complete frames, and finally call :meth:`finish` to drain
the look-ahead delay.  The limiter always applies one gain value to every
channel in a frame, so the relative level of the channels is preserved.
"""

from __future__ import division

import array
import math
from collections import deque

from .audio_segment import AudioSegment
from .exceptions import InvalidAudioFormat, InvalidLimiterSettings
from .utils import audioop, db_to_float, ratio_to_db, register_pydub_effect


_EPSILON = 1e-12


def _is_finite(value):
    return not (math.isinf(value) or math.isnan(value))


def _as_number(name, value):
    if isinstance(value, bool):
        raise InvalidLimiterSettings("{0} must be a finite number".format(name))
    try:
        value = float(value)
    except (TypeError, ValueError):
        raise InvalidLimiterSettings(
            "{0} must be a finite number, got {1!r}".format(name, value))
    if not _is_finite(value):
        raise InvalidLimiterSettings("{0} must be finite".format(name))
    return value


def _as_positive_int(name, value):
    if isinstance(value, bool):
        raise InvalidLimiterSettings("{0} must be a positive integer".format(name))
    try:
        converted = int(value)
    except (TypeError, ValueError):
        raise InvalidLimiterSettings(
            "{0} must be a positive integer, got {1!r}".format(name, value))
    if converted != value or converted <= 0:
        raise InvalidLimiterSettings(
            "{0} must be a positive integer, got {1!r}".format(name, value))
    return converted


def _lookup(mapping, *names):
    for name in names:
        if name in mapping and mapping[name] is not None:
            return mapping.pop(name)
    return None


class LookaheadLimiter(object):
    """
    A streaming, channel-linked peak limiter with look-ahead.

    Parameters
    ----------
    sample_width : int
        PCM sample width in bytes (1, 2, 3, or 4).
    frame_rate : int
        Sample rate in Hz.  ``sample_rate`` and ``rate`` are accepted aliases.
    channels : int
        Number of interleaved channels.
    ceiling : float
        Output peak ceiling.  Negative values are interpreted as dBFS
        (``-1.0`` means -1 dBFS).  Values in ``(0, 1]`` are normalized
        amplitude, and values greater than one are absolute sample units.
    ceiling_db : float, optional
        Explicit dBFS ceiling.  It takes precedence over ``ceiling``.
    max_peak, peak_limit : float, optional
        Explicit normalized (``<= 1``) or absolute sample-unit ceiling.
    lookahead_ms : float
        Look-ahead time in milliseconds.  ``lookahead``,
        ``lookahead_time_ms`` and ``lookahead_duration_ms`` are aliases.
    release_ms : float
        Recovery time in milliseconds.  ``release``, ``recovery_ms`` and
        ``recovery_time_ms`` are aliases.
    segment : AudioSegment, optional
        Copy the PCM format from this segment.
    metadata, format : dict, optional
        A mapping containing ``sample_width``, ``frame_rate`` and
        ``channels``.

    The returned PCM has the same frames and peak ceiling as a single,
    whole-buffer run, independent of how complete frames are split between
    calls to :meth:`process`.
    """

    def __init__(self, sample_width=None, frame_rate=None, channels=None,
                 ceiling=-1.0, lookahead_ms=5.0, release_ms=50.0,
                 segment=None, metadata=None, format=None,
                 sample_rate=None, lookahead=None, release=None,
                 recovery_ms=None, recovery_time_ms=None,
                 ceiling_db=None, max_peak=None, peak_limit=None,
                 **kwargs):
        if isinstance(sample_width, dict):
            metadata = sample_width
            sample_width = None
        elif sample_width is not None and hasattr(sample_width, "raw_data"):
            segment = sample_width
            sample_width = None

        rate_alias = _lookup(kwargs, "rate", "sample_rate_hz")
        if sample_rate is None:
            sample_rate = rate_alias
        if sample_rate is not None:
            frame_rate = sample_rate

        channel_alias = _lookup(
            kwargs, "n_channels", "num_channels", "channel_count")
        if channel_alias is not None and channels is None:
            channels = channel_alias
        width_alias = _lookup(kwargs, "bytes_per_sample", "sample_size")
        if width_alias is not None and sample_width is None:
            sample_width = width_alias

        lookahead_alias = _lookup(
            kwargs, "lookahead_time_ms", "lookahead_duration_ms",
            "lookahead_time", "lookahead_duration", "attack_ms", "attack")
        if lookahead_alias is not None:
            lookahead_ms = lookahead_alias
        if lookahead is not None:
            lookahead_ms = lookahead

        release_alias = _lookup(
            kwargs, "release_time_ms", "release_time", "recovery_time")
        if release_alias is not None:
            release_ms = release_alias
        if recovery_time_ms is not None:
            release_ms = recovery_time_ms
        if recovery_ms is not None:
            release_ms = recovery_ms
        if release is not None:
            release_ms = release

        ceiling_alias = _lookup(
            kwargs, "ceiling_dbfs", "ceiling_dBFS", "output_ceiling_db",
            "peak_limit_dbfs", "threshold_db", "max_peak_db")
        if ceiling_alias is not None:
            ceiling_db = ceiling_alias
        absolute_alias = _lookup(
            kwargs, "max_output", "output_peak", "peak", "limit",
            "peak_ceiling", "output_ceiling")
        if absolute_alias is not None and max_peak is None:
            max_peak = absolute_alias

        if kwargs:
            unknown = ", ".join(sorted(kwargs))
            raise TypeError("unexpected limiter argument(s): {0}".format(unknown))

        declared_format = {
            "sample_width": sample_width,
            "frame_rate": frame_rate,
            "channels": channels,
        }
        declared_format = dict(
            (key, value) for key, value in declared_format.items()
            if value is not None)

        if metadata is not None:
            metadata = self._normalize_metadata(metadata)

        if segment is not None:
            segment_format = self._format_from_segment(segment)
            for key, value in declared_format.items():
                if value != segment_format[key]:
                    raise InvalidAudioFormat(
                        "declared {0}={1} does not match segment {2}={3}".format(
                            key, value, key, segment_format[key]))
            if metadata:
                for key, value in metadata.items():
                    if (key in segment_format and
                            value != segment_format[key]):
                        raise InvalidAudioFormat(
                            "metadata {0}={1} does not match segment {0}={2}".format(
                                key, value, segment_format[key]))
            metadata = dict(metadata or {}, **segment_format)

        if format is not None:
            normalized_format = self._normalize_metadata(format)
            if metadata:
                for key, value in normalized_format.items():
                    if key in metadata and value != metadata[key]:
                        raise InvalidAudioFormat(
                            "format {0}={1} conflicts with metadata {0}={2}".format(
                                key, value, metadata[key]))
            metadata = dict(metadata or {}, **normalized_format)

        if metadata:
            metadata = self._normalize_metadata(metadata)
            for key in ("sample_width", "frame_rate", "channels"):
                explicit = declared_format.get(key)
                if (explicit is not None and key in metadata and
                        explicit != metadata[key]):
                    raise InvalidAudioFormat(
                        "declared {0}={1} does not match metadata {0}={2}".format(
                            key, explicit, metadata[key]))
            if sample_width is None:
                sample_width = metadata.get("sample_width")
            if frame_rate is None:
                frame_rate = metadata.get("frame_rate")
            if channels is None:
                channels = metadata.get("channels")

        if sample_width is None or frame_rate is None or channels is None:
            raise InvalidAudioFormat(
                "sample_width, frame_rate and channels must all be specified")

        self.sample_width = _as_positive_int("sample_width", sample_width)
        if self.sample_width not in (1, 2, 3, 4):
            raise InvalidLimiterSettings(
                "sample_width must be 1, 2, 3, or 4 bytes")
        self.frame_rate = _as_positive_int("frame_rate", frame_rate)
        self.channels = _as_positive_int("channels", channels)
        self.frame_width = self.sample_width * self.channels

        self.lookahead_ms = _as_number("lookahead_ms", lookahead_ms)
        if self.lookahead_ms <= 0:
            raise InvalidLimiterSettings("lookahead_ms must be greater than zero")

        self.release_ms = _as_number("release_ms", release_ms)
        if self.release_ms <= 0:
            raise InvalidLimiterSettings("release_ms must be greater than zero")

        self.lookahead_frames = max(
            1, int(round(self.frame_rate * self.lookahead_ms / 1000.0)))
        self.release_frames = max(
            1, int(round(self.frame_rate * self.release_ms / 1000.0)))

        self.max_possible_amplitude = 2 ** (self.sample_width * 8 - 1)
        self.ceiling_amplitude = self._resolve_ceiling(
            ceiling, ceiling_db=ceiling_db,
            max_peak=max_peak, peak_limit=peak_limit)
        self.ceiling = self.ceiling_amplitude / self.max_possible_amplitude
        self.ceiling_db = ratio_to_db(self.ceiling)

        if self.sample_width == 1:
            self._array_type = "b"
        elif self.sample_width == 2:
            self._array_type = "h"
        elif self.sample_width == 4:
            self._array_type = "i"
        else:
            self._array_type = None

        self._delay = deque()
        self._peak_queue = deque()
        self._next_input_frame = 0
        self._next_output_frame = 0
        self._gain = 1.0
        self._attack_remaining = 0
        self._attack_target = 1.0
        self._releasing = False
        self._release_remaining = 0
        self._release_target = 1.0
        self._finished = False
        self.frames_in = 0
        self.frames_out = 0
        self.frames_discarded = 0

    @staticmethod
    def _normalize_metadata(metadata):
        if not isinstance(metadata, dict):
            try:
                metadata = {
                    "sample_width": metadata.sample_width,
                    "frame_rate": metadata.frame_rate,
                    "channels": metadata.channels,
                }
            except AttributeError:
                raise InvalidAudioFormat(
                    "metadata must provide sample_width, frame_rate and channels")

        aliases = {
            "sample_rate": "frame_rate",
            "rate": "frame_rate",
            "channel_count": "channels",
            "sample_size": "sample_width",
        }
        normalized = {}
        for key, value in metadata.items():
            normalized[aliases.get(key, key)] = value
        return normalized

    @staticmethod
    def _format_from_segment(segment):
        try:
            return {
                "sample_width": segment.sample_width,
                "frame_rate": segment.frame_rate,
                "channels": segment.channels,
            }
        except AttributeError:
            raise InvalidAudioFormat(
                "segment must provide sample_width, frame_rate and channels")

    def _resolve_ceiling(self, ceiling, ceiling_db=None,
                         max_peak=None, peak_limit=None):
        if ceiling_db is not None:
            value = _as_number("ceiling_db", ceiling_db)
            if value > 0:
                raise InvalidLimiterSettings("ceiling_db must be <= 0 dBFS")
            amplitude = self.max_possible_amplitude * db_to_float(value)
        elif max_peak is not None or peak_limit is not None:
            value = max_peak if max_peak is not None else peak_limit
            value = _as_number("max_peak", value)
            if value <= 0:
                raise InvalidLimiterSettings("max_peak must be greater than zero")
            amplitude = (self.max_possible_amplitude * value
                         if value <= 1 else value)
        else:
            value = _as_number("ceiling", ceiling)
            if value < 0:
                amplitude = self.max_possible_amplitude * db_to_float(value)
            elif value <= 1:
                amplitude = self.max_possible_amplitude * value
            else:
                amplitude = value

        amplitude = int(math.floor(amplitude))
        if amplitude < 1:
            raise InvalidLimiterSettings(
                "ceiling must be greater than zero after integer rounding")
        if amplitude > self.max_possible_amplitude:
            raise InvalidLimiterSettings(
                "ceiling must not exceed {0}".format(self.max_possible_amplitude))
        return amplitude

    @classmethod
    def for_segment(cls, segment, **kwargs):
        kwargs["segment"] = segment
        return cls(**kwargs)

    @property
    def input_format(self):
        return {
            "sample_width": self.sample_width,
            "frame_rate": self.frame_rate,
            "channels": self.channels,
        }

    @property
    def latency_frames(self):
        return self.lookahead_frames

    @property
    def latency_ms(self):
        return 1000.0 * self.lookahead_frames / self.frame_rate

    @property
    def pending_frames(self):
        return len(self._delay)

    @property
    def buffered_frames(self):
        return len(self._delay)

    @property
    def max_pending_frames(self):
        return self.lookahead_frames + 1

    @property
    def finished(self):
        return self._finished

    @property
    def closed(self):
        return self._finished

    @property
    def current_gain(self):
        return self._gain

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self.finish()
        return False

    def check_format(self, metadata=None, **format_kwargs):
        """
        Validate a chunk's declared format without processing audio.

        Only fields supplied by the caller are checked.  Raw PCM does not
        carry format metadata, so callers that may change format should pass
        the format explicitly (or feed an AudioSegment).
        """
        supplied = self._collect_format(metadata, format_kwargs)
        expected = self.input_format
        for key in ("sample_width", "frame_rate", "channels"):
            if key in supplied and supplied[key] != expected[key]:
                raise InvalidAudioFormat(
                    "input format changed: expected {0}={1}, got {2}".format(
                        key, expected[key], supplied[key]))
        return True

    def _collect_format(self, metadata, format_kwargs):
        supplied = {}
        if metadata is not None:
            supplied.update(self._normalize_metadata(metadata))
        supplied.update(self._normalize_metadata(format_kwargs))
        supplied = dict(
            (key, value) for key, value in supplied.items()
            if value is not None)
        return supplied

    def process(self, chunk, metadata=None, format=None, **format_kwargs):
        """
        Limit one PCM chunk and return the frames that are ready.

        ``chunk`` may be bytes, bytearray, memoryview, array.array, or an
        AudioSegment.  AudioSegments are returned as AudioSegments; all other
        inputs return bytes.  Chunks must contain complete frames and must
        use the format declared when the limiter was created.
        """
        if self._finished:
            raise RuntimeError("cannot process audio after finish()")

        if format is not None:
            format = self._normalize_metadata(format)
            if metadata is None:
                metadata = format
            else:
                metadata = dict(metadata)
                for key, value in format.items():
                    if key in metadata and metadata[key] != value:
                        raise InvalidAudioFormat(
                            "format {0}={1} conflicts with metadata {0}={2}".format(
                                key, value, metadata[key]))
                    metadata[key] = value

        metadata = dict(metadata) if isinstance(metadata, dict) else metadata
        if metadata is not None:
            metadata = self._normalize_metadata(metadata)
        supplied = self._collect_format(metadata, format_kwargs)
        return_segment = isinstance(chunk, AudioSegment)
        if return_segment:
            supplied.update(self._format_from_segment(chunk))

        self.check_format(supplied)

        if isinstance(chunk, AudioSegment):
            raw = chunk.raw_data
        elif isinstance(chunk, array.array):
            if chunk.itemsize != self.sample_width:
                raise InvalidAudioFormat(
                    "array itemsize {0} does not match sample_width {1}".format(
                        chunk.itemsize, self.sample_width))
            try:
                raw = chunk.tobytes()
            except AttributeError:  # pragma: no cover - Python 2 compatibility
                raw = chunk.tostring()
        elif isinstance(chunk, memoryview):
            raw = chunk.tobytes()
        elif isinstance(chunk, (bytes, bytearray)):
            raw = bytes(chunk)
        else:
            raise TypeError(
                "chunk must be bytes-like or an AudioSegment, got {0}".format(
                    type(chunk).__name__))

        if len(raw) % self.frame_width:
            raise InvalidAudioFormat(
                "chunk length {0} is not a multiple of frame_width {1}; "
                "incomplete frames cannot be buffered indefinitely".format(
                    len(raw), self.frame_width))

        output = []
        for offset in range(0, len(raw), self.frame_width):
            frame = raw[offset:offset + self.frame_width]
            limited = self._push_frame(frame)
            if limited is not None:
                output.append(limited)

        raw_output = b"".join(output)
        if return_segment:
            return chunk._spawn(data=raw_output)
        return raw_output

    def feed(self, chunk, **kwargs):
        return self.process(chunk, **kwargs)

    def push(self, chunk, **kwargs):
        return self.process(chunk, **kwargs)

    def write(self, chunk, **kwargs):
        return self.process(chunk, **kwargs)

    def process_chunk(self, chunk, **kwargs):
        return self.process(chunk, **kwargs)

    def feed_chunk(self, chunk, **kwargs):
        return self.process(chunk, **kwargs)

    def finish(self):
        """
        Drain the look-ahead delay and mark the stream finished.

        The first call returns every frame that has not been emitted yet.
        Later calls return an empty byte string and never repeat audio.
        """
        if self._finished:
            return b""

        output = []
        while self._delay:
            output.append(self._emit_one())
        self._finished = True
        return b"".join(output)

    def flush(self):
        return self.finish()

    def drain(self):
        return self.finish()

    def close(self):
        return self.finish()

    def finalize(self):
        return self.finish()

    def end_stream(self):
        return self.finish()

    def flush_remaining(self):
        return self.finish()

    def reset(self):
        """
        Reset stream state while keeping the current format and settings.
        """
        self._delay.clear()
        self._peak_queue.clear()
        self._next_input_frame = 0
        self._next_output_frame = 0
        self._gain = 1.0
        self._attack_remaining = 0
        self._attack_target = 1.0
        self._releasing = False
        self._release_remaining = 0
        self._release_target = 1.0
        self._finished = False
        self.frames_in = 0
        self.frames_out = 0
        self.frames_discarded = 0

    def _push_frame(self, frame):
        index = self._next_input_frame
        peak = audioop.max(frame, self.sample_width)
        self._delay.append((index, frame, peak))
        self.frames_in += 1

        while self._peak_queue and self._peak_queue[-1][1] < peak:
            self._peak_queue.pop()
        self._peak_queue.append((index, peak))
        self._next_input_frame += 1

        if len(self._delay) > self.lookahead_frames:
            return self._emit_one()
        return None

    def _emit_one(self):
        index, frame, frame_peak = self._delay.popleft()
        window_peak = self._peak_queue[0][1]

        target_gain = 1.0
        if window_peak > self.ceiling_amplitude:
            target_gain = self.ceiling_amplitude / float(window_peak)
        peak_index = self._peak_queue[0][0]
        frames_until_peak = peak_index - index + 1
        self._advance_gain(target_gain, frames_until_peak)

        # The envelope normally gets below this value before a peak arrives.
        # This guard makes the ceiling absolute for streams that begin with a
        # peak (where no earlier samples exist for an attack ramp).
        if frame_peak > self.ceiling_amplitude:
            required_gain = self.ceiling_amplitude / float(frame_peak)
            if required_gain < self._gain:
                self._gain = required_gain

        if self._peak_queue and self._peak_queue[0][0] == index:
            self._peak_queue.popleft()

        output = self._apply_gain(frame, self._gain)
        self._next_output_frame += 1
        self.frames_out += 1
        return output

    def _advance_gain(self, target_gain, frames_until_peak):
        if target_gain < self._gain - _EPSILON:
            self._releasing = False
            self._release_remaining = 0
            if (target_gain < self._attack_target - _EPSILON or
                    self._attack_remaining <= 0):
                self._attack_target = target_gain
                self._attack_remaining = max(
                    1, min(self.lookahead_frames, frames_until_peak))

            if self._attack_remaining > 0:
                remaining = float(self._attack_remaining)
                self._gain = (
                    target_gain +
                    (self._gain - target_gain) * (remaining - 1.0) / remaining)
                self._attack_remaining -= 1
            else:  # pragma: no cover - guarded above
                self._gain = target_gain
        elif target_gain > self._gain + _EPSILON:
            if (not self._releasing or
                    target_gain > self._release_target + _EPSILON):
                self._releasing = True
                self._release_target = target_gain
                self._release_remaining = self.release_frames

            remaining = float(self._release_remaining)
            if remaining <= 0:
                self._gain = target_gain
            else:
                self._gain = (
                    target_gain +
                    (self._gain - target_gain) *
                    (remaining - 1.0) / remaining)
                self._release_remaining -= 1
            self._attack_remaining = 0
            self._attack_target = self._gain
        else:
            self._gain = target_gain
            self._attack_remaining = 0
            self._attack_target = self._gain
            self._releasing = False
            self._release_remaining = 0

    def _apply_gain(self, frame, gain):
        if gain >= 1.0 - _EPSILON:
            return frame

        gain = max(0.0, min(1.0, gain))
        if self._array_type is None:
            return self._apply_gain_24bit(frame, gain)

        samples = array.array(self._array_type, frame)
        for i, sample in enumerate(samples):
            samples[i] = int(sample * gain)
        try:
            return samples.tobytes()
        except AttributeError:  # pragma: no cover - Python 2 compatibility
            return samples.tostring()

    def _apply_gain_24bit(self, frame, gain):
        output = bytearray(len(frame))
        for offset in range(0, len(frame), 3):
            sample = int.from_bytes(
                frame[offset:offset + 3], byteorder="little", signed=True)
            scaled = int(sample * gain)
            scaled = max(-0x800000, min(0x7fffff, scaled))
            output[offset:offset + 3] = scaled.to_bytes(
                3, byteorder="little", signed=True)
        return bytes(output)


StreamingLimiter = LookaheadLimiter
LookAheadLimiter = LookaheadLimiter
StreamingLookaheadLimiter = LookaheadLimiter
StreamingLookAheadLimiter = LookaheadLimiter
PCMStreamLimiter = LookaheadLimiter
PCMChunkLimiter = LookaheadLimiter
Limiter = LookaheadLimiter


def lookahead_limiter(seg, ceiling=-1.0, lookahead_ms=5.0, release_ms=50.0,
                      **kwargs):
    """
    Apply a look-ahead limiter to a whole AudioSegment.

    This is registered as ``AudioSegment.lookahead_limiter()`` and also as
    ``AudioSegment.limiter()`` / ``AudioSegment.limit()``.  The result has
    exactly the same number of frames as ``seg``; the look-ahead is drained
    internally and no padding is added.
    """
    limiter = LookaheadLimiter(
        segment=seg,
        ceiling=ceiling,
        lookahead_ms=lookahead_ms,
        release_ms=release_ms,
        **kwargs)
    output = limiter.process(seg.raw_data) + limiter.finish()
    return seg._spawn(data=output)


lookahead_limiter = register_pydub_effect(lookahead_limiter)
AudioSegment.limiter = lookahead_limiter
AudioSegment.limit = lookahead_limiter
AudioSegment.apply_lookahead_limiter = lookahead_limiter
AudioSegment.look_ahead_limiter = lookahead_limiter
AudioSegment.limit_peaks = lookahead_limiter
streaming_limiter = lookahead_limiter
streaming_lookahead_limiter = lookahead_limiter
look_ahead_limiter = lookahead_limiter
apply_lookahead_limiter = lookahead_limiter
limit_peaks = lookahead_limiter
limiter = lookahead_limiter

