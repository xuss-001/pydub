"""
Streaming volume envelopes for raw PCM audio.

``VolumeEnvelopeProcessor`` is designed for audio that is too large to decode
and process as a single buffer.  It accepts consecutive PCM chunks and applies
an envelope described by absolute sample-frame positions and gain values in
decibels.  The gain is interpolated linearly in dB between keypoints and the
nearest keypoint value is held before the first and after the last keypoint.

Example (stereo, 16-bit little-endian PCM, changing only the left channel)::

    from pydub.volume_envelope import VolumeEnvelopeProcessor

    processor = VolumeEnvelopeProcessor(
        [(0, -6.0), (48000, 0.0), (96000, -3.0)],
        channel=1,
        sample_width=2,
        channels=2,
        frame_rate=48000,
    )

    output_chunks = [
        processor.process(chunk)
        for chunk in pcm_chunks
    ]

Each call returns a ``bytes`` object containing exactly the same number of
audio frames as the input chunk.  Chunk boundaries have no effect on the
result: keypoint frames are addressed in the coordinate space of the complete
recording, never relative to a chunk.
"""

from __future__ import division

import bisect
import math
import numbers

from .utils import db_to_float


class InvalidVolumeEnvelope(ValueError):
    """Raised when an envelope or PCM stream is invalid."""


def _coerce_position(value, label="keypoint frame position"):
    if isinstance(value, bool) or not isinstance(value, numbers.Real):
        raise InvalidVolumeEnvelope("%s must be a number: %r" % (label, value))

    value = float(value)
    if not math.isfinite(value):
        raise InvalidVolumeEnvelope("%s must be finite: %r" % (label, value))
    if value < 0:
        raise InvalidVolumeEnvelope("%s must not be negative: %r" % (label, value))
    if not value.is_integer():
        raise InvalidVolumeEnvelope("%s must be a whole frame: %r" % (label, value))
    return int(value)


def _coerce_gain(value):
    if isinstance(value, bool) or not isinstance(value, numbers.Real):
        raise InvalidVolumeEnvelope("keypoint gain must be a number: %r" % (value,))

    value = float(value)
    if not math.isfinite(value):
        raise InvalidVolumeEnvelope("keypoint gain must be finite: %r" % (value,))
    return value


def _normalise_keypoints(keypoints):
    if keypoints is None:
        raise InvalidVolumeEnvelope("at least one (frame_position, gain_db) keypoint is required")

    if isinstance(keypoints, dict):
        keypoints = list(keypoints.items())

    try:
        keypoints = list(keypoints)
    except TypeError:
        raise InvalidVolumeEnvelope("keypoints must be an iterable of (frame_position, gain_db) pairs")

    if not keypoints:
        raise InvalidVolumeEnvelope("at least one (frame_position, gain_db) keypoint is required")

    normalised = []
    previous_position = None

    for keypoint in keypoints:
        if isinstance(keypoint, dict):
            position = keypoint.get("frame", keypoint.get("position"))
            gain = keypoint.get("gain_db", keypoint.get("gain", keypoint.get("db")))
        else:
            try:
                position, gain = keypoint
            except (TypeError, ValueError):
                raise InvalidVolumeEnvelope(
                    "each keypoint must be a (frame_position, gain_db) pair: %r" % (keypoint,)
                )

        position = _coerce_position(position)
        gain = _coerce_gain(gain)

        if previous_position is not None and position <= previous_position:
            raise InvalidVolumeEnvelope(
                "keypoint frame positions must be strictly increasing; got %r after %r"
                % (position, previous_position)
            )

        normalised.append((position, gain))
        previous_position = position

    return tuple(normalised)


def _normalise_channel_selector(selector, zero_based=False):
    if selector is None:
        return None
    if isinstance(selector, str):
        lowered = selector.strip().lower()
        if lowered in ("all", "*"):
            return None
        if lowered in ("left", "l", "mono"):
            selector = 1
        elif lowered in ("right", "r"):
            selector = 2
        else:
            raise InvalidVolumeEnvelope("unknown channel selector: %r" % (selector,))

    if isinstance(selector, numbers.Integral) and not isinstance(selector, bool):
        selector = (int(selector),)

    try:
        selector = tuple(selector)
    except TypeError:
        raise InvalidVolumeEnvelope("channel selector must be an integer or an iterable of integers")

    if not selector:
        raise InvalidVolumeEnvelope("channel selector must not be empty")

    channels = []
    for channel in selector:
        if isinstance(channel, bool) or not isinstance(channel, numbers.Integral):
            raise InvalidVolumeEnvelope("channel selector entries must be integers: %r" % (channel,))
        channel = int(channel)
        if zero_based:
            index = channel
        else:
            # pydub uses 1-based channels (1 is left/the first channel).  Zero
            # is accepted as an explicit alias for the first channel.
            index = 0 if channel == 0 else channel - 1
        if index < 0:
            raise InvalidVolumeEnvelope("channel selector must not be negative: %r" % (channel,))
        channels.append(index)

    return tuple(sorted(set(channels)))


def _normalise_positive_int(value, label):
    if isinstance(value, bool) or not isinstance(value, numbers.Real):
        raise InvalidVolumeEnvelope("%s must be an integer: %r" % (label, value))
    if not math.isfinite(float(value)) or not float(value).is_integer():
        raise InvalidVolumeEnvelope("%s must be a whole number: %r" % (label, value))
    value = int(value)
    if value <= 0:
        raise InvalidVolumeEnvelope("%s must be positive: %r" % (label, value))
    return value


def _normalise_frame_rate(value):
    if value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, numbers.Real):
        raise InvalidVolumeEnvelope("frame_rate must be a number: %r" % (value,))
    value = float(value)
    if not math.isfinite(value) or value <= 0:
        raise InvalidVolumeEnvelope("frame_rate must be finite and positive: %r" % (value,))
    return value


class VolumeEnvelopeProcessor(object):
    """
    Apply a dB volume envelope to a stream of raw PCM chunks.

    ``keypoints`` is an iterable of ``(absolute_frame_position, gain_db)``
    pairs.  Positions address frames in the complete recording, not frames
    within a chunk.  Positions must be strictly increasing and gains must be
    finite.  Between keypoints the gain changes linearly in decibels.  Before
    the first keypoint the first gain is used, and after the last keypoint the
    last gain is used.

    The PCM format is fixed by the first chunk.  Pass ``sample_width`` (bytes
    per sample), ``channels`` (number of interleaved channels), and optionally
    ``frame_rate`` to either the constructor or the first ``process`` call.
    Every later chunk must use the same format.

    ``channel`` selects the channel(s) to modify.  Channels use pydub's
    1-based convention (1 is the first/left channel); ``0`` is also accepted
    as an alias for the first channel.  ``None`` or ``"all"`` modifies every
    channel.  Channels that are not selected are copied byte-for-byte.

    The processor keeps only the envelope, the current absolute frame
    position, a keypoint cursor, and the detected format.  It never retains
    previously processed audio.
    """

    def __init__(self, keypoints=None, channel=None, sample_width=None,
                 channels=None, frame_rate=None, start_frame=0,
                 channel_index=None, format=None, **kwargs):
        if keypoints is None:
            for alias in ("keyframes", "envelope", "points"):
                if alias in kwargs:
                    keypoints = kwargs.pop(alias)
                    break

        for alias in ("selected_channels", "target_channels"):
            if alias in kwargs:
                if channel is not None or channel_index is not None:
                    raise TypeError("%s cannot be combined with channel/channel_index" % alias)
                channel = kwargs.pop(alias)

        if "channel_indices" in kwargs:
            if channel is not None or channel_index is not None:
                raise TypeError(
                    "channel_indices cannot be combined with channel/channel_index"
                )
            channel_index = kwargs.pop("channel_indices")

        if "sample_rate" in kwargs:
            if frame_rate is not None:
                raise TypeError("frame_rate and sample_rate cannot be used together")
            frame_rate = kwargs.pop("sample_rate")

        if kwargs:
            unexpected = next(iter(kwargs))
            raise TypeError("unexpected keyword argument %r" % unexpected)

        format_sample_width, format_channels, format_frame_rate = self._parse_format(format)
        if sample_width is None:
            sample_width = format_sample_width
        elif format_sample_width is not None and sample_width != format_sample_width:
            raise InvalidVolumeEnvelope("conflicting sample_width values")
        if channels is None:
            channels = format_channels
        elif format_channels is not None and channels != format_channels:
            raise InvalidVolumeEnvelope("conflicting channel counts")
        if frame_rate is None:
            frame_rate = format_frame_rate
        elif format_frame_rate is not None and frame_rate != format_frame_rate:
            raise InvalidVolumeEnvelope("conflicting frame rates")

        self._keypoints = _normalise_keypoints(keypoints)
        self._keypoint_positions = tuple(position for position, _ in self._keypoints)

        if channel is not None and channel_index is not None:
            raise TypeError("channel and channel_index cannot be used together")
        if channel_index is not None:
            self._selected_channels = _normalise_channel_selector(channel_index, zero_based=True)
        else:
            self._selected_channels = _normalise_channel_selector(channel)

        self._sample_width = (
            None if sample_width is None else _normalise_positive_int(sample_width, "sample_width")
        )
        self._channels = (
            None if channels is None else _normalise_positive_int(channels, "channels")
        )
        self._frame_rate = _normalise_frame_rate(frame_rate)
        self._frame_width = (
            None if self._sample_width is None or self._channels is None
            else self._sample_width * self._channels
        )
        self._format_locked = self._frame_width is not None

        start_frame = _coerce_position(start_frame, "start_frame")
        self._start_frame = start_frame
        self._position = start_frame
        self._segment_index = max(
            0, bisect.bisect_right(self._keypoint_positions, start_frame) - 1
        )

        self._validate_selected_channels()

    @property
    def keypoints(self):
        return self._keypoints

    @property
    def position(self):
        """Absolute frame position of the next input frame."""
        return self._position

    @property
    def frame_position(self):
        return self._position

    @property
    def frames_processed(self):
        return self._position - self._start_frame

    @property
    def sample_width(self):
        return self._sample_width

    @property
    def channels(self):
        return self._channels

    @property
    def frame_rate(self):
        return self._frame_rate

    @property
    def frame_width(self):
        return self._frame_width

    @property
    def channel(self):
        if self._selected_channels is None:
            return None
        if len(self._selected_channels) == 1:
            return self._selected_channels[0] + 1
        return tuple(channel + 1 for channel in self._selected_channels)

    def reset(self):
        """Reset the frame cursor without discarding the envelope or format."""
        self._position = self._start_frame
        self._segment_index = max(
            0, bisect.bisect_right(self._keypoint_positions, self._start_frame) - 1
        )

    def _validate_selected_channels(self):
        if self._selected_channels is None or self._channels is None:
            return
        invalid = [channel + 1 for channel in self._selected_channels
                   if channel >= self._channels]
        if invalid:
            raise InvalidVolumeEnvelope(
                "channel(s) %s do not exist in %d-channel audio"
                % (invalid, self._channels)
            )

    def _resolve_format(self, sample_width, channels, frame_rate):
        if sample_width is not None:
            sample_width = _normalise_positive_int(sample_width, "sample_width")
        if channels is not None:
            channels = _normalise_positive_int(channels, "channels")
        if frame_rate is not None:
            frame_rate = _normalise_frame_rate(frame_rate)

        if not self._format_locked:
            if sample_width is not None:
                self._sample_width = sample_width
            if channels is not None:
                self._channels = channels
            if frame_rate is not None:
                self._frame_rate = frame_rate

            if self._sample_width is None or self._channels is None:
                raise InvalidVolumeEnvelope(
                    "the first chunk must determine the audio format: "
                    "sample_width and channels are required"
                )

            self._frame_width = self._sample_width * self._channels
            self._format_locked = True
            self._validate_selected_channels()
            return

        expected = (
            ("sample_width", sample_width, self._sample_width),
            ("channels", channels, self._channels),
            ("frame_rate", frame_rate, self._frame_rate),
        )
        for name, actual, current in expected:
            if actual is None:
                continue
            if current is None:
                if name == "frame_rate":
                    self._frame_rate = actual
                continue
            if actual != current:
                raise InvalidVolumeEnvelope(
                    "audio format changed: %s was %r, got %r"
                    % (name, current, actual)
                )

    def _gain_db_at(self, frame):
        keypoints = self._keypoints
        first_position, first_gain = keypoints[0]
        if frame <= first_position:
            return first_gain

        last_position, last_gain = keypoints[-1]
        if frame >= last_position:
            return last_gain

        index = self._segment_index
        last_segment = len(keypoints) - 2
        if index > last_segment:
            index = last_segment

        while index < last_segment and keypoints[index + 1][0] <= frame:
            index += 1

        self._segment_index = index
        start_position, start_gain = keypoints[index]
        end_position, end_gain = keypoints[index + 1]
        fraction = (frame - start_position) / (end_position - start_position)
        return start_gain + (end_gain - start_gain) * fraction

    @staticmethod
    def _gain_factor(gain_db):
        if gain_db == 0:
            return 1.0
        try:
            return db_to_float(gain_db)
        except OverflowError:
            return float("inf")

    @staticmethod
    def _scale_sample(sample, factor, sample_width):
        bits = sample_width * 8
        minimum = -(1 << (bits - 1))
        maximum = (1 << (bits - 1)) - 1

        if math.isinf(factor):
            if sample > 0:
                return maximum
            if sample < 0:
                return minimum
            return 0

        try:
            product = sample * factor
        except OverflowError:
            return maximum if sample > 0 else minimum if sample < 0 else 0

        if not math.isfinite(product):
            return maximum if product > 0 else minimum

        # audioop.mul() uses floor(sample * factor); matching it keeps a
        # constant envelope byte-identical to AudioSegment.apply_gain().
        scaled = math.floor(product)
        if scaled < minimum:
            return minimum
        if scaled > maximum:
            return maximum
        return scaled

    def _process_bytes(self, data, sample_width, channels, frame_rate):
        self._resolve_format(sample_width, channels, frame_rate)

        if len(data) % self._frame_width:
            raise InvalidVolumeEnvelope(
                "chunk length %d is not a whole number of %d-byte frames"
                % (len(data), self._frame_width)
            )

        frames = len(data) // self._frame_width
        if frames == 0:
            return b""

        source = memoryview(data).cast("B")
        output = bytearray(source)
        selected_channels = self._selected_channels
        if selected_channels is None:
            selected_channels = range(self._channels)

        sample_width = self._sample_width
        frame_width = self._frame_width
        position = self._position

        for frame_index in range(frames):
            gain_factor = self._gain_factor(self._gain_db_at(position + frame_index))
            frame_offset = frame_index * frame_width
            for channel in selected_channels:
                offset = frame_offset + channel * sample_width
                sample = int.from_bytes(
                    source[offset:offset + sample_width], "little", signed=True
                )
                sample = self._scale_sample(sample, gain_factor, sample_width)
                output[offset:offset + sample_width] = sample.to_bytes(
                    sample_width, "little", signed=True
                )

        self._position += frames
        return bytes(output)

    @staticmethod
    def _parse_format(format_value):
        if format_value is None:
            return None, None, None
        if isinstance(format_value, dict):
            return (
                format_value.get("sample_width"),
                format_value.get("channels"),
                format_value.get("frame_rate", format_value.get("sample_rate")),
            )
        try:
            values = tuple(format_value)
        except TypeError:
            raise InvalidVolumeEnvelope(
                "format must be a (sample_width, channels[, frame_rate]) sequence"
            )
        if len(values) == 2:
            return values[0], values[1], None
        if len(values) == 3:
            return values
        raise InvalidVolumeEnvelope(
            "format must be a (sample_width, channels[, frame_rate]) sequence"
        )

    def process(self, chunk, sample_width=None, channels=None, frame_rate=None,
                format=None):
        """
        Process one PCM chunk and return a PCM buffer with the same frame count.

        ``chunk`` may be ``bytes``, ``bytearray``, ``memoryview``, or an
        ``array.array``.  An ``AudioSegment`` is also accepted; in that case an
        ``AudioSegment`` with the same metadata is returned.
        """
        format_sample_width, format_channels, format_frame_rate = self._parse_format(format)
        if sample_width is None:
            sample_width = format_sample_width
        elif format_sample_width is not None and sample_width != format_sample_width:
            raise InvalidVolumeEnvelope("conflicting sample_width values")
        if channels is None:
            channels = format_channels
        elif format_channels is not None and channels != format_channels:
            raise InvalidVolumeEnvelope("conflicting channel counts")
        if frame_rate is None:
            frame_rate = format_frame_rate
        elif format_frame_rate is not None and frame_rate != format_frame_rate:
            raise InvalidVolumeEnvelope("conflicting frame rates")

        is_audio_segment = (
            hasattr(chunk, "raw_data")
            and hasattr(chunk, "sample_width")
            and hasattr(chunk, "channels")
            and hasattr(chunk, "_spawn")
        )

        if is_audio_segment:
            segment = chunk
            chunk = segment.raw_data
            if sample_width is None:
                sample_width = segment.sample_width
            if channels is None:
                channels = segment.channels
            if frame_rate is None:
                frame_rate = segment.frame_rate

        if not isinstance(chunk, (bytes, bytearray, memoryview)):
            try:
                chunk = memoryview(chunk).cast("B")
            except TypeError:
                raise InvalidVolumeEnvelope(
                    "chunk must be a bytes-like object or an AudioSegment"
                )

        result = self._process_bytes(chunk, sample_width, channels, frame_rate)
        if is_audio_segment:
            return segment._spawn(result)
        return result

    # Friendly aliases for common streaming APIs.
    process_chunk = process
    feed = process
    write = process
    apply = process

    def __call__(self, chunk, sample_width=None, channels=None, frame_rate=None,
                 format=None):
        return self.process(chunk, sample_width, channels, frame_rate, format)


# Descriptive aliases; the implementation is the same streaming processor.
VolumeEnvelope = VolumeEnvelopeProcessor
StreamingVolumeEnvelope = VolumeEnvelopeProcessor
StreamingVolumeEnvelopeProcessor = VolumeEnvelopeProcessor
PCMVolumeEnvelopeProcessor = VolumeEnvelopeProcessor
PCMVolumeEnvelope = VolumeEnvelopeProcessor


def apply_volume_envelope(segment, keypoints, channel=None):
    """
    Apply an envelope to an ``AudioSegment`` (or an object with the same
    ``raw_data``/format attributes) and return the same kind of object.
    """
    processor = VolumeEnvelopeProcessor(
        keypoints,
        channel=channel,
        sample_width=segment.sample_width,
        channels=segment.channels,
        frame_rate=segment.frame_rate,
    )
    return processor.process(segment)
