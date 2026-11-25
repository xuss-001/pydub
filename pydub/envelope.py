"""
Streaming volume envelope processing for PCM audio.

The :class:`VolumeEnvelope` in this module applies a gain envelope to raw
PCM audio that arrives in consecutive chunks (e.g. while decoding or
streaming a long recording), without ever buffering the whole stream.

The envelope is described by ``(frame_position, gain_db)`` keypoints using
*absolute* sample frame positions over the whole recording.  Gain is
interpolated linearly in decibels between keypoints; before the first and
after the last keypoint the nearest gain is held constant.

Only the keypoints and the current stream position are kept as state, so
memory usage does not grow with the length of the audio.

Example - processing a long file chunk by chunk as raw PCM::

    from pydub.envelope import VolumeEnvelope

    # fade the left channel from silence to unity gain over 10 seconds
    envelope = VolumeEnvelope(
        keypoints=[(0, -60.0), (441000, 0.0)],
        channels=0,
    )

    out_chunks = []
    while True:
        pcm = source.read(8192)          # 16-bit stereo PCM bytes
        if not pcm:
            break
        out_chunks.append(
            envelope.process(pcm, sample_width=2, channel_count=2)
        )
    output_pcm = b"".join(out_chunks)

Example - processing an AudioSegment in slices::

    from pydub import AudioSegment
    from pydub.envelope import VolumeEnvelope

    segment = AudioSegment.from_file("voice_over.wav")
    envelope = VolumeEnvelope([(0, -12.0), (segment.frame_count(), 0.0)])

    processed = [envelope.process(chunk) for chunk in segment[::100]]
    result = AudioSegment(
        b"".join(processed),
        sample_width=segment.sample_width,
        channels=segment.channels,
        frame_rate=segment.frame_rate,
    )

Chunking is transparent: splitting the same PCM stream into any chunk
sizes yields exactly the same output as processing it in one piece.
"""

import math
import sys
from array import array
from bisect import bisect_right

from .utils import db_to_float

_SAMPLE_WIDTHS = (1, 2, 3, 4)

_ARRAY_TYPES = {2: "h", 4: "i"}


def _clip(sample, sample_width):
    bits = sample_width * 8
    minval = -(1 << (bits - 1))
    maxval = (1 << (bits - 1)) - 1
    return min(max(sample, minval), maxval)


def _decode_samples(data, sample_width):
    if sample_width == 1:
        return [byte - 128 for byte in data]
    if sample_width == 3:
        samples = []
        for i in range(0, len(data), 3):
            val = data[i] | (data[i + 1] << 8) | (data[i + 2] << 16)
            if val >= 1 << 23:
                val -= 1 << 24
            samples.append(val)
        return samples
    arr = array(_ARRAY_TYPES[sample_width])
    arr.frombytes(bytes(data))
    if sys.byteorder == "big":
        arr.byteswap()
    return list(arr)


def _encode_samples(samples, sample_width):
    if sample_width == 1:
        return bytes(sample + 128 for sample in samples)
    if sample_width == 3:
        out = bytearray()
        for sample in samples:
            if sample < 0:
                sample += 1 << 24
            out += bytes((sample & 0xFF, (sample >> 8) & 0xFF, (sample >> 16) & 0xFF))
        return bytes(out)
    arr = array(_ARRAY_TYPES[sample_width], samples)
    if sys.byteorder == "big":
        arr.byteswap()
    return arr.tobytes()


class VolumeEnvelope:
    """
    Applies a decibel gain envelope to a stream of PCM audio chunks.

    ``keypoints`` is an iterable of ``(frame_position, gain_db)`` pairs.
    Frame positions are absolute sample frame indices over the whole
    recording (one frame = one sample per channel), must be non-negative
    integers, and must be strictly increasing.  Gains are in decibels and
    must be finite.  Between keypoints the gain is interpolated linearly
    in decibels; outside the keypoint range the nearest gain is held.

    ``channels`` selects which channel indices the envelope is applied to
    (an int or an iterable of ints).  ``None`` (the default) applies the
    envelope to every channel.  Unselected channels are passed through
    unchanged.

    The audio format (sample width and channel count) is fixed by the
    first processed chunk; later chunks with a different format are
    rejected.  Each chunk must contain a whole number of frames so that
    no frame is ever skipped or processed twice across chunk boundaries.
    """

    def __init__(self, keypoints, channels=None):
        points = []
        for keypoint in keypoints:
            position, gain = keypoint
            if isinstance(position, float) and position.is_integer():
                position = int(position)
            if isinstance(position, bool) or not isinstance(position, int):
                raise ValueError(
                    "keypoint frame positions must be integers, got %r" % (position,)
                )
            if position < 0:
                raise ValueError(
                    "keypoint frame positions must be non-negative, got %r" % (position,)
                )
            gain = float(gain)
            if not math.isfinite(gain):
                raise ValueError("keypoint gains must be finite, got %r" % (gain,))
            if points and position <= points[-1][0]:
                raise ValueError(
                    "keypoint frame positions must be strictly increasing, "
                    "got %r after %r" % (position, points[-1][0])
                )
            points.append((position, gain))

        if not points:
            raise ValueError("at least one keypoint is required")

        self._keypoints = tuple(points)

        if channels is None:
            self._channels = None
        else:
            if isinstance(channels, int):
                channels = (channels,)
            selected = set()
            for channel in channels:
                if isinstance(channel, bool) or not isinstance(channel, int):
                    raise ValueError("channel indices must be integers, got %r" % (channel,))
                if channel < 0:
                    raise ValueError(
                        "channel indices must be non-negative, got %r" % (channel,)
                    )
                selected.add(channel)
            if not selected:
                raise ValueError("at least one channel must be selected")
            self._channels = frozenset(selected)

        self._sample_width = None
        self._channel_count = None
        self._position = 0
        self._segment = 0

    @property
    def keypoints(self):
        """The envelope keypoints as a tuple of ``(frame_position, gain_db)``."""
        return self._keypoints

    @property
    def position(self):
        """The absolute frame position of the next frame to be processed."""
        return self._position

    @property
    def sample_width(self):
        """The sample width in bytes, fixed by the first processed chunk."""
        return self._sample_width

    @property
    def channel_count(self):
        """The number of channels, fixed by the first processed chunk."""
        return self._channel_count

    def gain_db_at(self, frame_position):
        """Returns the envelope gain in dB at an absolute frame position."""
        points = self._keypoints
        if frame_position <= points[0][0]:
            return points[0][1]
        if frame_position >= points[-1][0]:
            return points[-1][1]
        positions = [p for p, _ in points]
        index = bisect_right(positions, frame_position) - 1
        (p0, g0), (p1, g1) = points[index], points[index + 1]
        return g0 + (g1 - g0) * (frame_position - p0) / (p1 - p0)

    def _check_format(self, sample_width, channel_count):
        if sample_width not in _SAMPLE_WIDTHS:
            raise ValueError(
                "sample_width must be one of %r, got %r" % (_SAMPLE_WIDTHS, sample_width)
            )
        if isinstance(channel_count, bool) or not isinstance(channel_count, int):
            raise ValueError("channel_count must be an integer, got %r" % (channel_count,))
        if channel_count < 1:
            raise ValueError("channel_count must be at least 1, got %r" % (channel_count,))

        if self._sample_width is None:
            self._sample_width = sample_width
            self._channel_count = channel_count
            if self._channels is not None and max(self._channels) >= channel_count:
                raise ValueError(
                    "selected channel %d is out of range for %d channel(s)"
                    % (max(self._channels), channel_count)
                )
        elif (sample_width, channel_count) != (self._sample_width, self._channel_count):
            raise ValueError(
                "chunk format (%d byte samples, %d channels) does not match the "
                "format of the first chunk (%d byte samples, %d channels)"
                % (sample_width, channel_count, self._sample_width, self._channel_count)
            )

    def _chunk_gains_db(self, frame_count):
        points = self._keypoints
        first_position, first_gain = points[0]
        last_position, last_gain = points[-1]
        segment = self._segment
        gains = []
        for offset in range(frame_count):
            frame = self._position + offset
            while segment + 1 < len(points) and frame > points[segment + 1][0]:
                segment += 1
            if frame <= first_position:
                gains.append(first_gain)
            elif frame >= last_position:
                gains.append(last_gain)
            else:
                (p0, g0), (p1, g1) = points[segment], points[segment + 1]
                gains.append(g0 + (g1 - g0) * (frame - p0) / (p1 - p0))
        self._segment = segment
        return gains

    def process(self, chunk, sample_width=None, channel_count=None):
        """
        Processes one chunk of PCM audio and returns the processed bytes.

        ``chunk`` is a bytes-like object of PCM data, or an
        :class:`~pydub.AudioSegment` (in which case the format is taken
        from the segment unless explicitly overridden).  The returned
        bytes always contain exactly as many frames as the input chunk.

        ``sample_width`` is the bytes per sample (1, 2, 3 or 4) and
        ``channel_count`` the number of interleaved channels.  The first
        processed chunk fixes the stream format; later chunks must match.
        """
        if hasattr(chunk, "raw_data"):
            if sample_width is None:
                sample_width = chunk.sample_width
            if channel_count is None:
                channel_count = chunk.channels
            chunk = chunk.raw_data

        if sample_width is None or channel_count is None:
            raise ValueError(
                "sample_width and channel_count are required when processing "
                "raw PCM bytes"
            )

        self._check_format(sample_width, channel_count)

        frame_size = self._sample_width * self._channel_count
        if len(chunk) % frame_size != 0:
            raise ValueError(
                "chunk of %d bytes is not a whole number of %d byte frames; "
                "partial frames would be skipped or repeated across chunks"
                % (len(chunk), frame_size)
            )

        frame_count = len(chunk) // frame_size
        if frame_count == 0:
            return b""

        samples = _decode_samples(chunk, self._sample_width)
        gains_db = self._chunk_gains_db(frame_count)
        factors = [db_to_float(gain_db) for gain_db in gains_db]

        if self._channels is None:
            selected = range(self._channel_count)
        else:
            selected = sorted(self._channels)

        channel_count = self._channel_count
        sample_width = self._sample_width
        out = list(samples)
        for frame in range(frame_count):
            factor = factors[frame]
            base = frame * channel_count
            for channel in selected:
                index = base + channel
                out[index] = _clip(int(samples[index] * factor), sample_width)

        self._position += frame_count
        return _encode_samples(out, sample_width)
