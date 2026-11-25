"""Stateful PCM sample-rate conversion.

``PCMResampler`` is intended for streamed or chunked PCM audio.  It keeps
only a short input look-ahead window and a small filter history; it never
retains the complete input.  Output samples are computed from their global
sample index, so the result does not depend on how the input was split into
chunks.

Example::

    from pydub import AudioSegment
    from pydub.resampler import PCMChunk, PCMResampler

    segment = AudioSegment.from_wav("input.wav")
    resampler = PCMResampler(output_frame_rate=16000)
    output = []
    frames_per_chunk = 320
    frame_width = segment.frame_width
    raw = segment.raw_data
    for start in range(0, int(segment.frame_count()), frames_per_chunk):
        begin = start * frame_width
        end = min(len(raw), (start + frames_per_chunk) * frame_width)
        chunk = PCMChunk(
            raw[begin:end],
            segment.sample_width,
            segment.channels,
            segment.frame_rate,
        )
        output.append(resampler.process(chunk))
    output.append(resampler.finish())

    audio = AudioSegment(
        data=b"".join(output),
        sample_width=resampler.sample_width,
        frame_rate=16000,
        channels=resampler.channels,
    )

The first chunk supplies the input format when it is an ``AudioSegment`` (or
another object carrying ``raw_data``, ``sample_width``, ``channels`` and
``frame_rate``).  For raw bytes the same values can be supplied to the
constructor or to the first ``process()`` call.
"""

from __future__ import division

import math
import operator
import sys
from array import array
from collections import namedtuple

from .exceptions import PCMFormatError, PCMFormatMismatchError

try:
    from fractions import gcd
except ImportError:  # Python 3.9+
    from math import gcd


PCMFormat = namedtuple("PCMFormat", ["sample_width", "channels", "frame_rate"])
PCMChunk = namedtuple("PCMChunk", ["data", "sample_width", "channels", "frame_rate"])

_ARRAY_CODES = {
    1: "b",
    2: "h",
    4: "i",
}

_SIGNED_MIN = {
    1: -0x80,
    2: -0x8000,
    3: -0x800000,
    4: -0x80000000,
}

_SIGNED_MAX = {
    1: 0x7f,
    2: 0x7fff,
    3: 0x7fffff,
    4: 0x7fffffff,
}


def _as_int(value, name, minimum=None):
    if isinstance(value, bool):
        raise PCMFormatError("%s must be an integer" % name)
    try:
        converted = int(value)
    except (TypeError, ValueError):
        raise PCMFormatError("%s must be an integer" % name)
    if converted != value:
        raise PCMFormatError("%s must be an integer" % name)
    if minimum is not None and converted < minimum:
        raise PCMFormatError("%s must be >= %s" % (name, minimum))
    return converted


def _validate_format(sample_width, channels, frame_rate):
    sample_width = _as_int(sample_width, "sample_width")
    channels = _as_int(channels, "channels", 1)
    frame_rate = _as_int(frame_rate, "frame_rate", 1)
    if sample_width not in (1, 2, 3, 4):
        raise PCMFormatError("sample_width must be 1, 2, 3 or 4 bytes")
    return PCMFormat(sample_width, channels, frame_rate)


def _decode_pcm(data, sample_width):
    """Decode little-endian signed PCM into an array of Python integers."""
    if sample_width == 3:
        count = len(data) // 3
        result = array("i")
        view = memoryview(data)
        for index in range(count):
            position = index * 3
            value = view[position] | (view[position + 1] << 8) | (view[position + 2] << 16)
            if value & 0x800000:
                value -= 0x1000000
            result.append(value)
        return result

    values = array(_ARRAY_CODES[sample_width])
    values.frombytes(data)
    if sys.byteorder == "big":
        values.byteswap()
    return values


def _encode_pcm(values, sample_width):
    """Encode clipped little-endian signed PCM from integer samples."""
    if sample_width == 3:
        result = bytearray(len(values) * 3)
        for index, value in enumerate(values):
            value = int(round(value))
            value = max(_SIGNED_MIN[3], min(_SIGNED_MAX[3], value))
            if value < 0:
                value += 0x1000000
            position = index * 3
            result[position] = value & 0xff
            result[position + 1] = (value >> 8) & 0xff
            result[position + 2] = (value >> 16) & 0xff
        return bytes(result)

    minimum = _SIGNED_MIN[sample_width]
    maximum = _SIGNED_MAX[sample_width]
    clipped = array(
        _ARRAY_CODES[sample_width],
        (max(minimum, min(maximum, int(round(value)))) for value in values),
    )
    if sys.byteorder == "big":
        clipped.byteswap()
    return clipped.tobytes()


def _round_half_up(numerator, denominator):
    """Round a non-negative rational number to the nearest integer."""
    return (numerator + denominator // 2) // denominator


def _blackman(value):
    # value is tau / half_width in [-1, 1].
    return 0.42 + 0.5 * math.cos(math.pi * value) + 0.08 * math.cos(2 * math.pi * value)


def _sinc(value):
    if value == 0.0:
        return 1.0
    scaled = math.pi * value
    return math.sin(scaled) / scaled


class PCMResampler(object):
    """Convert chunked PCM audio to another sample rate without seams.

    Parameters
    ----------
    output_frame_rate:
        Target sample rate in Hz.
    input_frame_rate, sample_width, channels:
        Optional format hints.  Any value omitted here is taken from the first
        input chunk.  Raw byte chunks require all three values to be known.
    output_sample_width:
        Optional output sample width.  Defaults to the input sample width.
        A different width is scaled in the same way as
        ``AudioSegment.set_sample_width``.
    filter_half_width:
        Half-width of the windowed-sinc kernel in input frames.  Larger values
        improve stop-band attenuation at the cost of more look-ahead.  The
        default of 64 adds about 1.33 ms of look-ahead at 48 kHz.

    Timing and boundaries
    ---------------------
    Output frame ``k`` represents input time ``k * input_rate / output_rate``.
    The processor does not add a time offset.  It only withholds samples until
    enough future input has arrived to evaluate the symmetric filter.  Missing
    input before the first frame and after the last frame is treated as zero,
    so the first and last full kernel windows are edge transients.  The number
    of output frames is ``round(input_frames * output_rate / input_rate)``.

    ``finish()`` emits those final frames and closes the input.  Calling it
    again returns ``b""``; it never emits duplicate samples.
    """

    def __init__(
        self,
        output_frame_rate=None,
        input_frame_rate=None,
        sample_width=None,
        channels=None,
        output_sample_width=None,
        filter_half_width=64,
        **kwargs
    ):
        if output_frame_rate is None:
            output_frame_rate = kwargs.pop(
                "out_frame_rate",
                kwargs.pop(
                    "out_rate",
                    kwargs.pop(
                        "output_sample_rate",
                        kwargs.pop(
                            "out_sample_rate",
                            kwargs.pop(
                                "target_frame_rate",
                                kwargs.pop(
                                    "target_sample_rate",
                                    kwargs.pop("output_rate", kwargs.pop("sample_rate", None)),
                                ),
                            ),
                        ),
                    ),
                ),
            )
        if input_frame_rate is None:
            input_frame_rate = kwargs.pop(
                "in_frame_rate",
                kwargs.pop(
                    "in_rate",
                    kwargs.pop(
                        "input_sample_rate",
                        kwargs.pop(
                            "in_sample_rate",
                            kwargs.pop(
                                "source_frame_rate",
                                kwargs.pop("input_rate", kwargs.pop("source_rate", None)),
                            ),
                        ),
                    ),
                ),
            )
        if channels is None:
            channels = kwargs.pop("nchannels", kwargs.pop("num_channels", None))
        if sample_width is None:
            sample_width = kwargs.pop("width", None)
        if kwargs:
            unexpected = next(iter(kwargs))
            raise TypeError("unexpected keyword argument %r" % unexpected)

        if output_frame_rate is None:
            raise TypeError("output_frame_rate is required")

        self.output_frame_rate = _as_int(output_frame_rate, "output_frame_rate", 1)
        self.filter_half_width = _as_int(filter_half_width, "filter_half_width", 1)
        self._input_frame_rate_hint = (
            None if input_frame_rate is None else _as_int(input_frame_rate, "input_frame_rate", 1)
        )
        self._sample_width_hint = (
            None if sample_width is None else _as_int(sample_width, "sample_width")
        )
        self._channels_hint = None if channels is None else _as_int(channels, "channels", 1)
        self._output_sample_width = (
            None if output_sample_width is None
            else _as_int(output_sample_width, "output_sample_width")
        )

        self._format = None
        self._sample_width = None
        self._channels = None
        self._input_frame_rate = None
        self._output_sample_width_resolved = None

        self._input_frames = 0
        self._next_output_index = 0
        self._history_start = 0
        self._history = []
        self._phase_cache = {}
        self._reduced_input = None
        self._reduced_output = None
        self._finished = False

    @property
    def output_frame_count(self):
        """Number of output frames emitted so far."""
        return self._next_output_index

    @property
    def input_frame_count(self):
        """Number of complete input frames accepted so far."""
        return self._input_frames

    @property
    def buffered_input_frames(self):
        """Number of input frames currently retained by the filter state."""
        if not self._history:
            return 0
        return len(self._history[0])

    @property
    def latency_frames(self):
        """Input look-ahead required before an output frame can be emitted."""
        return self.filter_half_width

    @property
    def finished(self):
        return self._finished

    @property
    def format(self):
        return self._format

    def _set_format(self, detected, supplied):
        candidates = []
        for candidate in (detected, supplied):
            if candidate is None:
                continue
            if isinstance(candidate, PCMFormat):
                candidates.append(candidate)
            else:
                candidates.append(PCMFormat(*candidate))

        if not candidates:
            raise PCMFormatError(
                "the first chunk did not provide a PCM format; supply "
                "sample_width, channels and input_frame_rate"
            )

        resolved = _validate_format(*candidates[0])
        for candidate in candidates[1:]:
            candidate = _validate_format(*candidate)
            if candidate != resolved:
                raise PCMFormatMismatchError(
                    "input format changed: expected %s, got %s" % (resolved, candidate)
                )

        if self._sample_width_hint is not None and resolved.sample_width != self._sample_width_hint:
            raise PCMFormatMismatchError(
                "sample_width changed: expected %s, got %s"
                % (self._sample_width_hint, resolved.sample_width)
            )
        if self._channels_hint is not None and resolved.channels != self._channels_hint:
            raise PCMFormatMismatchError(
                "channels changed: expected %s, got %s"
                % (self._channels_hint, resolved.channels)
            )
        if (
            self._input_frame_rate_hint is not None
            and resolved.frame_rate != self._input_frame_rate_hint
        ):
            raise PCMFormatMismatchError(
                "input_frame_rate changed: expected %s, got %s"
                % (self._input_frame_rate_hint, resolved.frame_rate)
            )

        self._format = resolved
        self._sample_width = resolved.sample_width
        self._channels = resolved.channels
        self._input_frame_rate = resolved.frame_rate
        history_code = _ARRAY_CODES.get(self._sample_width, "i")
        self._history = [array(history_code) for _ in range(self._channels)]

        if self._output_sample_width is None:
            self._output_sample_width_resolved = self._sample_width
        else:
            if self._output_sample_width not in (1, 2, 3, 4):
                raise PCMFormatError("output_sample_width must be 1, 2, 3 or 4 bytes")
            self._output_sample_width_resolved = self._output_sample_width

        divisor = gcd(self._input_frame_rate, self.output_frame_rate)
        self._reduced_input = self._input_frame_rate // divisor
        self._reduced_output = self.output_frame_rate // divisor

    def _coerce_chunk(self, chunk, supplied_format):
        if isinstance(chunk, (bytes, bytearray, memoryview)):
            return bytes(chunk), supplied_format

        data = None
        detected = None

        if isinstance(chunk, dict):
            data = chunk.get("raw_data", chunk.get("data", chunk.get("pcm")))
            if data is None:
                raise TypeError("chunk dictionary must contain 'data' or 'raw_data'")
            width = chunk.get("sample_width", chunk.get("width"))
            channels = chunk.get("channels", chunk.get("num_channels"))
            frame_rate = chunk.get("frame_rate", chunk.get("sample_rate"))
            if width is not None and channels is not None and frame_rate is not None:
                detected = PCMFormat(width, channels, frame_rate)
        elif isinstance(chunk, (tuple, list)):
            if len(chunk) == 2 and isinstance(chunk[0], (tuple, list)) and len(chunk[0]) == 3:
                detected = PCMFormat(*chunk[0])
                data = chunk[1]
            elif len(chunk) == 2 and isinstance(chunk[1], (tuple, list)) and len(chunk[1]) == 3:
                data = chunk[0]
                detected = PCMFormat(*chunk[1])
            elif len(chunk) == 2:
                data, frame_rate = chunk
                if self._input_frame_rate is not None:
                    detected = PCMFormat(
                        self._sample_width, self._channels, frame_rate
                    )
                elif (
                    self._sample_width_hint is not None
                    and self._channels_hint is not None
                ):
                    detected = PCMFormat(
                        self._sample_width_hint, self._channels_hint, frame_rate
                    )
                elif (
                    supplied_format is not None
                    and supplied_format.sample_width is not None
                    and supplied_format.channels is not None
                ):
                    detected = PCMFormat(
                        supplied_format.sample_width,
                        supplied_format.channels,
                        frame_rate,
                    )
                else:
                    raise PCMFormatError(
                        "a 2-item PCM chunk needs sample_width and channels "
                        "from the constructor or process()"
                    )
            elif len(chunk) == 3:
                data, sample_width, channels = chunk
                if self._input_frame_rate is not None:
                    detected = PCMFormat(sample_width, channels, self._input_frame_rate)
                elif self._input_frame_rate_hint is not None:
                    detected = PCMFormat(sample_width, channels, self._input_frame_rate_hint)
                elif supplied_format is not None and supplied_format.frame_rate is not None:
                    detected = PCMFormat(sample_width, channels, supplied_format.frame_rate)
                else:
                    raise PCMFormatError(
                        "a 3-item PCM chunk needs input_frame_rate from the "
                        "constructor or process()"
                    )
            elif len(chunk) == 4:
                data, sample_width, channels, frame_rate = chunk
                detected = PCMFormat(sample_width, channels, frame_rate)
            else:
                raise TypeError("PCM chunk tuple must have 2, 3 or 4 items")
        else:
            data = getattr(chunk, "raw_data", getattr(chunk, "data", None))
            if data is None:
                raise TypeError("chunk must be bytes, a PCM chunk, or AudioSegment")
            width = getattr(chunk, "sample_width", getattr(chunk, "width", None))
            channels = getattr(chunk, "channels", getattr(chunk, "num_channels", None))
            frame_rate = getattr(
                chunk, "frame_rate", getattr(chunk, "sample_rate", None)
            )
            if width is not None and channels is not None and frame_rate is not None:
                detected = PCMFormat(width, channels, frame_rate)

        if not isinstance(data, (bytes, bytearray, memoryview)):
            raise TypeError("PCM chunk data must be bytes-like")

        if detected is not None and supplied_format is not None:
            if _validate_format(*detected) != _validate_format(*supplied_format):
                raise PCMFormatMismatchError(
                    "chunk format %s does not match supplied format %s"
                    % (_validate_format(*detected), _validate_format(*supplied_format))
                )
            detected = None

        return bytes(data), detected if detected is not None else supplied_format

    def _phase(self, phase):
        cached = self._phase_cache.get(phase)
        if cached is not None:
            return cached

        ratio = self.output_frame_rate / self._input_frame_rate
        cutoff = min(1.0, ratio)
        offset = phase / self._reduced_output
        half_width = self.filter_half_width
        first = int(math.ceil(offset - half_width))
        last = int(math.floor(offset + half_width))

        coefficients = []
        for tap in range(first, last + 1):
            distance = offset - tap
            weight = cutoff * _sinc(cutoff * distance) * _blackman(distance / half_width)
            coefficients.append(weight)

        cached = (first, tuple(coefficients))
        if self._reduced_output <= 4096:
            self._phase_cache[phase] = cached
        return cached

    def _output_frame(self, output_index, include_future):
        numerator = output_index * self._reduced_input
        source_frame, phase = divmod(numerator, self._reduced_output)
        first_offset, coefficients = self._phase(phase)

        first_sample = source_frame + first_offset
        last_sample = first_sample + len(coefficients) - 1

        if not include_future and last_sample >= self._input_frames:
            return None

        if first_sample < 0:
            coefficient_start = -first_sample
        else:
            coefficient_start = 0

        if last_sample >= self._input_frames:
            coefficient_end = len(coefficients) - (last_sample - self._input_frames + 1)
        else:
            coefficient_end = len(coefficients)

        if coefficient_start >= coefficient_end:
            return [0] * self._channels

        sample_start = max(first_sample, 0)
        relative_start = sample_start - self._history_start
        relative_end = relative_start + (coefficient_end - coefficient_start)

        result = []
        for channel in range(self._channels):
            samples = self._history[channel][relative_start:relative_end]
            channel_coefficients = coefficients[coefficient_start:coefficient_end]
            value = sum(map(operator.mul, samples, channel_coefficients))
            result.append(value)
        return result

    def _emit_ready(self, output_chunks):
        while True:
            output = self._output_frame(self._next_output_index, include_future=False)
            if output is None:
                break
            output_chunks.append(output)
            self._next_output_index += 1

    def _trim_history(self):
        source_frame, phase = divmod(
            self._next_output_index * self._reduced_input, self._reduced_output
        )
        first_offset, _ = self._phase(phase)
        keep_from = max(0, source_frame + first_offset)
        if keep_from > self._history_start and self._history:
            amount = keep_from - self._history_start
            for channel in self._history:
                del channel[:amount]
            self._history_start = keep_from

    def _append_pcm(self, data, output_chunks):
        frame_width = self._sample_width * self._channels
        # Keep processor-owned state bounded even when a very large chunk is
        # supplied.  The input itself is never retained beyond this block.
        max_block_frames = max(1, self.filter_half_width * 2)
        block_bytes = max_block_frames * frame_width

        for start in range(0, len(data), block_bytes):
            block = data[start:start + block_bytes]
            values = _decode_pcm(block, self._sample_width)
            frames = len(values) // self._channels
            for channel in range(self._channels):
                self._history[channel].extend(values[channel::self._channels])
            self._input_frames += frames
            self._emit_ready(output_chunks)
            self._trim_history()

    def _encode_outputs(self, output_chunks):
        if not output_chunks:
            return b""
        if self._channels == 1:
            values = []
            for frame in output_chunks:
                values.append(frame[0])
        else:
            values = []
            for frame in output_chunks:
                values.extend(frame)
        if self._output_sample_width_resolved > self._sample_width:
            shift = 8 * (self._output_sample_width_resolved - self._sample_width)
            scale = 1 << shift
            values = [value * scale for value in values]
        elif self._output_sample_width_resolved < self._sample_width:
            shift = 8 * (self._sample_width - self._output_sample_width_resolved)
            divisor = 1 << shift
            values = [value / divisor for value in values]
        return _encode_pcm(values, self._output_sample_width_resolved)

    def process(
        self,
        chunk,
        sample_width=None,
        channels=None,
        frame_rate=None,
        **kwargs
    ):
        """Accept one complete-frame PCM chunk and return newly ready PCM.

        ``chunk`` may be bytes, an ``AudioSegment``, a ``PCMChunk`` or a tuple
        ``(data, sample_width, channels, frame_rate)``.  Raw bytes use format
        hints from the constructor or from the keyword arguments here.
        The input object is only read, never modified.
        """
        if frame_rate is None:
            frame_rate = kwargs.pop(
                "input_frame_rate",
                kwargs.pop("sample_rate", kwargs.pop("rate", None)),
            )
        if kwargs:
            unexpected = next(iter(kwargs))
            raise TypeError("unexpected keyword argument %r" % unexpected)

        if self._finished:
            raise ValueError("cannot process more audio after finish()")

        supplied = None
        if sample_width is not None or channels is not None or frame_rate is not None:
            if sample_width is None:
                sample_width = (
                    self._format.sample_width
                    if self._format is not None
                    else self._sample_width_hint
                )
            if channels is None:
                channels = (
                    self._format.channels
                    if self._format is not None
                    else self._channels_hint
                )
            if frame_rate is None:
                frame_rate = (
                    self._format.frame_rate
                    if self._format is not None
                    else self._input_frame_rate_hint
                )
            if sample_width is None or channels is None or frame_rate is None:
                raise PCMFormatError(
                    "sample_width, channels and frame_rate could not be resolved"
                )
            supplied = PCMFormat(sample_width, channels, frame_rate)

        data, detected = self._coerce_chunk(chunk, supplied)

        if self._format is None:
            if detected is None:
                hint_width = self._sample_width_hint
                hint_channels = self._channels_hint
                hint_rate = self._input_frame_rate_hint
                if hint_width is not None and hint_channels is not None and hint_rate is not None:
                    detected = PCMFormat(hint_width, hint_channels, hint_rate)
            self._set_format(detected, None)
        elif detected is not None:
            candidate = _validate_format(*detected)
            if candidate != self._format:
                raise PCMFormatMismatchError(
                    "input format changed: expected %s, got %s"
                    % (self._format, candidate)
                )

        if (
            self._sample_width_hint is not None
            and self._format.sample_width != self._sample_width_hint
        ):
            raise PCMFormatMismatchError(
                "sample_width changed: expected %s, got %s"
                % (self._sample_width_hint, self._format.sample_width)
            )
        if self._channels_hint is not None and self._format.channels != self._channels_hint:
            raise PCMFormatMismatchError(
                "channels changed: expected %s, got %s"
                % (self._channels_hint, self._format.channels)
            )
        if (
            self._input_frame_rate_hint is not None
            and self._format.frame_rate != self._input_frame_rate_hint
        ):
            raise PCMFormatMismatchError(
                "input_frame_rate changed: expected %s, got %s"
                % (self._input_frame_rate_hint, self._format.frame_rate)
            )

        frame_width = self._format.sample_width * self._format.channels
        if len(data) % frame_width:
            raise ValueError(
                "PCM chunk length %s is not a multiple of frame width %s"
                % (len(data), frame_width)
            )

        output_chunks = []
        self._append_pcm(data, output_chunks)
        return self._encode_outputs(output_chunks)

    feed = process
    push = process
    write = process
    process_chunk = process
    convert = process
    resample = process

    def finish(self):
        """Emit the remaining output frames; repeated calls return ``b""``."""
        if self._finished:
            return b""
        self._finished = True
        if self._format is None:
            return b""

        total_output_frames = _round_half_up(
            self._input_frames * self._reduced_output, self._reduced_input
        )
        output_chunks = []
        while self._next_output_index < total_output_frames:
            output = self._output_frame(self._next_output_index, include_future=True)
            output_chunks.append(output)
            self._next_output_index += 1

        encoded = self._encode_outputs(output_chunks)
        self._history = []
        self._history_start = self._input_frames
        return encoded

    flush = finish
    drain = finish
    end = finish
    close = finish

    def process_segment(self, segment):
        """Process an ``AudioSegment`` and return an ``AudioSegment``."""
        from .audio_segment import AudioSegment

        data = self.process(segment)
        return AudioSegment(
            data=data,
            sample_width=self.output_sample_width,
            frame_rate=self.output_frame_rate,
            channels=self.channels,
        )

    def finish_segment(self):
        """Return the drained tail as an ``AudioSegment``."""
        from .audio_segment import AudioSegment

        data = self.finish()
        return AudioSegment(
            data=data,
            sample_width=self.output_sample_width,
            frame_rate=self.output_frame_rate,
            channels=self.channels,
        )

    @property
    def sample_width(self):
        return self._sample_width

    @property
    def channels(self):
        return self._channels

    @property
    def input_frame_rate(self):
        return self._input_frame_rate

    @property
    def input_rate(self):
        return self._input_frame_rate

    @property
    def in_frame_rate(self):
        return self._input_frame_rate

    @property
    def output_rate(self):
        return self.output_frame_rate

    @property
    def out_frame_rate(self):
        return self.output_frame_rate

    @property
    def sample_rate(self):
        return self.output_frame_rate

    @property
    def output_sample_width(self):
        return self._output_sample_width_resolved


StatefulPCMResampler = PCMResampler
StreamingPCMResampler = PCMResampler
PCMStreamResampler = PCMResampler
PCMResamplingProcessor = PCMResampler
StatefulResampler = PCMResampler
StreamingResampler = PCMResampler
Resampler = PCMResampler


def resample_pcm(chunks, output_frame_rate, **kwargs):
    """Convenience helper that resamples an iterable of PCM chunks."""
    resampler = PCMResampler(output_frame_rate=output_frame_rate, **kwargs)
    output = []
    for chunk in chunks:
        output.append(resampler.process(chunk))
    output.append(resampler.finish())
    return b"".join(output)

