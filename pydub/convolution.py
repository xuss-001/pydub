"""
Streaming convolution reverb.

Convolution with an impulse response (IR) describes the acoustics of a room
and can be applied to arbitrarily long recordings one block at a time. The
:class:`ConvolutionReverb` keeps only the overlap history between blocks, so
the memory it requires depends on the length of the impulse response rather
than on the total length of the recording.

Example::

    reverb = ConvolutionReverb(impulse_response, dry_gain=1.0, wet_gain=0.8)
    output = AudioSegment.empty()
    for chunk in make_chunks(long_recording, 100):
        output += reverb.process(chunk)
    output += reverb.flush()

Impulse response samples are normalized against the full-scale amplitude of
their own bit depth, so an IR whose peak reaches full scale is treated as a
unit coefficient regardless of the source bit depth.
"""

import math

import numpy as np

from .exceptions import ConvolutionError


def _is_finite_number(value):
    if isinstance(value, bool):
        return False
    if isinstance(value, (int, float)):
        return math.isfinite(value)
    if isinstance(value, np.number):
        return bool(np.isfinite(value))
    return False


def _ensure_finite_gain(gain, name):
    if not _is_finite_number(gain):
        raise ValueError(
            "{0} must be a finite linear gain (got {1!r})".format(name, gain))
    return float(gain)


class ConvolutionReverb(object):
    """
    Apply convolution reverb to a stream of :class:`AudioSegment` blocks.

    Parameters
    ----------
    impulse_response : AudioSegment
        Audio describing the room acoustics. Its sample rate and sample
        width are converted to the format of the first input block. A mono
        response is applied to every input channel; a response with the same
        number of channels as the input is applied channel by channel.
    dry_gain : float
        Linear gain applied to the unprocessed (original) signal. Must be
        finite.
    wet_gain : float
        Linear gain applied to the convolved (reverberant) signal. Must be
        finite.

    Yielded blocks have the same frame rate, sample width and channel count
    as the first processed block. After all input blocks have been processed,
    :meth:`flush` returns the remaining reverb tail so that no samples are
    lost or duplicated by the block boundaries.
    """

    def __init__(self, impulse_response, dry_gain=1.0, wet_gain=1.0):
        self.dry_gain = _ensure_finite_gain(dry_gain, "dry_gain")
        self.wet_gain = _ensure_finite_gain(wet_gain, "wet_gain")

        if impulse_response.frame_count() < 1:
            raise ConvolutionError(
                "impulse_response must contain at least one sample")

        self._ir_segment = impulse_response
        self._ir_channels = None
        self._kernel_length = None
        self._history = None
        self._template = None
        self._dtype = None
        self._min_value = None
        self._max_value = None
        self._finished = False

    def _initialize(self, chunk):
        ir = self._ir_segment
        if ir.frame_rate != chunk.frame_rate:
            ir = ir.set_frame_rate(chunk.frame_rate)
        if ir.sample_width != chunk.sample_width:
            ir = ir.set_sample_width(chunk.sample_width)

        if ir.channels != 1 and ir.channels != chunk.channels:
            raise ConvolutionError(
                "impulse response has {0} channels but the input has {1}; "
                "expected 1 or {1}".format(ir.channels, chunk.channels))

        # PCM full scale is the largest positive value representable by the
        # response's bit depth (e.g. 32767 for 16-bit audio).
        full_scale = float(2 ** (ir.sample_width * 8 - 1) - 1)
        ir_samples = ir.get_array_of_samples()
        if ir.channels == 1:
            kernels = [np.fromiter(ir_samples, dtype=np.float64) / full_scale]
        else:
            interleaved = np.fromiter(ir_samples, dtype=np.float64)
            interleaved = interleaved.reshape(-1, ir.channels)
            kernels = [
                interleaved[:, channel] / full_scale
                for channel in range(ir.channels)
            ]

        kernel_length = len(kernels[0])
        if kernel_length < 1:
            raise ConvolutionError(
                "impulse_response must contain at least one sample")

        self._ir_channels = kernels
        self._kernel_length = kernel_length
        self._history = [
            np.zeros(kernel_length - 1, dtype=np.float64)
            for _ in range(chunk.channels)
        ]
        self._template = chunk
        self._dtype = {
            1: np.int8,
            2: np.int16,
            4: np.int32,
        }[chunk.sample_width]
        self._min_value = -(2 ** (chunk.sample_width * 8 - 1))
        self._max_value = 2 ** (chunk.sample_width * 8 - 1) - 1

    def _check_format(self, chunk):
        template = self._template
        if (chunk.frame_rate != template.frame_rate
                or chunk.sample_width != template.sample_width
                or chunk.channels != template.channels):
            raise ConvolutionError(
                "all input blocks must match the first block "
                "({0} Hz, {1}-byte samples, {2} channels); got "
                "({3} Hz, {4}-byte samples, {5} channels)".format(
                    template.frame_rate, template.sample_width,
                    template.channels, chunk.frame_rate,
                    chunk.sample_width, chunk.channels))

    def _input_channels(self, samples):
        channels = self._template.channels
        reshaped = samples.reshape(-1, channels)
        return [reshaped[:, channel] for channel in range(channels)]

    def _convolve_block(self, samples, kernel, history):
        # Full linear convolution of (history + new samples) with the
        # kernel. Only the part aligned with the new samples is emitted;
        # the tail is retained in ``history`` for the next block.
        combined = np.concatenate((history, samples))
        block_length = len(samples)
        fft_size = 1
        while fft_size < len(combined) + len(kernel) - 1:
            fft_size <<= 1
        spectrum = (
            np.fft.rfft(combined, fft_size)
            * np.fft.rfft(kernel, fft_size)
        )
        convolved = np.fft.irfft(spectrum, fft_size)
        start = len(history)
        output = convolved[start:start + block_length]
        new_history = combined[-(len(kernel) - 1):].copy()
        return output, new_history

    def process(self, chunk):
        """
        Process one input block and return the corresponding output block.

        The returned :class:`AudioSegment` has the same number of frames as
        the input block; the reverb tail that extends beyond it is emitted
        later by :meth:`flush`.
        """
        if self._finished:
            raise ConvolutionError(
                "this reverb has already been flushed; create a new "
                "ConvolutionReverb for another stream")

        if self._template is None:
            if chunk.frame_count() < 1:
                # Nothing to anchor the stream format to yet; an empty
                # block simply produces an empty result.
                return chunk
            self._initialize(chunk)
        elif chunk.frame_count() < 1:
            self._check_format(chunk)
            return chunk
        else:
            self._check_format(chunk)

        raw = np.frombuffer(chunk._data, dtype=self._dtype)
        input_channels = self._input_channels(raw)

        output_channels = []
        for index, samples in enumerate(input_channels):
            kernel = self._ir_channels[0 if len(self._ir_channels) == 1
                                       else index]
            wet, self._history[index] = self._convolve_block(
                samples.astype(np.float64), kernel, self._history[index])
            mixed = (self.dry_gain * samples.astype(np.float64)
                     + self.wet_gain * wet)
            output_channels.append(mixed)

        return self._build_segment(output_channels)

    def _build_segment(self, output_channels):
        interleaved = np.stack(output_channels, axis=1)
        clipped = np.clip(np.rint(interleaved),
                          self._min_value, self._max_value)
        integer_samples = clipped.astype(self._dtype)
        return self._template._spawn(integer_samples.tobytes())

    def flush(self):
        """
        Return the remaining reverb tail after the last input block.

        The tail contains ``len(impulse_response) - 1`` frames (zero frames
        when the impulse response is a single sample). Calling this more
        than once returns empty segments instead of repeating samples.
        """
        if self._template is None:
            from .audio_segment import AudioSegment
            return AudioSegment.empty()

        if self._finished:
            return self._build_empty()

        self._finished = True

        tail_length = self._kernel_length - 1
        if tail_length == 0:
            return self._build_empty()

        output_channels = []
        for index, history in enumerate(self._history):
            kernel = self._ir_channels[0 if len(self._ir_channels) == 1
                                       else index]
            # The retained history holds the last (kernel_length - 1) input
            # samples; its convolution with the kernel starts at kernel
            # tap 1, which is exactly the un-emitted tail of the full
            # input-by-kernel convolution.
            tail = np.convolve(history, kernel)[tail_length:]
            output_channels.append(self.wet_gain * tail)

        tail_segment = self._build_segment(output_channels)
        return tail_segment

    def _build_empty(self):
        return self._template._spawn(b"")


def convolve(segment, impulse_response, dry_gain=1.0, wet_gain=1.0):
    """
    Convolve an :class:`AudioSegment` with an impulse response in one call.

    This is a convenience wrapper around :class:`ConvolutionReverb` and
    returns the wet signal followed by its reverb tail. An empty input
    segment produces an empty result.
    """
    if segment.frame_count() < 1:
        return segment

    reverb = ConvolutionReverb(
        impulse_response, dry_gain=dry_gain, wet_gain=wet_gain)
    return reverb.process(segment) + reverb.flush()
