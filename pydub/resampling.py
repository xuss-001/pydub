"""
Stateful, streaming PCM resampling.

This module provides :class:`PCMResampler`, an incremental resampler for
raw PCM audio carried in :class:`pydub.AudioSegment` chunks.  It is designed
for scenarios where audio arrives in blocks (network streams, files read in
pieces, live capture) and must be resampled on the fly without buffering the
whole stream and without audible seams between blocks.

Design notes
------------
The resampler evaluates a Kaiser-windowed-sinc FIR low-pass filter at the
exact (fractional) input position of every output sample.  Because each
output sample is a pure function of the global input stream (never of how
the stream was chopped into chunks), feeding the same audio as one big
chunk, in arbitrary pieces, or frame by frame yields bit-identical output.

Anti-aliasing
-------------
The filter cut-off is ``0.5 * min(1, output_rate / input_rate)`` cycles per
input sample, i.e. the lower of the input and output Nyquist frequencies,
so downsampling suppresses images that would otherwise alias.  The Kaiser
window gives ~60 dB of stop-band attenuation.

Latency and head/tail handling
------------------------------
The FIR is causal-windowed around each output's ideal position, so an
output sample can only be emitted once all input samples under its window
have been received.  The steady-state latency is therefore half the filter
length, ``resampler.latency`` input samples (plus whatever buffering the
caller's chunk size adds).  Input samples before the start of the stream
and (after :meth:`flush`) past the end of the stream are treated as zeros.
Consequently the first and last ``latency`` input samples worth of output
form fade-in/fade-out transition regions; the interior is bit-exact.
"""

import math
from fractions import Fraction

import numpy as np

from .audio_segment import AudioSegment

_SAMPLE_WIDTH_TO_DTYPE = {
    1: np.int8,
    2: np.int16,
    4: np.int32,
}


class PCMResampler(object):
    """
    Incrementally resample a stream of PCM audio chunks.

    Usage::

        resampler = PCMResampler(output_rate=16000)
        for chunk in incoming_chunks:          # AudioSegment pieces
            out_chunk = resampler.process(chunk)
            send(out_chunk)
        tail = resampler.flush()               # drain remaining samples
        send(tail)

    The audio format (sample width, channel count and - unless given
    explicitly - the input frame rate) is locked in by the first chunk
    passed to :meth:`process`.  Later chunks with a different format raise
    :class:`ValueError`.

    Only whole sample frames may be passed; an :class:`AudioSegment` always
    satisfies this.  The total number of output frames for a finished stream
    of ``N`` input frames is ``round(N * output_rate / input_rate)``,
    regardless of chunking.

    The processor keeps only the short FIR tail it still needs (a few dozen
    frames); it never buffers the whole stream and never mutates its input.
    """

    #: Kaiser stop-band attenuation of the anti-aliasing filter, in dB.
    STOPBAND_ATTENUATION_DB = 60.0

    #: Transition-band width as a fraction of the cut-off frequency.
    TRANSITION_RELATIVE_WIDTH = 0.25

    #: Outputs are computed in blocks of at most this many frames.
    _BLOCK = 4096

    def __init__(self, output_rate, input_rate=None):
        """
        :param output_rate: frame rate of the produced audio, in Hz.
        :param input_rate: expected input frame rate, in Hz.  If omitted,
            the frame rate of the first chunk is used.
        """
        output_rate = int(output_rate)
        if output_rate <= 0:
            raise ValueError("output_rate must be a positive integer")
        if input_rate is not None:
            input_rate = int(input_rate)
            if input_rate <= 0:
                raise ValueError("input_rate must be a positive integer")
        self._out_rate = output_rate
        self._in_rate = input_rate

        self._format = None       # (sample_width, channels) of first chunk
        self._finished = False

        # filter state (built lazily once the input rate is known)
        self._half_len = None
        self._beta = None
        self._fc = None
        self._offsets = None
        self._step = None         # input frames advanced per output frame

        # streaming state
        self._buf = None          # float64 (frames, channels) tail buffer
        self._buf_start = 0       # global input index of _buf[0]
        self._in_count = 0        # total input frames received
        self._next_out = 0        # next output frame index to emit
        self._total_out = None    # known once flush() is called

    # ------------------------------------------------------------------
    # public properties
    # ------------------------------------------------------------------

    @property
    def output_rate(self):
        """Frame rate of the produced audio, in Hz."""
        return self._out_rate

    @property
    def input_rate(self):
        """Input frame rate in Hz, or None until the first chunk arrives."""
        return self._in_rate

    @property
    def format(self):
        """(sample_width, channels) of the stream, or None before any input."""
        return self._format

    @property
    def latency(self):
        """
        Algorithmic latency in *input* frames: half the FIR filter length.

        Output frame ``n`` is emitted once input up to roughly
        ``n * input_rate / output_rate + latency`` frames has been received.
        The first and last ``latency`` input frames worth of output are
        transition regions (the filter window hangs over the zero-padded
        edges of the stream).
        """
        if self._half_len is None:
            raise RuntimeError("latency is unknown until the first chunk is processed")
        return self._half_len

    @property
    def finished(self):
        """True after :meth:`flush` has been called."""
        return self._finished

    # ------------------------------------------------------------------
    # public API
    # ------------------------------------------------------------------

    def process(self, chunk):
        """
        Resample one chunk of input audio.

        :param chunk: an :class:`AudioSegment` containing whole sample
            frames.  Its format defines the stream format on the first
            call; later chunks must match it exactly.
        :return: an :class:`AudioSegment` at ``output_rate`` with the
            output frames that became available (possibly zero frames).
        """
        if self._finished:
            raise RuntimeError("input stream is already finished (flush() was called)")
        if not isinstance(chunk, AudioSegment):
            raise TypeError("chunk must be an AudioSegment")

        fmt = (chunk.sample_width, chunk.channels)
        if self._format is None:
            if self._in_rate is None:
                self._in_rate = int(chunk.frame_rate)
            elif int(chunk.frame_rate) != self._in_rate:
                raise ValueError(
                    "input frame rate mismatch: resampler was created with "
                    "input_rate=%d but the first chunk has frame_rate=%d"
                    % (self._in_rate, chunk.frame_rate))
            self._format = fmt
            self._init_filter()
            self._buf = np.zeros((0, chunk.channels), dtype=np.float64)
        else:
            if fmt != self._format or int(chunk.frame_rate) != self._in_rate:
                raise ValueError(
                    "chunk format mismatch: stream format is sample_width=%d, "
                    "channels=%d, frame_rate=%d but got sample_width=%d, "
                    "channels=%d, frame_rate=%d"
                    % (self._format[0], self._format[1], self._in_rate,
                       chunk.sample_width, chunk.channels, chunk.frame_rate))

        frames = self._decode(chunk)
        self._buf = np.concatenate([self._buf, frames], axis=0)
        self._in_count += frames.shape[0]
        return self._emit(final=False)

    def flush(self):
        """
        Signal the end of input and drain the remaining output frames.

        Input samples past the end of the stream are treated as zeros, so
        the tail fades out over ``latency`` input frames.  Calling
        :meth:`flush` again is allowed but returns an empty segment - the
        tail is never emitted twice.  Calling :meth:`process` afterwards
        raises :class:`RuntimeError`.
        """
        if self._finished:
            return self._make_segment(np.zeros((0, 0)))
        self._finished = True
        if self._format is None:
            return self._make_segment(np.zeros((0, 0)))
        self._total_out = int(
            Fraction(self._in_count * self._out_rate, self._in_rate)
            + Fraction(1, 2))
        return self._emit(final=True)

    # ------------------------------------------------------------------
    # internals
    # ------------------------------------------------------------------

    def _init_filter(self):
        ratio = self._out_rate / self._in_rate
        attenuation = self.STOPBAND_ATTENUATION_DB
        self._beta = 0.1102 * (attenuation - 8.7)
        self._fc = 0.5 * min(1.0, ratio)
        transition = self._fc * self.TRANSITION_RELATIVE_WIDTH
        # Kaiser FIR length estimate (in input-frame units)
        n_taps = (attenuation - 8.0) / (2.285 * 2.0 * math.pi * transition)
        self._half_len = max(4, int(math.ceil(n_taps / 2.0)) + 1)
        self._offsets = np.arange(-self._half_len, self._half_len + 1)
        self._step = self._in_rate / self._out_rate

    def _kernel(self, t):
        """Windowed-sinc kernel evaluated at offsets t (input-frame units)."""
        h = np.zeros_like(t)
        mask = np.abs(t) <= self._half_len
        tm = t[mask]
        window = np.i0(self._beta * np.sqrt(
            np.maximum(0.0, 1.0 - (tm / self._half_len) ** 2))) / np.i0(self._beta)
        h[mask] = 2.0 * self._fc * np.sinc(2.0 * self._fc * tm) * window
        return h

    def _decode(self, chunk):
        dtype = _SAMPLE_WIDTH_TO_DTYPE.get(chunk.sample_width)
        if dtype is None:
            raise ValueError("unsupported sample_width: %r" % (chunk.sample_width,))
        # np.frombuffer never writes into the (immutable) source bytes and
        # astype() copies, so the input chunk is left untouched.
        samples = np.frombuffer(chunk.raw_data, dtype=dtype)
        return samples.reshape(-1, chunk.channels).astype(np.float64)

    def _emit(self, final):
        if final:
            n_last = self._total_out - 1
        elif self._in_count == 0:
            n_last = -1
        else:
            # emit outputs whose whole kernel window is covered by input
            n_last = int(math.floor(
                (self._in_count - 1 - self._half_len) / self._step))

        blocks = []
        while self._next_out <= n_last:
            stop = min(n_last + 1, self._next_out + self._BLOCK)
            ns = np.arange(self._next_out, stop)
            blocks.append(self._render(ns))
            self._next_out = stop

        # drop input frames no future output can still need
        min_needed = int(math.floor(self._next_out * self._step)) - self._half_len
        drop = min(max(min_needed - self._buf_start, 0), len(self._buf))
        if drop:
            self._buf = self._buf[drop:]
            self._buf_start += drop

        if blocks:
            return self._make_segment(np.concatenate(blocks, axis=0))
        return self._make_segment(np.zeros((0, self._format[1])))

    def _render(self, ns):
        positions = ns * self._step
        centers = np.floor(positions).astype(np.int64)
        kmat = centers[:, None] + self._offsets[None, :]
        h = self._kernel(positions[:, None] - kmat)
        norm = h.sum(axis=1)

        valid = (kmat >= 0) & (kmat < self._in_count)
        if len(self._buf):
            idx = np.clip(kmat - self._buf_start, 0, len(self._buf) - 1)
            gathered = self._buf[idx] * valid[:, :, None]
            out = (gathered * h[:, :, None]).sum(axis=1) / norm[:, None]
        else:
            out = np.zeros((len(ns), self._format[1]))
        return out

    def _make_segment(self, frames):
        if self._format is None:
            return AudioSegment(data=b"", sample_width=1, channels=1,
                                frame_rate=self._out_rate)
        sample_width, channels = self._format
        dtype = _SAMPLE_WIDTH_TO_DTYPE[sample_width]
        info = np.iinfo(dtype)
        data = np.clip(np.rint(frames), info.min, info.max)
        return AudioSegment(data=data.astype(dtype).tobytes(),
                            sample_width=sample_width,
                            channels=channels,
                            frame_rate=self._out_rate)
