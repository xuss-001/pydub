import sys
import math
import array
import numbers
from .utils import (
    db_to_float,
    ratio_to_db,
    register_pydub_effect,
    make_chunks,
    audioop,
    get_min_max_value
)
from .silence import split_on_silence
from .exceptions import TooManyMissingFrames, InvalidDuration

if sys.version_info >= (3, 0):
    xrange = range


@register_pydub_effect
def apply_mono_filter_to_each_channel(seg, filter_fn):
    n_channels = seg.channels

    channel_segs = seg.split_to_mono()
    channel_segs = [filter_fn(channel_seg) for channel_seg in channel_segs]

    out_data = seg.get_array_of_samples()
    for channel_i, channel_seg in enumerate(channel_segs):
        for sample_i, sample in enumerate(channel_seg.get_array_of_samples()):
            index = (sample_i * n_channels) + channel_i
            out_data[index] = sample

    return seg._spawn(out_data)


@register_pydub_effect
def normalize(seg, headroom=0.1):
    """
    headroom is how close to the maximum volume to boost the signal up to (specified in dB)
    """
    peak_sample_val = seg.max
    
    # if the max is 0, this audio segment is silent, and can't be normalized
    if peak_sample_val == 0:
        return seg
    
    target_peak = seg.max_possible_amplitude * db_to_float(-headroom)

    needed_boost = ratio_to_db(target_peak / peak_sample_val)
    return seg.apply_gain(needed_boost)


@register_pydub_effect
def speedup(seg, playback_speed=1.5, chunk_size=150, crossfade=25):
    # we will keep audio in 150ms chunks since one waveform at 20Hz is 50ms long
    # (20 Hz is the lowest frequency audible to humans)

    # portion of AUDIO TO KEEP. if playback speed is 1.25 we keep 80% (0.8) and
    # discard 20% (0.2)
    atk = 1.0 / playback_speed

    if playback_speed < 2.0:
        # throwing out more than half the audio - keep 50ms chunks
        ms_to_remove_per_chunk = int(chunk_size * (1 - atk) / atk)
    else:
        # throwing out less than half the audio - throw out 50ms chunks
        ms_to_remove_per_chunk = int(chunk_size)
        chunk_size = int(atk * chunk_size / (1 - atk))

    # the crossfade cannot be longer than the amount of audio we're removing
    crossfade = min(crossfade, ms_to_remove_per_chunk - 1)

    # DEBUG
    #print("chunk: {0}, rm: {1}".format(chunk_size, ms_to_remove_per_chunk))

    chunks = make_chunks(seg, chunk_size + ms_to_remove_per_chunk)
    if len(chunks) < 2:
        raise Exception("Could not speed up AudioSegment, it was too short {2:0.2f}s for the current settings:\n{0}ms chunks at {1:0.1f}x speedup".format(
            chunk_size, playback_speed, seg.duration_seconds))

    # we'll actually truncate a bit less than we calculated to make up for the
    # crossfade between chunks
    ms_to_remove_per_chunk -= crossfade

    # we don't want to truncate the last chunk since it is not guaranteed to be
    # the full chunk length
    last_chunk = chunks[-1]
    chunks = [chunk[:-ms_to_remove_per_chunk] for chunk in chunks[:-1]]

    out = chunks[0]
    for chunk in chunks[1:]:
        out = out.append(chunk, crossfade=crossfade)

    out += last_chunk
    return out


def _wsola_time_stretch(samples, speed, n_out, frame_rate, window_ms=40.0):
    """
    WSOLA (Waveform Similarity Overlap-Add) time-scale modification.

    samples: float64 numpy array of shape (n_in, channels), normalized to
    roughly [-1.0, 1.0]. All channels are stretched using a single shared
    time mapping (the segment offset search is done on the channel mix) so
    inter-channel phase relations are preserved.

    Returns a float64 array of shape (n_out, channels).
    """
    import numpy as np

    n_in, channels = samples.shape

    win_len = int(round(frame_rate * window_ms / 1000.0))
    win_len = max(1, min(win_len, n_in))
    hop = max(1, win_len // 4)
    max_delta = hop

    window = np.hanning(win_len)
    if win_len >= 2 and window[1] == 0.0:
        # extremely short windows: np.hanning degenerates, fall back to ones
        window = np.ones(win_len)

    pad = win_len + max_delta + 2 * hop + 2
    if n_in >= 2:
        padded = np.pad(samples, ((pad, pad), (0, 0)), mode='reflect')
    else:
        padded = np.pad(samples, ((pad, pad), (0, 0)), mode='constant')
    mono = padded.mean(axis=1)

    out = np.zeros((n_out + win_len, channels), dtype=np.float64)
    out_mono = np.zeros(n_out + win_len, dtype=np.float64)
    weight = np.zeros(n_out + win_len, dtype=np.float64)

    ref_len = win_len - hop
    deltas = np.arange(-max_delta, max_delta + 1)

    k = 0
    while k * hop < n_out:
        o = k * hop
        # input position (in original, unpadded coordinates) that
        # corresponds to output position o
        nominal = o * speed + pad - win_len // 2
        start0 = int(round(nominal))

        delta = 0
        if k > 0 and ref_len > 0 and max_delta > 0:
            ref = out_mono[o:o + ref_len]
            ref_norm = np.sqrt(np.dot(ref, ref))
            if ref_norm > 1e-12:
                base = start0 - max_delta
                cand = np.lib.stride_tricks.as_strided(
                    mono[base:base + len(deltas) + ref_len - 1],
                    shape=(len(deltas), ref_len),
                    strides=(mono.strides[0], mono.strides[0]))
                corrs = cand.dot(ref)
                cand_norms = np.sqrt((cand ** 2).sum(axis=1)) * ref_norm
                scores = corrs / (cand_norms + 1e-20)
                best = np.flatnonzero(scores >= scores.max() - 1e-12)
                # prefer the offset closest to zero on ties
                delta = deltas[best[np.argmin(np.abs(deltas[best]))]]

        start = start0 + delta
        segment = padded[start:start + win_len]
        out[o:o + win_len] += segment * window[:, None]
        out_mono[o:o + win_len] += segment.mean(axis=1) * window
        weight[o:o + win_len] += window
        k += 1

    out = out[:n_out]
    weight = weight[:n_out]
    nonzero = weight > 1e-8
    out[nonzero] /= weight[nonzero, None]
    out[~nonzero] = 0.0
    return out


@register_pydub_effect
def time_stretch(seg, playback_speed=1.0, window_ms=40.0):
    """
    Change the duration of the audio without changing its pitch, using
    WSOLA (Waveform Similarity Overlap-Add).

    playback_speed values greater than 1.0 shorten the audio, values
    between 0.0 and 1.0 lengthen it. The output keeps the frame rate,
    sample width and channel count of the input, and its frame count is
    round(input_frames / playback_speed). All channels share a single
    time mapping so stereo phase relations are preserved. The original
    AudioSegment is not modified; a playback_speed of exactly 1.0
    returns the original samples unchanged.

    window_ms is the analysis window length in milliseconds (default 40).
    """
    import numpy as np

    if isinstance(playback_speed, bool) or not isinstance(
            playback_speed, numbers.Real):
        raise TypeError(
            "playback_speed must be a real number, got %r" % (playback_speed,))
    speed = float(playback_speed)
    if not math.isfinite(speed) or speed <= 0.0:
        raise ValueError(
            "playback_speed must be a positive, finite number, got %r"
            % (playback_speed,))

    frame_width = seg.sample_width * seg.channels
    n_in = len(seg.raw_data) // frame_width
    n_out = int(math.floor(n_in / speed + 0.5))

    if n_in == 0 or n_out == 0:
        return seg._spawn(b'')

    if speed == 1.0:
        return seg._spawn(seg.raw_data)

    int_dtypes = {1: np.int8, 2: np.int16, 4: np.int32}
    dtype = int_dtypes[seg.sample_width]
    full_scale = float(2 ** (8 * seg.sample_width - 1))

    samples = np.frombuffer(seg.raw_data, dtype=dtype)
    samples = samples.astype(np.float64).reshape(n_in, seg.channels)
    samples /= full_scale

    stretched = _wsola_time_stretch(
        samples, speed, n_out, seg.frame_rate, window_ms)

    out = np.round(stretched * full_scale)
    np.clip(out, -full_scale, full_scale - 1.0, out=out)
    return seg._spawn(out.astype(dtype).tobytes())


@register_pydub_effect
def strip_silence(seg, silence_len=1000, silence_thresh=-16, padding=100):
    if padding > silence_len:
        raise InvalidDuration("padding cannot be longer than silence_len")

    chunks = split_on_silence(seg, silence_len, silence_thresh, padding)
    crossfade = padding / 2

    if not len(chunks):
        return seg[0:0]

    seg = chunks[0]
    for chunk in chunks[1:]:
        seg = seg.append(chunk, crossfade=crossfade)

    return seg


@register_pydub_effect
def compress_dynamic_range(seg, threshold=-20.0, ratio=4.0, attack=5.0, release=50.0):
    """
    Keyword Arguments:
        
        threshold - default: -20.0
            Threshold in dBFS. default of -20.0 means -20dB relative to the
            maximum possible volume. 0dBFS is the maximum possible value so
            all values for this argument sould be negative.

        ratio - default: 4.0
            Compression ratio. Audio louder than the threshold will be 
            reduced to 1/ratio the volume. A ratio of 4.0 is equivalent to
            a setting of 4:1 in a pro-audio compressor like the Waves C1.
        
        attack - default: 5.0
            Attack in milliseconds. How long it should take for the compressor
            to kick in once the audio has exceeded the threshold.

        release - default: 50.0
            Release in milliseconds. How long it should take for the compressor
            to stop compressing after the audio has falled below the threshold.

    
    For an overview of Dynamic Range Compression, and more detailed explanation
    of the related terminology, see: 

        http://en.wikipedia.org/wiki/Dynamic_range_compression
    """

    thresh_rms = seg.max_possible_amplitude * db_to_float(threshold)
    
    look_frames = int(seg.frame_count(ms=attack))
    def rms_at(frame_i):
        return seg.get_sample_slice(frame_i - look_frames, frame_i).rms
    def db_over_threshold(rms):
        if rms == 0: return 0.0
        db = ratio_to_db(rms / thresh_rms)
        return max(db, 0)

    output = []

    # amount to reduce the volume of the audio by (in dB)
    attenuation = 0.0
    
    attack_frames = seg.frame_count(ms=attack)
    release_frames = seg.frame_count(ms=release)
    for i in xrange(int(seg.frame_count())):
        rms_now = rms_at(i)
        
        # with a ratio of 4.0 this means the volume will exceed the threshold by
        # 1/4 the amount (of dB) that it would otherwise
        max_attenuation = (1 - (1.0 / ratio)) * db_over_threshold(rms_now)
        
        attenuation_inc = max_attenuation / attack_frames
        attenuation_dec = max_attenuation / release_frames
        
        if rms_now > thresh_rms and attenuation <= max_attenuation:
            attenuation += attenuation_inc
            attenuation = min(attenuation, max_attenuation)
        else:
            attenuation -= attenuation_dec
            attenuation = max(attenuation, 0)
        
        frame = seg.get_frame(i)
        if attenuation != 0.0:
            frame = audioop.mul(frame,
                                seg.sample_width,
                                db_to_float(-attenuation))
        
        output.append(frame)
    
    return seg._spawn(data=b''.join(output))


# Invert the phase of the signal.

@register_pydub_effect

def invert_phase(seg, channels=(1, 1)):
    """
    channels- specifies which channel (left or right) to reverse the phase of.
    Note that mono AudioSegments will become stereo.
    """
    if channels == (1, 1):
        inverted = audioop.mul(seg._data, seg.sample_width, -1.0)  
        return seg._spawn(data=inverted)
    
    else:
        if seg.channels == 2:
            left, right = seg.split_to_mono()
        else:
            raise Exception("Can't implicitly convert an AudioSegment with " + str(seg.channels) + " channels to stereo.")
            
        if channels == (1, 0):    
            left = left.invert_phase()
        else:
            right = right.invert_phase()
        
        return seg.from_mono_audiosegments(left, right)
        


# High and low pass filters based on implementation found on Stack Overflow:
#   http://stackoverflow.com/questions/13882038/implementing-simple-high-and-low-pass-filters-in-c

@register_pydub_effect
def low_pass_filter(seg, cutoff):
    """
        cutoff - Frequency (in Hz) where higher frequency signal will begin to
            be reduced by 6dB per octave (doubling in frequency) above this point
    """
    RC = 1.0 / (cutoff * 2 * math.pi)
    dt = 1.0 / seg.frame_rate

    alpha = dt / (RC + dt)
    
    original = seg.get_array_of_samples()
    filteredArray = array.array(seg.array_type, original)
    
    frame_count = int(seg.frame_count())

    last_val = [0] * seg.channels
    for i in range(seg.channels):
        last_val[i] = filteredArray[i] = original[i]

    for i in range(1, frame_count):
        for j in range(seg.channels):
            offset = (i * seg.channels) + j
            last_val[j] = last_val[j] + (alpha * (original[offset] - last_val[j]))
            filteredArray[offset] = int(last_val[j])

    return seg._spawn(data=filteredArray)


@register_pydub_effect
def high_pass_filter(seg, cutoff):
    """
        cutoff - Frequency (in Hz) where lower frequency signal will begin to
            be reduced by 6dB per octave (doubling in frequency) below this point
    """
    RC = 1.0 / (cutoff * 2 * math.pi)
    dt = 1.0 / seg.frame_rate

    alpha = RC / (RC + dt)

    minval, maxval = get_min_max_value(seg.sample_width * 8)
    
    original = seg.get_array_of_samples()
    filteredArray = array.array(seg.array_type, original)
    
    frame_count = int(seg.frame_count())

    last_val = [0] * seg.channels
    for i in range(seg.channels):
        last_val[i] = filteredArray[i] = original[i]

    for i in range(1, frame_count):
        for j in range(seg.channels):
            offset = (i * seg.channels) + j
            offset_minus_1 = ((i-1) * seg.channels) + j

            last_val[j] = alpha * (last_val[j] + original[offset] - original[offset_minus_1])
            filteredArray[offset] = int(min(max(last_val[j], minval), maxval))

    return seg._spawn(data=filteredArray)
    
    
@register_pydub_effect
def pan(seg, pan_amount):
    """
    pan_amount should be between -1.0 (100% left) and +1.0 (100% right)
    
    When pan_amount == 0.0 the left/right balance is not changed.
    
    Panning does not alter the *perceived* loundness, but since loudness
    is decreasing on one side, the other side needs to get louder to
    compensate. When panned hard left, the left channel will be 3dB louder.
    """
    if not -1.0 <= pan_amount <= 1.0:
        raise ValueError("pan_amount should be between -1.0 (100% left) and +1.0 (100% right)")
    
    max_boost_db = ratio_to_db(2.0)
    boost_db = abs(pan_amount) * max_boost_db
    
    boost_factor = db_to_float(boost_db)
    reduce_factor = db_to_float(max_boost_db) - boost_factor
    
    reduce_db = ratio_to_db(reduce_factor)
    
    # Cut boost in half (max boost== 3dB) - in reality 2 speakers
    #   do not sum to a full 6 dB.
    boost_db = boost_db / 2.0
    
    if pan_amount < 0:
        return seg.apply_gain_stereo(boost_db, reduce_db)
    else:
        return seg.apply_gain_stereo(reduce_db, boost_db)
        
    
@register_pydub_effect
def apply_gain_stereo(seg, left_gain=0.0, right_gain=0.0):
    """
    left_gain - amount of gain to apply to the left channel (in dB)
    right_gain - amount of gain to apply to the right channel (in dB)
    
    note: mono audio segments will be converted to stereo
    """
    if seg.channels == 1:
        left = right = seg
    elif seg.channels == 2:
        left, right = seg.split_to_mono()
    
    l_mult_factor = db_to_float(left_gain)
    r_mult_factor = db_to_float(right_gain)
    
    left_data = audioop.mul(left._data, left.sample_width, l_mult_factor)
    left_data = audioop.tostereo(left_data, left.sample_width, 1, 0)
    
    right_data = audioop.mul(right._data, right.sample_width, r_mult_factor)
    right_data = audioop.tostereo(right_data, right.sample_width, 0, 1)
    
    output = audioop.add(left_data, right_data, seg.sample_width)
    
    return seg._spawn(data=output,
                overrides={'channels': 2,
                           'frame_width': 2 * seg.sample_width})
