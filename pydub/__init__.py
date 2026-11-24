from .audio_segment import AudioSegment, TimelineTrack


def mix_timeline(tracks, frame_rate=None, sample_width=None, channels=None,
                 **kwargs):
    """Module-level shortcut for :meth:`AudioSegment.mix_timeline`."""
    return AudioSegment.mix_timeline(
        tracks, frame_rate=frame_rate, sample_width=sample_width,
        channels=channels, **kwargs)


timeline_mix = mix_timeline
mix = mix_timeline
