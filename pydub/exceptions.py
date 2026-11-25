class PydubException(Exception):
    """
    Base class for any Pydub exception
    """


class TooManyMissingFrames(PydubException):
    pass


class InvalidDuration(PydubException):
    pass


class InvalidTag(PydubException):
    pass


class InvalidID3TagVersion(PydubException):
    pass


class CouldntDecodeError(PydubException):
    pass


class CouldntEncodeError(PydubException):
    pass


class MissingAudioParameter(PydubException):
    pass


class InvalidAudioFormat(PydubException, ValueError):
    """
    The audio format is invalid, or changed while a stream was active.
    """


class InvalidLimiterSettings(PydubException, ValueError):
    """
    A limiter setting is invalid.
    """
