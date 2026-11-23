# Reproducible initial environment

Python 3.12.13. Install requirements-userfeedback.txt in a separate environment; run from the repository root.
No feature implementation, task text or evaluation scripts are included.
Pydub uses Python 3.12 audioop. WAV/PCM checks do not require an encoder. FFmpeg for optional compressed media is provided by imageio-ffmpeg. The existing assertEquals alias test is incompatible with Python 3.12; its original failure is preserved externally.
