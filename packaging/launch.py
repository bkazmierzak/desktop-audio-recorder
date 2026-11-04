"""PyInstaller entry point (the console script ``desktop-audio-recorder`` in a frozen build)."""

import multiprocessing

from desktop_audio_recorder.app import main

if __name__ == "__main__":
    multiprocessing.freeze_support()  # in case joblib ever starts worker processes
    raise SystemExit(main())
