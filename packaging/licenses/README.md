# License texts for THIRD-PARTY-NOTICES.txt

`packaging/build.py` copies these into the notices it writes next to the exe. Most license
texts come from the bundled distributions' own metadata; these are the ones that don't:

| File | For | Taken from |
|---|---|---|
| `GPL-3.0.txt` | The GPL that LGPL-3.0 builds on (Qt, PySide6, lameenc) | numpy 2.5.4's `LICENSE.txt` (the GCC runtime section) |
| `LGPL-3.0.txt` | Qt, PySide6, shiboken6, lameenc | lameenc 1.9.0's `LICENSE` |
| `LGPL-2.1.txt` | libsndfile, mpg123 | soundfile 0.14.0's `_soundfile_data/COPYING` |
| `Apache-2.0.txt` | OpenSSL (in the Python runtime), PyAudioWPatch | packaging 26.3's `LICENSE.APACHE` |
| `PortAudio.txt` | PortAudio (in PyAudioWPatch and sounddevice) | the PortAudio license (portaudio.com) |
| `python-runtime.txt` | libffi, Expat, mpdecimal, zlib, XZ Utils (inside the Python runtime) | their upstream license notices |

The texts are unchanged apart from line endings.
