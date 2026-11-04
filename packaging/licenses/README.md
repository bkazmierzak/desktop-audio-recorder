# License texts for THIRD-PARTY-NOTICES.txt

`packaging/build.py` copies these into the notices it writes next to the exe. Most license
texts come from the bundled distributions' own metadata; these are the ones that don't:

| File | For | Taken from |
|---|---|---|
| `GPL-3.0.txt` | The GPL that LGPL-3.0 builds on (Qt, PySide6, lameenc) | numpy 2.5.4's `LICENSE.txt` (the GCC runtime section) |
| `LGPL-3.0.txt` | Qt, PySide6, shiboken6, lameenc | lameenc 1.9.0's `LICENSE` |
| `LGPL-2.1.txt` | libsndfile, mpg123 | soundfile 0.14.0's `_soundfile_data/COPYING` |
| `Apache-2.0.txt` | OpenSSL (in the Python runtime), PyAudioWPatch | packaging 26.3's `LICENSE.APACHE` |
| `PortAudio.txt` | PortAudio (in PyAudioWPatch and sounddevice) | the V19 license page, portaudio.com/license.html (the repository's `LICENSE.txt` is a header template) |
| `python-runtime.txt` | libffi, Expat, libmpdec, zlib, XZ Utils (inside the Python runtime) | the license files of the versions python-build-standalone 20261013 builds into Python 3.12.15 for Windows: libffi 3.4.2 (`python/cpython-source-deps`, the commit that build pins), Expat 2.8.5 and libmpdec 2.5.1 (CPython 3.12.15's `Modules/expat/COPYING` and `Doc/license.rst`), zlib 1.3.2 (`zlib.h`), XZ Utils 5.8.4 (`COPYING.0BSD`). `tests/test_packaging.py` fails if the running Python reports other versions. |

The texts are unchanged apart from line endings.
