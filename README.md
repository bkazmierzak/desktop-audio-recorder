# Desktop Audio Recorder

A Windows 11 desktop app that records whatever is playing on your PC from the **current default playback device** using WASAPI loopback. You can play back and edit the clip, optionally clean up background noise, and save it as **.wav** or **.mp3**.

No virtual audio cable, "Stereo Mix" or ffmpeg is needed: loopback capture comes from PyAudioWPatch, and MP3 encoding from the bundled LAME library (`lameenc`).

## Requirements
- Windows 10 or 11 (64-bit).
- To run from source: [uv](https://docs.astral.sh/uv/). uv installs Python 3.12 itself, so you don't need a system Python.
- To run the packaged build: nothing else (Python and all libraries are inside the folder).

## Setup and run from source
```powershell
uv sync                              # creates .venv with Python 3.12 and every dependency
uv run desktop-audio-recorder        # or: uv run python -m desktop_audio_recorder
```

## Build the Windows app (.exe)
```powershell
uv run python packaging/build.py              # build, self-test, zip
uv run python packaging/build.py --no-smoke   # build and zip without the self-test (no sound)
uv run python packaging/build.py --zip-only   # zip an existing build
```
This builds the folder `dist\DesktopAudioRecorder\` (about 200 MB) with `DesktopAudioRecorder.exe` inside, runs its self-test (below), and packs the folder plus this README into `dist\DesktopAudioRecorder-<version>-win64.zip`. To install, unzip anywhere and run `DesktopAudioRecorder.exe`; nothing else is installed. Keep the **whole folder**: the exe doesn't work on its own. It is a windowed app (no console), so messages go to the log file (see [Logs](#logs-and-errors)). The exe has the app's icon and a version resource (Properties → Details). `uv run pyinstaller packaging/desktop_audio_recorder.spec --noconfirm` builds without testing or zipping. The icon is drawn by `packaging/make_icon.py`.

To check a build on a machine, run its self-test. It plays a quiet 1 kHz tone (−26 dBFS) on the default playback device for about 3 s while it records, so expect to hear it:
```powershell
dist\DesktopAudioRecorder\DesktopAudioRecorder.exe --smoke-test C:\Temp\smoke
```
It records 2 s, plays the clip for a second, exports `smoke.wav` and `smoke.mp3`, reads both back, checks the log, saves `window.png` and writes `report.json`. Its exit code is 0 when every check passed. Its settings go to `smoke\settings.ini`; your own settings aren't touched.

## Using it

### Recording
- **Record** captures whatever is playing on the default playback device, shown as "Recording from: …".
  - The line follows Windows. It updates as soon as you change the default output in Settings → System → Sound, and **Refresh** or switching back to the window re-detects it as well.
  - Surround devices (for example 7.1 headsets) are mixed down to stereo.
- **Pause/Resume** skips audio without ending the take. **Stop** finishes it.
- The L/R meter shows peak level in dBFS. **CLIP** lights when the signal goes over full scale (click the meter to clear it). During playback the meter shows the clip being played.
- Silence is recorded as silence: the recording keeps the wall-clock length even when nothing is playing.

While a recording runs, the app watches the audio device:
- **The default playback device changes** (for example, you switch to headphones). The recording moves to the new device and carries on in the same take; the status bar says "Now recording from …". The switch takes about 0.15 s (up to 0.3 s the first time), which is recorded as silence, so the timeline stays true. A device with another sample rate is converted to the recording's rate. This is **Recording → Follow the default playback device**, on by default. With it off, a yellow banner says the default changed and the recording carries on from the device it started on.
- **The recorded device is unplugged or disabled.** If Windows picks another default device within 1.5 s, the recording moves there (when following is on). Otherwise the recording stops by itself and keeps everything captured until then; the status bar says why.
- **The new device can't be opened.** The recording stays on the device it was using, and a banner says so. If neither device can be opened, the recording stops and keeps what it has.
- **No audio arrives for 2 s or more** (the capture stream died). A warning appears and the gap is filled with silence.
- **Writing the recording fails** (for example, the disk is full). The recording stops and keeps what was written.

### Opening files
**File → Open…** (Ctrl+O) opens an existing audio file to play, edit and export, instead of a recording. You can also drop a file on the window, or name one on the command line (`DesktopAudioRecorder.exe song.mp3`, which also makes "Open with" work).
- WAV, MP3, FLAC, OGG/Vorbis and AIFF files can be opened. Mono and surround files are mixed to stereo, and the file's sample rate is kept.
- The file loads in the background and replaces the current clip. An unsaved clip gets the usual **Save… / Discard / Cancel** question first. If the file can't be read, the previous clip stays.
- An opened file counts as saved until you edit it. **Export…** suggests a new name next to it (`<name> (edited).wav`), so the original is never overwritten unless you choose it in the dialog. The app never deletes an opened file.

### Playback
After **Stop** (or opening a file), the clip's waveform appears and you can play it on the current default output:

| Action | How |
|---|---|
| Play / pause | **Play** button or **Space** |
| Stop playback (back to the start of the selection or clip) | **Stop** in the playback bar |
| Go to the start | **Home** |
| Seek | Click the waveform, or drag the orange playhead |
| Select a range | Drag across the waveform; drag the selection's edges to adjust it. Play then plays only the selection |
| Loop | **Loop** repeats the selection (or the whole clip); the setting is remembered |
| Zoom | Mouse wheel (around the pointer) |
| Scroll | Shift + wheel, the scroll bar, or drag with the middle/right button |

Recording is disabled while the clip is playing. The dashed lines mark full scale (±1.0); the Windows mix can go above it, and the waveform shows those overs (playback clips them).

### Editing
The edit toolbar above the waveform changes the clip. The waveform and playback update right away, and the zoom and scroll position stay where they were:

| Action | What it does |
|---|---|
| **Delete** (Del) | Removes the selection; the audio after it moves up |
| **Crop** | Keeps only the selection |
| **Silence** | Replaces the selection with silence |
| **Insert silence…** | Inserts the given number of seconds at the selection start (or at the cursor) |
| **Fade in / Fade out** | Linear fade across the selection |
| **Gain…** | Changes the volume of the selection, or the whole clip if nothing is selected, by a number of dB. Gain can push audio above full scale; the waveform shows it |
| **Normalize** | Scales the selection (or the whole clip) so its peak is at −1 dBFS. This also turns overs back down |
| **Noise reduction…** | Reduces background noise in the selection (or the whole clip). See below |
| **Undo / Redo** | Ctrl+Z, and Ctrl+Y or Ctrl+Shift+Z, in any order |

Editing while the clip plays pauses playback first. On long recordings, big edits (anything that changes the length, or gain over a long selection) take a fraction of a second and show "Editing…".

The undo history keeps up to about 500 MB of undo data in memory. Beyond that, older steps move to temporary files in `%TEMP%\DesktopAudioRecorder` (up to 10 GB, and never below 2 GB of free disk space), and only after that are the oldest steps dropped. The last edit can always be undone. On an hour-long recording a whole-clip edit needs 1.3 GB of undo data, so this keeps about eight such steps instead of one. Undoing a step from disk takes about 0.4 s per hour of audio. Writing the files happens in the background, but several whole-clip edits on an hour-long clip in quick succession can wait a few seconds each for the disk. The files are deleted when the clip is replaced or the app closes, and leftovers from a crash are removed at the next start.

### Noise reduction
**Noise reduction…** (for voice recordings) uses spectral gating (noisereduce). The options you choose are remembered.

- **Noise type.** *Steady* is for hiss, hum, a fan or air conditioning. It works best when you first select a stretch with only noise (no voice) and tick **Use the selection as the noise profile**; the whole clip is then cleaned. Without a profile it uses the quietest parts of the audio. *Changing* is for noise that comes and goes.
- **Strength** is how much of the noise to remove (90 % by default). Lower values sound more natural.
- **High-pass filter** (80 Hz by default) removes rumble. **Normalize** brings the peak to −1 dBFS afterwards.
- **Preview** processes the selection (up to 30 s) or the first 10 s and plays it in a loop. **Play original** and **Play processed** switch instantly at the same position, for an A/B comparison.
- **Apply** processes in the background with a progress bar (about 1.6 s per minute of stereo audio, half that when both channels are identical). **Cancel** leaves the clip unchanged. The result is one undoable step. The first run takes about 1.5 s longer while the library loads.

### Export
**Export…** (above the waveform, or Ctrl+S) saves the whole clip:

- **WAV**: 16-bit PCM (default, plays everywhere), 24-bit PCM, or 32-bit float. Float keeps peaks above 0 dBFS exactly. The integer formats clip them; the dialog warns when the clip has such peaks, and the status bar says how many samples were clipped.
- **Dither** (on by default, for 16-bit WAV and MP3): very quiet triangular (TPDF) noise, about −96 dBFS, is added before rounding to 16 bits. Quiet passages and fade-outs then keep their detail as a smooth noise floor instead of turning into grainy distortion. The same clip always gives the same file. Untick it in the export dialog for plain rounding; 24-bit and float output is never dithered.
- **MP3**: constant bitrate 128, 192 (default), 256 or 320 kbps, encoded with LAME. Clips at rates other than 44.1/48 kHz are resampled to 44.1 or 48 kHz. The file carries a LAME "Info" tag, so gapless players (foobar2000, VLC, mpv, …) play exactly the clip's length.
- **Browse…** opens the Windows "Save As" dialog. The folder and the format choices are remembered for next time. New files are named "Recording YYYY-MM-DD HH-MM-SS".
- Export runs in the background with a progress bar (an hour of audio takes about 3 s as WAV, 40 s as MP3). **Cancel** writes nothing: the file appears only once it is complete, and an existing file is only replaced at the end.

### Unsaved clips, quitting and crashes
- Starting a new recording, opening a file, or closing the app, with a clip that hasn't been exported (or has been edited since) asks **Save… / Discard / Cancel**. Undoing back to the exported state counts as saved. Closing during a recording, edit or export asks whether to close anyway.
- When Windows signs out, restarts or shuts down with an unsaved clip, you get the same question. **Save…** and **Cancel** cancel the sign-out (Windows shows the app as "preventing shutdown"), so you can finish the export first.
- While you record, audio goes to a temp file in `%TEMP%\DesktopAudioRecorder`, which is deleted when you discard the clip or close the app. If the app crashes or is killed, the file stays behind. On the next start the app offers to **Recover** it (it opens as an unsaved clip), **Delete** it, or ask again next time (**Not now**). Edits made before the crash aren't recovered, only the recording.

### Settings
These are remembered in the registry under `HKEY_CURRENT_USER\Software\DesktopAudioRecorder\Desktop Audio Recorder`:
- window size and position
- export folder, format, bit depth, bitrate and dither
- the folder of the last file opened
- noise reduction options
- Loop
- Follow the default playback device

Delete that key to reset them.

### Logs and errors
- Logs are written to `%LOCALAPPDATA%\DesktopAudioRecorder\logs\app.log` (4 × 1 MB, rotating). A native crash is dumped to `crash.log` in the same folder.
- An unexpected error shows a dialog with the details and an **Open log folder** button. The same error is shown only once, at most three dialogs appear per session, and nothing is shown while the app is quitting.

## Limitations
- **System output only.** It records what Windows plays on the default playback device. It doesn't record a microphone, and it doesn't record other devices except by making them the default.
- **A device switch mid-recording leaves a short silent gap** (about 0.15 s) where the old device's audio stops and the new one's starts.
- **Length.** Recordings are kept in memory (about 1.4 GB per hour of stereo audio) and written to a float WAV while they are made. That temp file tops out at 4 GB, about 3.1 hours.
  - Exported WAVs over 4 GB (32-bit float above about 3.1 h) are written as **RF64**, which some players don't open.
  - 16-bit WAV and MP3 have plenty of room.
- **MP3 durations in Windows.** Explorer and Media Player show MP3 durations about 50 ms longer than the clip, because they count LAME's start delay and end padding. Gapless decoders play the exact length.
- **Noise reduction is classic spectral gating, not a neural model.** At high strength it can sound "watery". Steady noise with a noise profile works best.
- **One clip at a time.** The app has no multi-track editing. Opened files are read whole into memory (about 1.4 GB per hour of audio at 48 kHz, whatever the file's format).

### Not yet checked by hand
Automated tests and measurements covered the following, but they haven't been tried by a person yet:
1. **Switching the default output.** Change the default output in Settings → Sound:
   - While idle, the "Recording from:" line should update within about a second.
   - During a recording, the recording should move to the new device ("Now recording from …") and the exported clip should contain both parts. With Recording → Follow the default playback device off, the yellow warning should appear instead.
   - **Play** should then play on the new device.
2. **Unplugging the recorded device mid-recording** (for example USB or Bluetooth headphones). The recording should move to the device Windows picks next, or stop with "… was disconnected" and keep the audio.
3. **The Save As and Open dialogs.** Export… → Browse…, pick the MP3 type and type a name without an extension. The dialog should switch to MP3 and show `<name>.mp3`. Picking an existing file should ask to replace it once. File → Open… should list audio files and open the chosen one.
4. **Opening from Explorer.** Drag an MP3 onto the window, and use "Open with" → `DesktopAudioRecorder.exe` on a WAV.
5. **Listening tests.** Listen to a recording (including one that switched devices), an edited clip, a noise-reduced voice clip, and exported WAV and MP3 files. In this project's testing, audio was checked by measurement, not by ear.
6. **Signing out with an unsaved clip.** The Save/Discard/Cancel question should appear.

## Debug recorder
A headless recorder that writes the default playback device to a WAV file:
```powershell
uv run python -m desktop_audio_recorder.audio.recorder --seconds 10 out.wav
uv run python -m desktop_audio_recorder.audio.recorder --list-devices
```

## Develop
```powershell
uv run pytest                    # all tests (hardware tests play a quiet test tone on the default device)
uv run pytest -m "not hardware"  # skip tests that use the real audio device
uv run ruff check .              # lint
uv run ruff format .             # format
```
