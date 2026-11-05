# Changelog

All notable changes to Desktop Audio Recorder are listed here. The format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and the project uses [Semantic Versioning](https://semver.org/).

## [1.0.0] - 2026-11-04

The first public release.

### Added
- **Recording** of the Windows default playback device through WASAPI loopback, without a virtual cable or "Stereo Mix". Surround devices are mixed down to stereo; silence is recorded as silence, so the length matches the wall clock.
- **Following the default device**: when Windows switches the default output mid-recording, the recording moves to the new device in the same take (Recording → Follow the default playback device, on by default). A device with another sample rate is resampled live. An unplugged device moves the recording to the next default, or stops it and keeps what was captured.
- Pause and resume within a take, an L/R peak meter with peak hold and a CLIP lamp, and warnings when the device changes, audio stops arriving, or writing fails.
- **Playback** with a waveform view: zoom, scroll, seek, selection, and looped playback of the selection.
- **Editing**: delete, crop, silence, insert silence, fade in and out, gain and normalize, with undo and redo. Undo data beyond 500 MB moves to temporary files (up to 10 GB), so long clips keep several undo steps.
- **Noise reduction** for voice recordings (spectral gating with noisereduce, plus a high-pass filter), with an A/B preview and a noise profile taken from a selection.
- **Export** to WAV (16-bit, 24-bit or 32-bit float; RF64 above 4 GB) and MP3 (128–320 kbps CBR with a LAME gapless tag), with TPDF dither for 16-bit output. Exports run in the background and can be cancelled.
- **File → Open** for WAV, MP3, FLAC, OGG and AIFF files, by drag and drop, or from the command line ("Open with").
- Save / Discard / Cancel for unsaved clips on a new recording, on opening a file, on closing, and when Windows signs out; recovery of the recording after a crash.
- Light and dark mode, following the Windows setting while the app runs.
- **Help** menu: About (version, license, project page, third-party notices), About Qt, Open log folder, and Report a problem.
- A rotating log in `%LOCALAPPDATA%\DesktopAudioRecorder\logs` and an error dialog for unexpected errors.
- A portable Windows build (a zip; nothing is installed) with an icon, a version resource, `LICENSE`, `THIRD-PARTY-NOTICES.txt`, and a `--smoke-test` self-check.

[1.0.0]: https://github.com/bkazmierzak/desktop-audio-recorder/releases/tag/v1.0.0
