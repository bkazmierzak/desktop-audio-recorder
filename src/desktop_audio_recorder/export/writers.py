"""Write a clip to WAV (soundfile) or MP3 (lameenc) (no Qt).

``write_wav(samples, rate, path, subtype)`` and ``write_mp3(samples, rate, path, bitrate)`` take the
app's float32 ``[frames, 2]`` samples, and ``export(samples, rate, path, settings)`` dispatches on
an ``ExportSettings``. All three take ``progress(fraction)`` and ``cancelled()`` callbacks like
``denoise()`` and raise ``Cancelled`` when asked to stop.

* **Never a broken file.** The file is written to a temporary name next to the target
  (``<name>.<random>.part``) and renamed over the target only when it is complete
  (``os.replace``). A cancel or a failure deletes the temporary file; an existing file at the
  target is untouched until the very end.
* **Chunks.** The audio is converted and written ``CHUNK_FRAMES`` at a time with reused buffers,
  so no temporary is clip-sized and progress/cancel are checked regularly. numpy, libsndfile
  (cffi) and LAME all release the GIL, so the GUI stays responsive.
* **Overs.** The app keeps samples above 1.0 until export. 32-bit float WAV keeps them exactly;
  integer formats (16/24-bit WAV, MP3's 16-bit input) clip them to full scale, and the result
  counts how many samples were clipped.
* **Quantization.** ``x * 2**(bits-1)``, rounded to nearest, clipped to the integer range: the
  inverse of how libsndfile (and most readers) turn integers back into floats, so a read-back
  differs by at most half a step (one step at exactly +1.0).
* **Dither.** 16-bit output (16-bit WAV and MP3's input) can get TPDF dither
  (``ExportSettings.dither``, on by default in the app): triangular noise of +-1 step is added
  before rounding, which turns the rounding error into a constant, signal-independent noise floor
  (about -96 dBFS RMS) instead of distortion on quiet passages and fades. The noise comes from a
  hash of each sample's position (splitmix64), not from a running generator, so chunked output is
  byte-identical to one pass and the same clip always gives the same file. 24-bit output isn't
  dithered (its rounding error is ~-144 dBFS, far below any playback chain). The raw
  ``write_wav``/``write_mp3`` functions don't dither unless asked (``dither=True``).
* **MP3** is CBR at 128-320 kbps, LAME quality 2 (``-h``, "recommended"). LAME only gets 44.1 or
  48 kHz: other rates are resampled first with ``scipy.signal.resample_poly`` (in chunks, with
  enough real context on both sides that the result equals a one-pass resample). The file starts
  with a LAME "Info" tag (frame count, size, encoder delay and padding), so players show the
  exact duration and gapless decoders (mpg123, ffmpeg, foobar2000, ...) trim LAME's start delay
  and end padding: the decoded audio has exactly the clip's frames. lameenc doesn't write that
  tag itself.
"""

from __future__ import annotations

import enum
import logging
import math
import os
import secrets
import time
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path

import numpy as np

log = logging.getLogger(__name__)

CHUNK_FRAMES = 1 << 20  # 21.8 s at 48 kHz; 8 MB of float32 stereo
MP3_CHUNK_FRAMES = 1 << 19  # LAME encodes ~20 s of audio per second: ~0.5 s per chunk
WAV_SUBTYPES = ("PCM_16", "PCM_24", "FLOAT")
DEFAULT_WAV_SUBTYPE = "PCM_16"
MP3_BITRATES = (128, 192, 256, 320)
DEFAULT_MP3_BITRATE = 192
DEFAULT_DITHER = True
MP3_RATES = (44100, 48000)
DITHER_SEED = 0x5EEDD17EA11D0001  # any fixed value: exports are reproducible
MP3_QUALITY = 2  # LAME -q 2 (same as -h): "recommended"; ~1.1 s per minute of stereo here
LAME_ENCODER_DELAY = 576  # LAME 3.100's ENCDELAY (the decoder adds its own 529 samples)
# A WAV's data chunk size is 32 bits; bigger files are written as RF64.
WAV_MAX_DATA_BYTES = 0xFFFFFFFF - 1024

ProgressCallback = Callable[[float], None]
CancelledCallback = Callable[[], bool]


class Cancelled(Exception):  # noqa: N818 (not an error: the user stopped it)
    """``cancelled()`` returned true; no file was written (the target is unchanged)."""


class ExportFormat(enum.Enum):
    WAV = "wav"
    MP3 = "mp3"

    @property
    def extension(self) -> str:
        return f".{self.value}"

    @classmethod
    def from_path(cls, path: str | Path) -> ExportFormat | None:
        suffix = Path(path).suffix.lower()
        return next((f for f in cls if f.extension == suffix), None)


WAV_SUBTYPE_LABELS = {
    "PCM_16": "16-bit PCM",
    "PCM_24": "24-bit PCM",
    "FLOAT": "32-bit float",
}


@dataclass(frozen=True, slots=True)
class ExportSettings:
    format: ExportFormat = ExportFormat.WAV
    wav_subtype: str = DEFAULT_WAV_SUBTYPE
    mp3_bitrate: int = DEFAULT_MP3_BITRATE
    dither: bool = DEFAULT_DITHER  # TPDF dither for 16-bit output (see ``dithers``)

    def __post_init__(self) -> None:
        if not isinstance(self.format, ExportFormat):
            raise ValueError(f"unknown format {self.format!r}")
        if self.wav_subtype not in WAV_SUBTYPES:
            raise ValueError(f"WAV subtype must be one of {WAV_SUBTYPES}, not {self.wav_subtype!r}")
        if self.mp3_bitrate not in MP3_BITRATES:
            raise ValueError(f"MP3 bitrate must be one of {MP3_BITRATES}, not {self.mp3_bitrate}")

    @property
    def is_16_bit(self) -> bool:
        """Whether the output is 16-bit (16-bit WAV, or MP3, which LAME gets as int16)."""
        return self.format is ExportFormat.MP3 or self.wav_subtype == "PCM_16"

    @property
    def dithers(self) -> bool:
        """Whether this export is dithered: ``dither`` is on and the output is 16-bit."""
        return self.dither and self.is_16_bit

    @property
    def clips_overs(self) -> bool:
        """Whether samples above full scale are clipped (everything but float WAV)."""
        return not (self.format is ExportFormat.WAV and self.wav_subtype == "FLOAT")

    def describe(self) -> str:
        """``"WAV, 16-bit PCM"`` / ``"MP3, 192 kbps"``."""
        if self.format is ExportFormat.WAV:
            return f"WAV, {WAV_SUBTYPE_LABELS[self.wav_subtype]}"
        return f"MP3, {self.mp3_bitrate} kbps"


@dataclass(frozen=True, slots=True)
class ExportResult:
    path: Path
    settings: ExportSettings
    frames: int  # frames in the file (after resampling, for MP3)
    sample_rate: int  # the file's rate
    bytes: int  # file size
    clipped: int  # samples above full scale that were clipped (0 for float WAV)
    seconds: float  # time the export took

    @property
    def duration(self) -> float:
        return self.frames / self.sample_rate


def estimate_bytes(frames: int, rate: int, settings: ExportSettings) -> int:
    """About the size of the file exported from ``frames`` at ``rate`` (exact for WAV but for
    the header; within a frame for MP3)."""
    if settings.format is ExportFormat.WAV:
        width = {"PCM_16": 2, "PCM_24": 3, "FLOAT": 4}[settings.wav_subtype]
        return 44 + frames * 2 * width
    out_rate = mp3_rate(rate)
    out_frames = -(-frames * out_rate // rate)
    # LAME's start delay and end padding (~1 frame each) plus the Info tag frame.
    mp3_frames = -(-(out_frames + LAME_ENCODER_DELAY) // 1152) + 2
    return int(mp3_frames * 144 * settings.mp3_bitrate * 1000 / out_rate)


def mp3_rate(rate: int) -> int:
    """The rate LAME gets: 44.1/48 kHz as is; the 44.1 kHz family -> 44.1 kHz; else 48 kHz."""
    if rate in MP3_RATES:
        return rate
    return 44100 if rate % 11025 == 0 else 48000


# --------------------------------------------------------------------------- helpers


def _check(samples: np.ndarray, rate: int) -> np.ndarray:
    samples = np.asarray(samples)
    if samples.ndim != 2 or samples.shape[1] != 2:
        raise ValueError(f"expected [frames, 2] samples, got shape {samples.shape}")
    if samples.dtype != np.float32:
        samples = samples.astype(np.float32)
    if int(rate) <= 0:
        raise ValueError(f"sample rate must be positive, got {rate}")
    if len(samples) == 0:
        raise ValueError("nothing to export: the clip is empty")
    return samples


def _report(progress: ProgressCallback | None, fraction: float) -> None:
    if progress is not None:
        progress(min(1.0, max(0.0, fraction)))


def _check_cancel(cancelled: CancelledCallback | None) -> None:
    if cancelled is not None and cancelled():
        raise Cancelled("Export cancelled")


def part_path(path: Path) -> Path:
    """A fresh temporary name next to ``path``."""
    return path.with_name(f"{path.name}.{secrets.token_hex(4)}.part")


@contextmanager
def _atomic(path: str | Path) -> Iterator[Path]:
    """Yield a temporary path next to ``path``; on success rename it to ``path`` (replacing
    it), on any error or cancel delete it."""
    target = Path(path)
    tmp = part_path(target)
    try:
        yield tmp
        os.replace(tmp, target)
    except BaseException:
        try:
            tmp.unlink(missing_ok=True)
        except OSError:
            log.warning("Could not delete the partial export %s", tmp, exc_info=True)
        raise


_SPLITMIX_GAMMA = np.uint64(0x9E3779B97F4A7C15)
_SPLITMIX_M1 = np.uint64(0xBF58476D1CE4E5B9)
_SPLITMIX_M2 = np.uint64(0x94D049BB133111EB)


def tpdf_noise(start: int, frames: int, seed: int = DITHER_SEED) -> np.ndarray:
    """TPDF dither for ``frames`` stereo frames from frame ``start``: float32 ``[frames, 2]`` in
    steps (the difference of two uniform values: triangular on (-1, 1), mean 0, variance 1/6).

    Each frame's noise depends only on ``(seed, frame)``: splitmix64 of the frame index gives 64
    bits, read as four 16-bit uniforms (left = u0 - u1, right = u2 - u3). So any chunking of a
    clip gives the same noise as one pass. (16-bit uniforms are far finer than dither needs.)
    """
    z = np.arange(start, start + frames, dtype=np.uint64)
    z += np.uint64(seed & 0xFFFFFFFFFFFFFFFF)
    z *= _SPLITMIX_GAMMA  # wraps mod 2**64 (numpy doesn't warn for arrays)
    z ^= z >> np.uint64(30)
    z *= _SPLITMIX_M1
    z ^= z >> np.uint64(27)
    z *= _SPLITMIX_M2
    z ^= z >> np.uint64(31)
    u = z.view(np.uint16).reshape(frames, 4).astype(np.int32)
    noise = (u[:, 0::2] - u[:, 1::2]).astype(np.float32)  # |u0 - u1| < 2**16: exact
    noise *= np.float32(2.0**-16)
    return noise


class _Quantizer:
    """float32 -> integers in reused chunk buffers: ``x * 2**(bits-1)``, (dither,) round, clip.

    Blocks must come in order: the quantizer counts frames to place the dither noise.
    """

    def __init__(self, bits: int, frames: int, *, dither: bool = False) -> None:
        self.bits = bits
        self.scale = float(2 ** (bits - 1))
        self.dtype = np.int16 if bits == 16 else np.int32
        self.dither = dither
        self.position = 0  # frames quantized so far
        self._f = np.empty((frames, 2), np.float32)
        self._i = np.empty((frames, 2), self.dtype)
        self.clipped = 0

    def __call__(self, block: np.ndarray) -> np.ndarray:
        n = len(block)
        if n > len(self._f):  # a resampled (upsampled) block can be longer than a chunk
            self._f = np.empty((n, 2), np.float32)
            self._i = np.empty((n, 2), self.dtype)
        f, out = self._f[:n], self._i[:n]
        np.multiply(block, np.float32(self.scale), out=f)
        # Overs beyond full scale (exactly +1.0 maps to scale and is clipped by one step; that
        # isn't counted, nor is dither pushing a sample just below full scale over it).
        self.clipped += int(np.count_nonzero(f > self.scale)) + int(
            np.count_nonzero(f < -self.scale)
        )
        if self.dither:
            f += tpdf_noise(self.position, n)
        self.position += n
        np.rint(f, out=f)
        np.clip(f, -self.scale, self.scale - 1, out=f)
        out[...] = f  # exact: |f| <= 2**23 fits float32's mantissa
        if self.bits == 24:
            np.left_shift(out, 8, out=out)  # libsndfile writes the top 24 bits of an int32
        return out


# --------------------------------------------------------------------------- WAV


def write_wav(
    samples: np.ndarray,
    rate: int,
    path: str | Path,
    subtype: str = DEFAULT_WAV_SUBTYPE,
    progress: ProgressCallback | None = None,
    cancelled: CancelledCallback | None = None,
    *,
    dither: bool = False,
    chunk_frames: int = CHUNK_FRAMES,
) -> ExportResult:
    """Write a stereo WAV: ``PCM_16`` (default), ``PCM_24`` or ``FLOAT`` (keeps overs).

    ``dither`` adds TPDF dither to 16-bit output (it is ignored for 24-bit and float)."""
    import soundfile as sf

    settings = ExportSettings(ExportFormat.WAV, wav_subtype=subtype, dither=dither)
    samples = _check(samples, rate)
    started = time.perf_counter()
    n = len(samples)
    width = {"PCM_16": 2, "PCM_24": 3, "FLOAT": 4}[subtype]
    container = "RF64" if n * 2 * width > WAV_MAX_DATA_BYTES else "WAV"
    if container == "RF64":
        log.info("Writing %s as RF64: the data is over 4 GB", path)
    quantize = (
        None
        if subtype == "FLOAT"
        else _Quantizer(width * 8, min(chunk_frames, n) or 1, dither=settings.dithers)
    )
    _report(progress, 0.0)
    with _atomic(path) as tmp:
        with sf.SoundFile(
            tmp, "w", samplerate=int(rate), channels=2, subtype=subtype, format=container
        ) as f:
            for a in range(0, n, chunk_frames):
                _check_cancel(cancelled)
                block = samples[a : a + chunk_frames]
                f.write(block if quantize is None else quantize(block))
                _report(progress, (a + len(block)) / n)
            _check_cancel(cancelled)
        size = tmp.stat().st_size
    result = ExportResult(
        Path(path),
        settings,
        n,
        int(rate),
        size,
        0 if quantize is None else quantize.clipped,
        time.perf_counter() - started,
    )
    _log_result(result)
    return result


# --------------------------------------------------------------------------- resampling


class _ChunkResampler:
    """``resample_poly`` in chunks, equal to resampling everything at once.

    Each chunk starts on a multiple of ``down`` input frames (so it maps to a whole output frame)
    and is resampled with ``context`` real neighbouring frames on both sides, more than the
    filter's half-length; the output for the context is cut off again. At the ends of the clip
    ``resample_poly`` pads with zeros, exactly as a one-pass resample does.
    """

    def __init__(self, samples: np.ndarray, rate: int, target: int) -> None:
        g = math.gcd(int(rate), int(target))
        self.up, self.down = target // g, rate // g
        self.samples = samples
        self.n = len(samples)
        self.out_frames = -(-self.n * self.up // self.down)  # ceil, like resample_poly
        # resample_poly's filter has 10 * max(up, down) taps per side at the upsampled rate,
        # i.e. 10 * max(up, down) / up input frames; take plenty more, rounded up to ``down``.
        half = 10 * max(self.up, self.down) / self.up
        self.context = self.down * math.ceil((2 * half + 64) / self.down)

    def chunks(self, chunk_frames: int) -> Iterator[tuple[int, np.ndarray]]:
        """``(input frames consumed so far, float32 output block)``."""
        from scipy.signal import resample_poly

        step = max(self.down, (chunk_frames // self.down) * self.down)
        for a in range(0, self.n, step):
            b = min(self.n, a + step)
            lo, hi = max(0, a - self.context), min(self.n, b + self.context)
            out = resample_poly(self.samples[lo:hi], self.up, self.down, axis=0)
            skip = (a - lo) * self.up // self.down  # exact: a and lo are multiples of down
            start = a * self.up // self.down
            stop = -(-b * self.up // self.down) if b == self.n else b * self.up // self.down
            yield b, np.asarray(out[skip : skip + stop - start], dtype=np.float32)


def resample(samples: np.ndarray, rate: int, target: int, chunk_frames: int = CHUNK_FRAMES):
    """Resample ``[frames, ch]`` audio (chunked; see ``_ChunkResampler``)."""
    if rate == target:
        return np.asarray(samples, dtype=np.float32)
    r = _ChunkResampler(samples, rate, target)
    return np.concatenate([block for _, block in r.chunks(chunk_frames)], axis=0)


# --------------------------------------------------------------------------- MP3

_MPEG1_L3_KBPS = (0, 32, 40, 48, 56, 64, 80, 96, 112, 128, 160, 192, 224, 256, 320)
_MPEG2_L3_KBPS = (0, 8, 16, 24, 32, 40, 48, 56, 64, 80, 96, 112, 128, 144, 160)
_MPEG1_RATES = (44100, 48000, 32000)


@dataclass(frozen=True, slots=True)
class Mp3Header:
    """The fields of an MPEG audio Layer III frame header that matter here."""

    mpeg1: bool
    kbps: int
    rate: int
    padding: bool
    mono: bool
    raw: bytes

    @classmethod
    def parse(cls, data: bytes) -> Mp3Header:
        if len(data) < 4:
            raise ValueError("an MP3 frame header has 4 bytes")
        h = int.from_bytes(data[:4], "big")
        version = (h >> 19) & 3  # 3 MPEG-1, 2 MPEG-2, 0 MPEG-2.5
        layer = (h >> 17) & 3  # 1 Layer III
        br_index = (h >> 12) & 15
        sr_index = (h >> 10) & 3
        if h >> 21 != 0x7FF or version == 1 or layer != 1 or br_index in (0, 15) or sr_index == 3:
            raise ValueError(f"not an MPEG Layer III frame header: {data[:4].hex()}")
        mpeg1 = version == 3
        rate = _MPEG1_RATES[sr_index] // (1 if mpeg1 else 2 if version == 2 else 4)
        kbps = (_MPEG1_L3_KBPS if mpeg1 else _MPEG2_L3_KBPS)[br_index]
        return cls(mpeg1, kbps, rate, bool((h >> 9) & 1), ((h >> 6) & 3) == 3, bytes(data[:4]))

    @property
    def samples_per_frame(self) -> int:
        return 1152 if self.mpeg1 else 576

    @property
    def frame_bytes(self) -> int:
        return (144 if self.mpeg1 else 72) * self.kbps * 1000 // self.rate + int(self.padding)

    @property
    def side_info_bytes(self) -> int:
        if self.mpeg1:
            return 17 if self.mono else 32
        return 9 if self.mono else 17


class FrameCounter:
    """Walks the MP3 frames of a byte stream fed in arbitrary pieces (validates it, too)."""

    def __init__(self) -> None:
        self.frames = 0
        self.bytes = 0
        self.first: Mp3Header | None = None
        self._skip = 0
        self._head = b""

    def feed(self, data: bytes) -> None:
        pos, n = 0, len(data)
        self.bytes += n
        while pos < n:
            if self._skip:
                take = min(self._skip, n - pos)
                self._skip -= take
                pos += take
                continue
            need = 4 - len(self._head)
            self._head += bytes(data[pos : pos + need])
            pos += need
            if len(self._head) < 4:
                return
            header = Mp3Header.parse(self._head)
            if self.first is None:
                self.first = header
            self.frames += 1
            self._skip = header.frame_bytes - 4
            self._head = b""

    @property
    def complete(self) -> bool:
        """Whether the stream ended on a frame boundary."""
        return self._skip == 0 and not self._head


def crc16(data: bytes, crc: int = 0) -> int:
    """CRC-16/ARC (polynomial 0x8005, reflected), the LAME tag's CRC."""
    for byte in data:
        crc ^= byte
        for _ in range(8):
            crc = (crc >> 1) ^ 0xA001 if crc & 1 else crc >> 1
    return crc


def info_frame_bytes(kbps: int, rate: int) -> int:
    """Size of the (unpadded) MPEG-1 frame that holds the Info tag."""
    return 144 * kbps * 1000 // rate


def info_tag_frame(
    first: Mp3Header, frames: int, total_bytes: int, delay: int, padding: int
) -> bytes:
    """A silent MP3 frame holding a Xing "Info" tag (CBR) with a LAME extension.

    ``frames`` counts the audio frames (not this one); ``total_bytes`` includes this frame.
    Layout: header, zeroed side info, "Info", flags (frames, bytes, TOC, quality), frames, bytes,
    a linear 100-entry TOC, quality, then the 36-byte LAME extension (encoder version, CBR
    method, bitrate, 12-bit encoder delay and padding, music length, CRCs).
    """
    h = int.from_bytes(first.raw, "big")
    h |= 1 << 16  # protection bit set: no CRC
    h &= ~(1 << 9)  # no padding
    h &= ~(1 << 8)  # private bit
    header = h.to_bytes(4, "big")
    size = (144 if first.mpeg1 else 72) * first.kbps * 1000 // first.rate
    frame = bytearray(size)
    frame[:4] = header
    pos = 4 + first.side_info_bytes
    toc = bytes(min(255, (i * 256) // 100) for i in range(100))
    xing = (
        b"Info"
        + (0x0F).to_bytes(4, "big")
        + frames.to_bytes(4, "big")
        + total_bytes.to_bytes(4, "big")
        + toc
        + (0).to_bytes(4, "big")
    )
    delay = min(max(delay, 0), 4095)
    padding = min(max(padding, 0), 4095)
    lame = (
        b"LAME3.100"  # the encoder: lameenc bundles LAME 3.100
        + bytes([0x01])  # tag revision 0, VBR method 1 = CBR
        + bytes([0])  # lowpass / 100 Hz (unknown)
        + bytes(8)  # peak amplitude, radio and audiophile replay gain
        + bytes([0])  # encoding flags, ATH type
        + bytes([min(first.kbps, 255)])  # CBR bitrate
        + ((delay << 12) | padding).to_bytes(3, "big")
        + bytes(4)  # misc, MP3 gain, preset and surround
        + total_bytes.to_bytes(4, "big")  # music length (tag frame + audio)
        + bytes(2)  # music CRC (not computed)
    )
    tag = xing + lame
    end = pos + len(tag)
    if end + 2 > size:
        raise ValueError(f"an Info tag doesn't fit a {size}-byte frame")
    frame[pos:end] = tag
    frame[end : end + 2] = crc16(bytes(frame[:end])).to_bytes(2, "big")
    return bytes(frame)


def write_mp3(
    samples: np.ndarray,
    rate: int,
    path: str | Path,
    bitrate: int = DEFAULT_MP3_BITRATE,
    progress: ProgressCallback | None = None,
    cancelled: CancelledCallback | None = None,
    *,
    dither: bool = False,
    chunk_frames: int = MP3_CHUNK_FRAMES,
    quality: int = MP3_QUALITY,
) -> ExportResult:
    """Write a CBR stereo MP3 (see the module docstring); ``dither`` dithers LAME's 16-bit
    input."""
    import lameenc

    settings = ExportSettings(ExportFormat.MP3, mp3_bitrate=bitrate, dither=dither)
    samples = _check(samples, rate)
    started = time.perf_counter()
    rate = int(rate)
    out_rate = mp3_rate(rate)
    if out_rate != rate:
        log.info("Resampling %d Hz to %d Hz for MP3", rate, out_rate)
        source = _ChunkResampler(samples, rate, out_rate)
        blocks = source.chunks(chunk_frames)
        out_frames = source.out_frames
    else:
        n = len(samples)
        blocks = (
            (min(n, a + chunk_frames), samples[a : a + chunk_frames])
            for a in range(0, n, chunk_frames)
        )
        out_frames = n
    n_in = len(samples)

    encoder = lameenc.Encoder()
    encoder.set_bit_rate(bitrate)
    encoder.set_in_sample_rate(out_rate)
    encoder.set_out_sample_rate(out_rate)  # never let LAME resample (it would change the delay)
    encoder.set_channels(2)
    encoder.set_quality(quality)
    quantize = _Quantizer(16, max(1, chunk_frames + 1), dither=dither)
    counter = FrameCounter()
    tag_bytes = info_frame_bytes(bitrate, out_rate)

    _report(progress, 0.0)
    with _atomic(path) as tmp:
        with open(tmp, "wb") as f:
            f.write(bytes(tag_bytes))  # the Info tag goes here once the counts are known
            for consumed, block in blocks:
                _check_cancel(cancelled)
                data = encoder.encode(quantize(block).tobytes())
                if data:
                    f.write(data)
                    counter.feed(data)
                _report(progress, consumed / n_in if n_in else 1.0)
            _check_cancel(cancelled)
            data = encoder.flush()
            if data:
                f.write(data)
                counter.feed(data)
            if counter.first is None or not counter.complete:
                raise RuntimeError("the MP3 encoder produced an incomplete stream")
            total = tag_bytes + counter.bytes
            padding = (
                counter.frames * counter.first.samples_per_frame - LAME_ENCODER_DELAY - out_frames
            )
            tag = info_tag_frame(counter.first, counter.frames, total, LAME_ENCODER_DELAY, padding)
            if len(tag) != tag_bytes:
                raise RuntimeError("Info tag size mismatch")
            f.seek(0)
            f.write(tag)
        size = tmp.stat().st_size
    result = ExportResult(
        Path(path),
        settings,
        out_frames,
        out_rate,
        size,
        quantize.clipped,
        time.perf_counter() - started,
    )
    _log_result(result)
    return result


# --------------------------------------------------------------------------- dispatch


def export(
    samples: np.ndarray,
    rate: int,
    path: str | Path,
    settings: ExportSettings,
    progress: ProgressCallback | None = None,
    cancelled: CancelledCallback | None = None,
) -> ExportResult:
    if settings.format is ExportFormat.WAV:
        return write_wav(
            samples, rate, path, settings.wav_subtype, progress, cancelled, dither=settings.dither
        )
    return write_mp3(
        samples, rate, path, settings.mp3_bitrate, progress, cancelled, dither=settings.dither
    )


def _log_result(result: ExportResult) -> None:
    log.info(
        "Exported %s (%s%s): %.3f s at %d Hz, %.1f MB, %d samples clipped, in %.2f s",
        result.path,
        result.settings.describe(),
        ", dithered" if result.settings.dithers else "",
        result.duration,
        result.sample_rate,
        result.bytes / 2**20,
        result.clipped,
        result.seconds,
    )


__all__ = [
    "Cancelled",
    "ExportFormat",
    "ExportResult",
    "ExportSettings",
    "MP3_BITRATES",
    "WAV_SUBTYPES",
    "estimate_bytes",
    "export",
    "resample",
    "tpdf_noise",
    "write_mp3",
    "write_wav",
]
