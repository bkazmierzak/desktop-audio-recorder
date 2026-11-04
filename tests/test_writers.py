"""export/writers.py: WAV (16/24-bit, float) and MP3 (lameenc + LAME Info tag), atomic writes."""

import numpy as np
import pytest
import soundfile as sf
from scipy.signal import resample_poly

from desktop_audio_recorder.export import writers
from desktop_audio_recorder.export.writers import (
    MP3_BITRATES,
    Cancelled,
    ExportFormat,
    ExportSettings,
    FrameCounter,
    Mp3Header,
    crc16,
    estimate_bytes,
    export,
    mp3_rate,
    resample,
    write_mp3,
    write_wav,
)

RATE = 48000


def tone(seconds, rate=RATE, hz=1000.0, amplitude=0.5, right_hz=None):
    t = np.arange(round(seconds * rate)) / rate
    left = amplitude * np.sin(2 * np.pi * hz * t)
    right = amplitude * np.sin(2 * np.pi * (right_hz or hz) * t)
    return np.stack([left, right], axis=1).astype(np.float32)


def dominant_hz(x, rate):
    x = x - x.mean()
    spectrum = np.abs(np.fft.rfft(x * np.hanning(len(x))))
    return np.argmax(spectrum) * rate / len(x)


def leftovers(folder):
    return sorted(p.name for p in folder.iterdir() if p.name.endswith(".part"))


# --------------------------------------------------------------------------- settings


def test_settings_defaults_and_describe():
    s = ExportSettings()
    assert (s.format, s.wav_subtype, s.mp3_bitrate) == (ExportFormat.WAV, "PCM_16", 192)
    assert s.describe() == "WAV, 16-bit PCM"
    assert ExportSettings(ExportFormat.WAV, "FLOAT").describe() == "WAV, 32-bit float"
    assert ExportSettings(ExportFormat.MP3, mp3_bitrate=320).describe() == "MP3, 320 kbps"
    assert s.clips_overs
    assert not ExportSettings(ExportFormat.WAV, "FLOAT").clips_overs
    assert ExportSettings(ExportFormat.MP3).clips_overs


@pytest.mark.parametrize(
    "kwargs",
    [{"wav_subtype": "PCM_8"}, {"mp3_bitrate": 160}, {"format": "wav"}],
)
def test_settings_reject_unknown_values(kwargs):
    with pytest.raises(ValueError):
        ExportSettings(**kwargs)


def test_format_from_path():
    assert ExportFormat.from_path("a/b.WAV") is ExportFormat.WAV
    assert ExportFormat.from_path("x.mp3") is ExportFormat.MP3
    assert ExportFormat.from_path("x.flac") is None
    assert ExportFormat.MP3.extension == ".mp3"


@pytest.mark.parametrize(
    ("rate", "expected"),
    [(48000, 48000), (44100, 44100), (96000, 48000), (88200, 44100), (22050, 44100)]
    + [(32000, 48000), (16000, 48000), (192000, 48000), (11025, 44100)],
)
def test_mp3_rate(rate, expected):
    assert mp3_rate(rate) == expected


# --------------------------------------------------------------------------- WAV


def test_pcm16_reads_back_within_half_a_step(tmp_path):
    rng = np.random.default_rng(1)
    x = rng.uniform(-0.999, 0.999, (50_000, 2)).astype(np.float32)
    x[:4] = [[0, 0], [-1, -1], [0.5, -0.5], [1 / 32768, -1 / 32768]]
    path = tmp_path / "a.wav"
    result = write_wav(x, RATE, path, "PCM_16")
    info = sf.info(str(path))
    assert (info.format, info.subtype, info.samplerate, info.channels, info.frames) == (
        "WAV",
        "PCM_16",
        RATE,
        2,
        len(x),
    )
    back, rate = sf.read(str(path), dtype="float32")
    assert rate == RATE
    assert np.max(np.abs(back - x)) <= 0.5 / 32768 + 1e-9
    np.testing.assert_array_equal(back[:4], x[:4])  # exactly representable values are exact
    assert result.clipped == 0
    assert (result.frames, result.sample_rate, result.bytes) == (len(x), RATE, path.stat().st_size)
    assert result.duration == pytest.approx(len(x) / RATE)


def test_pcm24_reads_back_within_half_a_step(tmp_path):
    rng = np.random.default_rng(2)
    x = rng.uniform(-0.999, 0.999, (50_000, 2)).astype(np.float32)
    path = tmp_path / "a.wav"
    write_wav(x, 44100, path, "PCM_24")
    info = sf.info(str(path))
    assert (info.subtype, info.samplerate) == ("PCM_24", 44100)
    back, _ = sf.read(str(path), dtype="float64")
    assert np.max(np.abs(back - x)) <= 0.5 / 2**23 + 1e-12
    raw, _ = sf.read(str(path), dtype="int32")
    assert np.all(raw & 0xFF == 0)  # 24 significant bits


def test_float_is_exact_including_overs(tmp_path):
    x = tone(0.5, amplitude=1.6)
    x[10] = [-2.5, 3.0]
    path = tmp_path / "a.wav"
    result = write_wav(x, RATE, path, "FLOAT")
    assert sf.info(str(path)).subtype == "FLOAT"
    back, _ = sf.read(str(path), dtype="float32")
    np.testing.assert_array_equal(back, x)
    assert result.clipped == 0


@pytest.mark.parametrize(("subtype", "bits"), [("PCM_16", 16), ("PCM_24", 24)])
def test_integer_formats_clip_overs(tmp_path, subtype, bits):
    x = np.zeros((1000, 2), np.float32)
    x[:, 0] = np.linspace(-1.5, 1.5, 1000)
    x[5] = [1.0, -1.0]  # full scale: not overs
    path = tmp_path / "a.wav"
    result = write_wav(x, RATE, path, subtype)
    back, _ = sf.read(str(path), dtype="float64")
    step = 1 / 2 ** (bits - 1)
    assert back.max() == pytest.approx(1 - step)
    assert back.min() == -1.0
    expected = int(np.count_nonzero(np.abs(x) > 1.0))
    assert result.clipped == expected > 0
    # Everything inside full scale is untouched by the clipping.
    inside = np.abs(x[:, 0]) < 1 - step
    assert np.max(np.abs(back[inside, 0] - x[inside, 0])) <= step / 2 + 1e-12


@pytest.mark.parametrize("subtype", ["PCM_16", "PCM_24", "FLOAT"])
def test_chunked_write_equals_one_chunk(tmp_path, subtype):
    rng = np.random.default_rng(3)
    x = rng.uniform(-1.2, 1.2, (10_007, 2)).astype(np.float32)
    write_wav(x, RATE, tmp_path / "one.wav", subtype, chunk_frames=1 << 20)
    write_wav(x, RATE, tmp_path / "many.wav", subtype, chunk_frames=999)
    assert (tmp_path / "one.wav").read_bytes() == (tmp_path / "many.wav").read_bytes()


def test_wav_progress_reaches_one(tmp_path):
    seen = []
    write_wav(tone(1.0), RATE, tmp_path / "a.wav", progress=seen.append, chunk_frames=10_000)
    assert seen[0] == 0.0
    assert seen[-1] == 1.0
    assert seen == sorted(seen)
    assert len(seen) > 4


def test_wav_switches_to_rf64_over_4_gb(tmp_path, monkeypatch):
    monkeypatch.setattr(writers, "WAV_MAX_DATA_BYTES", 1000)
    path = tmp_path / "big.wav"
    write_wav(tone(0.1), RATE, path, "PCM_16")
    info = sf.info(str(path))
    assert info.format == "RF64"
    assert info.frames == 4800


def test_wav_replaces_an_existing_file(tmp_path):
    path = tmp_path / "a.wav"
    path.write_bytes(b"old")
    write_wav(tone(0.1), RATE, path)
    assert sf.info(str(path)).frames == 4800
    assert leftovers(tmp_path) == []


@pytest.mark.parametrize("bad", [np.zeros((10, 1)), np.zeros(10), np.zeros((0, 2))])
def test_rejects_bad_shapes_and_empty_clips(tmp_path, bad):
    with pytest.raises(ValueError):
        write_wav(bad, RATE, tmp_path / "a.wav")
    with pytest.raises(ValueError):
        write_mp3(bad, RATE, tmp_path / "a.mp3")
    assert list(tmp_path.iterdir()) == []


# --------------------------------------------------------------------------- cancel / failure


def cancel_after(calls):
    count = {"n": 0}

    def cancelled():
        count["n"] += 1
        return count["n"] > calls

    return cancelled


@pytest.mark.parametrize("fmt", ["wav", "mp3"])
def test_cancel_leaves_no_file(tmp_path, fmt):
    path = tmp_path / f"a.{fmt}"
    settings = ExportSettings(ExportFormat(fmt))
    with pytest.raises(Cancelled):
        if fmt == "wav":
            write_wav(tone(2.0), RATE, path, cancelled=cancel_after(1), chunk_frames=20_000)
        else:
            write_mp3(tone(2.0), RATE, path, cancelled=cancel_after(1), chunk_frames=20_000)
    assert list(tmp_path.iterdir()) == []
    # The dispatcher too, cancelled before it starts.
    with pytest.raises(Cancelled):
        export(tone(0.5), RATE, path, settings, cancelled=lambda: True)
    assert list(tmp_path.iterdir()) == []


@pytest.mark.parametrize("fmt", ["wav", "mp3"])
def test_cancel_keeps_an_existing_file(tmp_path, fmt):
    path = tmp_path / f"a.{fmt}"
    path.write_bytes(b"precious")
    with pytest.raises(Cancelled):
        export(tone(2.0), RATE, path, ExportSettings(ExportFormat(fmt)), cancelled=cancel_after(1))
    assert path.read_bytes() == b"precious"
    assert leftovers(tmp_path) == []


def test_failure_mid_write_removes_the_partial_file(tmp_path):
    path = tmp_path / "a.wav"
    path.write_bytes(b"precious")

    def progress(fraction):
        if fraction > 0.3:
            raise OSError(28, "There is not enough space on the disk")

    with pytest.raises(OSError, match="not enough space"):
        write_wav(tone(2.0), RATE, path, progress=progress, chunk_frames=20_000)
    assert path.read_bytes() == b"precious"
    assert leftovers(tmp_path) == []


def test_missing_folder_fails_cleanly(tmp_path):
    with pytest.raises(Exception):  # noqa: B017 (libsndfile's error type)
        write_wav(tone(0.1), RATE, tmp_path / "missing" / "a.wav")
    with pytest.raises(OSError):
        write_mp3(tone(0.1), RATE, tmp_path / "missing" / "a.mp3")
    assert list(tmp_path.iterdir()) == []


# --------------------------------------------------------------------------- MP3


def mp3_frames(data):
    """Every frame header in a file, in order."""
    headers, pos = [], 0
    while pos < len(data):
        h = Mp3Header.parse(data[pos : pos + 4])
        headers.append((pos, h))
        pos += h.frame_bytes
    assert pos == len(data)
    return headers


@pytest.mark.parametrize("bitrate", MP3_BITRATES)
def test_mp3_is_valid_cbr_with_exact_duration_and_tone(tmp_path, bitrate):
    x = tone(1.5, hz=1000.0, right_hz=440.0)
    path = tmp_path / "a.mp3"
    result = write_mp3(x, RATE, path, bitrate)
    data = path.read_bytes()
    frames = mp3_frames(data)
    # The first frame is the Info tag; every frame is MPEG-1 Layer III at the bitrate.
    assert data[36:40] == b"Info"
    assert {(h.mpeg1, h.kbps, h.rate, h.mono) for _, h in frames} == {(True, bitrate, RATE, False)}
    info = sf.info(str(path))
    assert info.samplerate == RATE
    assert info.channels == 2
    back, rate = sf.read(str(path), dtype="float32")
    # Gapless decoders trim LAME's delay and padding (the Info tag): exactly the clip's frames.
    assert len(back) == len(x)
    assert abs(len(back) / rate - 1.5) < 0.05
    assert dominant_hz(back[:, 0], rate) == pytest.approx(1000, abs=2)
    assert dominant_hz(back[:, 1], rate) == pytest.approx(440, abs=2)
    # Aligned: no start delay left over.
    lag = np.argmax(np.correlate(back[:4800, 0], x[:4800, 0], "full")) - 4799
    assert lag == 0
    assert (result.frames, result.sample_rate, result.bytes) == (len(x), RATE, len(data))
    assert result.clipped == 0
    # The estimate is within a frame (frames alternate 417/418 bytes at 44.1 kHz).
    assert abs(len(data) - estimate_bytes(len(x), RATE, result.settings)) <= 1200


def test_mp3_info_tag_fields(tmp_path):
    x = tone(1.0, rate=44100)
    path = tmp_path / "a.mp3"
    write_mp3(x, 44100, path, 128)
    data = path.read_bytes()
    frames = mp3_frames(data)
    first = frames[0][1]
    tag_at = 4 + first.side_info_bytes
    assert data[tag_at : tag_at + 4] == b"Info"
    assert int.from_bytes(data[tag_at + 4 : tag_at + 8], "big") == 0x0F
    assert int.from_bytes(data[tag_at + 8 : tag_at + 12], "big") == len(frames) - 1
    assert int.from_bytes(data[tag_at + 12 : tag_at + 16], "big") == len(data)
    lame = tag_at + 120
    assert data[lame : lame + 9] == b"LAME3.100"
    assert data[lame + 9] & 0x0F == 1  # CBR
    assert data[lame + 20] == 128
    packed = int.from_bytes(data[lame + 21 : lame + 24], "big")
    delay, padding = packed >> 12, packed & 0xFFF
    assert delay == writers.LAME_ENCODER_DELAY
    assert (len(frames) - 1) * 1152 - delay - padding == len(x)
    assert int.from_bytes(data[lame + 28 : lame + 32], "big") == len(data)
    crc_at = lame + 34
    assert int.from_bytes(data[crc_at : crc_at + 2], "big") == crc16(data[:crc_at])
    assert crc_at == 190  # MPEG-1 stereo: the CRC covers the first 190 bytes


def test_mp3_clips_overs_and_counts_them(tmp_path):
    x = tone(0.5, amplitude=1.5)
    result = write_mp3(x, RATE, tmp_path / "a.mp3")
    assert result.clipped == int(np.count_nonzero(np.abs(x) > 1.0)) > 0
    back, _ = sf.read(str(tmp_path / "a.mp3"), dtype="float32")
    assert np.max(np.abs(back)) < 1.2  # a clipped sine (codec overshoot only), not wrapped


@pytest.mark.parametrize(("rate", "out_rate"), [(96000, 48000), (22050, 44100), (32000, 48000)])
def test_mp3_resamples_other_rates(tmp_path, rate, out_rate):
    x = tone(1.25, rate=rate, hz=1000.0)
    path = tmp_path / "a.mp3"
    result = write_mp3(x, rate, path, 192, chunk_frames=8192)
    assert (result.sample_rate, result.frames) == (out_rate, -(-len(x) * out_rate // rate))
    back, got_rate = sf.read(str(path), dtype="float32")
    assert got_rate == out_rate
    assert len(back) == result.frames
    assert abs(len(back) / got_rate - 1.25) < 0.05
    assert dominant_hz(back[:, 0], got_rate) == pytest.approx(1000, abs=2)


def test_mp3_progress_and_chunking(tmp_path):
    seen = []
    x = tone(2.0)
    write_mp3(x, RATE, tmp_path / "one.mp3", chunk_frames=1 << 20)
    write_mp3(x, RATE, tmp_path / "many.mp3", progress=seen.append, chunk_frames=7000)
    assert seen[0] == 0.0 and seen[-1] == 1.0 and seen == sorted(seen)
    # LAME's output doesn't depend on how the input was split up.
    assert (tmp_path / "one.mp3").read_bytes() == (tmp_path / "many.mp3").read_bytes()


def test_export_dispatches(tmp_path):
    x = tone(0.3)
    r = export(x, RATE, tmp_path / "a.wav", ExportSettings(ExportFormat.WAV, "PCM_24"))
    assert sf.info(str(r.path)).subtype == "PCM_24"
    r = export(x, RATE, tmp_path / "a.mp3", ExportSettings(ExportFormat.MP3, mp3_bitrate=256))
    assert {h.kbps for _, h in mp3_frames(r.path.read_bytes())} == {256}


# --------------------------------------------------------------------------- pieces


@pytest.mark.parametrize(("rate", "target"), [(96000, 48000), (22050, 44100), (44100, 48000)])
@pytest.mark.parametrize("chunk", [1000, 4096, 1 << 20])
def test_chunked_resample_equals_one_pass(rate, target, chunk):
    rng = np.random.default_rng(4)
    x = rng.standard_normal((12_345, 2)).astype(np.float32)
    got = resample(x, rate, target, chunk_frames=chunk)
    from math import gcd

    g = gcd(rate, target)
    want = resample_poly(x, target // g, rate // g, axis=0)
    assert got.shape == want.shape
    assert np.max(np.abs(got - want)) < 1e-5


def test_resample_same_rate_is_identity():
    x = tone(0.1)
    assert resample(x, RATE, RATE) is x


def test_crc16_check_value():
    assert crc16(b"123456789") == 0xBB3D  # CRC-16/ARC


def test_header_parse_rejects_garbage():
    for bad in (b"\x00\x00\x00\x00", b"ID3\x04", b"\xff\xfb\xf0\x00", b"\xff"):
        with pytest.raises(ValueError):
            Mp3Header.parse(bad)
    h = Mp3Header.parse(b"\xff\xfb\xb4\x04")  # MPEG-1 L3, 192 kbps, 48 kHz, joint stereo
    assert (h.mpeg1, h.kbps, h.rate, h.mono, h.frame_bytes) == (True, 192, 48000, False, 576)


def test_frame_counter_handles_any_split(tmp_path):
    write_mp3(tone(0.5, rate=44100), 44100, tmp_path / "a.mp3", 128)
    data = (tmp_path / "a.mp3").read_bytes()
    whole = FrameCounter()
    whole.feed(data)
    assert whole.complete
    pieces = FrameCounter()
    for i in range(0, len(data), 3):
        pieces.feed(data[i : i + 3])
    assert (pieces.frames, pieces.bytes, pieces.complete) == (whole.frames, len(data), True)
    partial = FrameCounter()
    partial.feed(data[:-10])
    assert not partial.complete


def test_estimate_bytes_wav_is_exact_size(tmp_path):
    x = tone(0.25)
    for subtype in ("PCM_16", "PCM_24", "FLOAT"):
        r = write_wav(x, RATE, tmp_path / f"{subtype}.wav", subtype)
        estimate = estimate_bytes(len(x), RATE, r.settings)
        assert abs(estimate - r.bytes) <= 100  # header size differs a little (FLOAT has 'fact')


# --------------------------------------------------------------------------- dither


def test_tpdf_noise_is_triangular_and_chunk_independent():
    n = writers.tpdf_noise(0, 400_000)
    assert n.dtype == np.float32 and n.shape == (400_000, 2)
    assert n.min() > -1 and n.max() < 1
    assert abs(n.mean()) < 0.003
    assert n.var() == pytest.approx(1 / 6, rel=0.01)
    # Triangular: a quarter of the mass lies beyond |0.5| (2 * 0.5 * 0.5**2).
    assert np.mean(np.abs(n) > 0.5) == pytest.approx(0.25, abs=0.003)
    # Independent channels and neighbours.
    assert abs(np.corrcoef(n[:, 0], n[:, 1])[0, 1]) < 0.01
    assert abs(np.corrcoef(n[:-1, 0], n[1:, 0])[0, 1]) < 0.01
    # Any split gives the same values; another seed gives others.
    parts = [writers.tpdf_noise(a, b - a) for a, b in [(0, 7), (7, 1000), (1000, 4321)]]
    assert np.array_equal(np.concatenate(parts), writers.tpdf_noise(0, 4321))
    assert not np.array_equal(writers.tpdf_noise(0, 100, seed=1), writers.tpdf_noise(0, 100))


def test_settings_dither_only_16_bit():
    assert ExportSettings().dither and ExportSettings().dithers
    assert ExportSettings(ExportFormat.MP3).dithers
    assert not ExportSettings(ExportFormat.WAV, "PCM_24").dithers
    assert not ExportSettings(ExportFormat.WAV, "FLOAT").dithers
    assert not ExportSettings(dither=False).dithers
    assert ExportSettings(ExportFormat.WAV, "PCM_24").is_16_bit is False


def test_dithered_pcm16_error_is_noise_not_distortion(tmp_path):
    rng = np.random.default_rng(5)
    x = rng.uniform(-0.9, 0.9, (200_000, 2)).astype(np.float32)
    write_wav(x, RATE, tmp_path / "d.wav", "PCM_16", dither=True)
    back, _ = sf.read(str(tmp_path / "d.wav"), dtype="float64")
    err = (back - x) * 32768  # in steps
    assert np.max(np.abs(err)) <= 1.5 + 1e-6  # TPDF (<1) plus rounding (<=1/2)
    assert abs(err.mean()) < 0.01
    assert err.var() == pytest.approx(1 / 6 + 1 / 12, rel=0.03)  # dither + rounding
    # The error doesn't follow the signal.
    assert abs(np.corrcoef(err[:, 0], x[:, 0])[0, 1]) < 0.01


def test_dither_keeps_a_tone_below_one_step(tmp_path):
    # A 1 kHz tone at 0.4 steps peak: plain rounding turns it into silence, dither keeps it
    # (as tone + noise), which is the point of dithering.
    x = tone(1.0, amplitude=0.4 / 32768)
    write_wav(x, RATE, tmp_path / "plain.wav", "PCM_16")
    write_wav(x, RATE, tmp_path / "dith.wav", "PCM_16", dither=True)
    plain, _ = sf.read(str(tmp_path / "plain.wav"), dtype="float64")
    dith, _ = sf.read(str(tmp_path / "dith.wav"), dtype="float64")
    assert not plain.any()
    assert dominant_hz(dith[:, 0], RATE) == pytest.approx(1000, abs=2)


@pytest.mark.parametrize("subtype", ["PCM_24", "FLOAT"])
def test_dither_is_ignored_above_16_bit(tmp_path, subtype):
    x = tone(0.2)
    write_wav(x, RATE, tmp_path / "a.wav", subtype)
    write_wav(x, RATE, tmp_path / "b.wav", subtype, dither=True)
    assert (tmp_path / "a.wav").read_bytes() == (tmp_path / "b.wav").read_bytes()


def test_dithered_output_is_reproducible_and_chunk_independent(tmp_path):
    rng = np.random.default_rng(6)
    x = rng.uniform(-1.2, 1.2, (30_011, 2)).astype(np.float32)
    write_wav(x, RATE, tmp_path / "one.wav", dither=True, chunk_frames=1 << 20)
    write_wav(x, RATE, tmp_path / "many.wav", dither=True, chunk_frames=997)
    write_wav(x, RATE, tmp_path / "again.wav", dither=True, chunk_frames=4096)
    one = (tmp_path / "one.wav").read_bytes()
    assert one == (tmp_path / "many.wav").read_bytes() == (tmp_path / "again.wav").read_bytes()
    write_wav(x, RATE, tmp_path / "plain.wav")
    assert one != (tmp_path / "plain.wav").read_bytes()


def test_dither_still_counts_and_clips_overs(tmp_path):
    x = np.full((1000, 2), 1.5, np.float32)
    x[500:] = -1.5
    result = write_wav(x, RATE, tmp_path / "a.wav", dither=True)
    assert result.clipped == 2000
    back, _ = sf.read(str(tmp_path / "a.wav"), dtype="int16")
    assert back[:500].min() == 32767 and back[500:].max() == -32768


@pytest.mark.parametrize("rate", [RATE, 96000])
def test_dithered_mp3_is_chunk_independent(tmp_path, rate):
    x = tone(1.5, rate=rate, amplitude=0.01)
    write_mp3(x, rate, tmp_path / "one.mp3", dither=True, chunk_frames=1 << 20)
    write_mp3(x, rate, tmp_path / "many.mp3", dither=True, chunk_frames=7000)
    assert (tmp_path / "one.mp3").read_bytes() == (tmp_path / "many.mp3").read_bytes()
    back, got_rate = sf.read(str(tmp_path / "one.mp3"), dtype="float32")
    assert len(back) == round(1.5 * got_rate)
    assert dominant_hz(back[:, 0], got_rate) == pytest.approx(1000, abs=2)


@pytest.mark.parametrize("fmt", ["wav", "mp3"])
def test_export_passes_the_dither_setting(tmp_path, fmt, monkeypatch):
    seen = []
    real = writers._Quantizer

    def spy(*args, **kwargs):
        q = real(*args, **kwargs)
        seen.append(q.dither)
        return q

    monkeypatch.setattr(writers, "_Quantizer", spy)
    for dither in (True, False):
        export(
            tone(0.1),
            RATE,
            tmp_path / f"{dither}.{fmt}",
            ExportSettings(ExportFormat(fmt), dither=dither),
        )
    assert seen == [True, False]
