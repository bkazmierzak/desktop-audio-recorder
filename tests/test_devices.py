"""DeviceManager with a mocked PyAudio module."""

import pytest
from conftest import loopback_device, make_fake_pyaudio, output_device

from desktop_audio_recorder.audio.devices import DeviceError, DeviceManager, resolve_loopback


def test_resolves_default_loopback(fake_pyaudio):
    device = DeviceManager(fake_pyaudio).resolve_default_loopback()
    assert device.index == 13
    assert device.name == "Speakers (INZONE H9 II - Game) [Loopback]"
    assert device.sample_rate == 48000
    assert isinstance(device.sample_rate, int)
    assert device.channels == 8
    assert device.output_index == 10
    assert device.output_name == "Speakers (INZONE H9 II - Game)"
    assert device.output_channels == 8


def test_each_call_uses_fresh_pyaudio_and_terminates_it(fake_pyaudio):
    dm = DeviceManager(fake_pyaudio)
    dm.resolve_default_loopback()
    dm.list_output_devices()
    assert len(fake_pyaudio.instances) == 2
    assert all(pa.terminated for pa in fake_pyaudio.instances)


def test_picks_up_default_device_change(fake_pyaudio):
    dm = DeviceManager(fake_pyaudio)
    assert dm.resolve_default_loopback().output_index == 10
    fake_pyaudio.default_output = 11  # user switches Windows default to the Realtek speakers
    device = dm.resolve_default_loopback()
    assert device.index == 14
    assert device.channels == 2
    assert device.output_name == "Speakers (Realtek(R) Audio)"


def test_uses_given_pyaudio_without_terminating(fake_pyaudio):
    dm = DeviceManager(fake_pyaudio)
    pa = dm.create_pyaudio()
    assert dm.resolve_default_loopback(pa).index == 13
    assert not pa.terminated
    assert len(fake_pyaudio.instances) == 1


def test_no_wasapi_raises():
    dm = DeviceManager(make_fake_pyaudio(has_wasapi=False))
    with pytest.raises(DeviceError, match="WASAPI"):
        dm.resolve_default_loopback()
    with pytest.raises(DeviceError):
        dm.list_output_devices()


def test_no_default_output_raises():
    dm = DeviceManager(make_fake_pyaudio(default_output=-1))
    with pytest.raises(DeviceError, match="No default playback"):
        dm.resolve_default_loopback()


def test_missing_loopback_raises():
    module = make_fake_pyaudio(devices=[output_device(0, "Speakers (USB)")], default_output=0)
    with pytest.raises(DeviceError, match="No loopback"):
        DeviceManager(module).resolve_default_loopback()


def test_loopback_matched_by_substring_when_name_differs():
    lb = loopback_device(1, "Speakers (USB)")
    lb["name"] = "Speakers (USB) [Loopback 2]"
    module = make_fake_pyaudio(devices=[output_device(0, "Speakers (USB)"), lb], default_output=0)
    assert DeviceManager(module).resolve_default_loopback().index == 1


def test_mono_device_and_non_integer_rate():
    module = make_fake_pyaudio(
        devices=[
            output_device(0, "Headset (BT Hands-Free)", 1, rate=16000.0),
            loopback_device(1, "Headset (BT Hands-Free)", 1, rate=16000.0),
        ],
        default_output=0,
    )
    device = DeviceManager(module).resolve_default_loopback()
    assert (device.channels, device.sample_rate, device.output_channels) == (1, 16000, 1)


def test_list_output_devices(fake_pyaudio):
    devices = DeviceManager(fake_pyaudio).list_output_devices()
    # Only WASAPI outputs: no MME duplicates, no loopbacks, no microphones.
    assert [d.index for d in devices] == [10, 11]
    inzone, realtek = devices
    assert inzone.is_default and not realtek.is_default
    assert (inzone.loopback_index, realtek.loopback_index) == (13, 14)
    assert (inzone.channels, realtek.channels) == (8, 2)
    assert inzone.sample_rate == 48000


def test_list_output_device_without_loopback():
    module = make_fake_pyaudio(devices=[output_device(0, "Speakers (USB)")], default_output=0)
    (device,) = DeviceManager(module).list_output_devices()
    assert device.loopback_index is None


def test_resolve_specific_output(fake_pyaudio):
    dm = DeviceManager(fake_pyaudio)
    with dm.session() as pa:
        device = resolve_loopback(pa, 11, fake_pyaudio)
        assert (device.index, device.output_name) == (14, "Speakers (Realtek(R) Audio)")
        with pytest.raises(DeviceError, match="not a WASAPI"):
            resolve_loopback(pa, 3, fake_pyaudio)  # the MME twin
