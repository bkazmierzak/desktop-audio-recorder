"""Downmix N-channel loopback audio to stereo.

Windows hands loopback data over in the endpoint's mix format, which can be mono, stereo or a
surround layout (the INZONE H9 II headset exposes 8-channel virtual 7.1). Channel order follows
the WAVEFORMATEXTENSIBLE default masks:

    1: FC
    2: FL FR
    3: FL FR FC
    4: FL FR BL BR                  (quad)
    5: FL FR FC BL BR               (5.0)
    6: FL FR FC LFE BL BR           (5.1)
    7: FL FR FC LFE BL BR BC        (6.1)
    8: FL FR FC LFE BL BR SL SR     (7.1)
   >8: 7.1 for the first 8, then extra channels alternate L/R

Front L/R stay at unity gain because stereo sources played on a surround endpoint arrive in
FL/FR only (measured on this machine). Centre and surrounds are added at -3 dB (1/sqrt 2),
a back-centre at -6 dB to both sides, and LFE is dropped.

No clipping by default: the Windows float mix itself goes above 1.0 (peaks of ~1.2 were
measured on loud game audio), and the float32 temp WAV can store those overs losslessly, so a
later gain/normalize can still recover them. Integer export must clip.
"""

from __future__ import annotations

from functools import lru_cache

import numpy as np

MINUS_3DB = float(1.0 / np.sqrt(2.0))
MINUS_6DB = 0.5

# Per-layout (left_gain, right_gain) per input channel.
_L = (1.0, 0.0)
_R = (0.0, 1.0)
_C = (MINUS_3DB, MINUS_3DB)
_LS = (MINUS_3DB, 0.0)
_RS = (0.0, MINUS_3DB)
_BC = (MINUS_6DB, MINUS_6DB)
_LFE = (0.0, 0.0)

_LAYOUTS: dict[int, tuple[tuple[float, float], ...]] = {
    1: ((1.0, 1.0),),
    2: (_L, _R),
    3: (_L, _R, _C),
    4: (_L, _R, _LS, _RS),
    5: (_L, _R, _C, _LS, _RS),
    6: (_L, _R, _C, _LFE, _LS, _RS),
    7: (_L, _R, _C, _LFE, _LS, _RS, _BC),
    8: (_L, _R, _C, _LFE, _LS, _RS, _LS, _RS),
}


@lru_cache(maxsize=16)
def downmix_matrix(channels: int) -> np.ndarray:
    """Return the ``[channels, 2]`` float32 gain matrix used to fold ``channels`` to stereo."""
    if channels < 1:
        raise ValueError(f"channels must be >= 1, got {channels}")
    if channels in _LAYOUTS:
        rows = list(_LAYOUTS[channels])
    else:
        rows = list(_LAYOUTS[8])
        rows += [_LS if i % 2 == 0 else _RS for i in range(channels - 8)]
    matrix = np.asarray(rows, dtype=np.float32)
    matrix.setflags(write=False)
    return matrix


def downmix_to_stereo(block: np.ndarray, *, clip: bool = False) -> np.ndarray:
    """Fold a ``[frames, channels]`` float block to a new ``[frames, 2]`` float32 array.

    With ``clip=True`` the result is limited to [-1, 1].
    """
    if block.ndim == 1:
        block = block[:, np.newaxis]
    if block.ndim != 2:
        raise ValueError(f"expected [frames, channels], got shape {block.shape}")
    channels = block.shape[1]
    if channels == 2:
        out = np.array(block, dtype=np.float32, copy=True)
    else:
        out = block.astype(np.float32, copy=False) @ downmix_matrix(channels)
    if clip:
        np.clip(out, -1.0, 1.0, out=out)
    return out


def interleaved_to_stereo(data: bytes, channels: int) -> np.ndarray:
    """Convert interleaved float32 bytes from PortAudio to a ``[frames, 2]`` array."""
    block = np.frombuffer(data, dtype=np.float32).reshape(-1, channels)
    return downmix_to_stereo(block)
