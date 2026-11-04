"""Draw the app icon and write it as a multi-size .ico (run it again after changing the design).

    uv run python packaging/make_icon.py

Writes ``src/desktop_audio_recorder/resources/app.ico`` (16-256 px, PNG-compressed entries, as
Windows Vista and later read them) and ``app.png`` (256 px, for previews). The design: a dark
blue rounded tile with a light waveform and a red "record" dot. Small sizes get fewer, wider
bars so they stay legible.
"""

from __future__ import annotations

import os
import struct
import sys
from pathlib import Path

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PySide6.QtCore import QBuffer, QByteArray, QIODevice, QPointF, QRectF, Qt  # noqa: E402
from PySide6.QtGui import (  # noqa: E402
    QColor,
    QGuiApplication,
    QImage,
    QLinearGradient,
    QPainter,
    QPen,
)

SIZES = (16, 20, 24, 32, 40, 48, 64, 128, 256)
OUT = Path(__file__).resolve().parent.parent / "src" / "desktop_audio_recorder" / "resources"

TILE_TOP = QColor(44, 62, 92)
TILE_BOTTOM = QColor(22, 30, 46)
WAVE = QColor(120, 205, 255)
DOT = QColor(232, 57, 53)


def render(size: int) -> QImage:
    image = QImage(size, size, QImage.Format.Format_ARGB32_Premultiplied)
    image.fill(Qt.GlobalColor.transparent)
    p = QPainter(image)
    p.setRenderHint(QPainter.RenderHint.Antialiasing)
    s = float(size)
    margin = 0.0 if size <= 24 else s * 0.04
    tile = QRectF(margin, margin, s - 2 * margin, s - 2 * margin)
    gradient = QLinearGradient(tile.topLeft(), tile.bottomLeft())
    gradient.setColorAt(0.0, TILE_TOP)
    gradient.setColorAt(1.0, TILE_BOTTOM)
    p.setPen(Qt.PenStyle.NoPen)
    p.setBrush(gradient)
    radius = tile.width() * 0.22
    p.drawRoundedRect(tile, radius, radius)

    # The waveform: symmetric bars around the middle, left of the dot.
    heights = (
        (0.30, 0.62, 1.0, 0.72, 0.42) if size <= 32 else (0.24, 0.5, 0.82, 1.0, 0.7, 0.44, 0.26)
    )
    n = len(heights)
    area = QRectF(
        tile.left() + tile.width() * 0.14,
        tile.top() + tile.height() * 0.18,
        tile.width() * 0.62,
        tile.height() * 0.50,
    )
    pitch = area.width() / n
    bar = max(1.0, pitch * (0.62 if size <= 32 else 0.55))
    p.setBrush(WAVE)
    for i, h in enumerate(heights):
        height = max(bar, area.height() * h)
        x = area.left() + pitch * i + (pitch - bar) / 2
        y = area.center().y() - height / 2
        if size <= 32:  # crisp edges at icon sizes
            x, bar_w = round(x), max(1, round(bar))
            p.drawRect(QRectF(x, round(y), bar_w, round(height)))
        else:
            p.drawRoundedRect(QRectF(x, y, bar, height), bar / 2, bar / 2)

    # The record dot, bottom right, with a ring in the tile colour so it reads on the bars.
    r = tile.width() * (0.19 if size <= 32 else 0.16)
    centre = QPointF(tile.right() - tile.width() * 0.27, tile.bottom() - tile.height() * 0.27)
    ring = max(1.0, s * 0.035)
    p.setPen(QPen(TILE_BOTTOM, ring))
    p.setBrush(DOT)
    p.drawEllipse(centre, r, r)
    p.end()
    return image


def png_bytes(image: QImage) -> bytes:
    data = QByteArray()
    buffer = QBuffer(data)
    buffer.open(QIODevice.OpenModeFlag.WriteOnly)
    image.save(buffer, "PNG")
    buffer.close()
    return bytes(data)


def write_ico(path: Path, images: dict[int, bytes]) -> None:
    """ICONDIR + ICONDIRENTRYs + PNG data (each entry: 32 bpp, PNG-compressed)."""
    count = len(images)
    header = struct.pack("<HHH", 0, 1, count)
    offset = 6 + 16 * count
    entries, blobs = [], []
    for size, blob in sorted(images.items()):
        dim = 0 if size >= 256 else size  # 0 means 256
        entries.append(struct.pack("<BBBBHHII", dim, dim, 0, 0, 1, 32, len(blob), offset))
        blobs.append(blob)
        offset += len(blob)
    path.write_bytes(header + b"".join(entries) + b"".join(blobs))


def main() -> int:
    app = QGuiApplication.instance() or QGuiApplication(sys.argv[:1])
    OUT.mkdir(parents=True, exist_ok=True)
    images = {size: png_bytes(render(size)) for size in SIZES}
    write_ico(OUT / "app.ico", images)
    (OUT / "app.png").write_bytes(images[256])
    print(f"Wrote {OUT / 'app.ico'} ({len(SIZES)} sizes) and app.png")
    del app
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
