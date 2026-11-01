"""Generate transparent-scope scientific workflow visuals for the Solar Filament README.

All shapes are conceptual and are not actual H-alpha observations, model predictions,
experimental measurements, or competition leaderboard results.
"""
from __future__ import annotations

import math
from pathlib import Path
from PIL import Image, ImageDraw, ImageFont

WIDTH, HEIGHT = 720, 216
FRAME_COUNT = 18
FRAME_DURATION_MS = 120
BACKGROUND = (13, 20, 35)
PANEL = (24, 35, 54)
BORDER = (64, 91, 118)
WHITE = (235, 243, 250)
MUTED = (159, 181, 200)
ACCENT = (104, 198, 221)
YELLOW = (248, 208, 114)


def draw_filaments(draw: ImageDraw.ImageDraw, x: int, y: int, stage: int, phase: float) -> None:
    """Render synthetic, labelled examples of source structure and segmentation."""
    if stage == 0:
        for offset in (-8, 0, 9):
            points = [
                (x + t, y + int(15 * math.sin((t + offset) / 35.0) + offset))
                for t in range(155)
            ]
            draw.line(points, fill=(56, 79, 109), width=4)
        return
    if stage == 1:
        for offset in (-8, 0, 9):
            points = [
                (x + t, y + int(15 * math.sin((t + offset) / 35.0) + offset))
                for t in range(155)
            ]
            draw.line(points, fill=ACCENT, width=4)
        return
    for index, offset in enumerate((-8, 0, 9)):
        points = [
            (x + t, y + int(15 * math.sin((t + offset) / 35.0) + offset))
            for t in range(155)
        ]
        draw.line(points, fill=(ACCENT, YELLOW, (184, 159, 221))[index], width=4)


def build_frame(phase: float) -> Image.Image:
    """Build one animation frame with explicit disclosure of conceptual scope."""
    canvas = Image.new("RGB", (WIDTH, HEIGHT), BACKGROUND)
    draw = ImageDraw.Draw(canvas)
    draw.text((22, 13), "SOLAR FILAMENT SEGMENTATION", fill=WHITE)
    draw.text((22, 35), "H-alpha observations  >  candidate mask  >  instance separation", fill=MUTED)
    for i, (x, label) in enumerate(((20, "01  OBSERVATION"), (263, "02  MASK"), (506, "03  INSTANCES"))):
        draw.rounded_rectangle((x, 65, x + 195, 164), radius=12, fill=PANEL, outline=BORDER, width=2)
        draw.text((x + 12, 75), label, fill=WHITE)
        draw_filaments(draw, x + 18, 125, i, phase)
    left = 223
    right = 465
    for anchor in (left, right):
        draw.line((anchor, 110, anchor + 27, 110), fill=ACCENT, width=3)
        draw.polygon([(anchor + 27, 110), (anchor + 18, 104), (anchor + 18, 116)], fill=ACCENT)
    moving_x = 30 + int(650 * phase)
    draw.rounded_rectangle((20, 178, 700, 184), radius=3, fill=(50, 65, 85))
    draw.rounded_rectangle((20, 178, moving_x, 184), radius=3, fill=ACCENT)
    draw.text((22, 193), "CONCEPTUAL ILLUSTRATION - NO EXPERIMENTAL IMAGE OR SCORE", fill=MUTED)
    return canvas


def main() -> None:
    """Write an optimized animated GIF and an accessible PNG fallback."""
    output = Path(__file__).resolve().parents[1] / "assets"
    output.mkdir(parents=True, exist_ok=True)
    frames = [build_frame(i / (FRAME_COUNT - 1)) for i in range(FRAME_COUNT)]
    frames[FRAME_COUNT // 2].save(output / "filament-workflow-concept.png", optimize=True)
    frames[0].save(
        output / "filament-workflow-concept.gif",
        save_all=True,
        append_images=frames[1:],
        duration=FRAME_DURATION_MS,
        loop=0,
        optimize=True,
    )
    print("Generated conceptual GIF and static PNG without experimental metrics.")


if __name__ == "__main__":
    main()
