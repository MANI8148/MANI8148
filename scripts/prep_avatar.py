"""
Lightweight background removal for GitHub avatars (solid background color).

Flood-fills from the image border using color distance, then composites the
subject onto white and boosts local contrast. Output is consumed by
make_ascii_svg.py. For real photos use prep_photo.py (rembg-based) instead.

    python scripts/prep_avatar.py <input.jpg> [output.png]
"""
import sys
from collections import deque

import numpy as np
from PIL import Image, ImageFilter

INP = sys.argv[1] if len(sys.argv) > 1 else "source-photo.jpg"
OUT = sys.argv[2] if len(sys.argv) > 2 else "source-prepped.png"
THRESH = 60
FEATHER = 2


def remove_bg(rgb):
    h, w, _ = rgb.shape
    dist = np.zeros((h, w), np.float32)
    dist += (rgb[..., 0] - rgb[0, 0, 0]) ** 2
    dist += (rgb[..., 1] - rgb[0, 0, 1]) ** 2
    dist += (rgb[..., 2] - rgb[0, 0, 2]) ** 2
    dist = np.sqrt(dist)

    mask = np.zeros((h, w), bool)
    q = deque()
    for x in range(w):
        for y in (0, h - 1):
            if dist[y, x] < THRESH and not mask[y, x]:
                mask[y, x] = True
                q.append((y, x))
    for y in range(h):
        for x in (0, w - 1):
            if dist[y, x] < THRESH and not mask[y, x]:
                mask[y, x] = True
                q.append((y, x))
    while q:
        y, x = q.popleft()
        for dy, dx in ((-1, 0), (1, 0), (0, -1), (0, 1)):
            ny, nx = y + dy, x + dx
            if 0 <= ny < h and 0 <= nx < w and not mask[ny, nx] and dist[ny, nx] < THRESH:
                mask[ny, nx] = True
                q.append((ny, nx))
    return ~mask


img = Image.open(INP).convert("RGB")
rgb = np.array(img)
alpha = remove_bg(rgb)
am = Image.fromarray((alpha * 255).astype(np.uint8)).filter(
    ImageFilter.GaussianBlur(FEATHER)
)

out = Image.new("RGB", img.size, (255, 255, 255))
out.paste(img, (0, 0), am)

gray = np.array(out.convert("L")).astype(np.float32)
blur = np.array(out.convert("L").filter(ImageFilter.GaussianBlur(radius=24))).astype(np.float32)
gray = gray * 1.12 + 0.35 * (gray - blur) + 20
gray = np.clip(gray, 0, 255).astype(np.uint8)

Image.fromarray(gray).save(OUT)
print("wrote", OUT, img.size)
