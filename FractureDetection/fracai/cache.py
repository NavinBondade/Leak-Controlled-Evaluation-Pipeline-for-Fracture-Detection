"""Decode images once into a uint8 memmap aligned with the metadata rows."""

import io
import zipfile

import numpy as np
from PIL import Image, ImageFile

ImageFile.LOAD_TRUNCATED_IMAGES = True


def normalise(im: Image.Image, size: int) -> np.ndarray:
    a = np.asarray(im.convert("I;16") if im.mode == "I;16" else im.convert("L"), dtype=np.float32)
    lo, hi = np.percentile(a, [1, 99])
    a = np.clip((a - lo) / max(hi - lo, 1e-6), 0, 1)
    h, w = a.shape
    s = size / max(h, w)
    nh, nw = max(1, round(h * s)), max(1, round(w * s))
    small = Image.fromarray((a * 255).astype(np.uint8)).resize((nw, nh), Image.BILINEAR)
    out = np.zeros((size, size), np.uint8)
    y0, x0 = (size - nh) // 2, (size - nw) // 2
    out[y0 : y0 + nh, x0 : x0 + nw] = np.asarray(small)
    return out


def read_path(path) -> Image.Image:
    return Image.open(path)


def iter_graz_zip(outer_zip: str):
    """Yield (filestem, PIL image) streamed from the nested zips without extracting them."""
    with zipfile.ZipFile(outer_zip) as outer:
        for part in sorted(n for n in outer.namelist() if n.startswith("images_part")):
            with zipfile.ZipFile(outer.open(part)) as inner:
                for name in inner.namelist():
                    if name.lower().endswith(".png"):
                        yield name[:-4], Image.open(io.BytesIO(inner.read(name)))
