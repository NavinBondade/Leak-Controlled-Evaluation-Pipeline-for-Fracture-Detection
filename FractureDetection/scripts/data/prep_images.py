"""Build the image caches: prep_images.py [fracatlas|graz] [size]."""
import sys

import numpy as np
from tqdm import tqdm

from fracai.cache import iter_graz_zip, normalise, read_path
from fracai.data import DATA, load_fracatlas, load_graz

which = sys.argv[1]
size = int(sys.argv[2]) if len(sys.argv) > 2 else 512
out_dir = DATA / "cache"
out_dir.mkdir(exist_ok=True)

if which == "fracatlas":
    df = load_fracatlas()
    arr = np.lib.format.open_memmap(out_dir / f"fracatlas_{size}.npy", "w+", np.uint8, (len(df), size, size))
    for i, p in enumerate(tqdm(df.path)):
        with read_path(p) as im:
            arr[i] = normalise(im, size)
elif which == "graz":
    df = load_graz()
    idx = {s: i for i, s in enumerate(df.case_id)}
    arr = np.lib.format.open_memmap(out_dir / f"graz_{size}.npy", "w+", np.uint8, (len(df), size, size))
    done = np.zeros(len(df), bool)
    for stem, im in tqdm(iter_graz_zip(str(DATA.parent / "SAM" / "GRAZPEDWRI-DX.zip")), total=len(df)):
        if stem in idx:
            arr[idx[stem]] = normalise(im, size)
            done[idx[stem]] = True
    print(f"images found for {done.sum()}/{len(df)} metadata rows")
    if not done.all():
        print("missing:", df.case_id[~done].head().tolist())
arr.flush()
print("wrote", out_dir, which, size)
