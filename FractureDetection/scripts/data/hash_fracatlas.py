"""Cache perceptual hashes for FracAtlas and list unreadable images."""
import numpy as np
from PIL import Image

from fracai.data import load_fracatlas
from fracai.split import phashes

df = load_fracatlas()
bad = []
for pth in df.path:
    try:
        with Image.open(pth) as im:
            im.load()
    except OSError as e:
        bad.append(f"{pth}\t{e}")
open("results/truncated_images.txt", "w").write("\n".join(bad))
print("truncated/unreadable images:", len(bad), *bad, sep="\n")

h = phashes(df.path)
np.save("results/fracatlas_phash.npy", h)
hh = h.astype(np.uint8)
nn = np.array([np.delete((hh != hh[i]).sum(1), i).min() for i in range(len(hh))])
print("n", len(df), "nearest-neighbour Hamming distance (of 256 bits) percentiles:")
for q in (1, 5, 10, 25, 50, 75, 90):
    print(q, np.percentile(nn, q))
print("histogram (bin width 5):", np.histogram(nn, bins=range(0, 140, 5))[0])
