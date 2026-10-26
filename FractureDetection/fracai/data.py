"""Metadata tables for FracAtlas and GRAZPEDWRI-DX in a common schema."""

import json
from pathlib import Path

import numpy as np
import pandas as pd

DATA = Path(r"D:\Ad_Pro\Oxford\data")
FRAC = DATA / "FracAtlas"
GRAZ = DATA / "GRAZPEDWRI-DX"


def load_fracatlas(root: Path = FRAC) -> pd.DataFrame:
    df = pd.read_csv(root / "dataset.csv")
    # two filenames (IMG0003375/6) exist in both folders; key on (name, folder) and pick by CSV label
    files = {(p.name, p.parent.name): p for p in (root / "images").rglob("*.jpg")}
    paths = {
        n: files.get((n, "Fractured" if y else "Non_fractured"), files.get((n, "Non_fractured" if y else "Fractured")))
        for n, y in zip(df.image_id, df.fractured)
    }
    paths = {n: p for n, p in paths.items() if p is not None}
    orphan_img = sorted({n for n, _ in files} - set(df.image_id))
    label_conflict = [
        n
        for n, y in zip(df.image_id, df.fractured)
        if n in paths and paths[n].parent.name != ("Fractured" if y else "Non_fractured")
    ]
    if label_conflict:
        print(f"[fracatlas] csv label disagrees with folder for: {label_conflict}")
    missing_img = sorted(set(df.image_id) - set(paths))
    if missing_img or orphan_img:
        print(f"[fracatlas] csv rows without image: {len(missing_img)}, images without csv row: {len(orphan_img)}")
    df = df[df.image_id.isin(paths)].copy()
    parts = ["hand", "leg", "hip", "shoulder", "mixed"]
    # the `mixed` flag always co-occurs with its constituent parts (398 images), so it takes priority
    df["bodypart"] = np.where(df.mixed == 1, "mixed", df[parts[:4]].idxmax(axis=1))
    df["view"] = df[["frontal", "lateral", "oblique"]].idxmax(axis=1)
    out = pd.DataFrame(
        {
            "case_id": df.image_id.str.replace(".jpg", "", regex=False),
            "path": df.image_id.map(paths).astype(str),
            "y": df.fractured.astype(int),
            "bodypart": df.bodypart,
            "view": df.view,
            "hardware": df.hardware.astype(int),
            "multiscan": df.multiscan.astype(int),
            "fracture_count": df.fracture_count.astype(int),
        }
    )
    return out.reset_index(drop=True)


def load_graz(root: Path = GRAZ, image_dir: Path | None = None) -> pd.DataFrame:
    """image_dir: folder holding the extracted images (filestem.png)."""
    df = pd.read_csv(root / "dataset.csv")
    df["fracture_visible"] = df.fracture_visible.fillna(0).astype(int)
    out = pd.DataFrame(
        {
            "case_id": df.filestem,
            "path": "" if image_dir is None else df.filestem.map(lambda s: str(image_dir / f"{s}.png")),
            "y": df.fracture_visible,
            "patient_id": df.patient_id,
            "bodypart": "wrist",
            "view": df.projection.map({1: "frontal", 2: "lateral", 3: "oblique"}),
            "hardware": df.metal.fillna(0).astype(int),
            "cast": df.cast.fillna(0).astype(int),
            "uncertain": df.diagnosis_uncertain.fillna(0).astype(int),
            "age": df.age,
            "sex": df.gender,
            "device": df.device_manufacturer,
        }
    )
    return out


def load_boxes(size: int = 512) -> dict[str, list[tuple[float, float, float, float]]]:
    """Annotated fracture boxes per case_id as (x, y, w, h) in the letterboxed size x size image."""
    coco = json.load(open(FRAC / "Annotations" / "COCO JSON" / "COCO_fracture_masks.json"))
    images = {i["id"]: i for i in coco["images"]}
    boxes: dict[str, list[tuple[float, float, float, float]]] = {}
    for a in coco["annotations"]:
        im = images[a["image_id"]]
        s = size / max(im["width"], im["height"])
        ox, oy = (size - im["width"] * s) / 2, (size - im["height"] * s) / 2
        x, y, w, h = a["bbox"]
        boxes.setdefault(im["file_name"].replace(".jpg", ""), []).append((x * s + ox, y * s + oy, w * s, h * s))
    return boxes
