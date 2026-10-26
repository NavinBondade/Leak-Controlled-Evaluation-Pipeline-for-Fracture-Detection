"""Join the single-reader blind visual annotations of the 180 contact-sheet images to their set and metadata, and tabulate.
Annotations were made by reading results/audit2/blind/page_XX.png (codes only, set and score hidden), plus zoomed
re-reads of ambiguous tiles. Codes: C = cast or splint/slab, M = metal (Mi internal hardware, Me external object such as
jewellery, watch, clip, wire), B = bandage/dressing, A = annotation text or side marker. Unlisted codes = none seen.
Outputs: results/audit2/C_annotations.csv, results/audit2/C_counts.csv
"""

import pandas as pd

ANN = {  # code: (labels, note)
    "B005": ("Me", "ring/bangle at wrist"),
    "B006": ("Me", "small metal object over neck/chest"),
    "B010": ("C", "foot and ankle in cast"),
    "B013": ("C", "long-leg cast"),
    "B024": ("Me", "bright object at wrist, likely watch/bangle"),
    "B043": ("Me", "bangle at wrist"),
    "B044": ("Me", "bangle at wrist"),
    "B050": ("C", "arm in posterior slab/splint"),
    "B059": ("A", "side marker with text"),
    "B061": ("Me", "wire/necklace loop over chest"),
    "B064": ("Me", "bangle"),
    "B072": ("Me", "necklace"),
    "B077": ("Me", "metal clip near shoulder (uncertain)"),
    "B078": ("C", "foot in cast"),
    "B079": ("A", "side marker with text"),
    "B082": ("C", "ankle/foot in cast"),
    "B104": ("Me", "several bangles at wrist"),
    "B106": ("Me", "tube/wire at edge (uncertain)"),
    "B111": ("Me", "bangle at wrist"),
    "B113": ("C", "forearm slab/splint in both views; distal radius looks possibly fractured"),
    "B116": ("Me", "watch/ring on holder's hand"),
    "B121": ("Me", "small metal clips near chest"),
    "B129": ("Me", "bangle"),
    "B137": ("Me", "small rings at edge"),
    "B140": ("C;Me", "upper-limb slab/splint plus bangle"),
    "B144": ("Me", "metal at wrist / holder's hand"),
    "B149": ("Me", "watch-like metal object"),
    "B163": ("A", "side markers"),
    "B164": ("A;Mi", "side markers plus internal screw/wire at distal tibia"),
    "B170": ("Me", "bracelet at wrist"),
    "B176": ("Me", "rings/jewellery near hand"),
    "B179": ("C", "distal limb in dense cast (moderate confidence)"),
    "B180": ("Mi", "hip screws"),
}
ANN_FX = {  # supplementary random 60 fractured (cast_contact_sheets_fxrand.py), read the same way
    "F001": ("C", "forearm slab (same image as B113)"),
    "F003": ("C", "forearm cast/slab"),
    "F004": ("C", "lower-leg cast (moderate confidence)"),
    "F008": ("C", "forearm slab"),
    "F011": ("Mi", "ankle plate and screws"),
    "F015": ("C;Mi", "long-arm slab plus K-wires"),
    "F016": ("Mi", "femoral rod/nail"),
    "F017": ("B", "soft wrapping/padding around mid-forearm"),
    "F018": ("C", "lower-leg cast"),
    "F019": ("C", "forearm cast"),
    "F021": ("C", "forearm/hand cast"),
    "F026": ("Mi", "humeral plate"),
    "F027": ("C;Me", "same image as B140"),
    "F028": ("C", "lower-leg cast"),
    "F029": ("Mi", "tibial plate"),
    "F030": ("A;Me", "side marker; bangle"),
    "F032": ("Mi", "fixation wires at elbow"),
    "F034": ("C", "forearm cast"),
    "F035": ("Mi", "humeral nail"),
    "F037": ("A", "side markers"),
    "F038": ("Me", "metal object near femur (uncertain)"),
    "F039": ("C", "forearm/hand casts, both views"),
    "F043": ("C", "forearm casts, both views"),
    "F044": ("Mi;Me", "hip plate; round external object"),
    "F047": ("C", "lower-leg cast"),
    "F048": ("A", "side marker"),
    "F055": ("Mi;A", "external fixator; side marker"),
    "F056": ("C", "lower-leg cast"),
    "F060": ("C", "forearm casts, both views"),
}
ANN.update(ANN_FX)

key = pd.concat(
    [pd.read_csv("results/audit2/blind_key.csv"), pd.read_csv("results/audit2/blind_key_fx.csv")], ignore_index=True
)
key["labels"] = key.code.map(lambda c: ANN.get(c, ("", ""))[0])
key["note"] = key.code.map(lambda c: ANN.get(c, ("", ""))[1])
lab = key.labels.str.split(";")
key["cast_splint"] = lab.map(lambda tags: "C" in tags)
key["metal_internal"] = lab.map(lambda tags: "Mi" in tags)
key["metal_external"] = lab.map(lambda tags: "Me" in tags)
key["metal_any"] = key.metal_internal | key.metal_external
key["bandage"] = lab.map(lambda tags: "B" in tags)
key["annotation"] = lab.map(lambda tags: "A" in tags)
key["cast_splint_or_hardware"] = key.cast_splint | key.metal_internal
key["none"] = key.labels == ""
key.sort_values(["set", "score"]).to_csv("results/audit2/C_annotations.csv", index=False)

cols = [
    "cast_splint",
    "metal_internal",
    "metal_external",
    "metal_any",
    "bandage",
    "annotation",
    "cast_splint_or_hardware",
    "none",
    "hardware",
]
counts = key.groupby("set")[cols].sum().astype(int)
counts.insert(0, "n", key.groupby("set").size())
counts.to_csv("results/audit2/C_counts.csv")
print(counts.T)
fa = key[key.set == "fa_top60"].sort_values("score", ascending=False)
for n in (10, 20, 60):
    t = fa.head(n)
    print(
        f"top{n} FA: cast/splint {t.cast_splint.sum()}, cast/splint/internal hw {t.cast_splint_or_hardware.sum()}, any metal-or-cast {(t.cast_splint | t.metal_any).sum()}"
    )
print(
    key[key.labels != ""][["code", "set", "case_id", "score", "bodypart", "hardware", "labels", "note"]]
    .sort_values("set")
    .to_string()
)

from scipy.stats import fisher_exact


def fe(a, b, col):
    x, y = key[key.set == a][col].sum(), key[key.set == b][col].sum()
    return int(x), int(y), fisher_exact([[x, 60 - x], [y, 60 - y]])[1]


for col in ["cast_splint", "cast_splint_or_hardware", "metal_any", "metal_external"]:
    print(col, "FA vs TN", fe("fa_top60", "tn_rand60", col), "| FXrand vs TN", fe("fx_rand60", "tn_rand60", col))
fx = key[key.set == "fx_rand60"]
print("fx_rand60 metadata hardware vs read internal hardware:", pd.crosstab(fx.hardware, fx.metal_internal))
print("fx_rand60 metadata hardware vs read cast:", pd.crosstab(fx.hardware, fx.cast_splint))
print(
    "fx_rand60 detected (score>=0) by cast:",
    fx.groupby("cast_splint").score.apply(lambda v: f"{(v >= 0).mean():.2f} of {len(v)}").to_dict(),
)
