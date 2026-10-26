"""Download the pretrained backbones to weights/. Run once while online."""
import sys
from pathlib import Path

import timm
import torch

out = Path(__file__).resolve().parents[2] / "weights"
out.mkdir(exist_ok=True)
for arch in sys.argv[1:] or ["convnext_tiny"]:
    m = timm.create_model(arch, pretrained=True, num_classes=0, in_chans=1)
    torch.save(m.state_dict(), out / f"{arch}.pth")
    print("saved", out / f"{arch}.pth", f"({sum(p.numel() for p in m.parameters()) / 1e6:.1f}M params)")
