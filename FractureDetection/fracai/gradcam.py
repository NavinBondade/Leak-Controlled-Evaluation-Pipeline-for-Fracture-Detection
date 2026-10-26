"""Grad-CAM for the ConvNeXt fracture logit."""

import numpy as np
import torch
import torch.nn.functional as F


def gradcam(model, x, size=512):
    """x: (1,1,H,W) normalised tensor on the model's device. Returns (cam (size,size) in [0,1], logit)."""
    model.eval()
    feats = {}
    layer = model.stages[-1]
    h = layer.register_forward_hook(lambda m, i, o: feats.__setitem__("a", o))
    try:
        with torch.enable_grad():
            logit = model(x).squeeze()
            a = feats["a"]
            g = torch.autograd.grad(logit, a)[0]
    finally:
        h.remove()
    w = g.mean(dim=(2, 3), keepdim=True)
    cam = F.relu((w * a).sum(1, keepdim=True))
    cam = F.interpolate(cam, size=(size, size), mode="bilinear", align_corners=False)[0, 0]
    cam = cam - cam.min()
    cam = cam / cam.max().clamp_min(1e-8)
    return cam.detach().cpu().numpy(), float(logit)


def pointing_hit(cam, boxes, margin=16):
    """True if the CAM's peak lies inside any annotated box (grown by `margin` px)."""
    y, x = np.unravel_index(int(np.argmax(cam)), cam.shape)
    return any(
        bx - margin <= x <= bx + bw + margin and by - margin <= y <= by + bh + margin for bx, by, bw, bh in boxes
    )


def layercam(model, x, stage=2, size=512):
    """Finer-grained saliency from an earlier stage (stride 16 -> 32x32 grid at 512 px).
    LayerCAM: activations weighted element-wise by the positive part of their gradients."""
    model.eval()
    feats = {}
    h = model.stages[stage].register_forward_hook(lambda m, i, o: feats.__setitem__("a", o))
    try:
        with torch.enable_grad():
            logit = model(x).squeeze()
            a = feats["a"]
            g = torch.autograd.grad(logit, a)[0]
    finally:
        h.remove()
    cam = F.relu((F.relu(g) * a).sum(1, keepdim=True))
    cam = F.interpolate(cam, size=(size, size), mode="bilinear", align_corners=False)[0, 0]
    cam = cam - cam.min()
    cam = cam / cam.max().clamp_min(1e-8)
    return cam.detach().cpu().numpy(), float(logit)
