#!/usr/bin/env python
"""Controlled color, occlusion, and absence contrasts for a per-token transcoder.

The task object stands in for the apple. At a frozen arm state, language
prompt, and noise seed, each condition changes one thing:

  base              original render
  recolor           object color only. Pixels that change are the object mask.
  absent            object teleported away; arm qpos unchanged
  occluded          gray paint on half of those pixels; every other pixel matches base
  slab_miss         the same gray shape, translated off the object
  occluded_absent   the occluded disk painted on the absent render

A TopK transcoder is fit on layer-5 tokens and, when the module exists, predicts
layer-6 tokens. Circuit tracing patches only the contrast-specific features and
compares them with norm-matched controls, plus full-token, pooled, and
token-structure patches on the same pairs.

Run inside the Colab runtime that already has scripts/collect_layer5_replay.py.
"""

from __future__ import annotations

import json
import os
import random
from pathlib import Path
from typing import Literal, assert_never

import cv2
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

CONDITION_NAMES = (
    "base",
    "recolor",
    "absent",
    "occluded",
    "slab_miss",
    "occluded_absent",
)
CONTRAST_NAMES = ("color", "absence", "occlusion")
ContrastName = Literal["color", "absence", "occlusion"]
GRAY = (128, 128, 128)
GREEN = (20, 170, 40)
RED = (210, 30, 30)


def seed_all(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def changed_pixels(before: np.ndarray, after: np.ndarray) -> np.ndarray:
    return np.any(before.astype(np.int16) != after.astype(np.int16), axis=-1)


def changed_fraction(before: np.ndarray, after: np.ndarray) -> float:
    return float(changed_pixels(before, after).mean())


def pixel_rmse(before: np.ndarray, after: np.ndarray) -> float:
    delta = before.astype(np.float32) - after.astype(np.float32)
    return float(np.sqrt(np.mean(delta * delta)))


def outside_fraction(before: np.ndarray, after: np.ndarray, mask: np.ndarray) -> float:
    changed = changed_pixels(before, after)
    return float(np.logical_and(changed, ~mask).mean())


def paint_mask(image: np.ndarray, mask: np.ndarray, color: tuple[int, int, int]) -> np.ndarray:
    painted = image.copy()
    painted[mask] = color
    return painted


def disk_mask(height: int, width: int, cx: float, cy: float, radius: float) -> np.ndarray:
    yy, xx = np.ogrid[:height, :width]
    return (xx - cx) ** 2 + (yy - cy) ** 2 <= radius ** 2


def dilate_mask(mask: np.ndarray, radius: int) -> np.ndarray:
    out = mask.copy()
    height, width = mask.shape
    for dy in range(-radius, radius + 1):
        for dx in range(-radius, radius + 1):
            if dx * dx + dy * dy > radius * radius:
                continue
            src_y0, src_y1 = max(0, -dy), min(height, height - dy)
            src_x0, src_x1 = max(0, -dx), min(width, width - dx)
            dst_y0, dst_x0 = max(0, dy), max(0, dx)
            dst_y1 = dst_y0 + (src_y1 - src_y0)
            dst_x1 = dst_x0 + (src_x1 - src_x0)
            out[dst_y0:dst_y1, dst_x0:dst_x1] |= mask[src_y0:src_y1, src_x0:src_x1]
    return out


def half_mask(full: np.ndarray) -> np.ndarray:
    ys, xs = np.nonzero(full)
    if len(xs) == 0:
        return full.copy()
    cx = float(np.median(xs))
    xx = np.arange(full.shape[1], dtype=np.float64)[None, :]
    left = full & (xx < cx)
    if int(left.sum()) < 10:
        left = full & (xx <= cx)
    return left


def shift_mask(mask: np.ndarray, dy: int, dx: int) -> np.ndarray:
    out = np.zeros_like(mask)
    height, width = mask.shape
    src_y0, src_y1 = max(0, -dy), min(height, height - dy)
    src_x0, src_x1 = max(0, -dx), min(width, width - dx)
    dst_y0, dst_x0 = max(0, dy), max(0, dx)
    dst_y1 = dst_y0 + (src_y1 - src_y0)
    dst_x1 = dst_x0 + (src_x1 - src_x0)
    out[dst_y0:dst_y1, dst_x0:dst_x1] = mask[src_y0:src_y1, src_x0:src_x1]
    return out


def translated_copy(mask: np.ndarray, avoid: np.ndarray) -> np.ndarray | None:
    """Copy `mask` to a corner. The copy keeps every pixel and misses `avoid`."""
    ys, xs = np.nonzero(mask)
    if len(xs) == 0:
        return None
    height, width = mask.shape
    box_h = int(ys.max() - ys.min())
    box_w = int(xs.max() - xs.min())
    dilated = dilate_mask(avoid, 4)
    corners = (
        (2, 2),
        (2, width - box_w - 3),
        (height - box_h - 3, 2),
        (height - box_h - 3, width - box_w - 3),
    )
    for top, left in corners:
        if top < 0 or left < 0:
            continue
        moved = shift_mask(mask, int(top - ys.min()), int(left - xs.min()))
        if int(moved.sum()) != int(mask.sum()):
            continue
        if np.logical_and(moved, dilated).any():
            continue
        return moved
    return None


def union_disks(height: int, width: int, disks: list[tuple[float, float, float]]) -> np.ndarray:
    mask = np.zeros((height, width), dtype=bool)
    for cx, cy, radius in disks:
        mask |= disk_mask(height, width, cx, cy, radius)
    return mask


def world_to_camera(xpos: np.ndarray, xmat: np.ndarray, point: np.ndarray) -> np.ndarray:
    rotation = np.asarray(xmat, dtype=np.float64).reshape(3, 3)
    return rotation.T @ (np.asarray(point, dtype=np.float64) - np.asarray(xpos, dtype=np.float64))


def project_point(
    xpos: np.ndarray,
    xmat: np.ndarray,
    fovy_deg: float,
    point: np.ndarray,
    height: int,
    width: int,
    flip180: bool,
) -> tuple[float, float, float] | None:
    """Project a world point. Returns (u, v, depth) in the image the policy sees."""
    camera = world_to_camera(xpos, xmat, point)
    depth = float(-camera[2])
    if depth <= 1e-6:
        return None
    fovy = np.deg2rad(float(fovy_deg))
    fy = 0.5 * height / np.tan(0.5 * fovy)
    fx = fy
    u = fx * (camera[0] / depth) + width / 2.0
    v = fy * (-camera[1] / depth) + height / 2.0
    if flip180:
        u = (width - 1) - u
        v = (height - 1) - v
    return float(u), float(v), depth


def pixel_radius(fovy_deg: float, height: int, depth: float, rbound: float) -> float:
    fovy = np.deg2rad(float(fovy_deg))
    fy = 0.5 * height / np.tan(0.5 * fovy)
    return float(fy * rbound / max(depth, 1e-6))


def swapped_rgba(current: np.ndarray) -> np.ndarray:
    """Red-dominant paint becomes green. Green-dominant paint becomes red."""
    rgb = np.asarray(current, dtype=np.float64)[:, :3].mean(axis=0)
    alpha = np.asarray(current, dtype=np.float32)[:, 3:4]
    if rgb[0] >= rgb[1]:
        color = np.array([0.05, 0.72, 0.12], dtype=np.float32)
    else:
        color = np.array([0.82, 0.08, 0.08], dtype=np.float32)
    tiled = np.repeat(color[None, :], len(current), axis=0)
    return np.concatenate([tiled, alpha], axis=1)


def mean_object_color(image: np.ndarray, mask: np.ndarray) -> np.ndarray:
    pixels = image[mask]
    if len(pixels) == 0:
        return np.zeros(3, dtype=np.float64)
    return pixels.astype(np.float64).mean(axis=0)


def pixel_recolor(image: np.ndarray, mask: np.ndarray) -> np.ndarray:
    rgb = mean_object_color(image, mask)
    color = GREEN if rgb[0] >= rgb[1] else RED
    return paint_mask(image, mask, color)


def action_array(chunk) -> np.ndarray:
    if isinstance(chunk, (tuple, list)):
        chunk = chunk[0]
    if isinstance(chunk, dict):
        chunk = chunk.get("action", next(iter(chunk.values())))
    action = chunk.detach().float().cpu().numpy()
    if action.ndim == 3:
        if action.shape[0] != 1:
            raise RuntimeError(f"Expected a batch of 1, got {action.shape}")
        action = action[0]
    if action.ndim != 2 or action.shape[0] < 2:
        raise RuntimeError("Need a full action chunk, not a single action.")
    return np.asarray(action, dtype=np.float32)


def action_metrics(edited: np.ndarray, source: np.ndarray, donor: np.ndarray) -> dict:
    edited64 = edited.astype(np.float64)
    source64 = source.astype(np.float64)
    donor64 = donor.astype(np.float64)
    baseline = float(np.sqrt(np.mean((source64 - donor64) ** 2)))
    error = float(np.sqrt(np.mean((edited64 - donor64) ** 2)))
    gap = None if baseline <= 1e-4 else (baseline - error) / baseline
    return {
        "source_to_donor_rmse": baseline,
        "edited_to_donor_rmse": error,
        "fraction_gap_closed": gap,
    }


def structure_delta(donor: np.ndarray, source: np.ndarray) -> np.ndarray:
    """Remove the token-mean from a [tokens, dim] difference."""
    delta = donor.astype(np.float32) - source.astype(np.float32)
    return delta - delta.mean(axis=0, keepdims=True)


def moving_token_mask(
    base_tokens: np.ndarray, edited_tokens: np.ndarray, quantile: float = 0.85
) -> np.ndarray:
    delta = np.linalg.norm(
        edited_tokens.astype(np.float32) - base_tokens.astype(np.float32), axis=-1
    )
    if not np.isfinite(delta).all():
        raise RuntimeError("Nonfinite token delta.")
    cut = float(np.quantile(delta, quantile))
    mask = delta >= cut
    if int(mask.sum()) < 8:
        mask = np.zeros(len(delta), dtype=bool)
        mask[np.argsort(-delta)[:8]] = True
    return mask


def contrast_score(base_code: np.ndarray, edited_code: np.ndarray, mask: np.ndarray) -> np.ndarray:
    return (edited_code[mask] - base_code[mask]).mean(axis=0)


def pick_features(
    primary: np.ndarray,
    others: list[np.ndarray],
    k: int,
    min_specificity: float = 0.5,
) -> tuple[np.ndarray, bool]:
    primary64 = np.asarray(primary, dtype=np.float64)
    other64 = [np.asarray(other, dtype=np.float64) for other in others]
    denominator = np.abs(primary64) + sum(np.abs(other) for other in other64) + 1e-8
    specificity = np.abs(primary64) / denominator
    magnitude_cut = float(np.quantile(np.abs(primary64), 0.90))
    eligible = np.flatnonzero(
        (specificity > min_specificity) & (np.abs(primary64) >= magnitude_cut)
    )
    specific = True
    if eligible.size < min(4, k):
        specific = False
        eligible = np.arange(primary64.size)
    order = eligible[np.argsort(-np.abs(primary64[eligible]))]
    return order[:k].astype(np.int64), specific


def off_contrast(name: ContrastName) -> ContrastName:
    match name:
        case "color":
            return "absence"
        case "absence":
            return "color"
        case "occlusion":
            return "color"
        case _ as unexpected:
            assert_never(unexpected)


def remap_features(code_delta: np.ndarray, src_ids: np.ndarray, dst_ids: np.ndarray) -> np.ndarray:
    remapped = np.zeros_like(code_delta)
    remapped[:, dst_ids] = code_delta[:, src_ids]
    return remapped


def keep_features(code_delta: np.ndarray, feature_ids: np.ndarray) -> np.ndarray:
    kept = np.zeros_like(code_delta)
    kept[:, feature_ids] = code_delta[:, feature_ids]
    return kept


def match_l2(delta: np.ndarray, reference: np.ndarray, eps: float = 1e-6) -> np.ndarray:
    ref = float(np.linalg.norm(reference))
    cur = float(np.linalg.norm(delta))
    if cur < eps or ref < eps:
        return delta
    return delta * (ref / cur)


def mean_std(chunks: list[np.ndarray]) -> tuple[np.ndarray, np.ndarray]:
    count = 0
    total = None
    for chunk in chunks:
        flat = chunk.reshape(-1, chunk.shape[-1]).astype(np.float64)
        total = flat.sum(axis=0) if total is None else total + flat.sum(axis=0)
        count += len(flat)
    if total is None or count < 2:
        raise RuntimeError("Not enough tokens to fit the transcoder.")
    mean = total / count
    var = np.zeros_like(mean)
    for chunk in chunks:
        flat = chunk.reshape(-1, chunk.shape[-1]).astype(np.float64)
        var += ((flat - mean) ** 2).sum(axis=0)
    std = np.sqrt(var / (count - 1)).clip(min=1e-6)
    return mean.astype(np.float32), std.astype(np.float32)


def r2_score(pred: np.ndarray, target: np.ndarray) -> float:
    resid = float(np.mean((pred - target) ** 2))
    var = float(np.mean((target - target.mean(axis=0)) ** 2))
    if var < 1e-12:
        return 0.0
    return 1.0 - resid / var


def cosine(a: np.ndarray, b: np.ndarray) -> float:
    denom = float(np.linalg.norm(a) * np.linalg.norm(b))
    if denom < 1e-12:
        return 0.0
    return float(np.dot(a, b) / denom)


def evenly(items: list, count: int) -> list:
    if len(items) <= count:
        return list(items)
    indexes = np.linspace(0, len(items) - 1, count).round().astype(int)
    chosen = []
    seen: set[int] = set()
    for index in indexes:
        key = int(index)
        if key not in seen:
            seen.add(key)
            chosen.append(items[key])
    return chosen


def choose_frames(records: list[dict], count: int) -> list[dict]:
    held = [row for row in records if str(row["label"]).startswith("held_")]
    rest = [row for row in records if not str(row["label"]).startswith("held_")]
    picked = evenly(held, count)
    if len(picked) < count:
        picked.extend(evenly(rest, count - len(picked)))
    return picked


def full_cover_interpretation(outside: float, action_rmse: float) -> str:
    """Describe the full-cover diagnostic.

    Images agree when fewer than 0.2% of pixels still differ. Action chunks
    agree below an RMSE of 1e-3. A larger RMSE on agreeing images is the
    residual floor for near-matched inputs. The identical-input re-forward is
    the determinism check, and this diagnostic does not replace it.
    """
    images_agree = outside < 0.002
    actions_agree = action_rmse < 1e-3
    if images_agree and actions_agree:
        return (
            "Full cover makes the present and absent images agree, and the action chunks agree. "
            "A single forward has no remaining evidence the object is behind the cover."
        )
    if images_agree:
        return (
            f"The covered images agree, and the action chunks differ by RMSE {action_rmse:.5f}. "
            "The identical-input re-forward already matched, so this gap is the residual floor "
            "for near-matched images. Read later gap-closed numbers against this floor."
        )
    return (
        "Pixels outside the cover still differ between present and absent. "
        "The partial-occlusion contrast and the absence contrast are the controlled comparisons."
    )


class TokenTranscoder(nn.Module):
    """Per-token TopK transcoder from layer 5, optionally predicting layer 6."""

    def __init__(self, dim: int, n_features: int, k: int, predict_next: bool) -> None:
        super().__init__()
        if k >= n_features:
            raise ValueError("k must be smaller than n_features")
        self.k = k
        self.predict_next = predict_next
        self.encoder = nn.Linear(dim, n_features)
        self.decoder = nn.Linear(n_features, dim, bias=False)
        if predict_next:
            self.next_decoder = nn.Linear(n_features, dim, bias=False)
        self.reset_parameters()

    def reset_parameters(self) -> None:
        nn.init.kaiming_uniform_(self.encoder.weight, a=5**0.5)
        nn.init.zeros_(self.encoder.bias)
        with torch.no_grad():
            decoder = self.encoder.weight.detach().T
            decoder = decoder / decoder.norm(dim=0, keepdim=True).clamp_min(1e-8)
            self.decoder.weight.copy_(decoder)
            if self.predict_next:
                self.next_decoder.weight.copy_(decoder)

    def encode(self, normalized: torch.Tensor) -> torch.Tensor:
        preactivations = self.encoder(normalized)
        values, indices = preactivations.topk(self.k, dim=-1)
        code = torch.zeros_like(preactivations)
        code.scatter_(-1, indices, F.relu(values))
        return code

    def forward(self, normalized: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor | None, torch.Tensor]:
        code = self.encode(normalized)
        recon = self.decoder(code)
        nxt = self.next_decoder(code) if self.predict_next else None
        return recon, nxt, code

    def renorm_decoder(self) -> None:
        with torch.no_grad():
            norms = self.decoder.weight.norm(dim=0).clamp_min(1e-8)
            self.decoder.weight.div_(norms)
            self.encoder.weight.mul_(norms[:, None])


def train_transcoder(
    layer5: list[np.ndarray],
    layer6: list[np.ndarray] | None,
    n_features: int,
    k: int,
    steps: int,
    batch: int,
    seed: int,
) -> tuple[TokenTranscoder, np.ndarray, np.ndarray, np.ndarray | None, np.ndarray | None]:
    mean5, std5 = mean_std(layer5)
    x = np.concatenate([chunk.reshape(-1, chunk.shape[-1]) for chunk in layer5], axis=0)
    x_norm = ((x - mean5) / std5).astype(np.float32)
    predict_next = layer6 is not None
    y_norm = None
    mean6 = None
    std6 = None
    if layer6 is not None:
        mean6, std6 = mean_std(layer6)
        y = np.concatenate([chunk.reshape(-1, chunk.shape[-1]) for chunk in layer6], axis=0)
        y_norm = ((y - mean6) / std6).astype(np.float32)
        if len(y_norm) != len(x_norm):
            raise RuntimeError("Layer 5 and layer 6 token counts differ.")
    model = TokenTranscoder(x_norm.shape[1], n_features, k, predict_next)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    rng = np.random.default_rng(seed)
    model.train()
    for step in range(steps):
        take = min(batch, len(x_norm))
        idx = rng.integers(0, len(x_norm), size=take)
        xb = torch.from_numpy(np.ascontiguousarray(x_norm[idx]))
        optimizer.zero_grad(set_to_none=True)
        recon, nxt, _code = model(xb)
        loss = F.mse_loss(recon, xb)
        if nxt is not None and y_norm is not None:
            yb = torch.from_numpy(np.ascontiguousarray(y_norm[idx]))
            loss = loss + F.mse_loss(nxt, yb)
        loss.backward()
        optimizer.step()
        model.renorm_decoder()
        if step == 0 or (step + 1) % 100 == 0 or step + 1 == steps:
            print(f"  transcoder step {step + 1}/{steps} loss={float(loss.detach()):.5f}", flush=True)
    model.eval()
    return model, mean5, std5, mean6, std6


def encode_tokens(model: TokenTranscoder, tokens: np.ndarray, mean: np.ndarray, std: np.ndarray) -> np.ndarray:
    normalized = (tokens.astype(np.float32) - mean) / std
    with torch.no_grad():
        code = model.encode(torch.from_numpy(np.ascontiguousarray(normalized)))
    return code.numpy()


def decode_delta(model: TokenTranscoder, code_delta: np.ndarray, std: np.ndarray) -> np.ndarray:
    with torch.no_grad():
        recon = model.decoder(torch.from_numpy(np.ascontiguousarray(code_delta.astype(np.float32))))
        raw = recon * torch.from_numpy(std.astype(np.float32))
    return raw.numpy().astype(np.float32)


def reconstruction(model: TokenTranscoder, tokens: np.ndarray, mean: np.ndarray, std: np.ndarray) -> np.ndarray:
    code = encode_tokens(model, tokens, mean, std)
    with torch.no_grad():
        recon = model.decoder(torch.from_numpy(code))
        raw = recon * torch.from_numpy(std) + torch.from_numpy(mean)
    return raw.numpy().astype(np.float32)


def next_prediction(
    model: TokenTranscoder,
    tokens: np.ndarray,
    mean5: np.ndarray,
    std5: np.ndarray,
    mean6: np.ndarray,
    std6: np.ndarray,
) -> np.ndarray:
    code = encode_tokens(model, tokens, mean5, std5)
    with torch.no_grad():
        pred = model.next_decoder(torch.from_numpy(code))
        raw = pred * torch.from_numpy(std6) + torch.from_numpy(mean6)
    return raw.numpy().astype(np.float32)


def load_collect_module():
    # Imported here so unit tests can load this module without LIBERO, and so
    # MUJOCO_GL / MAX_STEPS are set before collect_layer5_replay reads them.
    import collect_layer5_replay as collect

    return collect


def env_int(name: str, default: int) -> int:
    return int(os.environ.get(name, str(default)))


def configure_environment() -> None:
    os.environ.setdefault("MUJOCO_GL", "egl")
    os.environ.setdefault("PYOPENGL_PLATFORM", "egl")
    os.environ.setdefault("MPLBACKEND", "Agg")
    os.environ.setdefault("SUITE", "libero_spatial")
    os.environ.setdefault("TASK_IDS", "[0]")
    os.environ.setdefault(
        "LAYER_NAME",
        "model.paligemma_with_expert.paligemma.model.language_model.layers.5",
    )
    # These two override any earlier collection cell so this run stays on the short controlled set.
    os.environ["MAX_STEPS"] = os.environ.get("CTRL_MAX_STEPS", "24")
    os.environ["MAX_EPISODES"] = os.environ.get("CTRL_MAX_EPISODES", "2")
    os.environ.setdefault("LOCAL_REPO", "/content/groot-run")


def unpack_output(output):
    tensor = output[0] if isinstance(output, (tuple, list)) else output
    if not torch.is_tensor(tensor) or tensor.ndim != 3 or tensor.shape[0] != 1:
        shape = tuple(tensor.shape) if torch.is_tensor(tensor) else type(tensor)
        raise RuntimeError(f"Expected layer output [1, tokens, dim], got {shape}")
    return tensor


def repack_output(output, edited):
    if isinstance(output, tuple):
        return (edited,) + output[1:]
    if isinstance(output, list):
        return [edited] + output[1:]
    return edited


def camera_index(sim, needle: str) -> int | None:
    model = sim.model
    for index in range(int(model.ncam)):
        name = model.camera_id2name(index) or ""
        if needle in name:
            return index
    return None


def project_geoms(sim, geoms: list[int], cam: int, height: int, width: int) -> list[tuple[float, float, float]]:
    data = sim.data
    model = sim.model
    xpos = np.asarray(data.cam_xpos[cam], dtype=np.float64)
    xmat = np.asarray(data.cam_xmat[cam], dtype=np.float64)
    fovy = float(model.cam_fovy[cam])
    disks = []
    for geom in geoms:
        point = np.asarray(data.geom_xpos[geom], dtype=np.float64)
        projected = project_point(xpos, xmat, fovy, point, height, width, flip180=True)
        if projected is None:
            continue
        u, v, depth = projected
        radius = pixel_radius(fovy, height, depth, float(model.geom_rbound[geom]))
        if radius < 6 or u < -radius or v < -radius or u >= width + radius or v >= height + radius:
            continue
        disks.append((u, v, radius))
    return disks


def masks_from_disks(height: int, width: int, disks: list[tuple[float, float, float]]):
    if not disks:
        return None, None, None
    full = union_disks(height, width, disks)
    if int(full.sum()) < 80:
        return None, None, None
    radius = float(np.median([disk[2] for disk in disks]))
    if radius > 0.45 * min(height, width):
        return None, None, None
    partial = half_mask(full)
    if int(partial.sum()) < 40:
        return None, None, None
    control = translated_copy(partial, full)
    if control is None:
        return None, None, None
    return full, partial, control


def images_of(collect, raw: dict) -> tuple[np.ndarray, np.ndarray]:
    agent = collect.image_hwc(raw, "agentview_image", "agentview_rgb")
    wrist = collect.image_hwc(
        raw, "robot0_eye_in_hand_image", "eye_in_hand_rgb", "robot0_eye_in_hand_rgb"
    )
    return agent, wrist


def write_rgba(sim, geoms: list[int], rgba: np.ndarray) -> None:
    for index, geom in enumerate(geoms):
        sim.model.geom_rgba[geom, :] = rgba[index]


def render_rgba_images(collect, env, sim, geoms: list[int]):
    current = np.array(sim.model.geom_rgba[geoms], dtype=np.float32).copy()
    edited = swapped_rgba(current)
    write_rgba(sim, geoms, edited)
    sim.forward()
    try:
        return images_of(collect, collect.render_after_edit(env))
    finally:
        write_rgba(sim, geoms, current)
        sim.forward()


def masks_from_change(before: np.ndarray, after: np.ndarray):
    """Object mask is the set of pixels the color swap actually changed."""
    changed = changed_pixels(before, after)
    count = int(changed.sum())
    fraction = float(changed.mean())
    if count < 80 or fraction < 0.001 or fraction > 0.2:
        return None
    partial = half_mask(changed)
    if int(partial.sum()) < 40:
        return None
    control = translated_copy(partial, changed)
    if control is None:
        return None
    return changed, partial, control


def pixel_recolor_wrist(wrist: np.ndarray, mask: np.ndarray | None) -> np.ndarray:
    if mask is None:
        return wrist.copy()
    return pixel_recolor(wrist, mask)


def arm_unchanged(saved: np.ndarray, sim, span: tuple[int, int]) -> bool:
    start, stop = span
    before = np.concatenate([saved[:start], saved[stop:]])
    after = np.concatenate([np.asarray(sim.data.qpos[:start]), np.asarray(sim.data.qpos[stop:])])
    return bool(np.allclose(before, after, atol=1e-5))


def render_absent(collect, env, sim, body: int):
    span = collect.object_qpos_span(sim, body)
    saved = collect.move_object_away(sim, body)
    if saved is None or span is None:
        return None
    try:
        if not arm_unchanged(saved, sim, span):
            return None
        return images_of(collect, collect.render_after_edit(env))
    finally:
        collect.restore_qpos(sim, saved)


def paint_pair(agent, wrist, agent_mask, wrist_mask, color):
    agent_out = paint_mask(agent, agent_mask, color)
    wrist_out = wrist.copy() if wrist_mask is None else paint_mask(wrist, wrist_mask, color)
    return agent_out, wrist_out


def projected_masks(sim, geoms: list[int], image: np.ndarray, needle: str):
    cam = camera_index(sim, needle)
    if cam is None:
        return None
    disks = project_geoms(sim, geoms, cam, image.shape[0], image.shape[1])
    full, partial, control = masks_from_disks(image.shape[0], image.shape[1], disks)
    if full is None or partial is None or control is None:
        return None
    return full, partial, control


def build_conditions(collect, env, sim, body: int, geoms: list[int], raw: dict) -> dict | None:
    base_agent, base_wrist = images_of(collect, raw)
    rgba_agent, rgba_wrist = render_rgba_images(collect, env, sim, geoms)
    empirical = masks_from_change(base_agent, rgba_agent)
    wrist_empirical = masks_from_change(base_wrist, rgba_wrist)
    if empirical is not None:
        full, partial, control = empirical
        recolor_agent, recolor_wrist = rgba_agent, rgba_wrist
        recolor_method = "rgba"
        if wrist_empirical is not None:
            wrist_full, wrist_partial, wrist_control = wrist_empirical
        else:
            wrist_full = wrist_partial = wrist_control = None
    else:
        projected = projected_masks(sim, geoms, base_agent, "agentview")
        if projected is None:
            return None
        full, partial, control = projected
        wrist_projected = projected_masks(sim, geoms, base_wrist, "eye_in_hand")
        if wrist_projected is None:
            wrist_full = wrist_partial = wrist_control = None
        else:
            wrist_full, wrist_partial, wrist_control = wrist_projected
        recolor_agent = pixel_recolor(base_agent, full)
        recolor_wrist = pixel_recolor_wrist(base_wrist, wrist_full)
        recolor_method = "pixel_disk"
    absent = render_absent(collect, env, sim, body)
    if absent is None:
        return None
    absent_agent, absent_wrist = absent
    occluded = paint_pair(base_agent, base_wrist, partial, wrist_partial, GRAY)
    slab = paint_pair(base_agent, base_wrist, control, wrist_control, GRAY)
    occluded_absent = paint_pair(absent_agent, absent_wrist, partial, wrist_partial, GRAY)
    if np.logical_and(partial, control).any():
        raise RuntimeError("Occlusion disk and control disk overlap.")
    if changed_fraction(base_agent, recolor_agent) < 1e-6:
        return None
    if not mask_edit_is_exact(base_agent, occluded[0], partial):
        return None
    if not mask_edit_is_exact(base_agent, slab[0], control):
        return None
    if not mask_edit_is_exact(absent_agent, occluded_absent[0], partial):
        return None
    if wrist_partial is not None:
        mask_edit_is_exact(base_wrist, occluded[1], wrist_partial)
    if wrist_control is not None:
        mask_edit_is_exact(base_wrist, slab[1], wrist_control)
    images = {
        "base": (base_agent, base_wrist),
        "recolor": (recolor_agent, recolor_wrist),
        "absent": (absent_agent, absent_wrist),
        "occluded": occluded,
        "slab_miss": slab,
        "occluded_absent": occluded_absent,
    }
    return {
        "images": images,
        "recolor_method": recolor_method,
        "agent_full": full,
        "agent_partial": partial,
        "wrist_full": wrist_full,
        "wrist_partial": wrist_partial,
        "wrist_painted": wrist_partial is not None,
    }


def mask_edit_is_exact(before: np.ndarray, after: np.ndarray, mask: np.ndarray) -> bool:
    changed = changed_pixels(before, after)
    if np.logical_and(changed, ~mask).any():
        raise RuntimeError("A controlled paint changed pixels outside its mask.")
    return bool(changed.any())


def batch_from_images(collect, pre, agent, wrist, state: np.ndarray, sentence: str, policy):
    batch = {
        "observation.images.image": collect.as_image_tensor(agent),
        "observation.images.image2": collect.as_image_tensor(wrist),
        "observation.state": torch.from_numpy(np.asarray(state, dtype=np.float32)).unsqueeze(0),
        "task": [sentence],
    }
    features = getattr(policy.config, "input_features", {}) or {}
    reference = batch["observation.images.image"]
    for key in features:
        if key.startswith("observation.images.") and key not in batch:
            batch[key] = torch.zeros_like(reference)
    return pre(batch)


def find_layer6(policy, layer5):
    names = [name for name, module in policy.named_modules() if module is layer5]
    if len(names) != 1:
        raise RuntimeError(f"Layer 5 module identity matched {names}")
    layer5_name = names[0]
    named = dict(policy.named_modules())
    candidate = layer5_name.replace(".layers.5", ".layers.6") if layer5_name.endswith(".layers.5") else ""
    if candidate in named:
        print("Hooking layer 6", candidate, flush=True)
        return named[candidate]
    hits = [
        name
        for name in named
        if name.endswith("language_model.layers.6") and "gemma_expert" not in name and "vision" not in name
    ]
    if len(hits) == 1:
        print("Hooking layer 6", hits[0], flush=True)
        return named[hits[0]]
    print("Layer 6 was not found. The transcoder reconstructs layer 5 only.", flush=True)
    return None


def forward_policy(policy, layer5, layer6, batch, noise_seed: int, payload: torch.Tensor | None, mode: str):
    seed_all(noise_seed)
    if hasattr(policy, "reset"):
        policy.reset()
    captured5 = []
    captured6 = []

    def hook5(_module, _inputs, output):
        tensor = unpack_output(output)
        match mode:
            case "base":
                edited = tensor
            case "add":
                if payload is None or payload.shape != tensor.shape:
                    raise RuntimeError("Add-mode payload does not match the layer output.")
                edited = (tensor.float() + payload).to(dtype=tensor.dtype)
            case "replace":
                if payload is None or payload.shape != tensor.shape:
                    raise RuntimeError("Replace-mode payload does not match the layer output.")
                edited = payload.to(device=tensor.device, dtype=tensor.dtype)
            case _ as unexpected:
                assert_never(unexpected)
        captured5.append(tensor.detach())
        return repack_output(output, edited)

    def hook6(_module, _inputs, output):
        captured6.append(unpack_output(output).detach())
        return output

    handle5 = layer5.register_forward_hook(hook5)
    handle6 = layer6.register_forward_hook(hook6) if layer6 is not None else None
    try:
        with torch.inference_mode():
            chunk = policy.predict_action_chunk(batch) if hasattr(policy, "predict_action_chunk") else policy.select_action(batch)
    finally:
        handle5.remove()
        if handle6 is not None:
            handle6.remove()
    if len(captured5) != 1:
        raise RuntimeError(f"Expected one layer-5 call, found {len(captured5)}.")
    tokens6 = None
    if layer6 is not None:
        if len(captured6) != 1:
            raise RuntimeError(f"Expected one layer-6 call, found {len(captured6)}.")
        tokens6 = captured6[0][0].float().cpu().numpy()
    tokens5 = captured5[0][0].float().cpu().numpy()
    return tokens5, tokens6, action_array(chunk)


def to_gpu(payload: np.ndarray, device) -> torch.Tensor:
    tensor = torch.from_numpy(np.ascontiguousarray(payload.astype(np.float32)))
    return tensor.unsqueeze(0).to(device)


def thumb(picture: np.ndarray, label: str) -> np.ndarray:
    tile = cv2.resize(picture, (160, 160), interpolation=cv2.INTER_AREA)
    tile = cv2.copyMakeBorder(tile, 22, 0, 0, 0, cv2.BORDER_CONSTANT, value=(255, 255, 255))
    cv2.putText(tile, label, (4, 16), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 0), 1, cv2.LINE_AA)
    return tile


def write_sheet(rows: list[tuple[str, list[np.ndarray]]], path: Path) -> None:
    strips = []
    for label, frames in rows:
        tiles = [thumb(frame, f"{label[:12]} {index}") for index, frame in enumerate(frames)]
        strips.append(np.concatenate(tiles, axis=1))
    width = max(strip.shape[1] for strip in strips)
    padded = []
    for strip in strips:
        if strip.shape[1] < width:
            pad = np.full((strip.shape[0], width - strip.shape[1], 3), 255, dtype=np.uint8)
            strip = np.concatenate([strip, pad], axis=1)
        padded.append(strip)
    sheet = np.concatenate(padded, axis=0)
    bgr = cv2.cvtColor(sheet, cv2.COLOR_RGB2BGR)
    cv2.imwrite(str(path), bgr)


def json_ready(value):
    if isinstance(value, dict):
        return {str(key): json_ready(val) for key, val in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_ready(val) for val in value]
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.floating):
        return float(value)
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, float) and not np.isfinite(value):
        return None
    return value


def mean_defined(values: list[float | None]) -> float | None:
    kept = [value for value in values if value is not None]
    if not kept:
        return None
    return float(np.mean(kept))


def print_circuit(rows: list[dict]) -> None:
    print("\nCIRCUIT  fraction of action-chunk RMSE closed (base -> edited)", flush=True)
    print(f"{'contrast':<12}{'patch':<24}{'gap_closed':>12}{'edit_l2':>12}{'base_rmse':>12}{'n':>6}", flush=True)
    for row in rows:
        gap = row["mean_gap_closed"]
        gap_text = "undefined" if gap is None else f"{100 * gap:.2f}%"
        print(
            f"{row['contrast']:<12}{row['patch']:<24}{gap_text:>12}"
            f"{row['mean_edit_l2']:>12.3f}{row['mean_baseline_rmse']:>12.4f}{row['n_defined']:>6}",
            flush=True,
        )


def feature_table(scores: dict[str, np.ndarray], picks: dict[str, np.ndarray]) -> list[dict]:
    rows = []
    for contrast, ids in picks.items():
        others = [scores[name] for name in CONTRAST_NAMES if name != contrast]
        denominator = np.abs(scores[contrast]) + sum(np.abs(other) for other in others) + 1e-8
        specificity = np.abs(scores[contrast]) / denominator
        for feature in ids:
            index = int(feature)
            rows.append(
                {
                    "contrast": contrast,
                    "feature": index,
                    "specificity": float(specificity[index]),
                    "color": float(scores["color"][index]),
                    "absence": float(scores["absence"][index]),
                    "occlusion": float(scores["occlusion"][index]),
                    "slab": float(scores["slab"][index]),
                }
            )
    return rows


def print_features(rows: list[dict]) -> None:
    print("\nFEATURES  mean code difference on the tokens that moved", flush=True)
    print(f"{'contrast':<12}{'feature':>8}{'specificity':>13}{'color':>10}{'absence':>10}{'occlusion':>10}{'slab':>10}", flush=True)
    for row in rows:
        print(
            f"{row['contrast']:<12}{row['feature']:>8}{row['specificity']:>13.3f}"
            f"{row['color']:>10.3f}{row['absence']:>10.3f}{row['occlusion']:>10.3f}{row['slab']:>10.3f}",
            flush=True,
        )


def frame_label(result) -> tuple:
    """Accept both label_frame shapes: (label, body) and (label, body, geometry)."""
    if isinstance(result, tuple):
        label = result[0] if result else None
        body = result[1] if len(result) > 1 else None
        return label, body
    return result, None


def inspect_candidate(collect, env, sim, sentence: str, episode: int, step: int, sim_state: np.ndarray):
    raw = collect.observe_state(env, sim_state)
    bodies = collect.candidate_bodies(sim, sentence)
    label, touched = frame_label(collect.label_frame(sim, sentence, bodies))
    body = touched if touched is not None else None
    if body is None and bodies:
        site = None
        for name in ("gripper0_grip_site", "gripper0_eef", "robot0_eef", "grip_site"):
            try:
                site = int(sim.model.site_name2id(name))
                break
            except Exception:
                site = None
        if site is not None:
            grip = np.asarray(sim.data.site_xpos[site], dtype=np.float64)
            body = min(
                bodies,
                key=lambda item: np.linalg.norm(np.mean(sim.data.geom_xpos[bodies[item]], axis=0) - grip),
            )
    if body is None or body not in bodies:
        return None
    if collect.object_qpos_span(sim, body) is None:
        return None
    built = build_conditions(collect, env, sim, body, bodies[body], raw)
    if built is None:
        return None
    return {
        "episode": episode,
        "step": step,
        "sim_state": np.asarray(sim_state),
        "label": label or "visible",
        "body": int(body),
        "geoms": list(bodies[body]),
        "built": built,
        "state": collect.libero_state(raw),
        "sentence": sentence,
    }


def capture(policy, layer5, layer6, collect, pre, frame, condition: str, noise_seed: int):
    agent, wrist = frame["built"]["images"][condition]
    batch = batch_from_images(collect, pre, agent, wrist, frame["state"], frame["sentence"], policy)
    device = next(policy.parameters()).device
    tokens5, tokens6, action = forward_policy(
        policy, layer5, layer6, batch, noise_seed, payload=None, mode="base"
    )
    return tokens5, tokens6, action, device


def score_probe(model, mean5, std5, probe_tokens: dict) -> dict[str, np.ndarray]:
    """probe_tokens[frame][condition] = layer5 float32 [T, D]."""
    buckets = {name: [] for name in ("color", "absence", "occlusion", "slab")}
    edited_of = {"color": "recolor", "absence": "absent", "occlusion": "occluded", "slab": "slab_miss"}
    for frame in probe_tokens:
        base = frame["base"]
        base_code = encode_tokens(model, base, mean5, std5)
        for contrast, condition in edited_of.items():
            edited = frame[condition]
            mask = moving_token_mask(base, edited)
            code = encode_tokens(model, edited, mean5, std5)
            buckets[contrast].append(contrast_score(base_code, code, mask))
    return {name: np.mean(np.stack(rows, axis=0), axis=0) for name, rows in buckets.items()}


def main() -> None:
    configure_environment()
    if not torch.cuda.is_available():
        raise SystemExit("CUDA GPU required. Use the Colab L4 or A100 runtime.")
    collect = load_collect_module()
    max_episodes = env_int("CTRL_MAX_EPISODES", 2)
    probe_count = env_int("CTRL_PROBE_FRAMES", 4)
    train_extra = env_int("CTRL_TRAIN_EXTRA", 3)
    n_features = env_int("CTRL_N_FEATURES", 512)
    k = env_int("CTRL_K", 16)
    steps = env_int("CTRL_TRAIN_STEPS", 400)
    top_k_features = env_int("CTRL_TOP_FEATURES", 8)
    seed = env_int("CTRL_SEED", 0)
    repo = Path(os.environ.get("LOCAL_REPO", "/content/groot-run"))
    dest = repo / "outputs" / "permanence" / "controlled_contrasts"
    dest.mkdir(parents=True, exist_ok=True)

    from libero.libero import benchmark, get_libero_path
    from libero.libero.envs import OffScreenRenderEnv

    suite = benchmark.get_benchmark_dict()[collect.SUITE]()
    task_id = int(collect.TASK_IDS[0])
    task = suite.get_task(task_id)
    sentence = task.language
    demo_path = collect.find_demo_file(task)
    bddl = os.path.join(get_libero_path("bddl_files"), task.problem_folder, task.bddl_file)
    print(f"Task {task_id}: {sentence}", flush=True)
    print(f"Demos: {demo_path}", flush=True)
    print("Loading policy", flush=True)
    policy, pre, _post, layer5 = collect.load_policy()
    policy.eval()
    layer6 = find_layer6(policy, layer5)

    episodes = []
    for index in range(max_episodes):
        try:
            _actions, states = collect.load_demo(demo_path, index)
        except IndexError:
            break
        episodes.append((index, states))
    if not episodes:
        raise SystemExit("No demonstration states were loaded.")
    same_episode = len(episodes) == 1
    train_eps = {episodes[0][0]} if same_episode else {item[0] for item in episodes[:-1]}
    probe_eps = {episodes[0][0]} if same_episode else {episodes[-1][0]}
    print(f"Train episodes {sorted(train_eps)} | probe episodes {sorted(probe_eps)}", flush=True)

    env = OffScreenRenderEnv(bddl_file_name=bddl, camera_heights=256, camera_widths=256)
    env.seed(0)
    candidates = []
    try:
        for episode, states in episodes:
            print(f"Scanning episode {episode} ({len(states)} steps)", flush=True)
            for step, sim_state in enumerate(states):
                row = inspect_candidate(collect, env, collect.sim_of(env), sentence, episode, step, sim_state)
                if row is not None:
                    candidates.append(row)
        if len(candidates) < 2:
            raise SystemExit(
                "No controllable frames. The task object did not project into the agent camera "
                "with room for a matched off-object disk."
            )
        if same_episode:
            mid = sorted(row["step"] for row in candidates)[len(candidates) // 2]
            train_candidates = [row for row in candidates if row["step"] < mid]
            probe_candidates = [row for row in candidates if row["step"] >= mid]
        else:
            train_candidates = [row for row in candidates if row["episode"] in train_eps]
            probe_candidates = [row for row in candidates if row["episode"] in probe_eps]
        probe_frames = choose_frames(probe_candidates, probe_count)
        extra_frames = choose_frames(
            [row for row in train_candidates if (row["episode"], row["step"]) not in {(f["episode"], f["step"]) for f in probe_frames}],
            train_extra,
        )
        if len(probe_frames) < 2:
            raise SystemExit(
                f"Only {len(probe_frames)} controllable probe frames. Raise CTRL_MAX_STEPS or CTRL_MAX_EPISODES."
            )
        print(
            f"Controllable frames: train-extra {len(extra_frames)} | probe {len(probe_frames)} "
            f"(scanned {len(candidates)})",
            flush=True,
        )

        train_l5: list[np.ndarray] = []
        train_l6: list[np.ndarray] = []
        probe_cutoff = min(int(frame["step"]) for frame in probe_frames) if same_episode else None
        kept_frames = {(int(frame["episode"]), int(frame["step"])) for frame in probe_frames + extra_frames}
        for row in candidates:
            if (int(row["episode"]), int(row["step"])) not in kept_frames:
                row.pop("built", None)
        for episode, states in episodes:
            for step, sim_state in enumerate(states):
                if episode not in train_eps:
                    continue
                if probe_cutoff is not None and step >= probe_cutoff:
                    continue
                raw = collect.observe_state(env, sim_state)
                state = collect.libero_state(raw)
                agent, wrist = images_of(collect, raw)
                batch = batch_from_images(collect, pre, agent, wrist, state, sentence, policy)
                noise = 1000 + episode * 100 + step
                tokens5, tokens6, _action = forward_policy(
                    policy, layer5, layer6, batch, noise, None, "base"
                )
                train_l5.append(tokens5)
                if tokens6 is not None:
                    train_l6.append(tokens6)
                if step % 8 == 0:
                    print(f"  train base ep {episode} step {step}", flush=True)

        def capture_conditions(frame, conditions: tuple[str, ...]) -> dict:
            noise = 5000 + int(frame["episode"]) * 100 + int(frame["step"])
            packed = {}
            for condition in conditions:
                tokens5, tokens6, action, _device = capture(
                    policy, layer5, layer6, collect, pre, frame, condition, noise
                )
                packed[condition] = {"l5": tokens5, "l6": tokens6, "action": action}
                print(
                    f"  captured ep {frame['episode']} step {frame['step']} {condition} "
                    f"recolor={frame['built']['recolor_method']}",
                    flush=True,
                )
            return packed

        for frame in extra_frames:
            packed = capture_conditions(frame, ("recolor", "absent", "occluded", "slab_miss"))
            for condition in packed:
                train_l5.append(packed[condition]["l5"])
                if packed[condition]["l6"] is not None:
                    train_l6.append(packed[condition]["l6"])

        probe_packed = [capture_conditions(frame, CONDITION_NAMES) for frame in probe_frames]

        # Determinism: a second base forward must reproduce the saved tokens and actions.
        check = probe_frames[0]
        noise = 5000 + int(check["episode"]) * 100 + int(check["step"])
        again5, _again6, again_action, device = capture(
            policy, layer5, layer6, collect, pre, check, "base", noise
        )
        saved5 = probe_packed[0]["base"]["l5"]
        max_token = float(np.max(np.abs(again5 - saved5)))
        if max_token > 1e-3 or not np.array_equal(again_action, probe_packed[0]["base"]["action"]):
            raise RuntimeError(
                f"Repeating the same inputs changed the forward (token max abs {max_token}). "
                "Circuit tracing needs a deterministic forward."
            )
        print(f"Determinism check passed (token max abs {max_token:.3e})", flush=True)

        pixel_rows = []
        for frame in probe_frames:
            base_agent = frame["built"]["images"]["base"][0]
            base_wrist = frame["built"]["images"]["base"][1]
            row = {
                "episode": int(frame["episode"]),
                "step": int(frame["step"]),
                "label": frame["label"],
                "recolor_method": frame["built"]["recolor_method"],
            }
            for condition in CONDITION_NAMES:
                if condition == "base":
                    continue
                row[condition] = changed_fraction(base_agent, frame["built"]["images"][condition][0])
            row["wrist_recolor"] = changed_fraction(base_wrist, frame["built"]["images"]["recolor"][1])
            pixel_rows.append(row)
        print("\nPIXELS  fraction of agent-view pixels that differ from base", flush=True)
        for row in pixel_rows:
            print(
                f"  ep {row['episode']} step {row['step']} {row['label']} via {row['recolor_method']}: "
                f"recolor={row['recolor']:.4f} wrist_recolor={row['wrist_recolor']:.4f} "
                f"absent={row['absent']:.4f} occluded={row['occluded']:.4f} slab_miss={row['slab_miss']:.4f}",
                flush=True,
            )

        # Full-cover diagnostic on the first probe frame. Both cameras are covered when the
        # object projects into them, then the two forwards see those covered images.
        first = probe_frames[0]
        cover = first["built"]["agent_full"]
        present_img = paint_mask(first["built"]["images"]["base"][0], cover, GRAY)
        absent_img = paint_mask(first["built"]["images"]["absent"][0], cover, GRAY)
        present_wrist = first["built"]["images"]["base"][1]
        absent_wrist = first["built"]["images"]["absent"][1]
        agent_outside = outside_fraction(present_img, absent_img, cover)
        wrist_cover = first["built"]["wrist_full"]
        if wrist_cover is not None:
            present_wrist = paint_mask(present_wrist, wrist_cover, GRAY)
            absent_wrist = paint_mask(absent_wrist, wrist_cover, GRAY)
            wrist_outside = outside_fraction(present_wrist, absent_wrist, wrist_cover)
        else:
            wrist_outside = changed_fraction(present_wrist, absent_wrist)
        outside = max(agent_outside, wrist_outside)

        def covered_forward(agent_image, wrist_image):
            batch = batch_from_images(
                collect, pre, agent_image, wrist_image, first["state"], first["sentence"], policy
            )
            _t5, _t6, action = forward_policy(policy, layer5, layer6, batch, noise, None, "base")
            return action

        present_action = covered_forward(present_img, present_wrist)
        absent_action = covered_forward(absent_img, absent_wrist)
        cover_rmse = float(np.sqrt(np.mean((present_action - absent_action) ** 2)))
        cover_text = full_cover_interpretation(outside, cover_rmse)
        print(
            f"\nFULL COVER  outside-pixel fraction={outside:.5f}  action RMSE={cover_rmse:.5f}",
            flush=True,
        )
        print(cover_text, flush=True)
        print("Continuing into transcoder training and circuit tracing.", flush=True)

        sheet_rows = []
        for condition in CONDITION_NAMES:
            sheet_rows.append(
                (condition, [frame["built"]["images"][condition][0] for frame in probe_frames[:4]])
            )
        sheet_rows.append(("cover_present", [present_img]))
        sheet_rows.append(("cover_absent", [absent_img]))
        sheet_path = dest / "contact_sheet.png"
        write_sheet(sheet_rows, sheet_path)
        print("Wrote", sheet_path, flush=True)

        print("\nTraining transcoder on CPU so the policy can stay loaded", flush=True)
        model, mean5, std5, mean6, std6 = train_transcoder(
            train_l5,
            train_l6 if train_l6 else None,
            n_features,
            k,
            steps,
            batch=2048,
            seed=seed,
        )
        probe_base = np.concatenate([item["base"]["l5"] for item in probe_packed], axis=0)
        recon = reconstruction(model, probe_base, mean5, std5)
        r2_5 = r2_score(recon, probe_base)
        r2_6 = None
        if mean6 is not None and std6 is not None and probe_packed[0]["base"]["l6"] is not None:
            probe_next = np.concatenate([item["base"]["l6"] for item in probe_packed], axis=0)
            predicted = next_prediction(model, probe_base, mean5, std5, mean6, std6)
            r2_6 = r2_score(predicted, probe_next)
        print(f"Probe base R^2 layer5={r2_5:.3f} layer6={r2_6}", flush=True)

        probe_token_maps = [{name: item[name]["l5"] for name in CONDITION_NAMES} for item in probe_packed]
        scores = score_probe(model, mean5, std5, probe_token_maps)
        picks = {}
        specific_flags = {}
        for contrast in CONTRAST_NAMES:
            others = [scores[name] for name in ("color", "absence", "occlusion", "slab") if name != contrast]
            ids, flag = pick_features(scores[contrast], others, top_k_features)
            picks[contrast] = ids
            specific_flags[contrast] = flag
        feature_rows = feature_table(scores, picks)
        print_features(feature_rows)
        print(
            "Score cosines  color-absence={:.3f} color-occlusion={:.3f} absence-occlusion={:.3f}".format(
                cosine(scores["color"], scores["absence"]),
                cosine(scores["color"], scores["occlusion"]),
                cosine(scores["absence"], scores["occlusion"]),
            ),
            flush=True,
        )

        rng = np.random.default_rng(seed)
        random_ids = {}
        for contrast in CONTRAST_NAMES:
            pool = np.setdiff1d(np.arange(n_features), picks[contrast])
            random_ids[contrast] = rng.choice(pool, size=len(picks[contrast]), replace=False)

        circuit_records = []
        patch_names = (
            "transcoder_features",
            "random_remap",
            "off_contrast",
            "full_tokens",
            "pooled",
            "token_structure",
        )
        edited_of = {"color": "recolor", "absence": "absent", "occlusion": "occluded"}
        for frame_index, (frame, packed) in enumerate(zip(probe_frames, probe_packed)):
            noise = 5000 + int(frame["episode"]) * 100 + int(frame["step"])
            source_tokens = packed["base"]["l5"]
            source_action = packed["base"]["action"]
            for contrast in CONTRAST_NAMES:
                condition = edited_of[contrast]
                donor_tokens = packed[condition]["l5"]
                donor_action = packed[condition]["action"]
                source_code = encode_tokens(model, source_tokens, mean5, std5)
                donor_code = encode_tokens(model, donor_tokens, mean5, std5)
                code_delta = donor_code - source_code
                selected = decode_delta(model, keep_features(code_delta, picks[contrast]), std5)
                random_delta = decode_delta(
                    model,
                    remap_features(keep_features(code_delta, picks[contrast]), picks[contrast], random_ids[contrast]),
                    std5,
                )
                other = off_contrast(contrast)  # type: ignore[arg-type]
                off_delta = decode_delta(model, keep_features(code_delta, picks[other]), std5)
                random_delta = match_l2(random_delta, selected)
                off_delta = match_l2(off_delta, selected)
                pooled = (donor_tokens - source_tokens).mean(axis=0, keepdims=True)
                pooled = np.repeat(pooled, source_tokens.shape[0], axis=0)
                payloads = {
                    "transcoder_features": ("add", selected),
                    "random_remap": ("add", random_delta),
                    "off_contrast": ("add", off_delta),
                    "full_tokens": ("replace", donor_tokens),
                    "pooled": ("add", pooled),
                    "token_structure": ("add", structure_delta(donor_tokens, source_tokens)),
                }
                agent, wrist = frame["built"]["images"]["base"]
                batch = batch_from_images(collect, pre, agent, wrist, frame["state"], frame["sentence"], policy)
                for patch in patch_names:
                    mode, payload_np = payloads[patch]
                    _tokens, _tokens6, action = forward_policy(
                        policy,
                        layer5,
                        None,
                        batch,
                        noise,
                        to_gpu(payload_np, device),
                        mode,
                    )
                    metrics = action_metrics(action, source_action, donor_action)
                    if mode == "replace":
                        realized = float(np.linalg.norm(payload_np - source_tokens))
                    else:
                        realized = float(np.linalg.norm(payload_np))
                    circuit_records.append(
                        {
                            "episode": int(frame["episode"]),
                            "step": int(frame["step"]),
                            "contrast": contrast,
                            "patch": patch,
                            "edit_l2": realized,
                            **metrics,
                        }
                    )
            print(f"Circuit frame {frame_index + 1}/{len(probe_frames)} done", flush=True)

        summary_rows = []
        for contrast in CONTRAST_NAMES:
            for patch in patch_names:
                selected_rows = [
                    row for row in circuit_records if row["contrast"] == contrast and row["patch"] == patch
                ]
                summary_rows.append(
                    {
                        "contrast": contrast,
                        "patch": patch,
                        "mean_gap_closed": mean_defined([row["fraction_gap_closed"] for row in selected_rows]),
                        "mean_edit_l2": float(np.mean([row["edit_l2"] for row in selected_rows])),
                        "mean_baseline_rmse": float(np.mean([row["source_to_donor_rmse"] for row in selected_rows])),
                        "n_defined": int(sum(row["fraction_gap_closed"] is not None for row in selected_rows)),
                        "n_pairs": len(selected_rows),
                    }
                )
        print_circuit(summary_rows)

        checkpoint = {
            "state_dict": model.state_dict(),
            "dim": int(mean5.shape[0]),
            "n_features": n_features,
            "k": k,
            "predict_next": model.predict_next,
            "mean5": mean5,
            "std5": std5,
            "mean6": mean6,
            "std6": std6,
            "features": {name: picks[name].tolist() for name in picks},
        }
        torch.save(checkpoint, dest / "transcoder.pt")
        np.savez_compressed(
            dest / "feature_scores.npz",
            **{f"score_{name}": value for name, value in scores.items()},
            **{f"features_{name}": picks[name] for name in picks},
        )
        summary = {
            "task": sentence,
            "task_id": task_id,
            "probe_frames": [
                {"episode": int(frame["episode"]), "step": int(frame["step"]), "label": frame["label"]}
                for frame in probe_frames
            ],
            "pixel_changed_fraction": pixel_rows,
            "full_cover": {
                "outside_pixel_fraction": outside,
                "action_rmse": cover_rmse,
                "images_agree": outside < 0.002,
                "actions_agree": cover_rmse < 1e-3,
                "residual_floor_rmse": cover_rmse if outside < 0.002 else None,
                "interpretation": cover_text,
            },
            "transcoder": {
                "r2_layer5_probe_base": r2_5,
                "r2_layer6_probe_base": r2_6,
                "n_features": n_features,
                "k": k,
                "steps": steps,
                "train_tokens": int(sum(chunk.shape[0] for chunk in train_l5)),
                "specific_selection": specific_flags,
            },
            "features": feature_rows,
            "score_cosines": {
                "color_absence": cosine(scores["color"], scores["absence"]),
                "color_occlusion": cosine(scores["color"], scores["occlusion"]),
                "absence_occlusion": cosine(scores["absence"], scores["occlusion"]),
            },
            "circuit": summary_rows,
            "limitations": [
                "Development diagnostic on a few frozen frames.",
                "Occlusion paints half of the pixels the color swap changed. The control is that same shape moved off the object.",
                "Absence teleports the object. The arm qpos, language, and proprioceptive state stay fixed.",
                "Recolor is a 3D material edit when that edit stays in a compact region, and a mask paint otherwise.",
                "Sparse patches are norm-matched to each other. Full-token, pooled, and token-structure patches keep their natural magnitude.",
                "A full cover that makes present and absent images agree cannot leave a hidden-object signal in one forward.",
                "A leftover action RMSE on that covered pair is the residual floor for near-matched images. The identical-input re-forward is the determinism check.",
            ],
        }
        (dest / "summary.json").write_text(json.dumps(json_ready(summary), indent=2))
        (repo / "outputs" / "permanence" / "controlled_contrasts_latest.txt").write_text(str(dest))
        print("\nWrote", dest / "summary.json", flush=True)
        print("SHEET", sheet_path, flush=True)
        print("RESULT_DIR", dest, flush=True)
    finally:
        env.close()


if __name__ == "__main__":
    main()
