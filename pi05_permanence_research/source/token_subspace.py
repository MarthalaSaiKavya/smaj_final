#!/usr/bin/env python
"""Low-rank check on the tokens that move most.

Rebuilds the saved probe frames. On the top 5% and 15% of layer-5 tokens by
movement, patches the real per-token delta, its mean, and its rank-1, rank-4,
and rank-16 reconstructions. Full tokens and token structure stay the
reference lines. Also records where those token indices sit in the prefix.

Does not rewrite the contrast run, the feature sweep, or the token-position run.
"""

from __future__ import annotations

import json
import os
from pathlib import Path

import numpy as np
import torch

import controlled_contrasts as cc
import token_position_patch as tp

EDITED_OF = {"color": "recolor", "absence": "absent", "occlusion": "occluded"}
CAPTURE_CONDITIONS = ("base", "recolor", "absent", "occluded")
REFERENCE_PATCHES = ("full_tokens", "token_structure")


def parse_ranks(text: str) -> list[int]:
    ranks: list[int] = []
    seen: set[int] = set()
    for part in text.split(","):
        piece = part.strip()
        if not piece:
            continue
        value = int(piece)
        if value < 1:
            raise ValueError(f"Rank must be positive, got {value}")
        if value not in seen:
            seen.add(value)
            ranks.append(value)
    if not ranks:
        raise ValueError("CTRL_TOKEN_RANKS is empty")
    return ranks


def patch_names(ranks: list[int]) -> list[str]:
    return ["full", "mean"] + [f"rank{rank}" for rank in ranks]


def low_rank_payload(source: np.ndarray, donor: np.ndarray, indexes: np.ndarray, rank: int) -> np.ndarray:
    """Write a rank-k reconstruction of the selected token delta. Rank 0 is the mean."""
    chosen = np.asarray(indexes, dtype=np.int64)
    block = donor[chosen].astype(np.float64) - source[chosen].astype(np.float64)
    mean = block.mean(axis=0, keepdims=True)
    if rank <= 0 or block.shape[0] == 1:
        recon = np.repeat(mean, block.shape[0], axis=0)
    else:
        centered = block - mean
        _u, singular, vt = np.linalg.svd(centered, full_matrices=False)
        keep = min(int(rank), int(singular.size))
        recon = (centered @ vt[:keep].T) @ vt[:keep] + mean
    payload = np.zeros_like(source, dtype=np.float32)
    payload[chosen] = recon.astype(np.float32)
    return payload


def payload_share(payload: np.ndarray, source: np.ndarray, donor: np.ndarray) -> float:
    total = float(np.linalg.norm(donor.astype(np.float64) - source.astype(np.float64)))
    if total < 1e-12:
        return 0.0
    return float(np.linalg.norm(payload.astype(np.float64)) / total)


def subspace_verdict(contrast: str, fraction: float, full_gap: float | None, by_patch: dict[str, float | None]) -> str:
    """Compare low-rank patches with the full delta on the same tokens."""
    label = f"{round(100 * fraction)}%"
    if full_gap is None:
        return f"{contrast} at {label}: the full position patch is undefined."
    rank4 = by_patch.get("rank4")
    rank16 = by_patch.get("rank16")
    if rank4 is not None and full_gap > 0 and rank4 >= 0.75 * full_gap:
        return (
            f"{contrast} at {label}: rank 4 keeps {100 * rank4:.1f}% of the {100 * full_gap:.1f}% "
            "closed by the full token delta. The moving tokens share a low-rank subspace."
        )
    if rank16 is not None and full_gap > 0 and rank16 < 0.50 * full_gap:
        return (
            f"{contrast} at {label}: rank 16 keeps {100 * rank16:.1f}% of the {100 * full_gap:.1f}% "
            "closed by the full token delta. The residual is token-specific."
        )
    pieces = [f"full {100 * full_gap:.1f}%"]
    for name in ("mean", "rank1", "rank4", "rank16"):
        gap = by_patch.get(name)
        if gap is not None:
            pieces.append(f"{name} {100 * gap:.1f}%")
    return f"{contrast} at {label}: " + ", ".join(pieces) + "."


def _position_count(module) -> int | None:
    weight = getattr(module, "weight", None)
    if weight is None or not hasattr(weight, "shape") or len(tuple(weight.shape)) != 2:
        return None
    return int(weight.shape[0])


def tokens_per_image(policy) -> int | None:
    found: list[int] = []
    for name, module in policy.named_modules():
        if "vision" not in name or not name.endswith("position_embedding"):
            continue
        count = _position_count(module)
        if count is None:
            continue
        side = int(round(count ** 0.5))
        if side >= 4 and side * side == count:
            found.append(count)
    if not found:
        return None
    return int(max(found))


def image_count(policy) -> int:
    features = getattr(getattr(policy, "config", None), "input_features", {}) or {}
    return sum(1 for key in features if str(key).startswith("observation.images."))


def token_region(index: int, per_image: int | None, n_images: int, n_tokens: int) -> str:
    if per_image is None or n_images < 1:
        return "unknown"
    vision = per_image * n_images
    if vision > n_tokens:
        return "unknown"
    if int(index) >= vision:
        return "language"
    return f"image{int(index) // per_image}"


def decile_shares(index_lists: list[np.ndarray], n_tokens: int, bins: int = 10) -> list[float]:
    counts = np.zeros(bins, dtype=np.float64)
    for indexes in index_lists:
        for index in np.asarray(indexes, dtype=np.int64):
            bin_id = min(bins - 1, int(int(index) * bins / max(n_tokens, 1)))
            counts[bin_id] += 1
    total = float(counts.sum())
    if total <= 0:
        return [0.0] * bins
    return [float(value / total) for value in counts]


def region_shares(index_lists: list[np.ndarray], per_image: int | None, n_images: int, n_tokens: int) -> dict[str, float]:
    counts: dict[str, int] = {}
    total = 0
    for indexes in index_lists:
        for index in np.asarray(indexes, dtype=np.int64):
            name = token_region(int(index), per_image, n_images, n_tokens)
            counts[name] = counts.get(name, 0) + 1
            total += 1
    if total == 0:
        return {}
    return {name: counts[name] / total for name in sorted(counts)}


def gap_of(rows: list[dict], contrast: str, fraction: float | None, patch: str) -> float | None:
    return next(
        row["mean_gap_closed"]
        for row in rows
        if row["contrast"] == contrast and row["fraction"] == fraction and row["patch"] == patch
    )


def main() -> None:
    cc.configure_environment()
    if not torch.cuda.is_available():
        raise SystemExit("CUDA GPU required. Use the Colab L4 or A100 runtime.")
    collect = cc.load_collect_module()
    seed = cc.env_int("CTRL_SEED", 0)
    repo = Path(os.environ.get("LOCAL_REPO", "/content/groot-run"))
    source = repo / "outputs" / "permanence" / "controlled_contrasts"
    dest = repo / "outputs" / "permanence" / "token_subspace"
    dest.mkdir(parents=True, exist_ok=True)
    summary_path = source / "summary.json"
    if not summary_path.is_file():
        raise SystemExit(f"Missing {summary_path}. Run the contrast cell first.")
    prior = json.loads(summary_path.read_text())
    wanted = [(int(row["episode"]), int(row["step"])) for row in prior["probe_frames"]]
    if len(wanted) < 2:
        raise SystemExit("The saved summary has fewer than two probe frames.")
    fractions = tp.parse_fractions(os.environ.get("CTRL_TOKEN_FRACTIONS", "0.05,0.15"))
    ranks = parse_ranks(os.environ.get("CTRL_TOKEN_RANKS", "1,4,16"))
    names = patch_names(ranks)
    print("Probe frames", wanted, flush=True)
    print("Token fractions", [tp.fraction_label(item) for item in fractions], "ranks", ranks, flush=True)

    # Imported here so unit tests can load this module without LIBERO.
    from libero.libero import benchmark, get_libero_path
    from libero.libero.envs import OffScreenRenderEnv

    suite = benchmark.get_benchmark_dict()[collect.SUITE]()
    task_id = int(collect.TASK_IDS[0])
    task = suite.get_task(task_id)
    sentence = task.language
    demo_path = collect.find_demo_file(task)
    bddl = os.path.join(get_libero_path("bddl_files"), task.problem_folder, task.bddl_file)
    print(f"Task {task_id}: {sentence}", flush=True)
    print("Loading policy", flush=True)
    policy, pre, _post, layer5 = collect.load_policy()
    policy.eval()
    layer6 = cc.find_layer6(policy, layer5)

    demos: dict[int, list] = {}
    for episode, _step in wanted:
        if episode not in demos:
            _actions, states = collect.load_demo(demo_path, episode)
            demos[episode] = states

    env = OffScreenRenderEnv(bddl_file_name=bddl, camera_heights=256, camera_widths=256)
    env.seed(0)
    try:
        probe_frames = []
        for episode, step in wanted:
            states = demos[episode]
            if step >= len(states):
                raise SystemExit(f"Episode {episode} has {len(states)} steps; cannot rebuild step {step}.")
            print(f"Rebuilding ep {episode} step {step}", flush=True)
            row = cc.inspect_candidate(
                collect, env, collect.sim_of(env), sentence, episode, step, states[step]
            )
            if row is None:
                raise SystemExit(f"Could not rebuild ep {episode} step {step}.")
            probe_frames.append(row)

        def capture_conditions(frame) -> dict:
            noise = 5000 + int(frame["episode"]) * 100 + int(frame["step"])
            packed = {}
            for condition in CAPTURE_CONDITIONS:
                tokens5, _tokens6, action, _device = cc.capture(
                    policy, layer5, layer6, collect, pre, frame, condition, noise
                )
                packed[condition] = {"l5": tokens5, "action": action}
            return packed

        probe_packed = [capture_conditions(frame) for frame in probe_frames]
        check = probe_frames[0]
        noise = 5000 + int(check["episode"]) * 100 + int(check["step"])
        again5, _again6, again_action, device = cc.capture(
            policy, layer5, layer6, collect, pre, check, "base", noise
        )
        max_token = float(np.max(np.abs(again5 - probe_packed[0]["base"]["l5"])))
        if max_token > 1e-3 or not np.array_equal(again_action, probe_packed[0]["base"]["action"]):
            raise RuntimeError(f"Repeating the same inputs changed the forward (token max abs {max_token}).")
        print(f"Determinism check passed (token max abs {max_token:.3e})", flush=True)

        n_tokens = int(probe_packed[0]["base"]["l5"].shape[0])
        per_image = tokens_per_image(policy)
        n_images = image_count(policy)
        print(
            f"Prefix length {n_tokens} | vision tokens/image {per_image} | image inputs {n_images}",
            flush=True,
        )

        rng = np.random.default_rng(seed)
        records: list[dict] = []
        located: dict[tuple[str, float], list[np.ndarray]] = {}
        for frame_index, (frame, packed) in enumerate(zip(probe_frames, probe_packed)):
            noise = 5000 + int(frame["episode"]) * 100 + int(frame["step"])
            source_tokens = packed["base"]["l5"]
            source_action = packed["base"]["action"]
            n_here = int(source_tokens.shape[0])
            agent, wrist = frame["built"]["images"]["base"]
            batch = cc.batch_from_images(collect, pre, agent, wrist, frame["state"], frame["sentence"], policy)

            def run_patch(contrast, patch, fraction, mode, payload_np, target_action, share, n_patched):
                _tokens, _tokens6, action = cc.forward_policy(
                    policy, layer5, None, batch, noise, cc.to_gpu(payload_np, device), mode
                )
                metrics = cc.action_metrics(action, source_action, target_action)
                if mode == "replace":
                    realized = float(np.linalg.norm(payload_np.astype(np.float64) - source_tokens.astype(np.float64)))
                else:
                    realized = float(np.linalg.norm(payload_np.astype(np.float64)))
                records.append(
                    {
                        "episode": int(frame["episode"]),
                        "step": int(frame["step"]),
                        "contrast": contrast,
                        "fraction": fraction,
                        "patch": patch,
                        "edit_l2": realized,
                        "token_l2_share": share,
                        "n_tokens": n_patched,
                        **metrics,
                    }
                )

            for contrast in cc.CONTRAST_NAMES:
                condition = EDITED_OF[contrast]
                donor_tokens = packed[condition]["l5"]
                donor_action = packed[condition]["action"]
                order, _distance = tp.movement(source_tokens, donor_tokens)
                structure_payload = cc.structure_delta(donor_tokens, source_tokens)
                references = {
                    "full_tokens": ("replace", donor_tokens.astype(np.float32), 1.0, n_here),
                    "token_structure": (
                        "add",
                        structure_payload,
                        payload_share(structure_payload, source_tokens, donor_tokens),
                        n_here,
                    ),
                }
                for patch in REFERENCE_PATCHES:
                    mode, payload_np, share, n_patched = references[patch]
                    run_patch(contrast, patch, None, mode, payload_np, donor_action, share, n_patched)

                for fraction in fractions:
                    count = tp.token_count(n_here, fraction)
                    indexes = tp.select_positions(order, n_here, count, "largest", rng)
                    located.setdefault((contrast, fraction), []).append(indexes)
                    built = {"full": tp.position_delta(source_tokens, donor_tokens, indexes)}
                    built["mean"] = low_rank_payload(source_tokens, donor_tokens, indexes, 0)
                    for rank in ranks:
                        built[f"rank{rank}"] = low_rank_payload(source_tokens, donor_tokens, indexes, rank)
                    for patch in names:
                        payload_np = built[patch]
                        run_patch(
                            contrast,
                            patch,
                            fraction,
                            "add",
                            payload_np,
                            donor_action,
                            payload_share(payload_np, source_tokens, donor_tokens),
                            int(indexes.size),
                        )
            print(f"Subspace frame {frame_index + 1}/{len(probe_frames)} done", flush=True)

        rows: list[dict] = []
        for contrast in cc.CONTRAST_NAMES:
            for patch in REFERENCE_PATCHES:
                rows.append(tp.summarize(records, contrast, None, patch))
            for fraction in fractions:
                for patch in names:
                    rows.append(tp.summarize(records, contrast, fraction, patch))
        tp.print_table(rows)

        print("\nPOSITIONS  where the largest-moving tokens sit (share of selected indexes)", flush=True)
        position_rows = []
        for fraction in fractions:
            print(f"  fraction {tp.fraction_label(fraction)}", flush=True)
            for contrast in cc.CONTRAST_NAMES:
                groups = located.get((contrast, fraction), [])
                deciles = decile_shares(groups, n_tokens)
                regions = region_shares(groups, per_image, n_images, n_tokens)
                decile_text = " ".join(f"d{index}={100 * share:.0f}%" for index, share in enumerate(deciles))
                region_text = " ".join(f"{name}={100 * share:.0f}%" for name, share in regions.items())
                print(f"    {contrast:<12}{decile_text}  {region_text}", flush=True)
                position_rows.append(
                    {
                        "contrast": contrast,
                        "fraction": fraction,
                        "deciles": deciles,
                        "regions": regions,
                    }
                )

        verdicts = []
        for fraction in fractions:
            for contrast in cc.CONTRAST_NAMES:
                by_patch = {patch: gap_of(rows, contrast, fraction, patch) for patch in names if patch != "full"}
                text = subspace_verdict(contrast, fraction, gap_of(rows, contrast, fraction, "full"), by_patch)
                verdicts.append(text)
                print("\n" + text, flush=True)

        summary = {
            "task": sentence,
            "task_id": task_id,
            "source_run": str(source),
            "probe_frames": [
                {"episode": int(frame["episode"]), "step": int(frame["step"]), "label": frame["label"]}
                for frame in probe_frames
            ],
            "fractions": fractions,
            "ranks": ranks,
            "n_tokens": n_tokens,
            "tokens_per_image": per_image,
            "n_images": n_images,
            "rows": rows,
            "positions": position_rows,
            "verdicts": verdicts,
        }
        (dest / "summary.json").write_text(json.dumps(cc.json_ready(summary), indent=2))
        print("\nWrote", dest / "summary.json", flush=True)
        print("RESULT_DIR", dest, flush=True)
    finally:
        env.close()


if __name__ == "__main__":
    main()
