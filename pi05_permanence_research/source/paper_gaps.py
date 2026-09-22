#!/usr/bin/env python
"""Close the gaps that block reading the occlusion run as a paper.

One rule for features: the code must rise on paint over the bowl and stay quiet
on paint off the bowl, on color, and on absence. The largest feature that fails
is reported, including when it falls or also moves for absence.

The same frames get the causal ceiling (full prefix, token structure, pooled),
a second noise seed, and a bootstrap interval. A short rollout asks whether the
chunk changes task success. A scan of the demonstrations counts gripper contact
and gripper hiding. The figure shows the agent view, the wrist view, the mask,
and a zoom of the paint.

Does not retrain the transcoder and does not rewrite earlier output directories.
"""

from __future__ import annotations

import json
import os
import re
from pathlib import Path

import cv2
import h5py
import numpy as np
import torch

import controlled_contrasts as cc
import occlusion_features as oc

CAPTURE = ("base", "recolor", "absent", "occluded", "slab_miss")
HELD_LABELS = ("held_hidden", "held_visible")


def bootstrap_interval(values: list[float], rng: np.random.Generator, draws: int = 2000) -> dict:
    arr = np.asarray([value for value in values if value is not None and np.isfinite(value)], dtype=np.float64)
    if arr.size == 0:
        return {"mean": None, "low": None, "high": None, "n": 0}
    if arr.size == 1:
        value = float(arr[0])
        return {"mean": value, "low": value, "high": value, "n": 1}
    picks = rng.integers(0, int(arr.size), size=(int(draws), int(arr.size)))
    means = arr[picks].mean(axis=1)
    return {
        "mean": float(arr.mean()),
        "low": float(np.quantile(means, 0.025)),
        "high": float(np.quantile(means, 0.975)),
        "n": int(arr.size),
    }


def zoom_box(mask: np.ndarray, pad: int, height: int, width: int) -> tuple[int, int, int, int]:
    ys, xs = np.nonzero(np.asarray(mask))
    if ys.size == 0:
        return 0, int(height), 0, int(width)
    y0 = max(0, int(ys.min()) - int(pad))
    y1 = min(int(height), int(ys.max()) + int(pad) + 1)
    x0 = max(0, int(xs.min()) - int(pad))
    x1 = min(int(width), int(xs.max()) + int(pad) + 1)
    return y0, y1, x0, x1


def crop(image: np.ndarray, box: tuple[int, int, int, int]) -> np.ndarray:
    y0, y1, x0, x1 = box
    return np.asarray(image)[y0:y1, x0:x1]


def spread_frames(rows: list[dict], limit: int) -> list[dict]:
    if limit < 1 or not rows:
        return []
    buckets: dict[int, list[dict]] = {}
    for row in rows:
        buckets.setdefault(int(row["episode"]), []).append(row)
    picked: list[dict] = []
    while len(picked) < limit:
        grew = False
        for episode in sorted(buckets):
            bucket = buckets[episode]
            if bucket and len(picked) < limit:
                picked.append(bucket.pop(0))
                grew = True
        if not grew:
            break
    return picked


def rejection_reasons(row: dict, min_specificity: float) -> list[str]:
    reasons: list[str] = []
    if float(row["occlusion"]) <= 0:
        reasons.append("falls instead of rising")
    if float(row["specificity"]) <= float(min_specificity):
        reasons.append(f"specificity {float(row['specificity']):.2f} is at or below {float(min_specificity):.2f}")
    if abs(float(row["absence"])) > 0.5 * max(abs(float(row["occlusion"])), 1e-8):
        reasons.append("also moves for absence")
    if abs(float(row["color"])) > 0.5 * max(abs(float(row["occlusion"])), 1e-8):
        reasons.append("also moves for color")
    if abs(float(row["slab"])) > 0.5 * max(abs(float(row["occlusion"])), 1e-8):
        reasons.append("also moves for paint off the bowl")
    return reasons


def claim_line(r2: float | None, full_gap: float | None, feature_gap: float | None) -> str:
    r2_text = "unknown" if r2 is None else f"{r2:.3f}"
    full_text = "unknown" if full_gap is None else f"{100 * full_gap:.1f}%"
    if feature_gap is None:
        feature_text = "no feature passed the rule, so there is no feature effect to report"
    else:
        feature_text = f"the features that pass close {100 * feature_gap:.1f}%"
    return (
        f"Layer-5 reconstruction R2 is {r2_text}. "
        f"The causal ceiling on this paint edit is the full prefix at {full_text}. "
        f"{feature_text}."
    )


def interval_text(stats: dict) -> str:
    if stats["mean"] is None:
        return "undefined"
    if stats["n"] < 2:
        return f"{100 * stats['mean']:.1f}% (n={stats['n']})"
    return (
        f"{100 * stats['mean']:.1f}% "
        f"[{100 * stats['low']:.1f}%, {100 * stats['high']:.1f}%] n={stats['n']}"
    )


def read_demo_states(path: Path, max_steps: int) -> list[np.ndarray]:
    demos: list[np.ndarray] = []
    with h5py.File(path, "r") as handle:
        names = sorted(
            handle["data"].keys(),
            key=lambda name: int(re.search(r"(\d+)$", name).group(1)),
        )
        for name in names:
            group = handle["data"][name]
            states = np.asarray(group["states"])
            actions = np.asarray(group["actions"])
            if len(states) == len(actions) + 1:
                states = states[:-1]
            demos.append(states[:max_steps])
    return demos


def object_xyz(sim, body: int, span: tuple[int, int] | None) -> np.ndarray | None:
    if span is None:
        return None
    start = int(span[0])
    return np.asarray(sim.data.qpos[start : start + 3], dtype=np.float64).copy()


def task_succeeded(env) -> bool:
    for owner in (env, getattr(env, "env", None)):
        if owner is None:
            continue
        for name in ("check_success", "_check_success"):
            fn = getattr(owner, name, None)
            if callable(fn):
                return bool(fn())
    return False


def step_env(env, action: np.ndarray) -> None:
    vector = np.asarray(action, dtype=np.float64).reshape(-1)
    space = getattr(env, "action_space", None)
    width = int(space.shape[0]) if space is not None and getattr(space, "shape", None) else vector.size
    vector = vector[:width]
    outcome = env.step(vector)
    if not isinstance(outcome, tuple) or len(outcome) not in (4, 5):
        raise RuntimeError(f"env.step returned {type(outcome)}")


def overlay(image: np.ndarray, mask: np.ndarray) -> np.ndarray:
    out = np.array(image, copy=True)
    if mask is None or not np.asarray(mask).any():
        return out
    red = np.zeros_like(out)
    red[..., 0] = 255
    out[mask] = (0.45 * out[mask] + 0.55 * red[mask]).astype(np.uint8)
    return out


def enlarge(image: np.ndarray, factor: int = 4) -> np.ndarray:
    return np.repeat(np.repeat(np.asarray(image), factor, axis=0), factor, axis=1)


def write_edit_sheet(frame: dict, path: Path) -> None:
    images = frame["built"]["images"]
    partial = frame["built"].get("agent_partial")
    wrist_partial = frame["built"].get("wrist_partial")
    agent0 = images["base"][0]
    box = zoom_box(partial if partial is not None else np.zeros(agent0.shape[:2], dtype=bool), 36, agent0.shape[0], agent0.shape[1])
    wrist0 = images["base"][1]
    wrist_box = zoom_box(
        wrist_partial if wrist_partial is not None else np.zeros(wrist0.shape[:2], dtype=bool),
        36,
        wrist0.shape[0],
        wrist0.shape[1],
    )
    columns = ("base", "recolor", "occluded", "slab_miss", "absent")
    rows = []
    for kind, region in (("agent", box), ("wrist", wrist_box)):
        view = 0 if kind == "agent" else 1
        mask = partial if kind == "agent" else wrist_partial
        picture_row = []
        zoom_row = []
        mask_row = []
        for condition in columns:
            picture = images[condition][view]
            picture_row.append(picture)
            zoom_row.append(crop(picture, region if kind == "agent" else wrist_box))
            mask_row.append(overlay(picture, mask) if mask is not None else picture)
        rows.append((f"{kind}", picture_row))
        rows.append((f"{kind}_zoom", zoom_row))
        rows.append((f"{kind}_mask", mask_row))
    cc.write_sheet(rows, path)
    occluded = images["occluded"][0]
    zoom = enlarge(crop(occluded, box), 4)
    base_zoom = enlarge(crop(agent0, box), 4)
    pair = np.concatenate([base_zoom, zoom], axis=1)
    cv2.imwrite(str(path.with_name(path.stem + "_zoom.png")), cv2.cvtColor(pair, cv2.COLOR_RGB2BGR))


def cover_pair(frame: dict) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, float] | None:
    cover = frame["built"].get("agent_full")
    if cover is None:
        return None
    images = frame["built"]["images"]
    present = cc.paint_mask(images["base"][0], cover, cc.GRAY)
    absent = cc.paint_mask(images["absent"][0], cover, cc.GRAY)
    present_wrist = images["base"][1]
    absent_wrist = images["absent"][1]
    outside = cc.outside_fraction(present, absent, cover)
    wrist_cover = frame["built"].get("wrist_full")
    if wrist_cover is not None:
        present_wrist = cc.paint_mask(present_wrist, wrist_cover, cc.GRAY)
        absent_wrist = cc.paint_mask(absent_wrist, wrist_cover, cc.GRAY)
        outside = max(outside, cc.outside_fraction(present_wrist, absent_wrist, wrist_cover))
    return present, present_wrist, absent, absent_wrist, float(outside)


def per_frame_gaps(records: list[dict], patch: str, seed: int | None) -> list[float]:
    return [
        row["fraction_gap_closed"]
        for row in records
        if row["patch"] == patch and row["seed_tag"] == seed and row["fraction_gap_closed"] is not None
    ]


def main() -> None:
    cc.configure_environment()
    if not torch.cuda.is_available():
        raise SystemExit("CUDA GPU required. Use the Colab L4 or A100 runtime.")
    os.environ["MAX_STEPS"] = os.environ.get("CTRL_SCAN_STEPS", "200")
    collect = cc.load_collect_module()
    seed = cc.env_int("CTRL_SEED", 0)
    min_specificity = float(os.environ.get("CTRL_MIN_SPECIFICITY", "0.5"))
    scan_episodes = cc.env_int("CTRL_SCAN_EPISODES", 20)
    scan_steps = cc.env_int("CTRL_SCAN_STEPS", 200)
    stride = cc.env_int("CTRL_SCAN_STRIDE", 2)
    held_limit = cc.env_int("CTRL_HELD_FRAMES", 4)
    rollout_steps = cc.env_int("CTRL_ROLLOUT_STEPS", 10)
    draws = cc.env_int("CTRL_BOOTSTRAP", 2000)
    repo = Path(os.environ.get("LOCAL_REPO", "/content/groot-run"))
    source = repo / "outputs" / "permanence" / "controlled_contrasts"
    dest = repo / "outputs" / "permanence" / "paper_gaps"
    dest.mkdir(parents=True, exist_ok=True)
    summary_path = source / "summary.json"
    checkpoint_path = source / "transcoder.pt"
    if not summary_path.is_file() or not checkpoint_path.is_file():
        raise SystemExit(f"Missing {summary_path} or {checkpoint_path}. Run the contrast cell first.")
    prior = json.loads(summary_path.read_text())
    replicate = [(int(row["episode"]), int(row["step"])) for row in prior["probe_frames"]]
    r2 = (prior.get("transcoder") or {}).get("r2_layer5_probe_base")
    model, mean5, std5, n_features = oc.load_transcoder(checkpoint_path)
    print(f"Loaded transcoder features={n_features} R2={r2}", flush=True)

    # Imported here so unit tests can load this module without LIBERO.
    from libero.libero import benchmark, get_libero_path
    from libero.libero.envs import OffScreenRenderEnv

    suite = benchmark.get_benchmark_dict()[collect.SUITE]()
    task_id = int(collect.TASK_IDS[0])
    task = suite.get_task(task_id)
    sentence = task.language
    demo_path = collect.find_demo_file(task)
    bddl = os.path.join(get_libero_path("bddl_files"), task.problem_folder, task.bddl_file)
    demos = read_demo_states(demo_path, scan_steps)
    print(f"Task {task_id}: {sentence}", flush=True)
    print(f"Demos in file: {len(demos)}. Scanning {min(scan_episodes, len(demos))} at stride {stride}.", flush=True)

    env = OffScreenRenderEnv(bddl_file_name=bddl, camera_heights=256, camera_widths=256)
    env.seed(0)
    try:
        census: dict[str, int] = {}
        held_rows: list[dict] = []
        for episode, states in enumerate(demos[:scan_episodes]):
            print(f"Scanning episode {episode} ({len(states)} steps)", flush=True)
            for step in range(0, len(states), stride):
                collect.observe_state(env, states[step])
                sim = collect.sim_of(env)
                label, body = cc.frame_label(collect.label_frame(sim, sentence, collect.candidate_bodies(sim, sentence)))
                name = label or "unlabeled"
                census[name] = census.get(name, 0) + 1
                if name in HELD_LABELS and body is not None:
                    held_rows.append({"episode": episode, "step": step, "label": name, "body": int(body)})
        print("CENSUS", census, flush=True)
        hidden = [row for row in held_rows if row["label"] == "held_hidden"]
        visible = [row for row in held_rows if row["label"] == "held_visible"]
        chosen_held = spread_frames(hidden or visible, held_limit)
        print(
            f"Gripper contact frames: hidden={len(hidden)} visible={len(visible)}. "
            f"Using {len(chosen_held)} for the policy.",
            flush=True,
        )

        print("Loading policy", flush=True)
        policy, pre, _post, layer5 = collect.load_policy()
        policy.eval()
        layer6 = cc.find_layer6(policy, layer5)

        def rebuild(episode: int, step: int):
            states = demos[episode]
            if step >= len(states):
                raise SystemExit(f"Episode {episode} has {len(states)} scanned steps; cannot rebuild step {step}.")
            print(f"Rebuilding ep {episode} step {step}", flush=True)
            row = cc.inspect_candidate(collect, env, collect.sim_of(env), sentence, episode, step, states[step])
            return row

        replicate_frames = []
        for episode, step in replicate:
            row = rebuild(episode, step)
            if row is None:
                raise SystemExit(f"Could not rebuild replicate ep {episode} step {step}.")
            replicate_frames.append(row)
        held_frames = []
        for item in chosen_held:
            row = rebuild(int(item["episode"]), int(item["step"]))
            if row is None:
                print(f"Skipped held ep {item['episode']} step {item['step']}: controlled images failed.", flush=True)
                continue
            held_frames.append(row)

        sheet = dest / "edit_sheet.png"
        write_edit_sheet(replicate_frames[0], sheet)
        print("Wrote", sheet, "and", sheet.with_name(sheet.stem + "_zoom.png"), flush=True)
        if held_frames:
            held_sheet = dest / "held_sheet.png"
            write_edit_sheet(held_frames[0], held_sheet)
            print("Wrote", held_sheet, flush=True)

        def capture_frame(frame, noise: int) -> dict:
            packed = {}
            for condition in CAPTURE:
                tokens5, _tokens6, action, _device = cc.capture(
                    policy, layer5, layer6, collect, pre, frame, condition, noise
                )
                packed[condition] = {"l5": tokens5, "action": action}
            return packed

        groups = {"paint": replicate_frames, "gripper": held_frames}
        records: list[dict] = []
        rollouts: list[dict] = []
        covers: list[dict] = []
        passed_rows: list[dict] = []
        leader_row: dict | None = None
        leader_reasons: list[str] = []
        rng = np.random.default_rng(seed)

        for group_name, frames in groups.items():
            if not frames:
                continue
            packed_seed_a = []
            for frame in frames:
                noise_a = 5000 + int(frame["episode"]) * 100 + int(frame["step"])
                packed_seed_a.append(capture_frame(frame, noise_a))
            check = frames[0]
            noise_a = 5000 + int(check["episode"]) * 100 + int(check["step"])
            again5, _again6, again_action, device = cc.capture(
                policy, layer5, layer6, collect, pre, check, "base", noise_a
            )
            max_token = float(np.max(np.abs(again5 - packed_seed_a[0]["base"]["l5"])))
            if max_token > 1e-3 or not np.array_equal(again_action, packed_seed_a[0]["base"]["action"]):
                raise RuntimeError(f"{group_name} determinism failed (token max abs {max_token}).")
            print(f"{group_name} determinism passed (token max abs {max_token:.3e})", flush=True)

            token_maps = [{name: item[name]["l5"] for name in CAPTURE} for item in packed_seed_a]
            scores = cc.score_probe(model, mean5, std5, token_maps)
            occlusion_ids = oc.specific_ids(scores, "occlusion", oc.QUIET_FOR["occlusion"], min_specificity)
            color_ids = oc.specific_ids(scores, "color", oc.QUIET_FOR["color"], min_specificity)
            leader_id = int(np.argmax(np.abs(np.asarray(scores["occlusion"], dtype=np.float64))))
            leader_rows = oc.feature_rows(np.array([leader_id]), scores, "occlusion", oc.QUIET_FOR["occlusion"])
            leader = leader_rows[0]
            reasons = rejection_reasons(leader, min_specificity)
            if group_name == "paint":
                passed_rows = oc.feature_rows(occlusion_ids, scores, "occlusion", oc.QUIET_FOR["occlusion"])
                leader_row = leader
                leader_reasons = reasons
            print(
                f"{group_name}: passed {occlusion_ids.tolist() or 'none'} | "
                f"largest |occlusion| feature {leader_id} reasons={reasons or ['passes the rule']}",
                flush=True,
            )
            oc.print_feature_table(f"{group_name} PASSED", oc.feature_rows(occlusion_ids, scores, "occlusion", oc.QUIET_FOR["occlusion"]))

            pool = np.setdiff1d(np.arange(n_features), occlusion_ids) if occlusion_ids.size else np.arange(n_features)
            random_ids = (
                rng.choice(pool, size=int(occlusion_ids.size), replace=False).astype(np.int64)
                if occlusion_ids.size
                else np.array([], dtype=np.int64)
            )

            for frame, packed in zip(frames, packed_seed_a):
                noise_a = 5000 + int(frame["episode"]) * 100 + int(frame["step"])
                noise_b = 9000 + int(frame["episode"]) * 100 + int(frame["step"])
                source_tokens = packed["base"]["l5"]
                source_action = packed["base"]["action"]
                donor_tokens = packed["occluded"]["l5"]
                donor_action = packed["occluded"]["action"]
                agent, wrist = frame["built"]["images"]["base"]
                batch = cc.batch_from_images(collect, pre, agent, wrist, frame["state"], frame["sentence"], policy)
                code_delta = cc.encode_tokens(model, donor_tokens, mean5, std5) - cc.encode_tokens(
                    model, source_tokens, mean5, std5
                )

                def run_patch(patch: str, mode: str, payload_np: np.ndarray) -> np.ndarray:
                    _tokens, _tokens6, action = cc.forward_policy(
                        policy, layer5, None, batch, noise_a, cc.to_gpu(payload_np, device), mode
                    )
                    metrics = cc.action_metrics(action, source_action, donor_action)
                    records.append(
                        {
                            "group": group_name,
                            "episode": int(frame["episode"]),
                            "step": int(frame["step"]),
                            "label": frame["label"],
                            "patch": patch,
                            "seed_tag": 0,
                            "edit_l2": float(np.linalg.norm(payload_np.astype(np.float64))),
                            **metrics,
                        }
                    )
                    return action

                run_patch("full_tokens", "replace", donor_tokens.astype(np.float32))
                run_patch("token_structure", "add", cc.structure_delta(donor_tokens, source_tokens))
                pooled = np.repeat((donor_tokens - source_tokens).mean(axis=0, keepdims=True), source_tokens.shape[0], axis=0)
                run_patch("pooled", "add", pooled.astype(np.float32))
                feature_action = None
                if occlusion_ids.size:
                    feature_payload = oc.decoded_patch(model, code_delta, occlusion_ids, std5, None)
                    moved = cc.remap_features(cc.keep_features(code_delta, occlusion_ids), occlusion_ids, random_ids)
                    random_payload = cc.match_l2(cc.decode_delta(model, moved, std5), feature_payload)
                    color_take = color_ids[: int(occlusion_ids.size)]
                    color_payload = oc.decoded_patch(model, code_delta, color_take, std5, feature_payload)
                    feature_action = run_patch("occlusion_features", "add", feature_payload)
                    run_patch("random_remap", "add", random_payload)
                    run_patch("color_features", "add", color_payload)
                leader_payload = oc.decoded_patch(model, code_delta, np.array([leader_id]), std5, None)
                run_patch("rejected_leader", "add", leader_payload)
                absence_rmse = float(np.sqrt(np.mean((source_action - packed["absent"]["action"]) ** 2)))
                records.append(
                    {
                        "group": group_name,
                        "episode": int(frame["episode"]),
                        "step": int(frame["step"]),
                        "label": frame["label"],
                        "patch": "absence_rmse",
                        "seed_tag": 0,
                        "edit_l2": absence_rmse,
                        "source_to_donor_rmse": absence_rmse,
                        "edited_to_donor_rmse": None,
                        "fraction_gap_closed": None,
                    }
                )

                packed_b = capture_frame(frame, noise_b)
                source_b = packed_b["base"]["action"]
                donor_b = packed_b["occluded"]["action"]
                batch_b_tokens = packed_b["occluded"]["l5"]
                _tokens, _tokens6, action_b = cc.forward_policy(
                    policy, layer5, None, batch, noise_b, cc.to_gpu(batch_b_tokens.astype(np.float32), device), "replace"
                )
                records.append(
                    {
                        "group": group_name,
                        "episode": int(frame["episode"]),
                        "step": int(frame["step"]),
                        "label": frame["label"],
                        "patch": "full_tokens",
                        "seed_tag": 1,
                        "edit_l2": float(np.linalg.norm(batch_b_tokens.astype(np.float64) - packed_b["base"]["l5"].astype(np.float64))),
                        **cc.action_metrics(action_b, source_b, donor_b),
                    }
                )
                structure_b = cc.structure_delta(packed_b["occluded"]["l5"], packed_b["base"]["l5"])
                _tokens, _tokens6, action_s = cc.forward_policy(
                    policy, layer5, None, batch, noise_b, cc.to_gpu(structure_b, device), "add"
                )
                records.append(
                    {
                        "group": group_name,
                        "episode": int(frame["episode"]),
                        "step": int(frame["step"]),
                        "label": frame["label"],
                        "patch": "token_structure",
                        "seed_tag": 1,
                        "edit_l2": float(np.linalg.norm(structure_b.astype(np.float64))),
                        **cc.action_metrics(action_s, source_b, donor_b),
                    }
                )

                if group_name == "paint":
                    chunks = {
                        "base": source_action,
                        "occluded_image": donor_action,
                    }
                    if feature_action is not None:
                        chunks["occlusion_features"] = feature_action
                    body = int(frame["body"])
                    for name, chunk in chunks.items():
                        try:
                            collect.observe_state(env, frame["sim_state"])
                            sim = collect.sim_of(env)
                            span = collect.object_qpos_span(sim, body)
                            start = object_xyz(sim, body, span)
                            before = task_succeeded(env)
                            for action in np.asarray(chunk)[:rollout_steps]:
                                step_env(env, action)
                            end = object_xyz(collect.sim_of(env), body, span)
                            shift = None if start is None or end is None else float(np.linalg.norm(end - start))
                            rollouts.append(
                                {
                                    "episode": int(frame["episode"]),
                                    "step": int(frame["step"]),
                                    "chunk": name,
                                    "success_before": before,
                                    "success_after": task_succeeded(env),
                                    "bowl_shift_m": shift,
                                }
                            )
                        except Exception as exc:
                            rollouts.append(
                                {
                                    "episode": int(frame["episode"]),
                                    "step": int(frame["step"]),
                                    "chunk": name,
                                    "success_before": None,
                                    "success_after": None,
                                    "bowl_shift_m": None,
                                    "error": str(exc),
                                }
                            )

            for frame in frames:
                covered = cover_pair(frame)
                if covered is None:
                    continue
                present, present_wrist, absent, absent_wrist, outside = covered
                noise = 5000 + int(frame["episode"]) * 100 + int(frame["step"])

                def covered_forward(agent_image, wrist_image):
                    batch = cc.batch_from_images(
                        collect, pre, agent_image, wrist_image, frame["state"], frame["sentence"], policy
                    )
                    _t5, _t6, action = cc.forward_policy(policy, layer5, layer6, batch, noise, None, "base")
                    return action

                present_action = covered_forward(present, present_wrist)
                absent_action = covered_forward(absent, absent_wrist)
                cover_rmse = float(np.sqrt(np.mean((present_action - absent_action) ** 2)))
                covers.append(
                    {
                        "group": group_name,
                        "episode": int(frame["episode"]),
                        "step": int(frame["step"]),
                        "label": frame["label"],
                        "outside": outside,
                        "action_rmse": cover_rmse,
                        "text": cc.full_cover_interpretation(outside, cover_rmse),
                    }
                )

        def mean_gap(group: str, patch: str, seed_tag: int) -> float | None:
            chosen = [
                row["fraction_gap_closed"]
                for row in records
                if row["group"] == group and row["patch"] == patch and row["seed_tag"] == seed_tag
            ]
            return cc.mean_defined(chosen)

        paint_full = bootstrap_interval(per_frame_gaps([row for row in records if row["group"] == "paint"], "full_tokens", 0), rng, draws)
        paint_structure = bootstrap_interval(
            per_frame_gaps([row for row in records if row["group"] == "paint"], "token_structure", 0), rng, draws
        )
        paint_features = bootstrap_interval(
            per_frame_gaps([row for row in records if row["group"] == "paint"], "occlusion_features", 0), rng, draws
        )
        seed_a_full = mean_gap("paint", "full_tokens", 0)
        seed_b_full = mean_gap("paint", "full_tokens", 1)
        seed_a_structure = mean_gap("paint", "token_structure", 0)
        seed_b_structure = mean_gap("paint", "token_structure", 1)

        print("\nGAP 1  reconstruction is not causation", flush=True)
        line = claim_line(r2, seed_a_full, paint_features["mean"])
        print(line, flush=True)
        print(f"  full prefix {interval_text(paint_full)}", flush=True)
        print(f"  token structure {interval_text(paint_structure)}", flush=True)
        def pct(value: float | None) -> str:
            return "undefined" if value is None else f"{100 * value:.1f}%"

        print(f"  pooled {pct(mean_gap('paint', 'pooled', 0))}", flush=True)
        print(f"  strict features {interval_text(paint_features)}", flush=True)
        print(f"  random {pct(mean_gap('paint', 'random_remap', 0))}", flush=True)
        print(f"  color features {pct(mean_gap('paint', 'color_features', 0))}", flush=True)
        print(f"  rejected leader {pct(mean_gap('paint', 'rejected_leader', 0))}", flush=True)

        def absence_text(group: str) -> str:
            vals = [
                row["source_to_donor_rmse"]
                for row in records
                if row["group"] == group and row["patch"] == "absence_rmse"
            ]
            if not vals:
                return "none"
            return f"mean RMSE {float(np.mean(vals)):.4f} over {len(vals)} frames"

        print("\nGAP 2  gripper hiding, separate from the paint blob", flush=True)
        print("  census", census, flush=True)
        print("  removing the bowl, paint frames:", absence_text("paint"), flush=True)
        print("  removing the bowl, gripper frames:", absence_text("gripper"), flush=True)
        if not held_frames:
            print("  No held frame produced controlled images in this scan. The paint edit is still a paint blob.", flush=True)
        else:
            print(f"  held frames {[ (f['episode'], f['step'], f['label']) for f in held_frames ]}", flush=True)
            print(f"  held paint full prefix {pct(mean_gap('gripper', 'full_tokens', 0))}", flush=True)
            print(f"  held strict features {pct(mean_gap('gripper', 'occlusion_features', 0))}", flush=True)
        for cover in covers:
            print(
                f"  cover ep {cover['episode']} step {cover['step']} {cover['label']} "
                f"outside={cover['outside']:.5f} rmse={cover['action_rmse']:.5f}",
                flush=True,
            )
            print("   ", cover["text"], flush=True)

        print("\nGAP 3  more than a point estimate", flush=True)
        print(f"  paint frames {len(replicate_frames)} | held frames {len(held_frames)} | one task, task {task_id}", flush=True)
        print(f"  seed A full {pct(seed_a_full)} | seed B full {pct(seed_b_full)}", flush=True)
        print(f"  seed A structure {pct(seed_a_structure)} | seed B structure {pct(seed_b_structure)}", flush=True)
        print(f"  rollout steps {rollout_steps}", flush=True)
        for row in rollouts:
            print(
                f"  ep {row['episode']} step {row['step']} {row['chunk']}: "
                f"success {row['success_before']} -> {row['success_after']} bowl_shift_m={row.get('bowl_shift_m')} {row.get('error', '')}",
                flush=True,
            )

        print("\nGAP 4  figure", flush=True)
        wrist = replicate_frames[0]["built"]["images"]
        agent_recolor = cc.changed_fraction(wrist["base"][0], wrist["recolor"][0])
        wrist_recolor = cc.changed_fraction(wrist["base"][1], wrist["recolor"][1])
        agent_paint = cc.changed_fraction(wrist["base"][0], wrist["occluded"][0])
        print(f"  agent recolor {agent_recolor:.4f} | wrist recolor {wrist_recolor:.4f} | agent paint {agent_paint:.4f}", flush=True)
        print("  sheet", sheet, flush=True)
        print("  zoom", sheet.with_name(sheet.stem + "_zoom.png"), flush=True)

        print("\nGAP 5  one rule", flush=True)
        print(f"  passed {[row['feature'] for row in passed_rows] or 'none'}", flush=True)
        if leader_row is not None:
            print(
                f"  largest |occlusion| feature {leader_row['feature']} "
                f"occlusion={leader_row['occlusion']:.3f} slab={leader_row['slab']:.3f} "
                f"color={leader_row['color']:.3f} absence={leader_row['absence']:.3f} "
                f"specificity={leader_row['specificity']:.3f}",
                flush=True,
            )
            print("  rejected because:", "; ".join(leader_reasons) if leader_reasons else "it passes the rule", flush=True)

        summary = {
            "task": sentence,
            "task_id": task_id,
            "census": census,
            "replicate_frames": [
                {"episode": int(frame["episode"]), "step": int(frame["step"]), "label": frame["label"]}
                for frame in replicate_frames
            ],
            "held_frames": [
                {"episode": int(frame["episode"]), "step": int(frame["step"]), "label": frame["label"]}
                for frame in held_frames
            ],
            "passed": passed_rows,
            "leader": leader_row,
            "leader_reasons": leader_reasons,
            "claim": line,
            "intervals": {"full_tokens": paint_full, "token_structure": paint_structure, "occlusion_features": paint_features},
            "seed_replicate": {
                "full_a": seed_a_full,
                "full_b": seed_b_full,
                "structure_a": seed_a_structure,
                "structure_b": seed_b_structure,
            },
            "covers": covers,
            "rollouts": rollouts,
            "records": records,
            "figure": str(sheet),
        }
        (dest / "summary.json").write_text(json.dumps(cc.json_ready(summary), indent=2))
        print("\nWrote", dest / "summary.json", flush=True)
        print("RESULT_DIR", dest, flush=True)
    finally:
        env.close()


if __name__ == "__main__":
    main()
