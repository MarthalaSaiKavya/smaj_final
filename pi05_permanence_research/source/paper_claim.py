#!/usr/bin/env python
"""Close the remaining gaps on the occlusion run.

The wrist mask was refused when a color swap changed more than 20% of that
camera, so two held frames still showed the bowl. This run rebuilds the mask
without that cap, paints any pixels that still differ, places another object
on the camera-to-bowl ray, rescores the feature rule on more frames, and runs
a closed-loop episode with the normal 10-step re-query.

Does not retrain and does not rewrite earlier output directories.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Literal, assert_never

import cv2
import numpy as np
import torch

import controlled_contrasts as cc
import occlusion_features as oc
import paper_gaps

PAINT_FRAMES = ((1, 0), (1, 8), (1, 15), (1, 23))
HELD_FRAMES = ((0, 44), (1, 44), (2, 40), (3, 46))
FIGURE_FRAMES = ((0, 44), (1, 44), (3, 46))
OCCLUDER_FRAMES = ((1, 0), (1, 44))
FROZEN_PAPER = (498, 366)
FROZEN_STRICT = (206, 366, 65)
EditName = Literal["base", "covered", "absent"]


def uncapped_change_mask(before: np.ndarray, after: np.ndarray, min_pixels: int = 20) -> np.ndarray | None:
    """Pixels the color swap changed, including when they fill more than 20% of the image."""
    changed = cc.changed_pixels(before, after)
    if int(changed.sum()) < int(min_pixels):
        return None
    return changed


def paint_camera(image: np.ndarray, mask: np.ndarray | None) -> np.ndarray:
    if mask is None:
        return np.array(image, copy=True)
    return cc.paint_mask(image, mask, cc.GRAY)


def diff_stats(before: np.ndarray, after: np.ndarray) -> tuple[int, int]:
    delta = np.abs(np.asarray(before).astype(np.int16) - np.asarray(after).astype(np.int16))
    changed = delta.max(axis=-1) > 0
    peak = int(delta.max()) if changed.any() else 0
    return int(changed.sum()), peak


def action_rmse(source: np.ndarray, edited: np.ndarray) -> float:
    left = np.asarray(source, dtype=np.float64)
    right = np.asarray(edited, dtype=np.float64)
    if left.size == 0 or right.size == 0:
        return float("nan")
    return float(np.sqrt(np.mean((left - right) ** 2)))


def residue_reading(pixels: int, peak: int, rmse: float) -> str:
    if pixels == 0 and rmse < 1e-3:
        return "images match and the action matches"
    if pixels == 0:
        return "images match and the action still moves"
    if peak >= 30:
        return "visible residue remains"
    if rmse < 0.05:
        return "a faint residue remains and the action gap is small"
    return "a faint residue remains and the action still moves"


def point_on_ray(camera: np.ndarray, target: np.ndarray, fraction: float) -> np.ndarray:
    origin = np.asarray(camera, dtype=np.float64)
    end = np.asarray(target, dtype=np.float64)
    return origin + float(fraction) * (end - origin)


def covering_ratio(camera: np.ndarray, front: np.ndarray, target: np.ndarray, radius: float) -> tuple[float, float]:
    to_target = np.asarray(target, dtype=np.float64) - np.asarray(camera, dtype=np.float64)
    to_front = np.asarray(front, dtype=np.float64) - np.asarray(camera, dtype=np.float64)
    target_dist = float(np.linalg.norm(to_target))
    front_dist = float(np.linalg.norm(to_front))
    if target_dist < 1e-6 or front_dist < 1e-6:
        return float("nan"), float("nan")
    cosine = float(np.clip(np.dot(to_target, to_front) / (target_dist * front_dist), -1.0, 1.0))
    angle = float(np.arccos(cosine))
    front_angle = float(np.arctan2(radius, front_dist))
    ratio = angle / front_angle if front_angle > 1e-9 else float("inf")
    return target_dist - front_dist, ratio


def prefer_occluder(names: list[tuple[int, str]], target: int) -> int | None:
    pool = [(body, name) for body, name in names if body != target]
    for needle in ("ramekin", "plate"):
        for body, name in pool:
            if needle in name:
                return body
    if not pool:
        return None
    return pool[0][0]


def overlap(found: list[int], frozen: tuple[int, ...]) -> list[int]:
    found_set = set(int(item) for item in found)
    return [int(item) for item in frozen if int(item) in found_set]


def mask_cover_fraction(before: np.ndarray, after: np.ndarray, bowl_mask: np.ndarray | None) -> float:
    if bowl_mask is None or not np.asarray(bowl_mask).any():
        return float("nan")
    changed = cc.changed_pixels(before, after)
    bowl = np.asarray(bowl_mask, dtype=bool)
    return float(np.logical_and(changed, bowl).sum() / max(int(bowl.sum()), 1))


def highlight(image: np.ndarray, mask: np.ndarray | None) -> np.ndarray:
    out = np.array(image, copy=True)
    if mask is None or not np.asarray(mask).any():
        return out
    red = np.zeros_like(out)
    red[..., 0] = 255
    chosen = np.asarray(mask, dtype=bool)
    out[chosen] = (0.45 * out[chosen] + 0.55 * red[chosen]).astype(np.uint8)
    return out


def pct(value: float | None) -> str:
    if value is None or not np.isfinite(value):
        return "undefined"
    return f"{100 * float(value):.1f}%"


def task_count(suite) -> int:
    value = getattr(suite, "n_tasks", None)
    if value is None:
        value = getattr(suite, "get_num_tasks", 10)
    if callable(value):
        value = value()
    return int(value)


def choose_placement(candidates: list[dict]) -> dict | None:
    usable = [row for row in candidates if np.isfinite(row["agent_cover"])]
    if not usable:
        return None
    bounded = [row for row in usable if row["scene_fraction"] < 0.45]
    pool = bounded or usable
    return max(pool, key=lambda row: float(row["agent_cover"]))


def scope_line(task_names: list[str], feature_frames: int, episodes: int, horizon: int) -> str:
    listed = "; ".join(task_names) if task_names else "none"
    return (
        f"Scope: lerobot/pi05_libero_finetuned, PaliGemma language-model layer 5, "
        f"proprioception taken from the unedited scene. "
        f"Feature rule scored on {feature_frames} task-0 frames. "
        f"Geometry scan: {listed}. "
        f"Closed loop: {episodes} episodes, {horizon} steps, re-query every 10 actions."
    )


def main() -> None:
    cc.configure_environment()
    if not torch.cuda.is_available():
        raise SystemExit("CUDA GPU required. Use the Colab L4 or A100 runtime.")
    os.environ["MAX_STEPS"] = os.environ.get("CTRL_SCAN_STEPS", "200")
    collect = cc.load_collect_module()
    # Simulator imports stay inside main so unit tests do not load LIBERO or MuJoCo.
    from libero.libero import benchmark, get_libero_path
    from libero.libero.envs import OffScreenRenderEnv

    try:
        import mujoco

        free_type = int(mujoco.mjtJoint.mjJNT_FREE)
    except Exception:
        free_type = 0

    scan_steps = cc.env_int("CTRL_SCAN_STEPS", 200)
    scope_episodes = cc.env_int("CTRL_SCOPE_EPISODES", 3)
    scope_stride = cc.env_int("CTRL_SCOPE_STRIDE", 16)
    scope_steps = cc.env_int("CTRL_SCOPE_STEPS", 96)
    extra_frames = cc.env_int("CTRL_EXTRA_FRAMES", 4)
    horizon = cc.env_int("CTRL_ROLLOUT_HORIZON", 120)
    replan = cc.env_int("CTRL_REPLAN", 10)
    rollout_episodes = cc.env_int("CTRL_ROLLOUT_EPISODES", 2)
    occluder_scale = float(os.environ.get("CTRL_OCCLUDER_SCALE", "3"))
    min_specificity = float(os.environ.get("CTRL_MIN_SPECIFICITY", "0.5"))
    repo = Path(os.environ.get("LOCAL_REPO", "/content/groot-run"))
    dest = repo / "outputs" / "permanence" / "paper_claim"
    dest.mkdir(parents=True, exist_ok=True)
    checkpoint_path = repo / "outputs" / "permanence" / "controlled_contrasts" / "transcoder.pt"
    if not checkpoint_path.is_file():
        raise SystemExit(f"Missing transcoder at {checkpoint_path}. Run the contrast cell first.")
    prior_path = repo / "outputs" / "permanence" / "paper_gaps" / "summary.json"
    paint = list(PAINT_FRAMES)
    held = list(HELD_FRAMES)
    if prior_path.is_file():
        prior = json.loads(prior_path.read_text())
        if prior.get("replicate_frames"):
            paint = [(int(row["episode"]), int(row["step"])) for row in prior["replicate_frames"]]
        if prior.get("held_frames"):
            held = [(int(row["episode"]), int(row["step"])) for row in prior["held_frames"]]

    suite = benchmark.get_benchmark_dict()[collect.SUITE]()
    n_tasks = task_count(suite)
    task_rows = []
    print(f"Geometry scan: {n_tasks} tasks, {scope_episodes} episodes, stride {scope_stride}", flush=True)
    for task_id in range(n_tasks):
        scan_env = None
        try:
            task = suite.get_task(task_id)
            sentence = task.language
            demo_path = collect.find_demo_file(task)
            bddl = os.path.join(get_libero_path("bddl_files"), task.problem_folder, task.bddl_file)
            demos = paper_gaps.read_demo_states(demo_path, scope_steps)
            scan_env = OffScreenRenderEnv(bddl_file_name=bddl, camera_heights=256, camera_widths=256)
            scan_env.seed(0)
            ratios: list[float] = []
            hidden = 0
            held_n = 0
            for states in demos[:scope_episodes]:
                for step in range(0, len(states), scope_stride):
                    collect.observe_state(scan_env, states[step])
                    sim = collect.sim_of(scan_env)
                    result = collect.label_frame(sim, sentence, collect.candidate_bodies(sim, sentence))
                    label = result[0] if isinstance(result, tuple) else result
                    geometry = result[2] if isinstance(result, tuple) and len(result) > 2 else None
                    if label not in ("held_visible", "held_hidden") or not geometry:
                        continue
                    held_n += 1
                    depth_gap = float(geometry["depth_gap"])
                    ratio = float(geometry["angle_ratio"])
                    ratios.append(ratio)
                    if depth_gap > 0.0 and ratio < 1.0:
                        hidden += 1
            ratio_min = float(np.min(ratios)) if ratios else None
            task_rows.append(
                {
                    "task_id": task_id,
                    "task": sentence,
                    "held_samples": held_n,
                    "hidden": hidden,
                    "angle_ratio_min": ratio_min,
                }
            )
            ratio_text = "none" if ratio_min is None else f"{ratio_min:.3f}"
            print(
                f"  task {task_id}: held {held_n} hidden {hidden} angle_ratio_min {ratio_text} | {sentence}",
                flush=True,
            )
        except Exception as exc:
            print(f"  task {task_id} scan failed: {exc}", flush=True)
            task_rows.append(
                {
                    "task_id": task_id,
                    "task": "",
                    "held_samples": 0,
                    "hidden": 0,
                    "angle_ratio_min": None,
                    "error": str(exc),
                }
            )
        finally:
            if scan_env is not None:
                scan_env.close()

    print("Loading policy", flush=True)
    policy, pre, _post, layer5 = collect.load_policy()
    policy.eval()
    layer6 = cc.find_layer6(policy, layer5)
    model, mean5, std5, n_features = oc.load_transcoder(checkpoint_path)

    task0 = suite.get_task(0)
    sentence0 = task0.language
    demo0 = paper_gaps.read_demo_states(collect.find_demo_file(task0), scan_steps)
    bddl0 = os.path.join(get_libero_path("bddl_files"), task0.problem_folder, task0.bddl_file)
    env = OffScreenRenderEnv(bddl_file_name=bddl0, camera_heights=256, camera_widths=256)
    env.seed(0)

    def rebuild(demos, sentence: str, episode: int, step: int):
        if episode >= len(demos) or step >= len(demos[episode]):
            print(f"Skip ep {episode} step {step}: outside the demos", flush=True)
            return None
        print(f"Rebuilding ep {episode} step {step}", flush=True)
        row = cc.inspect_candidate(
            collect, env, collect.sim_of(env), sentence, episode, step, demos[episode][step]
        )
        if row is None:
            print(f"Skip ep {episode} step {step}: no controllable object", flush=True)
            return None
        return row

    def forward(frame, agent, wrist, noise: int):
        batch = cc.batch_from_images(collect, pre, agent, wrist, frame["state"], frame["sentence"], policy)
        _tokens, _tokens6, action = cc.forward_policy(policy, layer5, layer6, batch, noise, None, "base")
        return action

    def camera_masks(frame) -> tuple[np.ndarray | None, np.ndarray | None, str, str]:
        images = frame["built"]["images"]
        agent_mask = uncapped_change_mask(images["base"][0], images["recolor"][0])
        wrist_mask = uncapped_change_mask(images["base"][1], images["recolor"][1])
        agent_source = "recolor" if agent_mask is not None else "none"
        wrist_source = "recolor" if wrist_mask is not None else "none"
        return agent_mask, wrist_mask, agent_source, wrist_source

    try:
        frames = []
        for episode, step in paint + held:
            row = rebuild(demo0, sentence0, episode, step)
            if row is None:
                continue
            row["group"] = "paint" if (episode, step) in set(paint) else "held"
            frames.append(row)

        print("\nCOVER  wrist mask without the 20% cap", flush=True)
        print(
            f"{'group':<8}{'ep':>4}{'step':>6}{'old_w':>8}{'new_w':>8}{'left':>8}{'peak':>6}"
            f"{'old_rmse':>10}{'new_rmse':>10}{'painted':>10}  reading",
            flush=True,
        )
        cover_rows = []
        figure_images: dict[tuple[int, int], dict] = {}
        for frame in frames:
            images = frame["built"]["images"]
            base_agent, base_wrist = images["base"]
            absent_agent, absent_wrist = images["absent"]
            old_agent = frame["built"].get("agent_full")
            old_wrist = frame["built"].get("wrist_full")
            new_agent, new_wrist, agent_source, wrist_source = camera_masks(frame)
            noise = 5000 + int(frame["episode"]) * 1000 + int(frame["step"])
            old_present = (paint_camera(base_agent, old_agent), paint_camera(base_wrist, old_wrist))
            old_absent = (paint_camera(absent_agent, old_agent), paint_camera(absent_wrist, old_wrist))
            new_present = (paint_camera(base_agent, new_agent), paint_camera(base_wrist, new_wrist))
            new_absent = (paint_camera(absent_agent, new_agent), paint_camera(absent_wrist, new_wrist))
            leftover = cc.changed_pixels(new_present[0], new_absent[0])
            leftover_wrist = cc.changed_pixels(new_present[1], new_absent[1])
            left_px, left_peak = diff_stats(new_present[0], new_absent[0])
            wrist_left, wrist_peak = diff_stats(new_present[1], new_absent[1])
            if wrist_left > left_px:
                left_px, left_peak = wrist_left, wrist_peak
            painted_present = (
                paint_camera(new_present[0], leftover),
                paint_camera(new_present[1], leftover_wrist),
            )
            painted_absent = (
                paint_camera(new_absent[0], leftover),
                paint_camera(new_absent[1], leftover_wrist),
            )
            old_action = forward(frame, old_present[0], old_present[1], noise)
            old_absent_action = forward(frame, old_absent[0], old_absent[1], noise)
            new_action = forward(frame, new_present[0], new_present[1], noise)
            new_absent_action = forward(frame, new_absent[0], new_absent[1], noise)
            same_action = forward(frame, new_present[0], new_present[1], noise)
            painted_action = forward(frame, painted_present[0], painted_present[1], noise)
            painted_absent_action = forward(frame, painted_absent[0], painted_absent[1], noise)
            old_rmse = action_rmse(old_action, old_absent_action)
            new_rmse = action_rmse(new_action, new_absent_action)
            painted_rmse = action_rmse(painted_action, painted_absent_action)
            same_rmse = action_rmse(new_action, same_action)
            reading = residue_reading(left_px, left_peak, new_rmse)
            old_wrist_px = 0 if old_wrist is None else int(np.asarray(old_wrist).sum())
            new_wrist_px = 0 if new_wrist is None else int(np.asarray(new_wrist).sum())
            print(
                f"{frame['group']:<8}{int(frame['episode']):>4}{int(frame['step']):>6}"
                f"{old_wrist_px:>8}{new_wrist_px:>8}{left_px:>8}{left_peak:>6}"
                f"{old_rmse:>10.4f}{new_rmse:>10.4f}{painted_rmse:>10.4f}  {reading}",
                flush=True,
            )
            print(
                f"    mask agent {agent_source} wrist {wrist_source} | identical re-forward {same_rmse:.4f} | "
                f"painted images differ by {diff_stats(painted_present[0], painted_absent[0])[0]} agent px and "
                f"{diff_stats(painted_present[1], painted_absent[1])[0]} wrist px",
                flush=True,
            )
            key = (int(frame["episode"]), int(frame["step"]))
            cover_rows.append(
                {
                    "group": frame["group"],
                    "episode": key[0],
                    "step": key[1],
                    "old_wrist_pixels": old_wrist_px,
                    "new_wrist_pixels": new_wrist_px,
                    "leftover_pixels": left_px,
                    "leftover_peak": left_peak,
                    "old_rmse": old_rmse,
                    "new_rmse": new_rmse,
                    "painted_rmse": painted_rmse,
                    "identical_rmse": same_rmse,
                    "agent_mask_source": agent_source,
                    "wrist_mask_source": wrist_source,
                    "reading": reading,
                }
            )
            if key in set(FIGURE_FRAMES):
                figure_images[key] = {
                    "base": base_wrist,
                    "absent": absent_wrist,
                    "old": old_absent[1],
                    "new": new_absent[1],
                    "leftover": leftover_wrist,
                }

        sheet = dest / "cover_sheet.png"
        sheet_rows = save_figure(figure_images, sheet)
        if sheet_rows:
            print("sheet", sheet, flush=True)

        print("\nOCCLUDER  another object on the camera-to-bowl ray", flush=True)
        occluder_rows = []
        by_key = {(int(frame["episode"]), int(frame["step"])): frame for frame in frames}
        for key in OCCLUDER_FRAMES:
            frame = by_key.get(key)
            if frame is None:
                continue
            collect.observe_state(env, frame["sim_state"])
            sim = collect.sim_of(env)
            bodies = free_body_map(sim, free_type)
            names = body_names(sim, bodies)
            occluder_body = prefer_occluder(names, int(frame["body"]))
            if occluder_body is None:
                print(f"  ep {key[0]} step {key[1]}: no second free body", flush=True)
                continue
            bowl_mask, wrist_bowl, _agent_source, _wrist_source = camera_masks(frame)
            base_agent, base_wrist = frame["built"]["images"]["base"]
            placed_rows = []
            for fraction in (0.50, 0.65, 0.80):
                placed = place_occluder(
                    collect, env, sim, occluder_body, bodies[occluder_body], frame["geoms"], fraction, occluder_scale
                )
                if placed is None:
                    continue
                placed_rows.append(
                    {
                        "fraction": fraction,
                        "agent": placed["agent"],
                        "wrist": placed["wrist"],
                        "agent_cover": mask_cover_fraction(base_agent, placed["agent"], bowl_mask),
                        "wrist_cover": mask_cover_fraction(base_wrist, placed["wrist"], wrist_bowl),
                        "scene_fraction": cc.changed_fraction(base_agent, placed["agent"]),
                        "depth_gap": placed["depth_gap"],
                        "angle_ratio": placed["angle_ratio"],
                    }
                )
            best = choose_placement(placed_rows)
            if best is None:
                print(f"  ep {key[0]} step {key[1]}: occluder could not be placed", flush=True)
                continue
            absent_agent, absent_wrist = frame["built"]["images"]["absent"]
            noise = 5000 + int(frame["episode"]) * 1000 + int(frame["step"])
            base_action = forward(frame, base_agent, base_wrist, noise)
            hidden_action = forward(frame, best["agent"], best["wrist"], noise)
            absent_action = forward(frame, absent_agent, absent_wrist, noise)
            to_hidden = action_rmse(base_action, hidden_action)
            to_absent = action_rmse(base_action, absent_action)
            closed = None if to_absent <= 1e-4 else (to_absent - action_rmse(hidden_action, absent_action)) / to_absent
            occluder_name = dict(names).get(occluder_body, str(occluder_body))
            print(
                f"  ep {key[0]} step {key[1]} {occluder_name}: along-ray {best['fraction']:.2f} "
                f"depth_gap {best['depth_gap']:.4f} angle_ratio {best['angle_ratio']:.3f} | "
                f"bowl pixels covered agent {best['agent_cover']:.3f} wrist {best['wrist_cover']:.3f} "
                f"scene {best['scene_fraction']:.3f} | "
                f"RMSE base→occluder {to_hidden:.4f} base→absent {to_absent:.4f} "
                f"absence gap closed by occluder {pct(closed)}",
                flush=True,
            )
            occluder_rows.append(
                {
                    "episode": key[0],
                    "step": key[1],
                    "occluder": occluder_name,
                    "fraction": best["fraction"],
                    "depth_gap": best["depth_gap"],
                    "angle_ratio": best["angle_ratio"],
                    "agent_cover": best["agent_cover"],
                    "wrist_cover": best["wrist_cover"],
                    "scene_fraction": best["scene_fraction"],
                    "rmse_base_to_occluder": to_hidden,
                    "rmse_base_to_absent": to_absent,
                    "absence_gap_closed": closed,
                }
            )

        print("\nFEATURES  same rule on more task-0 frames", flush=True)
        extra = find_extra_frames(collect, env, sentence0, demo0, frames, extra_frames)
        feature_frames = [frame for frame in frames if frame["group"] == "paint"] + extra
        probe = []
        packed_rows = []
        for frame in feature_frames:
            noise = 5000 + int(frame["episode"]) * 1000 + int(frame["step"])
            packed = {}
            for condition in oc.CAPTURE:
                tokens, _tokens6, action, device = cc.capture(
                    policy, layer5, layer6, collect, pre, frame, condition, noise
                )
                packed[condition] = {"l5": tokens, "action": action}
            probe.append({name: packed[name]["l5"] for name in oc.CAPTURE})
            packed_rows.append((frame, packed, device))
            print(
                f"  captured ep {frame['episode']} step {frame['step']} {frame['label']}",
                flush=True,
            )
        scores = cc.score_probe(model, mean5, std5, probe) if probe else {}
        passed = oc.specific_ids(scores, "occlusion", oc.QUIET_FOR["occlusion"], min_specificity) if scores else np.array([], dtype=np.int64)
        passed_list = [int(item) for item in passed.tolist()]
        print(f"  passed {passed_list or 'none'}", flush=True)
        print(f"  overlap with paper-gaps {list(FROZEN_PAPER)}: {overlap(passed_list, FROZEN_PAPER) or 'none'}", flush=True)
        print(f"  overlap with strict run {list(FROZEN_STRICT)}: {overlap(passed_list, FROZEN_STRICT) or 'none'}", flush=True)
        oc.print_feature_table(
            "PASSED",
            oc.feature_rows(passed, scores, "occlusion", oc.QUIET_FOR["occlusion"]) if scores else [],
        )
        feature_records = score_patches(
            policy, layer5, collect, pre, model, mean5, std5, n_features, packed_rows, passed
        )
        print_patch_means(feature_records)

        print("\nSECOND TASK  one covered frame outside task 0", flush=True)
        second = second_task_cover(
            collect, suite, pre, policy, layer5, layer6, scan_steps, get_libero_path, OffScreenRenderEnv
        )
        if second is None:
            print("  no second-task frame", flush=True)
        else:
            print(
                f"  task {second['task_id']}: {second['task']}\n"
                f"  ep {second['episode']} step {second['step']} new wrist px {second['new_wrist_pixels']} "
                f"leftover {second['leftover_pixels']} peak {second['leftover_peak']} "
                f"new RMSE {second['new_rmse']:.4f} painted RMSE {second['painted_rmse']:.4f} | {second['reading']}",
                flush=True,
            )

        print("\nCLOSED LOOP  re-query every 10 actions", flush=True)
        print(
            "The sim keeps the real bowl. covered paints the uncapped mask gray. "
            "absent shows the teleported bowl and then restores it before the step. "
            "Proprioception is the live unedited arm.",
            flush=True,
        )
        rollouts = run_closed_loop(
            collect,
            env,
            policy,
            pre,
            layer5,
            layer6,
            sentence0,
            demo0,
            rollout_episodes,
            horizon,
            replan,
        )

        summary = {
            "scope": scope_line([row["task"] for row in task_rows], len(feature_frames), rollout_episodes, horizon),
            "geometry": task_rows,
            "covers": cover_rows,
            "occluders": occluder_rows,
            "features": {
                "passed": passed_list,
                "overlap_paper_gaps": overlap(passed_list, FROZEN_PAPER),
                "overlap_strict": overlap(passed_list, FROZEN_STRICT),
                "frames": [
                    {"episode": int(frame["episode"]), "step": int(frame["step"]), "label": frame["label"]}
                    for frame in feature_frames
                ],
                "patches": patch_means(feature_records),
            },
            "second_task": second,
            "rollouts": rollouts,
            "figure": str(sheet) if sheet_rows else None,
        }
        (dest / "summary.json").write_text(json.dumps(cc.json_ready(summary), indent=2))
        print("\n" + summary["scope"], flush=True)
        print("Wrote", dest / "summary.json", flush=True)
        print("RESULT_DIR", dest, flush=True)
    finally:
        env.close()


def free_body_map(sim, free_type: int) -> dict[int, list[int]]:
    model = sim.model
    movable: set[int] = set()
    for body in range(int(model.nbody)):
        start = int(model.body_jntadr[body])
        count = int(model.body_jntnum[body])
        for joint in range(start, start + count):
            if joint >= 0 and int(model.jnt_type[joint]) == int(free_type):
                movable.add(body)
                break
    bodies: dict[int, list[int]] = {}
    for geom in range(int(model.ngeom)):
        body = int(model.geom_bodyid[geom])
        if body in movable:
            bodies.setdefault(body, []).append(geom)
    return bodies


def body_names(sim, bodies: dict[int, list[int]]) -> list[tuple[int, str]]:
    model = sim.model
    named = []
    for body, geoms in bodies.items():
        blob = " ".join((model.geom_id2name(geom) or "") for geom in geoms).lower()
        blob += " " + (model.body_id2name(body) or "").lower()
        named.append((body, blob))
    return named


def place_occluder(collect, env, sim, body: int, geoms: list[int], bowl_geoms: list[int], fraction: float, scale: float):
    span = collect.object_qpos_span(sim, body)
    if span is None or not geoms or not bowl_geoms:
        return None
    camera = np.asarray(sim.data.cam_xpos[collect.camera_id(sim)], dtype=np.float64)
    bowl = np.mean([np.asarray(sim.data.geom_xpos[geom], dtype=np.float64) for geom in bowl_geoms], axis=0)
    point = point_on_ray(camera, bowl, fraction)
    qpos = np.array(sim.data.qpos, copy=True)
    qvel = np.array(sim.data.qvel, copy=True)
    sizes = np.array(sim.model.geom_size, copy=True)
    start, stop = span
    try:
        sim.data.qpos[start : start + 3] = point
        sim.data.qpos[start + 3 : stop] = np.array([1.0, 0.0, 0.0, 0.0])
        sim.data.qvel[:] = 0
        for geom in geoms:
            sim.model.geom_size[geom] = sizes[geom] * float(scale)
        sim.forward()
        agent, wrist = cc.images_of(collect, collect.render_after_edit(env))
        front = np.mean([np.asarray(sim.data.geom_xpos[geom], dtype=np.float64) for geom in geoms], axis=0)
        depth_gap, ratio = covering_ratio(camera, front, bowl, 0.04 * float(scale))
        return {"agent": agent, "wrist": wrist, "depth_gap": depth_gap, "angle_ratio": ratio}
    finally:
        sim.model.geom_size[:] = sizes
        sim.data.qpos[:] = qpos
        sim.data.qvel[:] = qvel
        sim.forward()
        collect.render_after_edit(env)


def find_extra_frames(collect, env, sentence: str, demos, existing: list, limit: int) -> list:
    have = {(int(frame["episode"]), int(frame["step"])) for frame in existing}
    found = []
    for episode in (4, 6, 8, 10, 12):
        if episode >= len(demos):
            continue
        for step in (0, 24, 48):
            if len(found) >= limit or (episode, step) in have or step >= len(demos[episode]):
                continue
            row = cc.inspect_candidate(collect, env, collect.sim_of(env), sentence, episode, step, demos[episode][step])
            if row is None or row["label"] == "held_hidden":
                continue
            if str(row["label"]).startswith("held_"):
                continue
            wrist = row["built"].get("wrist_full")
            agent = row["built"].get("agent_full")
            if wrist is None or agent is None:
                continue
            row["group"] = "extra"
            found.append(row)
            print(f"  extra ep {episode} step {step} {row['label']}", flush=True)
    return found


def score_patches(policy, layer5, collect, pre, model, mean5, std5, n_features, packed_rows, passed):
    records = []
    rng = np.random.default_rng(0)
    frozen_sets = {
        "frozen_paper": np.asarray(FROZEN_PAPER, dtype=np.int64),
        "frozen_strict": np.asarray(FROZEN_STRICT, dtype=np.int64),
    }
    for frame, packed, device in packed_rows:
        noise = 5000 + int(frame["episode"]) * 1000 + int(frame["step"])
        source_tokens = packed["base"]["l5"]
        source_action = packed["base"]["action"]
        donor_tokens = packed["occluded"]["l5"]
        donor_action = packed["occluded"]["action"]
        agent, wrist = frame["built"]["images"]["base"]
        batch = cc.batch_from_images(collect, pre, agent, wrist, frame["state"], frame["sentence"], policy)
        code_delta = cc.encode_tokens(model, donor_tokens, mean5, std5) - cc.encode_tokens(
            model, source_tokens, mean5, std5
        )

        def run_patch(name: str, mode: str, payload_np: np.ndarray) -> None:
            _tokens, _tokens6, action = cc.forward_policy(
                policy, layer5, None, batch, noise, cc.to_gpu(payload_np, device), mode
            )
            metrics = cc.action_metrics(action, source_action, donor_action)
            records.append({"patch": name, "episode": int(frame["episode"]), "step": int(frame["step"]), **metrics})

        run_patch("full_tokens", "replace", donor_tokens.astype(np.float32))
        if passed.size:
            feature_payload = oc.decoded_patch(model, code_delta, passed, std5, None)
            pool = np.setdiff1d(np.arange(n_features), passed)
            random_ids = rng.choice(pool, size=int(passed.size), replace=False).astype(np.int64)
            moved = cc.remap_features(cc.keep_features(code_delta, passed), passed, random_ids)
            random_payload = cc.match_l2(cc.decode_delta(model, moved, std5), feature_payload)
            run_patch("occlusion_features", "add", feature_payload)
            run_patch("random_remap", "add", random_payload)
        for name, feature_ids in frozen_sets.items():
            usable = feature_ids[feature_ids < n_features]
            if usable.size == 0:
                continue
            run_patch(name, "add", oc.decoded_patch(model, code_delta, usable, std5, None))
    return records


def patch_means(records: list[dict]) -> list[dict]:
    names = []
    for row in records:
        if row["patch"] not in names:
            names.append(row["patch"])
    means = []
    for name in names:
        chosen = [row["fraction_gap_closed"] for row in records if row["patch"] == name]
        means.append({"patch": name, "mean_gap_closed": cc.mean_defined(chosen), "n": len(chosen)})
    return means


def print_patch_means(records: list[dict]) -> None:
    print(f"{'patch':<22}{'gap_closed':>12}{'n':>6}", flush=True)
    for row in patch_means(records):
        print(f"{row['patch']:<22}{pct(row['mean_gap_closed']):>12}{row['n']:>6}", flush=True)


def tile(image: np.ndarray, text: str) -> np.ndarray:
    picture = np.asarray(image)
    if picture.shape[0] != 256 or picture.shape[1] != 256:
        picture = cv2.resize(picture, (256, 256), interpolation=cv2.INTER_NEAREST)
    canvas = cv2.copyMakeBorder(picture, 22, 0, 0, 0, cv2.BORDER_CONSTANT, value=(255, 255, 255))
    cv2.putText(canvas, text, (4, 16), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 0), 1, cv2.LINE_AA)
    return canvas


def save_figure(figure_images: dict, path: Path) -> bool:
    strips = []
    for key in FIGURE_FRAMES:
        pictures = figure_images.get(key)
        if pictures is None:
            continue
        box = paper_gaps.zoom_box(pictures["leftover"], 36, pictures["base"].shape[0], pictures["base"].shape[1])
        zoom = paper_gaps.enlarge(paper_gaps.crop(highlight(pictures["base"], pictures["leftover"]), box), 4)
        tiles = [
            tile(pictures["base"], f"e{key[0]}s{key[1]} wrist"),
            tile(pictures["absent"], "absent"),
            tile(pictures["old"], "old cover"),
            tile(pictures["new"], "uncapped cover"),
            tile(zoom, "residue zoom"),
        ]
        strips.append(np.concatenate(tiles, axis=1))
    if not strips:
        return False
    width = max(strip.shape[1] for strip in strips)
    padded = []
    for strip in strips:
        if strip.shape[1] < width:
            pad = np.full((strip.shape[0], width - strip.shape[1], 3), 255, dtype=np.uint8)
            strip = np.concatenate([strip, pad], axis=1)
        padded.append(strip)
    sheet = np.concatenate(padded, axis=0)
    cv2.imwrite(str(path), cv2.cvtColor(sheet, cv2.COLOR_RGB2BGR))
    return True


def second_task_cover(collect, suite, pre, policy, layer5, layer6, scan_steps, get_libero_path, env_cls):
    if task_count(suite) < 2:
        return None
    task = suite.get_task(1)
    sentence = task.language
    demos = paper_gaps.read_demo_states(collect.find_demo_file(task), scan_steps)
    if not demos or len(demos[0]) == 0:
        return None
    bddl = os.path.join(get_libero_path("bddl_files"), task.problem_folder, task.bddl_file)
    env = env_cls(bddl_file_name=bddl, camera_heights=256, camera_widths=256)
    env.seed(0)
    try:
        row = cc.inspect_candidate(collect, env, collect.sim_of(env), sentence, 0, 0, demos[0][0])
        if row is None:
            return None
        images = row["built"]["images"]
        agent_mask = uncapped_change_mask(images["base"][0], images["recolor"][0])
        wrist_mask = uncapped_change_mask(images["base"][1], images["recolor"][1])
        present = (paint_camera(images["base"][0], agent_mask), paint_camera(images["base"][1], wrist_mask))
        absent = (paint_camera(images["absent"][0], agent_mask), paint_camera(images["absent"][1], wrist_mask))
        leftover = cc.changed_pixels(present[0], absent[0])
        leftover_wrist = cc.changed_pixels(present[1], absent[1])
        left_px, left_peak = diff_stats(present[0], absent[0])
        wrist_left, wrist_peak = diff_stats(present[1], absent[1])
        if wrist_left > left_px:
            left_px, left_peak = wrist_left, wrist_peak
        painted_present = (paint_camera(present[0], leftover), paint_camera(present[1], leftover_wrist))
        painted_absent = (paint_camera(absent[0], leftover), paint_camera(absent[1], leftover_wrist))

        def once(agent, wrist):
            batch = cc.batch_from_images(collect, pre, agent, wrist, row["state"], row["sentence"], policy)
            _tokens, _tokens6, action = cc.forward_policy(policy, layer5, layer6, batch, 6100, None, "base")
            return action

        new_rmse = action_rmse(once(present[0], present[1]), once(absent[0], absent[1]))
        painted_rmse = action_rmse(once(painted_present[0], painted_present[1]), once(painted_absent[0], painted_absent[1]))
        return {
            "task_id": 1,
            "task": sentence,
            "episode": 0,
            "step": 0,
            "new_wrist_pixels": 0 if wrist_mask is None else int(np.asarray(wrist_mask).sum()),
            "leftover_pixels": left_px,
            "leftover_peak": left_peak,
            "new_rmse": new_rmse,
            "painted_rmse": painted_rmse,
            "reading": residue_reading(left_px, left_peak, new_rmse),
        }
    finally:
        env.close()


def live_images(collect, env, sim, body: int, geoms: list[int], kind: EditName):
    raw = collect.render_after_edit(env)
    agent, wrist = cc.images_of(collect, raw)
    state = collect.libero_state(raw)
    match kind:
        case "base":
            return agent, wrist, state
        case "covered":
            recolor_agent, recolor_wrist = cc.render_rgba_images(collect, env, sim, geoms)
            raw = collect.render_after_edit(env)
            agent, wrist = cc.images_of(collect, raw)
            state = collect.libero_state(raw)
            agent_mask = uncapped_change_mask(agent, recolor_agent, min_pixels=5)
            wrist_mask = uncapped_change_mask(wrist, recolor_wrist, min_pixels=5)
            return paint_camera(agent, agent_mask), paint_camera(wrist, wrist_mask), state
        case "absent":
            qpos = np.array(sim.data.qpos, copy=True)
            qvel = np.array(sim.data.qvel, copy=True)
            if collect.move_object_away(sim, body) is None:
                return agent, wrist, state
            gone_agent, gone_wrist = cc.images_of(collect, collect.render_after_edit(env))
            sim.data.qpos[:] = qpos
            sim.data.qvel[:] = qvel
            sim.forward()
            raw = collect.render_after_edit(env)
            return gone_agent, gone_wrist, collect.libero_state(raw)
        case _ as unexpected:
            assert_never(unexpected)


def run_closed_loop(collect, env, policy, pre, layer5, layer6, sentence, demos, episodes: int, horizon: int, replan: int):
    edits: tuple[EditName, ...] = ("base", "covered", "absent")
    rows = []
    for episode in range(episodes):
        if episode >= len(demos) or len(demos[episode]) == 0:
            continue
        for edit in edits:
            collect.observe_state(env, demos[episode][0])
            sim = collect.sim_of(env)
            mapping = collect.candidate_bodies(sim, sentence)
            body = None
            geoms = None
            for candidate, candidate_geoms in mapping.items():
                if collect.object_qpos_span(sim, candidate) is not None:
                    body = int(candidate)
                    geoms = list(candidate_geoms)
                    break
            if body is None or not geoms:
                print(f"  ep {episode} {edit}: no free target body", flush=True)
                continue
            span = collect.object_qpos_span(sim, body)
            start = paper_gaps.object_xyz(sim, body, span)
            success_before = paper_gaps.task_succeeded(env)
            taken = 0
            succeeded = success_before
            error = ""
            try:
                while taken < horizon and not succeeded:
                    agent, wrist, state = live_images(collect, env, sim, body, geoms, edit)
                    batch = cc.batch_from_images(collect, pre, agent, wrist, state, sentence, policy)
                    noise = 7000 + episode * 1000 + taken
                    _tokens, _tokens6, chunk = cc.forward_policy(policy, layer5, layer6, batch, noise, None, "base")
                    acted = 0
                    for action in np.asarray(chunk)[:replan]:
                        if taken >= horizon:
                            break
                        paper_gaps.step_env(env, action)
                        taken += 1
                        acted += 1
                        sim = collect.sim_of(env)
                        if paper_gaps.task_succeeded(env):
                            break
                    if acted == 0:
                        break
                    succeeded = paper_gaps.task_succeeded(env)
                    print(f"    ep {episode} {edit} step {taken} success {succeeded}", flush=True)
            except Exception as exc:
                error = str(exc)
                print(f"  ep {episode} {edit} stopped: {error}", flush=True)
            end = paper_gaps.object_xyz(collect.sim_of(env), body, span)
            shift = None if start is None or end is None else float(np.linalg.norm(end - start))
            print(
                f"  ep {episode} {edit}: success {success_before} -> {succeeded} "
                f"steps {taken} bowl_shift_m {shift} {error}",
                flush=True,
            )
            rows.append(
                {
                    "episode": episode,
                    "edit": edit,
                    "success_before": success_before,
                    "success_after": succeeded,
                    "steps": taken,
                    "bowl_shift_m": shift,
                    "error": error,
                }
            )
    return rows


if __name__ == "__main__":
    main()
