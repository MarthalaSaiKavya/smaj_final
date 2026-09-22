#!/usr/bin/env python
"""Explain the cover gaps and the inert rollout.

On each saved frame, compare three forwards: the natural covered pair, the same
covered image twice, and each camera swapped alone. A second pass records how
far the gripper is from hiding the bowl. A third pass compares the first 10
actions with the rest of the chunk, then steps the arm so a dead simulator
step is visible.

Does not retrain and does not rewrite earlier output directories.
"""

from __future__ import annotations

import json
import os
from pathlib import Path

import numpy as np
import torch

import controlled_contrasts as cc
import paper_gaps

PAINT_FRAMES = ((1, 0), (1, 8), (1, 15), (1, 23))
HELD_FRAMES = ((0, 44), (1, 44), (2, 40), (3, 46))


def cover_reading(outside: float, rmse: float) -> str:
    if outside >= 0.002:
        return "images still differ outside the cover"
    if rmse < 1e-3:
        return "images agree and the action agrees"
    if rmse < 0.05:
        return "images agree and the action gap is small"
    return "images agree and the action still moves"


def diff_stats(before: np.ndarray, after: np.ndarray) -> tuple[int, int]:
    delta = np.abs(np.asarray(before).astype(np.int16) - np.asarray(after).astype(np.int16))
    changed = delta.max(axis=-1) > 0
    peak = int(delta.max()) if changed.any() else 0
    return int(changed.sum()), peak


def window_rmse(source: np.ndarray, edited: np.ndarray, count: int) -> float:
    left = np.asarray(source, dtype=np.float64)[:count]
    right = np.asarray(edited, dtype=np.float64)[:count]
    if left.size == 0 or right.size == 0:
        return float("nan")
    return float(np.sqrt(np.mean((left - right) ** 2)))


def action_rmse(source: np.ndarray, edited: np.ndarray) -> float:
    return window_rmse(source, edited, max(np.asarray(source).shape[0], 1))


def main() -> None:
    cc.configure_environment()
    if not torch.cuda.is_available():
        raise SystemExit("CUDA GPU required. Use the Colab L4 or A100 runtime.")
    os.environ["MAX_STEPS"] = os.environ.get("CTRL_SCAN_STEPS", "200")
    collect = cc.load_collect_module()
    scan_episodes = cc.env_int("CTRL_SCAN_EPISODES", 20)
    scan_steps = cc.env_int("CTRL_SCAN_STEPS", 200)
    stride = cc.env_int("CTRL_GEOMETRY_STRIDE", 4)
    rollout_steps = cc.env_int("CTRL_ROLLOUT_STEPS", 10)
    repo = Path(os.environ.get("LOCAL_REPO", "/content/groot-run"))
    source = repo / "outputs" / "permanence" / "paper_gaps" / "summary.json"
    dest = repo / "outputs" / "permanence" / "cover_autopsy"
    dest.mkdir(parents=True, exist_ok=True)
    paint = list(PAINT_FRAMES)
    held = list(HELD_FRAMES)
    if source.is_file():
        prior = json.loads(source.read_text())
        if prior.get("replicate_frames"):
            paint = [(int(row["episode"]), int(row["step"])) for row in prior["replicate_frames"]]
        if prior.get("held_frames"):
            held = [(int(row["episode"]), int(row["step"])) for row in prior["held_frames"]]
    print("Paint frames", paint, flush=True)
    print("Held frames", held, flush=True)

    # Imported here so unit tests can load this module without LIBERO.
    from libero.libero import benchmark, get_libero_path
    from libero.libero.envs import OffScreenRenderEnv

    suite = benchmark.get_benchmark_dict()[collect.SUITE]()
    task_id = int(collect.TASK_IDS[0])
    task = suite.get_task(task_id)
    sentence = task.language
    demo_path = collect.find_demo_file(task)
    bddl = os.path.join(get_libero_path("bddl_files"), task.problem_folder, task.bddl_file)
    demos = paper_gaps.read_demo_states(demo_path, scan_steps)
    print(f"Task {task_id}: {sentence}", flush=True)

    env = OffScreenRenderEnv(bddl_file_name=bddl, camera_heights=256, camera_widths=256)
    env.seed(0)
    try:
        gaps: list[float] = []
        ratios: list[float] = []
        hidden_at_zero = 0
        held_n = 0
        print(f"Geometry scan: {min(scan_episodes, len(demos))} episodes, stride {stride}", flush=True)
        for episode, states in enumerate(demos[:scan_episodes]):
            for step in range(0, len(states), stride):
                collect.observe_state(env, states[step])
                sim = collect.sim_of(env)
                result = collect.label_frame(sim, sentence, collect.candidate_bodies(sim, sentence))
                label = result[0] if isinstance(result, tuple) else result
                geometry = result[2] if isinstance(result, tuple) and len(result) > 2 else None
                if label not in ("held_visible", "held_hidden") or not geometry:
                    continue
                held_n += 1
                depth_gap = float(geometry["depth_gap"])
                ratio = float(geometry["angle_ratio"])
                gaps.append(depth_gap)
                ratios.append(ratio)
                if depth_gap > 0.0 and ratio < 1.0:
                    hidden_at_zero += 1
        print(f"Held samples {held_n}. Hidden even with a zero depth margin: {hidden_at_zero}.", flush=True)
        if gaps:
            gap_arr = np.asarray(gaps)
            ratio_arr = np.asarray(ratios)
            print(
                f"  depth gap m: min {gap_arr.min():.4f} median {np.median(gap_arr):.4f} max {gap_arr.max():.4f}",
                flush=True,
            )
            print(
                f"  angle ratio: min {ratio_arr.min():.3f} median {np.median(ratio_arr):.3f} max {ratio_arr.max():.3f}",
                flush=True,
            )
            print(
                f"  depth gap > 0: {int((gap_arr > 0).sum())} | depth gap > 0.02: {int((gap_arr > 0.02).sum())} | "
                f"angle ratio < 1: {int((ratio_arr < 1).sum())}",
                flush=True,
            )

        print("Loading policy", flush=True)
        policy, pre, _post, layer5 = collect.load_policy()
        policy.eval()
        layer6 = cc.find_layer6(policy, layer5)

        def rebuild(episode: int, step: int):
            if episode >= len(demos) or step >= len(demos[episode]):
                raise SystemExit(f"Episode {episode} step {step} is outside the scanned demos.")
            print(f"Rebuilding ep {episode} step {step}", flush=True)
            row = cc.inspect_candidate(
                collect, env, collect.sim_of(env), sentence, episode, step, demos[episode][step]
            )
            if row is None:
                raise SystemExit(f"Could not rebuild ep {episode} step {step}.")
            return row

        frames = []
        for episode, step in paint:
            row = rebuild(episode, step)
            row["group"] = "paint"
            frames.append(row)
        for episode, step in held:
            row = rebuild(episode, step)
            row["group"] = "held"
            frames.append(row)

        def forward(frame, agent, wrist, noise: int):
            batch = cc.batch_from_images(collect, pre, agent, wrist, frame["state"], frame["sentence"], policy)
            _tokens, _tokens6, action = cc.forward_policy(policy, layer5, layer6, batch, noise, None, "base")
            return action

        rows = []
        print("\nCOVER  which camera still moves the action", flush=True)
        print(
            "agent_px and wrist_px are leftover pixels after the full cover, with the peak absolute change beside them.",
            flush=True,
        )
        print(
            "both = covered present vs covered absent. agent = agent camera only. wrist = wrist camera only.",
            flush=True,
        )
        print("same = the covered present image forwarded twice.", flush=True)
        print(
            f"{'group':<8}{'ep':>4}{'step':>6}{'agent_px':>10}{'peak':>6}{'wrist_px':>10}{'peak':>6}"
            f"{'both':>10}{'agent':>10}{'wrist':>10}{'same':>10}  reading",
            flush=True,
        )
        for frame in frames:
            covered = paper_gaps.cover_pair(frame)
            if covered is None:
                print(f"ep {frame['episode']} step {frame['step']}: no cover mask", flush=True)
                continue
            present, present_wrist, absent, absent_wrist, outside = covered
            agent_px, agent_peak = diff_stats(present, absent)
            wrist_px, wrist_peak = diff_stats(present_wrist, absent_wrist)
            noise = 5000 + int(frame["episode"]) * 100 + int(frame["step"])
            both_present = forward(frame, present, present_wrist, noise)
            both_absent = forward(frame, absent, absent_wrist, noise)
            agent_only = forward(frame, absent, present_wrist, noise)
            wrist_only = forward(frame, present, absent_wrist, noise)
            same_again = forward(frame, present, present_wrist, noise)
            both_rmse = action_rmse(both_present, both_absent)
            agent_rmse = action_rmse(both_present, agent_only)
            wrist_rmse = action_rmse(both_present, wrist_only)
            same_rmse = action_rmse(both_present, same_again)
            reading = cover_reading(outside, both_rmse)
            print(
                f"{frame['group']:<8}{int(frame['episode']):>4}{int(frame['step']):>6}"
                f"{agent_px:>10}{agent_peak:>6}{wrist_px:>10}{wrist_peak:>6}"
                f"{both_rmse:>10.4f}{agent_rmse:>10.4f}"
                f"{wrist_rmse:>10.4f}{same_rmse:>10.4f}  {reading}",
                flush=True,
            )
            rows.append(
                {
                    "group": frame["group"],
                    "episode": int(frame["episode"]),
                    "step": int(frame["step"]),
                    "label": frame["label"],
                    "outside": outside,
                    "agent_pixels": agent_px,
                    "agent_peak": agent_peak,
                    "wrist_pixels": wrist_px,
                    "wrist_peak": wrist_peak,
                    "both_rmse": both_rmse,
                    "agent_only_rmse": agent_rmse,
                    "wrist_only_rmse": wrist_rmse,
                    "identical_rmse": same_rmse,
                    "reading": reading,
                }
            )

        print("\nROLLOUT  first 10 actions versus the rest of the chunk", flush=True)
        print(
            "Read arm move against zeros. When base and occluded match zeros, those 10 steps leave the arm where it started.",
            flush=True,
        )
        print(
            "When the first-10 action RMSE is near 0 and the later RMSE is large, the chunk changes after the rollout window.",
            flush=True,
        )
        checks = [frame for frame in frames if (int(frame["episode"]), int(frame["step"])) in {(1, 0), (1, 44), (3, 46)}]
        rollout_rows = []
        for frame in checks:
            noise = 5000 + int(frame["episode"]) * 100 + int(frame["step"])
            base_tokens, _t6, base_action, _device = cc.capture(
                policy, layer5, layer6, collect, pre, frame, "base", noise
            )
            _occ_tokens, _t6b, occ_action, _device = cc.capture(
                policy, layer5, layer6, collect, pre, frame, "occluded", noise
            )
            del base_tokens
            early = window_rmse(base_action, occ_action, rollout_steps)
            late = window_rmse(
                np.asarray(base_action)[rollout_steps:],
                np.asarray(occ_action)[rollout_steps:],
                max(np.asarray(base_action).shape[0] - rollout_steps, 1),
            )
            motions = {}
            for name, chunk in (("base", base_action), ("occluded", occ_action), ("zeros", np.zeros_like(base_action))):
                collect.observe_state(env, frame["sim_state"])
                sim = collect.sim_of(env)
                start = np.asarray(sim.data.qpos[:7], dtype=np.float64).copy()
                for action in np.asarray(chunk)[:rollout_steps]:
                    paper_gaps.step_env(env, action)
                end = np.asarray(collect.sim_of(env).data.qpos[:7], dtype=np.float64).copy()
                motions[name] = float(np.linalg.norm(end - start))
            print(
                f"  ep {frame['episode']} step {frame['step']} {frame['group']}: "
                f"first {rollout_steps} action RMSE {early:.4f} | later RMSE {late:.4f} | "
                f"arm move base {motions['base']:.5f} occluded {motions['occluded']:.5f} zeros {motions['zeros']:.5f}",
                flush=True,
            )
            rollout_rows.append(
                {
                    "episode": int(frame["episode"]),
                    "step": int(frame["step"]),
                    "group": frame["group"],
                    "early_rmse": early,
                    "late_rmse": late,
                    "arm_move": motions,
                }
            )

        summary = {
            "task": sentence,
            "held_samples": held_n,
            "hidden_if_depth_margin_is_zero": hidden_at_zero,
            "depth_gap": {
                "min": float(np.min(gaps)) if gaps else None,
                "median": float(np.median(gaps)) if gaps else None,
                "max": float(np.max(gaps)) if gaps else None,
            },
            "angle_ratio": {
                "min": float(np.min(ratios)) if ratios else None,
                "median": float(np.median(ratios)) if ratios else None,
                "max": float(np.max(ratios)) if ratios else None,
            },
            "covers": rows,
            "rollouts": rollout_rows,
        }
        (dest / "summary.json").write_text(json.dumps(cc.json_ready(summary), indent=2))
        print("\nWrote", dest / "summary.json", flush=True)
        print("RESULT_DIR", dest, flush=True)
    finally:
        env.close()


if __name__ == "__main__":
    main()
