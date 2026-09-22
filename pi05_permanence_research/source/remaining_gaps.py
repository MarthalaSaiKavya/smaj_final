#!/usr/bin/env python
"""Hide the bowl from both cameras, and step unnormalized actions.

The ramekin on the agent-camera ray left the wrist image unchanged. This run
searches a second free body along the wrist-camera ray, places both bodies in
one scene when that body exists, and always forwards a composite of the two
renders. The closed loop passes each chunk through the policy postprocessor
before env.step.

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
import paper_claim as pc
import paper_gaps

FRAMES = ((1, 0), (1, 44))
AGENT_FRACTION = 0.50
AGENT_SCALE = 3.0
WRIST_FRACTIONS = (0.30, 0.50, 0.70)
WRIST_SCALES = (1.5, 3.0, 6.0)
COVER_READY = 0.8
SCENE_LIMIT = 0.5
PlacementKind = Literal["agent_only", "wrist_only", "physical_pair"]


def choose_both(candidates: list[dict]) -> dict | None:
    """Prefer a tight placement that covers the bowl in both cameras."""
    usable = [
        row
        for row in candidates
        if np.isfinite(row.get("agent_cover", np.nan)) and np.isfinite(row.get("wrist_cover", np.nan))
    ]
    if not usable:
        return None
    qualified = [
        row
        for row in usable
        if float(row["agent_cover"]) >= COVER_READY
        and float(row["wrist_cover"]) >= COVER_READY
        and _scene(row) < SCENE_LIMIT
    ]
    pool = qualified or usable
    return max(pool, key=lambda row: min(float(row["agent_cover"]), float(row["wrist_cover"])))


def choose_wrist(candidates: list[dict]) -> dict | None:
    usable = [row for row in candidates if np.isfinite(row.get("wrist_cover", np.nan))]
    if not usable:
        return None
    qualified = [row for row in usable if float(row["wrist_cover"]) >= COVER_READY and _scene(row) < SCENE_LIMIT]
    pool = qualified or usable
    return max(pool, key=lambda row: float(row["wrist_cover"]))


def _scene(row: dict) -> float:
    value = float(row.get("scene_fraction", 1.0))
    if not np.isfinite(value):
        return 1.0
    return value


def second_body(names: list[tuple[int, str]], blocked: set[int]) -> int | None:
    pool = [(body, name) for body, name in names if body not in blocked]
    for needle in ("plate", "ramekin"):
        for body, name in pool:
            if needle in name:
                return body
    if not pool:
        return None
    return pool[0][0]


def short_name(blob: str) -> str:
    text = blob.lower()
    for needle in ("ramekin", "plate", "bowl"):
        if needle in text:
            return needle
    compact = " ".join(text.split())
    return compact[:48] or "body"


def view_scores(
    base_agent: np.ndarray,
    base_wrist: np.ndarray,
    agent: np.ndarray,
    wrist: np.ndarray,
    agent_mask: np.ndarray | None,
    wrist_mask: np.ndarray | None,
) -> dict:
    agent_scene = cc.changed_fraction(base_agent, agent)
    wrist_scene = cc.changed_fraction(base_wrist, wrist)
    return {
        "agent_cover": pc.mask_cover_fraction(base_agent, agent, agent_mask),
        "wrist_cover": pc.mask_cover_fraction(base_wrist, wrist, wrist_mask),
        "scene_fraction": float(max(agent_scene, wrist_scene)),
        "agent_scene": agent_scene,
        "wrist_scene": wrist_scene,
        "agent": np.asarray(agent),
        "wrist": np.asarray(wrist),
    }


def absence_gap_closed(edit_to_absent: float, base_to_absent: float) -> float | None:
    if not np.isfinite(base_to_absent) or float(base_to_absent) <= 1e-4:
        return None
    if not np.isfinite(edit_to_absent):
        return None
    return float((float(base_to_absent) - float(edit_to_absent)) / float(base_to_absent))


def both_reading(agent_cover: float, wrist_cover: float, scene: float, composite: bool) -> str:
    covered = (
        np.isfinite(agent_cover)
        and np.isfinite(wrist_cover)
        and float(agent_cover) >= COVER_READY
        and float(wrist_cover) >= COVER_READY
        and np.isfinite(scene)
        and float(scene) < SCENE_LIMIT
    )
    if composite and covered:
        return "The composite hides the bowl from both cameras. It is two renders combined."
    if covered:
        return "One scene hides the bowl from both cameras."
    if np.isfinite(agent_cover) and float(agent_cover) >= COVER_READY and (
        not np.isfinite(wrist_cover) or float(wrist_cover) < 0.2
    ):
        return "The agent camera is covered. The wrist camera still shows the bowl."
    if np.isfinite(wrist_cover) and float(wrist_cover) >= COVER_READY and (
        not np.isfinite(agent_cover) or float(agent_cover) < 0.2
    ):
        return "The wrist camera is covered. The agent camera still shows the bowl."
    return "The bowl stays partly visible in a camera."


def mean_abs(chunk: np.ndarray) -> float:
    array = np.asarray(chunk, dtype=np.float64)
    if array.size == 0:
        return float("nan")
    return float(np.mean(np.abs(array)))


def harness_reading(raw_abs: float, post_abs: float, base_arm: float, zero_arm: float) -> str:
    scale = max(1e-3, 0.05 * abs(float(raw_abs)))
    unchanged = abs(float(post_abs) - float(raw_abs)) <= scale
    separates = float(base_arm) > max(float(zero_arm) * 1.25, float(zero_arm) + 0.02)
    if not unchanged and separates:
        return "Unnormalized actions move the arm farther than a zero action."
    if unchanged and not separates:
        return "The postprocessor left the action scale unchanged, and the arm matches a zero action."
    if unchanged:
        return "The postprocessor left the action scale unchanged."
    return "The postprocessor changed the action scale."


def parse_action(value, steps: int, width: int) -> np.ndarray | None:
    if isinstance(value, (tuple, list)):
        if not value:
            return None
        value = value[0]
    if isinstance(value, dict):
        if "action" not in value:
            return None
        value = value["action"]
    if torch.is_tensor(value):
        value = value.detach().float().cpu().numpy()
    array = np.asarray(value, dtype=np.float32)
    while array.ndim > 2 and array.shape[0] == 1:
        array = array[0]
    if array.ndim == 1 and steps == 1 and array.shape[0] == width and np.isfinite(array).all():
        return array.reshape(1, width)
    if (
        array.ndim == 2
        and array.shape[1] == width
        and 1 <= array.shape[0] <= steps
        and np.isfinite(array).all()
    ):
        return array
    return None


def apply_post(post, chunk: np.ndarray, device: str | None = None) -> np.ndarray:
    """Unnormalize a chunk. Tries a batched dict, a batched tensor, then one step at a time."""
    raw = np.asarray(chunk, dtype=np.float32)
    if raw.ndim != 2 or raw.shape[0] < 1 or raw.shape[1] < 1:
        raise RuntimeError(f"Expected an action chunk [steps, dim], got {getattr(raw, 'shape', None)}")
    steps, width = int(raw.shape[0]), int(raw.shape[1])
    errors: list[str] = []
    for place in _devices(device):
        base = torch.from_numpy(np.ascontiguousarray(raw)).to(place)
        parsed = _call_post(post, {"action": base.unsqueeze(0).clone()}, steps, width, errors, f"{place} dict")
        if parsed is not None:
            return parsed
        parsed = _call_post(post, base.unsqueeze(0).clone(), steps, width, errors, f"{place} tensor")
        if parsed is not None:
            return parsed
        parsed = _call_steps(post, base, steps, width, errors, place)
        if parsed is not None:
            return parsed
    raise RuntimeError("The postprocessor did not return an action chunk. " + " | ".join(errors))


def _devices(device: str | None) -> list[str]:
    found = ["cpu"]
    if device and not str(device).startswith("cpu") and str(device) not in found:
        found.append(str(device))
    return found


def _call_post(post, payload, steps: int, width: int, errors: list[str], label: str) -> np.ndarray | None:
    try:
        out = post(payload)
    except Exception as exc:
        errors.append(f"{label}: {type(exc).__name__}: {exc}")
        return None
    parsed = parse_action(out, steps, width)
    if parsed is None:
        errors.append(f"{label}: returned a value that is not a finite action chunk")
        return None
    return np.asarray(parsed, dtype=np.float32)


def _call_steps(post, base: torch.Tensor, steps: int, width: int, errors: list[str], place: str) -> np.ndarray | None:
    rows = []
    for index in range(steps):
        try:
            out = post({"action": base[index].unsqueeze(0).clone()})
        except Exception as exc:
            errors.append(f"{place} step {index}: {type(exc).__name__}: {exc}")
            return None
        parsed = parse_action(out, 1, width)
        if parsed is None:
            errors.append(f"{place} step {index}: returned a value that is not a finite action")
            return None
        rows.append(parsed[0])
    return np.stack(rows, axis=0).astype(np.float32)


def plain_row(row: dict | None) -> dict | None:
    if row is None:
        return None
    return {key: value for key, value in row.items() if key not in ("agent", "wrist")}


def save_both_sheet(frames: list[dict], path: Path) -> bool:
    strips = []
    for frame in frames:
        pictures = frame.get("pictures")
        if not pictures:
            continue
        label = f"e{frame['episode']}s{frame['step']}"
        strips.append(
            np.concatenate(
                [
                    pc.tile(pictures["base_agent"], f"{label} base agent"),
                    pc.tile(pictures["physical_agent"], "physical agent"),
                    pc.tile(pictures["composite_agent"], "composite agent"),
                    pc.tile(pictures["absent_agent"], "absent agent"),
                ],
                axis=1,
            )
        )
        strips.append(
            np.concatenate(
                [
                    pc.tile(pictures["base_wrist"], f"{label} base wrist"),
                    pc.tile(pictures["physical_wrist"], "physical wrist"),
                    pc.tile(pictures["composite_wrist"], "composite wrist"),
                    pc.tile(pictures["absent_wrist"], "absent wrist"),
                ],
                axis=1,
            )
        )
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


def render_placed(collect, env, sim, placements: list[tuple[int, list[int], np.ndarray, float]]):
    qpos = np.array(sim.data.qpos, copy=True)
    qvel = np.array(sim.data.qvel, copy=True)
    sizes = np.array(sim.model.geom_size, copy=True)
    try:
        fronts: dict[int, np.ndarray] = {}
        for body, geoms, point, scale in placements:
            span = collect.object_qpos_span(sim, body)
            if span is None or not geoms:
                return None
            start, stop = span
            sim.data.qpos[start : start + 3] = np.asarray(point, dtype=np.float64)
            sim.data.qpos[start + 3 : stop] = np.array([1.0, 0.0, 0.0, 0.0])
            for geom in geoms:
                sim.model.geom_size[geom] = sizes[geom] * float(scale)
        sim.data.qvel[:] = 0
        sim.forward()
        agent, wrist = cc.images_of(collect, collect.render_after_edit(env))
        for body, geoms, _point, _scale in placements:
            fronts[int(body)] = np.mean(
                [np.asarray(sim.data.geom_xpos[geom], dtype=np.float64) for geom in geoms],
                axis=0,
            )
        return {"agent": agent, "wrist": wrist, "fronts": fronts}
    finally:
        sim.model.geom_size[:] = sizes
        sim.data.qpos[:] = qpos
        sim.data.qvel[:] = qvel
        sim.forward()
        collect.render_after_edit(env)


def _fmt(value: float | None, digits: int = 3) -> str:
    if value is None or not np.isfinite(value):
        return "undefined"
    return f"{float(value):.{digits}f}"


def _print_view(name: str, row: dict, base_to_view: float | None, base_to_absent: float | None, closed: float | None) -> None:
    line = (
        f"  {name} [{row.get('kind', '')}] {row.get('body', '')} fraction {_fmt(row.get('fraction'), 2)} "
        f"scale {_fmt(row.get('scale'), 1)}: "
        f"agent cover {_fmt(row.get('agent_cover'))} wrist cover {_fmt(row.get('wrist_cover'))} "
        f"scene {_fmt(row.get('scene_fraction'))} "
        f"depth {_fmt(row.get('depth_gap'), 4)} angle {_fmt(row.get('angle_ratio'), 3)} "
        f"wrist depth {_fmt(row.get('wrist_depth_gap'), 4)} wrist angle {_fmt(row.get('wrist_angle_ratio'), 3)}"
    )
    if base_to_view is not None or base_to_absent is not None:
        line += (
            f" | RMSE base→view {_fmt(base_to_view, 4)} base→absent {_fmt(base_to_absent, 4)} "
            f"absence gap closed {pc.pct(closed)}"
        )
    print(line, flush=True)


def policy_device(policy) -> str:
    try:
        return str(next(policy.parameters()).device)
    except StopIteration:
        return "cpu"


def _motion(sim, body: int, span, arm0: np.ndarray, bowl0: np.ndarray | None) -> tuple[float, float | None]:
    arm = float(np.linalg.norm(np.asarray(sim.data.qpos[:7], dtype=np.float64) - arm0))
    bowl = paper_gaps.object_xyz(sim, body, span)
    shift = None if bowl0 is None or bowl is None else float(np.linalg.norm(bowl - bowl0))
    return arm, shift


def run_edited(
    collect,
    env,
    policy,
    pre,
    post,
    layer5,
    layer6,
    sentence: str,
    body: int,
    geoms: list[int],
    edit: pc.EditName,
    episode: int,
    horizon: int,
    replan: int,
    device: str,
    noise_base: int,
) -> dict:
    sim = collect.sim_of(env)
    span = collect.object_qpos_span(sim, body)
    arm0 = np.asarray(sim.data.qpos[:7], dtype=np.float64).copy()
    bowl0 = paper_gaps.object_xyz(sim, body, span)
    success_before = paper_gaps.task_succeeded(env)
    taken = 0
    succeeded = success_before
    error = ""
    raw_abs = None
    post_abs = None
    action_width = None
    try:
        while taken < horizon and not succeeded:
            agent, wrist, state = pc.live_images(collect, env, sim, body, geoms, edit)
            batch = cc.batch_from_images(collect, pre, agent, wrist, state, sentence, policy)
            _tokens, _tokens6, raw = cc.forward_policy(
                policy, layer5, layer6, batch, noise_base + taken, None, "base"
            )
            chunk = apply_post(post, raw, device)
            if raw_abs is None:
                raw_abs = mean_abs(raw)
                post_abs = mean_abs(chunk)
                action_width = int(chunk.shape[-1])
                print(
                    f"    ep {episode} {edit} raw mean abs {raw_abs:.4f} post mean abs {post_abs:.4f} "
                    f"raw {tuple(np.asarray(raw).shape)} post {tuple(chunk.shape)}",
                    flush=True,
                )
            acted = 0
            for action in np.asarray(chunk)[:replan]:
                if taken >= horizon:
                    break
                paper_gaps.step_env(env, action)
                taken += 1
                acted += 1
                sim = collect.sim_of(env)
                succeeded = paper_gaps.task_succeeded(env)
                if taken % 40 == 0 or succeeded:
                    arm, shift = _motion(sim, body, span, arm0, bowl0)
                    print(
                        f"    ep {episode} {edit} step {taken} success {succeeded} "
                        f"arm_move {_fmt(arm, 5)} bowl_shift_m {_fmt(shift, 6)}",
                        flush=True,
                    )
                if succeeded:
                    break
            if acted == 0:
                break
    except RuntimeError:
        raise
    except Exception as exc:
        error = str(exc)
        print(f"  ep {episode} {edit} stopped: {error}", flush=True)
    sim = collect.sim_of(env)
    arm, shift = _motion(sim, body, span, arm0, bowl0)
    print(
        f"  ep {episode} {edit}: success {success_before} -> {succeeded} steps {taken} "
        f"bowl_shift_m {_fmt(shift, 6)} arm_move {_fmt(arm, 5)} {error}",
        flush=True,
    )
    return {
        "episode": episode,
        "edit": edit,
        "success_before": success_before,
        "success_after": succeeded,
        "steps": taken,
        "bowl_shift_m": shift,
        "arm_move": arm,
        "raw_mean_abs": raw_abs,
        "post_mean_abs": post_abs,
        "action_width": action_width,
        "error": error,
    }


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

    horizon = cc.env_int("CTRL_ROLLOUT_HORIZON", 220)
    replan = cc.env_int("CTRL_REPLAN", 10)
    episodes = cc.env_int("CTRL_ROLLOUT_EPISODES", 2)
    harness_steps = cc.env_int("CTRL_HARNESS_STEPS", 40)
    scan_steps = cc.env_int("CTRL_SCAN_STEPS", 200)
    repo = Path(os.environ.get("LOCAL_REPO", "/content/groot-run"))
    dest = repo / "outputs" / "permanence" / "remaining_gaps"
    dest.mkdir(parents=True, exist_ok=True)

    print("Loading policy", flush=True)
    policy, pre, post, layer5 = collect.load_policy()
    if post is None:
        raise SystemExit("load_policy did not return a postprocessor.")
    policy.eval()
    layer6 = cc.find_layer6(policy, layer5)
    device = policy_device(policy)
    suite = benchmark.get_benchmark_dict()[collect.SUITE]()
    task = suite.get_task(0)
    sentence = task.language
    demos = paper_gaps.read_demo_states(collect.find_demo_file(task), scan_steps)
    bddl = os.path.join(get_libero_path("bddl_files"), task.problem_folder, task.bddl_file)
    env = OffScreenRenderEnv(bddl_file_name=bddl, camera_heights=256, camera_widths=256)
    env.seed(0)

    def target_body(sim):
        mapping = collect.candidate_bodies(sim, sentence)
        for candidate, candidate_geoms in mapping.items():
            if collect.object_qpos_span(sim, candidate) is not None and candidate_geoms:
                return int(candidate), list(candidate_geoms)
        return None, None

    def forward(agent, wrist, state, noise: int):
        batch = cc.batch_from_images(collect, pre, agent, wrist, state, sentence, policy)
        _tokens, _tokens6, action = cc.forward_policy(policy, layer5, layer6, batch, noise, None, "base")
        return action

    def bowl_views(sim):
        _body, geoms = target_body(sim)
        raw = collect.render_after_edit(env)
        agent, wrist = cc.images_of(collect, raw)
        state = collect.libero_state(raw)
        if not geoms:
            return agent, wrist, state, None, None
        recolor_agent, recolor_wrist = cc.render_rgba_images(collect, env, sim, geoms)
        raw = collect.render_after_edit(env)
        agent, wrist = cc.images_of(collect, raw)
        state = collect.libero_state(raw)
        return (
            agent,
            wrist,
            state,
            pc.uncapped_change_mask(agent, recolor_agent, min_pixels=5),
            pc.uncapped_change_mask(wrist, recolor_wrist, min_pixels=5),
        )

    frame_rows: list[dict] = []
    try:
        print("\nBOTH CAMERAS  agent ray, wrist ray, and a labeled composite", flush=True)
        for episode, step in FRAMES:
            if episode >= len(demos) or step >= len(demos[episode]):
                print(f"  ep {episode} step {step}: outside the demos", flush=True)
                continue
            try:
                frame_rows.append(
                    measure_frame(
                        collect,
                        env,
                        sim_state=demos[episode][step],
                        episode=episode,
                        step=step,
                        free_type=free_type,
                        target_body=target_body,
                        bowl_views=bowl_views,
                        forward=forward,
                    )
                )
            except Exception as exc:
                print(f"  ep {episode} step {step} occluder failed: {exc}", flush=True)
                frame_rows.append({"episode": episode, "step": step, "error": str(exc)})

        sheet = dest / "both_cameras.png"
        if save_both_sheet(frame_rows, sheet):
            print("sheet", sheet, flush=True)

        print("\nHARNESS  40 steps of base actions through the postprocessor, then zeros", flush=True)
        harness = None
        if demos and len(demos[0]) > 0:
            collect.observe_state(env, demos[0][0])
            sim = collect.sim_of(env)
            body, geoms = target_body(sim)
            if body is None or not geoms:
                print("  no free target body", flush=True)
            else:
                harness = run_harness(
                    collect,
                    env,
                    policy,
                    pre,
                    post,
                    layer5,
                    layer6,
                    sentence,
                    demos[0][0],
                    body,
                    geoms,
                    harness_steps,
                    replan,
                    device,
                )
                print(" ", harness["reading"], flush=True)

        print("\nCLOSED LOOP  unnormalized chunks, re-query every 10 actions", flush=True)
        print(
            f"Horizon {horizon}, episodes {episodes}. covered paints the uncapped mask gray. "
            "absent shows the teleported bowl and restores it before the step. "
            "Proprioception is the live arm.",
            flush=True,
        )
        rollouts = []
        edits: tuple[pc.EditName, ...] = ("base", "covered", "absent")
        for episode in range(episodes):
            if episode >= len(demos) or len(demos[episode]) == 0:
                continue
            for edit in edits:
                collect.observe_state(env, demos[episode][0])
                sim = collect.sim_of(env)
                body, geoms = target_body(sim)
                if body is None or not geoms:
                    print(f"  ep {episode} {edit}: no free target body", flush=True)
                    continue
                rollouts.append(
                    run_edited(
                        collect,
                        env,
                        policy,
                        pre,
                        post,
                        layer5,
                        layer6,
                        sentence,
                        body,
                        geoms,
                        edit,
                        episode,
                        horizon,
                        replan,
                        device,
                        8000 + episode * 1000,
                    )
                )

        summary = {
            "scope": (
                "Scope: lerobot/pi05_libero_finetuned, PaliGemma language-model layer 5, "
                "proprioception taken from the unedited scene on the frozen frames. "
                f"Both-camera frames: {list(FRAMES)}. "
                f"Closed loop: {episodes} episodes, {horizon} steps, re-query every {replan} actions, "
                "chunks passed through the policy postprocessor."
            ),
            "frames": [{key: value for key, value in row.items() if key != "pictures"} for row in frame_rows],
            "harness": harness,
            "rollouts": rollouts,
            "figure": str(sheet) if sheet.is_file() else None,
        }
        (dest / "summary.json").write_text(json.dumps(cc.json_ready(summary), indent=2))
        print("\n" + summary["scope"], flush=True)
        print("Wrote", dest / "summary.json", flush=True)
        print("RESULT_DIR", dest, flush=True)
    finally:
        env.close()


def measure_frame(collect, env, sim_state, episode: int, step: int, free_type: int, target_body, bowl_views, forward) -> dict:
    collect.observe_state(env, sim_state)
    sim = collect.sim_of(env)
    bowl, bowl_geoms = target_body(sim)
    if bowl is None or not bowl_geoms:
        print(f"  ep {episode} step {step}: no free target body", flush=True)
        return {"episode": episode, "step": step, "error": "no free target body"}
    bodies = pc.free_body_map(sim, free_type)
    names = pc.body_names(sim, bodies)
    named = dict(names)
    agent_body = pc.prefer_occluder(names, bowl)
    if agent_body is None or agent_body not in bodies:
        print(f"  ep {episode} step {step}: no occluder body", flush=True)
        return {"episode": episode, "step": step, "error": "no occluder body"}
    other = second_body(names, {bowl, agent_body})
    agent_cam = collect.camera_id(sim)
    wrist_cam = cc.camera_index(sim, "eye_in_hand")
    agent_pos = np.asarray(sim.data.cam_xpos[agent_cam], dtype=np.float64)
    wrist_pos = None if wrist_cam is None else np.asarray(sim.data.cam_xpos[wrist_cam], dtype=np.float64)
    bowl_center = np.mean([np.asarray(sim.data.geom_xpos[geom], dtype=np.float64) for geom in bowl_geoms], axis=0)
    distance = None if wrist_pos is None else float(np.linalg.norm(bowl_center - wrist_pos))
    base_agent, base_wrist, state, agent_mask, wrist_mask = bowl_views(sim)
    print(
        f"  ep {episode} step {step} agent {short_name(named.get(agent_body, ''))} "
        f"wrist body {short_name(named.get(other, '')) if other is not None else 'same object, separate render'} "
        f"wrist-camera-to-bowl_m {_fmt(distance, 4)}",
        flush=True,
    )

    def place(kind: PlacementKind, placements: list[tuple[int, list[int], np.ndarray, float]], fraction: float, scale: float, body: int):
        rendered = render_placed(collect, env, sim, placements)
        if rendered is None:
            return None
        row = view_scores(base_agent, base_wrist, rendered["agent"], rendered["wrist"], agent_mask, wrist_mask)
        depth, ratio = (float("nan"), float("nan"))
        wrist_depth, wrist_ratio = (float("nan"), float("nan"))
        match kind:
            case "agent_only":
                wrist_body_id = None
                if agent_body in rendered["fronts"]:
                    depth, ratio = pc.covering_ratio(
                        agent_pos, rendered["fronts"][agent_body], bowl_center, 0.04 * AGENT_SCALE
                    )
            case "wrist_only":
                wrist_body_id = body
            case "physical_pair":
                wrist_body_id = other
                if agent_body in rendered["fronts"]:
                    depth, ratio = pc.covering_ratio(
                        agent_pos, rendered["fronts"][agent_body], bowl_center, 0.04 * AGENT_SCALE
                    )
            case _ as unexpected:
                assert_never(unexpected)
        if wrist_pos is not None and wrist_body_id is not None and wrist_body_id in rendered["fronts"]:
            wrist_depth, wrist_ratio = pc.covering_ratio(
                wrist_pos, rendered["fronts"][wrist_body_id], bowl_center, 0.04 * float(scale)
            )
        row.update(
            {
                "kind": kind,
                "body": short_name(named.get(body, "")),
                "fraction": float(fraction),
                "scale": float(scale),
                "depth_gap": depth,
                "angle_ratio": ratio,
                "wrist_depth_gap": wrist_depth,
                "wrist_angle_ratio": wrist_ratio,
            }
        )
        return row

    candidates = []
    agent_point = pc.point_on_ray(agent_pos, bowl_center, AGENT_FRACTION)
    agent_row = place(
        "agent_only",
        [(agent_body, bodies[agent_body], agent_point, AGENT_SCALE)],
        AGENT_FRACTION,
        AGENT_SCALE,
        agent_body,
    )
    if agent_row is not None:
        candidates.append(agent_row)
    search_body = other if other is not None else agent_body
    if wrist_pos is not None and search_body in bodies:
        for fraction in WRIST_FRACTIONS:
            for scale in WRIST_SCALES:
                point = pc.point_on_ray(wrist_pos, bowl_center, fraction)
                row = place(
                    "wrist_only",
                    [(search_body, bodies[search_body], point, scale)],
                    fraction,
                    scale,
                    search_body,
                )
                if row is not None:
                    candidates.append(row)
    if other is not None and other in bodies and wrist_pos is not None:
        for fraction in WRIST_FRACTIONS:
            for scale in WRIST_SCALES:
                wrist_point = pc.point_on_ray(wrist_pos, bowl_center, fraction)
                row = place(
                    "physical_pair",
                    [
                        (agent_body, bodies[agent_body], agent_point, AGENT_SCALE),
                        (other, bodies[other], wrist_point, scale),
                    ],
                    fraction,
                    scale,
                    other,
                )
                if row is not None:
                    candidates.append(row)
    print(f"  searched {len(candidates)} placements", flush=True)
    physical = choose_both([row for row in candidates if row["kind"] in ("agent_only", "wrist_only", "physical_pair")])
    wrist_choice = choose_wrist([row for row in candidates if row["kind"] == "wrist_only"])
    if agent_row is None or wrist_choice is None or physical is None:
        print(f"  ep {episode} step {step}: placement search produced no image", flush=True)
        return {
            "episode": episode,
            "step": step,
            "camera_to_bowl_m": distance,
            "candidates": [plain_row(row) for row in candidates],
            "error": "placement search produced no image",
        }
    composite = view_scores(
        base_agent,
        base_wrist,
        agent_row["agent"],
        wrist_choice["wrist"],
        agent_mask,
        wrist_mask,
    )
    composite.update(
        {
            "kind": "composite",
            "body": "two renders",
            "fraction": wrist_choice["fraction"],
            "scale": wrist_choice["scale"],
            "depth_gap": agent_row.get("depth_gap"),
            "angle_ratio": agent_row.get("angle_ratio"),
            "wrist_depth_gap": wrist_choice.get("wrist_depth_gap"),
            "wrist_angle_ratio": wrist_choice.get("wrist_angle_ratio"),
        }
    )
    noise = 8000 + episode * 1000 + step
    base_action = forward(base_agent, base_wrist, state, noise)
    physical_action = forward(physical["agent"], physical["wrist"], state, noise)
    composite_action = forward(composite["agent"], composite["wrist"], state, noise)
    absent_agent, absent_wrist, _absent_state = pc.live_images(collect, env, sim, bowl, bowl_geoms, "absent")
    absent_action = forward(absent_agent, absent_wrist, state, noise)
    to_absent = pc.action_rmse(base_action, absent_action)
    physical_rmse = pc.action_rmse(base_action, physical_action)
    composite_rmse = pc.action_rmse(base_action, composite_action)
    physical_closed = absence_gap_closed(pc.action_rmse(physical_action, absent_action), to_absent)
    composite_closed = absence_gap_closed(pc.action_rmse(composite_action, absent_action), to_absent)
    _print_view("agent_only", agent_row, None, None, None)
    _print_view("wrist_only", wrist_choice, None, None, None)
    _print_view("physical", physical, physical_rmse, to_absent, physical_closed)
    _print_view("composite", composite, composite_rmse, to_absent, composite_closed)
    physical_text = both_reading(
        physical["agent_cover"], physical["wrist_cover"], physical["scene_fraction"], False
    )
    composite_text = both_reading(
        composite["agent_cover"], composite["wrist_cover"], composite["scene_fraction"], True
    )
    print(f"  physical: {physical_text}", flush=True)
    print(f"  composite: {composite_text}", flush=True)
    return {
        "episode": episode,
        "step": step,
        "camera_to_bowl_m": distance,
        "agent_only": plain_row(agent_row),
        "wrist_only": plain_row(wrist_choice),
        "physical": plain_row(physical),
        "composite": plain_row(composite),
        "rmse_base_to_absent": to_absent,
        "rmse_base_to_physical": physical_rmse,
        "rmse_base_to_composite": composite_rmse,
        "physical_absence_gap_closed": physical_closed,
        "composite_absence_gap_closed": composite_closed,
        "physical_reading": physical_text,
        "composite_reading": composite_text,
        "candidates": [plain_row(row) for row in candidates],
        "pictures": {
            "base_agent": base_agent,
            "base_wrist": base_wrist,
            "physical_agent": physical["agent"],
            "physical_wrist": physical["wrist"],
            "composite_agent": composite["agent"],
            "composite_wrist": composite["wrist"],
            "absent_agent": absent_agent,
            "absent_wrist": absent_wrist,
        },
    }


def run_harness(
    collect,
    env,
    policy,
    pre,
    post,
    layer5,
    layer6,
    sentence: str,
    state0,
    body: int,
    geoms: list[int],
    harness_steps: int,
    replan: int,
    device: str,
) -> dict:
    collect.observe_state(env, state0)
    base = run_edited(
        collect,
        env,
        policy,
        pre,
        post,
        layer5,
        layer6,
        sentence,
        body,
        geoms,
        "base",
        0,
        harness_steps,
        replan,
        device,
        7900,
    )
    collect.observe_state(env, state0)
    sim = collect.sim_of(env)
    span = collect.object_qpos_span(sim, body)
    arm0 = np.asarray(sim.data.qpos[:7], dtype=np.float64).copy()
    bowl0 = paper_gaps.object_xyz(sim, body, span)
    space = getattr(env, "action_space", None)
    width = int(base["action_width"] or 0)
    if width < 1 and space is not None and getattr(space, "shape", None):
        width = int(space.shape[0])
    if width < 1:
        width = 7
    chunk = np.zeros((max(replan, 1), width), dtype=np.float32)
    taken = 0
    while taken < harness_steps:
        for action in chunk:
            if taken >= harness_steps:
                break
            paper_gaps.step_env(env, action)
            taken += 1
    arm, shift = _motion(collect.sim_of(env), body, span, arm0, bowl0)
    print(
        f"  zeros: steps {taken} arm_move {_fmt(arm, 5)} bowl_shift_m {_fmt(shift, 6)}",
        flush=True,
    )
    reading = harness_reading(
        float(base["raw_mean_abs"] or np.nan),
        float(base["post_mean_abs"] or np.nan),
        float(base["arm_move"]),
        arm,
    )
    return {
        "base_arm_move": base["arm_move"],
        "base_bowl_shift_m": base["bowl_shift_m"],
        "zero_arm_move": arm,
        "zero_bowl_shift_m": shift,
        "raw_mean_abs": base["raw_mean_abs"],
        "post_mean_abs": base["post_mean_abs"],
        "reading": reading,
    }


if __name__ == "__main__":
    main()
