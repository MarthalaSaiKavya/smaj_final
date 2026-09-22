#!/usr/bin/env python
"""Score the tight both-camera hide, and roll it out from a fresh reset.

The previous selector kept the first full cover, so the plate that fills 42%
of the wrist was forwarded and the pair that fills 17% was not. This run
forwards the smallest full cover, including a closer and smaller plate on the
held frame, then rolls out base, gray paint, absence, and that tight plate.
Each episode is reset and the done flag is cleared before the first step.

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
import remaining_gaps as rg

FRAMES = ((1, 0), (1, 44))
LARGE_FRACTION = 0.30
LARGE_SCALE = 1.5
TIGHT_FRACTIONS = (0.55, 0.70, 0.80, 0.90)
TIGHT_SCALES = (0.75, 1.0, 1.5)
RollEdit = Literal["base", "covered", "absent", "physical"]


def choose_tight(candidates: list[dict]) -> dict | None:
    """Among full covers, keep the one that changes the least of either image."""
    usable = [
        row
        for row in candidates
        if np.isfinite(row.get("agent_cover", np.nan)) and np.isfinite(row.get("wrist_cover", np.nan))
    ]
    if not usable:
        return None
    full = [
        row
        for row in usable
        if float(row["agent_cover"]) >= rg.COVER_READY and float(row["wrist_cover"]) >= rg.COVER_READY
    ]
    pool = full or usable
    return min(pool, key=lambda row: (rg._scene(row), -min(float(row["agent_cover"]), float(row["wrist_cover"]))))


def tight_reading(agent_cover: float, wrist_cover: float, scene: float, distance: float | None) -> str:
    full = (
        np.isfinite(agent_cover)
        and np.isfinite(wrist_cover)
        and float(agent_cover) >= rg.COVER_READY
        and float(wrist_cover) >= rg.COVER_READY
    )
    if full and np.isfinite(scene) and float(scene) < rg.SCENE_LIMIT:
        return "A tight scene hides the bowl from both cameras."
    if full and np.isfinite(scene):
        dist = "an unknown distance" if distance is None or not np.isfinite(distance) else f"{float(distance):.3f} m"
        return (
            f"Both cameras lose the bowl and the plate fills {100 * float(scene):.0f}% of a frame. "
            f"The wrist camera is {dist} from the bowl."
        )
    if np.isfinite(agent_cover) and float(agent_cover) >= rg.COVER_READY and (
        not np.isfinite(wrist_cover) or float(wrist_cover) < 0.2
    ):
        return "The agent camera is covered. The wrist camera still shows the bowl."
    return "The bowl stays partly visible in a camera."


def owners_of(env) -> list:
    found = []
    seen: set[int] = set()
    pending = [env]
    while pending:
        owner = pending.pop()
        if owner is None or id(owner) in seen:
            continue
        seen.add(id(owner))
        found.append(owner)
        for attr in ("env", "unwrapped"):
            inner = getattr(owner, attr, None)
            if inner is not None and inner is not owner:
                pending.append(inner)
    return found


def clear_episode(env) -> None:
    """Clear the done flag and the step counter so the next edit can step."""
    for owner in owners_of(env):
        if hasattr(owner, "done"):
            owner.done = False
        for name in ("timestep", "_elapsed_steps"):
            if hasattr(owner, name):
                setattr(owner, name, 0)


def episode_done(env) -> bool:
    for owner in owners_of(env):
        if bool(getattr(owner, "done", False)):
            return True
    return False


def task_succeeded(env) -> bool:
    for owner in owners_of(env):
        for name in ("check_success", "_check_success"):
            fn = getattr(owner, name, None)
            if callable(fn) and bool(fn()):
                return True
    return False


def step_done(outcome) -> bool:
    if not isinstance(outcome, tuple):
        return False
    if len(outcome) == 5:
        return bool(outcome[2] or outcome[3])
    if len(outcome) == 4:
        return bool(outcome[2])
    return False


def step_action(env, action: np.ndarray) -> bool:
    vector = np.asarray(action, dtype=np.float64).reshape(-1)
    space = getattr(env, "action_space", None)
    width = int(space.shape[0]) if space is not None and getattr(space, "shape", None) else vector.size
    vector = vector[:width]
    outcome = env.step(vector)
    if not isinstance(outcome, tuple) or len(outcome) not in (4, 5):
        raise RuntimeError(f"env.step returned {type(outcome)}")
    return step_done(outcome) or episode_done(env)


def begin_episode(collect, env, state) -> None:
    if hasattr(env, "reset"):
        try:
            env.reset()
        except Exception as exc:
            print(f"  reset skipped: {exc}", flush=True)
    collect.observe_state(env, state)
    clear_episode(env)


def rate_rows(rows: list[dict]) -> list[dict]:
    edits: list[str] = []
    for row in rows:
        if row["edit"] not in edits:
            edits.append(row["edit"])
    summary = []
    for edit in edits:
        chosen = [row for row in rows if row["edit"] == edit]
        summary.append(
            {
                "edit": edit,
                "episodes": len(chosen),
                "successes": sum(1 for row in chosen if row["success_after"]),
                "no_step": sum(1 for row in chosen if int(row["steps"]) == 0),
            }
        )
    return summary


def plain_row(row: dict | None) -> dict | None:
    if row is None:
        return None
    return {key: value for key, value in row.items() if key not in ("agent", "wrist")}


def save_sheet(frames: list[dict], path: Path) -> bool:
    strips = []
    for frame in frames:
        pictures = frame.get("pictures")
        if not pictures:
            continue
        label = f"e{frame['episode']}s{frame['step']}"
        strips.append(
            np.concatenate(
                [
                    pc.tile(pictures["base_agent"], f"{label} base"),
                    pc.tile(pictures["large_agent"], "large agent"),
                    pc.tile(pictures["tight_agent"], "tight agent"),
                    pc.tile(pictures["absent_agent"], "absent agent"),
                ],
                axis=1,
            )
        )
        strips.append(
            np.concatenate(
                [
                    pc.tile(pictures["base_wrist"], f"{label} wrist"),
                    pc.tile(pictures["large_wrist"], "large wrist"),
                    pc.tile(pictures["tight_wrist"], "tight wrist"),
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
    cv2.imwrite(str(path), cv2.cvtColor(np.concatenate(padded, axis=0), cv2.COLOR_RGB2BGR))
    return True


def _fmt(value: float | None, digits: int = 3) -> str:
    if value is None or not np.isfinite(value):
        return "undefined"
    return f"{float(value):.{digits}f}"


def _print_forward(name: str, row: dict, base_to_view: float, base_to_absent: float, closed: float | None) -> None:
    print(
        f"  {name} fraction {_fmt(row.get('fraction'), 2)} scale {_fmt(row.get('scale'), 2)}: "
        f"agent cover {_fmt(row.get('agent_cover'))} wrist cover {_fmt(row.get('wrist_cover'))} "
        f"scene {_fmt(row.get('scene_fraction'))} | "
        f"RMSE base→view {_fmt(base_to_view, 4)} base→absent {_fmt(base_to_absent, 4)} "
        f"absence gap closed {pc.pct(closed)}",
        flush=True,
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

    horizon = cc.env_int("CTRL_ROLLOUT_HORIZON", 220)
    replan = cc.env_int("CTRL_REPLAN", 10)
    episodes = cc.env_int("CTRL_ROLLOUT_EPISODES", 10)
    scan_steps = cc.env_int("CTRL_SCAN_STEPS", 200)
    repo = Path(os.environ.get("LOCAL_REPO", "/content/groot-run"))
    dest = repo / "outputs" / "permanence" / "tight_cover"
    dest.mkdir(parents=True, exist_ok=True)

    print("Loading policy", flush=True)
    policy, pre, post, layer5 = collect.load_policy()
    if post is None:
        raise SystemExit("load_policy did not return a postprocessor.")
    policy.eval()
    layer6 = cc.find_layer6(policy, layer5)
    device = rg.policy_device(policy)
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

    frame_rows: list[dict] = []
    tight_params = {"fraction": 0.70, "scale": 1.5}
    try:
        print("\nTIGHT COVER  smallest full cover of both cameras", flush=True)
        for episode, step in FRAMES:
            if episode >= len(demos) or step >= len(demos[episode]):
                print(f"  ep {episode} step {step}: outside the demos", flush=True)
                continue
            row = score_frame(
                collect,
                env,
                demos[episode][step],
                episode,
                step,
                free_type,
                target_body,
                forward,
            )
            frame_rows.append(row)
            if episode == 1 and step == 0 and row.get("tight"):
                tight_params = {
                    "fraction": float(row["tight"]["fraction"]),
                    "scale": float(row["tight"]["scale"]),
                }
        sheet = dest / "tight_cover.png"
        if save_sheet(frame_rows, sheet):
            print("sheet", sheet, flush=True)

        print("\nCLOSED LOOP  reset before every edit, unnormalized chunks", flush=True)
        print(
            f"Horizon {horizon}, episodes {min(episodes, len(demos))}, re-query every {replan}. "
            f"physical re-places the table-frame tight pair "
            f"(fraction {tight_params['fraction']:.2f}, scale {tight_params['scale']:.2f}) on the current rays. "
            "covered paints the uncapped mask gray. absent teleports the bowl and restores it before the step.",
            flush=True,
        )
        rollouts = run_rollouts(
            collect,
            env,
            policy,
            pre,
            post,
            layer5,
            layer6,
            sentence,
            demos,
            target_body,
            free_type,
            tight_params,
            min(episodes, len(demos)),
            horizon,
            replan,
            device,
        )
        rates = rate_rows(rollouts)
        print("\nSUCCESS", flush=True)
        for rate in rates:
            print(
                f"  {rate['edit']}: {rate['successes']}/{rate['episodes']} success, "
                f"{rate['no_step']} episodes with no step",
                flush=True,
            )
        summary = {
            "scope": (
                "Scope: lerobot/pi05_libero_finetuned, PaliGemma language-model layer 5. "
                f"Tight-cover frames: {list(FRAMES)}. "
                f"Rollout plate fraction {tight_params['fraction']:.2f}, scale {tight_params['scale']:.2f}. "
                f"Closed loop: {min(episodes, len(demos))} episodes, {horizon} steps, "
                f"re-query every {replan}, reset and done flag cleared before each edit."
            ),
            "frames": [{key: value for key, value in row.items() if key != "pictures"} for row in frame_rows],
            "tight_params": tight_params,
            "rollouts": rollouts,
            "rates": rates,
            "figure": str(sheet) if sheet.is_file() else None,
        }
        (dest / "summary.json").write_text(json.dumps(cc.json_ready(summary), indent=2))
        print("\n" + summary["scope"], flush=True)
        print("Wrote", dest / "summary.json", flush=True)
        print("RESULT_DIR", dest, flush=True)
    finally:
        env.close()


def score_frame(collect, env, sim_state, episode: int, step: int, free_type: int, target_body, forward) -> dict:
    begin_episode(collect, env, sim_state)
    sim = collect.sim_of(env)
    bowl, bowl_geoms = target_body(sim)
    if bowl is None or not bowl_geoms:
        print(f"  ep {episode} step {step}: no free target body", flush=True)
        return {"episode": episode, "step": step, "error": "no free target body"}
    bodies = pc.free_body_map(sim, free_type)
    names = dict(pc.body_names(sim, bodies))
    agent_body = pc.prefer_occluder(list(names.items()), bowl)
    other = rg.second_body(list(names.items()), {bowl, agent_body} if agent_body is not None else {bowl})
    if agent_body is None or other is None or agent_body not in bodies or other not in bodies:
        print(f"  ep {episode} step {step}: need a ramekin and a second body", flush=True)
        return {"episode": episode, "step": step, "error": "need a ramekin and a second body"}
    wrist_cam = cc.camera_index(sim, "eye_in_hand")
    if wrist_cam is None:
        print(f"  ep {episode} step {step}: wrist camera was not found", flush=True)
        return {"episode": episode, "step": step, "error": "wrist camera was not found"}
    agent_pos = np.asarray(sim.data.cam_xpos[collect.camera_id(sim)], dtype=np.float64)
    wrist_pos = np.asarray(sim.data.cam_xpos[wrist_cam], dtype=np.float64)
    bowl_center = np.mean([np.asarray(sim.data.geom_xpos[geom], dtype=np.float64) for geom in bowl_geoms], axis=0)
    distance = float(np.linalg.norm(bowl_center - wrist_pos))
    raw = collect.render_after_edit(env)
    base_agent, base_wrist = cc.images_of(collect, raw)
    state = collect.libero_state(raw)
    recolor_agent, recolor_wrist = cc.render_rgba_images(collect, env, sim, bowl_geoms)
    raw = collect.render_after_edit(env)
    base_agent, base_wrist = cc.images_of(collect, raw)
    state = collect.libero_state(raw)
    agent_mask = pc.uncapped_change_mask(base_agent, recolor_agent, min_pixels=5)
    wrist_mask = pc.uncapped_change_mask(base_wrist, recolor_wrist, min_pixels=5)
    print(
        f"  ep {episode} step {step} wrist-camera-to-bowl_m {_fmt(distance, 4)}",
        flush=True,
    )

    def one(fraction: float, scale: float) -> dict | None:
        rendered = rg.render_placed(
            collect,
            env,
            sim,
            [
                (agent_body, bodies[agent_body], pc.point_on_ray(agent_pos, bowl_center, rg.AGENT_FRACTION), rg.AGENT_SCALE),
                (other, bodies[other], pc.point_on_ray(wrist_pos, bowl_center, fraction), scale),
            ],
        )
        if rendered is None:
            return None
        row = rg.view_scores(base_agent, base_wrist, rendered["agent"], rendered["wrist"], agent_mask, wrist_mask)
        row.update({"fraction": float(fraction), "scale": float(scale), "kind": "physical_pair"})
        return row

    candidates = []
    large = one(LARGE_FRACTION, LARGE_SCALE)
    if large is not None:
        candidates.append(large)
    for fraction in TIGHT_FRACTIONS:
        for scale in TIGHT_SCALES:
            row = one(fraction, scale)
            if row is not None:
                candidates.append(row)
    tight = choose_tight(candidates)
    if large is None or tight is None:
        print(f"  ep {episode} step {step}: placement search produced no image", flush=True)
        return {"episode": episode, "step": step, "camera_to_bowl_m": distance, "error": "placement search produced no image"}
    noise = 8100 + episode * 1000 + step
    base_action = forward(base_agent, base_wrist, state, noise)
    large_action = forward(large["agent"], large["wrist"], state, noise)
    tight_action = forward(tight["agent"], tight["wrist"], state, noise)
    absent_agent, absent_wrist, _absent_state = pc.live_images(collect, env, sim, bowl, bowl_geoms, "absent")
    absent_action = forward(absent_agent, absent_wrist, state, noise)
    to_absent = pc.action_rmse(base_action, absent_action)
    large_rmse = pc.action_rmse(base_action, large_action)
    tight_rmse = pc.action_rmse(base_action, tight_action)
    large_closed = rg.absence_gap_closed(pc.action_rmse(large_action, absent_action), to_absent)
    tight_closed = rg.absence_gap_closed(pc.action_rmse(tight_action, absent_action), to_absent)
    _print_forward("large", large, large_rmse, to_absent, large_closed)
    _print_forward("tight", tight, tight_rmse, to_absent, tight_closed)
    text = tight_reading(tight["agent_cover"], tight["wrist_cover"], tight["scene_fraction"], distance)
    print(f"  tight: {text}", flush=True)
    return {
        "episode": episode,
        "step": step,
        "camera_to_bowl_m": distance,
        "large": plain_row(large),
        "tight": plain_row(tight),
        "rmse_base_to_absent": to_absent,
        "rmse_base_to_large": large_rmse,
        "rmse_base_to_tight": tight_rmse,
        "large_absence_gap_closed": large_closed,
        "tight_absence_gap_closed": tight_closed,
        "tight_reading": text,
        "candidates": [plain_row(row) for row in candidates],
        "pictures": {
            "base_agent": base_agent,
            "base_wrist": base_wrist,
            "large_agent": large["agent"],
            "large_wrist": large["wrist"],
            "tight_agent": tight["agent"],
            "tight_wrist": tight["wrist"],
            "absent_agent": absent_agent,
            "absent_wrist": absent_wrist,
        },
    }


def current_pair(collect, env, sim, agent_body, agent_geoms, wrist_body, wrist_geoms, bowl_geoms, fraction: float, scale: float):
    wrist_cam = cc.camera_index(sim, "eye_in_hand")
    if wrist_cam is None or not bowl_geoms:
        return None
    agent_pos = np.asarray(sim.data.cam_xpos[collect.camera_id(sim)], dtype=np.float64)
    wrist_pos = np.asarray(sim.data.cam_xpos[wrist_cam], dtype=np.float64)
    bowl_center = np.mean([np.asarray(sim.data.geom_xpos[geom], dtype=np.float64) for geom in bowl_geoms], axis=0)
    return rg.render_placed(
        collect,
        env,
        sim,
        [
            (agent_body, agent_geoms, pc.point_on_ray(agent_pos, bowl_center, rg.AGENT_FRACTION), rg.AGENT_SCALE),
            (wrist_body, wrist_geoms, pc.point_on_ray(wrist_pos, bowl_center, fraction), scale),
        ],
    )


def rollout_images(
    kind: RollEdit,
    collect,
    env,
    sim,
    body: int,
    geoms: list[int],
    pair,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    match kind:
        case "base" | "covered" | "absent":
            return pc.live_images(collect, env, sim, body, geoms, kind)
        case "physical":
            agent, wrist, state = pc.live_images(collect, env, sim, body, geoms, "base")
            if pair is None:
                return agent, wrist, state
            placed = current_pair(collect, env, sim, *pair)
            if placed is None:
                return agent, wrist, state
            return placed["agent"], placed["wrist"], state
        case _ as unexpected:
            assert_never(unexpected)


def run_rollouts(
    collect,
    env,
    policy,
    pre,
    post,
    layer5,
    layer6,
    sentence: str,
    demos,
    target_body,
    free_type: int,
    tight_params: dict,
    episodes: int,
    horizon: int,
    replan: int,
    device: str,
) -> list[dict]:
    edits: tuple[RollEdit, ...] = ("base", "covered", "absent", "physical")
    rows = []
    for episode in range(episodes):
        if episode >= len(demos) or len(demos[episode]) == 0:
            continue
        for edit in edits:
            begin_episode(collect, env, demos[episode][0])
            sim = collect.sim_of(env)
            body, geoms = target_body(sim)
            if body is None or not geoms:
                print(f"  ep {episode} {edit}: no free target body", flush=True)
                continue
            bodies = pc.free_body_map(sim, free_type)
            names = dict(pc.body_names(sim, bodies))
            agent_body = pc.prefer_occluder(list(names.items()), body)
            other = rg.second_body(list(names.items()), {body, agent_body} if agent_body is not None else {body})
            pair = None
            if edit == "physical" and (agent_body is None or other is None):
                print(f"  ep {episode} physical: no second body, using the live image", flush=True)
            if agent_body is not None and other is not None and agent_body in bodies and other in bodies:
                pair = (
                    agent_body,
                    bodies[agent_body],
                    other,
                    bodies[other],
                    geoms,
                    float(tight_params["fraction"]),
                    float(tight_params["scale"]),
                )
            span = collect.object_qpos_span(sim, body)
            arm0 = np.asarray(sim.data.qpos[:7], dtype=np.float64).copy()
            bowl0 = paper_gaps.object_xyz(sim, body, span)
            success_before = task_succeeded(env)
            taken = 0
            succeeded = success_before
            error = ""
            env_done = False
            raw_abs = None
            post_abs = None
            try:
                while taken < horizon and not succeeded and not episode_done(env):
                    sim = collect.sim_of(env)
                    agent, wrist, state = rollout_images(edit, collect, env, sim, body, geoms, pair)
                    batch = cc.batch_from_images(collect, pre, agent, wrist, state, sentence, policy)
                    _tokens, _tokens6, raw = cc.forward_policy(
                        policy, layer5, layer6, batch, 8200 + episode * 1000 + taken, None, "base"
                    )
                    chunk = rg.apply_post(post, raw, device)
                    if raw_abs is None:
                        raw_abs = rg.mean_abs(raw)
                        post_abs = rg.mean_abs(chunk)
                        print(
                            f"    ep {episode} {edit} raw mean abs {raw_abs:.4f} post mean abs {post_abs:.4f}",
                            flush=True,
                        )
                    acted = 0
                    for action in np.asarray(chunk)[:replan]:
                        if taken >= horizon or episode_done(env):
                            break
                        env_done = step_action(env, action)
                        taken += 1
                        acted += 1
                        sim = collect.sim_of(env)
                        succeeded = task_succeeded(env)
                        if taken % 40 == 0 or succeeded or env_done:
                            arm, shift = rg._motion(sim, body, span, arm0, bowl0)
                            print(
                                f"    ep {episode} {edit} step {taken} success {succeeded} "
                                f"env_done {env_done} arm_move {_fmt(arm, 5)} bowl_shift_m {_fmt(shift, 6)}",
                                flush=True,
                            )
                        if succeeded or env_done:
                            break
                    if acted == 0:
                        break
            except RuntimeError:
                raise
            except Exception as exc:
                error = str(exc)
                print(f"  ep {episode} {edit} stopped: {error}", flush=True)
            sim = collect.sim_of(env)
            arm, shift = rg._motion(sim, body, span, arm0, bowl0)
            print(
                f"  ep {episode} {edit}: success {success_before} -> {succeeded} steps {taken} "
                f"bowl_shift_m {_fmt(shift, 6)} arm_move {_fmt(arm, 5)} env_done {env_done} {error}",
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
                    "arm_move": arm,
                    "env_done": env_done,
                    "raw_mean_abs": raw_abs,
                    "post_mean_abs": post_abs,
                    "error": error,
                }
            )
    return rows


if __name__ == "__main__":
    main()
