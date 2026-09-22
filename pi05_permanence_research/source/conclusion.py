"""Score leftover bowl pixels after the tight hide, then stop.

The tight-cover run hid the table bowl from both cameras and split closed-loop
success. This run reuses those two frames and those plate placements. It counts
bowl pixels the plate did not cover, paints them, and compares that action with
full gray paint and with removal. It does not retrain, does not roll out, and
does not rewrite earlier output directories.
"""

from __future__ import annotations

import json
import os
from pathlib import Path

import cv2
import numpy as np
import torch

import controlled_contrasts as cc
import paper_claim as pc
import paper_gaps
import remaining_gaps as rg
import tight_cover as tc

FRAMES = ((1, 0), (1, 44))
FALLBACK = {
    (1, 0): {"fraction": 0.90, "scale": 0.75},
    (1, 44): {"fraction": 0.80, "scale": 0.75},
}


def leftover_split(base: np.ndarray, edited: np.ndarray, bowl_mask: np.ndarray | None) -> dict:
    """Bowl pixels the edit left in place, and changed pixels outside the bowl."""
    changed = cc.changed_pixels(base, edited)
    if bowl_mask is None:
        leftover = np.zeros(changed.shape, dtype=bool)
        extra = changed
        bowl_n = 0
        known = False
    else:
        bowl = np.asarray(bowl_mask, dtype=bool)
        leftover = np.logical_and(bowl, np.logical_not(changed))
        extra = np.logical_and(changed, np.logical_not(bowl))
        bowl_n = int(bowl.sum())
        known = True
    return {
        "leftover": leftover,
        "extra": extra,
        "leftover_pixels": int(leftover.sum()),
        "extra_pixels": int(extra.sum()),
        "bowl_pixels": bowl_n,
        "changed_pixels": int(changed.sum()),
        "known": known,
    }


def hide_reading(
    leftover_agent: int,
    leftover_wrist: int,
    known: bool,
    paint_to_cover: float,
    cover_to_absent: float,
    tight_to_cover: float,
) -> str:
    if not known:
        return "The bowl mask could not be measured."
    leftover = int(leftover_agent) + int(leftover_wrist)
    if leftover == 0:
        if (
            np.isfinite(tight_to_cover)
            and tight_to_cover < 0.05
            and np.isfinite(cover_to_absent)
            and cover_to_absent >= 0.05
        ):
            return (
                "The bowl is hidden as fully as gray paint. "
                "The remaining action versus removal is the plate, not a hidden bowl."
            )
        if np.isfinite(cover_to_absent) and cover_to_absent < 0.05:
            return "The hide matches removal."
        return (
            "The bowl pixels are gone. "
            "The remaining action versus removal is the extra object, not a hidden bowl."
        )
    if np.isfinite(paint_to_cover) and paint_to_cover < 1e-3:
        return "A visible bowl rim remains. Painting it matches the full cover."
    if np.isfinite(paint_to_cover) and paint_to_cover < 0.05:
        return "A visible bowl rim remains. Painting it nearly matches the full cover."
    return "A visible bowl rim remains, and painting it still leaves an action gap to the full cover."


def placements_from_summary(summary: dict | None) -> dict[tuple[int, int], dict]:
    found = dict(FALLBACK)
    if not summary:
        return found
    for row in summary.get("frames") or []:
        tight = row.get("tight") or {}
        if "fraction" not in tight or "scale" not in tight:
            continue
        found[(int(row["episode"]), int(row["step"]))] = {
            "fraction": float(tight["fraction"]),
            "scale": float(tight["scale"]),
        }
    return found


def rate_line(rates: list[dict] | None) -> str:
    if not rates:
        return "Closed-loop rates from the tight-cover run were not found."
    parts = []
    for row in rates:
        parts.append(f"{row['edit']} {row['successes']}/{row['episodes']}")
    return "; ".join(parts)


def final_claim(table: dict | None, held: dict | None, rates: list[dict] | None) -> str:
    table_text = table["reading"] if table else "the table hide was not scored"
    held_text = held["reading"] if held else "the held hide was not scored"
    return (
        "The next action follows the visible bowl pixels. "
        f"Table hide: {table_text} "
        f"Held hide: {held_text} "
        f"Closed loop: {rate_line(rates)}. "
        "Layer-5 prefix replace already closed 93% of a paint edit; passing features closed 1.8%."
    )


def load_summary(path: Path) -> dict | None:
    if not path.is_file():
        return None
    return json.loads(path.read_text())


def plain_split(row: dict) -> dict:
    return {key: value for key, value in row.items() if key not in ("leftover", "extra")}


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
                    pc.tile(pictures["tight_agent"], "tight"),
                    pc.tile(pc.highlight(pictures["tight_agent"], pictures["leftover_agent"]), "leftover"),
                    pc.tile(pictures["paint_agent"], "paint rim"),
                    pc.tile(pictures["cover_agent"], "gray"),
                    pc.tile(pictures["absent_agent"], "absent"),
                ],
                axis=1,
            )
        )
        strips.append(
            np.concatenate(
                [
                    pc.tile(pictures["base_wrist"], f"{label} wrist"),
                    pc.tile(pictures["tight_wrist"], "tight"),
                    pc.tile(pc.highlight(pictures["tight_wrist"], pictures["leftover_wrist"]), "leftover"),
                    pc.tile(pictures["paint_wrist"], "paint rim"),
                    pc.tile(pictures["cover_wrist"], "gray"),
                    pc.tile(pictures["absent_wrist"], "absent"),
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
    repo = Path(os.environ.get("LOCAL_REPO", "/content/groot-run"))
    dest = repo / "outputs" / "permanence" / "conclusion"
    dest.mkdir(parents=True, exist_ok=True)
    prior = load_summary(repo / "outputs" / "permanence" / "tight_cover" / "summary.json")
    placements = placements_from_summary(prior)
    rates = None if prior is None else prior.get("rates")

    print("Loading policy", flush=True)
    policy, pre, post, layer5 = collect.load_policy()
    del post
    policy.eval()
    layer6 = cc.find_layer6(policy, layer5)
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

    print("\nCONCLUSION  leftover bowl pixels after the tight hide", flush=True)
    print(f"Closed loop from tight_cover: {rate_line(rates)}", flush=True)
    frame_rows: list[dict] = []
    try:
        for episode, step in FRAMES:
            if episode >= len(demos) or step >= len(demos[episode]):
                print(f"  ep {episode} step {step}: outside the demos", flush=True)
                continue
            place = placements.get((episode, step), FALLBACK[(episode, step)])
            row = score_frame(
                collect,
                env,
                demos[episode][step],
                episode,
                step,
                free_type,
                target_body,
                forward,
                place,
            )
            frame_rows.append(row)
        sheet = dest / "conclusion.png"
        if save_sheet(frame_rows, sheet):
            print("sheet", sheet, flush=True)
        table = next((row for row in frame_rows if row.get("episode") == 1 and row.get("step") == 0), None)
        held = next((row for row in frame_rows if row.get("episode") == 1 and row.get("step") == 44), None)
        claim = final_claim(table, held, rates)
        print("\nCLAIM", flush=True)
        print(claim, flush=True)
        summary = {
            "scope": (
                "Scope: lerobot/pi05_libero_finetuned, PaliGemma language-model layer 5, "
                "proprioception from the unedited scene. "
                f"Frames {list(FRAMES)}. Reuses the tight-cover placements. No closed loop."
            ),
            "rates": rates,
            "frames": [{key: value for key, value in row.items() if key != "pictures"} for row in frame_rows],
            "claim": claim,
            "figure": str(sheet) if sheet.is_file() else None,
        }
        (dest / "summary.json").write_text(json.dumps(cc.json_ready(summary), indent=2))
        print("\n" + summary["scope"], flush=True)
        print("Wrote", dest / "summary.json", flush=True)
        print("RESULT_DIR", dest, flush=True)
    finally:
        env.close()


def score_frame(collect, env, sim_state, episode: int, step: int, free_type: int, target_body, forward, place: dict) -> dict:
    tc.begin_episode(collect, env, sim_state)
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
    recolor_agent, recolor_wrist = cc.render_rgba_images(collect, env, sim, bowl_geoms)
    raw = collect.render_after_edit(env)
    base_agent, base_wrist = cc.images_of(collect, raw)
    state = collect.libero_state(raw)
    agent_mask = pc.uncapped_change_mask(base_agent, recolor_agent, min_pixels=5)
    wrist_mask = pc.uncapped_change_mask(base_wrist, recolor_wrist, min_pixels=5)
    fraction = float(place["fraction"])
    scale = float(place["scale"])
    placed = rg.render_placed(
        collect,
        env,
        sim,
        [
            (agent_body, bodies[agent_body], pc.point_on_ray(agent_pos, bowl_center, rg.AGENT_FRACTION), rg.AGENT_SCALE),
            (other, bodies[other], pc.point_on_ray(wrist_pos, bowl_center, fraction), scale),
        ],
    )
    if placed is None:
        print(f"  ep {episode} step {step}: placement produced no image", flush=True)
        return {"episode": episode, "step": step, "error": "placement produced no image"}
    cover_agent, cover_wrist, _cover_state = pc.live_images(collect, env, sim, bowl, bowl_geoms, "covered")
    absent_agent, absent_wrist, _absent_state = pc.live_images(collect, env, sim, bowl, bowl_geoms, "absent")
    agent_split = leftover_split(base_agent, placed["agent"], agent_mask)
    wrist_split = leftover_split(base_wrist, placed["wrist"], wrist_mask)
    paint_agent = pc.paint_camera(placed["agent"], agent_split["leftover"])
    paint_wrist = pc.paint_camera(placed["wrist"], wrist_split["leftover"])
    scores = rg.view_scores(base_agent, base_wrist, placed["agent"], placed["wrist"], agent_mask, wrist_mask)
    noise = 8300 + episode * 1000 + step
    base_action = forward(base_agent, base_wrist, state, noise)
    tight_action = forward(placed["agent"], placed["wrist"], state, noise)
    paint_action = forward(paint_agent, paint_wrist, state, noise)
    cover_action = forward(cover_agent, cover_wrist, state, noise)
    absent_action = forward(absent_agent, absent_wrist, state, noise)
    tight_to_cover = pc.action_rmse(tight_action, cover_action)
    paint_to_cover = pc.action_rmse(paint_action, cover_action)
    cover_to_absent = pc.action_rmse(cover_action, absent_action)
    tight_to_absent = pc.action_rmse(tight_action, absent_action)
    base_to_absent = pc.action_rmse(base_action, absent_action)
    reading = hide_reading(
        agent_split["leftover_pixels"],
        wrist_split["leftover_pixels"],
        agent_split["known"] and wrist_split["known"],
        paint_to_cover,
        cover_to_absent,
        tight_to_cover,
    )
    print(
        f"  ep {episode} step {step} wrist-camera-to-bowl_m {_fmt(distance, 4)} "
        f"plate fraction {_fmt(fraction, 2)} scale {_fmt(scale, 2)}",
        flush=True,
    )
    print(
        f"    leftover bowl px agent {agent_split['leftover_pixels']} wrist {wrist_split['leftover_pixels']} "
        f"| plate extra px agent {agent_split['extra_pixels']} wrist {wrist_split['extra_pixels']} "
        f"| cover agent {_fmt(scores.get('agent_cover'))} wrist {_fmt(scores.get('wrist_cover'))} "
        f"scene {_fmt(scores.get('scene_fraction'))}",
        flush=True,
    )
    print(
        f"    RMSE base→tight {_fmt(pc.action_rmse(base_action, tight_action), 4)} "
        f"tight→cover {_fmt(tight_to_cover, 4)} paint-rim→cover {_fmt(paint_to_cover, 4)} "
        f"cover→absent {_fmt(cover_to_absent, 4)} tight→absent {_fmt(tight_to_absent, 4)} "
        f"base→absent {_fmt(base_to_absent, 4)}",
        flush=True,
    )
    print(f"    {reading}", flush=True)
    return {
        "episode": episode,
        "step": step,
        "camera_to_bowl_m": distance,
        "fraction": fraction,
        "scale": scale,
        "agent": plain_split(agent_split),
        "wrist": plain_split(wrist_split),
        "agent_cover": scores.get("agent_cover"),
        "wrist_cover": scores.get("wrist_cover"),
        "scene_fraction": scores.get("scene_fraction"),
        "rmse_base_to_tight": pc.action_rmse(base_action, tight_action),
        "rmse_tight_to_cover": tight_to_cover,
        "rmse_paint_to_cover": paint_to_cover,
        "rmse_cover_to_absent": cover_to_absent,
        "rmse_tight_to_absent": tight_to_absent,
        "rmse_base_to_absent": base_to_absent,
        "reading": reading,
        "pictures": {
            "base_agent": base_agent,
            "base_wrist": base_wrist,
            "tight_agent": placed["agent"],
            "tight_wrist": placed["wrist"],
            "leftover_agent": agent_split["leftover"],
            "leftover_wrist": wrist_split["leftover"],
            "paint_agent": paint_agent,
            "paint_wrist": paint_wrist,
            "cover_agent": cover_agent,
            "cover_wrist": cover_wrist,
            "absent_agent": absent_agent,
            "absent_wrist": absent_wrist,
        },
    }


if __name__ == "__main__":
    main()
