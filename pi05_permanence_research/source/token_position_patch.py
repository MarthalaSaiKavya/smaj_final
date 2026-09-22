#!/usr/bin/env python
"""Token-position patch on the saved controlled-contrast frames.

Rebuilds the probe frames from the contrast run and adds the real per-token
layer-5 delta on the tokens that moved most. Each fraction has two controls
of the same count: random positions, and the tokens that moved least.
Full-token and token-structure patches are the reference lines.

Does not rewrite outputs/permanence/controlled_contrasts/ or feature_count_sweep/.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Literal, assert_never

import numpy as np
import torch

import controlled_contrasts as cc

EDITED_OF = {"color": "recolor", "absence": "absent", "occlusion": "occluded"}
CAPTURE_CONDITIONS = ("base", "recolor", "absent", "occluded")
POSITION_PATCHES = ("largest", "random_positions", "least_moved")
REFERENCE_PATCHES = ("full_tokens", "token_structure")
PositionKind = Literal["largest", "random", "least"]


def parse_fractions(text: str) -> list[float]:
    fractions: list[float] = []
    seen: set[float] = set()
    for part in text.split(","):
        piece = part.strip()
        if not piece:
            continue
        value = float(piece)
        if not 0.0 < value <= 1.0:
            raise ValueError(f"Token fraction must be in (0, 1], got {value}")
        if value not in seen:
            seen.add(value)
            fractions.append(value)
    if not fractions:
        raise ValueError("CTRL_TOKEN_FRACTIONS is empty")
    return fractions


def token_count(n_tokens: int, fraction: float) -> int:
    if n_tokens < 1:
        raise ValueError("Need at least one token")
    return max(1, min(n_tokens, int(round(n_tokens * fraction))))


def movement(source: np.ndarray, donor: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Largest-first token order and the per-token L2 of donor minus source."""
    delta = donor.astype(np.float64) - source.astype(np.float64)
    distance = np.linalg.norm(delta, axis=-1)
    order = np.argsort(-distance, kind="mergesort")
    return order.astype(np.int64), distance


def select_positions(
    order: np.ndarray,
    n_tokens: int,
    count: int,
    kind: PositionKind,
    rng: np.random.Generator,
) -> np.ndarray:
    match kind:
        case "largest":
            return order[:count].astype(np.int64)
        case "least":
            return order[-count:].astype(np.int64)
        case "random":
            return rng.choice(n_tokens, size=count, replace=False).astype(np.int64)
        case _ as unexpected:
            assert_never(unexpected)


def position_delta(source: np.ndarray, donor: np.ndarray, indexes: np.ndarray) -> np.ndarray:
    """Per-token delta, zero outside `indexes`."""
    delta = np.zeros_like(source, dtype=np.float32)
    chosen = np.asarray(indexes, dtype=np.int64)
    delta[chosen] = donor[chosen].astype(np.float32) - source[chosen].astype(np.float32)
    return delta


def token_l2_share(source: np.ndarray, donor: np.ndarray, indexes: np.ndarray) -> float:
    delta = donor.astype(np.float64) - source.astype(np.float64)
    total = float(np.linalg.norm(delta))
    if total < 1e-12:
        return 0.0
    part = float(np.linalg.norm(delta[np.asarray(indexes, dtype=np.int64)]))
    return part / total


def position_verdict(
    contrast: str,
    curve: list[tuple[float, float | None, float | None, float | None]],
    structure: float | None,
) -> str:
    """Read one contrast. `curve` is (fraction, largest, random, least)."""
    usable = [
        row
        for row in curve
        if row[1] is not None and row[2] is not None and row[3] is not None
    ]
    if len(usable) < 2 or structure is None:
        return f"{contrast}: not enough token fractions to read the curve."
    head = min(usable, key=lambda row: abs(row[0] - 0.05))
    _fraction, top, random_gap, _least = head
    if structure > 0 and top >= 0.75 * structure and top - random_gap >= 0.10:
        keep = head[0]
        for fraction, largest, rand, _least_gap in usable:
            if largest >= 0.75 * structure and largest - rand >= 0.10:
                keep = fraction
                break
        return (
            f"{contrast}: the action sits in the tokens that move most. "
            f"Smallest fraction that reaches 75% of token structure: {round(100 * keep)}%."
        )
    first, last = usable[0], usable[-1]
    rose = last[1] - first[1]
    if rose >= 0.20 and first[1] < 0.5 * structure:
        return (
            f"{contrast}: closure rises from {100 * first[1]:.1f}% at {round(100 * first[0])}% of tokens "
            f"to {100 * last[1]:.1f}% at {round(100 * last[0])}%. "
            f"At 5% of tokens the largest patch closes {100 * top:.1f}%. "
            "The effect is spread across the prefix."
        )
    return (
        f"{contrast}: largest-token closure is {100 * top:.1f}% at 5% of tokens "
        f"and {100 * last[1]:.1f}% at {round(100 * last[0])}%. "
        f"Token structure is {100 * structure:.1f}%."
    )


def fraction_label(fraction: float | None) -> str:
    if fraction is None:
        return "ref"
    return f"{round(100 * fraction)}%"


def summarize(records: list[dict], contrast: str, fraction: float | None, patch: str) -> dict:
    chosen = [
        row
        for row in records
        if row["contrast"] == contrast and row["patch"] == patch and row["fraction"] == fraction
    ]
    def mean_of(key: str) -> float | None:
        if not chosen:
            return None
        return float(np.mean([row[key] for row in chosen]))

    return {
        "contrast": contrast,
        "fraction": fraction,
        "patch": patch,
        "mean_gap_closed": cc.mean_defined([row["fraction_gap_closed"] for row in chosen]),
        "mean_edit_l2": mean_of("edit_l2"),
        "mean_baseline_rmse": mean_of("source_to_donor_rmse"),
        "mean_token_l2_share": mean_of("token_l2_share"),
        "mean_n_tokens": mean_of("n_tokens"),
        "n_defined": int(sum(row["fraction_gap_closed"] is not None for row in chosen)),
        "n_pairs": len(chosen),
    }


def print_table(rows: list[dict]) -> None:
    print("\nTOKENS  fraction of action-chunk RMSE closed (base -> edited)", flush=True)
    print(
        f"{'contrast':<12}{'frac':>8}{'patch':<20}{'gap_closed':>12}{'edit_l2':>12}"
        f"{'base_rmse':>12}{'token_l2':>10}{'n':>6}",
        flush=True,
    )
    for row in rows:
        gap = row["mean_gap_closed"]
        gap_text = "undefined" if gap is None else f"{100 * gap:.2f}%"
        edit = row["mean_edit_l2"]
        base = row["mean_baseline_rmse"]
        share = row["mean_token_l2_share"]
        edit_text = "nan" if edit is None else f"{edit:.3f}"
        base_text = "nan" if base is None else f"{base:.4f}"
        share_text = "nan" if share is None else f"{100 * share:.1f}%"
        print(
            f"{row['contrast']:<12}{fraction_label(row['fraction']):>8}{row['patch']:<20}{gap_text:>12}"
            f"{edit_text:>12}{base_text:>12}{share_text:>10}{row['n_defined']:>6}",
            flush=True,
        )


def curve_for(rows: list[dict], contrast: str, fractions: list[float]):
    curve = []
    for fraction in fractions:
        def gap(patch: str, fraction: float = fraction) -> float | None:
            return next(
                row["mean_gap_closed"]
                for row in rows
                if row["contrast"] == contrast and row["fraction"] == fraction and row["patch"] == patch
            )

        curve.append((fraction, gap("largest"), gap("random_positions"), gap("least_moved")))
    return curve


def reference_gap(rows: list[dict], contrast: str, patch: str) -> float | None:
    return next(
        row["mean_gap_closed"]
        for row in rows
        if row["contrast"] == contrast and row["fraction"] is None and row["patch"] == patch
    )


def main() -> None:
    cc.configure_environment()
    if not torch.cuda.is_available():
        raise SystemExit("CUDA GPU required. Use the Colab L4 or A100 runtime.")
    collect = cc.load_collect_module()
    seed = cc.env_int("CTRL_SEED", 0)
    repo = Path(os.environ.get("LOCAL_REPO", "/content/groot-run"))
    source = repo / "outputs" / "permanence" / "controlled_contrasts"
    dest = repo / "outputs" / "permanence" / "token_position_patch"
    dest.mkdir(parents=True, exist_ok=True)
    summary_path = source / "summary.json"
    if not summary_path.is_file():
        raise SystemExit(f"Missing {summary_path}. Run the contrast cell first.")
    prior = json.loads(summary_path.read_text())
    wanted = [(int(row["episode"]), int(row["step"])) for row in prior["probe_frames"]]
    if len(wanted) < 2:
        raise SystemExit("The saved summary has fewer than two probe frames.")
    fractions = parse_fractions(os.environ.get("CTRL_TOKEN_FRACTIONS", "0.01,0.05,0.15,0.50"))
    print("Probe frames", wanted, flush=True)
    print("Token fractions", [fraction_label(item) for item in fractions], flush=True)

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
                tokens5, tokens6, action, _device = cc.capture(
                    policy, layer5, layer6, collect, pre, frame, condition, noise
                )
                packed[condition] = {"l5": tokens5, "l6": tokens6, "action": action}
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

        print("\nPIXELS  fraction of agent-view pixels that differ from base", flush=True)
        for frame in probe_frames:
            base_agent = frame["built"]["images"]["base"][0]
            parts = [
                f"{condition}={cc.changed_fraction(base_agent, frame['built']['images'][condition][0]):.4f}"
                for condition in ("recolor", "absent", "occluded")
            ]
            print(
                f"  ep {frame['episode']} step {frame['step']} {frame['label']}: " + " ".join(parts),
                flush=True,
            )

        rng = np.random.default_rng(seed)
        records: list[dict] = []
        for frame_index, (frame, packed) in enumerate(zip(probe_frames, probe_packed)):
            noise = 5000 + int(frame["episode"]) * 100 + int(frame["step"])
            source_tokens = packed["base"]["l5"]
            source_action = packed["base"]["action"]
            n_tokens = int(source_tokens.shape[0])
            agent, wrist = frame["built"]["images"]["base"]
            batch = cc.batch_from_images(collect, pre, agent, wrist, frame["state"], frame["sentence"], policy)

            def run_patch(contrast, patch, fraction, mode, payload_np, target_action, share, n_patched):
                _tokens, _tokens6, action = cc.forward_policy(
                    policy, layer5, None, batch, noise, cc.to_gpu(payload_np, device), mode
                )
                metrics = cc.action_metrics(action, source_action, target_action)
                if mode == "replace":
                    realized = float(np.linalg.norm(payload_np - source_tokens))
                else:
                    realized = float(np.linalg.norm(payload_np))
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
                order, _distance = movement(source_tokens, donor_tokens)
                structure_payload = cc.structure_delta(donor_tokens, source_tokens)
                raw_norm = float(np.linalg.norm(donor_tokens.astype(np.float64) - source_tokens.astype(np.float64)))
                structure_share = 0.0 if raw_norm < 1e-12 else float(np.linalg.norm(structure_payload) / raw_norm)
                references = {
                    "full_tokens": ("replace", donor_tokens, 1.0, n_tokens),
                    "token_structure": ("add", structure_payload, structure_share, n_tokens),
                }
                for patch in REFERENCE_PATCHES:
                    mode, payload_np, share, n_patched = references[patch]
                    run_patch(contrast, patch, None, mode, payload_np, donor_action, share, n_patched)

                for fraction in fractions:
                    count = token_count(n_tokens, fraction)
                    largest = select_positions(order, n_tokens, count, "largest", rng)
                    least = select_positions(order, n_tokens, count, "least", rng)
                    random_idx = select_positions(order, n_tokens, count, "random", rng)
                    chosen = {
                        "largest": largest,
                        "random_positions": random_idx,
                        "least_moved": least,
                    }
                    for patch in POSITION_PATCHES:
                        indexes = chosen[patch]
                        payload_np = position_delta(source_tokens, donor_tokens, indexes)
                        share = token_l2_share(source_tokens, donor_tokens, indexes)
                        run_patch(
                            contrast, patch, fraction, "add", payload_np, donor_action, share, int(indexes.size)
                        )
            print(f"Token frame {frame_index + 1}/{len(probe_frames)} done", flush=True)

        rows: list[dict] = []
        for contrast in cc.CONTRAST_NAMES:
            for patch in REFERENCE_PATCHES:
                rows.append(summarize(records, contrast, None, patch))
            for fraction in fractions:
                for patch in POSITION_PATCHES:
                    rows.append(summarize(records, contrast, fraction, patch))
        print_table(rows)

        verdicts = []
        for contrast in cc.CONTRAST_NAMES:
            text = position_verdict(
                contrast,
                curve_for(rows, contrast, fractions),
                reference_gap(rows, contrast, "token_structure"),
            )
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
            "prior_residual_floor_rmse": (prior.get("full_cover") or {}).get("action_rmse"),
            "tokens": rows,
            "verdicts": verdicts,
        }
        (dest / "summary.json").write_text(json.dumps(cc.json_ready(summary), indent=2))
        print("\nWrote", dest / "summary.json", flush=True)
        print("RESULT_DIR", dest, flush=True)
    finally:
        env.close()


if __name__ == "__main__":
    main()
