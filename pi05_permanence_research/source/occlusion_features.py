#!/usr/bin/env python
"""Find occlusion features in the saved per-token transcoder and circuit-trace them.

Reloads the transcoder from the contrast run. A feature is kept only when its
code rises on gray paint covering the bowl and stays quiet on the same paint
moved off the bowl, on the color swap, and on removing the bowl. Those features
are patched from the base frame toward the occluded frame and compared with a
same-size random patch and with the color features.

Does not retrain the transcoder and does not rewrite the contrast run.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Literal, assert_never

import numpy as np
import torch

import controlled_contrasts as cc

QUIET_FOR = {
    "occlusion": ("slab", "color", "absence"),
    "color": ("slab", "occlusion", "absence"),
}
CAPTURE = ("base", "recolor", "absent", "occluded", "slab_miss")
PATCHES = ("occlusion_features", "random_remap", "color_features")
REFERENCES = ("full_tokens", "token_structure")
PatchName = Literal["occlusion_features", "random_remap", "color_features"]


def parse_counts(text: str) -> list[int]:
    counts: list[int] = []
    seen: set[int] = set()
    for part in text.split(","):
        piece = part.strip()
        if not piece:
            continue
        value = int(piece)
        if value < 1:
            raise ValueError(f"Feature count must be positive, got {value}")
        if value not in seen:
            seen.add(value)
            counts.append(value)
    if not counts:
        raise ValueError("CTRL_OCCLUSION_COUNTS is empty")
    return counts


def prefix_counts(n_specific: int, requested: list[int]) -> list[int]:
    """Nested prefixes of the features that passed. Never invents a fallback set."""
    counts: list[int] = []
    for value in requested:
        take = min(int(value), int(n_specific))
        if take >= 1 and take not in counts:
            counts.append(take)
    return counts


def specificity_of(primary: np.ndarray, others: list[np.ndarray]) -> np.ndarray:
    primary64 = np.asarray(primary, dtype=np.float64)
    rest = [np.asarray(other, dtype=np.float64) for other in others]
    denominator = np.abs(primary64) + sum(np.abs(other) for other in rest) + 1e-8
    return np.abs(primary64) / denominator


def specific_ids(
    scores: dict[str, np.ndarray],
    primary: str,
    quiet: tuple[str, ...],
    min_specificity: float,
) -> np.ndarray:
    """Features that rise on `primary` and stay quiet on every contrast in `quiet`.

    An empty result means none passed. There is no magnitude fallback.
    """
    primary64 = np.asarray(scores[primary], dtype=np.float64)
    others = [np.asarray(scores[name], dtype=np.float64) for name in quiet]
    specificity = specificity_of(primary64, others)
    magnitude_cut = float(np.quantile(np.abs(primary64), 0.90))
    eligible = np.flatnonzero(
        (primary64 > 0) & (specificity > min_specificity) & (np.abs(primary64) >= magnitude_cut)
    )
    order = eligible[np.argsort(-primary64[eligible], kind="mergesort")]
    return order.astype(np.int64)


def feature_rows(
    feature_ids: np.ndarray,
    scores: dict[str, np.ndarray],
    primary: str,
    quiet: tuple[str, ...],
) -> list[dict]:
    primary64 = np.asarray(scores[primary], dtype=np.float64)
    others = [np.asarray(scores[name], dtype=np.float64) for name in quiet]
    specificity = specificity_of(primary64, others)
    rows = []
    for feature_id in np.asarray(feature_ids, dtype=np.int64):
        index = int(feature_id)
        row = {
            "feature": index,
            "occlusion": float(scores["occlusion"][index]),
            "slab": float(scores["slab"][index]),
            "color": float(scores["color"][index]),
            "absence": float(scores["absence"][index]),
            "specificity": float(specificity[index]),
        }
        rows.append(row)
    return rows


def nearest_rejected(
    scores: dict[str, np.ndarray],
    quiet: tuple[str, ...],
    limit: int = 8,
) -> list[dict]:
    """Largest positive occlusion scores, so a failed filter can be read."""
    primary = np.asarray(scores["occlusion"], dtype=np.float64)
    order = np.argsort(-primary, kind="mergesort")
    positive = order[primary[order] > 0][:limit]
    return feature_rows(positive.astype(np.int64), scores, "occlusion", quiet)


def load_transcoder(path: Path) -> tuple[cc.TokenTranscoder, np.ndarray, np.ndarray, int]:
    try:
        checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:
        checkpoint = torch.load(path, map_location="cpu")
    n_features = int(checkpoint["n_features"])
    model = cc.TokenTranscoder(
        int(checkpoint["dim"]),
        n_features,
        int(checkpoint["k"]),
        bool(checkpoint["predict_next"]),
    )
    model.load_state_dict(checkpoint["state_dict"])
    model.eval()
    mean5 = np.asarray(checkpoint["mean5"], dtype=np.float32)
    std5 = np.asarray(checkpoint["std5"], dtype=np.float32)
    return model, mean5, std5, n_features


def summarize(records: list[dict], count: int | None, patch: str) -> dict:
    chosen = [row for row in records if row["count"] == count and row["patch"] == patch]
    return {
        "count": count,
        "patch": patch,
        "mean_gap_closed": cc.mean_defined([row["fraction_gap_closed"] for row in chosen]),
        "mean_edit_l2": float(np.mean([row["edit_l2"] for row in chosen])) if chosen else None,
        "mean_baseline_rmse": float(np.mean([row["source_to_donor_rmse"] for row in chosen])) if chosen else None,
        "n_defined": int(sum(row["fraction_gap_closed"] is not None for row in chosen)),
        "n_pairs": len(chosen),
    }


def gap_of(rows: list[dict], count: int | None, patch: str) -> float | None:
    return next(row["mean_gap_closed"] for row in rows if row["count"] == count and row["patch"] == patch)


def occlusion_verdict(rows: list[dict], counts: list[int], n_specific: int) -> str:
    if n_specific == 0 or not counts:
        return (
            "No occlusion feature passed. None rose on paint covering the bowl while staying "
            "quiet on paint off the bowl, the color swap, and removal."
        )
    passing: list[int] = []
    lines: list[str] = []
    for count in counts:
        feature_gap = gap_of(rows, count, "occlusion_features")
        random_gap = gap_of(rows, count, "random_remap")
        color_gap = gap_of(rows, count, "color_features")
        if feature_gap is None or random_gap is None or color_gap is None:
            lines.append(f"{count} features: a control patch is missing.")
            continue
        lines.append(
            f"{count} features close {100 * feature_gap:.1f}% "
            f"(random {100 * random_gap:.1f}%, color features {100 * color_gap:.1f}%)."
        )
        if feature_gap - random_gap >= 0.10 and feature_gap - color_gap >= 0.10:
            passing.append(count)
    if passing:
        return (
            f"Smallest occlusion set that beats random and the color features by 10 points: {passing[0]}. "
            + " ".join(lines)
        )
    return "No occlusion set beat both controls by 10 points. " + " ".join(lines)


def print_feature_table(title: str, rows: list[dict]) -> None:
    print(f"\n{title}", flush=True)
    print(
        f"{'feature':>8}{'occlusion':>12}{'slab':>10}{'color':>10}{'absence':>10}{'specificity':>13}",
        flush=True,
    )
    if not rows:
        print("  none", flush=True)
        return
    for row in rows:
        print(
            f"{row['feature']:>8}{row['occlusion']:>12.3f}{row['slab']:>10.3f}"
            f"{row['color']:>10.3f}{row['absence']:>10.3f}{row['specificity']:>13.3f}",
            flush=True,
        )


def print_circuit(rows: list[dict]) -> None:
    print("\nCIRCUIT  fraction of occlusion action-chunk RMSE closed (base -> occluded)", flush=True)
    print(
        f"{'count':>8}{'patch':<22}{'gap_closed':>12}{'edit_l2':>12}{'base_rmse':>12}{'n':>6}",
        flush=True,
    )
    for row in rows:
        gap = row["mean_gap_closed"]
        gap_text = "undefined" if gap is None else f"{100 * gap:.2f}%"
        count = "ref" if row["count"] is None else str(row["count"])
        edit = row["mean_edit_l2"]
        base = row["mean_baseline_rmse"]
        edit_text = "nan" if edit is None else f"{edit:.3f}"
        base_text = "nan" if base is None else f"{base:.4f}"
        print(
            f"{count:>8}{row['patch']:<22}{gap_text:>12}{edit_text:>12}{base_text:>12}{row['n_defined']:>6}",
            flush=True,
        )


def decoded_patch(
    model: cc.TokenTranscoder,
    code_delta: np.ndarray,
    feature_ids: np.ndarray,
    std5: np.ndarray,
    reference: np.ndarray | None,
) -> np.ndarray:
    payload = cc.decode_delta(model, cc.keep_features(code_delta, feature_ids), std5)
    if reference is not None:
        payload = cc.match_l2(payload, reference)
    return payload


def main() -> None:
    cc.configure_environment()
    if not torch.cuda.is_available():
        raise SystemExit("CUDA GPU required. Use the Colab L4 or A100 runtime.")
    collect = cc.load_collect_module()
    seed = cc.env_int("CTRL_SEED", 0)
    min_specificity = float(os.environ.get("CTRL_MIN_SPECIFICITY", "0.5"))
    requested = parse_counts(os.environ.get("CTRL_OCCLUSION_COUNTS", "8,32,128"))
    repo = Path(os.environ.get("LOCAL_REPO", "/content/groot-run"))
    source = repo / "outputs" / "permanence" / "controlled_contrasts"
    dest = repo / "outputs" / "permanence" / "occlusion_features"
    dest.mkdir(parents=True, exist_ok=True)
    summary_path = source / "summary.json"
    checkpoint_path = source / "transcoder.pt"
    if not summary_path.is_file() or not checkpoint_path.is_file():
        raise SystemExit(
            "Missing the controlled-contrast run. Run that cell first so "
            f"{summary_path} and {checkpoint_path} exist."
        )
    prior = json.loads(summary_path.read_text())
    wanted = [(int(row["episode"]), int(row["step"])) for row in prior["probe_frames"]]
    if len(wanted) < 2:
        raise SystemExit("The saved summary has fewer than two probe frames.")

    model, mean5, std5, n_features = load_transcoder(checkpoint_path)
    print(
        f"Loaded transcoder dim={mean5.shape[0]} features={n_features} "
        f"specificity>{min_specificity} counts={requested}",
        flush=True,
    )
    print("Probe frames", wanted, flush=True)

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
            for condition in CAPTURE:
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

        print("\nPIXELS  fraction of agent-view pixels that differ from base", flush=True)
        for frame in probe_frames:
            base_agent = frame["built"]["images"]["base"][0]
            parts = []
            for condition in ("recolor", "absent", "occluded", "slab_miss"):
                fraction = cc.changed_fraction(base_agent, frame["built"]["images"][condition][0])
                parts.append(f"{condition}={fraction:.4f}")
            print(
                f"  ep {frame['episode']} step {frame['step']} {frame['label']}: " + " ".join(parts),
                flush=True,
            )

        probe_token_maps = [{name: item[name]["l5"] for name in CAPTURE} for item in probe_packed]
        scores = cc.score_probe(model, mean5, std5, probe_token_maps)
        occlusion_ids = specific_ids(scores, "occlusion", QUIET_FOR["occlusion"], min_specificity)
        color_ids = specific_ids(scores, "color", QUIET_FOR["color"], min_specificity)
        counts = prefix_counts(int(occlusion_ids.size), requested)
        print(
            f"\nPassed: occlusion {int(occlusion_ids.size)} | color {int(color_ids.size)} "
            f"| patch counts {counts or 'none'}",
            flush=True,
        )
        passed_rows = feature_rows(occlusion_ids, scores, "occlusion", QUIET_FOR["occlusion"])
        print_feature_table("OCCLUSION FEATURES that passed", passed_rows)
        if occlusion_ids.size == 0:
            print_feature_table(
                "LARGEST occlusion scores that did not pass",
                nearest_rejected(scores, QUIET_FOR["occlusion"]),
            )

        rng = np.random.default_rng(seed)
        random_plan: dict[int, np.ndarray] = {}
        for count in counts:
            chosen = occlusion_ids[:count]
            pool = np.setdiff1d(np.arange(n_features), chosen)
            random_plan[count] = rng.choice(pool, size=int(chosen.size), replace=False).astype(np.int64)

        records: list[dict] = []
        for frame_index, (frame, packed) in enumerate(zip(probe_frames, probe_packed)):
            noise = 5000 + int(frame["episode"]) * 100 + int(frame["step"])
            source_tokens = packed["base"]["l5"]
            source_action = packed["base"]["action"]
            donor_tokens = packed["occluded"]["l5"]
            donor_action = packed["occluded"]["action"]
            agent, wrist = frame["built"]["images"]["base"]
            batch = cc.batch_from_images(collect, pre, agent, wrist, frame["state"], frame["sentence"], policy)
            code_delta = cc.encode_tokens(model, donor_tokens, mean5, std5) - cc.encode_tokens(
                model, source_tokens, mean5, std5
            )

            def run_patch(patch: str, count: int | None, mode: str, payload_np: np.ndarray) -> None:
                _tokens, _tokens6, action = cc.forward_policy(
                    policy, layer5, None, batch, noise, cc.to_gpu(payload_np, device), mode
                )
                metrics = cc.action_metrics(action, source_action, donor_action)
                if mode == "replace":
                    realized = float(np.linalg.norm(payload_np.astype(np.float64) - source_tokens.astype(np.float64)))
                else:
                    realized = float(np.linalg.norm(payload_np.astype(np.float64)))
                records.append(
                    {
                        "episode": int(frame["episode"]),
                        "step": int(frame["step"]),
                        "count": count,
                        "patch": patch,
                        "edit_l2": realized,
                        **metrics,
                    }
                )

            run_patch("full_tokens", None, "replace", donor_tokens.astype(np.float32))
            run_patch("token_structure", None, "add", cc.structure_delta(donor_tokens, source_tokens))

            for count in counts:
                selected = occlusion_ids[:count]
                feature_payload = decoded_patch(model, code_delta, selected, std5, None)
                moved = cc.remap_features(
                    cc.keep_features(code_delta, selected), selected, random_plan[count]
                )
                random_payload = cc.match_l2(cc.decode_delta(model, moved, std5), feature_payload)
                color_take = color_ids[: min(count, int(color_ids.size))]
                color_payload = decoded_patch(model, code_delta, color_take, std5, feature_payload)
                built: dict[PatchName, np.ndarray] = {
                    "occlusion_features": feature_payload,
                    "random_remap": random_payload,
                    "color_features": color_payload,
                }
                for patch in PATCHES:
                    match patch:
                        case "occlusion_features" | "random_remap" | "color_features":
                            run_patch(patch, count, "add", built[patch])
                        case _ as unexpected:
                            assert_never(unexpected)
            print(f"Occlusion frame {frame_index + 1}/{len(probe_frames)} done", flush=True)

        rows: list[dict] = []
        for patch in REFERENCES:
            rows.append(summarize(records, None, patch))
        for count in counts:
            for patch in PATCHES:
                rows.append(summarize(records, count, patch))
        print_circuit(rows)
        verdict = occlusion_verdict(rows, counts, int(occlusion_ids.size))
        print("\n" + verdict, flush=True)

        summary = {
            "task": sentence,
            "task_id": task_id,
            "source_run": str(source),
            "probe_frames": [
                {"episode": int(frame["episode"]), "step": int(frame["step"]), "label": frame["label"]}
                for frame in probe_frames
            ],
            "min_specificity": min_specificity,
            "requested_counts": requested,
            "counts": counts,
            "n_features": n_features,
            "occlusion_features": passed_rows,
            "color_features": feature_rows(color_ids, scores, "color", QUIET_FOR["color"]),
            "rejected_preview": nearest_rejected(scores, QUIET_FOR["occlusion"]) if occlusion_ids.size == 0 else [],
            "rows": rows,
            "verdict": verdict,
        }
        (dest / "summary.json").write_text(json.dumps(cc.json_ready(summary), indent=2))
        print("\nWrote", dest / "summary.json", flush=True)
        print("RESULT_DIR", dest, flush=True)
    finally:
        env.close()


if __name__ == "__main__":
    main()
