#!/usr/bin/env python
"""Feature-count mediation sweep on the saved controlled-contrast frames.

Loads the transcoder written by controlled_contrasts.py, rebuilds only the
probe frames named in that run's summary, and patches the top 8, 32, 128, and
all dictionary features. Each count has a same-size random remap and an
off-contrast patch. Full-token, pooled, and token-structure patches are the
reference lines and do not depend on the count.

Does not rewrite outputs/permanence/controlled_contrasts/.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Literal

import numpy as np
import torch

import controlled_contrasts as cc

EDITED_OF = {"color": "recolor", "absence": "absent", "occlusion": "occluded"}
SPARSE_PATCHES = ("transcoder_features", "random_remap", "off_contrast")
REFERENCE_PATCHES = ("full_tokens", "pooled", "token_structure")
ControlKind = Literal["disjoint", "permute_all"]


def parse_counts(text: str, n_features: int) -> list[int]:
    counts: list[int] = []
    seen: set[int] = set()
    for part in text.split(","):
        piece = part.strip()
        if not piece:
            continue
        value = int(piece)
        if value < 1:
            raise ValueError(f"Feature count must be positive, got {value}")
        clipped = min(value, n_features)
        if clipped not in seen:
            seen.add(clipped)
            counts.append(clipped)
    if not counts:
        raise ValueError("CTRL_FEATURE_COUNTS is empty")
    return counts


def top_features(score: np.ndarray, count: int) -> np.ndarray:
    """Largest-magnitude features. A smaller count is a prefix of a larger one."""
    order = np.argsort(-np.abs(np.asarray(score, dtype=np.float64)), kind="mergesort")
    take = min(int(count), int(order.size))
    return order[:take].astype(np.int64)


def control_destination(
    selected: np.ndarray, n_features: int, rng: np.random.Generator
) -> tuple[np.ndarray, ControlKind]:
    """Same-size random features. A full dictionary is a channel permutation."""
    chosen = np.asarray(selected, dtype=np.int64)
    if chosen.size >= n_features:
        return rng.permutation(n_features).astype(np.int64), "permute_all"
    pool = np.setdiff1d(np.arange(n_features), chosen)
    picked = rng.choice(pool, size=int(chosen.size), replace=False)
    return picked.astype(np.int64), "disjoint"


def sweep_verdict(
    contrast: str,
    curve: list[tuple[int, float | None, float | None]],
    structure: float | None,
) -> str:
    """Read one contrast. `curve` is (count, feature gap, random gap), ascending."""
    usable = [(count, feat, rand) for count, feat, rand in curve if feat is not None and rand is not None]
    if len(usable) < 2:
        return f"{contrast}: not enough feature counts to read the curve."
    small_count, small_gap, _small_rand = usable[0]
    large_count, large_gap, large_rand = usable[-1]
    rose = large_gap - small_gap
    above_random = large_gap - large_rand
    reaches_structure = structure is not None and structure > 0 and large_gap >= 0.5 * structure
    if reaches_structure and above_random >= 0.10:
        keep = usable[0][0]
        for count, feat, rand in usable:
            if feat - rand >= 0.10:
                keep = count
                break
        return (
            f"{contrast}: closure climbs toward token structure and stays above the random patch. "
            f"Smallest count that beats random by 10 points: {keep}."
        )
    if rose < 0.05 and above_random < 0.05:
        return (
            f"{contrast}: closure stays near the {small_count}-feature result "
            f"({100 * small_gap:.1f}% to {100 * large_gap:.1f}% at {large_count}). "
            "Next run is a token-position patch."
        )
    structure_text = "the token-structure line"
    if structure is not None:
        structure_text = f"token structure at {100 * structure:.1f}%"
    return (
        f"{contrast}: closure moved from {100 * small_gap:.1f}% at {small_count} features "
        f"to {100 * large_gap:.1f}% at {large_count} (random {100 * large_rand:.1f}%). "
        f"The reference is {structure_text}."
    )


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


def summarize(records: list[dict], contrast: str, count: int | None, patch: str) -> dict:
    chosen = [
        row
        for row in records
        if row["contrast"] == contrast and row["patch"] == patch and row["count"] == count
    ]
    return {
        "contrast": contrast,
        "count": count,
        "patch": patch,
        "mean_gap_closed": cc.mean_defined([row["fraction_gap_closed"] for row in chosen]),
        "mean_edit_l2": float(np.mean([row["edit_l2"] for row in chosen])) if chosen else None,
        "mean_baseline_rmse": float(np.mean([row["source_to_donor_rmse"] for row in chosen])) if chosen else None,
        "n_defined": int(sum(row["fraction_gap_closed"] is not None for row in chosen)),
        "n_pairs": len(chosen),
        "random_kind": chosen[0]["random_kind"] if chosen and patch == "random_remap" else None,
    }


def print_sweep(rows: list[dict]) -> None:
    print("\nSWEEP  fraction of action-chunk RMSE closed (base -> edited)", flush=True)
    print(
        f"{'contrast':<12}{'count':>8}{'patch':<22}{'gap_closed':>12}{'edit_l2':>12}{'base_rmse':>12}{'n':>6}",
        flush=True,
    )
    for row in rows:
        gap = row["mean_gap_closed"]
        gap_text = "undefined" if gap is None else f"{100 * gap:.2f}%"
        count = row["count"]
        count_text = "ref" if count is None else str(count)
        edit = row["mean_edit_l2"]
        base = row["mean_baseline_rmse"]
        edit_text = "nan" if edit is None else f"{edit:.3f}"
        base_text = "nan" if base is None else f"{base:.4f}"
        print(
            f"{row['contrast']:<12}{count_text:>8}{row['patch']:<22}{gap_text:>12}"
            f"{edit_text:>12}{base_text:>12}{row['n_defined']:>6}",
            flush=True,
        )


def curve_for(rows: list[dict], contrast: str, counts: list[int]) -> list[tuple[int, float | None, float | None]]:
    curve = []
    for count in counts:
        feature = next(
            row["mean_gap_closed"]
            for row in rows
            if row["contrast"] == contrast and row["count"] == count and row["patch"] == "transcoder_features"
        )
        random_gap = next(
            row["mean_gap_closed"]
            for row in rows
            if row["contrast"] == contrast and row["count"] == count and row["patch"] == "random_remap"
        )
        curve.append((count, feature, random_gap))
    return curve


def reference_gap(rows: list[dict], contrast: str, patch: str) -> float | None:
    return next(
        row["mean_gap_closed"]
        for row in rows
        if row["contrast"] == contrast and row["count"] is None and row["patch"] == patch
    )


def main() -> None:
    cc.configure_environment()
    if not torch.cuda.is_available():
        raise SystemExit("CUDA GPU required. Use the Colab L4 or A100 runtime.")
    collect = cc.load_collect_module()
    seed = cc.env_int("CTRL_SEED", 0)
    repo = Path(os.environ.get("LOCAL_REPO", "/content/groot-run"))
    source = repo / "outputs" / "permanence" / "controlled_contrasts"
    dest = repo / "outputs" / "permanence" / "feature_count_sweep"
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
    counts = parse_counts(os.environ.get("CTRL_FEATURE_COUNTS", "8,32,128,512"), n_features)
    print(
        f"Loaded transcoder dim={mean5.shape[0]} features={n_features} counts={counts}",
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
        if episode in demos:
            continue
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

        def capture_conditions(frame, conditions: tuple[str, ...]) -> dict:
            noise = 5000 + int(frame["episode"]) * 100 + int(frame["step"])
            packed = {}
            for condition in conditions:
                tokens5, tokens6, action, _device = cc.capture(
                    policy, layer5, layer6, collect, pre, frame, condition, noise
                )
                packed[condition] = {"l5": tokens5, "l6": tokens6, "action": action}
            return packed

        probe_packed = [capture_conditions(frame, cc.CONDITION_NAMES) for frame in probe_frames]
        check = probe_frames[0]
        noise = 5000 + int(check["episode"]) * 100 + int(check["step"])
        again5, _again6, again_action, device = cc.capture(
            policy, layer5, layer6, collect, pre, check, "base", noise
        )
        max_token = float(np.max(np.abs(again5 - probe_packed[0]["base"]["l5"])))
        if max_token > 1e-3 or not np.array_equal(again_action, probe_packed[0]["base"]["action"]):
            raise RuntimeError(
                f"Repeating the same inputs changed the forward (token max abs {max_token})."
            )
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

        probe_token_maps = [{name: item[name]["l5"] for name in cc.CONDITION_NAMES} for item in probe_packed]
        scores = cc.score_probe(model, mean5, std5, probe_token_maps)
        rankings = {contrast: top_features(scores[contrast], n_features) for contrast in cc.CONTRAST_NAMES}
        rng = np.random.default_rng(seed)
        random_plan: dict[tuple[str, int], tuple[np.ndarray, ControlKind]] = {}
        for contrast in cc.CONTRAST_NAMES:
            for count in counts:
                selected = rankings[contrast][:count]
                random_plan[(contrast, count)] = control_destination(selected, n_features, rng)

        records: list[dict] = []
        for frame_index, (frame, packed) in enumerate(zip(probe_frames, probe_packed)):
            noise = 5000 + int(frame["episode"]) * 100 + int(frame["step"])
            source_tokens = packed["base"]["l5"]
            source_action = packed["base"]["action"]
            agent, wrist = frame["built"]["images"]["base"]
            batch = cc.batch_from_images(collect, pre, agent, wrist, frame["state"], frame["sentence"], policy)
            prepared = []
            for contrast in cc.CONTRAST_NAMES:
                condition = EDITED_OF[contrast]
                donor_tokens = packed[condition]["l5"]
                donor_action = packed[condition]["action"]
                code_delta = cc.encode_tokens(model, donor_tokens, mean5, std5) - cc.encode_tokens(
                    model, source_tokens, mean5, std5
                )
                prepared.append((contrast, donor_tokens, donor_action, code_delta))

            def run_patch(contrast, patch, count, mode, payload_np, random_kind, target_action):
                _tokens, _tokens6, action = cc.forward_policy(
                    policy,
                    layer5,
                    None,
                    batch,
                    noise,
                    cc.to_gpu(payload_np, device),
                    mode,
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
                        "count": count,
                        "patch": patch,
                        "random_kind": random_kind,
                        "edit_l2": realized,
                        **metrics,
                    }
                )

            for contrast, donor_tokens, donor_action, _code_delta in prepared:
                pooled = (donor_tokens - source_tokens).mean(axis=0, keepdims=True)
                pooled = np.repeat(pooled, source_tokens.shape[0], axis=0)
                references = {
                    "full_tokens": ("replace", donor_tokens),
                    "pooled": ("add", pooled),
                    "token_structure": ("add", cc.structure_delta(donor_tokens, source_tokens)),
                }
                for patch in REFERENCE_PATCHES:
                    mode, payload_np = references[patch]
                    run_patch(contrast, patch, None, mode, payload_np, None, donor_action)

            for count in counts:
                for contrast, _donor_tokens, donor_action, code_delta in prepared:
                    selected_ids = rankings[contrast][:count]
                    other = cc.off_contrast(contrast)  # type: ignore[arg-type]
                    other_ids = rankings[other][:count]
                    destination, kind = random_plan[(contrast, count)]
                    selected = cc.decode_delta(model, cc.keep_features(code_delta, selected_ids), std5)
                    random_delta = cc.decode_delta(
                        model,
                        cc.remap_features(cc.keep_features(code_delta, selected_ids), selected_ids, destination),
                        std5,
                    )
                    off_delta = cc.decode_delta(model, cc.keep_features(code_delta, other_ids), std5)
                    random_delta = cc.match_l2(random_delta, selected)
                    off_delta = cc.match_l2(off_delta, selected)
                    payloads = {
                        "transcoder_features": ("add", selected, None),
                        "random_remap": ("add", random_delta, kind),
                        "off_contrast": ("add", off_delta, None),
                    }
                    for patch in SPARSE_PATCHES:
                        mode, payload_np, random_kind = payloads[patch]
                        run_patch(contrast, patch, count, mode, payload_np, random_kind, donor_action)
            print(f"Sweep frame {frame_index + 1}/{len(probe_frames)} done", flush=True)

        rows: list[dict] = []
        for contrast in cc.CONTRAST_NAMES:
            for patch in REFERENCE_PATCHES:
                rows.append(summarize(records, contrast, None, patch))
            for count in counts:
                for patch in SPARSE_PATCHES:
                    rows.append(summarize(records, contrast, count, patch))
        print_sweep(rows)

        verdicts = []
        for contrast in ("color", "absence"):
            text = sweep_verdict(
                contrast,
                curve_for(rows, contrast, counts),
                reference_gap(rows, contrast, "token_structure"),
            )
            verdicts.append(text)
            print("\n" + text, flush=True)
        occlusion_text = sweep_verdict(
            "occlusion",
            curve_for(rows, "occlusion", counts),
            reference_gap(rows, "occlusion", "token_structure"),
        )
        print("\n" + occlusion_text, flush=True)

        summary = {
            "task": sentence,
            "task_id": task_id,
            "source_run": str(source),
            "probe_frames": [
                {"episode": int(frame["episode"]), "step": int(frame["step"]), "label": frame["label"]}
                for frame in probe_frames
            ],
            "counts": counts,
            "n_features": n_features,
            "prior_residual_floor_rmse": (prior.get("full_cover") or {}).get("action_rmse"),
            "sweep": rows,
            "verdicts": verdicts + [occlusion_text],
            "rankings_prefix": {contrast: rankings[contrast][: counts[0]].tolist() for contrast in cc.CONTRAST_NAMES},
        }
        (dest / "summary.json").write_text(json.dumps(cc.json_ready(summary), indent=2))
        print("\nWrote", dest / "summary.json", flush=True)
        print("RESULT_DIR", dest, flush=True)
    finally:
        env.close()


if __name__ == "__main__":
    main()
