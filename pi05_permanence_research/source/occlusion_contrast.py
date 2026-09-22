"""Occlusion contrast for one frozen action-expert transcoder feature.

The transcoders, the feature report, and the circuit tracer already exist.
This script only replays LIBERO demonstrations, scores one grasp feature on
held_visible / held_hidden / gone, and, when that feature passes, asks the
existing tracer for a compact circuit and compares one ablation against a
random feature of similar firing rate.

Pi0.5 is never allowed to move the arm. The simulator is set to saved
demonstration states, and the policy is only a probe.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import sys
from pathlib import Path
from typing import Any, Literal

import h5py
import numpy as np
import torch

FAIL_SENTENCE = (
    "no action-expert feature keeps the object when it is hidden and drops it when it is gone."
)
FIRING_FLOOR = 1e-4
GONE_FRACTION = 0.25
GRIPPER_R2 = 0.5
RESIDUAL_FRACTION = 0.25
ARM_ATOL = 1e-6
DEPTH_MARGIN = 0.02
GRIP_RADIUS = 0.03
Label = Literal["held_visible", "held_hidden", "not_held"]


def user_feature_id(layer: int, timestep: float, feature: int) -> str:
    return f"L{int(layer)}/tau{float(timestep):.4g}.F{int(feature)}"


def tracer_feature_key(layer: int, timestep: float, feature: int) -> str:
    return f"L{int(layer):02d}:tau{float(timestep):.4g}:F{int(feature)}"


def task_text(value: Any) -> str:
    if isinstance(value, (list, tuple)):
        return " ".join(task_text(item) for item in value)
    if value is None:
        return ""
    return str(value)


def same_language(left: str, right: str) -> bool:
    def norm(text: str) -> str:
        return " ".join(re.findall(r"[a-z0-9]+", text.lower()))

    a, b = norm(left), norm(right)
    if not a or not b:
        return False
    return a == b or a in b or b in a


def demo_index(rows: list[dict], episode_index: int) -> int | None:
    """Rank of one dataset episode among episodes that share its task text."""
    episode_index = int(episode_index)
    task = None
    for row in rows:
        if int(row["episode_index"]) == episode_index:
            task = task_text(row.get("task"))
            break
    if task is None:
        return None
    same = sorted(int(row["episode_index"]) for row in rows if task_text(row.get("task")) == task)
    return same.index(episode_index)


def episode_rows_from_meta(episodes: Any) -> list[dict]:
    """Read episode index and task text from a LeRobot metadata object."""
    if isinstance(episodes, dict):
        tasks = episodes.get("tasks", episodes.get("task"))
        length = len(episodes.get("dataset_from_index", tasks or []))
        rows = []
        for index in range(length):
            task = "" if tasks is None else tasks[index]
            rows.append({"episode_index": index, "task": task_text(task)})
        return rows
    if not hasattr(episodes, "__getitem__") or not hasattr(episodes, "__len__"):
        return []
    try:
        length = len(episodes)
    except TypeError:
        return []
    rows = []
    for index in range(length):
        item = episodes[index]
        if not isinstance(item, dict):
            continue
        rows.append(
            {
                "episode_index": int(item.get("episode_index", index)),
                "task": task_text(item.get("tasks", item.get("task", ""))),
            }
        )
    return rows


def label_from_measurement(
    contact_known: bool,
    touching: bool,
    angle_known: bool,
    hidden: bool,
) -> Label | None:
    """Drop a frame when contact or the held-frame camera angle cannot be measured."""
    if not contact_known:
        return None
    if not touching:
        return "not_held"
    if not angle_known:
        return None
    if hidden:
        return "held_hidden"
    return "held_visible"


def arm_joints(qpos: np.ndarray, span: tuple[int, int] | None) -> np.ndarray:
    values = np.asarray(qpos, dtype=np.float64).reshape(-1)
    if span is None:
        return values.copy()
    start, stop = span
    return np.concatenate([values[:start], values[stop:]])


def arm_unchanged(before: np.ndarray, after: np.ndarray, atol: float = ARM_ATOL) -> bool:
    before = np.asarray(before, dtype=np.float64)
    after = np.asarray(after, dtype=np.float64)
    if before.shape != after.shape:
        return False
    return float(np.max(np.abs(before - after))) <= atol


def choose_grasp(rows: list[dict]) -> dict | None:
    """Pick the report feature whose top frames are most often a grasp.

    A grasp frame is one where a finger is touching the object. Ties go to the
    feature the report already ranked higher.
    """
    usable = [row for row in rows if int(row["top_count"]) > 0 and int(row["held_top"]) > 0]
    if not usable:
        return None
    usable.sort(key=lambda row: (-int(row["held_top"]) / int(row["top_count"]), int(row["rank"])))
    return usable[0]


def _fit_line(x: np.ndarray, y: np.ndarray) -> tuple[float, float, float]:
    x = np.asarray(x, dtype=np.float64).reshape(-1)
    y = np.asarray(y, dtype=np.float64).reshape(-1)
    if x.size < 2 or np.allclose(x, x[0]):
        mean = float(y.mean()) if y.size else 0.0
        return mean, 0.0, 0.0
    design = np.column_stack([np.ones(x.size), x])
    coef, _, _, _ = np.linalg.lstsq(design, y, rcond=None)
    predicted = design @ coef
    total = float(np.sum((y - y.mean()) ** 2))
    residual = float(np.sum((y - predicted) ** 2))
    r2 = 0.0 if total <= 1e-12 else 1.0 - residual / total
    return float(coef[0]), float(coef[1]), float(r2)


def gripper_correlation_failed(scores: np.ndarray, grippers: np.ndarray, held: np.ndarray) -> bool:
    """True when gripper opening alone accounts for the held-versus-open firing."""
    scores = np.asarray(scores, dtype=np.float64).reshape(-1)
    grippers = np.asarray(grippers, dtype=np.float64).reshape(-1)
    held = np.asarray(held, dtype=bool).reshape(-1)
    if scores.size < 4 or int(held.sum()) == 0 or int((~held).sum()) == 0:
        return False
    intercept, slope, r2 = _fit_line(grippers, scores)
    residual = scores - (intercept + slope * grippers)
    gap = abs(float(scores[held].mean() - scores[~held].mean()))
    residual_gap = abs(float(residual[held].mean() - residual[~held].mean()))
    if gap < 1e-8:
        return r2 >= GRIPPER_R2
    return r2 >= GRIPPER_R2 and residual_gap <= RESIDUAL_FRACTION * gap


def occlusion_passes(
    held_visible: float | None,
    held_hidden: float | None,
    gone: float | None,
    gripper_failed: bool,
) -> bool:
    if gripper_failed:
        return False
    rates = (held_visible, held_hidden, gone)
    if any(rate is None or not np.isfinite(rate) for rate in rates):
        return False
    assert held_visible is not None and held_hidden is not None and gone is not None
    if held_visible <= FIRING_FLOOR or held_hidden <= FIRING_FLOOR:
        return False
    return gone <= GONE_FRACTION * min(held_visible, held_hidden)


def similar_feature(frequencies: np.ndarray, target: int, seed: int) -> int:
    """Pick one other feature whose firing rate is close to the target."""
    freq = np.asarray(frequencies, dtype=np.float64).reshape(-1)
    if freq.size < 2:
        raise ValueError("Need at least two features to choose a random control.")
    distance = np.abs(freq - freq[int(target)])
    distance[int(target)] = np.inf
    order = np.argsort(distance, kind="mergesort")
    closest = float(distance[int(order[0])])
    pool = [int(index) for index in order if abs(float(distance[int(index)]) - closest) <= 1e-12]
    if not pool:
        raise ValueError("No control feature is available.")
    return int(np.random.default_rng(seed).choice(np.asarray(pool, dtype=np.int64)))


def action_rmse(source: np.ndarray, edited: np.ndarray) -> float:
    source = np.asarray(source, dtype=np.float64)
    edited = np.asarray(edited, dtype=np.float64)
    return float(np.sqrt(np.mean((source - edited) ** 2)))


def real_feature_counts(real_delta: float, random_delta: float) -> bool:
    return float(real_delta) > float(random_delta)


def circuit_path(graph: dict) -> str | None:
    """Strongest incoming edge, walked back from the traced target."""
    nodes = {str(node["node_key"]): node for node in graph.get("nodes", [])}
    if not nodes:
        return None
    target = str((graph.get("config") or {}).get("target") or "")
    if target not in nodes:
        target = max(nodes, key=lambda key: int(nodes[key].get("depth") or 0))
    incoming: dict[str, list[dict]] = {}
    for edge in graph.get("edges", []):
        incoming.setdefault(str(edge["target_key"]), []).append(edge)
    chain = [target]
    seen = {target}
    while chain[-1] in incoming and len(chain) < 8:
        options = incoming[chain[-1]]
        best = max(
            options,
            key=lambda edge: float(
                edge.get("edge_score")
                or edge.get("mean_abs_contribution")
                or edge.get("edge_influence")
                or 0.0
            ),
        )
        source = str(best["source_key"])
        if source in seen:
            break
        chain.append(source)
        seen.add(source)
    chain.reverse()
    return " -> ".join(_node_user_id(key, nodes) for key in chain)


def _node_user_id(key: str, nodes: dict[str, dict]) -> str:
    node = nodes.get(key)
    if node is not None and "layer" in node and "feature" in node:
        return user_feature_id(int(node["layer"]), float(node["timestep"]), int(node["feature"]))
    parts = key.split(":")
    if len(parts) == 3 and parts[0].startswith("L") and parts[1].startswith("tau") and parts[2].startswith("F"):
        return user_feature_id(int(parts[0][1:]), float(parts[1][3:]), int(parts[2][1:]))
    return key


def mean_or_none(values: list[float]) -> float | None:
    if not values:
        return None
    return float(np.mean(np.asarray(values, dtype=np.float64)))


def verdict_document(
    *,
    passed: bool,
    feature_id: str | None,
    firing_rates: dict[str, float | None],
    gripper_correlation_failed_flag: bool | None,
    circuit_path_text: str | None,
    real_delta: float | None,
    random_delta: float | None,
    frame_counts: dict[str, int] | None = None,
) -> dict:
    real_counts = None
    delta = None
    if real_delta is not None and random_delta is not None:
        real_counts = real_feature_counts(real_delta, random_delta)
        delta = {"real": real_delta, "random": random_delta, "real_counts": real_counts}
    if not passed:
        text = FAIL_SENTENCE
        circuit_path_text = None
        delta = None
    elif real_counts:
        text = (
            f"{feature_id} fires while the object is hidden and drops when it is gone, "
            "and ablating it moves the action more than a random feature."
        )
    else:
        text = (
            f"{feature_id} fires while the object is hidden and drops when it is gone. "
            "Ablating it does not move the action more than a random feature."
        )
    document = {
        "verdict": text,
        "feature_id": feature_id,
        "firing_rates": {
            "held_visible": firing_rates.get("held_visible"),
            "held_hidden": firing_rates.get("held_hidden"),
            "gone": firing_rates.get("gone"),
        },
        "gripper_correlation_failed": gripper_correlation_failed_flag,
        "circuit_path": circuit_path_text,
        "real_versus_random_action_delta": delta,
    }
    if frame_counts is not None:
        document["frame_counts"] = frame_counts
    return document


def spread_frames(frames: list[dict], limit: int) -> list[dict]:
    """Take frames round-robin across episodes, up to limit."""
    if limit <= 0:
        return []
    buckets: dict[int, list[dict]] = {}
    for frame in frames:
        buckets.setdefault(int(frame["episode"]), []).append(frame)
    ordered = [buckets[key] for key in sorted(buckets)]
    chosen: list[dict] = []
    while ordered and len(chosen) < limit:
        nxt: list[list[dict]] = []
        for bucket in ordered:
            if bucket and len(chosen) < limit:
                chosen.append(bucket.pop(0))
            if bucket:
                nxt.append(bucket)
        ordered = nxt
    return chosen


def top_examples_for_candidate(topk: dict, observations: dict[int, dict], candidate: dict, limit: int) -> list[dict]:
    name = candidate["layer_name"]
    timestep = candidate["timestep_key"]
    feature = int(candidate["feature"])
    store = topk["topk"][name][timestep]
    scores = store["scores"][feature]
    obs_ids = store["observation_ids"][feature]
    rows = []
    width = min(limit, int(scores.shape[0]))
    for rank in range(width):
        score = float(scores[rank])
        if not np.isfinite(score):
            continue
        observation_id = int(obs_ids[rank])
        if observation_id < 0:
            continue
        observation = dict(observations.get(observation_id, {"observation_id": observation_id}))
        observation["activation"] = score
        rows.append(observation)
    return rows


def main() -> None:
    # LIBERO, MuJoCo, and LeRobot are imported here on purpose. The unit tests
    # import this module without a simulator or a policy install.
    # LeRobot and the action-expert package are imported here so unit tests can
    # load this module without that install. LIBERO is imported the same way.
    from lerobot.datasets.factory import make_dataset
    from lerobot.policies import make_policy

    from collect_pi05_transcoder_features import _freeze_policy, _load_transcoders
    from pi05_mi.patch_pi05 import Pi05TranscoderContext, install_pi05_action_expert_wrappers
    from train_pi05_transcoders import (
        _configure_train_config,
        _make_preprocessor,
        patch_pi05_checkpoint_key_compat,
        patch_transformers_causal_mask_compat,
        resolve_device,
        resolve_policy_dtype,
    )

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--feature-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--suite", default="libero_spatial")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--policy-path", default="lerobot/pi05_libero_finetuned")
    args = parser.parse_args()

    if not torch.cuda.is_available():
        raise SystemExit("This contrast needs the Colab GPU.")

    feature_dir = args.feature_dir
    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    candidates = _candidate_rows(json.loads((feature_dir / "feature_candidates.json").read_text()))
    observations = _read_jsonl(feature_dir / "observations.jsonl")
    topk = torch.load(feature_dir / "feature_topk.pt", map_location="cpu", weights_only=False)
    stats = torch.load(feature_dir / "feature_stats.pt", map_location="cpu", weights_only=False)

    episode_rows = _episode_rows(args, resolve_device, resolve_policy_dtype, _configure_train_config, make_dataset)
    labeled_candidates = []
    with _suite_env(args.suite) as environment:
        for candidate in candidates:
            examples = top_examples_for_candidate(topk, observations, candidate, limit=5)
            held_top = 0
            measured = 0
            task_hits: dict[str, int] = {}
            demo_ids: list[int] = []
            for example in examples:
                located = _locate_example(example, episode_rows, environment["suite"])
                if located is None:
                    continue
                task_index, demo_index_value, frame_index, sentence = located
                label = _label_demo_frame(environment, task_index, demo_index_value, frame_index)
                if label is None:
                    continue
                measured += 1
                if label.startswith("held_"):
                    held_top += 1
                    task_hits[sentence] = task_hits.get(sentence, 0) + 1
                    demo_ids.append(demo_index_value)
            labeled_candidates.append(
                {
                    **candidate,
                    "held_top": held_top,
                    "top_count": measured,
                    "task_hits": task_hits,
                    "demo_ids": demo_ids,
                }
            )
            print(
                f"report L{candidate['layer']}/tau{float(candidate['timestep']):.4g}.F{candidate['feature']} "
                f"rank {candidate['rank']}: grasp frames {held_top}/{measured}",
                flush=True,
            )
        chosen = choose_grasp(labeled_candidates)
        if chosen is None:
            document = verdict_document(
                passed=False,
                feature_id=None,
                firing_rates={"held_visible": None, "held_hidden": None, "gone": None},
                gripper_correlation_failed_flag=None,
                circuit_path_text=None,
                real_delta=None,
                random_delta=None,
                frame_counts={"held_visible": 0, "held_hidden": 0, "not_held": 0, "gone": 0, "dropped": 0},
            )
            _write_verdict(output_dir, document)
            print(document["verdict"], flush=True)
            return

        feature_id = user_feature_id(int(chosen["layer"]), float(chosen["timestep"]), int(chosen["feature"]))
        print(f"chosen grasp feature {feature_id}", flush=True)
        sentence, task_index = _chosen_task(chosen, environment["suite"])
        demos = _chosen_demos(chosen)
        print(f"replaying {args.suite} task {task_index}: {sentence}", flush=True)
        print(f"demos {demos}", flush=True)
        catalog = _catalog_frames(environment, task_index, demos, sentence)
        print(
            "labels "
            f"held_visible={len(catalog['held_visible'])} "
            f"held_hidden={len(catalog['held_hidden'])} "
            f"not_held={len(catalog['not_held'])} "
            f"dropped={catalog['dropped']}",
            flush=True,
        )
        selected = {
            "held_visible": spread_frames(catalog["held_visible"], 8),
            "held_hidden": spread_frames(catalog["held_hidden"], 8),
            "not_held": spread_frames(catalog["not_held"], 8),
        }
        policy, preprocessor, context, ablate = _load_probe(
            args,
            make_policy,
            make_dataset,
            _freeze_policy,
            _load_transcoders,
            Pi05TranscoderContext,
            install_pi05_action_expert_wrappers,
            _configure_train_config,
            _make_preprocessor,
            patch_transformers_causal_mask_compat,
            patch_pi05_checkpoint_key_compat,
            resolve_device,
            resolve_policy_dtype,
            str(chosen["layer_name"]),
        )
        scored = _score_selected(
            environment,
            policy,
            preprocessor,
            context,
            ablate,
            selected,
            sentence,
            task_index=task_index,
            layer=int(chosen["layer"]),
            timestep=float(chosen["timestep"]),
            feature=int(chosen["feature"]),
            seed=args.seed,
        )
        rates = {
            "held_visible": mean_or_none(scored["held_visible"]),
            "held_hidden": mean_or_none(scored["held_hidden"]),
            "gone": mean_or_none(scored["gone"]),
        }
        gripper_failed = gripper_correlation_failed(scored["grip_scores"], scored["grip_values"], scored["grip_held"])
        passed = occlusion_passes(rates["held_visible"], rates["held_hidden"], rates["gone"], gripper_failed)
        counts = {
            "held_visible": len(scored["held_visible"]),
            "held_hidden": len(scored["held_hidden"]),
            "not_held": len(scored["not_held"]),
            "gone": len(scored["gone"]),
            "dropped": int(catalog["dropped"]),
        }
        print(
            f"firing held_visible={rates['held_visible']} held_hidden={rates['held_hidden']} "
            f"gone={rates['gone']} gripper_correlation_failed={gripper_failed}",
            flush=True,
        )
        if not passed:
            document = verdict_document(
                passed=False,
                feature_id=feature_id,
                firing_rates=rates,
                gripper_correlation_failed_flag=gripper_failed,
                circuit_path_text=None,
                real_delta=None,
                random_delta=None,
                frame_counts=counts,
            )
            _write_verdict(output_dir, document)
            print(document["verdict"], flush=True)
            return

        _release_policy(policy)
        path_text = _trace_circuit(args, chosen, feature_dir, output_dir)
        policy, preprocessor, context, ablate = _load_probe(
            args,
            make_policy,
            make_dataset,
            _freeze_policy,
            _load_transcoders,
            Pi05TranscoderContext,
            install_pi05_action_expert_wrappers,
            _configure_train_config,
            _make_preprocessor,
            patch_transformers_causal_mask_compat,
            patch_pi05_checkpoint_key_compat,
            resolve_device,
            resolve_policy_dtype,
            str(chosen["layer_name"]),
        )
        frequency = stats["stats"][chosen["layer_name"]][chosen["timestep_key"]]["firing_frequency"]
        control = similar_feature(frequency.detach().float().cpu().numpy(), int(chosen["feature"]), args.seed)
        print(f"random control feature {control}", flush=True)
        real_delta, random_delta = _ablate(
            environment,
            policy,
            preprocessor,
            context,
            ablate,
            selected["held_hidden"],
            sentence,
            task_index=task_index,
            feature=int(chosen["feature"]),
            control=control,
            seed=args.seed,
        )
        document = verdict_document(
            passed=True,
            feature_id=feature_id,
            firing_rates=rates,
            gripper_correlation_failed_flag=gripper_failed,
            circuit_path_text=path_text,
            real_delta=real_delta,
            random_delta=random_delta,
            frame_counts=counts,
        )
        _write_verdict(output_dir, document)
        print(json.dumps(document, indent=2, sort_keys=True), flush=True)


def _candidate_rows(payload: Any) -> list[dict]:
    if isinstance(payload, dict):
        return list(payload["candidates"])
    return list(payload)


def _read_jsonl(path: Path) -> dict[int, dict]:
    rows: dict[int, dict] = {}
    with path.open() as handle:
        for line in handle:
            row = json.loads(line)
            rows[int(row["observation_id"])] = row
    return rows


def _write_verdict(output_dir: Path, document: dict) -> None:
    path = output_dir / "verdict.json"
    path.write_text(json.dumps(document, indent=2, sort_keys=True) + "\n")
    print(f"verdict {path}", flush=True)


def _episode_rows(args, resolve_device, resolve_policy_dtype, configure, make_dataset) -> list[dict]:
    try:
        device = resolve_device("cpu")
        namespace = _namespace(args, device, resolve_policy_dtype("auto", device))
        cfg = configure(namespace, episodes=None)
        dataset = make_dataset(cfg)
        rows = episode_rows_from_meta(dataset.meta.episodes)
        print(f"dataset episodes with task text: {len(rows)}", flush=True)
        return rows
    except Exception as exc:
        print(f"dataset episode map unavailable ({exc}); demo index will follow episode_index", flush=True)
        return []


def _namespace(args, device, dtype: str):
    return argparse.Namespace(
        policy_path=args.policy_path,
        batch_size=1,
        num_workers=0,
        local_files_only=os.environ.get("HF_HUB_OFFLINE") == "1",
        resolved_device=device,
        resolved_policy_dtype=dtype,
    )


def _suite_env(suite_name: str):
    return _Suite(suite_name)


class _Suite:
    """One LIBERO suite kept open while frames are labeled and probed."""

    def __init__(self, suite_name: str):
        self.suite_name = suite_name
        self.suite = None
        self.env = None
        self.task_index = None

    def __enter__(self):
        # LIBERO is imported on use so tests do not require the simulator.
        from libero.libero import benchmark, get_libero_path
        from libero.libero.envs import OffScreenRenderEnv

        self._benchmark = benchmark
        self._get_libero_path = get_libero_path
        self._env_cls = OffScreenRenderEnv
        suite_cls = benchmark.get_benchmark_dict()[self.suite_name]
        self.suite = suite_cls()
        return {"suite": self, "name": self.suite_name}

    def __exit__(self, *_exc):
        self.close()

    def close(self) -> None:
        if self.env is not None:
            self.env.close()
            self.env = None

    def task_count(self) -> int:
        count = self.suite.n_tasks
        return int(count() if callable(count) else count)

    def task(self, index: int):
        return self.suite.get_task(int(index))

    def ensure(self, task_index: int):
        if self.env is not None and self.task_index == int(task_index):
            return self.env
        self.close()
        task = self.task(task_index)
        bddl = os.path.join(self._get_libero_path("bddl_files"), task.problem_folder, task.bddl_file)
        self.env = self._env_cls(bddl_file_name=bddl, camera_heights=256, camera_widths=256)
        self.task_index = int(task_index)
        return self.env


def _locate_example(example: dict, episode_rows: list[dict], suite: _Suite):
    if "episode_index" not in example or "frame_index" not in example:
        return None
    episode = int(example["episode_index"])
    frame = int(example["frame_index"])
    language = task_text(example.get("task"))
    if episode_rows:
        mapped = demo_index(episode_rows, episode)
        demo = episode if mapped is None else mapped
    else:
        demo = episode
    match = _match_task(suite, language)
    if match is None:
        return None
    task_index, sentence = match
    return task_index, demo, frame, sentence


def _match_task(suite: _Suite, language: str):
    for index in range(suite.task_count()):
        task = suite.task(index)
        if same_language(task.language, language):
            return index, task.language
    return None


def _label_demo_frame(environment: dict, task_index: int, demo_index_value: int, frame_index: int) -> str | None:
    suite: _Suite = environment["suite"]
    try:
        _actions, states = _load_demo(suite, task_index, demo_index_value)
    except (IndexError, FileNotFoundError, KeyError, OSError):
        return None
    if frame_index < 0 or frame_index >= len(states):
        return None
    env = suite.ensure(task_index)
    raw = _observe_state(env, states[frame_index])
    return _label_raw(env.sim, suite.task(task_index).language, raw)


def _chosen_task(chosen: dict, suite: _Suite) -> tuple[str, int]:
    hits: dict[str, int] = chosen.get("task_hits") or {}
    if hits:
        sentence = max(hits, key=hits.get)
        match = _match_task(suite, sentence)
        if match is not None:
            return match[1], match[0]
    task = suite.task(0)
    return task.language, 0


def _chosen_demos(chosen: dict) -> list[int]:
    demos = []
    for demo in chosen.get("demo_ids") or []:
        if int(demo) not in demos:
            demos.append(int(demo))
    if not demos:
        demos = [0, 1, 2]
    return demos[:3]


def _catalog_frames(environment: dict, task_index: int, demos: list[int], sentence: str) -> dict:
    suite: _Suite = environment["suite"]
    stride = int(os.environ.get("OCCLUSION_STRIDE", "8"))
    max_steps = int(os.environ.get("OCCLUSION_MAX_STEPS", "80"))
    catalog: dict[str, Any] = {"held_visible": [], "held_hidden": [], "not_held": [], "dropped": 0}
    for demo in demos:
        try:
            _actions, states = _load_demo(suite, task_index, demo)
        except (IndexError, FileNotFoundError, KeyError, OSError) as exc:
            print(f"skip demo {demo}: {exc}", flush=True)
            continue
        env = suite.ensure(task_index)
        for frame_index in range(0, min(len(states), max_steps), stride):
            raw = _observe_state(env, states[frame_index])
            label = _label_raw(env.sim, sentence, raw)
            if label is None:
                catalog["dropped"] += 1
                continue
            catalog[label].append(
                {
                    "episode": int(demo),
                    "step": int(frame_index),
                    "label": label,
                    "state": np.asarray(states[frame_index], dtype=np.float64),
                }
            )
    return catalog


def _load_demo(suite: _Suite, task_index: int, demo_index_value: int):
    task = suite.task(task_index)
    path = _find_demo_file(task)
    with h5py.File(path, "r") as handle:
        names = sorted(handle["data"].keys(), key=lambda name: int(re.search(r"(\d+)$", name).group(1)))
        if demo_index_value >= len(names):
            raise IndexError(f"{path} has {len(names)} demos, asked for {demo_index_value}")
        group = handle["data"][names[demo_index_value]]
        actions = np.asarray(group["actions"])
        states = np.asarray(group["states"])
    if len(states) == len(actions) + 1:
        states = states[:-1]
    return actions, states


def _find_demo_file(task) -> Path:
    name = task.name
    folder = task.problem_folder
    roots = []
    if os.environ.get("LIBERO_DATASET_DIR"):
        roots.append(Path(os.environ["LIBERO_DATASET_DIR"]))
    # LIBERO dataset root is resolved only when a demonstration is opened.
    from libero.libero import get_libero_path

    roots.append(Path(get_libero_path("datasets")))
    exact_names = [f"{name}_demo.hdf5", f"{name}.hdf5"]
    tried = []
    for root in roots:
        for exact in exact_names:
            candidate = root / folder / exact
            tried.append(candidate)
            if candidate.is_file():
                return candidate
        if root.is_dir():
            for hit in sorted(root.rglob("*.hdf5")):
                if name in hit.stem:
                    return hit
    raise FileNotFoundError("No LIBERO demonstration file found:\n" + "\n".join(str(path) for path in tried[:8]))


def _observe_state(env, state: np.ndarray) -> dict:
    state = np.asarray(state, dtype=np.float64)
    if hasattr(env, "set_init_state"):
        return env.set_init_state(state)
    env.reset()
    env.sim.set_state_from_flattened(state)
    env.sim.forward()
    inner = env.env if hasattr(env, "env") else env
    return inner._get_observations()


def _label_raw(sim, sentence: str, raw: dict) -> str | None:
    del raw
    bodies = _candidate_bodies(sim, sentence)
    fingers = _finger_geoms(sim)
    touching, body = _contact_body(sim, fingers, bodies)
    if touching is None:
        return label_from_measurement(False, False, False, False)
    if not touching:
        return label_from_measurement(True, False, False, False)
    geoms = bodies.get(body) if body is not None else None
    measured = _occlusion(sim, geoms or [], _camera_id(sim), _site_id(sim))
    if measured is None:
        return label_from_measurement(True, True, False, False)
    hidden, _depth, _ratio = measured
    return label_from_measurement(True, True, True, hidden)


def _candidate_bodies(sim, sentence: str) -> dict[int, list[int]]:
    # MuJoCo is imported on use so tests do not require the simulator.
    import mujoco

    model = sim.model
    free = int(mujoco.mjtJoint.mjJNT_FREE)
    movable = set()
    for body in range(int(model.nbody)):
        start = int(model.body_jntadr[body])
        count = int(model.body_jntnum[body])
        for joint in range(start, start + count):
            if joint >= 0 and int(model.jnt_type[joint]) == free:
                movable.add(body)
                break
    by_body: dict[int, list[int]] = {}
    for geom in range(int(model.ngeom)):
        body = int(model.geom_bodyid[geom])
        if body in movable:
            by_body.setdefault(body, []).append(geom)
    words = _target_words(sentence)
    named = {}
    for body, geoms in by_body.items():
        blob = " ".join((model.geom_id2name(geom) or "") for geom in geoms).lower()
        blob += " " + (model.body_id2name(body) or "").lower()
        if any(word in blob for word in words):
            named[body] = geoms
    return named or by_body


def _target_words(sentence: str) -> list[str]:
    low = sentence.lower()
    verbs = ("pick up", "pick", "grasp", "take", "put", "place", "move", "open", "close", "turn")
    rels = (" between ", " next to ", " on top of ", " on the ", " in the ", " and ", " near ", " into ", " onto ")
    segment = low
    for verb in verbs:
        at = segment.find(verb)
        if at >= 0:
            segment = segment[at + len(verb) :]
            break
    cut = len(segment)
    for rel in rels:
        at = segment.find(rel)
        if 0 <= at < cut:
            cut = at
    stop = {"the", "a", "an", "and", "that", "with", "your", "from", "into", "onto"}
    words = [word for word in re.findall(r"[a-z]+", segment[:cut]) if len(word) > 3 and word not in stop]
    return words or [word for word in re.findall(r"[a-z]+", low) if len(word) > 3 and word not in stop]


def _finger_geoms(sim) -> list[int]:
    model = sim.model
    return [index for index in range(int(model.ngeom)) if "finger" in (model.geom_id2name(index) or "").lower()]


def _contact_body(sim, fingers: list[int], bodies: dict[int, list[int]]):
    if not fingers or not bodies:
        return None, None
    geom_to_body = {geom: body for body, geoms in bodies.items() for geom in geoms}
    finger_set = set(fingers)
    data = sim.data
    for index in range(int(data.ncon)):
        contact = data.contact[index]
        left, right = int(contact.geom1), int(contact.geom2)
        if left in finger_set and right in geom_to_body:
            return True, geom_to_body[right]
        if right in finger_set and left in geom_to_body:
            return True, geom_to_body[left]
    return False, None


def _site_id(sim) -> int | None:
    for name in ("gripper0_grip_site", "gripper0_eef", "robot0_eef", "grip_site"):
        try:
            return int(sim.model.site_name2id(name))
        except Exception:
            continue
    return None


def _camera_id(sim) -> int:
    model = sim.model
    for index in range(int(model.ncam)):
        if (model.camera_id2name(index) or "") == "agentview":
            return index
    return 0


def _occlusion(sim, geoms: list[int], cam: int, site: int | None):
    if not geoms or site is None:
        return None
    data = sim.data
    camera = np.asarray(data.cam_xpos[cam], dtype=np.float64)
    obj = np.mean([np.asarray(data.geom_xpos[geom], dtype=np.float64) for geom in geoms], axis=0)
    grip = np.asarray(data.site_xpos[site], dtype=np.float64)
    to_obj = obj - camera
    to_grip = grip - camera
    obj_dist = float(np.linalg.norm(to_obj))
    grip_dist = float(np.linalg.norm(to_grip))
    if obj_dist < 1e-6 or grip_dist < 1e-6:
        return None
    cosine = float(np.clip(np.dot(to_obj, to_grip) / (obj_dist * grip_dist), -1.0, 1.0))
    angle = float(np.arccos(cosine))
    grip_angle = float(np.arctan2(GRIP_RADIUS, grip_dist))
    depth_gap = obj_dist - grip_dist
    hidden = depth_gap > DEPTH_MARGIN and angle < grip_angle
    return hidden, depth_gap, (angle / grip_angle if grip_angle > 1e-9 else float("inf"))


def _object_span(sim, body: int) -> tuple[int, int] | None:
    # MuJoCo is imported on use so tests do not require the simulator.
    import mujoco

    model = sim.model
    free = int(mujoco.mjtJoint.mjJNT_FREE)
    start = int(model.body_jntadr[body])
    count = int(model.body_jntnum[body])
    for joint in range(start, start + count):
        if joint >= 0 and int(model.jnt_type[joint]) == free:
            address = int(model.jnt_qposadr[joint])
            return address, address + 7
    return None


def _move_object_away(sim, body: int):
    span = _object_span(sim, body)
    if span is None:
        return None, None
    saved = np.array(sim.data.qpos, copy=True)
    start, stop = span
    sim.data.qpos[start : start + 3] = np.array([5.0, 5.0, -1.0])
    sim.data.qpos[start + 3 : stop] = np.array([1.0, 0.0, 0.0, 0.0])
    sim.data.qvel[:] = 0
    sim.forward()
    return saved, span


def _restore_qpos(sim, saved: np.ndarray) -> None:
    sim.data.qpos[:] = saved
    sim.data.qvel[:] = 0
    sim.forward()


def _render_after_edit(env) -> dict:
    inner = env.env if hasattr(env, "env") else env
    try:
        return inner._get_observations(force_update=True)
    except TypeError:
        if hasattr(inner, "_update_observables"):
            inner._update_observables(force=True)
        return inner._get_observations()


def _load_probe(
    args,
    make_policy,
    make_dataset,
    freeze_policy,
    load_transcoders,
    context_cls,
    install,
    configure,
    make_preprocessor,
    patch_mask,
    patch_keys,
    resolve_device,
    resolve_policy_dtype,
    layer_name: str,
):
    patch_mask()
    patch_keys()
    device = resolve_device("cuda")
    dtype = resolve_policy_dtype("auto", device)
    namespace = _namespace(args, device, dtype)
    cfg = configure(namespace, episodes="0")
    dataset = make_dataset(cfg)
    preprocessor = make_preprocessor(cfg, dataset, args.policy_path)
    cfg.policy.device = str(device)
    cfg.policy.dtype = dtype
    cfg.policy.pretrained_path = Path(args.policy_path)
    cfg.policy.compile_model = False
    cfg.policy.gradient_checkpointing = False
    print("loading frozen Pi0.5 policy", flush=True)
    policy = make_policy(cfg.policy, ds_meta=dataset.meta, rename_map=cfg.rename_map)
    freeze_policy(policy)
    transcoders = load_transcoders(args.checkpoint, device=device)
    context = context_cls(
        mode="probe",
        capture_records=False,
        capture_latents=True,
        latent_top_k=0,
        save_full_latents=False,
        store_latent_summaries=False,
    )
    _context, _names = install(policy, context=context, transcoders=transcoders, mode="probe")
    ablate = {"index": None}
    module = dict(policy.named_modules())[layer_name]
    original = module.forward

    def forward(x):
        if module.context.mode != "replace" or ablate["index"] is None:
            return original(x)
        timestep = module.context.timestep_for(x)
        _y_hat, latent, preactivation = module.transcoder(x, timestep, return_preactivation=True)
        module.context.record_trace(module.name, module.layer_index, preactivation, latent, timestep)
        module.context.record_latent(module.name, module.layer_index, latent, timestep)
        latent = latent.clone()
        latent[..., int(ablate["index"])] = 0
        y_hat = module.transcoder.decoder(latent.to(dtype=module.transcoder.decoder.weight.dtype))
        return y_hat.to(dtype=x.dtype)

    module.forward = forward
    return policy, preprocessor, context, ablate


def _release_policy(policy) -> None:
    del policy
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def _score_selected(
    environment, policy, preprocessor, context, ablate, selected, sentence, task_index, layer, timestep, feature, seed
):
    suite: _Suite = environment["suite"]
    ablate["index"] = None
    context.mode = "probe"
    scored: dict[str, Any] = {
        "held_visible": [],
        "held_hidden": [],
        "not_held": [],
        "gone": [],
        "grip_scores": [],
        "grip_values": [],
        "grip_held": [],
    }
    for label, frames in selected.items():
        for frame in frames:
            env = suite.ensure(task_index)
            raw = _observe_state(env, frame["state"])
            again = _label_raw(env.sim, sentence, raw)
            if again != label:
                continue
            value = _probe_feature(
                policy,
                preprocessor,
                context,
                raw,
                sentence,
                layer,
                timestep,
                feature,
                seed + int(frame["episode"]) * 1000 + int(frame["step"]),
            )
            if value is None:
                continue
            scored[label].append(value)
            scored["grip_scores"].append(value)
            scored["grip_values"].append(_gripper_value(raw))
            scored["grip_held"].append(label.startswith("held_"))
            frame["raw_seed"] = seed + int(frame["episode"]) * 1000 + int(frame["step"])
    gone_limit = int(os.environ.get("OCCLUSION_GONE", "4"))
    gone_frames = spread_frames(selected["held_hidden"], gone_limit)
    if len(gone_frames) < gone_limit:
        gone_frames = gone_frames + spread_frames(selected["held_visible"], gone_limit - len(gone_frames))
    for frame in gone_frames:
        env = suite.ensure(task_index)
        _observe_state(env, frame["state"])
        gone = _gone_probe(
            env,
            policy,
            preprocessor,
            context,
            sentence,
            layer,
            timestep,
            feature,
            int(frame.get("raw_seed", seed + int(frame["episode"]) * 1000 + int(frame["step"]))),
        )
        if gone is not None:
            scored["gone"].append(gone)
    return scored


def _gone_probe(env, policy, preprocessor, context, sentence, layer, timestep, feature, seed):
    sim = env.sim
    bodies = _candidate_bodies(sim, sentence)
    _touching, body = _contact_body(sim, _finger_geoms(sim), bodies)
    if body is None:
        return None
    before = arm_joints(np.array(sim.data.qpos, copy=True), _object_span(sim, body))
    saved, span = _move_object_away(sim, body)
    if saved is None:
        return None
    try:
        after = arm_joints(np.array(sim.data.qpos, copy=True), span)
        if not arm_unchanged(before, after):
            return None
        raw = _render_after_edit(env)
        after_render = arm_joints(np.array(sim.data.qpos, copy=True), span)
        if not arm_unchanged(before, after_render):
            return None
        return _probe_feature(policy, preprocessor, context, raw, sentence, layer, timestep, feature, seed)
    finally:
        _restore_qpos(sim, saved)


def _probe_feature(policy, preprocessor, context, raw, sentence, layer, timestep, feature, seed) -> float | None:
    box: dict[str, float] = {}

    def callback(name, layer_index, latent, step_time):
        del name
        if int(layer_index) != int(layer):
            return
        current = float(step_time.detach().float().reshape(-1)[0].cpu())
        if abs(current - float(timestep)) > 1e-3:
            return
        values = latent.detach().float()
        if values.ndim == 3:
            score = float(values[0, :, int(feature)].max().cpu())
        elif values.ndim == 2:
            score = float(values[0, int(feature)].cpu())
        else:
            return
        box["score"] = max(score, box.get("score", score))

    context.mode = "probe"
    context.latent_callback = callback
    context.clear_records()
    _seed_torch(seed)
    with torch.no_grad():
        _predict(policy, preprocessor, raw, sentence)
    context.latent_callback = None
    context.clear_records()
    if "score" not in box:
        return None
    return float(box["score"])


def _ablate(environment, policy, preprocessor, context, ablate, frames, sentence, task_index, feature, control, seed):
    suite: _Suite = environment["suite"]
    real_deltas = []
    random_deltas = []
    context.mode = "replace"
    for frame in frames:
        env = suite.ensure(task_index)
        raw = _observe_state(env, frame["state"])
        frame_seed = seed + int(frame["episode"]) * 1000 + int(frame["step"])
        base = _action_chunk(policy, preprocessor, context, ablate, raw, sentence, None, frame_seed)
        real = _action_chunk(policy, preprocessor, context, ablate, raw, sentence, feature, frame_seed)
        other = _action_chunk(policy, preprocessor, context, ablate, raw, sentence, control, frame_seed)
        real_deltas.append(action_rmse(base, real))
        random_deltas.append(action_rmse(base, other))
    if not real_deltas:
        raise RuntimeError("The passing feature has no held_hidden frame to ablate.")
    return float(np.mean(real_deltas)), float(np.mean(random_deltas))


def _action_chunk(policy, preprocessor, context, ablate, raw, sentence, feature_index, seed):
    ablate["index"] = feature_index
    context.mode = "replace"
    context.latent_callback = None
    context.clear_records()
    _seed_torch(seed)
    with torch.no_grad():
        chunk = _predict(policy, preprocessor, raw, sentence)
    ablate["index"] = None
    return _chunk_array(chunk)


def _predict(policy, preprocessor, raw, sentence):
    batch = {
        "observation.images.image": _image_tensor(raw, "agentview_image", "agentview_rgb"),
        "observation.images.image2": _image_tensor(
            raw, "robot0_eye_in_hand_image", "eye_in_hand_rgb", "robot0_eye_in_hand_rgb"
        ),
        "observation.state": torch.from_numpy(_libero_state(raw)).unsqueeze(0),
        "task": [sentence],
    }
    features = getattr(policy.config, "input_features", {}) or {}
    reference = batch["observation.images.image"]
    for key in features:
        if key.startswith("observation.images.") and key not in batch:
            batch[key] = torch.zeros_like(reference)
    prepared = preprocessor(batch)
    return policy.predict_action_chunk(prepared, num_steps=10)


def _image_tensor(raw: dict, *keys: str):
    image = None
    for key in keys:
        if key in raw:
            image = np.ascontiguousarray(raw[key])
            break
    if image is None:
        raise KeyError(f"None of {keys} are in the simulator observation. Keys: {sorted(raw)}")
    image = image[::-1, ::-1].copy()
    return torch.from_numpy(image).permute(2, 0, 1).float().div(255.0).unsqueeze(0)


def _libero_state(raw: dict) -> np.ndarray:
    state = np.concatenate(
        [
            np.asarray(raw["robot0_eef_pos"], dtype=np.float32).reshape(3),
            _quat_xyzw_to_axisangle(raw["robot0_eef_quat"]),
            np.asarray(raw["robot0_gripper_qpos"], dtype=np.float32).reshape(2),
        ]
    )
    if state.shape != (8,):
        raise RuntimeError(f"Expected an 8-number arm state, got {state.shape}")
    return state


def _quat_xyzw_to_axisangle(quat: np.ndarray) -> np.ndarray:
    quat = np.asarray(quat, dtype=np.float64)
    norm = np.linalg.norm(quat)
    if norm < 1e-8:
        return np.zeros(3, dtype=np.float32)
    quat = quat / norm
    w = float(np.clip(quat[3], -1.0, 1.0))
    den = float(np.sqrt(max(1.0 - w * w, 0.0)))
    if den < 1e-8:
        return np.zeros(3, dtype=np.float32)
    return (quat[:3] * 2.0 * np.arccos(w) / den).astype(np.float32)


def _gripper_value(raw: dict) -> float:
    return float(np.sum(np.asarray(raw["robot0_gripper_qpos"], dtype=np.float64).reshape(-1)))


def _chunk_array(chunk) -> np.ndarray:
    if isinstance(chunk, (tuple, list)):
        chunk = chunk[0]
    if isinstance(chunk, dict):
        chunk = chunk["action"]
    action = chunk.detach().float().cpu().numpy()
    if action.ndim == 3:
        action = action[0]
    if action.ndim != 2:
        raise RuntimeError(f"Expected an action chunk, got shape {action.shape}")
    return action


def _seed_torch(seed: int) -> None:
    torch.manual_seed(int(seed))
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(int(seed))


def _trace_circuit(args, chosen: dict, feature_dir: Path, output_dir: Path) -> str | None:
    target = tracer_feature_key(int(chosen["layer"]), float(chosen["timestep"]), int(chosen["feature"]))
    circuit_dir = output_dir / "circuit"
    script = Path(__file__).resolve().parent / "trace_pi05_transcoder_circuit.py"
    command = [
        sys.executable,
        "-u",
        str(script),
        "--checkpoint",
        str(args.checkpoint),
        "--feature-dir",
        str(feature_dir),
        "--output-dir",
        str(circuit_dir),
        "--target",
        target,
        "--trace-mode",
        "diffract-frontier",
        "--max-nodes",
        "5",
        "--min-attribution",
        "0.005",
        "--node-cumulative-threshold",
        "0.03",
        "--edge-cumulative-threshold",
        "0.7",
        "--source-policy",
        "all-earlier",
        "--num-inference-steps",
        "10",
        "--top-examples",
        "5",
        "--device",
        "cuda",
    ]
    if os.environ.get("HF_HUB_OFFLINE") == "1":
        command.append("--local-files-only")
    print("tracing " + " ".join(command), flush=True)
    env = os.environ.copy()
    root = Path(__file__).resolve().parents[1]
    env["PYTHONPATH"] = os.pathsep.join([str(root / "src"), str(root / "scripts"), env.get("PYTHONPATH", "")])
    env["PYTHONUNBUFFERED"] = "1"
    process = subprocess.Popen(
        command,
        cwd=str(root / "scripts"),
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
    )
    assert process.stdout is not None
    for line in process.stdout:
        print(line, end="")
    code = process.wait()
    if code != 0:
        raise RuntimeError(f"trace_pi05_transcoder_circuit.py exited with code {code}")
    graph = json.loads((circuit_dir / "graph.json").read_text())
    return circuit_path(graph)


if __name__ == "__main__":
    main()
