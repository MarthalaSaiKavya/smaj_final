"""Pack the π0.5 permanence study into one research zip.

Keeps the Colab notebook, experiment source, tests, figures, and summary JSON.
Skips caches, public model weights, LIBERO archives, secrets, and raw dumps.
Does not rewrite outputs/permanence/.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import zipfile
from pathlib import Path
from typing import Literal, assert_never

Decision = Literal["keep", "skip"]

SOURCE_NAMES = (
    "collect_layer5_replay.py",
    "controlled_contrasts.py",
    "feature_count_sweep.py",
    "token_position_patch.py",
    "token_subspace.py",
    "occlusion_features.py",
    "paper_gaps.py",
    "cover_autopsy.py",
    "paper_claim.py",
    "remaining_gaps.py",
    "tight_cover.py",
    "conclusion.py",
    "occlusion_contrast.py",
    "pack_research.py",
)

NOTEBOOK_NAMES = (
    "final_smaj.ipynb",
    "controlled_contrasts.ipynb",
    "feature_count_sweep.ipynb",
    "token_position_patch.ipynb",
    "token_subspace.ipynb",
)

RESULT_SUFFIXES = {".json", ".png", ".txt", ".csv", ".md"}
WEIGHT_KEEP = "transcoder.pt"
DEFAULT_ZIP_NAME = "pi05_permanence_research.zip"
MAX_KEEP_BYTES = 250 * 1024 * 1024


def skip_roots() -> list[tuple[str, str]]:
    return [
        ("lerobot-venv", "runtime virtualenv"),
        (".venv", "runtime virtualenv"),
        ("hf_home", "HuggingFace cache; Pi0.5 and PaliGemma stay on the Hub"),
        ("huggingface", "HuggingFace cache"),
        ("datasets--", "Hub snapshot; not a paper figure"),
        (".git", "version control internals"),
        ("__pycache__", "bytecode"),
        ("secrets", "tokens and keys"),
        ("libero_hdf5", "public LIBERO demonstration archive"),
        ("libero-assets", "public LIBERO assets"),
        ("uploads", "scratch upload folder, not the paper drop"),
    ]


def skip_suffixes() -> tuple[str, ...]:
    return (
        ".tar",
        ".tar.gz",
        ".tgz",
        ".hdf5",
        ".h5",
        ".safetensors",
        ".bin",
        ".ckpt",
        ".npz",
        ".pyc",
        ".whl",
    )


def is_smaj_notebook(path: Path) -> bool:
    name = path.name.lower()
    return path.suffix == ".ipynb" and "smaj" in name


def under_permanence_results(path: Path) -> bool:
    parts = set(path.parts)
    return "outputs" in parts and "permanence" in parts


def under_occlusion_results(path: Path) -> bool:
    parts = path.parts
    return "outputs" in parts and "occlusion_contrast" in parts


def skip_root_reason(path: Path) -> str | None:
    parts = [part.lower() for part in path.parts]
    for name, reason in skip_roots():
        key = name.lower()
        if key.endswith("--"):
            if any(part.startswith(key) for part in parts):
                return reason
        elif key in parts:
            return reason
    lowered = path.name.lower()
    if lowered in {"hf_token.txt", ".env"} or lowered.startswith("hf_token"):
        return "secret"
    if lowered.endswith(skip_suffixes()):
        if lowered.endswith(".npz"):
            return "raw replay or feature dump; the JSON summaries are enough"
        return "cache, archive, or public checkpoint"
    if lowered.endswith(".pt") and lowered != WEIGHT_KEEP:
        return "policy or action-expert weight; Pi0.5 is on HuggingFace, the Drive checkpoint stays on Drive"
    return None


def keep_reason(path: Path) -> str | None:
    name = path.name
    if is_smaj_notebook(path):
        return "Colab run notebook"
    if name in NOTEBOOK_NAMES:
        return "experiment notebook"
    if name == "README.md":
        return "how to run the study"
    if name in SOURCE_NAMES:
        return "experiment source"
    if path.parent.name == "tests" and name.startswith("test_") and name.endswith(".py"):
        return "unit test"
    if name == WEIGHT_KEEP and under_permanence_results(path):
        return "trained layer-5 dictionary used by the paper figures"
    if under_permanence_results(path) and path.suffix.lower() in RESULT_SUFFIXES:
        return "permanence result"
    if under_occlusion_results(path) and path.suffix.lower() in RESULT_SUFFIXES:
        return "action-expert verdict or figure"
    return None


def classify(path: Path) -> tuple[Decision, str]:
    if not path.is_file():
        return "skip", "not a file"
    skipped = skip_root_reason(path)
    if skipped is not None:
        if path.suffix.lower() in RESULT_SUFFIXES and (
            under_permanence_results(path) or under_occlusion_results(path) or is_smaj_notebook(path)
        ):
            pass
        else:
            return "skip", skipped
    kept = keep_reason(path)
    if kept is None:
        return "skip", "not a paper artifact"
    size = path.stat().st_size
    if size > MAX_KEEP_BYTES:
        return "skip", f"larger than {MAX_KEEP_BYTES} bytes"
    return "keep", kept


def search_roots(cwd: Path | None = None) -> list[Path]:
    here = cwd if cwd is not None else Path.cwd()
    drive = os.environ.get("DRIVE_ROOT", "/content/drive/MyDrive/groot-run-shared-programmer908")
    roots = [
        here,
        Path("/content"),
        Path("/content/groot-run"),
        Path("/content/pi05-run"),
        Path(drive),
    ]
    found: list[Path] = []
    seen: set[Path] = set()
    for root in roots:
        try:
            resolved = root.resolve()
        except OSError:
            continue
        if not resolved.exists() or resolved in seen:
            continue
        seen.add(resolved)
        found.append(resolved)
    return found


def iter_candidates(roots: list[Path]) -> list[Path]:
    files: list[Path] = []
    seen: set[Path] = set()

    def add(path: Path) -> None:
        try:
            resolved = path.resolve()
        except OSError:
            return
        if resolved in seen or not resolved.is_file():
            return
        if "uploads" in resolved.parts:
            return
        seen.add(resolved)
        files.append(resolved)

    for root in roots:
        for name in SOURCE_NAMES:
            add(root / name)
            add(root / "scripts" / name)
        add(root / "README.md")
        tests = root / "tests"
        if tests.is_dir():
            for path in sorted(tests.glob("test_*.py")):
                add(path)
        for name in NOTEBOOK_NAMES:
            add(root / name)
        try:
            for path in list(root.glob("*smaj*.ipynb")) + list(root.glob("*/*smaj*.ipynb")):
                add(path)
        except OSError:
            pass
        permanence = root / "outputs" / "permanence"
        if permanence.is_dir():
            for path in permanence.rglob("*"):
                if path.is_file():
                    add(path)
        occlusion = root / "outputs" / "occlusion_contrast"
        if occlusion.is_dir():
            for path in occlusion.rglob("*"):
                if path.is_file():
                    add(path)
        scripts = root / "scripts"
        if scripts.is_dir():
            for name in SOURCE_NAMES:
                add(scripts / name)
    return files


def zip_name_for(path: Path) -> str:
    name = path.name
    if is_smaj_notebook(path) or name in NOTEBOOK_NAMES:
        return f"notebooks/{name}"
    if name in SOURCE_NAMES:
        return f"source/{name}"
    if path.parent.name == "tests" and name.startswith("test_"):
        return f"tests/{name}"
    if name == "README.md":
        return "README.md"
    parts = list(path.parts)
    if "permanence" in parts:
        index = parts.index("permanence")
        rel = "/".join(parts[index:])
        return f"results/{rel}"
    if "occlusion_contrast" in parts:
        index = parts.index("occlusion_contrast")
        rel = "/".join(parts[index:])
        return f"results/{rel}"
    digest = hashlib.sha1(str(path).encode("utf-8")).hexdigest()[:8]
    return f"other/{digest}_{name}"


def inventory(paths: list[Path]) -> list[dict]:
    rows: list[dict] = []
    used: dict[str, Path] = {}
    for path in paths:
        decision, reason = classify(path)
        row = {
            "path": str(path),
            "name": path.name,
            "bytes": path.stat().st_size if path.is_file() else 0,
            "decision": decision,
            "reason": reason,
            "zip_name": None,
        }
        if decision == "keep":
            entry = zip_name_for(path)
            taken = used.get(entry)
            if taken is not None and taken != path:
                if taken.stat().st_mtime >= path.stat().st_mtime:
                    row["decision"] = "skip"
                    row["reason"] = f"duplicate of {taken}"
                else:
                    for old in rows:
                        if old.get("zip_name") == entry and old["decision"] == "keep":
                            old["decision"] = "skip"
                            old["reason"] = f"duplicate of {path}"
                            break
                    row["zip_name"] = entry
                    used[entry] = path
            else:
                row["zip_name"] = entry
                used[entry] = path
        elif decision == "skip":
            pass
        else:
            assert_never(decision)
        rows.append(row)
    return rows


def manifest_text(rows: list[dict], zip_path: Path) -> str:
    kept = [row for row in rows if row["decision"] == "keep"]
    skipped = [row for row in rows if row["decision"] == "skip"]
    lines = [
        "π0.5 permanence research zip",
        f"archive: {zip_path.name}",
        "",
        "KEEP these files. They are the paper: notebook, source, tests, figures, summary JSON.",
        "",
    ]
    for row in kept:
        lines.append(f"KEEP  {row['zip_name']}  ({row['bytes']} bytes)  {row['reason']}")
    lines.extend(
        [
            "",
            "SKIP these. They are runtime, public data, or secrets. Do not put them in a paper drop.",
            "",
        ]
    )
    for name, reason in skip_roots():
        lines.append(f"SKIP  {name}/  {reason}")
    lines.append("SKIP  *.pt except transcoder.pt  policy weights stay on HuggingFace or Drive")
    lines.append("SKIP  *.npz  raw replay dumps")
    lines.append("SKIP  *.hdf5 *.tar  LIBERO archives")
    extra = [
        row
        for row in skipped
        if row["reason"] not in {reason for _name, reason in skip_roots()}
        and "duplicate" not in row["reason"]
        and row["reason"] != "not a paper artifact"
    ]
    if extra:
        lines.append("")
        lines.append("Skipped files that were considered and rejected:")
        for row in extra[:80]:
            lines.append(f"SKIP  {row['path']}  {row['reason']}")
    lines.extend(
        [
            "",
            "How to read this zip",
            "- notebooks/ holds the Colab run, including names like 3final_smaj (1).ipynb.",
            "- source/ is the experiment code. tests/ checks the scoring rules without LIBERO.",
            "- results/permanence/*/summary.json and *.png are the numbers and figures.",
            "- results/occlusion_contrast/verdict.json is the separate action-expert check.",
            "- Pi0.5 weights: HuggingFace lerobot/pi05_libero_finetuned.",
            "- Demonstrations: HuggingFaceVLA/libero and the public LIBERO suite.",
            "",
        ]
    )
    return "\n".join(lines) + "\n"


def write_zip(rows: list[dict], zip_path: Path) -> Path:
    zip_path.parent.mkdir(parents=True, exist_ok=True)
    if zip_path.exists():
        zip_path.unlink()
    kept = [row for row in rows if row["decision"] == "keep"]
    with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_DEFLATED) as bundle:
        bundle.writestr("MANIFEST.txt", manifest_text(rows, zip_path))
        bundle.writestr(
            "inventory.json",
            json.dumps({"files": rows, "kept": len(kept)}, indent=2) + "\n",
        )
        for row in kept:
            bundle.write(row["path"], arcname=row["zip_name"])
    return zip_path


def print_inventory(rows: list[dict], zip_path: Path | None) -> None:
    kept = [row for row in rows if row["decision"] == "keep"]
    print("KEEP")
    for row in kept:
        print(f"  {row['zip_name']}  {row['reason']}")
    print("SKIP roots")
    for name, reason in skip_roots():
        print(f"  {name}/  {reason}")
    print(f"kept {len(kept)} files")
    if zip_path is not None:
        print("ZIP", zip_path, "bytes", zip_path.stat().st_size)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Pack the permanence study into one research zip.")
    parser.add_argument("--cwd", type=Path, default=None)
    parser.add_argument("--zip", type=Path, default=None)
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    cwd = args.cwd.resolve() if args.cwd is not None else Path.cwd()
    roots = search_roots(cwd)
    rows = inventory(iter_candidates(roots))
    if args.dry_run:
        print_inventory(rows, None)
        return 0
    zip_path = args.zip
    if zip_path is None:
        colab = Path("/content")
        zip_path = (colab if colab.is_dir() else cwd) / DEFAULT_ZIP_NAME
    zip_path = zip_path.resolve()
    write_zip(rows, zip_path)
    print_inventory(rows, zip_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
