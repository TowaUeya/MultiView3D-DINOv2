from __future__ import annotations

import logging
import random
from pathlib import Path
from typing import Iterable, List, Sequence

import numpy as np
import yaml

try:
    import torch
except Exception:  # pragma: no cover
    torch = None

LOGGER = logging.getLogger(__name__)

MESH_EXTENSIONS = {".ply", ".obj", ".stl", ".off"}


def setup_logging(level: int = logging.INFO) -> None:
    logging.basicConfig(
        level=level,
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    )


def ensure_dir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


def list_mesh_files(input_dir: Path) -> List[Path]:
    if input_dir.is_file() and input_dir.suffix.lower() in MESH_EXTENSIONS:
        return [input_dir]
    return sorted(
        p
        for p in input_dir.rglob("*")
        if p.is_file() and p.suffix.lower() in MESH_EXTENSIONS
    )


def list_image_files(input_dir: Path) -> List[Path]:
    return sorted(
        p
        for p in input_dir.rglob("*")
        if p.is_file() and p.suffix.lower() in {".png", ".jpg", ".jpeg"}
    )


def stem(path: Path) -> str:
    return path.stem


def specimen_id_from_render(render_path: Path, root_dir: Path | None = None) -> str:
    # expected: {specimen_id}_viewXX.png
    if root_dir is not None:
        path_for_id = render_path.relative_to(root_dir)
    else:
        path_for_id = render_path

    name = path_for_id.stem
    if "_view" in name:
        base = name.rsplit("_view", 1)[0]
    else:
        base = name

    rel_parent = path_for_id.parent
    if str(rel_parent) == ".":
        return base
    return str(rel_parent / base)


def group_renders_by_specimen(render_files: Iterable[Path], root_dir: Path | None = None) -> dict[str, list[Path]]:
    grouped: dict[str, list[Path]] = {}
    for fp in render_files:
        sid = specimen_id_from_render(fp, root_dir=root_dir)
        grouped.setdefault(sid, []).append(fp)
    for sid in grouped:
        grouped[sid] = sorted(grouped[sid])
    return grouped


def save_ids(ids: list[str], path: Path) -> None:
    with path.open("w", encoding="utf-8") as f:
        for sid in ids:
            f.write(f"{sid}\n")


def load_ids(path: Path) -> list[str]:
    return [line.strip() for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def set_seed(seed: int | None) -> None:
    if seed is None:
        return
    random.seed(seed)
    np.random.seed(seed)
    if torch is not None:
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)


def load_yaml(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def resolve_file_or_recursive_search(
    path: Path,
    *,
    patterns: Sequence[str],
    label: str,
    fallback_patterns: Sequence[str] | None = None,
) -> Path:
    """Resolve file path. If a directory is given, recursively search candidates."""
    if path.is_file():
        return path
    if not path.exists():
        raise FileNotFoundError(f"{label} path does not exist: {path}")
    if not path.is_dir():
        raise FileNotFoundError(f"{label} must be a file or directory: {path}")

    candidates: list[Path] = []
    for pattern in patterns:
        candidates.extend(fp for fp in path.rglob(pattern) if fp.is_file())
    if not candidates and fallback_patterns:
        for pattern in fallback_patterns:
            candidates.extend(fp for fp in path.rglob(pattern) if fp.is_file())
        if candidates:
            LOGGER.warning(
                "Canonical %s was not found under %s (patterns=%s). Fallback matched by %s.",
                label,
                path,
                ",".join(patterns),
                ",".join(fallback_patterns),
            )

    if not candidates:
        pats = ", ".join(patterns)
        if fallback_patterns:
            pats = f"{pats} | fallback: {', '.join(fallback_patterns)}"
        raise FileNotFoundError(f"{label} not found under directory: {path} (patterns: {pats})")

    unique_candidates = sorted(set(candidates), key=lambda fp: (len(fp.relative_to(path).parts), fp.as_posix()))
    chosen = unique_candidates[0]
    if len(unique_candidates) > 1:
        LOGGER.warning(
            "Multiple %s files found under %s; using %s",
            label,
            path,
            chosen,
        )
    return chosen
