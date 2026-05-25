from __future__ import annotations

import logging
import random
from pathlib import Path
from typing import Iterable, List

import numpy as np
import torch

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


def specimen_id_from_render(render_path: Path, root_dir: Path | None = None) -> str:
    if root_dir is not None:
        path_for_id = render_path.relative_to(root_dir)
    else:
        path_for_id = render_path

    name = path_for_id.stem
    base = name.rsplit("_view", 1)[0] if "_view" in name else name

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


def set_seed(seed: int | None) -> None:
    if seed is None:
        return
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
