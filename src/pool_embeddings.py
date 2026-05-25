from __future__ import annotations

import argparse
import logging
from pathlib import Path

import numpy as np
from tqdm import tqdm

from src.utils.io import ensure_dir, save_ids, set_seed, setup_logging

LOGGER = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Pool multi-view features into specimen embeddings")
    parser.add_argument("--features", type=Path, required=True, help="Directory with per-specimen feature npy files (e.g. [V,D] or [V,T,D])")
    parser.add_argument("--out", type=Path, required=True, help="Output directory for embeddings")
    parser.add_argument("--pool", type=str, choices=["mean", "max"], default="mean")
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def pool_features(arr: np.ndarray, method: str) -> np.ndarray:
    if arr.ndim < 2:
        raise ValueError(f"Expected [...,D], got shape={arr.shape}")

    pool_axes = tuple(range(arr.ndim - 1))
    if method == "mean":
        return arr.mean(axis=pool_axes)
    if method == "max":
        return arr.max(axis=pool_axes)
    raise ValueError(f"Unsupported pooling method: {method}")


def main() -> None:
    args = parse_args()
    setup_logging()
    set_seed(args.seed)
    ensure_dir(args.out)

    feature_files = sorted(
        fp
        for fp in args.features.rglob("*.npy")
        if fp.name not in {"embeddings.npy"}
    )
    if not feature_files:
        LOGGER.warning("No feature files in %s", args.features)
        return

    all_embs = []
    all_ids = []
    for fp in tqdm(feature_files, desc="Pooling"):
        sid = str(fp.relative_to(args.features).with_suffix(""))
        try:
            feat = np.load(fp)
            emb = pool_features(feat, args.pool).astype(np.float32)
        except Exception as e:
            LOGGER.exception("Failed to pool %s: %s", fp, e)
            continue

        out_path = args.out / f"{sid}.npy"
        out_path.parent.mkdir(parents=True, exist_ok=True)
        np.save(out_path, emb)
        all_embs.append(emb)
        all_ids.append(sid)

    if not all_embs:
        LOGGER.warning("No embeddings generated")
        return

    embs = np.stack(all_embs, axis=0)
    np.save(args.out / "embeddings.npy", embs)
    save_ids(all_ids, args.out / "ids.txt")
    LOGGER.info("Saved %d embeddings to %s", len(all_ids), args.out)


if __name__ == "__main__":
    main()
