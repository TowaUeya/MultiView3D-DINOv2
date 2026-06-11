from __future__ import annotations

import argparse
import logging
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

from src.utils.io import ensure_dir, group_renders_by_specimen, list_image_files, set_seed, setup_logging
from src.utils.vision import build_transform, forward_embedding, load_dinov3_model, load_image_tensor, resolve_device

LOGGER = logging.getLogger(__name__)


class SpecimenViewsDataset(Dataset):

    def __init__(self, items: list[tuple[str, list[Path]]], transform) -> None:
        self._items = items
        self._transform = transform

    def __len__(self) -> int:
        return len(self._items)

    def __getitem__(self, index: int):
        sid, image_paths = self._items[index]
        views: list[torch.Tensor] = []
        for ip in image_paths:
            try:
                views.append(load_image_tensor(ip, self._transform))
            except Exception as exc:
                LOGGER.exception("Failed to read image %s: %s", ip, exc)
        if not views:
            return sid, None
        return sid, torch.stack(views)  # [V, C, H, W]


def _collate_single(batch):
    return batch[0]


def _is_cuda_oom(exc: BaseException) -> bool:
    """Return True if the exception looks like a CUDA out-of-memory error."""
    return isinstance(exc, RuntimeError) and "out of memory" in str(exc).lower()


def _embed_specimen_views(
    model: torch.nn.Module,
    views: torch.Tensor,
    device: torch.device,
    batch_size: int,
    keep_tokens: bool,
) -> np.ndarray:
    """Embed all views of one specimen and return a stacked feature array.

    Views are processed in chunks of at most ``batch_size``. On a CUDA
    out-of-memory error the failing chunk is retried with a halved chunk size
    (down to a single view) after clearing the allocator cache, so that
    low-memory GPUs can finish instead of aborting the whole run.
    """
    parts: list[np.ndarray] = []
    num_views = int(views.shape[0])
    idx = 0
    chunk = max(1, batch_size)
    while idx < num_views:
        bt = views[idx : idx + chunk].to(device, non_blocking=True)
        try:
            with torch.inference_mode():
                z = forward_embedding(model, bt)  # [b, T, D] (CLS+patch, register tokens already dropped)
                if not keep_tokens:
                    # Pre-mean on the same token axis (=1) that pool_embeddings reduces -> [b, D]
                    z = z.mean(dim=1)
                arr = z.detach().cpu().numpy()
            parts.append(arr)
            idx += chunk
        except RuntimeError as exc:
            # Only recover from CUDA OOM, and only while there is still room to shrink.
            if not _is_cuda_oom(exc) or chunk == 1:
                raise
            del bt
            if device.type == "cuda":
                torch.cuda.empty_cache()
            reduced = max(1, chunk // 2)
            LOGGER.warning(
                "CUDA OOM while embedding %d view(s); retrying with chunk size %d",
                chunk,
                reduced,
            )
            chunk = reduced

    if len(parts) == 1:
        feature_arr = parts[0]
    else:
        feature_arr = np.concatenate(parts, axis=0)  # [V, T, D] or [V, D]
    return feature_arr.astype(np.float32, copy=False)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Extract frozen DINOv3 features from rendered images")
    parser.add_argument("--renders", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--model", type=str, default="dinov3_vitb16")
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument("--image-size", type=int, default=768)
    parser.add_argument("--crop-size", type=int, default=768)
    parser.add_argument("--batch-size", type=int, default=16, help="Max images per forward pass")
    parser.add_argument(
        "--num-workers",
        type=int,
        default=4,
        help="DataLoader worker processes for image decode/transform (0 = main process)",
    )
    parser.add_argument(
        "--safe-mode",
        action="store_true",
        help=(
            "Run with the most conservative DataLoader settings "
            "(num_workers=0, pin_memory off, persistent_workers off). "
            "Use this on low-memory GPUs that hit CUDA out-of-memory during extraction."
        ),
    )
    parser.add_argument(
        "--no-pin-memory",
        action="store_true",
        help="Disable pinned memory for host-to-GPU transfer (lowers host memory pressure).",
    )
    parser.add_argument(
        "--no-persistent-workers",
        action="store_true",
        help="Disable persistent DataLoader workers (only relevant when --num-workers > 0).",
    )
    parser.add_argument(
        "--keep-tokens",
        action="store_true",
        help=(
            "Save full per-view token features [V,T,D]. "
            "Default: token-mean pooled to [V,D] (required for --pool max downstream)."
        ),
    )
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    setup_logging()
    set_seed(args.seed)
    ensure_dir(args.out)

    render_files = list_image_files(args.renders)
    if not render_files:
        LOGGER.warning("No render images found: %s", args.renders)
        return

    grouped = group_renders_by_specimen(render_files, root_dir=args.renders)
    items = sorted(grouped.items())  # stable ordering of (sid, [paths])
    transform = build_transform(args.image_size, args.crop_size)
    device = resolve_device(args.device)
    model = load_dinov3_model(args.model, device)

    # Resolve DataLoader settings. Safe mode forces the most conservative
    # configuration, which avoids the pinned-memory and worker-prefetch pressure
    # that can trigger CUDA out-of-memory on low-memory GPUs.
    if args.safe_mode:
        num_workers = 0
        use_pin_memory = False
        use_persistent_workers = False
    else:
        num_workers = args.num_workers
        use_pin_memory = (device.type == "cuda") and not args.no_pin_memory
        use_persistent_workers = (num_workers > 0) and not args.no_persistent_workers

    LOGGER.info(
        "Using device=%s model=%s keep_tokens=%s safe_mode=%s num_workers=%d pin_memory=%s persistent_workers=%s",
        device,
        args.model,
        args.keep_tokens,
        args.safe_mode,
        num_workers,
        use_pin_memory,
        use_persistent_workers,
    )

    dataset = SpecimenViewsDataset(items, transform)
    loader = DataLoader(
        dataset,
        batch_size=1,
        shuffle=False,
        drop_last=False,
        num_workers=num_workers,
        pin_memory=use_pin_memory,
        persistent_workers=use_persistent_workers,
        collate_fn=_collate_single,
    )

    n_saved = 0
    for sid, views in tqdm(loader, total=len(dataset), desc="Extracting"):
        if views is None or views.shape[0] == 0:
            LOGGER.warning("No valid images for specimen %s", sid)
            continue

        feature_arr = _embed_specimen_views(model, views, device, args.batch_size, args.keep_tokens)

        out_path = args.out / f"{sid}.npy"
        out_path.parent.mkdir(parents=True, exist_ok=True)
        np.save(out_path, feature_arr)
        n_saved += 1

    LOGGER.info("Feature extraction finished for %d specimens", n_saved)


if __name__ == "__main__":
    main()
