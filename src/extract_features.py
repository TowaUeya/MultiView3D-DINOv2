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
    items = sorted(grouped.items())  # 安定した順序の (sid, [paths])
    transform = build_transform(args.image_size, args.crop_size)
    device = resolve_device(args.device)
    model = load_dinov3_model(args.model, device)
    LOGGER.info(
        "Using device=%s model=%s keep_tokens=%s num_workers=%d",
        device,
        args.model,
        args.keep_tokens,
        args.num_workers,
    )

    dataset = SpecimenViewsDataset(items, transform)
    loader = DataLoader(
        dataset,
        batch_size=1,
        shuffle=False,
        drop_last=False,
        num_workers=args.num_workers,
        pin_memory=(device.type == "cuda"),
        persistent_workers=(args.num_workers > 0),
        collate_fn=_collate_single,
    )

    n_saved = 0
    for sid, views in tqdm(loader, total=len(dataset), desc="Extracting"):
        if views is None or views.shape[0] == 0:
            LOGGER.warning("No valid images for specimen %s", sid)
            continue

        parts: list[np.ndarray] = []
        with torch.inference_mode():
            for idx in range(0, views.shape[0], args.batch_size):
                bt = views[idx : idx + args.batch_size].to(device, non_blocking=True)
                z = forward_embedding(model, bt)  # [b, T, D] (CLS+patch, register除外済み)
                if not args.keep_tokens:
                    # pool_embeddings が reduce するのと同一のトークン軸(=1)で事前 mean -> [b, D]
                    z = z.mean(dim=1)
                parts.append(z.detach().cpu().numpy())

        if len(parts) == 1:
            feature_arr = parts[0]
        else:
            feature_arr = np.concatenate(parts, axis=0)  # [V,T,D] または [V,D]
        feature_arr = feature_arr.astype(np.float32, copy=False)

        out_path = args.out / f"{sid}.npy"
        out_path.parent.mkdir(parents=True, exist_ok=True)
        np.save(out_path, feature_arr)
        n_saved += 1

    LOGGER.info("Feature extraction finished for %d specimens", n_saved)


if __name__ == "__main__":
    main()
