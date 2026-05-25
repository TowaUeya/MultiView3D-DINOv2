from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
from PIL import Image
import timm
import torch
from torchvision import transforms

LOGGER = logging.getLogger(__name__)

DINOV2_MEAN = (0.485, 0.456, 0.406)
DINOV2_STD = (0.229, 0.224, 0.225)
MODEL_ALIASES = {
    "dinov2_vits14": "vit_small_patch14_dinov2.lvd142m",
}


def resolve_device(device: str = "auto") -> torch.device:
    if device == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device == "cuda" and not torch.cuda.is_available():
        LOGGER.warning("CUDA requested but unavailable. Falling back to CPU.")
        return torch.device("cpu")
    return torch.device(device)


def build_transform(image_size: int = 224, crop_size: int = 224) -> transforms.Compose:
    return transforms.Compose(
        [
            transforms.Resize(image_size, interpolation=transforms.InterpolationMode.BICUBIC),
            transforms.CenterCrop(crop_size),
            transforms.ToTensor(),
            transforms.Normalize(mean=DINOV2_MEAN, std=DINOV2_STD),
        ]
    )


def load_image_tensor(image_path: Path, transform: transforms.Compose) -> torch.Tensor:
    with Image.open(image_path) as img:
        return transform(img.convert("RGB"))


def l2_normalize(array: np.ndarray, eps: float = 1e-8) -> np.ndarray:
    norm = np.linalg.norm(array, axis=-1, keepdims=True)
    return array / np.maximum(norm, eps)


def load_dinov2_model(model_name: str, device: torch.device) -> torch.nn.Module:
    timm_name = MODEL_ALIASES.get(model_name, model_name)
    model = timm.create_model(timm_name, pretrained=True)
    model.eval()
    model.to(device)
    return model


def _extract_embedding_tensor(out: torch.Tensor | dict[str, torch.Tensor]) -> torch.Tensor:
    if isinstance(out, dict):
        if "x_norm_clstoken" in out:
            return out["x_norm_clstoken"]
        if "x_cls" in out:
            return out["x_cls"]
    if isinstance(out, torch.Tensor):
        return out
    raise RuntimeError("Unsupported model output format for embeddings")


def forward_embedding(model: torch.nn.Module, batch: torch.Tensor, *, enable_grad: bool = False) -> torch.Tensor:
    if enable_grad:
        out = model.forward_features(batch)
        return _extract_embedding_tensor(out)

    with torch.inference_mode():
        out = model.forward_features(batch)
    return _extract_embedding_tensor(out)
