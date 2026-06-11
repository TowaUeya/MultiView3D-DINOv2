from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
from PIL import Image
import timm
import torch
from torchvision import transforms

LOGGER = logging.getLogger(__name__)

DINOV3_MEAN = (0.485, 0.456, 0.406)
DINOV3_STD = (0.229, 0.224, 0.225)
MODEL_ALIASES = {
    "dinov3_vits16": "vit_small_patch16_dinov3.lvd1689m",
    "dinov3_vitb16": "vit_base_patch16_dinov3.lvd1689m",  # default
    "dinov3_vitl16": "vit_large_patch16_dinov3.lvd1689m",
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
            transforms.Normalize(mean=DINOV3_MEAN, std=DINOV3_STD),
        ]
    )


def load_image_tensor(image_path: Path, transform: transforms.Compose) -> torch.Tensor:
    with Image.open(image_path) as img:
        return transform(img.convert("RGB"))


def l2_normalize(array: np.ndarray, eps: float = 1e-8) -> np.ndarray:
    norm = np.linalg.norm(array, axis=-1, keepdims=True)
    return array / np.maximum(norm, eps)


def load_dinov3_model(model_name: str, device: torch.device) -> torch.nn.Module:
    timm_name = MODEL_ALIASES.get(model_name, model_name)
    model = timm.create_model(timm_name, pretrained=True)
    model.eval()
    model.to(device)
    return model


def _drop_register_tokens(tokens: torch.Tensor, num_prefix_tokens: int) -> torch.Tensor:

    if num_prefix_tokens <= 1:
        # No register tokens (CLS only): return as-is
        return tokens
    cls = tokens[:, :1]
    patches = tokens[:, num_prefix_tokens:]
    return torch.cat([cls, patches], dim=1)


def _extract_embedding_tensor(
    out: torch.Tensor | dict[str, torch.Tensor],
    num_prefix_tokens: int = 1,
) -> torch.Tensor:
    if isinstance(out, dict):
        if "x_norm_clstoken" in out:
            return out["x_norm_clstoken"]
        if "x_cls" in out:
            return out["x_cls"]
    if isinstance(out, torch.Tensor):
        # timm's forward_features returns all tokens [B, N, D].
        # Drop register tokens only when the output is a 3D tensor.
        if out.dim() == 3:
            return _drop_register_tokens(out, num_prefix_tokens)
        return out
    raise RuntimeError("Unsupported model output format for embeddings")


def forward_embedding(model: torch.nn.Module, batch: torch.Tensor, *, enable_grad: bool = False) -> torch.Tensor:
    num_prefix_tokens = int(getattr(model, "num_prefix_tokens", 1))
    if enable_grad:
        out = model.forward_features(batch)
        return _extract_embedding_tensor(out, num_prefix_tokens)

    with torch.inference_mode():
        out = model.forward_features(batch)
    return _extract_embedding_tensor(out, num_prefix_tokens)
