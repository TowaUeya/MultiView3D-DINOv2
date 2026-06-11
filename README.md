# MultiView3D-DINOv2: DINOv2 embedding generation from multi-view renderings of 3D specimen models
Software pipeline for rendering multi-view images from 3D specimens and extracting frozen DINOv3 embeddings.

## Requirements
- Python 3.10+
- PyTorch-compatible environment (CPU or CUDA)
- Dependencies in `requirements.txt`

## Installation
```bash
pip install -r requirements.txt
```

## Data Layout
Create the following folders under project root:

```text
data/
  meshes/
  renders/
  features/
  embeddings/
```

## Usage
Run the pipeline in three steps:

1) **render_multiview**
```bash
python -m src.render_multiview --in data/meshes --out data/renders --views 12 --size 768 --auto-zoom --auto-zoom-probes 12 --jobs 8
```

2) **extract_features**
```bash
python -m src.extract_features --renders data/renders --out data/features --model dinov3_vitb16 --device auto --image-size 768 --crop-size 768
```

On low-memory GPUs that crash with `CUDA error: out of memory`, add `--safe-mode`:
```bash
python -m src.extract_features --renders data/renders --out data/features --model dinov3_vitb16 --device auto --image-size 768 --crop-size 768 --safe-mode
```
See [Troubleshooting](#troubleshooting) for details.

3) **pool_embeddings**
```bash
python -m src.pool_embeddings --features data/features --out data/embeddings --pool mean
```

## Outputs
The pipeline produces:
- rendered multi-view images under `data/renders/`
- per-view DINOv3 features under `data/features/`
- specimen-level embeddings under `data/embeddings/`
- `embeddings.npy` and `ids.txt` for downstream analysis

## Notes
- This repository contains only the embedding pipeline.
- No training or fine-tuning is performed in this repository.
- DINOv3 is used as a frozen feature extractor.
- `render_multiview` requires an OpenGL runtime (for example `libGL.so.1`).
- `extract_features` downloads pretrained weights on first run unless already cached.

## Troubleshooting

### CUDA out of memory during `extract_features`

On GPUs with limited memory (for example, 11 GB), feature extraction can fail with:

```text
torch.AcceleratorError: CUDA error: out of memory
```

This often originates from the DataLoader: pinned memory and persistent worker
prefetching keep several large multi-view batches resident at once. The first
thing to try is **safe mode**, which forces the most conservative DataLoader
configuration (`num_workers=0`, `pin_memory` off, `persistent_workers` off):

```bash
python -m src.extract_features \
  --renders data/renders \
  --out data/features \
  --model dinov3_vitb16 \
  --device auto \
  --image-size 768 \
  --crop-size 768 \
  --safe-mode
```

The relevant flags are:

| Flag | Effect |
| --- | --- |
| `--safe-mode` | Forces `num_workers=0`, `pin_memory` off, and `persistent_workers` off in one step. |
| `--no-pin-memory` | Disables pinned host memory (a transfer-speed optimization, not required). |
| `--no-persistent-workers` | Disables worker reuse across iterations (only relevant when `--num-workers > 0`). |

`pin_memory=True` only speeds up host-to-GPU transfer and `persistent_workers=True`
only helps across multiple epochs, so disabling them is safe for a single
extraction pass.

Independently of these flags, the forward pass already recovers from CUDA
out-of-memory by automatically halving the per-forward chunk size (down to a
single view) and retrying, so a transient spike does not abort the whole run.

If safe mode alone is not enough, reduce the input resolution before pushing it
higher again, e.g. `--image-size 384 --crop-size 384`, then `518`, then `768`.
You can also lower `--batch-size` (images per forward pass).

## Citation
```bibtex
@software{multiview3d_dinov2,
  title  = {MultiView3D-DINOv2},
  author = {Ueya, Towa and Iba, Yasuhiro},
  year   = {2026},
  url    = {https://github.com/TowaUeya/MultiView3D-DINOv2},
  doi    = {10.5281/zenodo.20258321},
  note   = {Software pipeline for multi-view rendering and frozen DINOv2 embedding generation}
}
```

## Links
* Source code: [https://github.com/TowaUeya/MultiView3D-DINOv2](https://github.com/TowaUeya/MultiView3D-DINOv2)
* Archived version: [https://doi.org/10.5281/zenodo.20258321](https://doi.org/10.5281/zenodo.20258321)

## Related Repositories

MultiView3D-DINOv2 is the embedding-generation component of this ecosystem. The workflow starts with multi-view rendering and frozen DINOv2 feature extraction in this repository, followed by embedding-space analysis and explainability visualization in the related repositories.

This repository is part of a small research software ecosystem for morphology-based analysis of 3D specimen models.

- **Embedding generation**  
  **MultiView3D-DINOv2**  
  [https://github.com/TowaUeya/MultiView3D-DINOv2](https://github.com/TowaUeya/MultiView3D-DINOv2)  
  Renders multi-view images from 3D specimen models and extracts frozen DINOv2 features, producing specimen-level embeddings and rendered views for downstream analysis and visualization.

- **Embedding-space analysis**  
  **Morphological-Embedding-Space-Analyzer**  
  [https://github.com/TowaUeya/Morphological-Embedding-Space-Analyzer](https://github.com/TowaUeya/Morphological-Embedding-Space-Analyzer)  
  Performs downstream analysis of specimen-level embeddings, including retrieval evaluation, HDBSCAN-based clustering, leaf-core and residual sample extraction, embedding-space visualization, and publication-oriented figure generation.

- **Embedding explainability**  
  **Morphological-Embedding-Explainability**  
  [https://github.com/TowaUeya/Morphological-Embedding-Explainability](https://github.com/TowaUeya/Morphological-Embedding-Explainability)  
  Uses rendered multi-view images, embeddings, specimen IDs, and optional cluster information to visualize attention rollout and image-level visual cues associated with ViT-based embedding formation.
