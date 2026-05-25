# MultiView3D-DINOv2: DINOv2 embedding generation from multi-view renderings of 3D specimen models
Software pipeline for rendering multi-view images from 3D specimens and extracting frozen DINOv2 embeddings.

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
python -m src.render_multiview --in data/meshes --out data/renders --views 12 --size 518
```

2) **extract_features**
```bash
python -m src.extract_features --renders data/renders --out data/features --model dinov2_vits14 --device auto
```

3) **pool_embeddings**
```bash
python -m src.pool_embeddings --features data/features --out data/embeddings --pool mean
```

## Outputs
The pipeline produces:
- rendered multi-view images under `data/renders/`
- per-view DINOv2 features under `data/features/`
- specimen-level embeddings under `data/embeddings/`
- `embeddings.npy` and `ids.txt` for downstream analysis

## Notes
- This repository contains only the embedding pipeline.
- No training or fine-tuning is performed in this repository.
- DINOv2 is used as a frozen feature extractor.
- `render_multiview` requires an OpenGL runtime (for example `libGL.so.1`).
- `extract_features` downloads pretrained weights on first run unless already cached.

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
