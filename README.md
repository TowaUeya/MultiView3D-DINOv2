# MultiView3D-DINOv2: DINOv2 embedding generation from multi view renderings of 3D specimen models
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

## Notes
- This repository contains only the embedding pipeline.
- No training or fine-tuning is performed in this repository.
- DINOv2 is used as a frozen feature extractor.
- `render_multiview` requires an OpenGL runtime (for example `libGL.so.1`).
- `extract_features` downloads pretrained weights on first run unless already cached.

## Citation
```bibtex
@software{multiview3d_dinov2,
  title = {MultiView3D-DINOv2},
  author = {Your Name or Team},
  year = {2026},
  note = {Software pipeline for multi-view rendering and frozen DINOv2 embedding generation}
}
```

## Links
* Source code: [https://github.com/TowaUeya/MultiView3D-DINOv2](https://github.com/TowaUeya/MultiView3D-DINOv2)
* Archived version: [https://doi.org/10.5281/zenodo.20258321](https://doi.org/10.5281/zenodo.20258321)

## Related Repositories

This repository provides the embedding-generation pipeline. Downstream analyses and related tools are maintained in the following repositories:

- **Morphological-Embedding-Space-Analyzer**  
  [https://github.com/TowaUeya/Morphological-Embedding-Space-Analyzer](https://github.com/TowaUeya/Morphological-Embedding-Space-Analyzer)  
  Analysis toolkit for evaluating nearest-neighbor retrieval, HDBSCAN-based auxiliary structure, leaf-core regions, residual samples, and publication figures.

- **Repository-Name-2**  
  https://github.com/TowaUeya/Repository-Name-2  
  Briefly describe what this repository is used for and how it relates to MultiView3D-DINOv2.
