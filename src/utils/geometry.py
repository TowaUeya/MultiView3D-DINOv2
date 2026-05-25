from __future__ import annotations

import logging
import math
import tempfile
from pathlib import Path

import numpy as np
import open3d as o3d

LOGGER = logging.getLogger(__name__)


def _normalize_off_header(raw_text: str) -> str | None:
    first_line, has_newline, rest = raw_text.partition("\n")
    if first_line.startswith("OFF") and len(first_line) > 3 and first_line[3].isdigit():
        return f"OFF\n{first_line[3:]}\n{rest}" if has_newline else f"OFF\n{first_line[3:]}\n"
    return None


def load_geometry(path: Path) -> o3d.geometry.Geometry:
    suffix = path.suffix.lower()
    point_cloud_suffixes = {".ply", ".pcd", ".pts", ".xyz", ".xyzn", ".xyzrgb"}

    if suffix in {".obj", ".stl"}:
        mesh = o3d.io.read_triangle_mesh(str(path))
        if mesh.is_empty():
            raise ValueError("triangle mesh is empty")
        mesh.compute_vertex_normals()
        return mesh

    if suffix == ".off":
        try:
            raw_text = path.read_text(encoding="utf-8", errors="replace")
        except OSError:
            raw_text = ""

        normalized = _normalize_off_header(raw_text)
        candidate_path = path
        tmp_path: Path | None = None

        try:
            if normalized is not None:
                with tempfile.NamedTemporaryFile("w", suffix=".off", delete=False, encoding="utf-8") as tmp:
                    tmp.write(normalized)
                    tmp_path = Path(tmp.name)
                candidate_path = tmp_path
                LOGGER.warning("OFF header normalized fallback used: %s", path)

            mesh = o3d.io.read_triangle_mesh(str(candidate_path))
            if not mesh.is_empty() and len(mesh.triangles) > 0:
                mesh.compute_vertex_normals()
                return mesh
        finally:
            if tmp_path is not None:
                tmp_path.unlink(missing_ok=True)

        raise ValueError("failed to read geometry as mesh/point cloud")

    mesh = o3d.io.read_triangle_mesh(str(path))
    if not mesh.is_empty() and len(mesh.triangles) > 0:
        mesh.compute_vertex_normals()
        return mesh

    if suffix in point_cloud_suffixes:
        pcd = o3d.io.read_point_cloud(str(path))
        if not pcd.is_empty():
            return pcd

    raise ValueError("failed to read geometry as mesh/point cloud")


def get_points(geom: o3d.geometry.Geometry) -> np.ndarray:
    if isinstance(geom, o3d.geometry.TriangleMesh):
        pts = np.asarray(geom.vertices)
    elif isinstance(geom, o3d.geometry.PointCloud):
        pts = np.asarray(geom.points)
    else:
        raise TypeError(f"Unsupported geometry type: {type(geom)}")

    if pts.size == 0:
        raise ValueError("geometry has no points")
    return pts


def normalize_geometry(
    geom: o3d.geometry.Geometry,
    target_extent: float = 1.0,
    quantile_low: float = 0.01,
    quantile_high: float = 0.99,
    drop_trimmed_points: bool = False,
) -> o3d.geometry.Geometry:
    if not 0.0 <= quantile_low < quantile_high <= 1.0:
        raise ValueError("quantile range must satisfy 0.0 <= low < high <= 1.0")

    points = get_points(geom)
    raw_min_b = points.min(axis=0)
    raw_max_b = points.max(axis=0)
    robust_min_b = np.quantile(points, quantile_low, axis=0)
    robust_max_b = np.quantile(points, quantile_high, axis=0)

    raw_extent_xyz = raw_max_b - raw_min_b
    robust_extent_xyz = robust_max_b - robust_min_b
    robust_extent = float(np.max(robust_extent_xyz))
    if robust_extent <= 0:
        raise ValueError("invalid geometry extent")

    LOGGER.info(
        "Geometry normalization extents raw_extent=%s robust_extent=%s quantiles=(%.4f, %.4f)",
        np.array2string(raw_extent_xyz, precision=6),
        np.array2string(robust_extent_xyz, precision=6),
        quantile_low,
        quantile_high,
    )

    center = (robust_min_b + robust_max_b) / 2.0
    inlier_mask = np.logical_and(points >= robust_min_b, points <= robust_max_b).all(axis=1)
    trimmed_count = int((~inlier_mask).sum())
    if drop_trimmed_points and trimmed_count > 0:
        if isinstance(geom, o3d.geometry.PointCloud):
            inlier_indices = np.flatnonzero(inlier_mask).tolist()
            geom = geom.select_by_index(inlier_indices)
        elif isinstance(geom, o3d.geometry.TriangleMesh):
            geom.remove_vertices_by_mask((~inlier_mask).tolist())
            geom.remove_unreferenced_vertices()
        else:
            LOGGER.warning("drop_trimmed_points ignored for unsupported geometry type: %s", type(geom))
        LOGGER.info(
            "Dropped %d / %d points outside robust bbox before normalization",
            trimmed_count,
            points.shape[0],
        )

    geom = geom.translate(-center)
    scale = target_extent / robust_extent
    geom = geom.scale(scale, center=(0.0, 0.0, 0.0))
    return geom


def fibonacci_sphere_points(n_views: int, radius: float = 2.0) -> np.ndarray:
    if n_views < 1:
        raise ValueError("n_views must be >= 1")

    points: list[np.ndarray] = []
    phi = math.pi * (3.0 - math.sqrt(5.0))
    for i in range(n_views):
        y = 1 - (i / float(max(n_views - 1, 1))) * 2
        r = math.sqrt(max(0.0, 1 - y * y))
        theta = phi * i
        x = math.cos(theta) * r
        z = math.sin(theta) * r
        points.append(np.array([x, y, z], dtype=np.float32) * radius)
    return np.asarray(points, dtype=np.float32)
