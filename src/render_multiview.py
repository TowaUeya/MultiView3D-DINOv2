from __future__ import annotations

import argparse
import copy
import csv
import logging
from pathlib import Path

import numpy as np
import open3d as o3d
from tqdm import tqdm

from src.utils.geometry import fibonacci_sphere_points, get_points, load_geometry, normalize_geometry
from src.utils.io import ensure_dir, list_mesh_files, set_seed, setup_logging

LOGGER = logging.getLogger(__name__)


def _compute_camera_light_direction(eye: np.ndarray) -> np.ndarray:
    """Return camera-following sun-light direction."""
    eye_np = np.asarray(eye, dtype=np.float32)
    norm = float(np.linalg.norm(eye_np))
    if norm <= 0:
        raise ValueError("Camera eye vector has zero norm; cannot compute camera light direction.")
    eye_dir = eye_np / norm

    return -eye_dir




def _prepare_geometry_for_appearance(
    geom: o3d.geometry.Geometry,
    appearance: str,
    gray_rgb: tuple[float, float, float] = (0.8, 0.8, 0.8),
) -> o3d.geometry.Geometry:
    """Prepare geometry for rendering appearance modes."""
    if appearance == "color_lit":
        return geom

    if appearance != "gray_lit":
        raise ValueError(f"Unsupported appearance: {appearance}")

    gray = np.asarray(gray_rgb, dtype=np.float64)

    if isinstance(geom, o3d.geometry.TriangleMesh):
        mesh = copy.deepcopy(geom)
        n_vertices = int(np.asarray(mesh.vertices).shape[0])
        mesh.vertex_colors = o3d.utility.Vector3dVector(np.tile(gray, (n_vertices, 1)))

        try:
            mesh.textures = []
        except Exception:
            LOGGER.debug("Failed to clear textures for gray_lit.", exc_info=True)

        try:
            mesh.triangle_uvs = o3d.utility.Vector2dVector()
        except Exception:
            LOGGER.debug("Failed to clear triangle_uvs for gray_lit.", exc_info=True)

        try:
            mesh.triangle_material_ids = o3d.utility.IntVector()
        except Exception:
            LOGGER.debug("Failed to clear triangle_material_ids for gray_lit.", exc_info=True)

        return mesh

    if isinstance(geom, o3d.geometry.PointCloud):
        pcd = copy.deepcopy(geom)
        n_points = int(np.asarray(pcd.points).shape[0])
        pcd.colors = o3d.utility.Vector3dVector(np.tile(gray, (n_points, 1)))
        return pcd

    return geom

def _make_material_for_appearance(
    geom: o3d.geometry.Geometry,
    appearance: str,
) -> o3d.visualization.rendering.MaterialRecord:
    """Build a material for the selected appearance mode."""
    mat = o3d.visualization.rendering.MaterialRecord()
    mat.shader = "defaultLit"

    if appearance == "gray_lit":
        mat.base_color = (0.8, 0.8, 0.8, 1.0)
    elif appearance == "color_lit":
        if isinstance(geom, o3d.geometry.TriangleMesh):
            has_vertex_colors = geom.has_vertex_colors()
            textures = getattr(geom, "textures", [])
            num_textures = len(textures) if textures is not None else 0
            if num_textures > 0:
                try:
                    mat.albedo_img = textures[0]
                except Exception:
                    LOGGER.debug("Failed to assign albedo texture; continuing without albedo_img.", exc_info=True)

            if not has_vertex_colors and num_textures == 0:
                mat.base_color = (0.8, 0.8, 0.8, 1.0)
        elif isinstance(geom, o3d.geometry.PointCloud):
            if not geom.has_colors():
                mat.base_color = (0.8, 0.8, 0.8, 1.0)
    else:
        raise ValueError(f"Unsupported appearance: {appearance}")

    if isinstance(geom, o3d.geometry.PointCloud):
        mat.point_size = 3.0
    return mat


def _compute_bbox_fill_ratio(
    depth_image: o3d.geometry.Image | None = None,
    image: o3d.geometry.Image | None = None,
    bg_threshold: int = 245,
) -> float:
    if depth_image is not None:
        depth_np = np.asarray(depth_image)
        if depth_np.ndim == 3 and depth_np.shape[2] == 1:
            depth_np = depth_np[..., 0]
        if depth_np.ndim != 2:
            return 0.0

        finite_mask = np.isfinite(depth_np)
        positive_mask = depth_np > 0
        valid_mask = finite_mask & positive_mask
        if not np.any(valid_mask):
            return 0.0

        bg_depth = float(np.max(depth_np[valid_mask]))
        fg_mask = valid_mask & (depth_np < (bg_depth - 1e-6))
        if not np.any(fg_mask):
            return 0.0

        fg_indices = np.argwhere(fg_mask)
        ymin, xmin = fg_indices.min(axis=0)
        ymax, xmax = fg_indices.max(axis=0)
        bbox_area = float((ymax - ymin + 1) * (xmax - xmin + 1))
        image_area = float(depth_np.shape[0] * depth_np.shape[1])
        return bbox_area / image_area if image_area > 0 else 0.0

    if image is None:
        return 0.0

    image_np = np.asarray(image)
    if image_np.ndim != 3 or image_np.shape[2] < 3:
        return 0.0

    rgb = image_np[..., :3]
    non_bg_mask = np.any(rgb < bg_threshold, axis=2)
    non_bg_indices = np.argwhere(non_bg_mask)
    if non_bg_indices.size == 0:
        return 0.0

    ymin, xmin = non_bg_indices.min(axis=0)
    ymax, xmax = non_bg_indices.max(axis=0)
    bbox_area = float((ymax - ymin + 1) * (xmax - xmin + 1))
    image_area = float(image_np.shape[0] * image_np.shape[1])
    return bbox_area / image_area if image_area > 0 else 0.0


def _compute_bbox_fill_and_touches_border(
    depth_image: o3d.geometry.Image | None = None,
    image: o3d.geometry.Image | None = None,
    bg_threshold: int = 245,
) -> tuple[float, bool]:
    if depth_image is not None:
        depth_np = np.asarray(depth_image)
        if depth_np.ndim == 3 and depth_np.shape[2] == 1:
            depth_np = depth_np[..., 0]
        if depth_np.ndim != 2:
            return 0.0, False

        finite_mask = np.isfinite(depth_np)
        positive_mask = depth_np > 0
        valid_mask = finite_mask & positive_mask
        if not np.any(valid_mask):
            return 0.0, False

        bg_depth = float(np.max(depth_np[valid_mask]))
        fg_mask = valid_mask & (depth_np < (bg_depth - 1e-6))
        if not np.any(fg_mask):
            return 0.0, False

        fg_indices = np.argwhere(fg_mask)
        ymin, xmin = fg_indices.min(axis=0)
        ymax, xmax = fg_indices.max(axis=0)
        bbox_area = float((ymax - ymin + 1) * (xmax - xmin + 1))
        image_area = float(depth_np.shape[0] * depth_np.shape[1])
        touches_border = bool(
            ymin == 0 or xmin == 0 or ymax == (depth_np.shape[0] - 1) or xmax == (depth_np.shape[1] - 1)
        )
        fill_ratio = bbox_area / image_area if image_area > 0 else 0.0
        return fill_ratio, touches_border

    if image is None:
        return 0.0, False

    image_np = np.asarray(image)
    if image_np.ndim != 3 or image_np.shape[2] < 3:
        return 0.0, False

    rgb = image_np[..., :3]
    non_bg_mask = np.any(rgb < bg_threshold, axis=2)
    non_bg_indices = np.argwhere(non_bg_mask)
    if non_bg_indices.size == 0:
        return 0.0, False

    ymin, xmin = non_bg_indices.min(axis=0)
    ymax, xmax = non_bg_indices.max(axis=0)
    bbox_area = float((ymax - ymin + 1) * (xmax - xmin + 1))
    image_area = float(image_np.shape[0] * image_np.shape[1])
    touches_border = bool(
        ymin == 0 or xmin == 0 or ymax == (image_np.shape[0] - 1) or xmax == (image_np.shape[1] - 1)
    )
    fill_ratio = bbox_area / image_area if image_area > 0 else 0.0
    return fill_ratio, touches_border


def _autotune_camera_radius(
    renderer: o3d.visualization.rendering.OffscreenRenderer,
    center: np.ndarray,
    up: np.ndarray,
    fov_deg: float,
    target_fill_min: float,
    target_fill_max: float,
    preview_directions: np.ndarray | None = None,
    initial_radius: float = 2.0,
    min_radius: float = 0.25,
    max_radius: float = 8.0,
    max_iter: int = 12,
    log_prefix: str = "",
) -> tuple[float, float, float, int]:
    if preview_directions is None:
        preview_directions = np.array([[0.0, 0.0, 1.0]], dtype=np.float32)
    if preview_directions.ndim != 2 or preview_directions.shape[1] != 3:
        raise ValueError("preview_directions must be an array with shape [N, 3]")
    depth_fallback_logged = False

    def render_fill_stats(radius: float) -> tuple[float, float]:
        nonlocal depth_fallback_logged
        fill_values: list[float] = []
        for direction in preview_directions:
            eye = direction * radius
            renderer.setup_camera(fov_deg, center, eye, up)
            preview = renderer.render_to_image()
            try:
                preview_depth = renderer.render_to_depth_image()
                fill_values.append(_compute_bbox_fill_ratio(depth_image=preview_depth))
            except Exception:
                if not depth_fallback_logged:
                    LOGGER.warning(
                        "Depth preview failed for auto-zoom%s; fallback to RGB thresholding only.",
                        f" ({log_prefix})" if log_prefix else "",
                        exc_info=True,
                    )
                    depth_fallback_logged = True
                fill_values.append(_compute_bbox_fill_ratio(image=preview))

        if not fill_values:
            return 0.0, 0.0
        return float(np.min(fill_values)), float(np.max(fill_values))

    lo = min_radius
    hi = max_radius
    current_radius = float(np.clip(initial_radius, lo, hi))
    current_fill_min, current_fill_max = render_fill_stats(current_radius)
    trace: list[tuple[int, float, float, float]] = [(0, current_radius, current_fill_min, current_fill_max)]
    best_radius = current_radius
    best_fill_min = current_fill_min
    best_fill_max = current_fill_max
    best_penalty = max(0.0, current_fill_max - target_fill_max) + max(0.0, target_fill_min - current_fill_min)

    if target_fill_min <= current_fill_min and current_fill_max <= target_fill_max:
        trace_text = ", ".join(
            [
                f"iter={it}:radius={rad:.4f},fill_min={fill_min:.4f},fill_max={fill_max:.4f}"
                for it, rad, fill_min, fill_max in trace
            ]
        )
        LOGGER.info("Auto-zoom trace%s %s", f" ({log_prefix})" if log_prefix else "", trace_text)
        return best_radius, best_fill_min, best_fill_max, 0

    for it in range(1, max_iter + 1):
        if current_fill_max > target_fill_max:
            lo = current_radius
        elif current_fill_min < target_fill_min:
            hi = current_radius
        else:
            break
        current_radius = (lo + hi) * 0.5
        current_fill_min, current_fill_max = render_fill_stats(current_radius)
        trace.append((it, current_radius, current_fill_min, current_fill_max))

        penalty = max(0.0, current_fill_max - target_fill_max) + max(0.0, target_fill_min - current_fill_min)
        if penalty < best_penalty:
            best_radius = current_radius
            best_fill_min = current_fill_min
            best_fill_max = current_fill_max
            best_penalty = penalty

        if target_fill_min <= current_fill_min and current_fill_max <= target_fill_max:
            trace_text = ", ".join(
                [
                    f"iter={trace_it}:radius={rad:.4f},fill_min={fill_min:.4f},fill_max={fill_max:.4f}"
                    for trace_it, rad, fill_min, fill_max in trace
                ]
            )
            LOGGER.info("Auto-zoom trace%s %s", f" ({log_prefix})" if log_prefix else "", trace_text)
            return current_radius, current_fill_min, current_fill_max, it

    trace_text = ", ".join(
        [
            f"iter={it}:radius={rad:.4f},fill_min={fill_min:.4f},fill_max={fill_max:.4f}"
            for it, rad, fill_min, fill_max in trace
        ]
    )
    LOGGER.info("Auto-zoom trace%s %s", f" ({log_prefix})" if log_prefix else "", trace_text)
    return best_radius, best_fill_min, best_fill_max, max_iter


def _evaluate_radius_on_directions(
    renderer: o3d.visualization.rendering.OffscreenRenderer,
    center: np.ndarray,
    up: np.ndarray,
    fov_deg: float,
    directions: np.ndarray,
    radius: float,
) -> tuple[float, float, int]:
    fill_values: list[float] = []
    border_touch_count = 0
    depth_fallback_logged = False
    for direction in directions:
        eye = direction * radius
        renderer.setup_camera(fov_deg, center, eye, up)
        img = renderer.render_to_image()
        try:
            depth = renderer.render_to_depth_image()
            fill_ratio, touches_border = _compute_bbox_fill_and_touches_border(depth_image=depth)
        except Exception:
            if not depth_fallback_logged:
                LOGGER.warning("Depth post-check failed; fallback to RGB thresholding only.", exc_info=True)
                depth_fallback_logged = True
            fill_ratio, touches_border = _compute_bbox_fill_and_touches_border(image=img)
        fill_values.append(fill_ratio)
        if touches_border:
            border_touch_count += 1

    if not fill_values:
        return 0.0, 0.0, 0
    return float(np.min(fill_values)), float(np.max(fill_values)), border_touch_count


def _apply_auto_zoom_safety_adjustment(
    renderer: o3d.visualization.rendering.OffscreenRenderer,
    center: np.ndarray,
    up: np.ndarray,
    fov_deg: float,
    directions: np.ndarray,
    initial_radius: float,
    safe_margin: float,
    max_safety_steps: int,
    max_radius: float = 8.0,
) -> tuple[float, int, int, float, float]:
    radius = min(initial_radius, max_radius)
    safety_steps_used = 0
    fill_min, fill_max, border_touch_count = _evaluate_radius_on_directions(
        renderer=renderer,
        center=center,
        up=up,
        fov_deg=fov_deg,
        directions=directions,
        radius=radius,
    )
    while border_touch_count > 0 and safety_steps_used < max_safety_steps and radius < max_radius:
        next_radius = min(radius * (1.0 + safe_margin), max_radius)
        if next_radius <= radius:
            break
        radius = next_radius
        safety_steps_used += 1
        fill_min, fill_max, border_touch_count = _evaluate_radius_on_directions(
            renderer=renderer,
            center=center,
            up=up,
            fov_deg=fov_deg,
            directions=directions,
            radius=radius,
        )

    return radius, border_touch_count, safety_steps_used, fill_min, fill_max


def _compute_safe_min_camera_radius(
    geom: o3d.geometry.Geometry,
    min_radius_floor: float = 0.25,
    margin_ratio: float = 0.05,
) -> float:
    points = get_points(geom)
    max_norm = float(np.max(np.linalg.norm(points, axis=1)))
    if not np.isfinite(max_norm) or max_norm <= 0:
        return min_radius_floor
    return max(min_radius_floor, max_norm * (1.0 + margin_ratio))


def _render_scale_views(
    renderer: o3d.visualization.rendering.OffscreenRenderer,
    scene: o3d.visualization.rendering.Open3DScene,
    specimen_out_dir: Path,
    sid: str,
    scale_name: str,
    views: int,
    center: np.ndarray,
    up: np.ndarray,
    fov_deg: float,
    final_radius: float,
    multiscale: bool,
    light_mode: str,
    light_direction: tuple[float, float, float],
    light_color: tuple[float, float, float],
    light_intensity: float,
    lighting_enabled: bool,
) -> bool:
    camera_positions = fibonacci_sphere_points(views, radius=final_radius)
    ok = True
    for i, eye in enumerate(camera_positions):
        try:
            if lighting_enabled:
                if light_mode == "camera":
                    camera_light_direction = _compute_camera_light_direction(eye)
                    scene.scene.set_sun_light(camera_light_direction, light_color, light_intensity)
                    scene.scene.enable_sun_light(True)
                    LOGGER.debug("Camera light direction for view %d: %s", i, camera_light_direction.tolist())
                elif light_mode == "world":
                    scene.scene.set_sun_light(light_direction, light_color, light_intensity)
                    scene.scene.enable_sun_light(True)
                else:
                    raise ValueError(f"Unsupported light_mode: {light_mode}")
            else:
                scene.scene.enable_sun_light(False)

            renderer.setup_camera(fov_deg, center, eye, up)
            img = renderer.render_to_image()
            if multiscale:
                out_path = specimen_out_dir / f"{sid}_{scale_name}_view{i:02d}.png"
            else:
                out_path = specimen_out_dir / f"{sid}_view{i:02d}.png"
            o3d.io.write_image(str(out_path), img)
        except Exception as e:
            ok = False
            LOGGER.exception("Render failed specimen=%s scale=%s view=%d: %s", sid, scale_name, i, e)
    return ok


def render_specimen(
    renderer: o3d.visualization.rendering.OffscreenRenderer,
    mesh_path: Path,
    input_root: Path,
    out_dir: Path,
    views: int,
    size: int,
    light_direction: tuple[float, float, float],
    light_color: tuple[float, float, float],
    light_intensity: float,
    light_mode: str,
    appearance: str,
    auto_zoom: bool,
    target_fill_min: float,
    target_fill_max: float,
    multiscale_zoom: bool,
    loose_fill_min: float,
    loose_fill_max: float,
    up_fill_min: float,
    up_fill_max: float,
    auto_zoom_probes: int,
    auto_zoom_safe_margin: float,
    auto_zoom_max_safety_steps: int,
) -> tuple[bool, list[dict[str, str | float | bool | int]]]:
    mesh_rel = mesh_path.relative_to(input_root)
    sid = mesh_rel.stem
    specimen_out_dir = out_dir / mesh_rel.parent
    ensure_dir(specimen_out_dir)
    try:
        geom = normalize_geometry(load_geometry(mesh_path))
    except Exception as e:
        LOGGER.exception("Failed to load/normalize %s: %s", mesh_path, e)
        return False, []

    if appearance == "color_lit":
        if isinstance(geom, o3d.geometry.TriangleMesh):
            has_vertex_colors = geom.has_vertex_colors()
            has_triangle_uvs = geom.has_triangle_uvs()
            textures = getattr(geom, "textures", [])
            num_textures = len(textures) if textures is not None else 0
            LOGGER.debug(
                "Color appearance for %s: has_vertex_colors=%s has_triangle_uvs=%s num_textures=%d",
                mesh_rel.as_posix(),
                has_vertex_colors,
                has_triangle_uvs,
                num_textures,
            )
            if not has_vertex_colors and num_textures == 0:
                LOGGER.warning(
                    "color_lit requested but no vertex colors/textures were detected for %s; "
                    "falling back to gray material.",
                    mesh_rel.as_posix(),
                )
        elif isinstance(geom, o3d.geometry.PointCloud):
            has_point_colors = geom.has_colors()
            LOGGER.debug(
                "Color appearance for %s: has_point_colors=%s",
                mesh_rel.as_posix(),
                has_point_colors,
            )
            if not has_point_colors:
                LOGGER.warning(
                    "color_lit requested but no point colors were detected for %s; "
                    "falling back to gray material.",
                    mesh_rel.as_posix(),
                )

    if appearance == "gray_lit":
        if isinstance(geom, o3d.geometry.TriangleMesh):
            has_vertex_colors = geom.has_vertex_colors()
            has_triangle_uvs = geom.has_triangle_uvs()
            textures = getattr(geom, "textures", [])
            num_textures = len(textures) if textures is not None else 0
            if has_vertex_colors or has_triangle_uvs or num_textures > 0:
                LOGGER.info(
                    "gray_lit requested; overriding vertex colors/textures for shape-only rendering: %s",
                    mesh_rel.as_posix(),
                )
        elif isinstance(geom, o3d.geometry.PointCloud) and geom.has_colors():
            LOGGER.info(
                "gray_lit requested; overriding point colors for shape-only rendering: %s",
                mesh_rel.as_posix(),
            )

    render_geom = _prepare_geometry_for_appearance(geom=geom, appearance=appearance)
    mat = _make_material_for_appearance(geom=render_geom, appearance=appearance)

    scene = renderer.scene
    scene.clear_geometry()
    scene.add_geometry("specimen", render_geom, mat)
    lighting_enabled = appearance.endswith("_lit")
    if lighting_enabled:
        scene.scene.enable_sun_light(True)
    else:
        scene.scene.enable_sun_light(False)

    center = np.array([0.0, 0.0, 0.0], dtype=np.float32)
    up = np.array([0.0, 1.0, 0.0], dtype=np.float32)
    fov_deg = 60.0
    if multiscale_zoom:
        views_per_scale = views // 2
        scale_configs = [
            {"name": "loose", "views": views_per_scale, "target_fill_min": loose_fill_min, "target_fill_max": loose_fill_max},
            {"name": "up", "views": views_per_scale, "target_fill_min": up_fill_min, "target_fill_max": up_fill_max},
        ]
    else:
        scale_configs = [
            {"name": "single", "views": views, "target_fill_min": target_fill_min, "target_fill_max": target_fill_max},
        ]

    ok = True
    zoom_rows: list[dict[str, str | float | bool | int]] = []
    for scale_cfg in scale_configs:
        scale_name = str(scale_cfg["name"])
        scale_views = int(scale_cfg["views"])
        scale_target_fill_min = float(scale_cfg["target_fill_min"])
        scale_target_fill_max = float(scale_cfg["target_fill_max"])
        final_directions = fibonacci_sphere_points(scale_views, radius=1.0).astype(np.float32)

        final_radius = 2.0
        preview_fill_min = float("nan")
        preview_fill_max = float("nan")
        postcheck_border_touch_count = 0
        safety_steps_used = 0
        initial_autozoom_radius = float("nan")
        if auto_zoom:
            safe_min_radius = _compute_safe_min_camera_radius(geom)
            probe_count = max(1, min(auto_zoom_probes, scale_views))
            probe_dirs = final_directions
            final_radius, preview_fill_min, preview_fill_max, iters = _autotune_camera_radius(
                renderer=renderer,
                center=center,
                up=up,
                fov_deg=fov_deg,
                target_fill_min=scale_target_fill_min,
                target_fill_max=scale_target_fill_max,
                preview_directions=probe_dirs,
                min_radius=safe_min_radius,
                log_prefix=f"{mesh_rel.as_posix()}:{scale_name}",
            )
            initial_autozoom_radius = final_radius
            if final_radius < safe_min_radius:
                LOGGER.warning(
                    "Auto-zoom radius clamped to safe minimum for %s scale=%s: %.4f -> %.4f",
                    mesh_rel.as_posix(),
                    scale_name,
                    final_radius,
                    safe_min_radius,
                )
                final_radius = safe_min_radius
            final_radius, postcheck_border_touch_count, safety_steps_used, preview_fill_min, preview_fill_max = (
                _apply_auto_zoom_safety_adjustment(
                    renderer=renderer,
                    center=center,
                    up=up,
                    fov_deg=fov_deg,
                    directions=final_directions,
                    initial_radius=final_radius,
                    safe_margin=auto_zoom_safe_margin,
                    max_safety_steps=auto_zoom_max_safety_steps,
                )
            )
            LOGGER.info(
                "Auto-zoom specimen=%s scale=%s fill_min=%.4f fill_max=%.4f initial_radius=%.4f safety_radius=%.4f safe_min_radius=%.4f iterations=%d probes=%d views_for_zoom=%d safety_steps=%d final_border_touches=%d target=[%.2f, %.2f]",
                mesh_rel.as_posix(),
                scale_name,
                preview_fill_min,
                preview_fill_max,
                initial_autozoom_radius,
                final_radius,
                safe_min_radius,
                iters,
                probe_count,
                scale_views,
                safety_steps_used,
                postcheck_border_touch_count,
                scale_target_fill_min,
                scale_target_fill_max,
            )

        scale_ok = _render_scale_views(
            renderer=renderer,
            scene=scene,
            specimen_out_dir=specimen_out_dir,
            sid=sid,
            scale_name=scale_name,
            views=scale_views,
            center=center,
            up=up,
            fov_deg=fov_deg,
            final_radius=final_radius,
            multiscale=multiscale_zoom,
            light_mode=light_mode,
            light_direction=light_direction,
            light_color=light_color,
            light_intensity=light_intensity,
            lighting_enabled=lighting_enabled,
        )
        ok = ok and scale_ok
        zoom_rows.append(
            {
                "specimen": mesh_rel.as_posix(),
                "ok": scale_ok,
                "auto_zoom": auto_zoom,
                "appearance": appearance,
                "light_mode": light_mode,
                "scale": scale_name,
                "scale_view_count": scale_views,
                "preview_fill_min": preview_fill_min,
                "preview_fill_max": preview_fill_max,
                "final_radius": final_radius,
                "postcheck_border_touch_count": postcheck_border_touch_count,
                "safety_steps_used": safety_steps_used,
                "final_radius_after_safety": final_radius,
                "target_fill_min": scale_target_fill_min,
                "target_fill_max": scale_target_fill_max,
            }
        )

    return ok, zoom_rows


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Render multi-view PNG images from 3D meshes/point clouds.")
    parser.add_argument("--in", dest="input_dir", type=Path, required=True, help="Input directory with .ply/.obj/.stl/.off")
    parser.add_argument("--out", dest="output_dir", type=Path, required=True, help="Output directory for rendered PNGs")
    parser.add_argument("--views", type=int, default=12)
    parser.add_argument("--size", type=int, default=512)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--light-mode",
        choices=["camera", "world"],
        default="camera",
        help=(
            "Lighting mode for lit rendering. "
            "camera: update sun light direction for each view to illuminate the camera-facing surface "
            "(camera-following directional light; avoids excessively dark back-side views; intended for stable DINOv2 multi-view input). "
            "world: use the fixed world-space sun light direction for all views "
            "(preserves previous behavior; useful for comparison / compatibility)."
        ),
    )
    parser.add_argument(
        "--appearance",
        choices=["gray_lit", "color_lit"],
        default="gray_lit",
        help=(
            "Rendering appearance. "
            "gray_lit: fixed gray material with fixed lighting for shape-only rendering. "
            "color_lit: use original vertex colors/textures when available with fixed lighting."
        ),
    )
    parser.add_argument("--auto-zoom", action="store_true", help="Automatically tune camera radius per specimen.")
    parser.add_argument("--target-fill-min", type=float, default=0.20, help="Minimum target preview fill ratio.")
    parser.add_argument("--target-fill-max", type=float, default=0.35, help="Maximum target preview fill ratio.")
    parser.add_argument(
        "--auto-zoom-probes",
        type=int,
        default=6,
        help="Number of preview directions used to determine per-specimen auto-zoom radius.",
    )
    parser.add_argument(
        "--auto-zoom-safe-margin",
        type=float,
        default=0.05,
        help="Per-step radius expansion ratio for auto-zoom safety post-check.",
    )
    parser.add_argument(
        "--auto-zoom-max-safety-steps",
        type=int,
        default=5,
        help="Maximum safety post-check adjustment iterations for auto-zoom.",
    )
    parser.add_argument(
        "--multiscale-zoom",
        action="store_true",
        help=(
            "Render two auto-zoom scales per specimen: loose and up. "
            "When enabled, --views must be divisible by the number of scales, "
            "and each scale gets views / num_scales views."
        ),
    )
    parser.add_argument(
        "--loose-fill-min",
        type=float,
        default=0.35,
        help="Minimum target bbox fill ratio for the loose scale when --multiscale-zoom is enabled.",
    )
    parser.add_argument(
        "--loose-fill-max",
        type=float,
        default=0.55,
        help="Maximum target bbox fill ratio for the loose scale when --multiscale-zoom is enabled.",
    )
    parser.add_argument(
        "--up-fill-min",
        type=float,
        default=0.65,
        help="Minimum target bbox fill ratio for the up scale when --multiscale-zoom is enabled.",
    )
    parser.add_argument(
        "--up-fill-max",
        type=float,
        default=0.85,
        help="Maximum target bbox fill ratio for the up scale when --multiscale-zoom is enabled.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    setup_logging()
    set_seed(args.seed)
    if not 0 < args.target_fill_min < args.target_fill_max < 1:
        raise ValueError("--target-fill-min / --target-fill-max must satisfy 0 < min < max < 1")
    if args.multiscale_zoom:
        if not args.auto_zoom:
            raise ValueError("--multiscale-zoom requires --auto-zoom")
        if args.views % 2 != 0:
            raise ValueError("--multiscale-zoom requires --views to be divisible by 2")
        if not 0 < args.loose_fill_min < args.loose_fill_max < 1:
            raise ValueError("--loose-fill-min / --loose-fill-max must satisfy 0 < min < max < 1")
        if not 0 < args.up_fill_min < args.up_fill_max < 1:
            raise ValueError("--up-fill-min / --up-fill-max must satisfy 0 < min < max < 1")
        if not args.loose_fill_max < args.up_fill_min:
            raise ValueError("--multiscale-zoom requires --loose-fill-max < --up-fill-min")
    if args.auto_zoom_probes < 1:
        raise ValueError("--auto-zoom-probes must be >= 1")
    if args.auto_zoom_safe_margin < 0:
        raise ValueError("--auto-zoom-safe-margin must be >= 0")
    if args.auto_zoom_max_safety_steps < 0:
        raise ValueError("--auto-zoom-max-safety-steps must be >= 0")

    ensure_dir(args.output_dir)
    LOGGER.info("Rendering appearance: %s", args.appearance)
    LOGGER.info("Rendering light mode: %s", args.light_mode)
    mesh_files = list_mesh_files(args.input_dir)
    if not mesh_files:
        LOGGER.warning("No mesh files found in %s", args.input_dir)
        return

    renderer = o3d.visualization.rendering.OffscreenRenderer(args.size, args.size)
    renderer.scene.set_background([1.0, 1.0, 1.0, 1.0])
    zoom_report_path = args.output_dir / "auto_zoom_report.csv"
    zoom_rows: list[dict[str, str | float | bool | int]] = []

    success = 0
    for mesh_path in tqdm(mesh_files, desc="Rendering"):
        ok, specimen_zoom_rows = render_specimen(
            renderer=renderer,
            mesh_path=mesh_path,
            input_root=args.input_dir,
            out_dir=args.output_dir,
            views=args.views,
            size=args.size,
            light_direction=(0.577, -0.577, -0.577),
            light_color=(1.0, 1.0, 1.0),
            light_intensity=50000,
            light_mode=args.light_mode,
            appearance=args.appearance,
            auto_zoom=args.auto_zoom,
            target_fill_min=args.target_fill_min,
            target_fill_max=args.target_fill_max,
            multiscale_zoom=args.multiscale_zoom,
            loose_fill_min=args.loose_fill_min,
            loose_fill_max=args.loose_fill_max,
            up_fill_min=args.up_fill_min,
            up_fill_max=args.up_fill_max,
            auto_zoom_probes=args.auto_zoom_probes,
            auto_zoom_safe_margin=args.auto_zoom_safe_margin,
            auto_zoom_max_safety_steps=args.auto_zoom_max_safety_steps,
        )
        zoom_rows.extend(specimen_zoom_rows)
        if ok:
            success += 1

    with zoom_report_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "specimen",
                "ok",
                "auto_zoom",
                "appearance",
                "light_mode",
                "scale",
                "scale_view_count",
                "preview_fill_min",
                "preview_fill_max",
                "final_radius",
                "postcheck_border_touch_count",
                "safety_steps_used",
                "final_radius_after_safety",
                "target_fill_min",
                "target_fill_max",
            ],
        )
        writer.writeheader()
        writer.writerows(zoom_rows)
    LOGGER.info("Auto-zoom report written: %s", zoom_report_path)
    if args.multiscale_zoom and args.auto_zoom:
        radii_by_specimen: dict[str, dict[str, float]] = {}
        for row in zoom_rows:
            specimen = str(row["specimen"])
            scale = str(row["scale"])
            final_radius = float(row["final_radius"])
            radii_by_specimen.setdefault(specimen, {})[scale] = final_radius

        same_radius_specimens = []
        different_radius_count = 0
        for specimen, scales in radii_by_specimen.items():
            if "loose" in scales and "up" in scales:
                if np.isclose(scales["loose"], scales["up"]):
                    same_radius_specimens.append(specimen)
                else:
                    different_radius_count += 1

        LOGGER.info(
            "Multiscale radius check: loose/up different for %d specimens, equal for %d specimens.",
            different_radius_count,
            len(same_radius_specimens),
        )
        if same_radius_specimens:
            LOGGER.warning(
                "loose/up final_radius are equal for specimens: %s",
                ", ".join(same_radius_specimens[:20]),
            )
    LOGGER.info("Rendered %d/%d specimens", success, len(mesh_files))


if __name__ == "__main__":
    main()
