from __future__ import annotations

import argparse
import copy
import csv
import logging
import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import open3d as o3d
from tqdm import tqdm

from src.utils.geometry import fibonacci_sphere_points, get_points, load_geometry, normalize_geometry
from src.utils.io import ensure_dir, list_mesh_files, set_seed, setup_logging

LOGGER = logging.getLogger(__name__)

# Auto-zoom の占有率が縮退しているかを判定する閾値。
# 正常な標本では占有率は target(既定 0.20-0.35)付近に収まるため、
# fill_max<=EMPTY_FILL_EPS(空フレーム=何も写らない)や
# fill_min>=FULL_FILL_EPS(全面占有=カメラめり込み)は、
# OffscreenRenderer の累積的な状態破壊(リソースリーク)の兆候とみなす。
EMPTY_FILL_EPS = 0.02
FULL_FILL_EPS = 0.98

# auto_zoom_report.csv の列順(ワーカーの部分CSVと結合後CSVで共有する)。
CSV_FIELDNAMES = [
    "specimen",
    "ok",
    "auto_zoom",
    "appearance",
    "light_mode",
    "scale",
    "scale_view_count",
    "preview_fill_min",
    "preview_fill_max",
    "degenerate",
    "final_radius",
    "postcheck_border_touch_count",
    "safety_steps_used",
    "final_radius_after_safety",
    "target_fill_min",
    "target_fill_max",
]


def _is_degenerate_fill(
    fill_min: float,
    fill_max: float,
    empty_eps: float = EMPTY_FILL_EPS,
    full_eps: float = FULL_FILL_EPS,
) -> bool:
    """auto-zoom の占有率が縮退(空フレーム/全面めり込み)しているか判定する。

    fill_max <= empty_eps は深度プレビューに何も写っていない空フレーム、
    fill_min >= full_eps は画面全体が手前ジオメトリで埋まるカメラめり込みで、
    いずれも OffscreenRenderer の累積的な状態破壊の兆候。
    NaN(auto_zoom 無効で占有率未測定)は判定不能として False を返す。
    """
    if not np.isfinite(fill_min) or not np.isfinite(fill_max):
        return False
    if fill_max <= empty_eps:
        return True
    if fill_min >= full_eps:
        return True
    return False


def _compute_camera_light_direction(eye: np.ndarray) -> np.ndarray:
    """Return camera-following sun-light direction for stable DINOv3 multi-view renders.

    camera mode follows each view to avoid excessively dark back-side views.
    world mode keeps fixed world-space lighting for comparison/compatibility.
    """
    eye_np = np.asarray(eye, dtype=np.float32)
    norm = float(np.linalg.norm(eye_np))
    if norm <= 0:
        raise ValueError("Camera eye vector has zero norm; cannot compute camera light direction.")
    eye_dir = eye_np / norm

    # Open3D sun light direction can be visually ambiguous depending on convention.
    # This sign should illuminate the camera-facing surface; flip if test renders look dark.
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
    """Build material for rendering appearance modes.

    gray_lit uses a fixed gray material to suppress specimen color/texture differences
    for shape-only rendering. color_lit keeps original vertex colors/textures when
    available for optional appearance-aware rendering.
    """
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
            # 占有率は深度画像のみで計算する(_compute_bbox_fill_ratio に depth を渡す)。
            # RGB(render_to_image)は深度取得が失敗したときのフォールバックでしか使わない
            # ため、深度が取れる限り描画しない。RGB レンダは陰影計算を伴い深度の約3.6倍
            # 重く、二分探索で大量に繰り返すプレビューの主コストになっていた。深度成功時の
            # 占有率(=最終半径)は従来と完全一致する。RGBレンダ回数が減ることで本番出力に
            # Filament自動露出の±2階調シフトが生じるが、DINOv3埋め込みはcos≈0.99999で
            # retrieval評価は不変(実測確認済み)。
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
                preview = renderer.render_to_image()
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
        # 占有率・境界接触は深度画像のみで判定する。RGB は深度取得が失敗したときの
        # フォールバックでしか使わないため、深度が取れる限り描画しない(プレビュー高速化。
        # 本番出力に露出の±2階調シフトのみ生じ評価は不変、詳細は render_fill_stats のコメント参照)。
        try:
            depth = renderer.render_to_depth_image()
            fill_ratio, touches_border = _compute_bbox_fill_and_touches_border(depth_image=depth)
        except Exception:
            if not depth_fallback_logged:
                LOGGER.warning("Depth post-check failed; fallback to RGB thresholding only.", exc_info=True)
                depth_fallback_logged = True
            img = renderer.render_to_image()
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
) -> tuple[bool, list[dict[str, str | float | bool | int]], bool]:
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
    specimen_degenerate = False
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
            # Keep --auto-zoom-probes for backward compatibility; use final
            # render directions to avoid probe/view mismatch misses.
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
        scale_degenerate = bool(auto_zoom and _is_degenerate_fill(preview_fill_min, preview_fill_max))
        if scale_degenerate:
            specimen_degenerate = True
            LOGGER.warning(
                "Degenerate auto-zoom fill for %s scale=%s: fill_min=%.4f fill_max=%.4f "
                "(empty or fully-clipped frame; OffscreenRenderer may be corrupted).",
                mesh_rel.as_posix(),
                scale_name,
                preview_fill_min,
                preview_fill_max,
            )
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
                "degenerate": scale_degenerate,
                "final_radius": final_radius,
                "postcheck_border_touch_count": postcheck_border_touch_count,
                "safety_steps_used": safety_steps_used,
                "final_radius_after_safety": final_radius,
                "target_fill_min": scale_target_fill_min,
                "target_fill_max": scale_target_fill_max,
            }
        )

    return ok, zoom_rows, specimen_degenerate


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Render multi-view PNG images from 3D meshes/point clouds.")
    parser.add_argument("--in", dest="input_dir", type=Path, required=True, help="Input directory with .ply/.obj/.stl/.off")
    parser.add_argument("--out", dest="output_dir", type=Path, required=True, help="Output directory for rendered PNGs")
    parser.add_argument("--views", type=int, default=12)
    parser.add_argument("--size", type=int, default=768)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--subprocess-chunk-size",
        type=int,
        default=100,
        help=(
            "Render specimens in subprocesses of N each. Open3D/Filament cannot recreate an "
            "OffscreenRenderer in-process (it segfaults), so the only reliable fix for the "
            "cumulative renderer-state corruption (empty frames / camera clipping) is to "
            "render each chunk in a fresh subprocess. A chunk that crashes or reports a "
            "degenerate frame is re-rendered one specimen per subprocess for recovery. "
            "0 disables subprocess isolation (single in-process run; original behavior)."
        ),
    )
    parser.add_argument(
        "--jobs",
        type=int,
        default=1,
        help=(
            "並列に起動する子プロセス(チャンク)の数。既定1は従来どおりの逐次起動。"
            "2以上で複数チャンクを同時にレンダリングし、遊休CPUコアを活用して高速化する。"
            "各標本の出力は独立、part CSV は start 連番で結合されるため、同一の "
            "--subprocess-chunk-size なら並列でも出力・auto_zoom_report の行順は不変。"
            "本環境(CPUソフトレンダ llvmpipe)はメモリ帯域律速で jobs=4-8 付近で頭打ち。"
        ),
    )
    # 内部用(オーケストレータが子プロセスへ担当範囲を渡す)。利用者は直接指定しない。
    parser.add_argument("--_worker-start", type=int, default=None, help=argparse.SUPPRESS)
    parser.add_argument("--_worker-count", type=int, default=None, help=argparse.SUPPRESS)
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


def _create_offscreen_renderer(size: int) -> o3d.visualization.rendering.OffscreenRenderer:
    """背景白で初期化した OffscreenRenderer を生成する(main 初回・再生成で共用)。"""
    renderer = o3d.visualization.rendering.OffscreenRenderer(size, size)
    renderer.scene.set_background([1.0, 1.0, 1.0, 1.0])
    return renderer


def _render_specimen_range(
    args: argparse.Namespace,
    mesh_files: list[Path],
    start: int,
    count: int,
) -> tuple[int, int, list[dict[str, str | float | bool | int]], int]:
    """指定範囲の標本を 1 つの OffscreenRenderer で描画する(チャンク内では再生成しない)。

    Open3D(Filament) はプロセス内でのレンダラ再生成が segfault するため、累積破壊への
    対処はチャンク単位のサブプロセス分離に委ねる。縮退フレームは検出して degenerate 列に
    記録するのみで、回復はオーケストレータが担当する。
    戻り値: (成功数, 試行標本数, zoom_rows, 縮退標本数)。
    """
    subset = mesh_files[start : start + count]
    renderer = _create_offscreen_renderer(args.size)
    rows: list[dict[str, str | float | bool | int]] = []
    success = 0
    degenerate_count = 0
    for mesh_path in tqdm(subset, desc=f"Rendering[{start}:{start + len(subset)}]"):
        ok, specimen_rows, degenerate = render_specimen(
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
        rows.extend(specimen_rows)
        if ok:
            success += 1
        if degenerate:
            degenerate_count += 1
    return success, len(subset), rows, degenerate_count


def _write_zoom_csv(path: Path, rows: list[dict[str, str | float | bool | int]]) -> None:
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=CSV_FIELDNAMES)
        writer.writeheader()
        writer.writerows(rows)


def _part_csv_path(output_dir: Path, start: int) -> Path:
    """子プロセス(ワーカー)が書く部分CSVのパス。start 連番で命名し結合順を安定させる。"""
    return output_dir / f"auto_zoom_report.part{start:06d}.csv"


def _part_has_problem(output_dir: Path, start: int) -> bool:
    """部分CSVが無い(=子が異常終了)か、縮退標本を含むなら True を返す。"""
    path = _part_csv_path(output_dir, start)
    if not path.exists():
        return True
    with path.open(encoding="utf-8") as f:
        for row in csv.DictReader(f):
            if str(row.get("degenerate", "")).strip().lower() == "true":
                return True
    return False


def _combine_part_csvs(output_dir: Path) -> list[dict[str, str]]:
    """全 part CSV を start 順に結合して auto_zoom_report.csv を書き、part を削除する。"""
    parts = sorted(output_dir.glob("auto_zoom_report.part*.csv"))
    rows: list[dict[str, str]] = []
    for part in parts:
        with part.open(encoding="utf-8") as f:
            rows.extend(csv.DictReader(f))
    with (output_dir / "auto_zoom_report.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=CSV_FIELDNAMES)
        writer.writeheader()
        writer.writerows(rows)
    for part in parts:
        part.unlink()
    return rows


def _check_multiscale_radii(rows: list[dict]) -> None:
    """multiscale 時に loose/up の半径が分離できているかを確認する(従来挙動を維持)。"""
    radii_by_specimen: dict[str, dict[str, float]] = {}
    for row in rows:
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


def _worker_command(start: int, count: int) -> list[str]:
    """元の CLI 引数を引き継ぎ、担当範囲を指定した子プロセス(ワーカー)コマンドを作る。"""
    return [
        sys.executable,
        "-m",
        "src.render_multiview",
        *sys.argv[1:],
        "--_worker-start",
        str(start),
        "--_worker-count",
        str(count),
    ]


def _build_worker_env(jobs: int) -> dict[str, str] | None:
    """並列実行(jobs>1)時の子プロセス環境変数を作る。jobs<=1 では None(環境を変えない)。

    CPUソフトレンダ(llvmpipe)は1レンダリングを内部で複数スレッド化する。各子の llvmpipe
    スレッド数を「論理コア数 / jobs」に制限し、総スレッド数を物理資源に合わせる
    (LP_NUM_THREADS)。これにより jobs が少なければ各プロセスはマルチスレッドでコアを使い、
    jobs が論理コア数に近づくと1スレッド/プロセスに収束する。LP_NUM_THREADS=1 を固定すると
    低並列で各プロセスが1スレッドに絞られコアが遊ぶため、動的に分配する。スレッド数は出力を
    変えない(バイト不変)。GPUバックエンドでは無視される(無害)。ユーザー明示値は尊重する。
    """
    if jobs <= 1:
        return None
    env = os.environ.copy()
    threads_per_job = max(1, (os.cpu_count() or 1) // jobs)
    env.setdefault("LP_NUM_THREADS", str(threads_per_job))
    # 親が全体進捗を表示するため、子プロセス側の tqdm バーは抑制する。
    env.setdefault("TQDM_DISABLE", "1")
    return env


def _run_chunk_subprocess(
    start: int,
    count: int,
    env: dict[str, str] | None = None,
    log_path: Path | None = None,
) -> int:
    """1 チャンク(または 1 標本)を新しい子プロセスで描画し、終了コードを返す。

    env は子プロセスへ渡す環境変数(並列時の LP_NUM_THREADS 等)。None なら親環境を継承。
    log_path 指定時は子の標準出力/エラーをそのファイルへ送り、並列時にログが混ざるのを防ぐ。
    """
    if log_path is None:
        return subprocess.run(_worker_command(start, count), env=env).returncode
    with log_path.open("w", encoding="utf-8") as logf:
        return subprocess.run(
            _worker_command(start, count), env=env, stdout=subprocess.DEVNULL, stderr=logf
        ).returncode


def _run_worker(args: argparse.Namespace) -> None:
    """子プロセス本体: 担当範囲のみ描画して部分CSVを書く。"""
    mesh_files = list_mesh_files(args.input_dir)
    start = int(args._worker_start)
    count = int(args._worker_count)
    success, total, rows, degenerate_count = _render_specimen_range(args, mesh_files, start, count)
    _write_zoom_csv(_part_csv_path(args.output_dir, start), rows)
    LOGGER.info(
        "Worker [%d:%d] rendered %d/%d specimens (degenerate=%d).",
        start,
        start + total,
        success,
        total,
        degenerate_count,
    )


def _run_chunks_sequential(
    args: argparse.Namespace,
    mesh_files: list[Path],
    starts: list[int],
    chunk: int,
    n: int,
) -> list[int]:
    """逐次オーケストレータ(従来挙動): チャンクを1つずつ子プロセスで描画する。

    子プロセスのログ(auto-zoom trace 等)はそのまま親コンソールへ流す。
    戻り値: 隔離しても回復しなかった標本インデックスの一覧。
    """
    unrecovered: list[int] = []
    for ci, start in enumerate(starts):
        count = min(chunk, n - start)
        LOGGER.info("=== Chunk %d/%d: specimens [%d:%d] ===", ci + 1, len(starts), start, start + count)
        rc = _run_chunk_subprocess(start, count)
        if rc == 0 and not _part_has_problem(args.output_dir, start):
            continue

        # チャンクが crash したか縮退を記録 → 標本単位の新プロセスで再描画して回復させる。
        LOGGER.warning(
            "Chunk [%d:%d] unhealthy (exit=%d or degenerate); re-rendering one specimen per subprocess.",
            start,
            start + count,
            rc,
        )
        if count == 1:
            unrecovered.append(start)
            _log_unrecovered(start, mesh_files[start].as_posix(), rc)
            continue
        stale_part = _part_csv_path(args.output_dir, start)
        if stale_part.exists():
            stale_part.unlink()
        for k in range(count):
            rc1 = _run_chunk_subprocess(start + k, 1)
            if rc1 != 0 or _part_has_problem(args.output_dir, start + k):
                unrecovered.append(start + k)
                _log_unrecovered(start + k, mesh_files[start + k].as_posix(), rc1)
    return unrecovered


def _run_chunks_parallel(
    args: argparse.Namespace,
    mesh_files: list[Path],
    starts: list[int],
    chunk: int,
    n: int,
    env: dict[str, str] | None,
) -> list[int]:
    """並列オーケストレータ: チャンクを最大 args.jobs 個まで同時に子プロセスで描画する。

    逐次版 _run_chunks_sequential と同じ回復ロジック(不健全チャンクは標本単位で再描画)を、
    完了したそばから動的に再投入する形で並列化する。チャンク内のレンダリング順序・レンダラ
    状態は逐次版と同一なので、同じ --subprocess-chunk-size なら出力は完全に不変。子プロセス
    のログは output_dir/render_logs/chunk{start}.log に保存する。
    戻り値: 隔離しても回復しなかった標本インデックスの一覧。
    """
    from concurrent.futures import FIRST_COMPLETED, ThreadPoolExecutor, wait

    log_dir = args.output_dir / "render_logs"
    ensure_dir(log_dir)
    unrecovered: list[int] = []
    with ThreadPoolExecutor(max_workers=args.jobs) as executor:
        pending: dict = {}
        for start in starts:
            count = min(chunk, n - start)
            fut = executor.submit(
                _run_chunk_subprocess, start, count, env, log_dir / f"chunk{start:06d}.log"
            )
            pending[fut] = (start, count)

        with tqdm(total=n, desc=f"Rendering(jobs={args.jobs})") as pbar:
            while pending:
                done, _ = wait(list(pending), return_when=FIRST_COMPLETED)
                for fut in done:
                    start, count = pending.pop(fut)
                    rc = fut.result()
                    if rc == 0 and not _part_has_problem(args.output_dir, start):
                        pbar.update(count)
                        continue

                    LOGGER.warning(
                        "Chunk [%d:%d] unhealthy (exit=%d or degenerate); re-rendering one specimen per subprocess.",
                        start,
                        start + count,
                        rc,
                    )
                    if count == 1:
                        unrecovered.append(start)
                        LOGGER.error(
                            "Specimen index=%d (%s) unhealthy in isolation (exit=%d); output may be invalid.",
                            start,
                            mesh_files[start].as_posix(),
                            rc,
                        )
                        pbar.update(1)
                        continue

                    # チャンクをばらして標本単位で再投入(健全標本も含めて描き直す=逐次版と同じ)。
                    stale_part = _part_csv_path(args.output_dir, start)
                    if stale_part.exists():
                        stale_part.unlink()
                    for k in range(count):
                        f2 = executor.submit(
                            _run_chunk_subprocess,
                            start + k,
                            1,
                            env,
                            log_dir / f"chunk{start + k:06d}.log",
                        )
                        pending[f2] = (start + k, 1)
    return unrecovered


def _run_orchestrator(args: argparse.Namespace) -> None:
    """親プロセス: チャンクごとに子プロセスを起動し、異常チャンクは標本単位で再実行する。"""
    ensure_dir(args.output_dir)
    LOGGER.info("Rendering appearance: %s", args.appearance)
    LOGGER.info("Rendering light mode: %s", args.light_mode)
    mesh_files = list_mesh_files(args.input_dir)
    if not mesh_files:
        LOGGER.warning("No mesh files found in %s", args.input_dir)
        return
    n = len(mesh_files)

    # 分離無効: 単一プロセスで全標本を描画する(従来挙動)。
    if args.subprocess_chunk_size <= 0:
        if args.jobs > 1:
            LOGGER.warning(
                "--jobs=%d is ignored because --subprocess-chunk-size 0 runs a single in-process renderer.",
                args.jobs,
            )
        success, total, rows, degenerate_count = _render_specimen_range(args, mesh_files, 0, n)
        _write_zoom_csv(args.output_dir / "auto_zoom_report.csv", rows)
        LOGGER.info("Auto-zoom report written: %s", args.output_dir / "auto_zoom_report.csv")
        if args.multiscale_zoom and args.auto_zoom:
            _check_multiscale_radii(rows)
        LOGGER.info("Rendered %d/%d specimens (degenerate=%d).", success, total, degenerate_count)
        return

    # サブプロセス分離: チャンクごとに新プロセス=まっさらなレンダラ(累積破壊が原理的に起きない)。
    for stale in args.output_dir.glob("auto_zoom_report.part*.csv"):
        stale.unlink()
    chunk = args.subprocess_chunk_size
    starts = list(range(0, n, chunk))
    env = _build_worker_env(args.jobs)
    LOGGER.info(
        "Subprocess isolation: %d specimens in %d chunk(s) of up to %d (jobs=%d%s).",
        n,
        len(starts),
        chunk,
        args.jobs,
        ", LP_NUM_THREADS=1" if env is not None and env.get("LP_NUM_THREADS") == "1" else "",
    )
    if args.jobs > 1:
        # 並列: 複数チャンクを同時起動。チャンク内のレンダリングは逐次版と同一=出力不変。
        unrecovered = _run_chunks_parallel(args, mesh_files, starts, chunk, n, env)
    else:
        # 逐次(従来挙動)。
        unrecovered = _run_chunks_sequential(args, mesh_files, starts, chunk, n)

    rows = _combine_part_csvs(args.output_dir)
    LOGGER.info("Auto-zoom report written: %s (%d rows)", args.output_dir / "auto_zoom_report.csv", len(rows))
    if args.multiscale_zoom and args.auto_zoom:
        _check_multiscale_radii(rows)
    rendered = len({str(r["specimen"]) for r in rows})
    LOGGER.info("Rendered %d/%d specimens via subprocess isolation.", rendered, n)
    if unrecovered:
        LOGGER.error(
            "%d specimen(s) could not be rendered even in isolation: indices %s",
            len(unrecovered),
            unrecovered[:50],
        )


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
    if args.subprocess_chunk_size < 0:
        raise ValueError("--subprocess-chunk-size must be >= 0")
    if args.jobs < 1:
        raise ValueError("--jobs must be >= 1")

    # 子プロセス(ワーカー)は割り当てられた担当範囲だけを描画して終了する。
    if args._worker_start is not None:
        _run_worker(args)
        return
    _run_orchestrator(args)


if __name__ == "__main__":
    main()
