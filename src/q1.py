"""Question 1 kinematics, full-cylinder shadow test, and validation outputs."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.optimize import brentq, minimize_scalar

from mmkit.artifacts import configure_plotting, prepare_output_paths, save_figure, save_table

MISSILE_INITIAL = np.array([20000.0, 0.0, 2000.0])
UAV_INITIAL = np.array([17800.0, 0.0, 1800.0])
TARGET_CENTER = np.array([0.0, 200.0, 5.0])
MISSILE_SPEED = 300.0
UAV_SPEED = 120.0
RELEASE_TIME = 1.5
FUSE_DELAY = 3.6
EXPLOSION_TIME = RELEASE_TIME + FUSE_DELAY
CLOUD_RADIUS = 10.0
CLOUD_DESCENT_SPEED = 3.0
CLOUD_LIFETIME = 20.0
TARGET_RADIUS = 7.0
TARGET_HEIGHT = 10.0


@dataclass(frozen=True, slots=True)
class OcclusionInterval:
    """One continuous effective-occlusion interval."""

    start: float
    end: float

    @property
    def duration(self) -> float:
        """Return interval length in seconds."""
        return self.end - self.start


@dataclass(frozen=True, slots=True)
class Q1Result:
    """Question 1 geometry and both requested occlusion interpretations."""

    gravity: float
    release_point: tuple[float, float, float]
    explosion_point: tuple[float, float, float]
    strict_intervals: tuple[OcclusionInterval, ...]
    center_intervals: tuple[OcclusionInterval, ...]

    @property
    def strict_duration(self) -> float:
        """Total full-cylinder occlusion time."""
        return sum(interval.duration for interval in self.strict_intervals)

    @property
    def center_duration(self) -> float:
        """Total center-line approximation time."""
        return sum(interval.duration for interval in self.center_intervals)


def missile_position(time: float) -> np.ndarray:
    """Return M1 position at time measured from radar detection."""
    direction = -MISSILE_INITIAL / np.linalg.norm(MISSILE_INITIAL)
    return MISSILE_INITIAL + MISSILE_SPEED * time * direction


def release_point() -> np.ndarray:
    """Return the fixed Question 1 release point."""
    return UAV_INITIAL + np.array([-UAV_SPEED * RELEASE_TIME, 0.0, 0.0])


def explosion_point(gravity: float) -> np.ndarray:
    """Return the fixed Question 1 explosion point for a declared gravity."""
    return release_point() + np.array(
        [-UAV_SPEED * FUSE_DELAY, 0.0, -0.5 * gravity * FUSE_DELAY**2]
    )


def cloud_position(time: float, gravity: float) -> np.ndarray:
    """Return the cloud center during its effective lifetime."""
    if not EXPLOSION_TIME <= time <= EXPLOSION_TIME + CLOUD_LIFETIME:
        raise ValueError("time is outside the effective cloud lifetime")
    return explosion_point(gravity) + np.array(
        [0.0, 0.0, -CLOUD_DESCENT_SPEED * (time - EXPLOSION_TIME)]
    )


def _cone_margin(points: np.ndarray, missile: np.ndarray, cloud: np.ndarray) -> np.ndarray:
    """Return signed tangent-cone margins; non-positive points lie in the shadow cone."""
    cloud_vector = cloud - missile
    cloud_distance = float(np.linalg.norm(cloud_vector))
    if cloud_distance <= CLOUD_RADIUS:
        return np.full(len(points), cloud_distance - CLOUD_RADIUS)
    axis = cloud_vector / cloud_distance
    point_vectors = np.asarray(points, dtype=float) - missile
    axial = point_vectors @ axis
    perpendicular = point_vectors - np.outer(axial, axis)
    radial = np.linalg.norm(perpendicular, axis=1)
    tangent = CLOUD_RADIUS / np.sqrt(cloud_distance**2 - CLOUD_RADIUS**2)
    return radial - tangent * axial


def _minimum_target_axial(missile: np.ndarray, cloud: np.ndarray) -> float:
    """Return the exact minimum axial coordinate over the solid target cylinder."""
    axis = (cloud - missile) / np.linalg.norm(cloud - missile)
    centered = float((TARGET_CENTER - missile) @ axis)
    horizontal_support = TARGET_RADIUS * float(np.hypot(axis[0], axis[1]))
    vertical_support = 0.5 * TARGET_HEIGHT * abs(float(axis[2]))
    return centered - horizontal_support - vertical_support


def _rim_point(angle: float, height: float) -> np.ndarray:
    return np.array(
        [
            TARGET_RADIUS * np.cos(angle),
            200.0 + TARGET_RADIUS * np.sin(angle),
            height,
        ]
    )


def _maximum_rim_margin(
    missile: np.ndarray,
    cloud: np.ndarray,
    *,
    rim_samples: int,
) -> tuple[float, np.ndarray]:
    """Maximize cone margin over the two extreme circular rims of the cylinder."""
    if rim_samples < 32:
        raise ValueError("rim_samples must be at least 32")
    angles = np.linspace(0.0, 2.0 * np.pi, rim_samples, endpoint=False)
    step = 2.0 * np.pi / rim_samples
    best_margin = -np.inf
    best_point = np.zeros(3)

    for height in (0.0, TARGET_HEIGHT):
        points = np.column_stack(
            (
                TARGET_RADIUS * np.cos(angles),
                200.0 + TARGET_RADIUS * np.sin(angles),
                np.full_like(angles, height),
            )
        )
        margins = _cone_margin(points, missile, cloud)
        candidate_indices = np.argpartition(margins, -min(4, rim_samples))[-min(4, rim_samples) :]
        for index in candidate_indices:
            center_angle = float(angles[index])

            def negative_margin(angle: float, fixed_height: float = height) -> float:
                point = _rim_point(angle, fixed_height)
                return -float(_cone_margin(point[None, :], missile, cloud)[0])

            optimized = minimize_scalar(
                negative_margin,
                bounds=(center_angle - step, center_angle + step),
                method="bounded",
                options={"xatol": 1e-13},
            )
            margin = -float(optimized.fun)
            if margin > best_margin:
                best_margin = margin
                best_point = _rim_point(float(optimized.x), height)
    return best_margin, best_point


def strict_shadow_margin(time: float, gravity: float, *, rim_samples: int) -> float:
    """Return full-cylinder shadow margin; the target is occluded iff it is non-positive."""
    missile = missile_position(time)
    cloud = cloud_position(time, gravity)
    cloud_distance = float(np.linalg.norm(cloud - missile))
    if cloud_distance <= CLOUD_RADIUS:
        return cloud_distance - CLOUD_RADIUS
    cone_margin, _ = _maximum_rim_margin(missile, cloud, rim_samples=rim_samples)
    tangent_plane_distance = np.sqrt(cloud_distance**2 - CLOUD_RADIUS**2)
    behind_margin = tangent_plane_distance - _minimum_target_axial(missile, cloud)
    return max(cone_margin, behind_margin)


def center_shadow_margin(time: float, gravity: float) -> float:
    """Return center-point shadow margin; non-positive means its sightline is occluded."""
    missile = missile_position(time)
    cloud = cloud_position(time, gravity)
    cloud_distance = float(np.linalg.norm(cloud - missile))
    if cloud_distance <= CLOUD_RADIUS:
        return cloud_distance - CLOUD_RADIUS
    cone_margin = float(_cone_margin(TARGET_CENTER[None, :], missile, cloud)[0])
    axis = (cloud - missile) / cloud_distance
    target_axial = float((TARGET_CENTER - missile) @ axis)
    behind_margin = np.sqrt(cloud_distance**2 - CLOUD_RADIUS**2) - target_axial
    return max(cone_margin, behind_margin)


def _dense_target_surface(angle_count: int, level_count: int) -> np.ndarray:
    """Sample the complete cylinder surface for an independent segment-distance check."""
    if angle_count < 32 or level_count < 3:
        raise ValueError("surface validation requires at least 32 angles and 3 levels")
    angles = np.linspace(0.0, 2.0 * np.pi, angle_count, endpoint=False)
    heights = np.linspace(0.0, TARGET_HEIGHT, level_count)
    angle_grid, height_grid = np.meshgrid(angles, heights, indexing="ij")
    side = np.column_stack(
        (
            TARGET_RADIUS * np.cos(angle_grid).ravel(),
            200.0 + TARGET_RADIUS * np.sin(angle_grid).ravel(),
            height_grid.ravel(),
        )
    )

    radii = np.linspace(0.0, TARGET_RADIUS, level_count)
    angle_grid, radius_grid = np.meshgrid(angles, radii, indexing="ij")
    cap_x = radius_grid.ravel() * np.cos(angle_grid).ravel()
    cap_y = 200.0 + radius_grid.ravel() * np.sin(angle_grid).ravel()
    bottom = np.column_stack((cap_x, cap_y, np.zeros_like(cap_x)))
    top = np.column_stack((cap_x, cap_y, np.full_like(cap_x, TARGET_HEIGHT)))
    return np.vstack((side, bottom, top))


def _maximum_segment_distance(
    points: np.ndarray,
    missile: np.ndarray,
    cloud: np.ndarray,
) -> tuple[float, np.ndarray]:
    """Independently maximize distance from the cloud center to sightline segments."""
    directions = points - missile
    squared_lengths = np.einsum("ij,ij->i", directions, directions)
    parameters = ((cloud - missile) @ directions.T) / squared_lengths
    parameters = np.clip(parameters, 0.0, 1.0)
    closest = missile + parameters[:, None] * directions
    distances = np.linalg.norm(closest - cloud, axis=1)
    index = int(np.argmax(distances))
    return float(distances[index]), points[index]


def _closest_missile_cloud_approach(gravity: float) -> tuple[float, float]:
    """Return closest approach time and distance during the cloud lifetime."""

    def squared_distance(time: float) -> float:
        difference = missile_position(time) - cloud_position(time, gravity)
        return float(difference @ difference)

    optimized = minimize_scalar(
        squared_distance,
        bounds=(EXPLOSION_TIME, EXPLOSION_TIME + CLOUD_LIFETIME),
        method="bounded",
        options={"xatol": 1e-13},
    )
    return float(optimized.x), float(np.sqrt(optimized.fun))


def _find_intervals(
    margin_function: Any,
    *,
    start: float,
    end: float,
    scan_step: float,
    root_tolerance: float,
) -> tuple[OcclusionInterval, ...]:
    """Locate non-positive intervals using a scan only for brackets and Brent roots for edges."""
    count = int(np.ceil((end - start) / scan_step))
    times = np.linspace(start, end, count + 1)
    margins = np.array([margin_function(float(time)) for time in times])
    inside = margins <= 0.0
    boundaries: list[tuple[float, bool]] = []
    for index in np.flatnonzero(inside[1:] != inside[:-1]):
        left = float(times[index])
        right = float(times[index + 1])
        root = float(brentq(margin_function, left, right, xtol=root_tolerance, rtol=1e-14))
        boundaries.append((root, bool(inside[index + 1])))

    intervals: list[OcclusionInterval] = []
    current_start = start if inside[0] else None
    for boundary, enters in boundaries:
        if enters:
            current_start = boundary
        elif current_start is not None:
            intervals.append(OcclusionInterval(current_start, boundary))
            current_start = None
    if current_start is not None:
        intervals.append(OcclusionInterval(current_start, end))
    return tuple(intervals)


def solve_question_1(config: dict[str, Any], gravity: float) -> Q1Result:
    """Solve Question 1 for one declared gravity value."""
    start = EXPLOSION_TIME
    end = EXPLOSION_TIME + CLOUD_LIFETIME
    rim_samples = int(config["rim_samples"])
    strict_intervals = _find_intervals(
        lambda time: strict_shadow_margin(time, gravity, rim_samples=rim_samples),
        start=start,
        end=end,
        scan_step=float(config["time_scan_step"]),
        root_tolerance=float(config["root_tolerance"]),
    )
    center_intervals = _find_intervals(
        lambda time: center_shadow_margin(time, gravity),
        start=start,
        end=end,
        scan_step=float(config["time_scan_step"]),
        root_tolerance=float(config["root_tolerance"]),
    )
    return Q1Result(
        gravity=gravity,
        release_point=tuple(float(value) for value in release_point()),
        explosion_point=tuple(float(value) for value in explosion_point(gravity)),
        strict_intervals=strict_intervals,
        center_intervals=center_intervals,
    )


def _interval_rows(result: Q1Result, label: str) -> list[dict[str, float | str]]:
    rows: list[dict[str, float | str]] = []
    for criterion, intervals in (
        ("strict-full-cylinder", result.strict_intervals),
        ("center-line", result.center_intervals),
    ):
        for index, interval in enumerate(intervals, start=1):
            rows.append(
                {
                    "scenario": label,
                    "gravity_m_s2": result.gravity,
                    "criterion": criterion,
                    "interval": index,
                    "start_s": interval.start,
                    "end_s": interval.end,
                    "duration_s": interval.duration,
                }
            )
    return rows


def _validation_record(config: dict[str, Any], result: Q1Result) -> dict[str, Any]:
    convergence: list[dict[str, float | int]] = []
    gravity = result.gravity
    for rim_samples in config["validation_rim_samples"]:
        intervals = _find_intervals(
            lambda time, samples=int(rim_samples): strict_shadow_margin(
                time, gravity, rim_samples=samples
            ),
            start=EXPLOSION_TIME,
            end=EXPLOSION_TIME + CLOUD_LIFETIME,
            scan_step=float(config["time_scan_step"]),
            root_tolerance=float(config["root_tolerance"]),
        )
        convergence.append(
            {
                "rim_samples": int(rim_samples),
                "duration_s": sum(interval.duration for interval in intervals),
                "interval_count": len(intervals),
            }
        )

    time_scan_convergence: list[dict[str, float | int]] = []
    for scan_step in config["validation_time_steps"]:
        intervals = _find_intervals(
            lambda time: strict_shadow_margin(
                time, gravity, rim_samples=int(config["rim_samples"])
            ),
            start=EXPLOSION_TIME,
            end=EXPLOSION_TIME + CLOUD_LIFETIME,
            scan_step=float(scan_step),
            root_tolerance=float(config["root_tolerance"]),
        )
        time_scan_convergence.append(
            {
                "scan_step_s": float(scan_step),
                "duration_s": sum(interval.duration for interval in intervals),
                "interval_count": len(intervals),
            }
        )

    boundary_checks: list[dict[str, float | bool | str]] = []
    boundary_tolerance = float(config["boundary_margin_tolerance"])
    for criterion, intervals, margin_function in (
        (
            "strict-full-cylinder",
            result.strict_intervals,
            lambda time: strict_shadow_margin(
                time, gravity, rim_samples=int(config["rim_samples"])
            ),
        ),
        ("center-line", result.center_intervals, lambda time: center_shadow_margin(time, gravity)),
    ):
        for interval in intervals:
            midpoint = 0.5 * (interval.start + interval.end)
            probe = float(config["boundary_probe_seconds"])
            for location, time, expected_state in (
                ("before-start", interval.start - probe, "outside"),
                ("start", interval.start, "boundary"),
                ("midpoint", midpoint, "inside"),
                ("end", interval.end, "boundary"),
                ("after-end", interval.end + probe, "outside"),
            ):
                margin = margin_function(time)
                if expected_state == "boundary":
                    check_passed = abs(margin) <= boundary_tolerance
                elif expected_state == "inside":
                    check_passed = margin < -boundary_tolerance
                else:
                    check_passed = margin > boundary_tolerance
                boundary_checks.append(
                    {
                        "criterion": criterion,
                        "location": location,
                        "time_s": time,
                        "margin_m": margin,
                        "expected_state": expected_state,
                        "check_passed": check_passed,
                    }
                )

    surface_points = _dense_target_surface(
        int(config["surface_validation_angles"]),
        int(config["surface_validation_levels"]),
    )
    surface_checks: list[dict[str, Any]] = []
    for interval in result.strict_intervals:
        midpoint = 0.5 * (interval.start + interval.end)
        probe = float(config["boundary_probe_seconds"])
        for location, time in (
            ("before-start", interval.start - probe),
            ("start", interval.start),
            ("midpoint", midpoint),
            ("end", interval.end),
            ("after-end", interval.end + probe),
        ):
            missile = missile_position(time)
            cloud = cloud_position(time, gravity)
            maximum_distance, worst_point = _maximum_segment_distance(
                surface_points, missile, cloud
            )
            surface_checks.append(
                {
                    "location": location,
                    "time_s": time,
                    "sampled_surface_points": int(len(surface_points)),
                    "maximum_segment_distance_m": maximum_distance,
                    "distance_margin_m": maximum_distance - CLOUD_RADIUS,
                    "occluded": maximum_distance <= CLOUD_RADIUS,
                    "worst_sampled_target_point": worst_point.tolist(),
                }
            )

    closest_time, closest_distance = _closest_missile_cloud_approach(gravity)
    return {
        "model": "q1-shadow-cone-v1",
        "criterion": "full target cylinder contained in the missile-view shadow cone",
        "target_extreme_set": "top and bottom circular rims",
        "gravity_m_s2": gravity,
        "time_scan_step_s": float(config["time_scan_step"]),
        "root_tolerance_s": float(config["root_tolerance"]),
        "boundary_margin_tolerance_m": boundary_tolerance,
        "rim_convergence": convergence,
        "time_scan_convergence": time_scan_convergence,
        "boundary_checks": boundary_checks,
        "independent_surface_segment_checks": surface_checks,
        "derived_checks": {
            "missile_hit_time_s": float(np.linalg.norm(MISSILE_INITIAL) / MISSILE_SPEED),
            "release_time_s": RELEASE_TIME,
            "explosion_time_s": EXPLOSION_TIME,
            "cloud_effective_end_s": EXPLOSION_TIME + CLOUD_LIFETIME,
            "explosion_above_ground": bool(result.explosion_point[2] > 0.0),
            "closest_missile_cloud_time_s": closest_time,
            "closest_missile_cloud_distance_m": closest_distance,
            "missile_enters_cloud": closest_distance < CLOUD_RADIUS,
        },
    }


def run_question_1(project_root: Path, config: dict[str, Any]) -> list[Path]:
    """Run Question 1 and write tables, plot, and a validation record."""
    paths = prepare_output_paths(project_root / "outputs")
    primary = solve_question_1(config, float(config["gravity"]))
    sensitivity = solve_question_1(config, float(config["gravity_sensitivity"]))

    summary = pd.DataFrame(
        [
            {
                "scenario": "primary",
                "gravity_m_s2": primary.gravity,
                "release_x_m": primary.release_point[0],
                "release_y_m": primary.release_point[1],
                "release_z_m": primary.release_point[2],
                "explosion_x_m": primary.explosion_point[0],
                "explosion_y_m": primary.explosion_point[1],
                "explosion_z_m": primary.explosion_point[2],
                "strict_duration_s": primary.strict_duration,
                "center_duration_s": primary.center_duration,
            },
            {
                "scenario": "gravity-sensitivity",
                "gravity_m_s2": sensitivity.gravity,
                "release_x_m": sensitivity.release_point[0],
                "release_y_m": sensitivity.release_point[1],
                "release_z_m": sensitivity.release_point[2],
                "explosion_x_m": sensitivity.explosion_point[0],
                "explosion_y_m": sensitivity.explosion_point[1],
                "explosion_z_m": sensitivity.explosion_point[2],
                "strict_duration_s": sensitivity.strict_duration,
                "center_duration_s": sensitivity.center_duration,
            },
        ]
    )
    intervals = pd.DataFrame(
        _interval_rows(primary, "primary") + _interval_rows(sensitivity, "gravity-sensitivity")
    )
    summary_path = save_table(summary, paths.tables / "q1_summary.csv", overwrite=True)
    intervals_path = save_table(intervals, paths.tables / "q1_intervals.csv", overwrite=True)

    times = np.linspace(
        EXPLOSION_TIME,
        EXPLOSION_TIME + CLOUD_LIFETIME,
        int(config["plot_points"]),
    )
    strict_margins = np.array(
        [
            strict_shadow_margin(
                float(time), primary.gravity, rim_samples=int(config["rim_samples"])
            )
            for time in times
        ]
    )
    center_margins = np.array(
        [center_shadow_margin(float(time), primary.gravity) for time in times]
    )
    configure_plotting()
    figure, axis = plt.subplots(figsize=(8.5, 4.8))
    axis.plot(times, strict_margins, label="Full cylinder", linewidth=2)
    axis.plot(times, center_margins, label="Center-line approximation", linestyle="--")
    axis.axhline(0.0, color="black", linewidth=1)
    axis.set_xlabel("Time since task assignment (s)")
    axis.set_ylabel("Shadow-cone margin (m)")
    axis.set_title("Question 1 occlusion margins")
    axis.legend()
    figure_path = save_figure(figure, paths.figures / "q1_occlusion_margin.svg", overwrite=True)
    plt.close(figure)

    validation = _validation_record(config, primary)
    validation["primary_result"] = {
        **asdict(primary),
        "strict_duration": primary.strict_duration,
        "center_duration": primary.center_duration,
    }
    validation["gravity_sensitivity_result"] = {
        **asdict(sensitivity),
        "strict_duration": sensitivity.strict_duration,
        "center_duration": sensitivity.center_duration,
    }
    validation_path = paths.logs / "q1_validation.json"
    validation_path.write_text(
        json.dumps(validation, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    return [summary_path, intervals_path, figure_path, validation_path]
