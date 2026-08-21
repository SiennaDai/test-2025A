"""Question 2 optimization using the audited Question 1 occlusion geometry."""

from __future__ import annotations

import json
from collections.abc import Callable
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from scipy.optimize import differential_evolution, minimize

from mmkit.artifacts import prepare_output_paths, save_table

from .q1 import (
    CLOUD_DESCENT_SPEED,
    CLOUD_LIFETIME,
    CLOUD_RADIUS,
    MISSILE_INITIAL,
    MISSILE_SPEED,
    TARGET_CENTER,
    TARGET_HEIGHT,
    TARGET_RADIUS,
    UAV_INITIAL,
    OcclusionInterval,
    _cone_margin,
    _dense_target_surface,
    _find_intervals,
    _maximum_rim_margin,
    _maximum_segment_distance,
    _minimum_target_axial,
    missile_position,
)

MINIMUM_SPEED = 70.0
MAXIMUM_SPEED = 140.0
UAV_ALTITUDE = float(UAV_INITIAL[2])


@dataclass(frozen=True, slots=True)
class Q2Strategy:
    """One feasible FY1 release strategy."""

    theta_rad: float
    speed_m_s: float
    release_time_s: float
    fuse_delay_s: float

    @property
    def explosion_time_s(self) -> float:
        """Return explosion time from task assignment."""
        return self.release_time_s + self.fuse_delay_s


@dataclass(frozen=True, slots=True)
class Q2Evaluation:
    """Formal strict and comparison evaluation for one feasible strategy."""

    strategy: Q2Strategy
    release_point: tuple[float, float, float]
    explosion_point: tuple[float, float, float]
    strict_intervals: tuple[OcclusionInterval, ...]
    center_intervals: tuple[OcclusionInterval, ...]
    minimum_cloud_center_height_m: float
    cloud_center_below_ground: bool
    cloud_center_ground_crossing_time_s: float | None

    @property
    def strict_duration_s(self) -> float:
        """Return union length of strict full-cylinder intervals."""
        return sum(interval.duration for interval in self.strict_intervals)

    @property
    def center_duration_s(self) -> float:
        """Return union length of center-line comparison intervals."""
        return sum(interval.duration for interval in self.center_intervals)


def missile_hit_time() -> float:
    """Return time at which M1 reaches the false target at the origin."""
    return float(np.linalg.norm(MISSILE_INITIAL) / MISSILE_SPEED)


def _heading(theta_rad: float) -> np.ndarray:
    return np.array([np.cos(theta_rad), np.sin(theta_rad), 0.0])


def release_point(strategy: Q2Strategy) -> np.ndarray:
    """Return the strategy's release point."""
    return UAV_INITIAL + strategy.speed_m_s * strategy.release_time_s * _heading(strategy.theta_rad)


def explosion_point(strategy: Q2Strategy, gravity: float) -> np.ndarray:
    """Return the ballistic explosion point."""
    horizontal = strategy.speed_m_s * strategy.explosion_time_s * _heading(strategy.theta_rad)
    return (
        UAV_INITIAL + horizontal + np.array([0.0, 0.0, -0.5 * gravity * strategy.fuse_delay_s**2])
    )


def cloud_position(time: float, strategy: Q2Strategy, gravity: float) -> np.ndarray:
    """Return cloud center during the declared 20-second effective window."""
    if not strategy.explosion_time_s <= time <= strategy.explosion_time_s + CLOUD_LIFETIME:
        raise ValueError("time is outside the effective cloud lifetime")
    return explosion_point(strategy, gravity) + np.array(
        [0.0, 0.0, -CLOUD_DESCENT_SPEED * (time - strategy.explosion_time_s)]
    )


def _strategy_from_array(values: np.ndarray) -> Q2Strategy:
    return Q2Strategy(*(float(value) for value in values))


def _feasibility_violation(strategy: Q2Strategy, gravity: float) -> float:
    """Return a scaled nonnegative violation measure."""
    hit_time = missile_hit_time()
    explosion_height = float(explosion_point(strategy, gravity)[2])
    violation = max(0.0, strategy.explosion_time_s - hit_time) / hit_time
    violation += max(0.0, -explosion_height) / UAV_ALTITUDE
    return violation


def _strict_margin(
    time: float,
    strategy: Q2Strategy,
    gravity: float,
    *,
    rim_samples: int,
    refine_rim: bool,
) -> float:
    """Return the generalized full-cylinder margin for one strategy."""
    missile = missile_position(time)
    cloud = cloud_position(time, strategy, gravity)
    cloud_distance = float(np.linalg.norm(cloud - missile))
    if cloud_distance <= CLOUD_RADIUS:
        return cloud_distance - CLOUD_RADIUS
    if refine_rim:
        cone_margin, _ = _maximum_rim_margin(missile, cloud, rim_samples=rim_samples)
    else:
        angles = np.linspace(0.0, 2.0 * np.pi, rim_samples, endpoint=False)
        x = TARGET_RADIUS * np.cos(angles)
        y = 200.0 + TARGET_RADIUS * np.sin(angles)
        points = np.vstack(
            (
                np.column_stack((x, y, np.zeros_like(x))),
                np.column_stack((x, y, np.full_like(x, TARGET_HEIGHT))),
            )
        )
        cone_margin = float(np.max(_cone_margin(points, missile, cloud)))
    tangent_plane_distance = np.sqrt(cloud_distance**2 - CLOUD_RADIUS**2)
    behind_margin = tangent_plane_distance - _minimum_target_axial(missile, cloud)
    return max(cone_margin, behind_margin)


def _center_margin(time: float, strategy: Q2Strategy, gravity: float) -> float:
    """Return the center-line comparison margin for one strategy."""
    missile = missile_position(time)
    cloud = cloud_position(time, strategy, gravity)
    cloud_distance = float(np.linalg.norm(cloud - missile))
    if cloud_distance <= CLOUD_RADIUS:
        return cloud_distance - CLOUD_RADIUS
    cone_margin = float(_cone_margin(TARGET_CENTER[None, :], missile, cloud)[0])
    axis = (cloud - missile) / cloud_distance
    target_axial = float((TARGET_CENTER - missile) @ axis)
    behind_margin = np.sqrt(cloud_distance**2 - CLOUD_RADIUS**2) - target_axial
    return max(cone_margin, behind_margin)


def _effective_window(strategy: Q2Strategy) -> tuple[float, float]:
    start = strategy.explosion_time_s
    return start, min(start + CLOUD_LIFETIME, missile_hit_time())


def _sampled_metrics(
    strategy: Q2Strategy,
    margin_function: Callable[[float], float],
    *,
    time_step: float,
) -> tuple[float, float]:
    start, end = _effective_window(strategy)
    if end <= start:
        return 0.0, float("inf")
    count = max(1, int(np.ceil((end - start) / time_step)))
    times = np.linspace(start, end, count + 1)
    margins = np.array([margin_function(float(time)) for time in times])
    inside = (margins <= 0.0).astype(float)
    return float(np.trapezoid(inside, times)), float(np.min(margins))


def _objective(
    values: np.ndarray,
    gravity: float,
    *,
    criterion: str,
    time_step: float,
    rim_samples: int = 96,
) -> float:
    strategy = _strategy_from_array(values)
    violation = _feasibility_violation(strategy, gravity)
    if violation > 0.0:
        return 100.0 + 100.0 * violation
    if criterion == "center":

        def margin(time: float) -> float:
            return _center_margin(time, strategy, gravity)
    else:

        def margin(time: float) -> float:
            return _strict_margin(
                time,
                strategy,
                gravity,
                rim_samples=rim_samples,
                refine_rim=False,
            )

    duration, minimum_margin = _sampled_metrics(strategy, margin, time_step=time_step)
    if duration > 0.0:
        return -duration
    return max(minimum_margin, 0.0) / 100.0


def _root_refined_objective(
    values: np.ndarray,
    gravity: float,
    *,
    time_step: float,
    rim_samples: int,
    root_tolerance: float,
) -> float:
    """Return a locally useful duration objective with root-refined boundaries."""
    strategy = _strategy_from_array(values)
    violation = _feasibility_violation(strategy, gravity)
    if violation > 0.0:
        return 100.0 + 100.0 * violation
    start, end = _effective_window(strategy)

    def margin(time: float) -> float:
        return _strict_margin(
            time,
            strategy,
            gravity,
            rim_samples=rim_samples,
            refine_rim=False,
        )

    sampled_duration, minimum_margin = _sampled_metrics(strategy, margin, time_step=time_step)
    if sampled_duration <= 0.0:
        return max(minimum_margin, 0.0) / 100.0
    intervals = _find_intervals(
        margin,
        start=start,
        end=end,
        scan_step=time_step,
        root_tolerance=root_tolerance,
    )
    return -sum(interval.duration for interval in intervals)


def _formal_evaluation(
    strategy: Q2Strategy,
    gravity: float,
    config: dict[str, Any],
) -> Q2Evaluation:
    settings = config["formal_evaluation"]
    start, end = _effective_window(strategy)
    strict_intervals = _find_intervals(
        lambda time: _strict_margin(
            time,
            strategy,
            gravity,
            rim_samples=int(settings["rim_samples"]),
            refine_rim=True,
        ),
        start=start,
        end=end,
        scan_step=float(settings["time_scan_step_s"]),
        root_tolerance=float(settings["root_tolerance_s"]),
    )
    center_intervals = _find_intervals(
        lambda time: _center_margin(time, strategy, gravity),
        start=start,
        end=end,
        scan_step=float(settings["time_scan_step_s"]),
        root_tolerance=float(settings["root_tolerance_s"]),
    )
    release = release_point(strategy)
    explosion = explosion_point(strategy, gravity)
    minimum_height = float(explosion[2] - CLOUD_DESCENT_SPEED * CLOUD_LIFETIME)
    ground_crossing_time = (
        strategy.explosion_time_s + float(explosion[2]) / CLOUD_DESCENT_SPEED
        if minimum_height < 0.0
        else None
    )
    return Q2Evaluation(
        strategy=strategy,
        release_point=tuple(float(value) for value in release),
        explosion_point=tuple(float(value) for value in explosion),
        strict_intervals=strict_intervals,
        center_intervals=center_intervals,
        minimum_cloud_center_height_m=minimum_height,
        cloud_center_below_ground=minimum_height < 0.0,
        cloud_center_ground_crossing_time_s=ground_crossing_time,
    )


def _bounds(config: dict[str, Any]) -> list[tuple[float, float]]:
    configured = config["bounds"]
    return [
        tuple(configured["theta_rad"]),
        tuple(configured["speed_m_s"]),
        tuple(configured["release_time_s"]),
        tuple(configured["fuse_delay_s"]),
    ]


def _initial_population(bounds: list[tuple[float, float]], popsize: int, seed: int) -> np.ndarray:
    """Build a seeded population containing a known positive-duration baseline."""
    rng = np.random.default_rng(seed)
    population_count = max(5, popsize * len(bounds))
    lower = np.array([bound[0] for bound in bounds])
    upper = np.array([bound[1] for bound in bounds])
    population = rng.uniform(lower, upper, size=(population_count, len(bounds)))
    population[0] = np.array([np.pi, 120.0, 1.5, 3.6])
    population[1] = np.array([np.pi, 140.0, 0.0, 3.6])
    population[2] = np.array([np.pi, 100.0, 0.0, 5.0])
    return population


def _deduplicate(candidates: list[np.ndarray]) -> list[np.ndarray]:
    unique: list[np.ndarray] = []
    for candidate in candidates:
        if not any(np.linalg.norm(candidate - existing) < 1e-7 for existing in unique):
            unique.append(np.asarray(candidate, dtype=float))
    return unique


def optimize_question_2(config: dict[str, Any]) -> tuple[list[Q2Evaluation], dict[str, Any]]:
    """Run center-screened and strict derivative-free optimization."""
    gravity = float(config["gravity"])
    seed = int(config["seed"])
    bounds = _bounds(config)
    center_settings = config["center_search"]
    strict_settings = config["strict_search"]
    center_init = _initial_population(bounds, int(center_settings["popsize"]), seed)

    center_result = differential_evolution(
        lambda values: _objective(
            values,
            gravity,
            criterion="center",
            time_step=float(center_settings["time_step_s"]),
        ),
        bounds,
        seed=seed,
        maxiter=int(center_settings["maxiter"]),
        popsize=int(center_settings["popsize"]),
        tol=float(center_settings["tol"]),
        polish=bool(center_settings["polish"]),
        init=center_init,
        workers=1,
    )
    center_population = np.asarray(center_result.population, dtype=float)
    center_order = np.argsort(center_result.population_energies)
    strict_init = center_population[center_order]

    strict_result = differential_evolution(
        lambda values: _objective(
            values,
            gravity,
            criterion="strict",
            time_step=float(strict_settings["time_step_s"]),
            rim_samples=int(strict_settings["rim_samples"]),
        ),
        bounds,
        seed=seed + 1,
        maxiter=int(strict_settings["maxiter"]),
        popsize=int(strict_settings["popsize"]),
        tol=float(strict_settings["tol"]),
        polish=bool(strict_settings["polish"]),
        init=strict_init,
        workers=1,
    )

    boundary_settings = config["boundary_refinement"]

    def boundary_objective(values: np.ndarray) -> float:
        full_values = np.array(
            [
                values[0],
                float(boundary_settings["speed_m_s"]),
                float(boundary_settings["release_time_s"]),
                values[1],
            ]
        )
        return _root_refined_objective(
            full_values,
            gravity,
            time_step=float(boundary_settings["time_step_s"]),
            rim_samples=int(boundary_settings["rim_samples"]),
            root_tolerance=float(config["formal_evaluation"]["root_tolerance_s"]),
        )

    boundary_result = differential_evolution(
        boundary_objective,
        [
            tuple(boundary_settings["theta_rad"]),
            tuple(boundary_settings["fuse_delay_s"]),
        ],
        seed=int(boundary_settings["seed"]),
        maxiter=int(boundary_settings["maxiter"]),
        popsize=int(boundary_settings["popsize"]),
        tol=float(boundary_settings["tol"]),
        polish=True,
        workers=1,
    )
    boundary_candidate = np.array(
        [
            boundary_result.x[0],
            float(boundary_settings["speed_m_s"]),
            float(boundary_settings["release_time_s"]),
            boundary_result.x[1],
        ]
    )
    warm_start = boundary_settings["independent_warm_start"]
    independent_warm_start = np.array(
        [
            float(warm_start["theta_rad"]),
            float(boundary_settings["speed_m_s"]),
            float(boundary_settings["release_time_s"]),
            float(warm_start["fuse_delay_s"]),
        ]
    )

    raw_candidates = [
        center_result.x,
        strict_result.x,
        boundary_candidate,
        independent_warm_start,
    ]
    strict_population = np.asarray(strict_result.population, dtype=float)
    strict_order = np.argsort(strict_result.population_energies)
    retained_population = max(8, int(config["local_refinement"]["starts"]))
    raw_candidates.extend(strict_population[strict_order[:retained_population]])

    local_results = []
    local_settings = config["local_refinement"]
    for start_values in _deduplicate(raw_candidates)[: int(local_settings["starts"])]:
        result = minimize(
            lambda values: _root_refined_objective(
                values,
                gravity,
                time_step=float(strict_settings["time_step_s"]),
                rim_samples=int(strict_settings["rim_samples"]),
                root_tolerance=float(config["formal_evaluation"]["root_tolerance_s"]),
            ),
            start_values,
            method="Nelder-Mead",
            bounds=bounds,
            options={
                "maxfev": int(local_settings["maxfev"]),
                "xatol": float(local_settings["xtol"]),
                "fatol": float(local_settings["ftol"]),
            },
        )
        local_results.append(result)
        raw_candidates.append(result.x)

    feasible_candidates = [
        candidate
        for candidate in _deduplicate(raw_candidates)
        if _feasibility_violation(_strategy_from_array(candidate), gravity) <= 1e-12
    ]
    ranked_coarse = sorted(
        feasible_candidates,
        key=lambda values: _root_refined_objective(
            values,
            gravity,
            time_step=float(strict_settings["time_step_s"]),
            rim_samples=int(strict_settings["rim_samples"]),
            root_tolerance=float(config["formal_evaluation"]["root_tolerance_s"]),
        ),
    )
    formal_count = min(int(config["formal_evaluation"]["candidate_count"]), len(ranked_coarse))
    formal = [
        _formal_evaluation(_strategy_from_array(values), gravity, config)
        for values in ranked_coarse[:formal_count]
    ]
    formal.sort(key=lambda evaluation: evaluation.strict_duration_s, reverse=True)

    solver_record = {
        "seed": seed,
        "variable_order": ["theta_rad", "speed_m_s", "release_time_s", "fuse_delay_s"],
        "bounds": config["bounds"],
        "candidate_initialization": {
            "mechanism": "seeded uniform population with Q1 and two heading baselines injected",
            "reason": "positive-duration strategies occupy a sparse subset of the box bounds",
            "baseline_q1": [float(np.pi), 120.0, 1.5, 3.6],
        },
        "center_differential_evolution": {
            "success": bool(center_result.success),
            "message": str(center_result.message),
            "iterations": int(center_result.nit),
            "evaluations": int(center_result.nfev),
            "best_sampled_duration_s": -float(center_result.fun),
            "settings": center_settings,
        },
        "strict_differential_evolution": {
            "success": bool(strict_result.success),
            "message": str(strict_result.message),
            "iterations": int(strict_result.nit),
            "evaluations": int(strict_result.nfev),
            "best_sampled_duration_s": -float(strict_result.fun),
            "settings": strict_settings,
        },
        "independent_boundary_refinement": {
            "success": bool(boundary_result.success),
            "message": str(boundary_result.message),
            "iterations": int(boundary_result.nit),
            "evaluations": int(boundary_result.nfev),
            "root_refined_duration_s": -float(boundary_result.fun),
            "theta_rad": float(boundary_result.x[0]),
            "fuse_delay_s": float(boundary_result.x[1]),
            "settings": boundary_settings,
        },
        "local_refinements": [
            {
                "success": bool(result.success),
                "message": str(result.message),
                "evaluations": int(result.nfev),
                "feasible": _feasibility_violation(_strategy_from_array(result.x), gravity)
                <= 1e-12,
                "root_refined_duration_s": (
                    -float(result.fun) if float(result.fun) <= 0.0 else None
                ),
            }
            for result in local_results
        ],
        "formal_evaluation": config["formal_evaluation"],
        "feasible_candidate_count": len(feasible_candidates),
        "formal_candidate_count": len(formal),
        "convergence_claimed": bool(center_result.success and strict_result.success),
        "status": (
            "converged"
            if center_result.success and strict_result.success
            else "budget-exhausted-with-feasible-candidates"
        ),
    }
    return formal, solver_record


def _summary_row(evaluation: Q2Evaluation, rank: int) -> dict[str, Any]:
    strategy = evaluation.strategy
    return {
        "rank": rank,
        "theta_rad": strategy.theta_rad,
        "theta_deg": float(np.degrees(strategy.theta_rad) % 360.0),
        "speed_m_s": strategy.speed_m_s,
        "release_time_s": strategy.release_time_s,
        "release_x_m": evaluation.release_point[0],
        "release_y_m": evaluation.release_point[1],
        "release_z_m": evaluation.release_point[2],
        "fuse_delay_s": strategy.fuse_delay_s,
        "explosion_time_s": strategy.explosion_time_s,
        "explosion_x_m": evaluation.explosion_point[0],
        "explosion_y_m": evaluation.explosion_point[1],
        "explosion_z_m": evaluation.explosion_point[2],
        "strict_duration_s": evaluation.strict_duration_s,
        "center_duration_s": evaluation.center_duration_s,
        "minimum_cloud_center_height_m": evaluation.minimum_cloud_center_height_m,
        "cloud_center_below_ground": evaluation.cloud_center_below_ground,
        "cloud_center_ground_crossing_time_s": (evaluation.cloud_center_ground_crossing_time_s),
    }


def _strict_intervals_with_settings(
    strategy: Q2Strategy,
    gravity: float,
    *,
    time_scan_step: float,
    rim_samples: int,
    root_tolerance: float,
) -> tuple[OcclusionInterval, ...]:
    start, end = _effective_window(strategy)
    return _find_intervals(
        lambda time: _strict_margin(
            time,
            strategy,
            gravity,
            rim_samples=rim_samples,
            refine_rim=True,
        ),
        start=start,
        end=end,
        scan_step=time_scan_step,
        root_tolerance=root_tolerance,
    )


def _q2_validation_record(best: Q2Evaluation, config: dict[str, Any]) -> dict[str, Any]:
    """Build numerical, boundary, independent-geometry, and perturbation checks."""
    gravity = float(config["gravity"])
    validation = config["validation"]
    root_tolerance = float(config["formal_evaluation"]["root_tolerance_s"])
    convergence = []
    for settings in validation["convergence_settings"]:
        intervals = _strict_intervals_with_settings(
            best.strategy,
            gravity,
            time_scan_step=float(settings["time_scan_step_s"]),
            rim_samples=int(settings["rim_samples"]),
            root_tolerance=root_tolerance,
        )
        convergence.append(
            {
                **settings,
                "interval_count": len(intervals),
                "strict_duration_s": sum(interval.duration for interval in intervals),
            }
        )

    boundary_checks = []
    probe = float(validation["boundary_probe_s"])
    for interval in best.strict_intervals:
        midpoint = 0.5 * (interval.start + interval.end)
        explosion_time = best.strategy.explosion_time_s
        locations = [
            ("pre-explosion", explosion_time - probe, "inactive"),
            ("start", interval.start, "boundary"),
            ("midpoint", midpoint, "inside"),
            ("end", interval.end, "boundary"),
            ("after-end", interval.end + probe, "outside"),
        ]
        if interval.start > explosion_time:
            active_left_probe = interval.start - min(probe, 0.5 * (interval.start - explosion_time))
            locations.insert(1, ("before-start", active_left_probe, "outside"))
        for location, time, expected in locations:
            active = (
                best.strategy.explosion_time_s
                <= time
                <= (best.strategy.explosion_time_s + CLOUD_LIFETIME)
            )
            margin = (
                _strict_margin(
                    time,
                    best.strategy,
                    gravity,
                    rim_samples=int(config["formal_evaluation"]["rim_samples"]),
                    refine_rim=True,
                )
                if active
                else None
            )
            boundary_checks.append(
                {
                    "location": location,
                    "time_s": time,
                    "expected_state": expected,
                    "strict_margin_m": margin,
                    "cloud_active": active,
                }
            )

    surface_points = _dense_target_surface(
        int(validation["surface_angles"]), int(validation["surface_levels"])
    )
    independent_checks = []
    for boundary in boundary_checks:
        time = float(boundary["time_s"])
        if not boundary["cloud_active"]:
            independent_checks.append(
                {
                    "location": boundary["location"],
                    "time_s": time,
                    "cloud_active": False,
                    "occluded": False,
                }
            )
            continue
        maximum_distance, worst_point = _maximum_segment_distance(
            surface_points,
            missile_position(time),
            cloud_position(time, best.strategy, gravity),
        )
        independent_checks.append(
            {
                "location": boundary["location"],
                "time_s": time,
                "cloud_active": True,
                "sampled_surface_points": len(surface_points),
                "maximum_segment_distance_m": maximum_distance,
                "distance_margin_m": maximum_distance - CLOUD_RADIUS,
                "occluded": maximum_distance <= CLOUD_RADIUS,
                "worst_sampled_target_point": worst_point.tolist(),
            }
        )

    perturbations = {
        "theta-minus": (-float(validation["perturbation_theta_rad"]), 0.0, 0.0, 0.0),
        "theta-plus": (float(validation["perturbation_theta_rad"]), 0.0, 0.0, 0.0),
        "speed-plus": (0.0, float(validation["perturbation_speed_m_s"]), 0.0, 0.0),
        "release-plus": (0.0, 0.0, float(validation["perturbation_release_time_s"]), 0.0),
        "fuse-minus": (0.0, 0.0, 0.0, -float(validation["perturbation_fuse_delay_s"])),
        "fuse-plus": (0.0, 0.0, 0.0, float(validation["perturbation_fuse_delay_s"])),
    }
    baseline = np.array(list(asdict(best.strategy).values()), dtype=float)
    perturbation_checks = []
    for label, offset in perturbations.items():
        candidate = baseline + np.asarray(offset)
        objective = _root_refined_objective(
            candidate,
            gravity,
            time_step=float(validation["perturbation_time_step_s"]),
            rim_samples=int(validation["perturbation_rim_samples"]),
            root_tolerance=root_tolerance,
        )
        perturbation_checks.append(
            {
                "perturbation": label,
                "strategy": asdict(_strategy_from_array(candidate)),
                "root_refined_strict_duration_s": -objective if objective <= 0.0 else 0.0,
                "feasible": _feasibility_violation(_strategy_from_array(candidate), gravity)
                <= 1e-12,
            }
        )

    q1_baseline = _formal_evaluation(Q2Strategy(np.pi, 120.0, 1.5, 3.6), gravity, config)
    return {
        "convergence": convergence,
        "boundary_checks": boundary_checks,
        "independent_full_surface_segment_checks": independent_checks,
        "local_perturbation_checks": perturbation_checks,
        "q1_kernel_regression": {
            "strategy": asdict(q1_baseline.strategy),
            "strict_duration_s": q1_baseline.strict_duration_s,
            "expected_q1_strict_duration_s": 1.3916426681980933,
            "absolute_difference_s": abs(q1_baseline.strict_duration_s - 1.3916426681980933),
        },
    }


def run_question_2(project_root: Path, config: dict[str, Any]) -> list[Path]:
    """Optimize Question 2 and write auditable tables and solver metadata."""
    paths = prepare_output_paths(project_root / "outputs")
    evaluations, solver_record = optimize_question_2(config)
    if not evaluations:
        raise RuntimeError("Question 2 optimization produced no feasible formal candidate")

    near_count = min(int(config["near_optimal_count"]), len(evaluations))
    candidate_rows = [
        _summary_row(item, rank) for rank, item in enumerate(evaluations[:near_count], 1)
    ]
    summary_path = save_table(
        pd.DataFrame(candidate_rows[:1]), paths.tables / "q2_summary.csv", overwrite=True
    )
    candidates_path = save_table(
        pd.DataFrame(candidate_rows), paths.tables / "q2_candidates.csv", overwrite=True
    )

    interval_rows: list[dict[str, Any]] = []
    for rank, evaluation in enumerate(evaluations[:near_count], 1):
        for criterion, intervals in (
            ("strict-full-cylinder", evaluation.strict_intervals),
            ("center-line", evaluation.center_intervals),
        ):
            for number, interval in enumerate(intervals, 1):
                interval_rows.append(
                    {
                        "rank": rank,
                        "criterion": criterion,
                        "interval": number,
                        "start_s": interval.start,
                        "end_s": interval.end,
                        "duration_s": interval.duration,
                    }
                )
    intervals_path = save_table(
        pd.DataFrame(interval_rows), paths.tables / "q2_intervals.csv", overwrite=True
    )

    best = evaluations[0]
    record = {
        "model": "q2-two-stage-derivative-free-v1",
        "objective": "maximize union duration of strict full-cylinder occlusion",
        "gravity_m_s2": float(config["gravity"]),
        "best": {
            **_summary_row(best, 1),
            "strict_intervals": [asdict(interval) for interval in best.strict_intervals],
            "center_intervals": [asdict(interval) for interval in best.center_intervals],
        },
        "near_optimal_candidates": candidate_rows,
        "solver": solver_record,
        "ground_rule": (
            "Cloud continues descending for 20 seconds as stated; no additional ground cutoff."
        ),
        "feasibility_checks": {
            "speed_within_bounds": MINIMUM_SPEED <= best.strategy.speed_m_s <= MAXIMUM_SPEED,
            "release_time_nonnegative": best.strategy.release_time_s >= 0.0,
            "fuse_delay_nonnegative": best.strategy.fuse_delay_s >= 0.0,
            "explosion_before_missile_hit": best.strategy.explosion_time_s <= missile_hit_time(),
            "explosion_above_ground": best.explosion_point[2] >= 0.0,
            "cloud_center_below_ground_during_effective_window": best.cloud_center_below_ground,
        },
    }
    log_path = paths.logs / "q2_optimization.json"
    log_path.write_text(json.dumps(record, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    validation_record = _q2_validation_record(best, config)
    validation_path = paths.logs / "q2_validation.json"
    validation_path.write_text(
        json.dumps(validation_record, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    return [summary_path, candidates_path, intervals_path, log_path, validation_path]
